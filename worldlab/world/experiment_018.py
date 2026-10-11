"""Experiment 018: mean decision-impact matching.

Calibration uses Experiment 017 S0 histories. Held-out worlds use new choice
seeds. This module does not modify Experiment 004, 005, or 017 production code
and does not call a model.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from random import Random
from typing import Any, Callable

import numpy as np

from agents.persona_generator import append_jsonl, write_json
from config.settings import PROJECT_ROOT
from experiments.reproducibility_check import canonical_sha256
from world.dynamics_calibration import load_initial_locations
from world.experiment_002 import REFERENCE_CHOICE_TEMPERATURE, REFERENCE_RECONSIDERATION_RATE
from world.experiment_004 import trajectory_hash
from world.experiment_005 import SOCIAL_WEIGHT, DecayMemory, RollingMemory
from world.experiment_017 import (
    CELL_BY_ID,
    SOCIAL_CELL_IDS,
    _base_utilities,
    _commit,
    file_sha256,
    score_memory,
    simulate_cell,
    unpack_series,
)
from world.experiment_017 import OUTPUT_DIR as WORLD_V017_DIR
from world.geography import PLACE_IDS, distance_matrix, occupancy_of, place_index
from world.utility_atlas import OUTPUT_DIR as ATLAS_DIR
from world.utility_atlas import load_frozen_atlas
from world.experiment_004 import empty_social

OUTPUT_DIR = PROJECT_ROOT / "data" / "worlds" / "world_v0_18"
PROTOCOL_PATH = PROJECT_ROOT / "docs" / "EXPERIMENT_018_PROTOCOL.md"
REPORT_PATH = PROJECT_ROOT / "docs" / "EXPERIMENT_018_REPORT.md"
EXPECTED_ATLAS_SHA256 = "664b511d208f7989e8a671d0e2d6e6433366120bb3d5d6a67db38277b4d5a373"
HASH_CHUNK_BYTES = 1024 * 1024
TRACE_SCAN_LIMIT = 2_000_000
JSON_SCAN_LIMIT = 20_000_000
TIMESTEPS = 500
LATE_START = 101
REFERENCE_SOCIAL_WEIGHT = 150.0
S0_WEIGHT = 0.0
Q_TARGETS = (
    {"target_id": "q05", "q_target": 0.05},
    {"target_id": "q15", "q_target": 0.15},
    {"target_id": "q25", "q_target": 0.25},
)
CALIBRATION_TOLERANCE = 0.0005
MATCH_BAND = 0.005
Q_COMPATIBILITY_TOLERANCE = 1e-10
MONOTONICITY_SLACK = 1e-8
WEIGHT_FLOOR = 0.0
WEIGHT_START = 150.0
WEIGHT_CAP = 1_000_000.0
MAX_BISECTIONS = 64
BOOTSTRAP_DRAWS = 10_000
ANALYSIS_SEED = 20260930
BUDGET_SECONDS = 5 * 60 * 60
RESERVE_SECONDS = 40 * 60
PERIODS = (
    ("t1_100", 0, 100),
    ("t101_200", 100, 200),
    ("t201_300", 200, 300),
    ("t301_400", 300, 400),
    ("t401_500", 400, 500),
)
CALIBRATION_SEEDS = tuple(range(200, 700))
HELDOUT_SEEDS = tuple(range(700, 1700))
LINEAGE_A_SEEDS = tuple(range(50))
LINEAGE_B_SEEDS = (200, 449, 699)
REPLAY_SEEDS = (700, 1199, 1699)
SHORT_MEMORY = {"cumulative": "CUM", "rolling": "RW50", "exponential": "EXP99"}
MEMORIES = ("CUM", "RW50", "EXP99")
TRANSFORMS = ("LOGMAX", "ZRAW")
_WORKER: dict[str, Any] = {}
_POOL: ProcessPoolExecutor | None = None


def percentile_indexes(draws: int) -> tuple[int, int]:
    return int(0.025 * (draws - 1)), int(0.975 * (draws - 1))


def refuse_nonempty_output(destination: Path) -> None:
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError("experiment 018 refuses to write into a non-empty output directory")


def condition_specs(weights: dict[str, float] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in Q_TARGETS:
        for cell_id in SOCIAL_CELL_IDS:
            condition_id = f"{cell_id}__{target['target_id']}"
            rows.append(
                {
                    "condition_id": condition_id,
                    "cell_id": cell_id,
                    "target_id": target["target_id"],
                    "q_target": target["q_target"],
                    "role": "matched",
                    "social_weight": None if weights is None else float(weights[condition_id]),
                }
            )
    for cell_id in SOCIAL_CELL_IDS:
        rows.append(
            {
                "condition_id": f"{cell_id}__w150",
                "cell_id": cell_id,
                "target_id": None,
                "q_target": None,
                "role": "reference_150",
                "social_weight": REFERENCE_SOCIAL_WEIGHT,
            }
        )
    rows.append(
        {
            "condition_id": "S0",
            "cell_id": "S0",
            "target_id": None,
            "q_target": None,
            "role": "s0",
            "social_weight": S0_WEIGHT,
        }
    )
    return rows


def assert_weight_schedule(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 25:
        raise RuntimeError("each held-out seed must keep 25 distinct conditions")
    if len({row["condition_id"] for row in rows}) != 25:
        raise RuntimeError("condition identifiers collided")
    matched = [row for row in rows if row["role"] == "matched"]
    if len(matched) != 18 or any(row["social_weight"] is None for row in matched):
        raise RuntimeError("matched conditions are missing calibrated weights")


def require_finite_array(values: np.ndarray, label: str) -> None:
    if not np.isfinite(values).all():
        raise RuntimeError(f"nonfinite {label}")


def require_indexed_trajectory(rows: list[dict[str, Any]], timesteps: int = TIMESTEPS) -> list[dict[str, Any]]:
    seen: dict[int, dict[str, Any]] = {}
    for row in rows:
        timestep = int(row["timestep"])
        if timestep in seen:
            raise RuntimeError(f"duplicate timestep {timestep}")
        seen[timestep] = row
    expected = set(range(timesteps + 1))
    if set(seen) != expected:
        raise RuntimeError("source trajectory is missing timesteps")
    return [seen[timestep] for timestep in range(timesteps + 1)]


def require_series(series: list[dict[str, str]], timesteps: int = TIMESTEPS) -> list[str]:
    if len(series) != timesteps + 1:
        raise RuntimeError("source trajectory does not contain timesteps 0 through the horizon")
    agents = sorted(series[0])
    if len(agents) != 20 or len(set(agents)) != 20:
        raise RuntimeError("source trajectory does not contain the frozen agent set")
    allowed = set(PLACE_IDS)
    for step in series:
        if sorted(step) != agents:
            raise RuntimeError("source trajectory agent set changed")
        if any(place_id not in allowed for place_id in step.values()):
            raise RuntimeError("source trajectory contains an unknown place")
    return agents


def build_decision_cache(
    atlas: Any,
    series: list[dict[str, str]],
    matrix: dict[str, dict[str, float]],
    places: dict[str, dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    agents = require_series(series, len(series) - 1)
    timesteps = len(series) - 1
    memories = {
        "cumulative": empty_social(agents),
        "rolling": RollingMemory(agents, 50),
        "exponential": DecayMemory(agents, 0.99),
    }
    specs = [CELL_BY_ID[cell_id] for cell_id in SOCIAL_CELL_IDS]
    bases = np.empty((timesteps, len(agents), 5), dtype=np.float64)
    scores = np.empty((len(specs), timesteps, len(agents), 5), dtype=np.float64)
    for timestep in range(timesteps):
        origins = series[timestep]
        occupancy = occupancy_of(origins)
        prepared: dict[tuple[str, str, str], dict[str, float]] = {}
        for spec in specs:
            memory = memories[spec["memory"]]
            for agent_id in agents:
                _unclipped, score, _loo = score_memory(memory, agent_id, spec["transform"])
                prepared[(spec["memory"], spec["transform"], agent_id)] = score
        for agent_index, agent_id in enumerate(agents):
            base = _base_utilities(atlas, matrix, places, agent_id, origins[agent_id], occupancy)
            for place_index, place_id in enumerate(PLACE_IDS):
                bases[timestep, agent_index, place_index] = base[place_id]
            for spec_index, spec in enumerate(specs):
                score = prepared[(spec["memory"], spec["transform"], agent_id)]
                for place_index, place_id in enumerate(PLACE_IDS):
                    scores[spec_index, timestep, agent_index, place_index] = score[place_id]
        destinations = series[timestep + 1]
        for name in memories:
            memories[name] = _commit(memories[name], destinations)
    require_finite_array(bases, "cached base utility")
    require_finite_array(scores, "cached score")
    return bases, scores


def softmax_utilities(utilities: np.ndarray) -> np.ndarray:
    peak = np.max(utilities, axis=-1, keepdims=True)
    weights = np.exp((utilities - peak) / REFERENCE_CHOICE_TEMPERATURE)
    require_finite_array(weights, "softmax weight")
    total = np.sum(weights, axis=-1, keepdims=True)
    return weights / total


def q_components(bases: np.ndarray, scores: np.ndarray, weight: float) -> np.ndarray:
    if bases.ndim != 3 or scores.shape != bases.shape:
        raise RuntimeError("Q cache axes are not timestep, agent, place")
    if float(weight) == 0.0:
        return np.zeros(bases.shape[:2], dtype=np.float64)
    base_probability = softmax_utilities(bases)
    social_probability = softmax_utilities(bases + np.float64(weight) * scores)
    components = 0.5 * np.abs(social_probability - base_probability).sum(axis=-1)
    require_finite_array(components, "decision Q")
    return components


def probability_tv(bases: np.ndarray, scores_a: np.ndarray, weight_a: float, scores_b: np.ndarray, weight_b: float) -> float:
    def _probs(scores: np.ndarray, weight: float) -> np.ndarray:
        if float(weight) == 0.0:
            return softmax_utilities(bases)
        return softmax_utilities(bases + np.float64(weight) * scores)

    gap = 0.5 * np.abs(_probs(scores_a, weight_a) - _probs(scores_b, weight_b)).sum(axis=-1)
    require_finite_array(gap, "probability TV")
    return float(gap.mean())


def summarize_q_map(components: np.ndarray) -> dict[str, Any]:
    if components.shape != (TIMESTEPS, 20):
        raise RuntimeError("formal Q summary requires 500 timesteps and 20 agents")
    agent_means = components.mean(axis=0)
    return {
        "q": float(components.mean()),
        "agent_means": [float(value) for value in agent_means],
        "agent_sd": float(agent_means.std(ddof=0)),
        "periods": {name: float(components[start:stop].mean()) for name, start, stop in PERIODS},
    }


def layer_a_for_weights(bases: np.ndarray, scores: np.ndarray, choice_seed_value: int, specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    stored: dict[str, np.ndarray] = {}
    for spec in specs:
        if spec["cell_id"] == "S0":
            continue
        cell_index = SOCIAL_CELL_IDS.index(spec["cell_id"])
        components = q_components(bases, scores[cell_index], float(spec["social_weight"]))
        stored[spec["condition_id"]] = components
        summary = summarize_q_map(components)
        summary.update(
            {
                "condition_id": spec["condition_id"],
                "cell_id": spec["cell_id"],
                "target_id": spec["target_id"],
                "role": spec["role"],
                "choice_seed": choice_seed_value,
                "social_weight": float(spec["social_weight"]),
                "q_target": spec["q_target"],
                "llm_calls": 0,
            }
        )
        rows.append(summary)
    for memory in MEMORIES:
        for target in Q_TARGETS:
            left = f"{memory}_LOGMAX__{target['target_id']}"
            right = f"{memory}_ZRAW__{target['target_id']}"
            _attach_tv(rows, bases, scores, left, right)
        _attach_tv(rows, bases, scores, f"{memory}_LOGMAX__w150", f"{memory}_ZRAW__w150")
    return rows


def _attach_tv(rows: list[dict[str, Any]], bases: np.ndarray, scores: np.ndarray, left_id: str, right_id: str) -> None:
    by_id = {row["condition_id"]: row for row in rows}
    left = by_id[left_id]
    right = by_id[right_id]
    tv = probability_tv(
        bases,
        scores[SOCIAL_CELL_IDS.index(left["cell_id"])],
        float(left["social_weight"]),
        scores[SOCIAL_CELL_IDS.index(right["cell_id"])],
        float(right["social_weight"]),
    )
    key = f"tv_{left['cell_id']}_vs_{right['cell_id']}"
    if left["target_id"] is not None:
        key = f"tv_{left['cell_id'].split('_')[0]}_{left['target_id']}"
    else:
        key = f"tv_{left['cell_id'].split('_')[0]}_w150"
    for row in (left, right):
        row.setdefault("probability_tv", {})[key] = tv


def seed_mean_from_cache(bases: np.ndarray, scores: np.ndarray, weight: float) -> float:
    return float(q_components(bases, scores, weight).mean())


def cohort_seed_q(bases: np.ndarray, scores: np.ndarray, weight: float) -> np.ndarray:
    if bases.ndim != 4 or scores.shape != bases.shape:
        raise RuntimeError("cohort cache is missing the seed axis")
    output = np.empty(bases.shape[0], dtype=np.float64)
    for start in range(0, bases.shape[0], 25):
        stop = min(start + 25, bases.shape[0])
        block_bases = bases[start:stop]
        if float(weight) == 0.0:
            output[start:stop] = 0.0
            continue
        block_scores = scores[start:stop]
        base_probability = softmax_utilities(block_bases)
        social_probability = softmax_utilities(block_bases + np.float64(weight) * block_scores)
        components = 0.5 * np.abs(social_probability - base_probability).sum(axis=-1)
        output[start:stop] = components.mean(axis=(1, 2))
    require_finite_array(output, "cohort Q")
    return output


def _points_are_monotone(points: list[tuple[float, float]]) -> bool:
    ordered = sorted(points, key=lambda item: (item[0], item[1]))
    for earlier, later in zip(ordered, ordered[1:]):
        if later[0] > earlier[0] and later[1] + MONOTONICITY_SLACK < earlier[1]:
            return False
    return True


def solve_weight(evaluate: Callable[[float], float], target: float, *, weight_cap: float = WEIGHT_CAP) -> dict[str, Any]:
    if target not in {item["q_target"] for item in Q_TARGETS}:
        raise RuntimeError("solver received a target outside the preregistered set")
    points: list[tuple[float, float]] = [(WEIGHT_FLOOR, 0.0)]
    trace: list[dict[str, Any]] = [{"weight": WEIGHT_FLOOR, "mean_q": 0.0, "abs_error": abs(target), "stage": "floor"}]

    def _evaluate(weight: float, stage: str) -> float:
        if weight < WEIGHT_FLOOR or weight > weight_cap:
            raise RuntimeError("solver evaluated a weight outside its preregistered bounds")
        mean_q = float(evaluate(weight))
        if not math.isfinite(mean_q):
            raise RuntimeError("solver received a nonfinite mean Q")
        points.append((weight, mean_q))
        trace.append({"weight": weight, "mean_q": mean_q, "abs_error": abs(mean_q - target), "stage": stage})
        if not _points_are_monotone(points):
            raise RuntimeError("calibration Q is not monotone at the evaluated weights")
        return mean_q

    hi = WEIGHT_START
    q_hi = _evaluate(hi, "upper")
    if abs(q_hi - target) <= CALIBRATION_TOLERANCE:
        return {"ok": True, "weight": hi, "mean_q": q_hi, "abs_error": abs(q_hi - target), "trace": trace, "reason": "tolerance_met"}
    while q_hi < target and hi < weight_cap:
        nxt = hi * 2.0
        if nxt > weight_cap:
            nxt = weight_cap
        if nxt == hi:
            break
        hi = nxt
        q_hi = _evaluate(hi, "expand")
        if abs(q_hi - target) <= CALIBRATION_TOLERANCE:
            return {"ok": True, "weight": hi, "mean_q": q_hi, "abs_error": abs(q_hi - target), "trace": trace, "reason": "tolerance_met"}
    if q_hi < target - CALIBRATION_TOLERANCE:
        return {"ok": False, "weight": None, "mean_q": q_hi, "abs_error": abs(q_hi - target), "trace": trace, "reason": "no_bracket"}
    lo = WEIGHT_FLOOR
    q_lo = 0.0
    if q_hi + MONOTONICITY_SLACK < target:
        return {"ok": False, "weight": None, "mean_q": q_hi, "abs_error": abs(q_hi - target), "trace": trace, "reason": "no_bracket"}
    for _step in range(MAX_BISECTIONS):
        mid = (lo + hi) / 2.0
        q_mid = _evaluate(mid, "bisection")
        if abs(q_mid - target) <= CALIBRATION_TOLERANCE:
            return {"ok": True, "weight": mid, "mean_q": q_mid, "abs_error": abs(q_mid - target), "trace": trace, "reason": "tolerance_met"}
        if q_mid < target:
            lo = mid
            q_lo = q_mid
        else:
            hi = mid
            q_hi = q_mid
    return {
        "ok": False,
        "weight": None,
        "mean_q": q_lo,
        "abs_error": min(abs(q_lo - target), abs(q_hi - target)),
        "trace": trace,
        "reason": "tolerance_not_met",
    }


def fit_targets(evaluate_cell: Callable[[str, float], float]) -> dict[str, Any]:
    fitted: dict[str, Any] = {}
    for cell_id in SOCIAL_CELL_IDS:
        for target in Q_TARGETS:
            condition_id = f"{cell_id}__{target['target_id']}"
            outcome = solve_weight(lambda weight, cell_id=cell_id: evaluate_cell(cell_id, weight), float(target["q_target"]))
            outcome["condition_id"] = condition_id
            outcome["cell_id"] = cell_id
            outcome["target_id"] = target["target_id"]
            outcome["q_target"] = target["q_target"]
            fitted[condition_id] = outcome
            print(f"calibration {condition_id} ok={outcome['ok']} reason={outcome['reason']} weight={outcome['weight']}", flush=True)
            if not outcome["ok"]:
                return {"ok": False, "weights": fitted, "failed_condition": condition_id}
    weights = {condition_id: float(item["weight"]) for condition_id, item in fitted.items()}
    return {"ok": True, "weights": fitted, "weight_values": weights, "failed_condition": None}


def hhi_contrasts(values: dict[str, float]) -> dict[str, float]:
    output: dict[str, float] = {}
    for memory in MEMORIES:
        for target in Q_TARGETS:
            target_id = target["target_id"]
            matched = values[f"{memory}_ZRAW__{target_id}"] - values[f"{memory}_LOGMAX__{target_id}"]
            output[f"D_matched_{memory}_{target_id}"] = matched
        reference = values[f"{memory}_ZRAW__w150"] - values[f"{memory}_LOGMAX__w150"]
        output[f"D_150_{memory}"] = reference
        for target in Q_TARGETS:
            target_id = target["target_id"]
            output[f"D_diff_{memory}_{target_id}"] = output[f"D_matched_{memory}_{target_id}"] - reference
    for suffix in [item["target_id"] for item in Q_TARGETS] + ["w150"]:
        for transform in TRANSFORMS:
            baseline = values[f"CUM_{transform}__{suffix}"]
            for memory in ("RW50", "EXP99"):
                output[f"mem_{memory}_minus_CUM_{transform}_{suffix}"] = values[f"{memory}_{transform}__{suffix}"] - baseline
    for memory in ("RW50", "EXP99"):
        for target in Q_TARGETS:
            target_id = target["target_id"]
            output[f"interaction_{memory}_{target_id}"] = output[f"D_matched_{memory}_{target_id}"] - output[f"D_matched_CUM_{target_id}"]
        output[f"interaction_{memory}_w150"] = output[f"D_150_{memory}"] - output["D_150_CUM"]
    return output


def q_contrasts(values: dict[str, float]) -> dict[str, float]:
    output: dict[str, float] = {}
    for memory in MEMORIES:
        for target in Q_TARGETS:
            target_id = target["target_id"]
            output[f"Qdiff_{memory}_{target_id}"] = values[f"{memory}_ZRAW__{target_id}"] - values[f"{memory}_LOGMAX__{target_id}"]
        output[f"Qdiff_{memory}_w150"] = values[f"{memory}_ZRAW__w150"] - values[f"{memory}_LOGMAX__w150"]
    return output


def bootstrap_means(values: np.ndarray, rng_seed: int, draws: int = BOOTSTRAP_DRAWS, batch: int = 16) -> np.ndarray:
    generator = Random(rng_seed)
    n_seeds, n_metrics = values.shape
    output = np.empty((draws, n_metrics), dtype=np.float64)
    cursor = 0
    while cursor < draws:
        width = min(batch, draws - cursor)
        indexes = np.empty((width, n_seeds), dtype=np.int32)
        for row in range(width):
            for column in range(n_seeds):
                indexes[row, column] = generator.randrange(n_seeds)
        output[cursor : cursor + width] = values[indexes].mean(axis=1)
        cursor += width
    return output


def distribution_summary(samples: list[float], replicates: list[float] | None = None) -> dict[str, Any]:
    ordered = sorted(samples)
    quantiles = statistics.quantiles(samples, n=4, method="inclusive")
    summary = {
        "mean": statistics.fmean(samples),
        "median": statistics.median(samples),
        "stdev": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "minimum": ordered[0],
        "maximum": ordered[-1],
        "q25": quantiles[0],
        "q75": quantiles[2],
        "n": len(samples),
    }
    if replicates is not None:
        low, high = percentile_indexes(len(replicates))
        ranked = sorted(replicates)
        summary["ci_low"] = ranked[low]
        summary["ci_high"] = ranked[high]
    return summary


def matching_flags(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for target in Q_TARGETS:
        band_low = float(target["q_target"]) - MATCH_BAND
        band_high = float(target["q_target"]) + MATCH_BAND
        for cell_id in SOCIAL_CELL_IDS:
            condition_id = f"{cell_id}__{target['target_id']}"
            item = summaries[condition_id]
            rows.append(
                {
                    "condition_id": condition_id,
                    "mean_q": item["mean"],
                    "ci_low": item["ci_low"],
                    "ci_high": item["ci_high"],
                    "abs_error": abs(item["mean"] - float(target["q_target"])),
                    "ci_inside_match_band": bool(item["ci_low"] >= band_low and item["ci_high"] <= band_high),
                    "band": [band_low, band_high],
                }
            )
    pairs = []
    for memory in MEMORIES:
        for target in Q_TARGETS:
            key = f"Qdiff_{memory}_{target['target_id']}"
            item = summaries[key]
            pairs.append(
                {
                    "contrast": key,
                    "mean": item["mean"],
                    "ci_low": item["ci_low"],
                    "ci_high": item["ci_high"],
                    "ci_inside_match_band": bool(item["ci_low"] >= -MATCH_BAND and item["ci_high"] <= MATCH_BAND),
                }
            )
    return {"cells": rows, "paired_q_differences": pairs, "match_band": MATCH_BAND, "llm_calls": 0}


def completion_decision(*, lineage_passed: bool, replay_passed: bool, worlds_complete: bool, calibration_passed: bool) -> str:
    if lineage_passed and replay_passed and worlds_complete and calibration_passed:
        return "COMPLETE"
    return "INCOMPLETE"


def _display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def _collect_choice_seeds(node: Any, found: set[int]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "choice_seeds" and isinstance(value, list) and all(isinstance(item, int) and not isinstance(item, bool) for item in value):
                found.update(int(item) for item in value)
            elif key == "choice_seed" and isinstance(value, int) and not isinstance(value, bool):
                found.add(int(value))
            else:
                _collect_choice_seeds(value, found)
    elif isinstance(node, list):
        for item in node:
            _collect_choice_seeds(item, found)


def _trajectory_seed(directory_name: str, key: str) -> int | None:
    parts = key.split(":")
    if directory_name == "world_v0_6":
        if len(parts) == 3 and parts[2].isdigit():
            return int(parts[2])
        return None
    if len(parts) >= 2 and parts[1].isdigit():
        return int(parts[1])
    return None


def audit_choice_seeds(data_root: Path | None = None) -> dict[str, Any]:
    root = data_root or (PROJECT_ROOT / "data")
    proposed = set(HELDOUT_SEEDS)
    sources = []
    overlaps = []
    unchecked = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in {".json", ".jsonl"}:
            continue
        if "world_v0_18" in path.parts:
            continue
        size = path.stat().st_size
        if path.suffix == ".jsonl" and size > TRACE_SCAN_LIMIT:
            unchecked.append({"path": _display(path), "bytes": size, "reason": "large trace; choice seeds are taken from configs and manifests"})
            continue
        if path.suffix == ".json" and size > JSON_SCAN_LIMIT:
            unchecked.append({"path": _display(path), "bytes": size, "reason": "json above the index scan limit"})
            overlaps.append(_display(path))
            continue
        choice_seeds: set[int] = set()
        trajectory_seeds: set[int] = set()
        if path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            _collect_choice_seeds(payload, choice_seeds)
            trajectories = payload.get("trajectory_sha256") if isinstance(payload, dict) else None
            if isinstance(trajectories, dict):
                for key in trajectories:
                    seed = _trajectory_seed(path.parent.name, key)
                    if seed is not None:
                        trajectory_seeds.add(seed)
        else:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if isinstance(row, dict) and isinstance(row.get("choice_seed"), int) and not isinstance(row.get("choice_seed"), bool):
                        choice_seeds.add(int(row["choice_seed"]))
        if not choice_seeds and not trajectory_seeds:
            continue
        hit = sorted((choice_seeds | trajectory_seeds) & proposed)
        record = {
            "path": _display(path),
            "choice_seed_min": min(choice_seeds) if choice_seeds else None,
            "choice_seed_max": max(choice_seeds) if choice_seeds else None,
            "trajectory_seed_min": min(trajectory_seeds) if trajectory_seeds else None,
            "trajectory_seed_max": max(trajectory_seeds) if trajectory_seeds else None,
        }
        if path.parent.name == "world_v0_6":
            record["trajectory_note"] = "world_v0_6 keys are architecture:social_weight:choice_seed; the choice seed is the third field"
        if hit:
            record["overlap"] = hit
            overlaps.append(record["path"])
        sources.append(record)
    return {
        "experiment_id": "experiment_018_mean_decision_impact_matching",
        "proposed_choice_seeds": "700-1699",
        "proposed_count": len(proposed),
        "passed": not overlaps,
        "overlap_sources": overlaps,
        "sources": sources,
        "unchecked_large_traces": unchecked,
        "scanned_documents": False,
        "includes_world_v0_17_outputs": True,
        "includes_world_v0_18_outputs": False,
        "judgment": (
            "Seeds 700-1699 do not appear as choice seeds in the checked configs, manifests, or small indexes, including Experiment 017."
            if not overlaps
            else "The proposed seeds collide with an existing choice-seed index, or a required index could not be read."
        ),
        "llm_calls": 0,
    }


def _source_hashes() -> dict[str, str]:
    names = (
        "world/experiment_004.py",
        "world/experiment_005.py",
        "world/experiment_017.py",
        "world/experiment_018.py",
    )
    return {name: file_sha256(PROJECT_ROOT / name) for name in names}


def build_protocol(audit_sha256: str, started_at: str) -> dict[str, Any]:
    payload = {
        "record_type": "experiment_018_preregistered_protocol",
        "world_id": "world_v0_18",
        "experiment_id": "experiment_018_mean_decision_impact_matching",
        "status": "PREREGISTERED_BEFORE_CALIBRATION",
        "started_at": started_at,
        "llm_calls": 0,
        "utility_atlas_sha256": EXPECTED_ATLAS_SHA256,
        "choice_temperature": REFERENCE_CHOICE_TEMPERATURE,
        "reconsideration_rate": REFERENCE_RECONSIDERATION_RATE,
        "familiarity_weight": 0.0,
        "timesteps": TIMESTEPS,
        "agents": 20,
        "places": list(PLACE_IDS),
        "reference_social_weight": REFERENCE_SOCIAL_WEIGHT,
        "s0_social_weight": S0_WEIGHT,
        "calibration_seeds": list(CALIBRATION_SEEDS),
        "heldout_seeds": list(HELDOUT_SEEDS),
        "lineage_a_seeds": list(LINEAGE_A_SEEDS),
        "lineage_b_seeds": list(LINEAGE_B_SEEDS),
        "replay_seeds": list(REPLAY_SEEDS),
        "q_targets": [dict(item) for item in Q_TARGETS],
        "calibration_tolerance": CALIBRATION_TOLERANCE,
        "match_band": MATCH_BAND,
        "q_compatibility_tolerance": Q_COMPATIBILITY_TOLERANCE,
        "monotonicity_slack": MONOTONICITY_SLACK,
        "solver": {
            "weight_floor": WEIGHT_FLOOR,
            "initial_upper": WEIGHT_START,
            "weight_cap": WEIGHT_CAP,
            "cap_meaning": "numerical work limit with no psychological interpretation",
            "expansion": "double the upper bound until the target is bracketed or the cap is reached",
            "method": "deterministic bisection",
            "max_bisections": MAX_BISECTIONS,
            "objective": "absolute error of the mean of seed-mean Q on the calibration S0 histories",
            "stop": "stop at the first evaluated weight whose absolute error is within tolerance",
        },
        "primary_outcome": "late visit-share HHI from cumulative presence at t=101-500, then share, then HHI",
        "primary_contrasts": ["D_matched(memory, target)", "D_150(memory)", "D_matched - D_150"],
        "secondary_contrasts": ["memory contrast within transformation and target", "memory by transformation interaction"],
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "rng": "random.Random",
            "rng_seed": ANALYSIS_SEED,
            "unit": "held-out choice seed",
            "row": "with replacement, length 1000, shared by every condition and contrast",
            "percentile_indexes": list(percentile_indexes(BOOTSTRAP_DRAWS)),
            "interval": "pointwise",
            "index_meaning": "position in seeds 700-1699; index 0 is choice seed 700",
            "includes_refitting_uncertainty": False,
        },
        "counts": {
            "lineage_worlds": 371,
            "new_worlds": 25000,
            "replays": 75,
            "total_world_executions": 25446,
            "conditions_per_heldout_seed": 25,
        },
        "seed_audit_sha256": audit_sha256,
        "source_sha256": _source_hashes(),
        "experiment_017_config_sha256": file_sha256(WORLD_V017_DIR / "world_v0_17_config.json"),
        "experiment_017_cohort_b_layer_a_sha256": file_sha256(WORLD_V017_DIR / "cohort_b" / "layer_a_seed_metrics.jsonl"),
        "budget_seconds": BUDGET_SECONDS,
        "reserve_seconds": RESERVE_SECONDS,
        "incomplete_rule": "Stop only at a completed held-out seed boundary, keep completed outputs, mark INCOMPLETE, and do not present a subset as the full cohort.",
        "scope": [
            "Matched rules are whole decision rules at one fixed social weight.",
            "Equal mean Q does not imply equal probability vectors, time profiles, agent profiles, or closed-loop impact.",
            "The design does not identify a pure scale effect or a pure memory timescale.",
            "Calibration is not human-behavior calibration and the held-out seeds are not an external population.",
            "An HHI difference is not evidence of path dependence.",
            "A confidence interval that contains zero is not an equivalence result.",
            "Held-out uncertainty is conditional on the frozen weights.",
        ],
    }
    payload["protocol_sha256"] = canonical_sha256(payload)
    return payload


def protocol_markdown(protocol: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Experiment 018 protocol",
            "",
            "狀態：校準前固定。本文件不含校準結果，也不含 seeds 700–1699 的世界結果。",
            "",
            f"Protocol SHA-256：`{protocol['protocol_sha256']}`",
            "",
            "Calibration cohort 是 Experiment 017 已保存的 S0，seeds 200–699。Held-out cohort 是 seeds 700–1699。",
            "Q targets 固定為 0.05、0.15、0.25。校準容差是 0.0005。Held-out 平均匹配帶是 target ± 0.005，配對 Q 差的帶是 [-0.005, 0.005]。",
            "Solver 從 weight 0 與上界 150 開始，必要時把上界倍增，上限 1,000,000，之後做最多 64 次二分。搜尋上限沒有心理學含義。",
            "主要閉環結果是 t=101–500 累積 visit share 的 HHI。主要對照是九個 D_matched、三個 D_150，以及 D_matched − D_150。",
            "Bootstrap 使用分析種子 20260930 的本地 Random，10000 次，每列 1000 個有放回索引，percentile indexes 249 與 9749。索引 0 是 choice seed 700。",
            "這個區間給的是已固定 weights 之下新 seeds 的不確定性，不包含重新 fitting 的校準不確定性。",
            "",
            "平均 Q 相同，不代表選擇機率向量、時間分布、agent 分布或閉環影響相同。",
            "這不是純尺度識別、純記憶時間尺度、真人行為校準、路徑依賴或外部效度。",
            "Held-out 匹配帶沒有通過時，仍依固定 weights 完成分析，不重新 fitting。",
            "",
            "計數：lineage 371、新世界 25000、replay 75、合計 25446。Calibration 不新增世界。Layer A 不另計 world。",
            "",
        ]
    )


def _init_worker() -> None:
    _WORKER["atlas"] = load_frozen_atlas()
    _WORKER["locations"] = load_initial_locations()
    _WORKER["matrix"] = distance_matrix()
    _WORKER["places"] = place_index()


def _shared_pool(workers: int) -> ProcessPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = ProcessPoolExecutor(max_workers=workers, initializer=_init_worker)
    return _POOL


def _shutdown_pool() -> None:
    global _POOL
    if _POOL is not None:
        _POOL.shutdown(wait=True, cancel_futures=False)
        _POOL = None


def _load_017_s0(seed: int) -> list[dict[str, str]]:
    path = WORLD_V017_DIR / "cohort_b" / "units" / f"S0__{seed}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    series = unpack_series(payload["series_packed"])
    if trajectory_hash(series) != payload["trajectory_sha256"]:
        raise RuntimeError(f"Experiment 017 S0 series does not match its saved hash at seed {seed}")
    require_series(series, TIMESTEPS)
    return series


def _cache_task(task: tuple[int, str]) -> str:
    if "atlas" not in _WORKER:
        _init_worker()
    seed, cache_dir = task
    bases, scores = build_decision_cache(_WORKER["atlas"], _load_017_s0(seed), _WORKER["matrix"], _WORKER["places"])
    path = Path(cache_dir) / f"{seed}.npz"
    np.savez(path, bases=bases, scores=scores)
    return str(seed)


def _load_cache_stack(cache_dir: Path, seeds: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    bases = []
    scores = []
    for seed in seeds:
        with np.load(cache_dir / f"{seed}.npz") as payload:
            bases.append(np.array(payload["bases"], dtype=np.float64))
            scores.append(np.array(payload["scores"], dtype=np.float64))
    return np.stack(bases), np.stack(scores)


def _load_017_q() -> dict[tuple[str, int], float]:
    found: dict[tuple[str, int], float] = {}
    path = WORLD_V017_DIR / "cohort_b" / "layer_a_seed_metrics.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            found[(str(row["cell_id"]), int(row["choice_seed"]))] = float(row["q"])
    return found


def _compare_q(bases: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    stored = _load_017_q()
    failures = []
    max_error = 0.0
    for cell_index, cell_id in enumerate(SOCIAL_CELL_IDS):
        got = cohort_seed_q(bases, scores[:, cell_index], REFERENCE_SOCIAL_WEIGHT)
        for seed_index, seed in enumerate(CALIBRATION_SEEDS):
            key = (cell_id, seed)
            if key not in stored:
                failures.append({"cell_id": cell_id, "choice_seed": seed, "reason": "missing_017_q"})
                continue
            error = abs(float(got[seed_index]) - stored[key])
            max_error = max(max_error, error)
            if error > Q_COMPATIBILITY_TOLERANCE:
                failures.append({"cell_id": cell_id, "choice_seed": seed, "abs_error": error, "got": float(got[seed_index]), "expected": stored[key]})
    return {"passed": not failures, "compared": len(CALIBRATION_SEEDS) * len(SOCIAL_CELL_IDS), "max_abs_error": max_error, "failures": failures, "llm_calls": 0}


def _world_task(task: dict[str, Any]) -> dict[str, Any]:
    if "atlas" not in _WORKER:
        _init_worker()
    record = simulate_cell(
        _WORKER["atlas"],
        _WORKER["locations"],
        task["cell_id"],
        int(task["choice_seed"]),
        social_weight=float(task["social_weight"]),
        matrix=_WORKER["matrix"],
        places=_WORKER["places"],
        keep_series=bool(task["keep_series"]),
        timesteps=int(task.get("timesteps", TIMESTEPS)),
    )
    record["condition_id"] = task["condition_id"]
    record["role"] = task["role"]
    record["target_id"] = task["target_id"]
    record["cohort"] = task["cohort"]
    record["llm_calls"] = 0
    if task.get("layer_a"):
        series = unpack_series(record["series_packed"])
        bases, scores = build_decision_cache(_WORKER["atlas"], series, _WORKER["matrix"], _WORKER["places"])
        record["layer_a"] = layer_a_for_weights(bases, scores, int(task["choice_seed"]), task["layer_specs"])
    return record


def _execute(tasks: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    if workers <= 1 or len(tasks) == 1:
        return [_world_task(task) for task in tasks]
    pool = _shared_pool(workers)
    futures = {pool.submit(_world_task, task): index for index, task in enumerate(tasks)}
    output: list[dict[str, Any] | None] = [None] * len(tasks)
    for future in as_completed(futures):
        output[futures[future]] = future.result(timeout=180)
    if any(item is None for item in output):
        raise RuntimeError("a world task returned no record")
    return [item for item in output if item is not None]


def _unit_path(destination: Path, cohort: str, condition_id: str, seed: int) -> Path:
    return destination / cohort / "units" / f"{condition_id}__{seed}.json"


def _append_index(destination: Path, cohort: str, record: dict[str, Any], digest: str) -> None:
    line = {
        "cohort": cohort,
        "condition_id": record["condition_id"],
        "cell_id": record["cell_id"],
        "choice_seed": record["choice_seed"],
        "social_weight": record["social_weight"],
        "file_sha256": digest,
        "trajectory_sha256": record["trajectory_sha256"],
        "llm_calls": 0,
    }
    append_jsonl(destination / cohort / "unit_index.jsonl", line)


def _compact_world(record: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "cohort",
        "condition_id",
        "cell_id",
        "role",
        "target_id",
        "choice_seed",
        "social_weight",
        "trajectory_sha256",
        "late_hhi",
        "late_entropy",
        "late_dominant_share",
        "movement_rate",
        "alternation_rate",
        "consecutive_occupancy_tv",
        "saturation_frequency",
        "empty_frequency",
        "clip_change_count",
        "nonfinite_count",
        "max_abs_raw",
        "max_abs_unclipped",
    )
    compact = {key: record[key] for key in keys}
    compact["llm_calls"] = 0
    return compact


def _write_world(destination: Path, cohort: str, record: dict[str, Any]) -> None:
    layer_a = record.get("layer_a")
    path = _unit_path(destination, cohort, record["condition_id"], int(record["choice_seed"]))
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, record)
    _append_index(destination, cohort, record, file_sha256(path))
    append_jsonl(destination / cohort / "world_metrics.jsonl", _compact_world(record))
    if layer_a is not None:
        for row in layer_a:
            row["cohort"] = cohort
            append_jsonl(destination / cohort / "layer_a_seed_metrics.jsonl", row)


def _lineage_tasks(cohort: str, seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    tasks = []
    for seed in seeds:
        for cell_id in list(SOCIAL_CELL_IDS) + ["S0"]:
            tasks.append(
                {
                    "cohort": cohort,
                    "condition_id": cell_id,
                    "cell_id": cell_id,
                    "role": "lineage",
                    "target_id": None,
                    "choice_seed": seed,
                    "social_weight": S0_WEIGHT if cell_id == "S0" else REFERENCE_SOCIAL_WEIGHT,
                    "keep_series": False,
                    "layer_a": False,
                }
            )
    return tasks


def _load_017_hashes() -> dict[tuple[str, str, int], str]:
    found = {}
    for cohort in ("cohort_a", "cohort_b"):
        path = WORLD_V017_DIR / cohort / "unit_index.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                found[(cohort, str(row["cell_id"]), int(row["choice_seed"]))] = str(row["trajectory_sha256"])
    return found


def _audit_lineage(destination: Path) -> dict[str, Any]:
    expected = _load_017_hashes()
    failures = []
    compared = 0
    mapping = {"lineage_a": "cohort_a", "lineage_b": "cohort_b"}
    for cohort, source in mapping.items():
        path = destination / cohort / "world_metrics.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                key = (source, row["cell_id"], int(row["choice_seed"]))
                compared += 1
                if key not in expected or expected[key] != row["trajectory_sha256"]:
                    failures.append({"cohort": cohort, "cell_id": row["cell_id"], "choice_seed": row["choice_seed"], "actual": row["trajectory_sha256"], "expected": expected.get(key)})
    return {"passed": compared == 371 and not failures, "compared": compared, "failures": failures, "llm_calls": 0}


def _heldout_tasks(seed: int, specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tasks = []
    for spec in specs:
        tasks.append(
            {
                "cohort": "heldout",
                "condition_id": spec["condition_id"],
                "cell_id": spec["cell_id"],
                "role": spec["role"],
                "target_id": spec["target_id"],
                "choice_seed": seed,
                "social_weight": spec["social_weight"],
                "keep_series": spec["condition_id"] == "S0",
                "layer_a": spec["condition_id"] == "S0",
                "layer_specs": specs,
            }
        )
    return tasks


def _replay(destination: Path, specs: list[dict[str, Any]], workers: int) -> dict[str, Any]:
    tasks = []
    for seed in REPLAY_SEEDS:
        for spec in specs:
            tasks.append(
                {
                    "cohort": "heldout",
                    "condition_id": spec["condition_id"],
                    "cell_id": spec["cell_id"],
                    "role": spec["role"],
                    "target_id": spec["target_id"],
                    "choice_seed": seed,
                    "social_weight": spec["social_weight"],
                    "keep_series": False,
                    "layer_a": False,
                }
            )
    results = []
    for offset in range(0, len(tasks), 25):
        for record in _execute(tasks[offset : offset + 25], workers):
            path = _unit_path(destination, "heldout", record["condition_id"], int(record["choice_seed"]))
            before = file_sha256(path)
            saved = json.loads(path.read_text(encoding="utf-8"))
            after = file_sha256(path)
            results.append(
                {
                    "condition_id": record["condition_id"],
                    "choice_seed": record["choice_seed"],
                    "match": record["trajectory_sha256"] == saved["trajectory_sha256"],
                    "replay_trajectory_sha256": record["trajectory_sha256"],
                    "saved_trajectory_sha256": saved["trajectory_sha256"],
                    "saved_file_rewritten": before != after,
                    "llm_calls": 0,
                }
            )
    return {"passed": len(results) == 75 and all(item["match"] and not item["saved_file_rewritten"] for item in results), "count": len(results), "results": results, "llm_calls": 0}


def _column_table(metrics_path: Path, layer_path: Path) -> tuple[list[int], list[str], np.ndarray]:
    worlds: dict[int, dict[str, dict[str, float]]] = {}
    with metrics_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            worlds.setdefault(int(row["choice_seed"]), {})[row["condition_id"]] = row
    layers: dict[int, dict[str, dict[str, Any]]] = {}
    with layer_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            layers.setdefault(int(row["choice_seed"]), {})[row["condition_id"]] = row
    seeds = list(HELDOUT_SEEDS)
    if set(worlds) != set(seeds) or set(layers) != set(seeds):
        raise RuntimeError("held-out metrics do not cover seeds 700-1699")
    names: list[str] = []
    columns: list[list[float]] = []

    def add(name: str, sample: list[float]) -> None:
        names.append(name)
        columns.append(sample)

    for spec in condition_specs():
        if spec["condition_id"] == "S0":
            continue
        add(f"q::{spec['condition_id']}", [float(layers[seed][spec["condition_id"]]["q"]) for seed in seeds])
        for period, _start, _stop in PERIODS:
            add(f"period::{spec['condition_id']}::{period}", [float(layers[seed][spec["condition_id"]]["periods"][period]) for seed in seeds])
    for memory in MEMORIES:
        for target in Q_TARGETS:
            key = f"tv_{memory}_{target['target_id']}"
            sample = []
            for seed in seeds:
                row = layers[seed][f"{memory}_LOGMAX__{target['target_id']}"]
                sample.append(float(row["probability_tv"][key]))
            add(f"tv::{key}", sample)
        key = f"tv_{memory}_w150"
        sample = []
        for seed in seeds:
            row = layers[seed][f"{memory}_LOGMAX__w150"]
            sample.append(float(row["probability_tv"][key]))
        add(f"tv::{key}", sample)
    for spec in condition_specs():
        add(f"hhi::{spec['condition_id']}", [float(worlds[seed][spec["condition_id"]]["late_hhi"]) for seed in seeds])
    q_by_seed = [{spec["condition_id"]: float(layers[seed][spec["condition_id"]]["q"]) for spec in condition_specs() if spec["condition_id"] != "S0"} for seed in seeds]
    hhi_by_seed = [{spec["condition_id"]: float(worlds[seed][spec["condition_id"]]["late_hhi"]) for spec in condition_specs()} for seed in seeds]
    q_names = list(q_contrasts(q_by_seed[0]))
    for name in q_names:
        add(f"qcontrast::{name}", [q_contrasts(row)[name] for row in q_by_seed])
    hhi_names = list(hhi_contrasts(hhi_by_seed[0]))
    for name in hhi_names:
        add(f"hhicontrast::{name}", [hhi_contrasts(row)[name] for row in hhi_by_seed])
    matrix = np.column_stack(columns)
    return seeds, names, matrix


def _analyze(destination: Path) -> dict[str, Any]:
    _seeds, names, matrix = _column_table(
        destination / "heldout" / "world_metrics.jsonl",
        destination / "heldout" / "layer_a_seed_metrics.jsonl",
    )
    replicates = bootstrap_means(matrix, ANALYSIS_SEED)
    summaries = {}
    for index, name in enumerate(names):
        summaries[name] = distribution_summary(matrix[:, index].tolist(), replicates[:, index].tolist())
    pair_summaries = {key.split("qcontrast::", 1)[1]: value for key, value in summaries.items() if key.startswith("qcontrast::")}
    flags = matching_flags(
        {
            **{key.split("q::", 1)[1]: value for key, value in summaries.items() if key.startswith("q::") and key.count("::") == 1},
            **pair_summaries,
        }
    )
    payload = {
        "analysis_rng_seed": ANALYSIS_SEED,
        "draws": BOOTSTRAP_DRAWS,
        "percentile_indexes": list(percentile_indexes(BOOTSTRAP_DRAWS)),
        "index_meaning": "position in seeds 700-1699; index 0 is choice seed 700",
        "seed_list_head": HELDOUT_SEEDS[0],
        "includes_refitting_uncertainty": False,
        "summaries": summaries,
        "matching": flags,
        "llm_calls": 0,
    }
    write_json(destination / "uncertainty_analysis.json", payload)
    write_json(destination / "matching_diagnostics.json", flags)
    return payload


def _agent_summary(destination: Path) -> dict[str, Any]:
    pooled: dict[str, list[float]] = {}
    within: dict[str, list[float]] = {}
    path = destination / "heldout" / "layer_a_seed_metrics.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            pooled.setdefault(row["condition_id"], []).extend(float(value) for value in row["agent_means"])
            within.setdefault(row["condition_id"], []).append(float(row["agent_sd"]))
    payload = {
        condition_id: {
            "agent_mean_distribution": distribution_summary(values),
            "mean_within_seed_agent_sd": statistics.fmean(within[condition_id]),
        }
        for condition_id, values in sorted(pooled.items())
    }
    write_json(destination / "agent_q_distribution.json", payload)
    return payload


def _diagnostics(destination: Path) -> None:
    clip_changes = 0
    nonfinite = 0
    rows = 0
    max_hhi = None
    min_entropy = None
    for cohort in ("lineage_a", "lineage_b", "heldout"):
        path = destination / cohort / "world_metrics.jsonl"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                rows += 1
                clip_changes += int(row["clip_change_count"])
                nonfinite += int(row["nonfinite_count"])
                if row["late_hhi"] is not None:
                    max_hhi = row["late_hhi"] if max_hhi is None else max(max_hhi, row["late_hhi"])
                if row["late_entropy"] is not None:
                    min_entropy = row["late_entropy"] if min_entropy is None else min(min_entropy, row["late_entropy"])
    write_json(
        destination / "numerical_diagnostics.json",
        {"rows": rows, "clip_change_count": clip_changes, "nonfinite_count": nonfinite, "max_late_hhi": max_hhi, "min_late_entropy": min_entropy, "llm_calls": 0},
    )


def _write_figures(destination: Path, analysis: dict[str, Any]) -> None:
    status: dict[str, Any] = {"written": False, "llm_calls": 0}
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import font_manager
        from matplotlib import pyplot as plt
    except ImportError as exc:
        status["reason"] = f"matplotlib unavailable: {exc}"
        write_json(destination / "figure_status.json", status)
        return
    try:
        _render_figures(destination, analysis, plt, font_manager, status)
    except Exception as exc:
        status["written"] = False
        status["reason"] = str(exc)
    write_json(destination / "figure_status.json", status)


def _render_figures(destination: Path, analysis: dict[str, Any], plt: Any, font_manager: Any, status: dict[str, Any]) -> None:
    names = {font.name for font in font_manager.fontManager.ttflist}
    font = "Microsoft JhengHei" if "Microsoft JhengHei" in names else "DejaVu Sans"
    plt.rcParams["font.sans-serif"] = [font]
    plt.rcParams["axes.unicode_minus"] = False
    figure_dir = destination / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    summaries = analysis["summaries"]
    _figure_weights(destination, plt, figure_dir / "calibration_q.png")
    _figure_periods(plt, summaries, figure_dir / "period_q.png")
    _figure_hhi(destination, plt, figure_dir / "late_hhi.png")
    _figure_contrasts(plt, summaries, figure_dir / "hhi_contrasts.png")
    _figure_q_box(destination, plt, figure_dir / "q_distribution.png")
    status.update({"written": True, "font": font})
    write_json(destination / "figure_status.json", status)


def _figure_weights(destination: Path, plt: Any, path: Path) -> None:
    weights = json.loads((destination / "calibrated_weights.json").read_text(encoding="utf-8"))
    labels = []
    calibration = []
    heldout = []
    targets = []
    analysis = json.loads((destination / "uncertainty_analysis.json").read_text(encoding="utf-8"))
    for item in weights["conditions"]:
        labels.append(item["condition_id"].replace("__", "\n"))
        calibration.append(item["calibration_mean_q"])
        heldout.append(analysis["summaries"][f"q::{item['condition_id']}"]["mean"])
        targets.append(item["q_target"])
    positions = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(14, 5))
    axis.scatter(positions - 0.1, calibration, label="calibration mean Q")
    axis.scatter(positions + 0.1, heldout, label="held-out mean Q")
    axis.scatter(positions, targets, marker="_", s=200, label="target")
    axis.set_xticks(positions, labels, rotation=90, fontsize=7)
    axis.set_ylabel("Q")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _figure_periods(plt: Any, summaries: dict[str, Any], path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    xs = np.arange(len(PERIODS))
    for axis, target in zip(axes, Q_TARGETS):
        for cell_id in SOCIAL_CELL_IDS:
            condition_id = f"{cell_id}__{target['target_id']}"
            ys = [summaries[f"period::{condition_id}::{name}"]["mean"] for name, _start, _stop in PERIODS]
            axis.plot(xs, ys, label=cell_id)
        axis.axhline(target["q_target"], color="black", linewidth=0.8)
        axis.set_title(target["target_id"])
        axis.set_xticks(xs, [name for name, _start, _stop in PERIODS], rotation=45, fontsize=7)
    axes[0].legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _figure_hhi(destination: Path, plt: Any, path: Path) -> None:
    grouped: dict[str, list[float]] = {}
    with (destination / "heldout" / "world_metrics.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            grouped.setdefault(row["condition_id"], []).append(float(row["late_hhi"]))
    labels = [spec["condition_id"] for spec in condition_specs()]
    figure, axis = plt.subplots(figsize=(14, 5))
    axis.boxplot([grouped[label] for label in labels], tick_labels=[label.replace("__", "\n") for label in labels], showfliers=False)
    axis.set_ylabel("late HHI")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _figure_contrasts(plt: Any, summaries: dict[str, Any], path: Path) -> None:
    names = [f"D_matched_{memory}_{target['target_id']}" for memory in MEMORIES for target in Q_TARGETS]
    names += [f"D_150_{memory}" for memory in MEMORIES]
    names += [f"D_diff_{memory}_{target['target_id']}" for memory in MEMORIES for target in Q_TARGETS]
    means = [summaries[f"hhicontrast::{name}"]["mean"] for name in names]
    low = [summaries[f"hhicontrast::{name}"]["ci_low"] for name in names]
    high = [summaries[f"hhicontrast::{name}"]["ci_high"] for name in names]
    positions = np.arange(len(names))
    figure, axis = plt.subplots(figsize=(14, 5))
    axis.errorbar(positions, means, yerr=[np.array(means) - np.array(low), np.array(high) - np.array(means)], fmt="o")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(positions, [name.replace("_", "\n") for name in names], rotation=90, fontsize=6)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _figure_q_box(destination: Path, plt: Any, path: Path) -> None:
    grouped: dict[str, list[float]] = {}
    with (destination / "heldout" / "layer_a_seed_metrics.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            grouped.setdefault(row["condition_id"], []).append(float(row["q"]))
    labels = [spec["condition_id"] for spec in condition_specs() if spec["condition_id"] != "S0"]
    figure, axis = plt.subplots(figsize=(14, 5))
    axis.boxplot([grouped[label] for label in labels], tick_labels=[label.replace("__", "\n") for label in labels], showfliers=False)
    axis.set_ylabel("seed-level Q")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _fmt(value: float) -> str:
    return f"{value:.6g}"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    rule = "| " + " | ".join("---" for _header in headers) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([head, rule, *body])


def write_report(destination: Path, analysis: dict[str, Any], elapsed: float) -> None:
    weights = json.loads((destination / "calibrated_weights.json").read_text(encoding="utf-8"))
    status = json.loads((destination / "completion_status.json").read_text(encoding="utf-8"))
    lineage = json.loads((destination / "lineage_audit.json").read_text(encoding="utf-8"))
    replay = json.loads((destination / "replay_audit.json").read_text(encoding="utf-8"))
    numerical = json.loads((destination / "numerical_diagnostics.json").read_text(encoding="utf-8"))
    agent = json.loads((destination / "agent_q_distribution.json").read_text(encoding="utf-8"))
    summaries = analysis["summaries"]
    weight_rows = []
    for item in weights["conditions"]:
        held = summaries[f"q::{item['condition_id']}"]
        weight_rows.append([item["condition_id"], _fmt(item["weight"]), _fmt(item["calibration_mean_q"]), _fmt(held["mean"]), f"[{_fmt(held['ci_low'])}, {_fmt(held['ci_high'])}]"])
    contrast_rows = []
    for name in [key.split("hhicontrast::", 1)[1] for key in summaries if key.startswith("hhicontrast::")]:
        item = summaries[f"hhicontrast::{name}"]
        contrast_rows.append([name, _fmt(item["mean"]), f"[{_fmt(item['ci_low'])}, {_fmt(item['ci_high'])}]", _fmt(item["median"])])
    period_rows = []
    for spec in condition_specs():
        if spec["role"] != "matched":
            continue
        values = [_fmt(summaries[f"period::{spec['condition_id']}::{name}"]["mean"]) for name, _start, _stop in PERIODS]
        period_rows.append([spec["condition_id"], *values])
    text = "\n".join(
        [
            "# Experiment 018 報告",
            "",
            f"狀態：{status['status']}。這是把六種 memory × transformation 的平均 Q 校準到 0.05、0.15、0.25 之後，在 seeds 700–1699 上做的閉環比較。",
            "",
            "比較的是經過平均 Q 校準的整套決策規則。平均 Q 相同，不代表選擇機率向量、時間分布、agent 分布或閉環中的實際影響相同。",
            "這不是純尺度效應的完整識別，不是每個 agent 或 timestep 的影響完全相等，不是純記憶時間尺度，不是真人行為校準，也不是路徑依賴或外部效度。",
            "HHI 差異不是歷史持續或路徑依賴。D_matched − D_150 描述校準後轉換差距的改變，不是可分配的因果占比。",
            "區間是已固定 weights 之下、新 stochastic seeds 的 pointwise 95% interval。它不包含重新 fitting weights 的不確定性。區間包含 0 不是等價證明。",
            "",
            f"Lineage passed={lineage['passed']}，compared={lineage['compared']}。Replay passed={replay['passed']}，count={replay['count']}。",
            f"世界執行計數：lineage 371、新世界 25000、replay 75、合計 25446。LLM calls = 0。正式執行器耗時 {elapsed:.1f} 秒。",
            "",
            "## 校準 weights",
            "",
            _table(["condition", "weight", "calibration Q", "held-out Q", "held-out CI"], weight_rows),
            "",
            "## 分時段 Q",
            "",
            _table(["condition", *[name for name, _start, _stop in PERIODS]], period_rows),
            "",
            "## HHI contrasts",
            "",
            _table(["contrast", "mean", "CI", "median"], contrast_rows),
            "",
            "## 匹配診斷",
            "",
            "目標帶是 target ± 0.005。配對 Q 差的帶是 [-0.005, 0.005]。這是操作性診斷。",
            "",
            _table(
                ["condition", "abs error", "CI inside band"],
                [[row["condition_id"], _fmt(row["abs_error"]), str(row["ci_inside_match_band"])] for row in analysis["matching"]["cells"]],
            ),
            "",
            _table(
                ["contrast", "mean", "CI inside band"],
                [[row["contrast"], _fmt(row["mean"]), str(row["ci_inside_match_band"])] for row in analysis["matching"]["paired_q_differences"]],
            ),
            "",
            "## Agent-level Q",
            "",
            "下表是各 condition 把每個 seed 的 20 個 agent 平均 Q 放在一起後的分布，另給 seed 內 agent 標準差的平均。",
            "",
            _table(
                ["condition", "agent mean", "agent sd", "q25", "q75", "mean within-seed sd"],
                [
                    [
                        condition_id,
                        _fmt(item["agent_mean_distribution"]["mean"]),
                        _fmt(item["agent_mean_distribution"]["stdev"]),
                        _fmt(item["agent_mean_distribution"]["q25"]),
                        _fmt(item["agent_mean_distribution"]["q75"]),
                        _fmt(item["mean_within_seed_agent_sd"]),
                    ]
                    for condition_id, item in agent.items()
                    if condition_id.endswith(("q05", "q15", "q25"))
                ],
            ),
            "",
            f"數值診斷：clip_change_count={numerical['clip_change_count']}，nonfinite_count={numerical['nonfinite_count']}。ZRAW 的分數標準差接近 1 與五場所 |z| 不超過 2 是公式性質。",
            "",
            "能分開的是：在同一組新 seeds 上，固定 weight 150 的轉換差距，以及把平均 Q 調到三個目標之後剩下的轉換差距。",
            "不能識別的是：純尺度、純時間尺度、回饋占比、路徑依賴、真人行為與外部效度，以及把 weights 重新 fitting 進去之後的兩階段誤差。",
            "",
        ]
    )
    (destination / "EXPERIMENT_018_REPORT.md").write_text(text, encoding="utf-8", newline="\n")
    REPORT_PATH.write_text(text, encoding="utf-8", newline="\n")


def _manifest(destination: Path) -> None:
    files = {}
    for path in sorted(destination.rglob("*")):
        if path.is_file() and path.name != "run_manifest.json":
            files[_display(path).split("world_v0_18/", 1)[-1]] = file_sha256(path)
    write_json(destination / "run_manifest.json", {"world_id": "world_v0_18", "files": files, "llm_calls": 0})


def _status(destination: Path, status: str, reason: str, started: float, **extra: Any) -> None:
    payload = {"status": status, "reason": reason, "elapsed_seconds": time.perf_counter() - started, "llm_calls": 0}
    payload.update(extra)
    write_json(destination / "completion_status.json", payload)


def _record_failure(destination: Path, stage: str, exc: BaseException, started: float) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    write_json(
        destination / "failure_record.json",
        {"stage": stage, "error": str(exc), "traceback": traceback.format_exc(), "elapsed_seconds": time.perf_counter() - started, "llm_calls": 0},
    )
    _status(destination, "INCOMPLETE", stage, started)


def _mark(destination: Path, stage: str, started: float, **extra: Any) -> None:
    payload = {"stage": stage, "elapsed_seconds": time.perf_counter() - started, "llm_calls": 0}
    payload.update(extra)
    append_jsonl(destination / "progress.jsonl", payload)
    print(f"{stage} elapsed={payload['elapsed_seconds']:.1f}s {extra}", flush=True)


def _require_atlas() -> None:
    manifest = json.loads((ATLAS_DIR / "atlas_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("sha256") != EXPECTED_ATLAS_SHA256:
        raise RuntimeError("utility atlas hash is not the frozen experiment atlas")


def _update_indexes(status: str) -> None:
    pointer = f"- 018 狀態為 {status}。見 [`docs/EXPERIMENT_018_REPORT.md`](docs/EXPERIMENT_018_REPORT.md)。協議見 [`docs/EXPERIMENT_018_PROTOCOL.md`](docs/EXPERIMENT_018_PROTOCOL.md)。"
    readme = PROJECT_ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    if "018 狀態為" not in text:
        text = text.replace(
            "- 017 已完成正式執行。見 [`docs/EXPERIMENT_017_REPORT.md`](docs/EXPERIMENT_017_REPORT.md)。設計稿仍是當時的 DESIGN_ONLY / NOT_RUN 紀錄\n",
            "- 017 已完成正式執行。見 [`docs/EXPERIMENT_017_REPORT.md`](docs/EXPERIMENT_017_REPORT.md)。設計稿仍是當時的 DESIGN_ONLY / NOT_RUN 紀錄\n" + pointer + "\n",
        )
        readme.write_text(text, encoding="utf-8", newline="\n")
    experiments = PROJECT_ROOT / "docs" / "EXPERIMENTS.md"
    index = experiments.read_text(encoding="utf-8")
    if "Experiment 018" not in index:
        block = (
            "\n### Experiment 018 — mean decision-impact matching\n\n"
            f"018 狀態為 {status}。協議見 [`EXPERIMENT_018_PROTOCOL.md`](EXPERIMENT_018_PROTOCOL.md)。結果見 [`EXPERIMENT_018_REPORT.md`](EXPERIMENT_018_REPORT.md)。"
            "Experiment 001–017 的輸出沒有因這次執行而改寫。\n\n"
        )
        index = index.replace("\n## 輸出契約\n", block + "## 輸出契約\n")
        experiments.write_text(index, encoding="utf-8", newline="\n")


def run_experiment(output_dir: Path | None = None, *, resume: bool = False, workers: int | None = None) -> Path:
    destination = output_dir or OUTPUT_DIR
    if destination.resolve() in {WORLD_V017_DIR.resolve(), ATLAS_DIR.resolve()}:
        raise RuntimeError("experiment 018 must not write into a frozen source directory")
    started = time.perf_counter()
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    deadline = time.time() + BUDGET_SECONDS - RESERVE_SECONDS
    worker_count = workers if workers is not None else min(12, max(1, (os.cpu_count() or 2) - 1))
    stage = "startup"
    try:
        _require_atlas()
        if resume:
            protocol = json.loads((destination / "preregistered_protocol.json").read_text(encoding="utf-8"))
            if protocol["source_sha256"]["world/experiment_018.py"] != file_sha256(Path(__file__).resolve()):
                raise RuntimeError("resume stopped because experiment_018.py no longer matches the frozen protocol")
            if file_sha256(destination / "seed_audit.json") != protocol["seed_audit_sha256"]:
                raise RuntimeError("resume stopped because the pre-run seed audit hash changed")
        else:
            refuse_nonempty_output(destination)
            destination.mkdir(parents=True, exist_ok=True)
            stage = "seed_audit"
            audit = audit_choice_seeds()
            write_json(destination / "seed_audit.json", audit)
            if not audit["passed"]:
                _status(destination, "INCOMPLETE", "seed_audit_overlap", started)
                raise RuntimeError("held-out choice seeds overlap an existing choice seed; seeds were not replaced")
            stage = "protocol"
            protocol = build_protocol(file_sha256(destination / "seed_audit.json"), started_at)
            write_json(destination / "preregistered_protocol.json", protocol)
            PROTOCOL_PATH.write_text(protocol_markdown(protocol), encoding="utf-8", newline="\n")
            write_json(
                destination / "source_manifest.json",
                {
                    "record_type": "experiment_018_source_manifest",
                    "not_an_experiment_run_manifest": True,
                    "source_sha256": protocol["source_sha256"],
                    "utility_atlas_sha256": EXPECTED_ATLAS_SHA256,
                    "experiment_017_config_sha256": protocol["experiment_017_config_sha256"],
                    "llm_calls": 0,
                },
            )
        _mark(destination, "protocol_ready", started, workers=worker_count)
        if not (destination / "lineage_audit.json").exists():
            stage = "lineage"
            for cohort, seeds in (("lineage_a", LINEAGE_A_SEEDS), ("lineage_b", LINEAGE_B_SEEDS)):
                tasks = [task for task in _lineage_tasks(cohort, seeds) if not _unit_path(destination, cohort, task["condition_id"], task["choice_seed"]).exists()]
                for offset in range(0, len(tasks), 36):
                    for record in _execute(tasks[offset : offset + 36], worker_count):
                        _write_world(destination, cohort, record)
                    _mark(destination, "lineage_batch", started, cohort=cohort, remaining=max(0, len(tasks) - offset - 36))
            lineage = _audit_lineage(destination)
            write_json(destination / "lineage_audit.json", lineage)
            if not lineage["passed"]:
                _status(destination, "INCOMPLETE", "lineage_mismatch", started, lineage_compared=lineage["compared"])
                raise RuntimeError("historical lineage hash mismatch; calibration did not start")
        _mark(destination, "lineage_passed", started)
        if not (destination / "world_v0_18_config.json").exists():
            stage = "calibration_cache"
            cache_dir = destination / "calibration_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            missing = [(seed, str(cache_dir)) for seed in CALIBRATION_SEEDS if not (cache_dir / f"{seed}.npz").exists()]
            if missing:
                pool = _shared_pool(worker_count)
                futures = [pool.submit(_cache_task, task) for task in missing]
                for index, future in enumerate(as_completed(futures), start=1):
                    future.result(timeout=180)
                    if index % 50 == 0:
                        _mark(destination, "cache_progress", started, done=index, total=len(missing))
            bases, scores = _load_cache_stack(cache_dir, CALIBRATION_SEEDS)
            stage = "q_compatibility"
            compatibility = _compare_q(bases, scores)
            write_json(destination / "q_compatibility.json", compatibility)
            if not compatibility["passed"]:
                _status(destination, "INCOMPLETE", "q_compatibility", started)
                raise RuntimeError("cached Q at weight 150 does not match Experiment 017 within 1e-10")
            _mark(destination, "q_compatibility_passed", started, max_abs_error=compatibility["max_abs_error"])
            stage = "solver"

            def evaluate_cell(cell_id: str, weight: float) -> float:
                cell_index = SOCIAL_CELL_IDS.index(cell_id)
                return float(cohort_seed_q(bases, scores[:, cell_index], weight).mean())

            fitted = fit_targets(evaluate_cell)
            trace = {key: {"ok": value["ok"], "reason": value["reason"], "trace": value["trace"], "q_target": value["q_target"]} for key, value in fitted["weights"].items()}
            write_json(destination / "calibration_trace.json", {"conditions": trace, "ok": fitted["ok"], "llm_calls": 0})
            if not fitted["ok"]:
                _status(destination, "INCOMPLETE", "calibration_failed", started, failed_condition=fitted["failed_condition"])
                raise RuntimeError(f"calibration failed at {fitted['failed_condition']}")
            weight_rows = []
            seed_rows = []
            for condition_id, outcome in fitted["weights"].items():
                cell_index = SOCIAL_CELL_IDS.index(outcome["cell_id"])
                seed_q = cohort_seed_q(bases, scores[:, cell_index], float(outcome["weight"]))
                weight_rows.append(
                    {
                        "condition_id": condition_id,
                        "cell_id": outcome["cell_id"],
                        "target_id": outcome["target_id"],
                        "q_target": outcome["q_target"],
                        "weight": float(outcome["weight"]),
                        "weight_hex": float(outcome["weight"]).hex(),
                        "calibration_mean_q": float(seed_q.mean()),
                        "abs_error": abs(float(seed_q.mean()) - float(outcome["q_target"])),
                    }
                )
                for seed_index, seed in enumerate(CALIBRATION_SEEDS):
                    seed_rows.append({"condition_id": condition_id, "choice_seed": seed, "q": float(seed_q[seed_index]), "llm_calls": 0})
            weights_payload = {"conditions": weight_rows, "llm_calls": 0}
            write_json(destination / "calibrated_weights.json", weights_payload)
            for row in seed_rows:
                append_jsonl(destination / "calibration_seed_q.jsonl", row)
            config = {
                "world_id": "world_v0_18",
                "experiment_id": "experiment_018_mean_decision_impact_matching",
                "status_at_config_freeze": "CONFIG_FROZEN_BEFORE_HELDOUT",
                "protocol_sha256": protocol["protocol_sha256"],
                "seed_audit_sha256": protocol["seed_audit_sha256"],
                "calibration_trace_sha256": file_sha256(destination / "calibration_trace.json"),
                "calibrated_weights_sha256": file_sha256(destination / "calibrated_weights.json"),
                "q_compatibility_sha256": file_sha256(destination / "q_compatibility.json"),
                "weights": weight_rows,
                "source_sha256": protocol["source_sha256"],
                "utility_atlas_sha256": EXPECTED_ATLAS_SHA256,
                "python": sys.version,
                "platform": platform.platform(),
                "social_weight_module_default": SOCIAL_WEIGHT,
                "llm_calls": 0,
                "started_at": started_at,
            }
            config["config_sha256"] = canonical_sha256(config)
            write_json(destination / "world_v0_18_config.json", config)
            del bases, scores
        _shutdown_pool()
        config = json.loads((destination / "world_v0_18_config.json").read_text(encoding="utf-8"))
        if any(unit.exists() for unit in (destination / "heldout" / "units").glob("*.json")) if (destination / "heldout" / "units").exists() else False:
            if config["status_at_config_freeze"] != "CONFIG_FROZEN_BEFORE_HELDOUT":
                raise RuntimeError("held-out units exist without a frozen config")
        weight_values = {row["condition_id"]: float.fromhex(row["weight_hex"]) for row in config["weights"]}
        specs = condition_specs(weight_values)
        assert_weight_schedule(specs)
        _mark(destination, "config_frozen", started)
        stage = "heldout"
        pending_seeds = [seed for seed in HELDOUT_SEEDS if any(not _unit_path(destination, "heldout", spec["condition_id"], seed).exists() for spec in specs)]
        for seed_index, seed in enumerate(pending_seeds):
            if time.time() > deadline:
                _status(destination, "INCOMPLETE", "budget", started, completed_heldout_seeds=len(HELDOUT_SEEDS) - len(pending_seeds) + seed_index)
                raise RuntimeError("budget reserve reached at a seed boundary")
            missing = [task for task in _heldout_tasks(seed, specs) if not _unit_path(destination, "heldout", task["condition_id"], seed).exists()]
            for record in _execute(missing, worker_count):
                _write_world(destination, "heldout", record)
            if (seed_index + 1) % 10 == 0:
                elapsed = time.perf_counter() - started
                rate = (seed_index + 1) / elapsed
                _mark(destination, "heldout_progress", started, seeds_done=seed_index + 1, seeds_total=len(pending_seeds), eta_seconds=(len(pending_seeds) - seed_index - 1) / rate)
        stage = "analysis"
        _shutdown_pool()
        analysis = _analyze(destination)
        _agent_summary(destination)
        _diagnostics(destination)
        stage = "replay"
        replay = _replay(destination, specs, worker_count)
        write_json(destination / "replay_audit.json", replay)
        _shutdown_pool()
        if not replay["passed"]:
            _status(destination, "INCOMPLETE", "replay_mismatch", started)
            raise RuntimeError("replay did not match the saved trajectory hash")
        lineage = json.loads((destination / "lineage_audit.json").read_text(encoding="utf-8"))
        decision = completion_decision(lineage_passed=bool(lineage["passed"]), replay_passed=True, worlds_complete=True, calibration_passed=True)
        _status(destination, decision, "verified", started, logical_new_worlds=25000, lineage_worlds=371, replays=75)
        _write_figures(destination, analysis)
        write_report(destination, analysis, time.perf_counter() - started)
        _manifest(destination)
        _update_indexes(decision)
        _mark(destination, "finished", started, status=decision)
        return destination
    except Exception as exc:
        _shutdown_pool()
        if not (destination / "failure_record.json").exists():
            _record_failure(destination, stage, exc, started)
        raise


def main() -> None:
    run_experiment()


if __name__ == "__main__":
    main()
