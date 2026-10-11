"""Experiment 019: closed-loop decision impact and cross-state audit.

This run replays Experiment 018's frozen seeds and weights. It was specified
after those results were known. It does not refit weights, add seeds, or call
a model.
"""

from __future__ import annotations

import json
import math
import os
import platform
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from random import Random
from typing import Any

import numpy as np

from agents.persona_generator import append_jsonl, write_json
from config.settings import PROJECT_ROOT
from experiments.reproducibility_check import canonical_sha256
from world.dynamics_calibration import load_initial_locations
from world.experiment_002 import REFERENCE_CHOICE_TEMPERATURE
from world.experiment_004 import _entropy, _hhi, _select, _share, _unique_leader, trajectory_hash
from world.experiment_005 import _presence
from world.experiment_017 import (
    SOCIAL_CELL_IDS,
    _base_utilities,
    _commit,
    file_sha256,
    make_memory,
    pack_series,
    score_memory,
    simulate_cell,
    unpack_series,
)
from world.experiment_018 import HELDOUT_SEEDS, condition_specs
from world.experiment_018 import OUTPUT_DIR as WORLD_V018_DIR
from world.geography import PLACE_IDS, distance_matrix, occupancy_of, place_index
from world.utility_atlas import OUTPUT_DIR as ATLAS_DIR
from world.utility_atlas import load_frozen_atlas

OUTPUT_DIR = PROJECT_ROOT / "data" / "worlds" / "world_v0_19"
PROTOCOL_DOC = PROJECT_ROOT / "docs" / "EXPERIMENT_019_PROTOCOL.md"
REPORT_DOC = PROJECT_ROOT / "docs" / "EXPERIMENT_019_REPORT.md"
EXPECTED_ATLAS_SHA256 = "664b511d208f7989e8a671d0e2d6e6433366120bb3d5d6a67db38277b4d5a373"
TIMESTEPS = 500
LATE_START = 101
REFERENCE_SOCIAL_WEIGHT = 150.0
Q_TOLERANCE = 1e-10
HHI_TOLERANCE = 1e-12
REPLAY_FLOAT_TOLERANCE = 1e-12
PROBABILITY_TOLERANCE = 1e-12
BOOTSTRAP_DRAWS = 10_000
ANALYSIS_SEED = 20261001
BUDGET_SECONDS = 5 * 60 * 60
RESERVE_SECONDS = 40 * 60
MEMORIES = ("CUM", "RW50", "EXP99")
TRANSFORMS = ("LOGMAX", "ZRAW")
SETTINGS = ("q05", "q15", "q25", "w150")
MATCHED_SETTINGS = ("q05", "q15", "q25")
HISTORIES = ("H0", "HL", "HZ")
GATE_SEEDS = (700, 1199, 1699)
ALIGNMENT_OFFSETS = tuple(range(-40, 50))
WINDOWS = (
    ("t1_500", 0, 500),
    ("t1_100", 0, 100),
    ("t101_200", 100, 200),
    ("t201_300", 200, 300),
    ("t301_400", 300, 400),
    ("t401_500", 400, 500),
    ("t101_500", 100, 500),
)
MEMORY_SPEC = {
    "CUM": {"memory": "cumulative", "window": None, "decay": None},
    "RW50": {"memory": "rolling", "window": 50, "decay": None},
    "EXP99": {"memory": "exponential", "window": None, "decay": 0.99},
}
_WORKER: dict[str, Any] = {}
_POOL: ProcessPoolExecutor | None = None
PERIOD_PLOT = ("t1_100", "t101_200", "t201_300", "t301_400", "t401_500")


class BudgetStop(RuntimeError):
    """The reserve was reached on a completed seed boundary."""


def percentile_indexes(draws: int) -> tuple[int, int]:
    return int(0.025 * (draws - 1)), int(0.975 * (draws - 1))


def refuse_nonempty_output(destination: Path) -> None:
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError("experiment 019 refuses to write into a non-empty output directory")


def resolve_source_path(raw: str) -> Path:
    normalized = str(raw).replace("\\", "/")
    path = Path(normalized)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def completion_decision(*, hashes_passed: bool, compatibility_passed: bool, replay_passed: bool, seeds_complete: bool) -> str:
    if hashes_passed and compatibility_passed and replay_passed and seeds_complete:
        return "COMPLETE"
    return "INCOMPLETE"


def condition_id(memory: str, transform: str, setting: str) -> str:
    return f"{memory}_{transform}__{setting}"


def load_frozen_weights() -> dict[str, float]:
    weights_path = WORLD_V018_DIR / "calibrated_weights.json"
    config_path = WORLD_V018_DIR / "world_v0_18_config.json"
    payload = json.loads(weights_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    from_weights = {row["condition_id"]: float.fromhex(row["weight_hex"]) for row in payload["conditions"]}
    from_config = {row["condition_id"]: float.fromhex(row["weight_hex"]) for row in config["weights"]}
    if from_weights != from_config:
        raise RuntimeError("Experiment 018 weight files do not agree at full precision")
    if len(from_weights) != 18:
        raise RuntimeError("Experiment 018 did not supply 18 calibrated weights")
    resolved = dict(from_weights)
    for memory in MEMORIES:
        for transform in TRANSFORMS:
            resolved[condition_id(memory, transform, "w150")] = REFERENCE_SOCIAL_WEIGHT
    resolved["S0"] = 0.0
    return resolved


def frozen_specs(weights: dict[str, float]) -> list[dict[str, Any]]:
    matched = {key: value for key, value in weights.items() if key.endswith(("q05", "q15", "q25"))}
    specs = condition_specs(matched)
    for spec in specs:
        spec["social_weight"] = float(weights[spec["condition_id"]])
        spec["weight_hex"] = float(spec["social_weight"]).hex()
    if len(specs) != 25:
        raise RuntimeError("the frozen schedule is not 25 conditions")
    return specs


def require_series_copy(series: list[dict[str, str]]) -> list[dict[str, str]]:
    copied = [{agent_id: place_id for agent_id, place_id in step.items()} for step in series]
    return copied


def walk_memory(
    atlas: Any,
    series: list[dict[str, str]],
    matrix: dict[str, dict[str, float]],
    places: dict[str, dict[str, Any]],
    memory_name: str,
) -> dict[str, np.ndarray]:
    original = [{agent_id: step[agent_id] for agent_id in step} for step in series]
    agents = sorted(series[0])
    timesteps = len(series) - 1
    memory = make_memory(MEMORY_SPEC[memory_name], agents)
    bases = np.empty((timesteps, len(agents), 5), dtype=np.float64)
    raw = np.empty((timesteps, len(agents), 5), dtype=np.float64)
    logmax = np.empty((timesteps, len(agents), 5), dtype=np.float64)
    zraw = np.empty((timesteps, len(agents), 5), dtype=np.float64)
    base_cache: dict[tuple[Any, ...], dict[str, float]] = {}
    for timestep in range(timesteps):
        origins = series[timestep]
        occupancy = occupancy_of(origins)
        occupancy_key = tuple(occupancy[place_id] for place_id in PLACE_IDS)
        for agent_index, agent_id in enumerate(agents):
            cache_key = (agent_id, origins[agent_id], occupancy_key)
            base = base_cache.get(cache_key)
            if base is None:
                base = _base_utilities(atlas, matrix, places, agent_id, origins[agent_id], occupancy)
                base_cache[cache_key] = base
            _log_score, log_score, log_raw = score_memory(memory, agent_id, "LOGMAX")
            _z_unclipped, z_score, z_raw = score_memory(memory, agent_id, "ZRAW")
            for place_index, place_id in enumerate(PLACE_IDS):
                if log_raw[place_id] != z_raw[place_id]:
                    raise RuntimeError("LOGMAX and ZRAW did not share one raw memory")
                bases[timestep, agent_index, place_index] = base[place_id]
                raw[timestep, agent_index, place_index] = log_raw[place_id]
                logmax[timestep, agent_index, place_index] = log_score[place_id]
                zraw[timestep, agent_index, place_index] = z_score[place_id]
        memory = _commit(memory, series[timestep + 1])
    if series != original:
        raise RuntimeError("scoring wrote back into the source history")
    if not np.isfinite(bases).all() or not np.isfinite(raw).all() or not np.isfinite(logmax).all() or not np.isfinite(zraw).all():
        raise RuntimeError("nonfinite value in a scoring walk")
    return {"bases": bases, "raw": raw, "LOGMAX": logmax, "ZRAW": zraw}


def softmax_utilities(utilities: np.ndarray) -> np.ndarray:
    peak = np.max(utilities, axis=-1, keepdims=True)
    weights = np.exp((utilities - peak) / REFERENCE_CHOICE_TEMPERATURE)
    if not np.isfinite(weights).all():
        raise RuntimeError("nonfinite softmax weight")
    total = np.sum(weights, axis=-1, keepdims=True)
    return weights / total


def probabilities_match_select(utilities: list[float]) -> float:
    array = np.asarray(utilities, dtype=np.float64)
    got = softmax_utilities(array)
    _choice, expected = _select(list(PLACE_IDS), utilities, 0.0)
    gaps = [abs(float(got[index]) - expected[place_id]) for index, place_id in enumerate(PLACE_IDS)]
    return max(gaps)


def readout_arrays(bases: np.ndarray, scores: np.ndarray, weight: float) -> dict[str, np.ndarray]:
    if float(weight) == 0.0:
        zeros = np.zeros(bases.shape[:2], dtype=np.float64)
        base_probability = softmax_utilities(bases)
        entropy = np.zeros(bases.shape[:2], dtype=np.float64)
        positive = base_probability > 0.0
        entropy_values = np.zeros_like(base_probability)
        entropy_values[positive] = base_probability[positive] * np.log(base_probability[positive])
        entropy = -entropy_values.sum(axis=-1)
        return {
            "q": zeros,
            "logit_range": zeros,
            "social_sd": zeros.copy(),
            "max_probability": base_probability.max(axis=-1),
            "entropy": entropy,
            "probability": base_probability,
            "social": np.zeros_like(bases),
        }
    social = np.float64(weight) * scores
    probability = softmax_utilities(bases + social)
    base_probability = softmax_utilities(bases)
    q = 0.5 * np.abs(probability - base_probability).sum(axis=-1)
    centered = social - social.mean(axis=-1, keepdims=True)
    social_sd = np.sqrt((centered * centered).mean(axis=-1)) / REFERENCE_CHOICE_TEMPERATURE
    logit_range = (social.max(axis=-1) - social.min(axis=-1)) / REFERENCE_CHOICE_TEMPERATURE
    positive = probability > 0.0
    entropy_values = np.zeros_like(probability)
    entropy_values[positive] = probability[positive] * np.log(probability[positive])
    return {
        "q": q,
        "logit_range": logit_range,
        "social_sd": social_sd,
        "max_probability": probability.max(axis=-1),
        "entropy": -entropy_values.sum(axis=-1),
        "probability": probability,
        "social": social,
    }


def window_mean(values: np.ndarray, start: int, stop: int) -> float:
    piece = values[start:stop]
    if piece.size == 0 or stop > values.shape[0]:
        raise RuntimeError("diagnostic window does not fit the trajectory")
    return float(piece.mean())


def summarize_readout(bundle: dict[str, np.ndarray], raw: np.ndarray, weight: float) -> dict[str, float]:
    raw_range = raw.max(axis=-1) - raw.min(axis=-1)
    zero = raw.max(axis=-1) == raw.min(axis=-1)
    summary: dict[str, float] = {"social_weight": float(weight)}
    for name, start, stop in WINDOWS:
        summary[f"q_{name}"] = window_mean(bundle["q"], start, stop)
        summary[f"logit_range_{name}"] = window_mean(bundle["logit_range"], start, stop)
        summary[f"social_sd_{name}"] = window_mean(bundle["social_sd"], start, stop)
        summary[f"memory_mass_{name}"] = window_mean(raw.sum(axis=-1), start, stop)
        summary[f"memory_range_{name}"] = window_mean(raw_range, start, stop)
        summary[f"zero_variance_{name}"] = window_mean(zero.astype(np.float64), start, stop)
        summary[f"max_probability_{name}"] = window_mean(bundle["max_probability"], start, stop)
        summary[f"entropy_{name}"] = window_mean(bundle["entropy"], start, stop)
    return summary


def score_from_walk(walked: dict[str, np.ndarray], weights_by_transform: dict[str, float]) -> dict[str, Any]:
    bundles = {transform: readout_arrays(walked["bases"], walked[transform], weights_by_transform[transform]) for transform in TRANSFORMS}
    tv = 0.5 * np.abs(bundles["LOGMAX"]["probability"] - bundles["ZRAW"]["probability"]).sum(axis=-1)
    rows = {}
    for transform in TRANSFORMS:
        summary = summarize_readout(bundles[transform], walked["raw"], weights_by_transform[transform])
        for name, start, stop in WINDOWS:
            summary[f"tv_{name}"] = window_mean(tv, start, stop)
        if float(weights_by_transform[transform]) == 0.0 and summary["q_t1_500"] != 0.0:
            raise RuntimeError("weight 0 did not produce Q = 0")
        if float(weights_by_transform[transform]) == 0.0 and np.any(bundles[transform]["social"] != 0.0):
            raise RuntimeError("weight 0 did not produce zero social utility")
        rows[transform] = summary
    return {"rows": rows, "q": {transform: bundles[transform]["q"] for transform in TRANSFORMS}, "logit_range": {transform: bundles[transform]["logit_range"] for transform in TRANSFORMS}}


def score_history(
    atlas: Any,
    series: list[dict[str, str]],
    matrix: dict[str, dict[str, float]],
    places: dict[str, dict[str, Any]],
    memory_name: str,
    weights_by_transform: dict[str, float],
) -> dict[str, Any]:
    return score_from_walk(walk_memory(atlas, series, matrix, places, memory_name), weights_by_transform)


def late_visit_metrics(series: list[dict[str, str]]) -> dict[str, Any]:
    if len(series) != TIMESTEPS + 1:
        raise RuntimeError("late metrics require timesteps 0 through 500")
    counts = _presence(series, LATE_START, TIMESTEPS)
    share = _share(counts)
    leader = _unique_leader(counts)
    return {
        "late_hhi": _hhi(share),
        "late_entropy": _entropy(share),
        "late_dominant_share": max(share.values()),
        "late_dominant": leader,
        "tie": leader is None,
        "late_share": {place_id: share[place_id] for place_id in PLACE_IDS},
    }


def concentration_event(series: list[dict[str, str]], threshold: int, run_length: int = 20) -> dict[str, Any]:
    last = len(series) - 1
    latest_start = last - run_length + 1
    empty = {"occurred": False, "start": None, "confirm": None, "place": None, "exited": None, "threshold": threshold, "run_length": run_length}
    if latest_start < 1:
        return empty
    for start in range(1, latest_start + 1):
        counts = occupancy_of(series[start])
        candidates = [place_id for place_id in PLACE_IDS if counts[place_id] >= threshold]
        for place_id in candidates:
            if any(sum(step[agent_id] == place_id for agent_id in step) < threshold for step in series[start : start + run_length]):
                continue
            confirm = start + run_length - 1
            exited = any(sum(step[agent_id] == place_id for agent_id in step) < threshold for step in series[confirm + 1 :])
            return {"occurred": True, "start": start, "confirm": confirm, "place": place_id, "exited": exited, "threshold": threshold, "run_length": run_length}
    return empty


def place_relation(labels: list[str | None]) -> str:
    if any(label is None for label in labels):
        return "tie"
    if len(set(labels)) == 1:
        return "same"
    return "different"


def share_tv(left: dict[str, float], right: dict[str, float]) -> float:
    return 0.5 * sum(abs(left[place_id] - right[place_id]) for place_id in PLACE_IDS)


def native_history(transform: str) -> str:
    if transform == "LOGMAX":
        return "HL"
    if transform == "ZRAW":
        return "HZ"
    raise RuntimeError("unknown readout")


def paired_contrasts(rows: dict[tuple[str, str, str, str], dict[str, float]]) -> dict[str, float]:
    output: dict[str, float] = {}
    for memory in MEMORIES:
        for setting in SETTINGS:
            for transform in TRANSFORMS:
                own = native_history(transform)
                native_key = f"delta_q_{memory}_{transform}_{setting}"
                output[native_key] = rows[(memory, setting, own, transform)]["q_t101_500"] - rows[(memory, setting, "H0", transform)]["q_t101_500"]
                output[f"delta_range_{memory}_{transform}_{setting}"] = (
                    rows[(memory, setting, own, transform)]["logit_range_t101_500"] - rows[(memory, setting, "H0", transform)]["logit_range_t101_500"]
                )
                for name, _start, _stop in WINDOWS:
                    if name == "t101_500":
                        continue
                    output[f"{native_key}_{name}"] = rows[(memory, setting, own, transform)][f"q_{name}"] - rows[(memory, setting, "H0", transform)][f"q_{name}"]
            for history in HISTORIES:
                for transform in TRANSFORMS:
                    output[f"q_{memory}_{setting}_{history}_{transform}"] = rows[(memory, setting, history, transform)]["q_t101_500"]
                output[f"rule_{memory}_{setting}_{history}"] = rows[(memory, setting, history, "ZRAW")]["q_t101_500"] - rows[(memory, setting, history, "LOGMAX")]["q_t101_500"]
            for transform in TRANSFORMS:
                for history in ("HL", "HZ"):
                    output[f"history_{memory}_{transform}_{setting}_{history}"] = (
                        rows[(memory, setting, history, transform)]["q_t101_500"] - rows[(memory, setting, "H0", transform)]["q_t101_500"]
                    )
            output[f"interaction_{memory}_{setting}_HL"] = output[f"rule_{memory}_{setting}_HL"] - output[f"rule_{memory}_{setting}_H0"]
            output[f"interaction_{memory}_{setting}_HZ"] = output[f"rule_{memory}_{setting}_HZ"] - output[f"rule_{memory}_{setting}_H0"]
    return output


def shared_indexes(n_seeds: int, draws: int, rng_seed: int) -> np.ndarray:
    generator = Random(rng_seed)
    indexes = np.empty((draws, n_seeds), dtype=np.int32)
    for draw in range(draws):
        for column in range(n_seeds):
            indexes[draw, column] = generator.randrange(n_seeds)
    return indexes


def bootstrap_column(values: np.ndarray, indexes: np.ndarray) -> np.ndarray:
    output = np.empty(indexes.shape[0], dtype=np.float64)
    batch = 20
    cursor = 0
    while cursor < indexes.shape[0]:
        stop = min(cursor + batch, indexes.shape[0])
        output[cursor:stop] = values[indexes[cursor:stop]].mean(axis=1)
        cursor = stop
    return output


def distribution_summary(samples: list[float], replicates: np.ndarray | None = None) -> dict[str, Any]:
    import statistics

    quantiles = statistics.quantiles(samples, n=4, method="inclusive")
    summary: dict[str, Any] = {
        "mean": statistics.fmean(samples),
        "median": statistics.median(samples),
        "stdev": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "minimum": min(samples),
        "maximum": max(samples),
        "q25": quantiles[0],
        "q75": quantiles[2],
        "n": len(samples),
    }
    if replicates is not None:
        low, high = percentile_indexes(len(replicates))
        ranked = np.sort(replicates)
        summary["ci_low"] = float(ranked[low])
        summary["ci_high"] = float(ranked[high])
    return summary


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
        _POOL.shutdown(wait=False, cancel_futures=True)
        _POOL = None


def _seed_task(task: dict[str, Any]) -> dict[str, Any]:
    if "atlas" not in _WORKER:
        _init_worker()
    seed = int(task["choice_seed"])
    specs = task["specs"]
    expected = task["expected"]
    layer_q = task["layer_q"]
    trajectories: dict[str, list[dict[str, str]]] = {}
    packed = []
    hash_failures = []
    hhi_failures = []
    q_failures = []
    events = []
    for spec in specs:
        record = simulate_cell(
            _WORKER["atlas"],
            _WORKER["locations"],
            spec["cell_id"],
            seed,
            social_weight=float(spec["social_weight"]),
            matrix=_WORKER["matrix"],
            places=_WORKER["places"],
            keep_series=True,
        )
        series = unpack_series(record["series_packed"])
        if trajectory_hash(series) != record["trajectory_sha256"]:
            raise RuntimeError("compact trajectory round-trip does not match the fresh hash")
        target = expected[spec["condition_id"]]
        if record["trajectory_sha256"] != target["trajectory_sha256"]:
            hash_failures.append({"condition_id": spec["condition_id"], "choice_seed": seed, "actual": record["trajectory_sha256"], "expected": target["trajectory_sha256"]})
        late = late_visit_metrics(series)
        hhi_error = abs(late["late_hhi"] - target["late_hhi"])
        share_error = abs(late["late_dominant_share"] - target["late_dominant_share"])
        entropy_error = abs(late["late_entropy"] - target["late_entropy"])
        place_match = late["late_dominant"] == target["late_dominant"]
        if hhi_error > HHI_TOLERANCE or share_error > HHI_TOLERANCE or entropy_error > HHI_TOLERANCE or not place_match:
            hhi_failures.append(
                {
                    "condition_id": spec["condition_id"],
                    "choice_seed": seed,
                    "hhi_error": hhi_error,
                    "share_error": share_error,
                    "entropy_error": entropy_error,
                    "place_match": place_match,
                    "actual_place": late["late_dominant"],
                    "expected_place": target["late_dominant"],
                }
            )
        event_row = {"condition_id": spec["condition_id"], "choice_seed": seed, "cell_id": spec["cell_id"], "role": spec["role"], "target_id": spec["target_id"], **late, "event18": concentration_event(series, 18), "event20": concentration_event(series, 20), "llm_calls": 0}
        events.append(event_row)
        trajectories[spec["condition_id"]] = series
        packed.append({"condition_id": spec["condition_id"], "trajectory_sha256": record["trajectory_sha256"], "series_packed": record["series_packed"]})
    scoring = []
    native_paths = {}
    if hash_failures or hhi_failures:
        return {"choice_seed": seed, "hash_failures": hash_failures, "hhi_failures": hhi_failures, "q_failures": q_failures, "llm_calls": 0}
    walks: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for memory in MEMORIES:
        for setting in SETTINGS:
            weights_by_transform = {transform: float(task["weights"][condition_id(memory, transform, setting)]) for transform in TRANSFORMS}
            for history in HISTORIES:
                source_id = "S0" if history == "H0" else condition_id(memory, "LOGMAX" if history == "HL" else "ZRAW", setting)
                walk_key = (memory, source_id)
                if walk_key not in walks:
                    walks[walk_key] = walk_memory(_WORKER["atlas"], trajectories[source_id], _WORKER["matrix"], _WORKER["places"], memory)
                scored = score_from_walk(walks[walk_key], weights_by_transform)
                for transform in TRANSFORMS:
                    row = {
                        "choice_seed": seed,
                        "memory": memory,
                        "setting": setting,
                        "history": history,
                        "readout": transform,
                        "condition_id": condition_id(memory, transform, setting),
                        "weight_hex": float(weights_by_transform[transform]).hex(),
                        **scored["rows"][transform],
                        "llm_calls": 0,
                    }
                    scoring.append(row)
                    if history == "H0":
                        expected_q = layer_q[condition_id(memory, transform, setting)]
                        error = abs(row["q_t1_500"] - expected_q)
                        if error > Q_TOLERANCE:
                            q_failures.append({"condition_id": row["condition_id"], "choice_seed": seed, "actual": row["q_t1_500"], "expected": expected_q, "abs_error": error})
                    if history == native_history(transform):
                        q_path = scored["q"][transform].mean(axis=1)
                        range_path = scored["logit_range"][transform].mean(axis=1)
                        if abs(float(q_path[100:500].mean()) - row["q_t101_500"]) > 1e-12:
                            raise RuntimeError("native path does not match the late window summary")
                        native_paths[condition_id(memory, transform, setting)] = {
                            "q": [float(value) for value in q_path],
                            "logit_range": [float(value) for value in range_path],
                        }
    if len(walks) != 27:
        raise RuntimeError("a seed did not reuse S0 and did not keep the 27 history walks")
    return {
        "choice_seed": seed,
        "hash_failures": hash_failures,
        "hhi_failures": hhi_failures,
        "q_failures": q_failures,
        "trajectories": packed,
        "scoring": scoring,
        "events": events,
        "native_paths": native_paths,
        "llm_calls": 0,
    }


def _load_expected() -> tuple[dict[tuple[str, int], dict[str, Any]], dict[tuple[str, int], float], dict[str, Any]]:
    metrics: dict[tuple[str, int], dict[str, Any]] = {}
    with (WORLD_V018_DIR / "heldout" / "world_metrics.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            metrics[(row["condition_id"], int(row["choice_seed"]))] = row
    layer_q: dict[tuple[str, int], float] = {}
    with (WORLD_V018_DIR / "heldout" / "layer_a_seed_metrics.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            layer_q[(row["condition_id"], int(row["choice_seed"]))] = float(row["q"])
    grouped: dict[int, dict[str, str | None]] = {seed: {} for seed in HELDOUT_SEEDS}
    specs = frozen_specs(load_frozen_weights())
    for spec_index, spec in enumerate(specs):
        spec_id = spec["condition_id"]
        for seed in HELDOUT_SEEDS:
            unit_path = WORLD_V018_DIR / "heldout" / "units" / f"{spec_id}__{seed}.json"
            payload = json.loads(unit_path.read_text(encoding="utf-8"))
            metrics[(spec_id, seed)]["late_dominant"] = payload["late_dominant"]
            if payload["trajectory_sha256"] != metrics[(spec_id, seed)]["trajectory_sha256"]:
                raise RuntimeError(f"Experiment 018 unit hash disagrees with world metrics for {spec_id} seed {seed}")
            if spec_id in {condition_id(memory, "LOGMAX", "q25") for memory in MEMORIES}:
                grouped[seed][spec["cell_id"].split("_")[0]] = payload["late_dominant"]
        if (spec_index + 1) % 5 == 0:
            print(f"sources units {spec_index + 1}/{len(specs)}", flush=True)
    counts = {"same": 0, "tie": 0, "different": 0}
    for row in grouped.values():
        counts[place_relation([row[memory] for memory in MEMORIES])] += 1
    counts["n"] = len(HELDOUT_SEEDS)
    counts["source"] = "experiment_018_unit_late_dominant"
    counts["llm_calls"] = 0
    return metrics, layer_q, counts


def _expected_grid(metrics: dict[tuple[str, int], dict[str, Any]], specs: list[dict[str, Any]]) -> None:
    expected = {(spec["condition_id"], seed) for spec in specs for seed in HELDOUT_SEEDS}
    if set(metrics) != expected:
        raise RuntimeError("Experiment 018 held-out metrics do not contain the frozen 25 by 1000 grid")


def _seed_path(destination: Path, seed: int) -> Path:
    return destination / "seeds" / f"{seed}.json"


def _write_seed(destination: Path, payload: dict[str, Any]) -> None:
    path = _seed_path(destination, int(payload["choice_seed"]))
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    write_json(path, payload)
    saved = json.loads(path.read_text(encoding="utf-8"))
    for item in saved["trajectories"]:
        if trajectory_hash(unpack_series(item["series_packed"])) != item["trajectory_sha256"]:
            raise RuntimeError("saved compact trajectory failed round-trip")
    append_jsonl(destination / "seed_index.jsonl", {"choice_seed": payload["choice_seed"], "file_sha256": file_sha256(path), "llm_calls": 0})


def _task_for(seed: int, specs: list[dict[str, Any]], weights: dict[str, float], metrics: dict[tuple[str, int], dict[str, Any]], layer_q: dict[tuple[str, int], float]) -> dict[str, Any]:
    return {
        "choice_seed": seed,
        "specs": specs,
        "weights": weights,
        "expected": {
            spec["condition_id"]: {
                "trajectory_sha256": metrics[(spec["condition_id"], seed)]["trajectory_sha256"],
                "late_hhi": metrics[(spec["condition_id"], seed)]["late_hhi"],
                "late_dominant_share": metrics[(spec["condition_id"], seed)]["late_dominant_share"],
                "late_entropy": metrics[(spec["condition_id"], seed)]["late_entropy"],
                "late_dominant": metrics[(spec["condition_id"], seed)]["late_dominant"],
            }
            for spec in specs
        },
        "layer_q": {spec["condition_id"]: layer_q[(spec["condition_id"], seed)] for spec in specs if spec["condition_id"] != "S0"},
    }


def _run_seeds(seeds: list[int], specs: list[dict[str, Any]], weights: dict[str, float], metrics: dict[tuple[str, int], dict[str, Any]], layer_q: dict[tuple[str, int], float], workers: int, destination: Path, started: float, deadline: float) -> list[int]:
    pending = [seed for seed in seeds if not _seed_path(destination, seed).exists()]
    completed: list[int] = []
    pool = _shared_pool(workers)
    for offset in range(0, len(pending), workers):
        if time.time() > deadline:
            raise BudgetStop("budget reserve reached at a seed boundary")
        batch = pending[offset : offset + workers]
        futures = {pool.submit(_seed_task, _task_for(seed, specs, weights, metrics, layer_q)): seed for seed in batch}
        for future in as_completed(futures):
            payload = future.result(timeout=900)
            if payload["hash_failures"] or payload["hhi_failures"] or payload["q_failures"]:
                write_json(
                    destination / "compatibility_failure.json",
                    {
                        "choice_seed": payload["choice_seed"],
                        "hash_failures": payload["hash_failures"],
                        "hhi_failures": payload["hhi_failures"],
                        "q_failures": payload["q_failures"],
                        "llm_calls": 0,
                    },
                )
                raise RuntimeError(f"compatibility gate failed at seed {payload['choice_seed']}")
            _write_seed(destination, payload)
            completed.append(int(payload["choice_seed"]))
        _mark(destination, "seeds", started, done=offset + len(batch), total=len(pending))
    return completed


def _iter_seeds(destination: Path) -> list[dict[str, Any]]:
    rows = []
    for seed in HELDOUT_SEEDS:
        path = _seed_path(destination, seed)
        if not path.exists():
            raise RuntimeError(f"missing seed file {seed}")
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def _analyze(destination: Path) -> dict[str, Any]:
    import statistics

    contrast_rows: dict[int, dict[str, float]] = {}
    period_totals: dict[str, list[float]] = {}
    alignment: dict[str, dict[int, list[float]]] = {}
    alignment_logit: dict[str, dict[int, list[float]]] = {}
    events: dict[str, list[dict[str, Any]]] = {}
    relations: dict[str, list[str]] = {}
    destinations: dict[str, dict[str, int]] = {}
    gap_values: dict[str, list[float]] = {}
    for seed in HELDOUT_SEEDS:
        payload = json.loads(_seed_path(destination, seed).read_text(encoding="utf-8"))
        index = {(row["memory"], row["setting"], row["history"], row["readout"]): row for row in payload["scoring"]}
        contrast_rows[seed] = paired_contrasts(index)
        for row in payload["scoring"]:
            if row["setting"] not in MATCHED_SETTINGS or row["history"] not in {native_history(row["readout"]), "H0"}:
                continue
            key = f"{row['memory']}|{row['readout']}|{row['setting']}|{row['history']}"
            bucket = period_totals.setdefault(key, [0.0 for _name in PERIOD_PLOT])
            for period_index, period_name in enumerate(PERIOD_PLOT):
                bucket[period_index] += float(row[f"q_{period_name}"])
        for condition, path in payload["native_paths"].items():
            q_bucket = alignment.setdefault(condition, {offset: [] for offset in ALIGNMENT_OFFSETS})
            range_bucket = alignment_logit.setdefault(condition, {offset: [] for offset in ALIGNMENT_OFFSETS})
            event = next(item for item in payload["events"] if item["condition_id"] == condition)["event18"]
            if not event["occurred"]:
                continue
            anchor = int(event["start"]) - 1
            for offset in ALIGNMENT_OFFSETS:
                index_t = anchor + offset
                if 0 <= index_t < TIMESTEPS:
                    q_bucket[offset].append(float(path["q"][index_t]))
                    range_bucket[offset].append(float(path["logit_range"][index_t]))
        by_condition = {item["condition_id"]: item for item in payload["events"]}
        for item in payload["events"]:
            contrast_rows[seed][f"event18_{item['condition_id']}"] = 1.0 if item["event18"]["occurred"] else 0.0
            events.setdefault(item["condition_id"], []).append(item)
            label = item["late_dominant"] if item["late_dominant"] is not None else "tie"
            destinations.setdefault(item["condition_id"], {})
            destinations[item["condition_id"]][label] = destinations[item["condition_id"]].get(label, 0) + 1
        for transform in TRANSFORMS:
            for setting in SETTINGS:
                labels = [by_condition[condition_id(memory, transform, setting)]["late_dominant"] for memory in MEMORIES]
                relations.setdefault(f"{transform}_{setting}", []).append(place_relation(labels))
                shares = [by_condition[condition_id(memory, transform, setting)]["late_share"] for memory in MEMORIES]
                contrast_rows[seed][f"share_tv_{transform}_{setting}_RW50_CUM"] = share_tv(shares[1], shares[0])
                contrast_rows[seed][f"share_tv_{transform}_{setting}_EXP99_CUM"] = share_tv(shares[2], shares[0])
                for left_name, right_name in (("RW50", "CUM"), ("EXP99", "CUM")):
                    left = by_condition[condition_id(left_name, transform, setting)]["event18"]
                    right = by_condition[condition_id(right_name, transform, setting)]["event18"]
                    gap_key = f"{transform}_{setting}_{left_name}_minus_{right_name}"
                    if left["occurred"] and right["occurred"]:
                        gap_values.setdefault(gap_key, []).append(float(left["start"]) - float(right["start"]))
                    else:
                        gap_values.setdefault(gap_key, []).append(float("nan"))
    names = sorted(contrast_rows[HELDOUT_SEEDS[0]])
    matrix = np.column_stack([[contrast_rows[seed][name] for seed in HELDOUT_SEEDS] for name in names])
    indexes = shared_indexes(len(HELDOUT_SEEDS), BOOTSTRAP_DRAWS, ANALYSIS_SEED)
    summaries = {}
    for column, name in enumerate(names):
        replicates = bootstrap_column(matrix[:, column], indexes)
        summaries[name] = distribution_summary(matrix[:, column].tolist(), replicates)
    event_summary = {}
    for condition, rows in events.items():
        occurred = [1.0 if row["event18"]["occurred"] else 0.0 for row in rows]
        starts = [float(row["event18"]["start"]) for row in rows if row["event18"]["occurred"]]
        exits = [1.0 if row["event18"]["exited"] else 0.0 for row in rows if row["event18"]["occurred"]]
        confirmations = [float(row["event18"]["confirm"]) for row in rows if row["event18"]["occurred"]]
        event_summary[condition] = {
            "n": len(rows),
            "event18_rate": statistics.fmean(occurred),
            "event18_absent": int(len(rows) - sum(occurred)),
            "event18_start": distribution_summary(starts) if starts else None,
            "event18_confirm": distribution_summary(confirmations) if confirmations else None,
            "event18_exit_rate": statistics.fmean(exits) if exits else None,
            "event20_rate": statistics.fmean(1.0 if row["event20"]["occurred"] else 0.0 for row in rows),
            "event20_absent": sum(1 for row in rows if not row["event20"]["occurred"]),
        }
    gap_summary = {}
    for key, values in gap_values.items():
        observed = [value for value in values if math.isfinite(value)]
        replicates, dropped = bootstrap_conditional(np.asarray(values, dtype=np.float64), indexes)
        item = distribution_summary(observed, replicates if len(replicates) else None) if observed else {"n": 0, "mean": None}
        item["absent_pairs"] = len(values) - len(observed)
        item["bootstrap_replicates_dropped"] = dropped
        gap_summary[key] = item
    alignment_summary = {
        condition: {str(offset): {"mean": statistics.fmean(values) if values else None, "n": len(values)} for offset, values in offsets.items()}
        for condition, offsets in alignment.items()
    }
    alignment_logit_summary = {
        condition: {str(offset): {"mean": statistics.fmean(values) if values else None, "n": len(values)} for offset, values in offsets.items()}
        for condition, offsets in alignment_logit.items()
    }
    period_means = {key: [value / len(HELDOUT_SEEDS) for value in totals] for key, totals in period_totals.items()}
    relation_summary = {key: {label: values.count(label) for label in ("same", "tie", "different")} for key, values in relations.items()}
    payload = {
        "analysis_rng_seed": ANALYSIS_SEED,
        "draws": BOOTSTRAP_DRAWS,
        "percentile_indexes": list(percentile_indexes(BOOTSTRAP_DRAWS)),
        "index_meaning": "position in seeds 700-1699; index 0 is choice seed 700",
        "seed_list_head": HELDOUT_SEEDS[0],
        "includes_refitting_uncertainty": False,
        "causal_mediation": False,
        "summaries": summaries,
        "events": event_summary,
        "event_time_gaps": gap_summary,
        "alignment": alignment_summary,
        "alignment_logit": alignment_logit_summary,
        "period_means": period_means,
        "period_names": list(PERIOD_PLOT),
        "place_relations": relation_summary,
        "destinations": destinations,
        "llm_calls": 0,
    }
    write_json(destination / "uncertainty_analysis.json", payload)
    return payload


def bootstrap_conditional(values: np.ndarray, indexes: np.ndarray) -> tuple[np.ndarray, int]:
    kept: list[float] = []
    dropped = 0
    for draw in range(indexes.shape[0]):
        sample = values[indexes[draw]]
        sample = sample[np.isfinite(sample)]
        if sample.size == 0:
            dropped += 1
            continue
        kept.append(float(sample.mean()))
    return np.asarray(kept, dtype=np.float64), dropped


def _replay_gap(saved: dict[str, Any], fresh: dict[str, Any]) -> float:
    gap = 0.0
    for key, value in saved.items():
        other = fresh[key]
        if isinstance(value, float):
            gap = max(gap, abs(value - float(other)))
        elif isinstance(value, dict):
            gap = max(gap, _replay_gap(value, other))
    return gap


def _replay_gate(destination: Path, specs: list[dict[str, Any]], weights: dict[str, float], metrics: dict[tuple[str, int], dict[str, Any]], layer_q: dict[tuple[str, int], float], workers: int) -> dict[str, Any]:
    pool = _shared_pool(workers)
    futures = [pool.submit(_seed_task, _task_for(seed, specs, weights, metrics, layer_q)) for seed in GATE_SEEDS]
    results = []
    for future in as_completed(futures):
        fresh = future.result(timeout=900)
        path = _seed_path(destination, int(fresh["choice_seed"]))
        before = file_sha256(path)
        saved = json.loads(path.read_text(encoding="utf-8"))
        after = file_sha256(path)
        saved_index = {(row["memory"], row["setting"], row["history"], row["readout"]): row for row in saved["scoring"]}
        fresh_index = {(row["memory"], row["setting"], row["history"], row["readout"]): row for row in fresh["scoring"]}
        max_gap = 0.0
        for key, row in saved_index.items():
            max_gap = max(max_gap, _replay_gap(row, fresh_index[key]))
        saved_events = {item["condition_id"]: item for item in saved["events"]}
        fresh_events = {item["condition_id"]: item for item in fresh["events"]}
        event_match = all(saved_events[key]["event18"] == fresh_events[key]["event18"] and saved_events[key]["event20"] == fresh_events[key]["event20"] for key in saved_events)
        hash_match = all(
            item["trajectory_sha256"] == next(other["trajectory_sha256"] for other in fresh["trajectories"] if other["condition_id"] == item["condition_id"])
            for item in saved["trajectories"]
        )
        results.append(
            {
                "choice_seed": fresh["choice_seed"],
                "hash_match": hash_match,
                "event_match": event_match,
                "max_diagnostic_gap": max_gap,
                "file_rewritten": before != after,
                "llm_calls": 0,
            }
        )
    passed = len(results) == 3 and all(
        item["hash_match"] and item["event_match"] and item["max_diagnostic_gap"] <= REPLAY_FLOAT_TOLERANCE and not item["file_rewritten"] for item in results
    )
    return {"passed": passed, "count": 75, "seeds": results, "tolerance": REPLAY_FLOAT_TOLERANCE, "llm_calls": 0}


def _mark(destination: Path, stage: str, started: float, **extra: Any) -> None:
    payload = {"stage": stage, "elapsed_seconds": time.perf_counter() - started, "llm_calls": 0}
    payload.update(extra)
    append_jsonl(destination / "progress.jsonl", payload)
    print(f"{stage} elapsed={payload['elapsed_seconds']:.1f}s {extra}", flush=True)


def _record_failure(destination: Path, stage: str, exc: BaseException, started: float) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if not (destination / "failure_record.json").exists():
        write_json(destination / "failure_record.json", {"stage": stage, "error": str(exc), "traceback": traceback.format_exc(), "elapsed_seconds": time.perf_counter() - started, "llm_calls": 0})
    write_json(destination / "completion_status.json", {"status": "INCOMPLETE", "reason": stage, "elapsed_seconds": time.perf_counter() - started, "llm_calls": 0})


def build_protocol(started_at: str, weights: dict[str, float]) -> dict[str, Any]:
    payload = {
        "record_type": "experiment_019_post_hoc_protocol",
        "world_id": "world_v0_19",
        "experiment_id": "experiment_019_closed_loop_decision_impact",
        "designed_after_experiment_018": True,
        "preregistered_before_018": False,
        "new_seeds": 0,
        "new_conditions": 0,
        "weights_refit": False,
        "started_at": started_at,
        "llm_calls": 0,
        "seeds": list(HELDOUT_SEEDS),
        "gate_seeds": list(GATE_SEEDS),
        "replay_seeds": list(GATE_SEEDS),
        "weights_hex": {key: float(value).hex() for key, value in sorted(weights.items())},
        "primary_window": "decision t=101-500",
        "windows": [name for name, _start, _stop in WINDOWS],
        "delta_q_native": "Q(native rule | own closed-loop history) - Q(native rule | S0 history), same seed, rule, weight, and window",
        "event18": "first committed run of 20 timesteps with at least 18 of 20 agents in one place",
        "event20": "first committed run of 20 timesteps with 20 of 20 agents in one place",
        "alignment_anchor": "decision that produces the first committed timestep of event18",
        "alignment_offsets": list(ALIGNMENT_OFFSETS),
        "q_tolerance": Q_TOLERANCE,
        "hhi_tolerance": HHI_TOLERANCE,
        "replay_float_tolerance": REPLAY_FLOAT_TOLERANCE,
        "probability_tolerance": PROBABILITY_TOLERANCE,
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": ANALYSIS_SEED, "percentile_indexes": list(percentile_indexes(BOOTSTRAP_DRAWS)), "index_0": 700},
        "counts": {"main_replays": 25000, "gate_subset_included": 75, "extra_replays": 75, "total_world_replays": 25075, "scoring_combinations": 72000},
        "source_sha256": {
            "world/experiment_018.py": file_sha256(PROJECT_ROOT / "world" / "experiment_018.py"),
            "world/experiment_019.py": file_sha256(Path(__file__).resolve()),
            "data/worlds/world_v0_18/calibrated_weights.json": file_sha256(WORLD_V018_DIR / "calibrated_weights.json"),
            "data/worlds/world_v0_18/world_v0_18_config.json": file_sha256(WORLD_V018_DIR / "world_v0_18_config.json"),
            "data/worlds/world_v0_18/heldout/world_metrics.jsonl": file_sha256(WORLD_V018_DIR / "heldout" / "world_metrics.jsonl"),
            "data/worlds/world_v0_18/heldout/layer_a_seed_metrics.jsonl": file_sha256(WORLD_V018_DIR / "heldout" / "layer_a_seed_metrics.jsonl"),
        },
        "utility_atlas_sha256": EXPECTED_ATLAS_SHA256,
        "budget_seconds": BUDGET_SECONDS,
        "reserve_seconds": RESERVE_SECONDS,
        "scope": [
            "Cross-scoring does not regenerate the world.",
            "Comparing HL with HZ changes locations, occupancy, base utility, and memory together.",
            "Contrasts are not feedback shares, mediation effects, or mechanism percentages.",
            "A concentration run of 20 steps is not a permanent absorbing state.",
            "Intervals are conditional on the frozen 018 weights and this existing seed cohort.",
        ],
    }
    payload["protocol_sha256"] = canonical_sha256(payload)
    return payload


def protocol_markdown(protocol: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Experiment 019 protocol",
            "",
            "這份診斷是在看過 Experiment 018 的結果之後制定的回溯稽核。它不是 018 之前的預註冊，也不是新的 held-out replication 或盲測。",
            "",
            f"Protocol SHA-256：`{protocol['protocol_sha256']}`",
            "",
            "Seeds 700–1699 與 018 的 18 個校準 weight、6 個 weight 150 reference、S0 全部沿用。Weight 從 018 的 weight_hex 讀取。",
            "主要診斷是同一規則在自身閉環歷史與 S0 歷史上、decision t=101–500 的 Q 差。",
            "集中事件是連續 20 個 committed timesteps 至少 18/20 agents 在同一場所。未發生者不記成第 500 步。",
            "對齊錨點是產生該連續區間第一個 committed state 的決策。相對時間是 -40 到 +49，缺歷史不填補。",
            "Q 對 018 Layer A 的容差是 1e-10。HHI、dominant share、entropy 的容差是 1e-12。Replay 浮點容差是 1e-12。機率對 _select 的容差是 1e-12。",
            "Bootstrap 種子是 20261001，10000 次，索引 0 對應 choice seed 700，percentile indexes 249 與 9749。",
            "交叉對照不是 feedback 占比或因果中介。HHI 到頂也不表示記憶沒有終局場所差異。",
            "",
        ]
    )


def _write_figures(destination: Path, analysis: dict[str, Any]) -> None:
    status: dict[str, Any] = {"written": False, "llm_calls": 0}
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import font_manager
        from matplotlib import pyplot as plt
    except ImportError as exc:
        status["reason"] = str(exc)
        write_json(destination / "figure_status.json", status)
        return
    try:
        names = {font.name for font in font_manager.fontManager.ttflist}
        font = "Microsoft JhengHei" if "Microsoft JhengHei" in names else "DejaVu Sans"
        plt.rcParams["font.sans-serif"] = [font]
        plt.rcParams["axes.unicode_minus"] = False
        figure_dir = destination / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        _figure_periods(destination, plt, figure_dir / "period_q.png")
        _figure_matrix(analysis, plt, figure_dir / "cross_matrix.png")
        _figure_alignment(analysis, plt, figure_dir / "alignment.png")
        _figure_events(analysis, plt, figure_dir / "events.png")
        _figure_places(analysis, plt, figure_dir / "destinations.png")
        status.update({"written": True, "font": font})
    except Exception as exc:
        status["reason"] = str(exc)
    write_json(destination / "figure_status.json", status)


def _figure_periods(destination: Path, plt: Any, path: Path) -> None:
    analysis_path = destination / "uncertainty_analysis.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    period_names = analysis["period_names"]
    figure, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    flat = [(memory, transform) for memory in MEMORIES for transform in TRANSFORMS]
    for axis, (memory, transform) in zip(axes.ravel(), flat):
        for setting in MATCHED_SETTINGS:
            own = analysis["period_means"][f"{memory}|{transform}|{setting}|{native_history(transform)}"]
            base = analysis["period_means"][f"{memory}|{transform}|{setting}|H0"]
            axis.plot(period_names, own, label=f"{setting} own")
            axis.plot(period_names, base, linestyle="--", label=f"{setting} S0")
        axis.set_title(f"{memory} {transform}")
        axis.tick_params(axis="x", labelrotation=45, labelsize=7)
    axes[0, 0].legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _figure_matrix(analysis: dict[str, Any], plt: Any, path: Path) -> None:
    labels = []
    grid = []
    for memory in MEMORIES:
        for setting in SETTINGS:
            for history in HISTORIES:
                labels.append(f"{memory} {setting} {history}")
                grid.append([analysis["summaries"][f"q_{memory}_{setting}_{history}_{transform}"]["mean"] for transform in TRANSFORMS])
    values = np.asarray(grid, dtype=np.float64)
    figure, axis = plt.subplots(figsize=(6, 12))
    image = axis.imshow(values, aspect="auto")
    axis.set_xticks([0, 1], list(TRANSFORMS))
    axis.set_yticks(range(len(labels)), labels, fontsize=6)
    axis.set_title("late Q by history and readout")
    figure.colorbar(image, ax=axis, fraction=0.05)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _figure_alignment(analysis: dict[str, Any], plt: Any, path: Path) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(10, 11))
    for condition in ("CUM_LOGMAX__q05", "CUM_LOGMAX__q15", "CUM_LOGMAX__q25", "CUM_ZRAW__q25"):
        series = analysis["alignment"].get(condition, {})
        logit = analysis["alignment_logit"].get(condition, {})
        xs = []
        q_values = []
        range_values = []
        for offset in ALIGNMENT_OFFSETS:
            point = series.get(str(offset))
            if point and point["n"]:
                xs.append(offset)
                q_values.append(point["mean"])
                range_values.append(logit[str(offset)]["mean"])
        if xs:
            axes[0].plot(xs, q_values, label=condition)
            axes[1].plot(xs, range_values, label=condition)
    for axis in axes[:2]:
        axis.axvline(0.0, color="black", linewidth=0.8)
        axis.axvline(19.0, color="gray", linewidth=0.8)
        axis.legend(fontsize=7)
    axes[0].set_ylabel("native Q")
    axes[1].set_ylabel("social logit range")
    labels = []
    starts = []
    deltas = []
    for memory in MEMORIES:
        for transform in TRANSFORMS:
            for setting in MATCHED_SETTINGS:
                condition = condition_id(memory, transform, setting)
                event = analysis["events"][condition]["event18_start"]
                if event is None:
                    continue
                labels.append(condition)
                starts.append(event["median"])
                deltas.append(analysis["summaries"][f"delta_q_{memory}_{transform}_{setting}"]["mean"])
    axes[2].scatter(starts, deltas)
    for label, start, delta in zip(labels, starts, deltas):
        axes[2].annotate(label, (start, delta), fontsize=5)
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_xlabel("median event start among seeds with an event")
    axes[2].set_ylabel("mean late native Q minus S0")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _figure_events(analysis: dict[str, Any], plt: Any, path: Path) -> None:
    labels = []
    rates = []
    exits = []
    for condition, item in analysis["events"].items():
        if item["event18_rate"] == 0.0 and not condition.endswith(("q15", "q25", "q05")):
            continue
        labels.append(condition.replace("__", "\n"))
        rates.append(item["event18_rate"])
        exits.append(0.0 if item["event18_exit_rate"] is None else item["event18_exit_rate"])
    positions = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(14, 5))
    axis.bar(positions - 0.15, rates, width=0.3, label="event rate")
    axis.bar(positions + 0.15, exits, width=0.3, label="exit rate among events")
    axis.set_xticks(positions, labels, rotation=90, fontsize=6)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _figure_places(analysis: dict[str, Any], plt: Any, path: Path) -> None:
    labels = list(analysis["place_relations"])
    same = [analysis["place_relations"][label]["same"] for label in labels]
    different = [analysis["place_relations"][label]["different"] for label in labels]
    tie = [analysis["place_relations"][label]["tie"] for label in labels]
    positions = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.bar(positions, same, label="same place")
    axis.bar(positions, different, bottom=same, label="different")
    axis.bar(positions, tie, bottom=np.array(same) + np.array(different), label="tie")
    axis.set_xticks(positions, [label.replace("_", "\n") for label in labels], rotation=90, fontsize=7)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _fmt(value: float) -> str:
    return f"{value:.6g}"


def write_report(destination: Path, analysis: dict[str, Any], source_places: dict[str, Any], elapsed: float) -> None:
    status = json.loads((destination / "completion_status.json").read_text(encoding="utf-8"))
    summaries = analysis["summaries"]
    delta_rows = []
    matched_abs = []
    for memory in MEMORIES:
        for transform in TRANSFORMS:
            for setting in SETTINGS:
                name = f"delta_q_{memory}_{transform}_{setting}"
                item = summaries[name]
                delta_rows.append([name, _fmt(item["mean"]), f"[{_fmt(item['ci_low'])}, {_fmt(item['ci_high'])}]"])
                if setting in MATCHED_SETTINGS:
                    matched_abs.append((abs(item["mean"]), name, item))
    matched_abs.sort(reverse=True)
    rule_rows = []
    for memory in MEMORIES:
        for setting in MATCHED_SETTINGS:
            for history in HISTORIES:
                name = f"rule_{memory}_{setting}_{history}"
                item = summaries[name]
                rule_rows.append([name, _fmt(item["mean"]), f"[{_fmt(item['ci_low'])}, {_fmt(item['ci_high'])}]"])
    event_rows = []
    for condition, item in analysis["events"].items():
        start = "未發生" if item["event18_start"] is None else _fmt(item["event18_start"]["median"])
        exit_rate = "—" if item["event18_exit_rate"] is None else _fmt(item["event18_exit_rate"])
        event_rows.append([condition, _fmt(item["event18_rate"]), str(item["event18_absent"]), start, exit_rate, _fmt(item["event20_rate"]), str(item["event20_absent"])])
    relation = analysis["place_relations"].get("LOGMAX_q25", {})
    largest = matched_abs[0][1] if matched_abs else ""
    text = "\n".join(
        [
            "# Experiment 019 報告",
            "",
            f"狀態：{status['status']}。這是看過 Experiment 018 之後做的閉環決策影響稽核，不是新的 held-out replication，也沒有重配 weights。",
            "",
            "交叉評分不重新生成世界。HL 與 HZ 的比較同時改變位置、occupancy、base utility 與歷史記憶，不能寫成純 feedback 占比或因果中介。",
            "連續 20 步的集中不是永久吸收態。區間條件於 018 已固定的 weights 與這 1000 個既有 seeds，不包含 weight fitting 的不確定性。",
            "",
            f"主重播 25000，其中整合子集 75 已包含在內。額外診斷 replay 75。合計世界重播 25075。新條件 0。新 seeds 0。交叉評分組合 72000。",
            f"正式執行器耗時 {elapsed:.1f} 秒。LLM calls = 0。",
            "",
            "## 自身閉環相對 S0 的 late Q 差",
            "",
            "ΔQ_native 是同一規則、同一 weight、decision t=101–500，在自身閉環減去 S0。方向沒有事先限定。",
            "",
            "| contrast | mean | CI |",
            "| --- | --- | --- |",
            *["| " + " | ".join(row) + " |" for row in delta_rows],
            "",
            f"18 個 matched 條件裡，絕對平均差最大的是 {largest}。",
            "",
            "## 固定歷史上的規則差",
            "",
            "這是同一來源歷史上 Q(ZRAW) − Q(LOGMAX)。它不是 feedback 占比。",
            "",
            "| contrast | mean | CI |",
            "| --- | --- | --- |",
            *["| " + " | ".join(row) + " |" for row in rule_rows],
            "",
            "## 集中事件",
            "",
            "分母是 1000 個 seeds。未發生者不記成第 500 步。起始時間只在有事件的 seeds 上摘要。",
            "",
            "| condition | event18 rate | absent | median start | exit rate | event20 rate | event20 absent |",
            "| --- | --- | --- | --- | --- | --- | --- |",
            *["| " + " | ".join(row) + " |" for row in event_rows],
            "",
            "## q25 LOGMAX 的終局場所",
            "",
            f"018 單位檔的三個 LOGMAX q25 memory，同一 seed 終局場所相同的有 {source_places['same']} / {source_places['n']}，不同 {source_places['different']}，含 tie {source_places['tie']}。",
            f"本輪由 hash 核對過的軌跡重建：same {relation.get('same')}，different {relation.get('different')}，tie {relation.get('tie')}。",
            "",
            "HHI 同為 1 只表示 late visit share 集中在單一場所。終局場所或集中時間若不同，記憶架構仍然可辨認。",
            "",
            "觀察到 Q 或 social logit range 的變化早於集中，仍然不是因果證明。",
            "",
        ]
    )
    (destination / "EXPERIMENT_019_REPORT.md").write_text(text, encoding="utf-8", newline="\n")
    REPORT_DOC.write_text(text, encoding="utf-8", newline="\n")


def _manifest(destination: Path) -> None:
    files = {}
    for path in sorted(destination.rglob("*")):
        if path.is_file() and path.name != "run_manifest.json":
            files[str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")] = file_sha256(path)
    write_json(destination / "run_manifest.json", {"world_id": "world_v0_19", "files": files, "llm_calls": 0})


def _update_indexes(status: str) -> None:
    pointer = f"- 019 狀態為 {status}。見 [`docs/EXPERIMENT_019_REPORT.md`](docs/EXPERIMENT_019_REPORT.md)。協議見 [`docs/EXPERIMENT_019_PROTOCOL.md`](docs/EXPERIMENT_019_PROTOCOL.md)。\n"
    readme = PROJECT_ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    if "019 狀態為" not in text:
        text = text.replace(
            "- 018 狀態為 COMPLETE。見 [`docs/EXPERIMENT_018_REPORT.md`](docs/EXPERIMENT_018_REPORT.md)。協議見 [`docs/EXPERIMENT_018_PROTOCOL.md`](docs/EXPERIMENT_018_PROTOCOL.md)。\n",
            "- 018 狀態為 COMPLETE。見 [`docs/EXPERIMENT_018_REPORT.md`](docs/EXPERIMENT_018_REPORT.md)。協議見 [`docs/EXPERIMENT_018_PROTOCOL.md`](docs/EXPERIMENT_018_PROTOCOL.md)。\n" + pointer,
        )
        readme.write_text(text, encoding="utf-8", newline="\n")
    experiments = PROJECT_ROOT / "docs" / "EXPERIMENTS.md"
    index = experiments.read_text(encoding="utf-8")
    if "Experiment 019" not in index:
        block = (
            "\n### Experiment 019 — closed-loop decision impact\n\n"
            f"019 狀態為 {status}。這是看過 018 之後的回溯稽核。協議見 [`EXPERIMENT_019_PROTOCOL.md`](EXPERIMENT_019_PROTOCOL.md)。結果見 [`EXPERIMENT_019_REPORT.md`](EXPERIMENT_019_REPORT.md)。\n\n"
        )
        index = index.replace("\n## 輸出契約\n", block + "## 輸出契約\n")
        experiments.write_text(index, encoding="utf-8", newline="\n")


def _verify_saved_seed(destination: Path, seed: int, metrics: dict[tuple[str, int], dict[str, Any]], specs: list[dict[str, Any]]) -> None:
    path = _seed_path(destination, seed)
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("hash_failures") or payload.get("hhi_failures") or payload.get("q_failures"):
        raise RuntimeError(f"saved seed {seed} contains a failed gate and cannot be resumed")
    for spec in specs:
        item = next(row for row in payload["trajectories"] if row["condition_id"] == spec["condition_id"])
        expected = metrics[(spec["condition_id"], seed)]["trajectory_sha256"]
        if item["trajectory_sha256"] != expected or trajectory_hash(unpack_series(item["series_packed"])) != expected:
            raise RuntimeError(f"saved seed {seed} does not match Experiment 018")


def run_experiment(output_dir: Path | None = None, *, resume: bool = False, workers: int | None = None) -> Path:
    destination = output_dir or OUTPUT_DIR
    if destination.resolve() in {WORLD_V018_DIR.resolve(), ATLAS_DIR.resolve()}:
        raise RuntimeError("experiment 019 must not write into a frozen source directory")
    started = time.perf_counter()
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    deadline = time.time() + BUDGET_SECONDS - RESERVE_SECONDS
    worker_count = workers if workers is not None else min(12, max(1, (os.cpu_count() or 2) - 1))
    stage = "startup"
    try:
        manifest = json.loads((ATLAS_DIR / "atlas_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("sha256") != EXPECTED_ATLAS_SHA256:
            raise RuntimeError("utility atlas hash is not the frozen experiment atlas")
        weights = load_frozen_weights()
        specs = frozen_specs(weights)
        for vector in ([0.0, 1.0, -2.0, 5.0, 3.5], [0.0, 0.0, 0.0, 0.0, 0.0], [10.0, 10.0, 10.0, 10.0, 0.0]):
            if probabilities_match_select(vector) > PROBABILITY_TOLERANCE:
                raise RuntimeError("vectorized probabilities do not match _select")
        if resume:
            protocol = json.loads((destination / "diagnostic_protocol.json").read_text(encoding="utf-8"))
            config = json.loads((destination / "world_v0_19_config.json").read_text(encoding="utf-8"))
            if protocol["source_sha256"]["world/experiment_019.py"] != file_sha256(Path(__file__).resolve()):
                raise RuntimeError("resume stopped because experiment_019.py no longer matches the frozen protocol")
            if config["protocol_sha256"] != protocol["protocol_sha256"] or config["source_sha256"] != protocol["source_sha256"]:
                raise RuntimeError("resume stopped because the frozen config does not match the protocol")
            if (destination / "failure_record.json").exists():
                raise RuntimeError("resume stopped because a previous attempt recorded a failure")
        else:
            refuse_nonempty_output(destination)
            destination.mkdir(parents=True, exist_ok=True)
        stage = "sources"
        metrics, layer_q, source_places = _load_expected()
        _expected_grid(metrics, specs)
        if resume:
            saved_places = json.loads((destination / "source_place_count.json").read_text(encoding="utf-8"))
            if {key: saved_places[key] for key in ("same", "tie", "different", "n")} != {key: source_places[key] for key in ("same", "tie", "different", "n")}:
                raise RuntimeError("resume stopped because the 018 place count changed")
            for seed in HELDOUT_SEEDS:
                _verify_saved_seed(destination, seed, metrics, specs)
        else:
            write_json(destination / "source_place_count.json", source_places)
            protocol = build_protocol(started_at, weights)
            write_json(destination / "diagnostic_protocol.json", protocol)
            PROTOCOL_DOC.write_text(protocol_markdown(protocol), encoding="utf-8", newline="\n")
            config = {
                "world_id": "world_v0_19",
                "status_at_config_freeze": "CONFIG_FROZEN_BEFORE_DIAGNOSTICS",
                "protocol_sha256": protocol["protocol_sha256"],
                "weights_hex": protocol["weights_hex"],
                "source_sha256": protocol["source_sha256"],
                "python": sys.version,
                "platform": platform.platform(),
                "llm_calls": 0,
                "started_at": started_at,
            }
            config["config_sha256"] = canonical_sha256(config)
            write_json(destination / "world_v0_19_config.json", config)
            write_json(
                destination / "source_manifest.json",
                {
                    "record_type": "experiment_019_source_manifest",
                    "source_sha256": protocol["source_sha256"],
                    "source_place_count": source_places,
                    "calibration_cache_scanned": False,
                    "llm_calls": 0,
                },
            )
            sample = resolve_source_path("data\\worlds\\world_v0_18\\calibrated_weights.json")
            if file_sha256(sample) != protocol["source_sha256"]["data/worlds/world_v0_18/calibrated_weights.json"]:
                raise RuntimeError("normalized 018 path did not resolve to the hashed weight file")
        _mark(destination, "protocol_ready", started, workers=worker_count)
        stage = "gate"
        gate_pending = [seed for seed in GATE_SEEDS if not _seed_path(destination, seed).exists()]
        if gate_pending:
            _run_seeds(gate_pending, specs, weights, metrics, layer_q, worker_count, destination, started, deadline)
        if any(not _seed_path(destination, seed).exists() for seed in GATE_SEEDS):
            raise RuntimeError("integration seeds did not all pass")
        write_json(destination / "integration_gate.json", {"passed": True, "seeds": list(GATE_SEEDS), "included_in_main_replay": True, "additional_count": 0, "llm_calls": 0})
        stage = "replay"
        _run_seeds([seed for seed in HELDOUT_SEEDS if seed not in GATE_SEEDS], specs, weights, metrics, layer_q, worker_count, destination, started, deadline)
        if any(not _seed_path(destination, seed).exists() for seed in HELDOUT_SEEDS):
            raise RuntimeError("the 1000-seed replay is incomplete")
        stage = "analysis"
        _shutdown_pool()
        analysis = _analyze(destination)
        stage = "diagnostic_replay"
        replay = _replay_gate(destination, specs, weights, metrics, layer_q, worker_count)
        write_json(destination / "replay_audit.json", replay)
        _shutdown_pool()
        if not replay["passed"]:
            raise RuntimeError("diagnostic replay did not match the saved summaries")
        decision = completion_decision(hashes_passed=True, compatibility_passed=True, replay_passed=True, seeds_complete=True)
        write_json(
            destination / "completion_status.json",
            {"status": decision, "reason": "verified", "elapsed_seconds": time.perf_counter() - started, "main_replays": 25000, "extra_replays": 75, "llm_calls": 0},
        )
        _write_figures(destination, analysis)
        write_report(destination, analysis, source_places, time.perf_counter() - started)
        _manifest(destination)
        _update_indexes(decision)
        _mark(destination, "finished", started, status=decision)
        return destination
    except BudgetStop as exc:
        _shutdown_pool()
        done = sum(1 for seed in HELDOUT_SEEDS if _seed_path(destination, seed).exists())
        write_json(destination / "incomplete_record.json", {"stage": stage, "error": str(exc), "completed_seeds": done, "llm_calls": 0})
        write_json(destination / "completion_status.json", {"status": "INCOMPLETE", "reason": "budget", "completed_seeds": done, "elapsed_seconds": time.perf_counter() - started, "llm_calls": 0})
        raise
    except Exception as exc:
        _shutdown_pool()
        _record_failure(destination, stage, exc, started)
        raise


def main() -> None:
    destination = OUTPUT_DIR
    resume = False
    if destination.exists() and any(destination.iterdir()):
        if (destination / "failure_record.json").exists():
            raise RuntimeError("experiment 019 has a failure record and will not resume a failed run")
        status_path = destination / "completion_status.json"
        if status_path.exists() and json.loads(status_path.read_text(encoding="utf-8")).get("status") == "COMPLETE":
            raise RuntimeError("experiment 019 is already complete")
        if not (destination / "diagnostic_protocol.json").exists():
            raise RuntimeError("experiment 019 output exists without a diagnostic protocol")
        resume = True
    run_experiment(resume=resume)


if __name__ == "__main__":
    main()
