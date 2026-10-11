"""Experiment 020: early formation and later maintenance by social-field source.

The schedule was specified after Experiments 018 and 019. It replays no fitted
weights and does not call a model. Replacing the social-field source is the
intervention. It is not an equal-Q decomposition and not a mechanism share.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
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
from world.choice import agent_reconsiders, choice_seed
from world.dynamics_calibration import load_initial_locations, measure_world
from world.experiment_002 import REFERENCE_CHOICE_TEMPERATURE, REFERENCE_RECONSIDERATION_RATE
from world.experiment_004 import _entropy, _hhi, _select, _share, _unique_leader, _utilities, trajectory_hash
from world.experiment_005 import _population_sd, _presence
from world.experiment_017 import (
    CELL_BY_ID,
    SOCIAL_CELL_IDS,
    _commit,
    file_sha256,
    make_memory,
    pack_series,
    score_memory,
    unpack_series,
)
from world.experiment_018 import OUTPUT_DIR as WORLD_V018_DIR
from world.experiment_018 import Q_TARGETS
from world.experiment_019 import concentration_event
from world.geography import PLACE_IDS, distance_matrix, occupancy_of, place_index
from world.utility_atlas import OUTPUT_DIR as ATLAS_DIR
from world.utility_atlas import load_frozen_atlas

OUTPUT_DIR = PROJECT_ROOT / "data" / "worlds" / "world_v0_20"
PROTOCOL_DOC = PROJECT_ROOT / "docs" / "EXPERIMENT_020_PROTOCOL.md"
REPORT_DOC = PROJECT_ROOT / "docs" / "EXPERIMENT_020_REPORT.md"
EXPECTED_ATLAS_SHA256 = "664b511d208f7989e8a671d0e2d6e6433366120bb3d5d6a67db38277b4d5a373"
TIMESTEPS = 500
LATE_START = 101
SWITCH_INDEX = 50
EARLY_CONFIRM_MAX = 50
ELIGIBILITY_START = 31
ELIGIBILITY_STOP = 50
SWITCH_T_START = 41
SWITCH_T_STOP = 70
FORMAL_SEEDS = tuple(range(1700, 2700))
LINEAGE_SEEDS = (700, 1199, 1699)
GATE_SEEDS = (1700, 2199, 2699)
PATHS = ("EE", "EO", "OE", "OO")
Q_LABELS = ("q05", "q15", "q25")
BOOTSTRAP_DRAWS = 10_000
ANALYSIS_SEED = 20261002
BUDGET_SECONDS = 5 * 60 * 60
RESERVE_SECONDS = 40 * 60
TRACE_SCAN_LIMIT = 2_000_000
JSON_SCAN_LIMIT = 20_000_000
TEXT_SCAN_LIMIT = 262_144
REPLAY_FLOAT_TOLERANCE = 1e-12
FIELD_GAP_TOLERANCE = 0.0
CHOICE_SEED_RE = re.compile(r'"choice_seed"\s*:\s*(\d+)')
CHOICE_SEEDS_RE = re.compile(r'"choice_seeds"\s*:\s*\[([^\]]*)\]')
_WORKER: dict[str, Any] = {}
_POOL: ProcessPoolExecutor | None = None


class BudgetStop(RuntimeError):
    """The reserve was reached on a completed seed boundary."""


def percentile_indexes(draws: int) -> tuple[int, int]:
    return int(0.025 * (draws - 1)), int(0.975 * (draws - 1))


def refuse_nonempty_output(destination: Path) -> None:
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError("experiment 020 refuses to write into a non-empty output directory")


def completion_decision(*, lineage_passed: bool, gates_passed: bool, replay_passed: bool, seeds_complete: bool) -> str:
    if lineage_passed and gates_passed and replay_passed and seeds_complete:
        return "COMPLETE"
    return "INCOMPLETE"


def condition_id(cell_id: str, label: str) -> str:
    return f"{cell_id}__{label}"


def source_at(schedule: str, timestep: int) -> str:
    if schedule not in PATHS:
        raise RuntimeError("unknown source schedule")
    return schedule[0] if timestep < SWITCH_INDEX else schedule[1]


def load_frozen_weights() -> dict[str, float]:
    payload = json.loads((WORLD_V018_DIR / "calibrated_weights.json").read_text(encoding="utf-8"))
    config = json.loads((WORLD_V018_DIR / "world_v0_18_config.json").read_text(encoding="utf-8"))
    from_weights = {row["condition_id"]: float.fromhex(row["weight_hex"]) for row in payload["conditions"]}
    from_config = {row["condition_id"]: float.fromhex(row["weight_hex"]) for row in config["weights"]}
    if from_weights != from_config:
        raise RuntimeError("Experiment 018 weight files do not agree at full precision")
    matched = {key: value for key, value in from_weights.items() if key.endswith(Q_LABELS)}
    if len(matched) != 18:
        raise RuntimeError("Experiment 018 did not supply 18 matched weights")
    return matched


def matched_conditions(weights: dict[str, float]) -> list[dict[str, Any]]:
    rows = []
    for label in Q_LABELS:
        for cell_id in SOCIAL_CELL_IDS:
            identifier = condition_id(cell_id, label)
            rows.append(
                {
                    "condition_id": identifier,
                    "cell_id": cell_id,
                    "target_id": label,
                    "q_target": next(item["q_target"] for item in Q_TARGETS if item["target_id"] == label),
                    "social_weight": float(weights[identifier]),
                    "weight_hex": float(weights[identifier]).hex(),
                }
            )
    if len(rows) != 18:
        raise RuntimeError("the matched schedule is not 18 conditions")
    return rows


def donor_score_cube(series: list[dict[str, str]], agents: list[str], spec: dict[str, Any]) -> np.ndarray:
    memory = make_memory(spec, agents)
    cube = np.empty((len(series) - 1, len(agents), 5), dtype=np.float64)
    for timestep in range(len(series) - 1):
        for agent_index, agent_id in enumerate(agents):
            _unclipped, scores, _loo = score_memory(memory, agent_id, spec["transform"])
            for place_index, place_id in enumerate(PLACE_IDS):
                cube[timestep, agent_index, place_index] = scores[place_id]
        memory = _commit(memory, series[timestep + 1])
    if not np.isfinite(cube).all():
        raise RuntimeError("nonfinite donor score")
    return cube


def field_sha256(cube: np.ndarray) -> str:
    payload = np.ascontiguousarray(cube, dtype="<f8")
    return hashlib.sha256(payload.tobytes()).hexdigest()


def eligibility_place(series: list[dict[str, str]], threshold: int = 18) -> str | None:
    if len(series) <= ELIGIBILITY_STOP:
        return None
    window = series[ELIGIBILITY_START : ELIGIBILITY_STOP + 1]
    for place_id in PLACE_IDS:
        if all(sum(step[agent_id] == place_id for agent_id in step) >= threshold for step in window):
            return place_id
    return None


def first_run(series: list[dict[str, str]], threshold: int, min_start: int, run_length: int = 20) -> dict[str, Any] | None:
    last = len(series) - 1
    latest = last - run_length + 1
    if min_start > latest or min_start < 1:
        return None
    for start in range(min_start, latest + 1):
        counts = occupancy_of(series[start])
        for place_id in PLACE_IDS:
            if counts[place_id] < threshold:
                continue
            if any(sum(step[agent_id] == place_id for agent_id in step) < threshold for step in series[start : start + run_length]):
                continue
            return {"start": start, "confirm": start + run_length - 1, "place": place_id}
    return None


def maintenance_record(series: list[dict[str, str]], place: str | None, threshold: int = 18) -> dict[str, Any]:
    empty = {
        "eligible": place is not None,
        "place": place,
        "maintained_to_horizon": None,
        "original_exit": None,
        "release_time": None,
        "released": None,
        "late_run_any": None,
        "late_run_other": None,
    }
    if place is None:
        return empty
    exit_time = None
    release_time = None
    for timestep in range(SWITCH_INDEX + 1, len(series)):
        counts = occupancy_of(series[timestep])
        if exit_time is None and counts[place] < threshold:
            exit_time = timestep
        if release_time is None and all(counts[item] < threshold for item in PLACE_IDS):
            release_time = timestep
    cursor = first_run(series, threshold, SWITCH_INDEX + 1, 20)
    any_late = cursor is not None
    other = False
    while cursor is not None and cursor["place"] == place:
        cursor = first_run(series, threshold, int(cursor["confirm"]) + 1, 20)
    if cursor is not None and cursor["place"] != place:
        other = True
    return {
        "eligible": True,
        "place": place,
        "maintained_to_horizon": exit_time is None,
        "original_exit": exit_time,
        "release_time": release_time,
        "released": release_time is not None,
        "late_run_any": any_late,
        "late_run_other": other,
    }


def early_formation(series: list[dict[str, str]]) -> bool:
    event = concentration_event(series, 18)
    return bool(event["occurred"] and int(event["confirm"]) <= EARLY_CONFIRM_MAX)


def simulate_recipient(
    atlas: Any,
    locations: dict[str, str],
    matrix: dict[str, dict[str, float]],
    places: dict[str, dict[str, Any]],
    cell_id: str,
    choice_seed_value: int,
    social_weight: float,
    schedule: str,
    donor_cube: np.ndarray,
    *,
    timesteps: int = TIMESTEPS,
    keep_switch_detail: bool = False,
) -> dict[str, Any]:
    spec = CELL_BY_ID[cell_id]
    weight = float(social_weight)
    agents = sorted(locations)
    if donor_cube.shape != (timesteps, len(agents), 5):
        raise RuntimeError("donor score cube does not match the recipient horizon")
    current = {agent_id: locations[agent_id] for agent_id in agents}
    memory = make_memory(spec, agents)
    series = [dict(current)]
    moves = 0
    q_steps = np.empty(timesteps, dtype=np.float64)
    range_steps = np.empty(timesteps, dtype=np.float64)
    score_steps = np.empty(timesteps, dtype=np.float64)
    gap_steps = np.empty(timesteps, dtype=np.float64)
    clip_changes = 0
    nonfinite = 0
    switch_detail = None
    for timestep in range(timesteps):
        occupancy = occupancy_of(current)
        source = source_at(schedule, timestep)
        decisions: dict[str, str] = {}
        used_rows = np.empty((len(agents), 5), dtype=np.float64)
        base_rows = np.empty((len(agents), 5), dtype=np.float64)
        social_rows = np.empty((len(agents), 5), dtype=np.float64)
        gap_rows = np.empty(len(agents), dtype=np.float64)
        score_rows = np.empty(len(agents), dtype=np.float64)
        for agent_index, agent_id in enumerate(agents):
            origin = current[agent_id]
            unclipped, own_scores, _loo = score_memory(memory, agent_id, spec["transform"])
            if source == "E":
                used = own_scores
            else:
                used = {place_id: float(donor_cube[timestep, agent_index, place_index]) for place_index, place_id in enumerate(PLACE_IDS)}
            bases, totals, social_term = _utilities(atlas, matrix, places, agent_id, origin, occupancy, used, weight)
            if weight == 0.0:
                for place_id in PLACE_IDS:
                    if social_term[place_id] != 0.0 or totals[place_id] != bases[place_id]:
                        raise RuntimeError("weight 0 did not remove social utility")
            if source == "E":
                for place_id in PLACE_IDS:
                    if used[place_id] != own_scores[place_id]:
                        raise RuntimeError("endogenous source did not use the recipient shadow score")
            does_reconsider = agent_reconsiders(choice_seed_value, timestep, agent_id, REFERENCE_RECONSIDERATION_RATE)
            if does_reconsider:
                draw = Random(choice_seed(choice_seed_value, timestep, agent_id)).random()
                destination, _probabilities = _select(list(PLACE_IDS), [totals[place_id] for place_id in PLACE_IDS], draw)
            else:
                destination = origin
            decisions[agent_id] = destination
            for place_index, place_id in enumerate(PLACE_IDS):
                used_rows[agent_index, place_index] = totals[place_id]
                base_rows[agent_index, place_index] = bases[place_id]
                social_rows[agent_index, place_index] = social_term[place_id]
                if not np.isfinite(totals[place_id]) or not np.isfinite(social_term[place_id]):
                    nonfinite += 1
                if own_scores[place_id] != unclipped[place_id]:
                    clip_changes += 1
            own_values = np.array([own_scores[place_id] for place_id in PLACE_IDS], dtype=np.float64)
            used_values = np.array([used[place_id] for place_id in PLACE_IDS], dtype=np.float64)
            gap_rows[agent_index] = float(np.mean(np.abs(used_values - own_values)))
            score_rows[agent_index] = float(np.max(used_values) - np.min(used_values))
            if keep_switch_detail and timestep == SWITCH_INDEX and agent_index == 0:
                switch_detail = {"source": source, "used": used_values.tolist(), "own": own_values.tolist()}
        peak_used = np.max(used_rows, axis=1, keepdims=True)
        peak_base = np.max(base_rows, axis=1, keepdims=True)
        prob_used = np.exp((used_rows - peak_used) / REFERENCE_CHOICE_TEMPERATURE)
        prob_base = np.exp((base_rows - peak_base) / REFERENCE_CHOICE_TEMPERATURE)
        prob_used /= np.sum(prob_used, axis=1, keepdims=True)
        prob_base /= np.sum(prob_base, axis=1, keepdims=True)
        q_steps[timestep] = float(np.mean(0.5 * np.abs(prob_used - prob_base).sum(axis=1)))
        social_range = (np.max(social_rows, axis=1) - np.min(social_rows, axis=1)) / REFERENCE_CHOICE_TEMPERATURE
        range_steps[timestep] = float(np.mean(social_range))
        score_steps[timestep] = float(np.mean(score_rows))
        gap_steps[timestep] = float(np.mean(gap_rows))
        moves += sum(1 for agent_id in agents if decisions[agent_id] != current[agent_id])
        current = {agent_id: decisions[agent_id] for agent_id in agents}
        memory = _commit(memory, decisions)
        series.append(dict(current))
    if nonfinite:
        raise RuntimeError("nonfinite utility in a recipient")
    packed = pack_series(series, agents)
    if trajectory_hash(unpack_series(packed)) != trajectory_hash(series):
        raise RuntimeError("packed recipient trajectory failed round-trip")
    diagnostics = measure_world(series, moves=moves, timesteps=timesteps)
    late = _late_metrics(series) if timesteps >= LATE_START else None
    event18 = concentration_event(series, 18) if timesteps >= 20 else concentration_event(series, 18)
    event20 = concentration_event(series, 20)
    place = eligibility_place(series) if timesteps > ELIGIBILITY_STOP else None
    switch_q = [float(q_steps[index]) for index in range(SWITCH_T_START - 1, min(SWITCH_T_STOP, timesteps))]
    switch_occ = []
    for timestep in range(SWITCH_T_START, min(SWITCH_T_STOP, len(series) - 1) + 1):
        counts = occupancy_of(series[timestep])
        switch_occ.append(max(counts.values()))
    result = {
        "cell_id": cell_id,
        "schedule": schedule,
        "choice_seed": choice_seed_value,
        "social_weight": weight,
        "trajectory_sha256": trajectory_hash(series),
        "prefix_sha256": trajectory_hash(series[: SWITCH_INDEX + 1]),
        "series_packed": packed,
        "q_t1_50": _mean_slice(q_steps, 0, min(50, timesteps)),
        "q_t51_100": _mean_slice(q_steps, 50, min(100, timesteps)),
        "q_t101_500": _mean_slice(q_steps, 100, timesteps) if timesteps >= LATE_START else None,
        "q_t1_500": float(np.mean(q_steps)),
        "logit_range_t1_50": _mean_slice(range_steps, 0, min(50, timesteps)),
        "logit_range_t51_100": _mean_slice(range_steps, 50, min(100, timesteps)),
        "logit_range_t101_500": _mean_slice(range_steps, 100, timesteps) if timesteps >= LATE_START else None,
        "score_range_t1_500": float(np.mean(score_steps)),
        "field_gap_t1_50": _mean_slice(gap_steps, 0, min(50, timesteps)),
        "field_gap_t51_100": _mean_slice(gap_steps, 50, min(100, timesteps)),
        "field_gap_t101_500": _mean_slice(gap_steps, 100, timesteps) if timesteps >= LATE_START else None,
        "field_gap_t1_500": float(np.mean(gap_steps)),
        "q_switch": switch_q,
        "occ_switch": switch_occ,
        "movement_rate": diagnostics["movement_rate"],
        "saturation_frequency": diagnostics["saturation_frequency"],
        "empty_frequency": diagnostics["empty_frequency"],
        "clip_change_count": clip_changes,
        "nonfinite_count": 0,
        "event18": event18,
        "event20": event20,
        "early_formed": early_formation(series) if timesteps > EARLY_CONFIRM_MAX else False,
        "maintenance": maintenance_record(series, place) if timesteps > ELIGIBILITY_STOP else maintenance_record(series, None),
        "late": late,
        "switch_detail": switch_detail,
        "llm_calls": 0,
    }
    if schedule[0] == "E" and result["field_gap_t1_50"] != FIELD_GAP_TOLERANCE:
        raise RuntimeError("early endogenous field gap was not zero")
    if schedule == "EE" and timesteps >= 100 and result["field_gap_t51_100"] != FIELD_GAP_TOLERANCE:
        raise RuntimeError("late endogenous field gap was not zero")
    return result


def _mean_slice(values: np.ndarray, start: int, stop: int) -> float:
    piece = values[start:stop]
    if piece.size == 0:
        raise RuntimeError("metric window is empty")
    return float(np.mean(piece))


def _late_metrics(series: list[dict[str, str]]) -> dict[str, Any]:
    counts = _presence(series, LATE_START, len(series) - 1)
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


def _build_donor(choice_seed_value: int, timesteps: int = TIMESTEPS) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if "atlas" not in _WORKER:
        _init_worker()
    agents = sorted(_WORKER["locations"])
    donor = simulate_recipient(
        _WORKER["atlas"],
        _WORKER["locations"],
        _WORKER["matrix"],
        _WORKER["places"],
        "S0",
        choice_seed_value,
        0.0,
        "EE",
        np.zeros((timesteps, len(agents), 5), dtype=np.float64),
        timesteps=timesteps,
    )
    series = unpack_series(donor["series_packed"])
    fields = {cell_id: donor_score_cube(series, agents, CELL_BY_ID[cell_id]) for cell_id in SOCIAL_CELL_IDS}
    frozen = {cell_id: np.array(values, dtype=np.float64, copy=True) for cell_id, values in fields.items()}
    return donor, frozen


def _formal_seed(task: dict[str, Any]) -> dict[str, Any]:
    if "atlas" not in _WORKER:
        _init_worker()
    seed = int(task["choice_seed"])
    timesteps = int(task.get("timesteps", TIMESTEPS))
    donor, fields = _build_donor(seed, timesteps)
    before = {cell_id: field_sha256(values) for cell_id, values in fields.items()}
    worlds = []
    prefix_failures = []
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for spec in task["conditions"]:
        grouped[spec["condition_id"]] = {}
        for schedule in PATHS:
            world = simulate_recipient(
                _WORKER["atlas"],
                _WORKER["locations"],
                _WORKER["matrix"],
                _WORKER["places"],
                spec["cell_id"],
                seed,
                float(spec["social_weight"]),
                schedule,
                fields[spec["cell_id"]],
                timesteps=timesteps,
                keep_switch_detail=bool(task.get("keep_switch_detail", False)),
            )
            world["condition_id"] = spec["condition_id"]
            world["target_id"] = spec["target_id"]
            world["weight_hex"] = spec["weight_hex"]
            grouped[spec["condition_id"]][schedule] = world
            worlds.append(world)
        pair = grouped[spec["condition_id"]]
        if pair["EE"]["prefix_sha256"] != pair["EO"]["prefix_sha256"]:
            prefix_failures.append({"condition_id": spec["condition_id"], "pair": "EE_EO"})
        if pair["OE"]["prefix_sha256"] != pair["OO"]["prefix_sha256"]:
            prefix_failures.append({"condition_id": spec["condition_id"], "pair": "OE_OO"})
        if pair["EE"]["early_formed"] != pair["EO"]["early_formed"] or pair["OE"]["early_formed"] != pair["OO"]["early_formed"]:
            prefix_failures.append({"condition_id": spec["condition_id"], "pair": "early_formation"})
        if pair["EE"]["maintenance"]["place"] != pair["EO"]["maintenance"]["place"]:
            prefix_failures.append({"condition_id": spec["condition_id"], "pair": "eligibility_E"})
        if pair["OE"]["maintenance"]["place"] != pair["OO"]["maintenance"]["place"]:
            prefix_failures.append({"condition_id": spec["condition_id"], "pair": "eligibility_O"})
    after = {cell_id: field_sha256(values) for cell_id, values in fields.items()}
    donor_unchanged = before == after
    return {
        "choice_seed": seed,
        "donor": {key: value for key, value in donor.items() if key != "series_packed"},
        "donor_series_packed": donor["series_packed"],
        "donor_trajectory_sha256": donor["trajectory_sha256"],
        "donor_field_sha256": before,
        "donor_unchanged": donor_unchanged,
        "prefix_failures": prefix_failures,
        "worlds": [{key: value for key, value in world.items() if key != "series_packed"} for world in worlds],
        "packed": [{"condition_id": world["condition_id"], "schedule": world["schedule"], "trajectory_sha256": world["trajectory_sha256"], "series_packed": world["series_packed"]} for world in worlds],
        "llm_calls": 0,
    }


def _lineage_seed(task: dict[str, Any]) -> dict[str, Any]:
    if "atlas" not in _WORKER:
        _init_worker()
    seed = int(task["choice_seed"])
    agents = sorted(_WORKER["locations"])
    zeros = np.zeros((TIMESTEPS, len(agents), 5), dtype=np.float64)
    failures = []
    compared = []
    donor = simulate_recipient(
        _WORKER["atlas"],
        _WORKER["locations"],
        _WORKER["matrix"],
        _WORKER["places"],
        "S0",
        seed,
        0.0,
        "EE",
        zeros,
    )
    expected = task["expected"]
    if donor["trajectory_sha256"] != expected["S0"]:
        failures.append({"condition_id": "S0", "choice_seed": seed, "actual": donor["trajectory_sha256"], "expected": expected["S0"]})
    compared.append("S0")
    for spec in task["conditions"]:
        world = simulate_recipient(
            _WORKER["atlas"],
            _WORKER["locations"],
            _WORKER["matrix"],
            _WORKER["places"],
            spec["cell_id"],
            seed,
            float(spec["social_weight"]),
            "EE",
            zeros,
        )
        target = expected[spec["condition_id"]]
        if world["trajectory_sha256"] != target:
            failures.append({"condition_id": spec["condition_id"], "choice_seed": seed, "actual": world["trajectory_sha256"], "expected": target})
        compared.append(spec["condition_id"])
    return {"choice_seed": seed, "compared": len(compared), "failures": failures, "llm_calls": 0}


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _seeds_in_text(text: str) -> set[int]:
    found = {int(item) for item in CHOICE_SEED_RE.findall(text)}
    for blob in CHOICE_SEEDS_RE.findall(text):
        for item in blob.split(","):
            token = item.strip()
            if token.isdigit():
                found.add(int(token))
    return found


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


def audit_choice_seeds(data_root: Path | None = None) -> dict[str, Any]:
    root = data_root or (PROJECT_ROOT / "data")
    proposed = set(FORMAL_SEEDS)
    sources = []
    overlaps = []
    unchecked = []
    seen_dirs = set()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in {".json", ".jsonl"}:
            continue
        if "world_v0_20" in path.parts:
            continue
        if any(name in path.parts for name in ("world_v0_17", "world_v0_18", "world_v0_19")):
            seen_dirs.add(next(name for name in path.parts if name in {"world_v0_17", "world_v0_18", "world_v0_19"}))
        size = path.stat().st_size
        display = _display_path(path, root)
        if path.suffix == ".jsonl" and size > TRACE_SCAN_LIMIT:
            unchecked.append({"path": display, "bytes": size, "reason": "large trace; choice seeds are taken from configs, manifests, and smaller indexes"})
            continue
        if path.suffix == ".json" and size > JSON_SCAN_LIMIT:
            unchecked.append({"path": display, "bytes": size, "reason": "json above the index scan limit"})
            overlaps.append(display)
            continue
        choice_seeds: set[int] = set()
        if path.suffix == ".json" and size <= TEXT_SCAN_LIMIT:
            payload = json.loads(path.read_text(encoding="utf-8"))
            _collect_choice_seeds(payload, choice_seeds)
        else:
            choice_seeds = _seeds_in_text(path.read_text(encoding="utf-8"))
        if not choice_seeds:
            continue
        hit = sorted(choice_seeds & proposed)
        record = {
            "path": display,
            "choice_seed_min": min(choice_seeds),
            "choice_seed_max": max(choice_seeds),
        }
        if hit:
            record["overlap"] = hit
            overlaps.append(record["path"])
        sources.append(record)
    missing = {"world_v0_17", "world_v0_18", "world_v0_19"} - seen_dirs
    if missing:
        overlaps.append("missing:" + ",".join(sorted(missing)))
    return {
        "experiment_id": "experiment_020_social_field_source_switching",
        "proposed_choice_seeds": "1700-2699",
        "proposed_count": len(proposed),
        "passed": not overlaps,
        "overlap_sources": overlaps,
        "source_count": len(sources),
        "unchecked_large_traces": unchecked,
        "scanned_documents": False,
        "includes_world_v0_17_outputs": "world_v0_17" in seen_dirs,
        "includes_world_v0_18_outputs": "world_v0_18" in seen_dirs,
        "includes_world_v0_19_outputs": "world_v0_19" in seen_dirs,
        "includes_world_v0_20_outputs": False,
        "calibration_cache_scanned": False,
        "llm_calls": 0,
    }


def load_lineage_targets() -> dict[tuple[str, int], str]:
    wanted = set(LINEAGE_SEEDS)
    found: dict[tuple[str, int], str] = {}
    with (WORLD_V018_DIR / "heldout" / "world_metrics.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            seed = int(row["choice_seed"])
            if seed in wanted and (row["condition_id"] == "S0" or row["condition_id"].endswith(Q_LABELS)):
                found[(row["condition_id"], seed)] = row["trajectory_sha256"]
    expected = {(condition_id(cell_id, label), seed) for cell_id in SOCIAL_CELL_IDS for label in Q_LABELS for seed in LINEAGE_SEEDS}
    expected.update({("S0", seed) for seed in LINEAGE_SEEDS})
    if set(found) != expected:
        raise RuntimeError("lineage targets do not contain the 57 frozen trajectories")
    return found


def _metrics_path(destination: Path, seed: int) -> Path:
    return destination / "seeds" / f"{seed}.json"


def _trajectory_path(destination: Path, seed: int) -> Path:
    return destination / "trajectories" / f"{seed}.json"


def _write_formal(destination: Path, payload: dict[str, Any]) -> None:
    seed = int(payload["choice_seed"])
    trajectory_path = _trajectory_path(destination, seed)
    metrics_path = _metrics_path(destination, seed)
    if metrics_path.exists() or trajectory_path.exists():
        raise RuntimeError(f"refusing to overwrite seed {seed}")
    packed = {"choice_seed": seed, "donor_series_packed": payload["donor_series_packed"], "donor_trajectory_sha256": payload["donor_trajectory_sha256"], "worlds": payload["packed"], "llm_calls": 0}
    write_json(trajectory_path, packed)
    saved = json.loads(trajectory_path.read_text(encoding="utf-8"))
    if trajectory_hash(unpack_series(saved["donor_series_packed"])) != saved["donor_trajectory_sha256"]:
        raise RuntimeError("saved donor trajectory failed round-trip")
    for item in saved["worlds"]:
        if trajectory_hash(unpack_series(item["series_packed"])) != item["trajectory_sha256"]:
            raise RuntimeError("saved recipient trajectory failed round-trip")
    metrics = {key: value for key, value in payload.items() if key not in {"packed", "donor_series_packed"}}
    metrics["trajectory_file_sha256"] = file_sha256(trajectory_path)
    write_json(metrics_path, metrics)
    append_jsonl(destination / "seed_index.jsonl", {"choice_seed": seed, "file_sha256": file_sha256(metrics_path), "llm_calls": 0})


def _run_pool(tasks: list[dict[str, Any]], workers: int, timeout: int) -> list[dict[str, Any]]:
    pool = _shared_pool(workers)
    futures = {pool.submit(_formal_seed if task["mode"] == "formal" else _lineage_seed, task): task for task in tasks}
    output = []
    for future in as_completed(futures):
        output.append(future.result(timeout=timeout))
    return output


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


def build_protocol(audit_sha256: str, started_at: str, weights: dict[str, float]) -> dict[str, Any]:
    payload = {
        "record_type": "experiment_020_protocol",
        "world_id": "world_v0_20",
        "experiment_id": "experiment_020_social_field_source_switching",
        "designed_after_018_and_019": True,
        "preregistered_before_those_results": False,
        "started_at": started_at,
        "seed_audit_sha256": audit_sha256,
        "seeds": list(FORMAL_SEEDS),
        "lineage_seeds": list(LINEAGE_SEEDS),
        "gate_seeds": list(GATE_SEEDS),
        "replay_seeds": list(GATE_SEEDS),
        "weights_hex": {key: float(value).hex() for key, value in sorted(weights.items())},
        "paths": list(PATHS),
        "switch": "decision t=51, before any agent chooses; commits 1-50 remain",
        "endogenous": "recipient shadow memory through t-1, same memory and transformation",
        "exogenous": "S0 donor commits 1 through t-1, leave-one-out of the same agent id, unweighted score, then the condition weight",
        "primary_outcome": "late visit-share HHI from cumulative presence at t=101-500",
        "contrasts": ["EE-OE", "EO-OO", "EE-EO", "OE-OO", "EE-EO-OE+OO"],
        "early_formation": "first 18/20 run confirms at or before committed t=50",
        "eligibility": "commits t=31-50 are one place with at least 18 agents; fixed before the switch",
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": ANALYSIS_SEED, "percentile_indexes": list(percentile_indexes(BOOTSTRAP_DRAWS)), "index_0": 1700},
        "replay_float_tolerance": REPLAY_FLOAT_TOLERANCE,
        "counts": {"lineage": 57, "new_worlds": 73000, "gate_subset_included": 219, "extra_replays": 219, "total_world_executions": 73276},
        "source_sha256": {
            "world/experiment_017.py": file_sha256(PROJECT_ROOT / "world" / "experiment_017.py"),
            "world/experiment_018.py": file_sha256(PROJECT_ROOT / "world" / "experiment_018.py"),
            "world/experiment_019.py": file_sha256(PROJECT_ROOT / "world" / "experiment_019.py"),
            "world/experiment_020.py": file_sha256(Path(__file__).resolve()),
            "data/worlds/world_v0_18/calibrated_weights.json": file_sha256(WORLD_V018_DIR / "calibrated_weights.json"),
            "data/worlds/world_v0_18/world_v0_18_config.json": file_sha256(WORLD_V018_DIR / "world_v0_18_config.json"),
        },
        "utility_atlas_sha256": EXPECTED_ATLAS_SHA256,
        "budget_seconds": BUDGET_SECONDS,
        "reserve_seconds": RESERVE_SECONDS,
        "scope": [
            "The two field sources need not have equal distributions or equal Q.",
            "Contrasts are effects of a source-and-timing policy, not feedback shares.",
            "An exogenous social field leaves recipient occupancy, movement, and base utility endogenous.",
            "A 20-step concentration run is not a permanent absorbing state.",
            "Intervals are conditional on the frozen 018 weights and this new seed cohort.",
        ],
        "llm_calls": 0,
    }
    payload["protocol_sha256"] = canonical_sha256(payload)
    return payload


def protocol_markdown(protocol: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Experiment 020 protocol",
            "",
            "這份干預是在看過 Experiment 018 與 019 之後制定的。它不是那兩輪之前的預註冊。",
            "",
            f"Protocol SHA-256：`{protocol['protocol_sha256']}`",
            "",
            "Seeds 1700–2699。每個 seed 有 18 個 matched conditions、四條來源時程，以及一個 S0 donor。",
            "Weights 從 018 的 weight_hex 讀取。q05、q15、q25 是校準標籤，不保證 recipient 的實際 Q 等於標籤。",
            "E 使用 recipient 自己截至 t−1 的 shadow memory。O 使用同 seed、weight 0 的 S0 donor 截至 t−1 的 leave-one-out score，再乘上該 condition 的固定 weight。",
            "切換點是 decision t=51，發生在該步所有 agents 決策之前。切換不清空 memory、不複製 donor memory、不重置位置、不消耗額外 RNG。",
            "四條路徑是 EE、EO、OE、OO。branch ID 不進入 choice seed。",
            "主要結果是 t=101–500 累積 visit share 的 HHI。對照是 EE−OE、EO−OO、EE−EO、OE−OO，以及 interaction EE−EO−OE+OO。",
            "早期形成是 18/20 的 20 步區間在 committed t=50 以前確認。維持資格只看 commits t=31–50，並且在切換前固定。",
            "Bootstrap 使用分析種子 20261002 的本地 Random，10000 次，索引 0 是 choice seed 1700，percentile indexes 249 與 9749。",
            "歷史 lineage 57 條必須與 018 的 trajectory hash exact match。額外 replay 的浮點容差是 1e-12。",
            "這不是等 Q 的純 feedback 效果，也不是機制貢獻百分比。外生 social field 不表示 occupancy 與 base utility 沒有 feedback。",
            "",
        ]
    )


def shared_indexes(n_seeds: int, draws: int, rng_seed: int) -> np.ndarray:
    generator = Random(rng_seed)
    indexes = np.empty((draws, n_seeds), dtype=np.int32)
    for draw in range(draws):
        for column in range(n_seeds):
            indexes[draw, column] = generator.randrange(n_seeds)
    return indexes


def bootstrap_column(values: np.ndarray, indexes: np.ndarray) -> np.ndarray:
    output = np.empty(indexes.shape[0], dtype=np.float64)
    cursor = 0
    while cursor < indexes.shape[0]:
        stop = min(cursor + 20, indexes.shape[0])
        output[cursor:stop] = values[indexes[cursor:stop]].mean(axis=1)
        cursor = stop
    return output


def conditional_means(values: np.ndarray, eligible: np.ndarray, indexes: np.ndarray) -> tuple[np.ndarray, int]:
    kept = []
    dropped = 0
    for draw in range(indexes.shape[0]):
        chosen = indexes[draw]
        mask = eligible[chosen]
        if not np.any(mask):
            dropped += 1
            continue
        sample = values[chosen][mask]
        sample = sample[np.isfinite(sample)]
        if sample.size == 0:
            dropped += 1
            continue
        kept.append(float(sample.mean()))
    return np.asarray(kept, dtype=np.float64), dropped


def distribution_summary(samples: list[float], replicates: np.ndarray | None = None) -> dict[str, Any]:
    import statistics

    if not samples:
        return {"n": 0, "mean": None}
    summary: dict[str, Any] = {
        "n": len(samples),
        "mean": statistics.fmean(samples),
        "median": statistics.median(samples),
        "minimum": min(samples),
        "maximum": max(samples),
    }
    if len(samples) > 1:
        summary["stdev"] = statistics.stdev(samples)
        quantiles = statistics.quantiles(samples, n=4, method="inclusive")
        summary["q25"] = quantiles[0]
        summary["q75"] = quantiles[2]
    if replicates is not None and len(replicates):
        low, high = percentile_indexes(len(replicates))
        ranked = np.sort(replicates)
        summary["ci_low"] = float(ranked[low])
        summary["ci_high"] = float(ranked[high])
    return summary


def _empty_level() -> dict[str, Any]:
    return {
        "n": 0,
        "early_formed": 0,
        "event_n": 0,
        "event_absent": 0,
        "late_new": 0,
        "spanning": 0,
        "eligible": 0,
        "maintained": 0,
        "released": 0,
        "transfer": 0,
        "late_any": 0,
        "exit_times": [],
        "release_times": [],
        "starts": [],
        "confirms": [],
        "terminals": {},
        "movement_sum": 0.0,
        "clip_sum": 0,
        "saturation_sum": 0.0,
        "empty_sum": 0.0,
        "field_gap_early_sum": 0.0,
        "field_gap_mid_sum": 0.0,
        "field_gap_late_sum": 0.0,
        "score_range_sum": 0.0,
        "logit_late_sum": 0.0,
        "nonfinite_sum": 0,
    }


def _accumulate_level(level: dict[str, Any], row: dict[str, Any]) -> None:
    level["n"] += 1
    event = row["event18"]
    if row["early_formed"]:
        level["early_formed"] += 1
    if event["occurred"]:
        level["event_n"] += 1
        level["starts"].append(int(event["start"]))
        level["confirms"].append(int(event["confirm"]))
        if row["early_formed"]:
            pass
        elif int(event["start"]) >= SWITCH_INDEX + 1:
            level["late_new"] += 1
        else:
            level["spanning"] += 1
    else:
        level["event_absent"] += 1
    record = row["maintenance"]
    if record["eligible"]:
        level["eligible"] += 1
        level["maintained"] += int(bool(record["maintained_to_horizon"]))
        level["released"] += int(bool(record["released"]))
        level["transfer"] += int(bool(record["late_run_other"]))
        level["late_any"] += int(bool(record["late_run_any"]))
        if record["original_exit"] is not None:
            level["exit_times"].append(int(record["original_exit"]))
        if record["release_time"] is not None:
            level["release_times"].append(int(record["release_time"]))
    terminal = row["late"]["late_dominant"]
    label = "tie" if terminal is None else str(terminal)
    level["terminals"][label] = int(level["terminals"].get(label, 0)) + 1
    level["movement_sum"] += float(row["movement_rate"])
    level["clip_sum"] += int(row["clip_change_count"])
    level["saturation_sum"] += float(row["saturation_frequency"])
    level["empty_sum"] += float(row["empty_frequency"])
    level["field_gap_early_sum"] += float(row["field_gap_t1_50"])
    level["field_gap_mid_sum"] += float(row["field_gap_t51_100"])
    level["field_gap_late_sum"] += float(row["field_gap_t101_500"])
    level["score_range_sum"] += float(row["score_range_t1_500"])
    level["logit_late_sum"] += float(row["logit_range_t101_500"])
    level["nonfinite_sum"] += int(row["nonfinite_count"])


def _rate(count: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return count / denominator


def _median(values: list[int]) -> float | None:
    import statistics

    if not values:
        return None
    return float(statistics.median(values))


def _finalize_level(level: dict[str, Any]) -> dict[str, Any]:
    count = int(level["n"])
    eligible = int(level["eligible"])
    return {
        "n": count,
        "early_formed": int(level["early_formed"]),
        "early_rate": _rate(int(level["early_formed"]), count),
        "event_n": int(level["event_n"]),
        "event_absent": int(level["event_absent"]),
        "late_new": int(level["late_new"]),
        "spanning": int(level["spanning"]),
        "event_start_median": _median(level["starts"]),
        "event_confirm_median": _median(level["confirms"]),
        "eligible_n": eligible,
        "maintained_n": None if eligible == 0 else int(level["maintained"]),
        "maintained_rate": _rate(int(level["maintained"]), eligible),
        "released_n": None if eligible == 0 else int(level["released"]),
        "released_rate": _rate(int(level["released"]), eligible),
        "transfer_n": None if eligible == 0 else int(level["transfer"]),
        "transfer_rate": _rate(int(level["transfer"]), eligible),
        "late_any_n": None if eligible == 0 else int(level["late_any"]),
        "late_any_rate": _rate(int(level["late_any"]), eligible),
        "exit_n": None if eligible == 0 else len(level["exit_times"]),
        "exit_time_median": None if eligible == 0 else _median(level["exit_times"]),
        "release_n": None if eligible == 0 else len(level["release_times"]),
        "release_time_median": None if eligible == 0 else _median(level["release_times"]),
        "terminals": level["terminals"],
        "movement_rate": None if count == 0 else level["movement_sum"] / count,
        "clip_change_mean": None if count == 0 else level["clip_sum"] / count,
        "saturation_frequency": None if count == 0 else level["saturation_sum"] / count,
        "empty_frequency": None if count == 0 else level["empty_sum"] / count,
        "field_gap_t1_50": None if count == 0 else level["field_gap_early_sum"] / count,
        "field_gap_t51_100": None if count == 0 else level["field_gap_mid_sum"] / count,
        "field_gap_t101_500": None if count == 0 else level["field_gap_late_sum"] / count,
        "score_range_mean": None if count == 0 else level["score_range_sum"] / count,
        "logit_range_t101_500": None if count == 0 else level["logit_late_sum"] / count,
        "nonfinite_count": int(level["nonfinite_sum"]),
    }


def _analyze(destination: Path) -> dict[str, Any]:
    columns: dict[str, list[float]] = {}
    eligible_e: dict[str, list[bool]] = {}
    eligible_o: dict[str, list[bool]] = {}
    maintain_e: dict[str, list[float]] = {}
    maintain_o: dict[str, list[float]] = {}
    exit_e: dict[str, list[float]] = {}
    release_e: dict[str, list[float]] = {}
    transfer_e: dict[str, list[float]] = {}
    exit_o: dict[str, list[float]] = {}
    release_o: dict[str, list[float]] = {}
    transfer_o: dict[str, list[float]] = {}
    switch_q: dict[str, list[float]] = {}
    switch_occ: dict[str, list[float]] = {}
    formation_counts = {}
    levels: dict[str, dict[str, dict[str, Any]]] = {}
    donor_changed = 0
    prefix_failed = 0
    for seed in FORMAL_SEEDS:
        payload = json.loads(_metrics_path(destination, seed).read_text(encoding="utf-8"))
        if payload["prefix_failures"]:
            prefix_failed += 1
        if not payload["donor_unchanged"]:
            donor_changed += 1
        by_key = {(row["condition_id"], row["schedule"]): row for row in payload["worlds"]}
        for spec_id in sorted({row["condition_id"] for row in payload["worlds"]}):
            rows = {schedule: by_key[(spec_id, schedule)] for schedule in PATHS}
            hhi = {schedule: float(rows[schedule]["late"]["late_hhi"]) for schedule in PATHS}
            names = {
                f"hhi_{spec_id}_{schedule}": hhi[schedule] for schedule in PATHS
            }
            names[f"early_given_E_{spec_id}"] = hhi["EE"] - hhi["OE"]
            names[f"early_given_O_{spec_id}"] = hhi["EO"] - hhi["OO"]
            names[f"late_given_E_{spec_id}"] = hhi["EE"] - hhi["EO"]
            names[f"late_given_O_{spec_id}"] = hhi["OE"] - hhi["OO"]
            names[f"interaction_{spec_id}"] = hhi["EE"] - hhi["EO"] - hhi["OE"] + hhi["OO"]
            names[f"formation_{spec_id}"] = (1.0 if rows["EE"]["early_formed"] else 0.0) - (1.0 if rows["OE"]["early_formed"] else 0.0)
            for schedule in PATHS:
                names[f"q_{spec_id}_{schedule}"] = float(rows[schedule]["q_t101_500"])
                names[f"q_early_{spec_id}_{schedule}"] = float(rows[schedule]["q_t1_50"])
                names[f"q_after_{spec_id}_{schedule}"] = float(rows[schedule]["q_t51_100"])
            for name, value in names.items():
                columns.setdefault(name, []).append(value)
            eligible_e.setdefault(spec_id, []).append(bool(rows["EE"]["maintenance"]["eligible"]))
            eligible_o.setdefault(spec_id, []).append(bool(rows["OE"]["maintenance"]["eligible"]))
            maintain_e.setdefault(spec_id, []).append(_flag(rows["EE"]["maintenance"], rows["EO"]["maintenance"], "maintained_to_horizon"))
            maintain_o.setdefault(spec_id, []).append(_flag(rows["OE"]["maintenance"], rows["OO"]["maintenance"], "maintained_to_horizon"))
            exit_e.setdefault(spec_id, []).append(_time_gap(rows["EE"]["maintenance"], rows["EO"]["maintenance"], "original_exit"))
            exit_o.setdefault(spec_id, []).append(_time_gap(rows["OE"]["maintenance"], rows["OO"]["maintenance"], "original_exit"))
            release_e.setdefault(spec_id, []).append(_flag(rows["EE"]["maintenance"], rows["EO"]["maintenance"], "released"))
            release_o.setdefault(spec_id, []).append(_flag(rows["OE"]["maintenance"], rows["OO"]["maintenance"], "released"))
            transfer_e.setdefault(spec_id, []).append(_flag(rows["EE"]["maintenance"], rows["EO"]["maintenance"], "late_run_other"))
            transfer_o.setdefault(spec_id, []).append(_flag(rows["OE"]["maintenance"], rows["OO"]["maintenance"], "late_run_other"))
            for schedule in PATHS:
                key = f"{spec_id}_{schedule}"
                switch_q.setdefault(key, [0.0 for _item in rows[schedule]["q_switch"]])
                switch_occ.setdefault(key, [0.0 for _item in rows[schedule]["occ_switch"]])
                for index, value in enumerate(rows[schedule]["q_switch"]):
                    switch_q[key][index] += float(value)
                for index, value in enumerate(rows[schedule]["occ_switch"]):
                    switch_occ[key][index] += float(value)
            formation_counts.setdefault(spec_id, {"E": 0, "O": 0, "n": 0})
            formation_counts[spec_id]["E"] += int(bool(rows["EE"]["early_formed"]))
            formation_counts[spec_id]["O"] += int(bool(rows["OE"]["early_formed"]))
            formation_counts[spec_id]["n"] += 1
            for schedule in PATHS:
                _accumulate_level(levels.setdefault(spec_id, {}).setdefault(schedule, _empty_level()), rows[schedule])
    names = sorted(columns)
    matrix = np.column_stack([columns[name] for name in names])
    indexes = shared_indexes(len(FORMAL_SEEDS), BOOTSTRAP_DRAWS, ANALYSIS_SEED)
    summaries = {}
    for column, name in enumerate(names):
        replicates = bootstrap_column(matrix[:, column], indexes)
        summaries[name] = distribution_summary(matrix[:, column].tolist(), replicates)
    maintenance = {}
    for spec_id in formation_counts:
        maintenance[spec_id] = {
            "E": _maintenance_summary(maintain_e[spec_id], exit_e[spec_id], release_e[spec_id], transfer_e[spec_id], eligible_e[spec_id], indexes),
            "O": _maintenance_summary(maintain_o[spec_id], exit_o[spec_id], release_o[spec_id], transfer_o[spec_id], eligible_o[spec_id], indexes),
        }
    n_seeds = float(len(FORMAL_SEEDS))
    finalized = {spec_id: {schedule: _finalize_level(levels[spec_id][schedule]) for schedule in PATHS} for spec_id in levels}
    payload = {
        "analysis_rng_seed": ANALYSIS_SEED,
        "draws": BOOTSTRAP_DRAWS,
        "percentile_indexes": list(percentile_indexes(BOOTSTRAP_DRAWS)),
        "index_0": FORMAL_SEEDS[0],
        "includes_refitting_uncertainty": False,
        "causal_mediation": False,
        "summaries": summaries,
        "formation": formation_counts,
        "maintenance": maintenance,
        "levels": finalized,
        "switch_q": {key: [value / n_seeds for value in values] for key, values in switch_q.items()},
        "switch_occ": {key: [value / n_seeds for value in values] for key, values in switch_occ.items()},
        "llm_calls": 0,
    }
    write_json(destination / "uncertainty_analysis.json", payload)
    write_json(
        destination / "prefix_donor_audit.json",
        {
            "seeds": len(FORMAL_SEEDS),
            "prefix_failure_seeds": prefix_failed,
            "donor_changed_seeds": donor_changed,
            "passed": prefix_failed == 0 and donor_changed == 0,
            "llm_calls": 0,
        },
    )
    return payload


def _flag(left: dict[str, Any], right: dict[str, Any], key: str) -> float:
    if not left["eligible"]:
        return float("nan")
    return (1.0 if left[key] else 0.0) - (1.0 if right[key] else 0.0)


def _time_gap(left: dict[str, Any], right: dict[str, Any], key: str) -> float:
    if not left["eligible"] or left[key] is None or right[key] is None:
        return float("nan")
    return float(left[key]) - float(right[key])


def _maintenance_summary(maintain: list[float], exit_gap: list[float], release: list[float], transfer: list[float], eligible: list[bool], indexes: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(eligible, dtype=bool)
    count = int(mask.sum())
    if count == 0:
        return {"eligible_n": 0, "maintain_diff": None, "release_diff": None, "transfer_diff": None, "exit_time_diff": None, "bootstrap_replicates_dropped": BOOTSTRAP_DRAWS}
    maintain_rep, maintain_drop = conditional_means(np.asarray(maintain, dtype=np.float64), mask, indexes)
    release_rep, _release_drop = conditional_means(np.asarray(release, dtype=np.float64), mask, indexes)
    transfer_rep, _transfer_drop = conditional_means(np.asarray(transfer, dtype=np.float64), mask, indexes)
    exit_values = np.asarray(exit_gap, dtype=np.float64)
    exit_mask = mask & np.isfinite(exit_values)
    exit_rep, exit_drop = conditional_means(exit_values, exit_mask, indexes) if np.any(exit_mask) else (np.asarray([]), BOOTSTRAP_DRAWS)
    observed_m = [value for value, flag in zip(maintain, eligible) if flag]
    observed_r = [value for value, flag in zip(release, eligible) if flag]
    observed_t = [value for value, flag in zip(transfer, eligible) if flag]
    observed_e = [value for value, flag in zip(exit_gap, eligible) if flag and value == value]
    return {
        "eligible_n": count,
        "maintain_diff": distribution_summary(observed_m, maintain_rep),
        "release_diff": distribution_summary(observed_r, release_rep),
        "transfer_diff": distribution_summary(observed_t, transfer_rep),
        "exit_time_diff": distribution_summary(observed_e, exit_rep if len(exit_rep) else None),
        "exit_pairs": len(observed_e),
        "bootstrap_replicates_dropped": maintain_drop,
        "exit_bootstrap_replicates_dropped": exit_drop,
    }


def _fmt(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.6g}"


def _ci(item: dict[str, Any] | None) -> str:
    if not item or item.get("mean") is None or "ci_low" not in item:
        return "NA"
    return f"[{_fmt(item['ci_low'])}, {_fmt(item['ci_high'])}]"


def write_report(destination: Path, analysis: dict[str, Any], elapsed: float) -> None:
    status = json.loads((destination / "completion_status.json").read_text(encoding="utf-8"))
    summaries = analysis["summaries"]
    contrast_rows = []
    formation_rows = []
    maintenance_rows = []
    for cell_id in SOCIAL_CELL_IDS:
        for label in Q_LABELS:
            spec_id = condition_id(cell_id, label)
            for name in (f"early_given_E_{spec_id}", f"early_given_O_{spec_id}", f"late_given_E_{spec_id}", f"late_given_O_{spec_id}", f"interaction_{spec_id}"):
                item = summaries[name]
                contrast_rows.append([name, _fmt(item["mean"]), _ci(item)])
            formed = analysis["formation"][spec_id]
            formation_rows.append([spec_id, str(formed["E"]), str(formed["O"]), str(formed["n"]), _fmt(summaries[f"formation_{spec_id}"]["mean"]), _ci(summaries[f"formation_{spec_id}"])])
            for stratum in ("E", "O"):
                block = analysis["maintenance"][spec_id][stratum]
                diff = block["maintain_diff"]
                maintenance_rows.append([spec_id, stratum, str(block["eligible_n"]), _fmt(None if diff is None else diff.get("mean")), _ci(diff)])
    level_rows = []
    for cell_id in SOCIAL_CELL_IDS:
        for label in Q_LABELS:
            spec_id = condition_id(cell_id, label)
            for schedule in PATHS:
                item = analysis["levels"][spec_id][schedule]
                level_rows.append(
                    [
                        spec_id,
                        schedule,
                        _fmt(item["early_rate"]),
                        str(item["event_absent"]),
                        str(item["late_new"]),
                        str(item["spanning"]),
                        str(item["eligible_n"]),
                        _fmt(item["maintained_rate"]),
                        _fmt(item["released_rate"]),
                        _fmt(item["transfer_rate"]),
                        _fmt(item["exit_time_median"]),
                        _fmt(item["field_gap_t101_500"]),
                        _fmt(summaries[f"q_{spec_id}_{schedule}"]["mean"]),
                        _fmt(summaries[f"hhi_{spec_id}_{schedule}"]["mean"]),
                    ]
                )
    text = "\n".join(
        [
            "# Experiment 020 報告",
            "",
            f"狀態：{status['status']}。這是看過 018 與 019 之後的 social-field 來源切換干預，不是等 Q 的純 feedback 分解，也沒有重配 weights。",
            "",
            "E 使用 recipient 自己的 shadow memory。O 使用同 seed 的 S0 donor field。Recipient 的 occupancy、移動與 base utility 始終來自自己。",
            "對照是指定時間與來源的政策效果，不是機制貢獻百分比。連續 20 步的集中不是永久吸收態。",
            "",
            "歷史 lineage 57。新正式世界 73000，其中整合子集 219 已包含在內。額外 replay 219。合計世界執行 73276。",
            f"正式執行器耗時 {elapsed:.1f} 秒。LLM calls = 0。",
            "",
            "## HHI factorial contrasts",
            "",
            "| contrast | mean | CI |",
            "| --- | --- | --- |",
            *["| " + " | ".join(row) + " |" for row in contrast_rows],
            "",
            "## 早期形成",
            "",
            "E 與 O 各算一次，不把 EE/EO 或 OE/OO 的共同 prefix 重複計入。",
            "",
            "| condition | E formed | O formed | n | risk difference | CI |",
            "| --- | --- | --- | --- | --- | --- |",
            *["| " + " | ".join(row) + " |" for row in formation_rows],
            "",
            "## 切換前已集中者的維持差",
            "",
            "資格由 commits t=31–50 決定。維持差是早期來源固定時，晚期 E 減晚期 O 的維持指標差。分母為 0 時結果是 NA。",
            "",
            "| condition | stratum | eligible n | maintain E minus O | CI |",
            "| --- | --- | --- | --- | --- |",
            *["| " + " | ".join(row) + " |" for row in maintenance_rows],
            "",
            "## 各路徑的形成、維持與實際影響",
            "",
            "維持率、解除率與場所轉移率的分母是切換前資格。分母為 0 時這些欄位是 NA，不是 0。晚期新形成是第一個 18/20 事件的起點在 t=51 之後。跨越切換是事件已開始但尚未在 t=50 前確認。",
            "",
            "| condition | path | early rate | absent | late new | spanning | eligible | maintained | released | transfer | exit median | late field gap | late Q | late HHI |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            *["| " + " | ".join(row) + " |" for row in level_rows],
            "",
            "區間條件於 018 已固定的 weights、本輪干預規則與 seeds 1700–2699，不包含重新 fitting 的不確定性。",
            "",
        ]
    )
    (destination / "EXPERIMENT_020_REPORT.md").write_text(text, encoding="utf-8", newline="\n")
    REPORT_DOC.write_text(text, encoding="utf-8", newline="\n")


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
        _figure_hhi_q(analysis, plt, figure_dir / "hhi_q.png")
        _figure_switch(analysis, plt, figure_dir / "switch.png")
        _figure_formation(analysis, plt, figure_dir / "formation.png")
        _figure_maintenance(analysis, plt, figure_dir / "maintenance.png")
        _figure_transfer(analysis, plt, figure_dir / "transfer.png")
        _figure_contrasts(analysis, plt, figure_dir / "contrasts.png")
        status.update({"written": True, "font": font})
    except Exception as exc:
        status["reason"] = str(exc)
    write_json(destination / "figure_status.json", status)


def _figure_hhi_q(analysis: dict[str, Any], plt: Any, path: Path) -> None:
    labels = [condition_id(cell_id, "q25") for cell_id in SOCIAL_CELL_IDS]
    figure, axes = plt.subplots(2, 1, figsize=(12, 8))
    positions = np.arange(len(labels))
    width = 0.18
    for axis, prefix in ((axes[0], "hhi"), (axes[1], "q")):
        for offset, schedule in enumerate(PATHS):
            means = [analysis["summaries"][f"{prefix}_{label}_{schedule}"]["mean"] for label in labels]
            axis.bar(positions + (offset - 1.5) * width, means, width=width, label=schedule)
        axis.set_xticks(positions, [label.replace("__", "\n") for label in labels], fontsize=7)
        axis.legend(fontsize=7)
    axes[0].set_ylabel("late HHI")
    axes[1].set_ylabel("Q used, t=101-500")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _figure_switch(analysis: dict[str, Any], plt: Any, path: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    key_root = "CUM_LOGMAX__q25"
    xs = list(range(SWITCH_T_START, SWITCH_T_STOP + 1))
    for schedule in PATHS:
        key = f"{key_root}_{schedule}"
        axes[0].plot(xs[: len(analysis["switch_q"][key])], analysis["switch_q"][key], label=schedule)
        axes[1].plot(xs[: len(analysis["switch_occ"][key])], analysis["switch_occ"][key], label=schedule)
    for axis in axes:
        axis.axvline(51, color="black", linewidth=0.8)
        axis.legend(fontsize=7)
    axes[0].set_ylabel("mean Q used")
    axes[1].set_ylabel("mean max occupancy")
    axes[1].set_xlabel("decision or committed timestep")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _figure_formation(analysis: dict[str, Any], plt: Any, path: Path) -> None:
    labels = []
    early = []
    other = []
    for cell_id in SOCIAL_CELL_IDS:
        for label in Q_LABELS:
            spec_id = condition_id(cell_id, label)
            labels.append(spec_id.replace("__", "\n"))
            formed = analysis["formation"][spec_id]
            early.append(formed["E"] / formed["n"])
            other.append(formed["O"] / formed["n"])
    positions = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(14, 5))
    axis.bar(positions - 0.15, early, width=0.3, label="E early")
    axis.bar(positions + 0.15, other, width=0.3, label="O early")
    axis.set_xticks(positions, labels, fontsize=6)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _figure_maintenance(analysis: dict[str, Any], plt: Any, path: Path) -> None:
    labels = []
    early = []
    later = []
    for cell_id in SOCIAL_CELL_IDS:
        for label in Q_LABELS:
            spec_id = condition_id(cell_id, label)
            ee = analysis["levels"][spec_id]["EE"]
            eo = analysis["levels"][spec_id]["EO"]
            if ee["eligible_n"] == 0:
                continue
            labels.append(f"{spec_id}\nn={ee['eligible_n']}")
            early.append(ee["maintained_rate"])
            later.append(eo["maintained_rate"])
    figure, axis = plt.subplots(figsize=(12, 5))
    if labels:
        positions = np.arange(len(labels))
        axis.bar(positions - 0.15, early, width=0.3, label="EE")
        axis.bar(positions + 0.15, later, width=0.3, label="EO")
        axis.set_xticks(positions, labels, fontsize=6)
        axis.legend()
    axis.set_ylabel("original place maintained to t=500")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _figure_transfer(analysis: dict[str, Any], plt: Any, path: Path) -> None:
    labels = []
    released = []
    transferred = []
    exited = []
    for cell_id in SOCIAL_CELL_IDS:
        for label in ("q15", "q25"):
            spec_id = condition_id(cell_id, label)
            item = analysis["levels"][spec_id]["EE"]
            if item["eligible_n"] == 0 or item["maintained_rate"] is None:
                continue
            labels.append(spec_id.replace("__", "\n"))
            released.append(item["released_rate"])
            transferred.append(item["transfer_rate"])
            exited.append(1.0 - item["maintained_rate"])
    positions = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(12, 5))
    if len(labels):
        axis.bar(positions - 0.25, exited, width=0.25, label="left original place")
        axis.bar(positions, released, width=0.25, label="concentration released")
        axis.bar(positions + 0.25, transferred, width=0.25, label="moved to another place")
        axis.set_xticks(positions, labels, fontsize=6)
        axis.legend(fontsize=7)
    axis.set_ylabel("share of pre-switch eligible seeds")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _figure_contrasts(analysis: dict[str, Any], plt: Any, path: Path) -> None:
    labels = []
    means = []
    for cell_id in SOCIAL_CELL_IDS:
        for label in Q_LABELS:
            spec_id = condition_id(cell_id, label)
            labels.append(spec_id.replace("__", "\n"))
            means.append(analysis["summaries"][f"interaction_{spec_id}"]["mean"])
    figure, axis = plt.subplots(figsize=(14, 5))
    axis.bar(np.arange(len(labels)), means)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(np.arange(len(labels)), labels, fontsize=6)
    axis.set_ylabel("HHI interaction")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _manifest(destination: Path) -> None:
    files = {}
    for path in sorted(destination.rglob("*")):
        if path.is_file() and path.name != "run_manifest.json":
            files[str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")] = file_sha256(path)
    write_json(destination / "run_manifest.json", {"world_id": "world_v0_20", "files": files, "llm_calls": 0})


def _update_indexes(status: str) -> None:
    pointer = f"- 020 狀態為 {status}。見 [`docs/EXPERIMENT_020_REPORT.md`](docs/EXPERIMENT_020_REPORT.md)。協議見 [`docs/EXPERIMENT_020_PROTOCOL.md`](docs/EXPERIMENT_020_PROTOCOL.md)。\n"
    readme = PROJECT_ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    if "020 狀態為" not in text:
        text = text.replace(
            "- 019 狀態為 COMPLETE。見 [`docs/EXPERIMENT_019_REPORT.md`](docs/EXPERIMENT_019_REPORT.md)。協議見 [`docs/EXPERIMENT_019_PROTOCOL.md`](docs/EXPERIMENT_019_PROTOCOL.md)。\n",
            "- 019 狀態為 COMPLETE。見 [`docs/EXPERIMENT_019_REPORT.md`](docs/EXPERIMENT_019_REPORT.md)。協議見 [`docs/EXPERIMENT_019_PROTOCOL.md`](docs/EXPERIMENT_019_PROTOCOL.md)。\n" + pointer,
        )
        readme.write_text(text, encoding="utf-8", newline="\n")
    experiments = PROJECT_ROOT / "docs" / "EXPERIMENTS.md"
    index = experiments.read_text(encoding="utf-8")
    if "Experiment 020" not in index:
        block = (
            "\n### Experiment 020 — social-field source switching\n\n"
            f"020 狀態為 {status}。這是看過 018 與 019 之後的來源切換干預。協議見 [`EXPERIMENT_020_PROTOCOL.md`](EXPERIMENT_020_PROTOCOL.md)。結果見 [`EXPERIMENT_020_REPORT.md`](EXPERIMENT_020_REPORT.md)。\n\n"
        )
        index = index.replace("\n## 輸出契約\n", block + "## 輸出契約\n")
        experiments.write_text(index, encoding="utf-8", newline="\n")


def _replay(destination: Path, conditions: list[dict[str, Any]], workers: int) -> dict[str, Any]:
    tasks = [{"mode": "formal", "choice_seed": seed, "conditions": conditions} for seed in GATE_SEEDS]
    fresh_rows = _run_pool(tasks, workers, 900)
    results = []
    for fresh in fresh_rows:
        seed = int(fresh["choice_seed"])
        metrics_path = _metrics_path(destination, seed)
        trajectory_path = _trajectory_path(destination, seed)
        before_metrics = file_sha256(metrics_path)
        before_trajectory = file_sha256(trajectory_path)
        saved = json.loads(metrics_path.read_text(encoding="utf-8"))
        saved_worlds = {(row["condition_id"], row["schedule"]): row for row in saved["worlds"]}
        fresh_worlds = {(row["condition_id"], row["schedule"]): row for row in fresh["worlds"]}
        hash_match = saved["donor_trajectory_sha256"] == fresh["donor_trajectory_sha256"] and saved["donor_field_sha256"] == fresh["donor_field_sha256"]
        max_gap = 0.0
        for key, row in saved_worlds.items():
            other = fresh_worlds[key]
            hash_match = hash_match and row["trajectory_sha256"] == other["trajectory_sha256"] and row["prefix_sha256"] == other["prefix_sha256"]
            max_gap = max(max_gap, abs(float(row["late"]["late_hhi"]) - float(other["late"]["late_hhi"])))
            max_gap = max(max_gap, abs(float(row["q_t101_500"]) - float(other["q_t101_500"])))
            hash_match = hash_match and row["event18"] == other["event18"] and row["maintenance"] == other["maintenance"]
        after_metrics = file_sha256(metrics_path)
        after_trajectory = file_sha256(trajectory_path)
        results.append(
            {
                "choice_seed": seed,
                "hash_match": hash_match,
                "max_diagnostic_gap": max_gap,
                "file_rewritten": before_metrics != after_metrics or before_trajectory != after_trajectory,
                "llm_calls": 0,
            }
        )
    passed = len(results) == 3 and all(item["hash_match"] and item["max_diagnostic_gap"] <= REPLAY_FLOAT_TOLERANCE and not item["file_rewritten"] for item in results)
    return {"passed": passed, "count": 219, "seeds": results, "tolerance": REPLAY_FLOAT_TOLERANCE, "llm_calls": 0}


def _formal_tasks(seeds: list[int], conditions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"mode": "formal", "choice_seed": seed, "conditions": conditions} for seed in seeds]


def run_experiment(output_dir: Path | None = None, *, resume: bool = False, workers: int | None = None) -> Path:
    destination = output_dir or OUTPUT_DIR
    if destination.resolve() in {WORLD_V018_DIR.resolve(), ATLAS_DIR.resolve()}:
        raise RuntimeError("experiment 020 must not write into a frozen source directory")
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
        conditions = matched_conditions(weights)
        if resume:
            protocol = json.loads((destination / "experiment_020_protocol.json").read_text(encoding="utf-8"))
            config = json.loads((destination / "world_v0_20_config.json").read_text(encoding="utf-8"))
            if protocol["source_sha256"]["world/experiment_020.py"] != file_sha256(Path(__file__).resolve()):
                raise RuntimeError("resume stopped because experiment_020.py no longer matches the frozen protocol")
            if config["protocol_sha256"] != protocol["protocol_sha256"] or config["weights_hex"] != protocol["weights_hex"]:
                raise RuntimeError("resume stopped because the frozen config does not match the protocol")
            if (destination / "failure_record.json").exists():
                raise RuntimeError("resume stopped because a previous attempt recorded a failure")
            audit = json.loads((destination / "seed_audit.json").read_text(encoding="utf-8"))
        else:
            stage = "seed_audit"
            audit = audit_choice_seeds()
            if not audit["passed"]:
                raise RuntimeError("proposed seeds overlap an existing choice seed or a required source is missing")
            refuse_nonempty_output(destination)
            destination.mkdir(parents=True, exist_ok=True)
            write_json(destination / "seed_audit.json", audit)
            protocol = build_protocol(file_sha256(destination / "seed_audit.json"), started_at, weights)
            write_json(destination / "experiment_020_protocol.json", protocol)
            PROTOCOL_DOC.write_text(protocol_markdown(protocol), encoding="utf-8", newline="\n")
            config = {
                "world_id": "world_v0_20",
                "status_at_config_freeze": "CONFIG_FROZEN_BEFORE_WORLDS",
                "protocol_sha256": protocol["protocol_sha256"],
                "weights_hex": protocol["weights_hex"],
                "source_sha256": protocol["source_sha256"],
                "python": sys.version,
                "platform": platform.platform(),
                "llm_calls": 0,
                "started_at": started_at,
            }
            config["config_sha256"] = canonical_sha256(config)
            write_json(destination / "world_v0_20_config.json", config)
            write_json(destination / "source_manifest.json", {"record_type": "experiment_020_source_manifest", "source_sha256": protocol["source_sha256"], "seed_audit_passed": True, "calibration_cache_scanned": False, "llm_calls": 0})
        _mark(destination, "protocol_ready", started, workers=worker_count, audit_passed=audit["passed"])
        stage = "lineage"
        if not (destination / "lineage_audit.json").exists():
            targets = load_lineage_targets()
            lineage_tasks = []
            for seed in LINEAGE_SEEDS:
                lineage_tasks.append(
                    {
                        "mode": "lineage",
                        "choice_seed": seed,
                        "conditions": conditions,
                        "expected": {"S0": targets[("S0", seed)], **{spec["condition_id"]: targets[(spec["condition_id"], seed)] for spec in conditions}},
                    }
                )
            lineage_rows = _run_pool(lineage_tasks, worker_count, 900)
            failures = [row for row in lineage_rows if row["failures"]]
            lineage = {"passed": not failures, "compared": sum(row["compared"] for row in lineage_rows), "failures": failures, "llm_calls": 0}
            write_json(destination / "lineage_audit.json", lineage)
            if not lineage["passed"]:
                raise RuntimeError("historical lineage did not exact-match Experiment 018")
        else:
            lineage = json.loads((destination / "lineage_audit.json").read_text(encoding="utf-8"))
            if not lineage["passed"]:
                raise RuntimeError("historical lineage did not exact-match Experiment 018")
        _mark(destination, "lineage_passed", started, compared=lineage["compared"])
        stage = "gate"
        pending_gate = [seed for seed in GATE_SEEDS if not _metrics_path(destination, seed).exists()]
        if pending_gate:
            if time.time() > deadline:
                raise BudgetStop("budget reserve reached before the integration seeds")
            for payload in _run_pool(_formal_tasks(pending_gate, conditions), worker_count, 900):
                if payload["prefix_failures"] or not payload["donor_unchanged"]:
                    write_json(destination / "prefix_failure.json", {"choice_seed": payload["choice_seed"], "prefix_failures": payload["prefix_failures"], "donor_unchanged": payload["donor_unchanged"], "llm_calls": 0})
                    raise RuntimeError(f"structural gate failed at seed {payload['choice_seed']}")
                _write_formal(destination, payload)
        write_json(destination / "integration_gate.json", {"passed": True, "seeds": list(GATE_SEEDS), "included_in_new_worlds": True, "additional_count": 0, "llm_calls": 0})
        stage = "worlds"
        remaining = [seed for seed in FORMAL_SEEDS if seed not in GATE_SEEDS and not _metrics_path(destination, seed).exists()]
        stage_started = time.perf_counter()
        for offset in range(0, len(remaining), worker_count):
            if time.time() > deadline:
                raise BudgetStop("budget reserve reached at a seed boundary")
            batch = remaining[offset : offset + worker_count]
            for payload in _run_pool(_formal_tasks(batch, conditions), worker_count, 900):
                if payload["prefix_failures"] or not payload["donor_unchanged"]:
                    write_json(destination / "prefix_failure.json", {"choice_seed": payload["choice_seed"], "prefix_failures": payload["prefix_failures"], "donor_unchanged": payload["donor_unchanged"], "llm_calls": 0})
                    raise RuntimeError(f"structural gate failed at seed {payload['choice_seed']}")
                _write_formal(destination, payload)
            done = offset + len(batch)
            elapsed = time.perf_counter() - stage_started
            rate = done / elapsed if elapsed else 0.0
            eta = (len(remaining) - done) / rate if rate else None
            _mark(destination, "worlds", started, done=done, total=len(remaining), eta_seconds=None if eta is None else round(eta, 1))
        if any(not _metrics_path(destination, seed).exists() for seed in FORMAL_SEEDS):
            raise RuntimeError("the 1000-seed cohort is incomplete")
        stage = "analysis"
        _shutdown_pool()
        analysis = _analyze(destination)
        audit = json.loads((destination / "prefix_donor_audit.json").read_text(encoding="utf-8"))
        if not audit["passed"]:
            raise RuntimeError("prefix or donor audit failed after the formal cohort")
        stage = "replay"
        replay = _replay(destination, conditions, worker_count)
        write_json(destination / "replay_audit.json", replay)
        _shutdown_pool()
        if not replay["passed"]:
            raise RuntimeError("diagnostic replay did not match the saved worlds")
        decision = completion_decision(lineage_passed=True, gates_passed=True, replay_passed=True, seeds_complete=True)
        write_json(
            destination / "completion_status.json",
            {
                "status": decision,
                "reason": "verified",
                "elapsed_seconds": time.perf_counter() - started,
                "lineage": 57,
                "new_worlds": 73000,
                "extra_replays": 219,
                "total_world_executions": 73276,
                "llm_calls": 0,
            },
        )
        _write_figures(destination, analysis)
        write_report(destination, analysis, time.perf_counter() - started)
        _manifest(destination)
        _update_indexes(decision)
        _mark(destination, "finished", started, status=decision)
        return destination
    except BudgetStop as exc:
        _shutdown_pool()
        done = sum(1 for seed in FORMAL_SEEDS if _metrics_path(destination, seed).exists())
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
            raise RuntimeError("experiment 020 has a failure record and will not resume a failed run")
        status_path = destination / "completion_status.json"
        if status_path.exists() and json.loads(status_path.read_text(encoding="utf-8")).get("status") == "COMPLETE":
            raise RuntimeError("experiment 020 is already complete")
        if not (destination / "experiment_020_protocol.json").exists():
            raise RuntimeError("experiment 020 output exists without a protocol")
        resume = True
    run_experiment(resume=resume)


if __name__ == "__main__":
    main()
