"""Experiment 017: memory representation crossed with score transformation.

The runner composes Experiment 004 and 005 functions. It does not modify those
modules, does not call a model, and writes only under data/worlds/world_v0_17.
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from random import Random
from typing import Any

from agents.persona_generator import write_json
from config.settings import PROJECT_ROOT
from experiments.reproducibility_check import canonical_sha256
from world.choice import agent_reconsiders, choice_seed
from world.dynamics_calibration import load_initial_locations, measure_world
from world.experiment_002 import REFERENCE_CHOICE_TEMPERATURE, REFERENCE_RECONSIDERATION_RATE
from world.experiment_004 import (
    SocialState,
    _argmax,
    _entropy,
    _hhi,
    _select,
    _share,
    _unique_leader,
    _utilities,
    commit_social,
    empty_social,
    popularity_scores,
    trajectory_hash,
)
from world.experiment_004 import OUTPUT_DIR as WORLD_V04_DIR
from world.experiment_005 import (
    SOCIAL_WEIGHT,
    DecayMemory,
    RollingMemory,
    _population_sd,
    _presence,
    centered_scores,
)
from world.experiment_005 import OUTPUT_DIR as WORLD_V05_DIR
from world.geography import PLACE_IDS, distance_matrix, occupancy_of, place_index, travel_distance
from world.utility_atlas import OUTPUT_DIR as ATLAS_DIR
from world.utility_atlas import load_frozen_atlas

OUTPUT_DIR = PROJECT_ROOT / "data" / "worlds" / "world_v0_17"
DESIGN_PROTOCOL = PROJECT_ROOT / "docs" / "experiment_017_design" / "protocol_draft.json"
REPORT_PATH = PROJECT_ROOT / "docs" / "EXPERIMENT_017_REPORT.md"
EXPECTED_ATLAS_SHA256 = "664b511d208f7989e8a671d0e2d6e6433366120bb3d5d6a67db38277b4d5a373"
HASH_CHUNK_BYTES = 1024 * 1024
TRACE_SCAN_LIMIT = 2_000_000
TIMESTEPS = 500
LATE_START = 101
FAMILIARITY_WEIGHT = 0.0
SOCIAL_CELL_WEIGHT = 150.0
S0_WEIGHT = 0.0
BOOTSTRAP_DRAWS = 10_000
COHORT_A_ANALYSIS_SEED = 20260928
COHORT_B_ANALYSIS_SEED = 20260929
BUDGET_START = datetime.fromisoformat("2026-09-28T03:06:15")
BUDGET_SECONDS = 5 * 60 * 60
RESERVE_SECONDS = 40 * 60
COHORT_A_SEEDS = tuple(range(50))
COHORT_B_SEEDS = tuple(range(200, 700))
CELLS = (
    {"cell_id": "CUM_LOGMAX", "memory": "cumulative", "transform": "LOGMAX", "window": None, "decay": None, "social_weight": SOCIAL_CELL_WEIGHT, "historical": True},
    {"cell_id": "CUM_ZRAW", "memory": "cumulative", "transform": "ZRAW", "window": None, "decay": None, "social_weight": SOCIAL_CELL_WEIGHT, "historical": False},
    {"cell_id": "RW50_LOGMAX", "memory": "rolling", "transform": "LOGMAX", "window": 50, "decay": None, "social_weight": SOCIAL_CELL_WEIGHT, "historical": False},
    {"cell_id": "RW50_ZRAW", "memory": "rolling", "transform": "ZRAW", "window": 50, "decay": None, "social_weight": SOCIAL_CELL_WEIGHT, "historical": True},
    {"cell_id": "EXP99_LOGMAX", "memory": "exponential", "transform": "LOGMAX", "window": None, "decay": 0.99, "social_weight": SOCIAL_CELL_WEIGHT, "historical": False},
    {"cell_id": "EXP99_ZRAW", "memory": "exponential", "transform": "ZRAW", "window": None, "decay": 0.99, "social_weight": SOCIAL_CELL_WEIGHT, "historical": True},
    {"cell_id": "S0", "memory": "cumulative", "transform": "LOGMAX", "window": None, "decay": None, "social_weight": S0_WEIGHT, "historical": True},
)
CELL_BY_ID = {spec["cell_id"]: spec for spec in CELLS}
SOCIAL_CELL_IDS = tuple(spec["cell_id"] for spec in CELLS if spec["cell_id"] != "S0")
LINEAGE_CELL_IDS = ("CUM_LOGMAX", "RW50_ZRAW", "EXP99_ZRAW", "S0")
NEW_CROSS_CELL_IDS = ("CUM_ZRAW", "RW50_LOGMAX", "EXP99_LOGMAX")
REPLAY_A = tuple((cell_id, seed) for cell_id in NEW_CROSS_CELL_IDS for seed in (0, 49))
REPLAY_B = tuple((cell_id, seed) for cell_id in tuple(CELL_BY_ID) for seed in (200, 699))
_WORKER: dict[str, Any] = {}
_POOL: ProcessPoolExecutor | None = None


def paired_seed_indexes(n_seeds: int, draws: int, rng_seed: int) -> list[list[int]]:
    generator = Random(rng_seed)
    return [[generator.randrange(n_seeds) for _column in range(n_seeds)] for _draw in range(draws)]


def percentile_indexes(draws: int) -> tuple[int, int]:
    return int(0.025 * (draws - 1)), int(0.975 * (draws - 1))


def resampled_mean(values: list[float], indexes: list[int]) -> float:
    return statistics.fmean(values[index] for index in indexes)


def refuse_nonempty_output(destination: Path) -> None:
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError("experiment 017 refuses to write into a non-empty output directory")


def file_sha256(path: Path, chunk_size: int = HASH_CHUNK_BYTES) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def pack_series(series: list[dict[str, str]], agents: list[str]) -> dict[str, Any]:
    digits = {place_id: str(index + 1) for index, place_id in enumerate(PLACE_IDS)}
    return {
        "agents": list(agents),
        "places": list(PLACE_IDS),
        "steps": ["".join(digits[step[agent_id]] for agent_id in agents) for step in series],
    }


def unpack_series(payload: dict[str, Any]) -> list[dict[str, str]]:
    agents = [str(agent_id) for agent_id in payload["agents"]]
    places = [str(place_id) for place_id in payload["places"]]
    if places != list(PLACE_IDS):
        raise RuntimeError("packed trajectory places are not the frozen order")
    series = []
    for step in payload["steps"]:
        text = str(step)
        if len(text) != len(agents):
            raise RuntimeError("packed trajectory width does not match agents")
        locations = {}
        for agent_id, digit in zip(agents, text, strict=True):
            index = int(digit) - 1
            if index < 0 or index >= len(PLACE_IDS):
                raise RuntimeError("packed trajectory digit is outside the place order")
            locations[agent_id] = PLACE_IDS[index]
        series.append(locations)
    return series


def make_memory(spec: dict[str, Any], agents: list[str]) -> SocialState | RollingMemory | DecayMemory:
    if spec["memory"] == "cumulative":
        return empty_social(agents)
    if spec["memory"] == "rolling":
        return RollingMemory(agents, int(spec["window"]))
    if spec["memory"] == "exponential":
        return DecayMemory(agents, float(spec["decay"]))
    raise RuntimeError("unknown memory representation")


def _require_finite(values: dict[str, float], label: str) -> None:
    for place_id in PLACE_IDS:
        if not math.isfinite(values[place_id]):
            raise RuntimeError(f"nonfinite {label} at {place_id}")


def _loo(memory: SocialState | RollingMemory | DecayMemory, agent_id: str) -> dict[str, float]:
    if isinstance(memory, SocialState):
        raw = {}
        for place_id in PLACE_IDS:
            count = memory.total[place_id] - memory.own[agent_id][place_id]
            if count < 0:
                raise RuntimeError("leave-one-out social presence went negative")
            raw[place_id] = float(count)
        return raw
    return {place_id: float(value) for place_id, value in memory.loo_raw(agent_id).items()}


def _logmax_from_loo(loo: dict[str, float]) -> dict[str, float]:
    state = getattr(_logmax_from_loo, "state", None)
    if state is None:
        state = empty_social(["focal", "other"])
        _logmax_from_loo.state = state
    for place_id in PLACE_IDS:
        value = loo[place_id]
        if value < 0:
            raise RuntimeError("leave-one-out social presence went negative")
        state.total[place_id] = value
        state.own["focal"][place_id] = 0
        state.own["other"][place_id] = value
    return popularity_scores(state, "focal")


def score_memory(memory: SocialState | RollingMemory | DecayMemory, agent_id: str, transform: str) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    if isinstance(memory, SocialState) and transform == "LOGMAX":
        loo = _loo(memory, agent_id)
        _require_finite(loo, "raw memory")
        scores = popularity_scores(memory, agent_id)
        _require_finite(scores, "logmax score")
        return scores, scores, loo
    loo = _loo(memory, agent_id)
    _require_finite(loo, "raw memory")
    if transform == "LOGMAX":
        scores = _logmax_from_loo(loo)
        _require_finite(scores, "logmax score")
        return scores, scores, loo
    if transform != "ZRAW":
        raise RuntimeError("unknown score transformation")
    unclipped, clipped = centered_scores(loo)
    _require_finite(unclipped, "unclipped z score")
    _require_finite(clipped, "clipped z score")
    return unclipped, clipped, loo


def _commit(memory: SocialState | RollingMemory | DecayMemory, destinations: dict[str, str]) -> SocialState | RollingMemory | DecayMemory:
    if isinstance(memory, SocialState):
        return commit_social(memory, destinations)
    memory.commit(destinations)
    return memory


def _base_utilities(atlas: Any, matrix: dict[str, dict[str, float]], places: dict[str, dict[str, Any]], agent_id: str, origin: str, occupancy: dict[str, int]) -> dict[str, float]:
    bases = {}
    for place_id in PLACE_IDS:
        capacity = int(places[place_id]["capacity"])
        distance = travel_distance(matrix, origin, place_id)
        bases[place_id] = float(atlas.lookup(agent_id, place_id, distance, occupancy[place_id], capacity))
        if not math.isfinite(bases[place_id]):
            raise RuntimeError("nonfinite base utility")
    return bases


def _probabilities(utilities: dict[str, float]) -> dict[str, float]:
    _choice, probabilities = _select(list(PLACE_IDS), [utilities[place_id] for place_id in PLACE_IDS], 0.0)
    return probabilities


def simulate_cell(
    atlas: Any,
    locations: dict[str, str],
    cell_id: str,
    choice_seed_value: int,
    *,
    timesteps: int = TIMESTEPS,
    social_weight: float | None = None,
    matrix: dict[str, dict[str, float]] | None = None,
    places: dict[str, dict[str, Any]] | None = None,
    keep_series: bool = False,
) -> dict[str, Any]:
    spec = CELL_BY_ID[cell_id]
    weight = float(spec["social_weight"] if social_weight is None else social_weight)
    agents = sorted(locations)
    matrix = distance_matrix() if matrix is None else matrix
    places = place_index() if places is None else places
    current = {agent_id: locations[agent_id] for agent_id in agents}
    memory = make_memory(spec, agents)
    series = [dict(current)]
    moves = 0
    decisions_n = 0
    reconsidered_n = 0
    rank_n = 0
    dest_n = 0
    sd_sum = 0.0
    range_sum = 0.0
    social_sd_sum = 0.0
    mass_sum = 0.0
    zero_var_n = 0
    clip_changes = 0
    max_abs_raw = 0.0
    max_abs_unclipped = 0.0
    for timestep in range(timesteps):
        occupancy = occupancy_of(current)
        decisions: dict[str, str] = {}
        for agent_id in agents:
            origin = current[agent_id]
            unclipped, scores, loo = score_memory(memory, agent_id, spec["transform"])
            bases, totals, social_term = _utilities(atlas, matrix, places, agent_id, origin, occupancy, scores, weight)
            _require_finite(social_term, "social utility")
            _require_finite(totals, "total utility")
            if weight == 0.0:
                for place_id in PLACE_IDS:
                    if social_term[place_id] != 0.0 or totals[place_id] != bases[place_id]:
                        raise RuntimeError("weight 0 did not remove social utility")
            does_reconsider = agent_reconsiders(choice_seed_value, timestep, agent_id, REFERENCE_RECONSIDERATION_RATE)
            if _argmax(totals) != _argmax(bases):
                rank_n += 1
            if does_reconsider:
                draw = Random(choice_seed(choice_seed_value, timestep, agent_id)).random()
                destination, _probabilities_social = _select(list(PLACE_IDS), [totals[place_id] for place_id in PLACE_IDS], draw)
                base_destination, _probabilities_base = _select(list(PLACE_IDS), [bases[place_id] for place_id in PLACE_IDS], draw)
                reconsidered_n += 1
                if destination != base_destination:
                    dest_n += 1
            else:
                destination = origin
            decisions[agent_id] = destination
            score_values = [scores[place_id] for place_id in PLACE_IDS]
            social_values = [social_term[place_id] for place_id in PLACE_IDS]
            raw_values = [loo[place_id] for place_id in PLACE_IDS]
            sd_sum += _population_sd(score_values)
            range_sum += max(score_values) - min(score_values)
            social_sd_sum += _population_sd(social_values) / REFERENCE_CHOICE_TEMPERATURE
            mass_sum += sum(raw_values)
            if max(raw_values) == min(raw_values):
                zero_var_n += 1
            for place_id in PLACE_IDS:
                max_abs_raw = max(max_abs_raw, abs(loo[place_id]))
                max_abs_unclipped = max(max_abs_unclipped, abs(unclipped[place_id]))
                if scores[place_id] != unclipped[place_id]:
                    clip_changes += 1
            decisions_n += 1
        moves += sum(1 for agent_id in agents if decisions[agent_id] != current[agent_id])
        current = {agent_id: decisions[agent_id] for agent_id in agents}
        memory = _commit(memory, decisions)
        series.append(dict(current))
    diagnostics = measure_world(series, moves=moves, timesteps=timesteps)
    if timesteps >= LATE_START:
        late_counts = _presence(series, LATE_START, timesteps)
        late_share = _share(late_counts)
        late_hhi = _hhi(late_share)
        late_entropy = _entropy(late_share)
        late_dominant_share = max(late_share.values())
        late_dominant = _unique_leader(late_counts)
    else:
        late_hhi = None
        late_entropy = None
        late_dominant_share = None
        late_dominant = None
    record = {
        "cell_id": cell_id,
        "choice_seed": choice_seed_value,
        "social_weight": weight,
        "memory": spec["memory"],
        "transform": spec["transform"],
        "timesteps": timesteps,
        "trajectory_sha256": trajectory_hash(series),
        "late_hhi": late_hhi,
        "late_entropy": late_entropy,
        "late_dominant_share": late_dominant_share,
        "late_dominant": late_dominant,
        "movement_rate": diagnostics["movement_rate"],
        "alternation_rate": diagnostics["site_01_site_02_alternation_rate"],
        "consecutive_occupancy_tv": diagnostics["mean_total_variation"],
        "saturation_frequency": diagnostics["saturation_frequency"],
        "empty_frequency": diagnostics["empty_frequency"],
        "lag1_site_01": diagnostics["lag1"]["site_01"],
        "lag1_site_02": diagnostics["lag1"]["site_02"],
        "score_sd": sd_sum / decisions_n,
        "score_range": range_sum / decisions_n,
        "social_utility_sd_over_temperature": social_sd_sum / decisions_n,
        "memory_mass": mass_sum / decisions_n,
        "zero_variance_rate": zero_var_n / decisions_n,
        "ranking_change_rate": rank_n / decisions_n,
        "destination_change_rate": None if reconsidered_n == 0 else dest_n / reconsidered_n,
        "reconsidered_count": reconsidered_n,
        "decision_count": decisions_n,
        "clip_change_count": clip_changes,
        "max_abs_raw": max_abs_raw,
        "max_abs_unclipped": max_abs_unclipped,
        "nonfinite_count": 0,
        "llm_calls": 0,
    }
    if keep_series:
        record["series_packed"] = pack_series(series, agents)
    return record


def layer_a_metrics(
    atlas: Any,
    series: list[dict[str, str]],
    choice_seed_value: int,
    *,
    matrix: dict[str, dict[str, float]] | None = None,
    places: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if len(series) != TIMESTEPS + 1:
        raise RuntimeError("Layer A requires timesteps 0 through 500")
    agents = sorted(series[0])
    matrix = distance_matrix() if matrix is None else matrix
    places = place_index() if places is None else places
    memories: dict[str, SocialState | RollingMemory | DecayMemory] = {
        "cumulative": empty_social(agents),
        "rolling": RollingMemory(agents, 50),
        "exponential": DecayMemory(agents, 0.99),
    }
    buckets = {cell_id: _empty_layer_bucket() for cell_id in SOCIAL_CELL_IDS}
    for timestep in range(TIMESTEPS):
        origins = series[timestep]
        occupancy = occupancy_of(origins)
        cached = {agent_id: _base_utilities(atlas, matrix, places, agent_id, origins[agent_id], occupancy) for agent_id in agents}
        prepared = {
            (spec["memory"], spec["transform"], agent_id): score_memory(memories[spec["memory"]], agent_id, spec["transform"])
            for spec in CELLS
            if spec["cell_id"] != "S0"
            for agent_id in agents
        }
        for agent_id in agents:
            bases = cached[agent_id]
            base_probability = _probabilities(bases)
            does_reconsider = agent_reconsiders(choice_seed_value, timestep, agent_id, REFERENCE_RECONSIDERATION_RATE)
            draw = Random(choice_seed(choice_seed_value, timestep, agent_id)).random() if does_reconsider else None
            for spec in CELLS:
                if spec["cell_id"] == "S0":
                    continue
                unclipped, scores, loo = prepared[(spec["memory"], spec["transform"], agent_id)]
                totals = {place_id: bases[place_id] + SOCIAL_CELL_WEIGHT * scores[place_id] for place_id in PLACE_IDS}
                social_term = {place_id: SOCIAL_CELL_WEIGHT * scores[place_id] for place_id in PLACE_IDS}
                _require_finite(totals, "layer A total utility")
                probability = _probabilities(totals)
                bucket = buckets[spec["cell_id"]]
                bucket["q"] += 0.5 * sum(abs(probability[place_id] - base_probability[place_id]) for place_id in PLACE_IDS)
                score_values = [scores[place_id] for place_id in PLACE_IDS]
                raw_values = [loo[place_id] for place_id in PLACE_IDS]
                bucket["sd"] += _population_sd(score_values)
                bucket["range"] += max(score_values) - min(score_values)
                bucket["social_sd"] += _population_sd([social_term[place_id] for place_id in PLACE_IDS]) / REFERENCE_CHOICE_TEMPERATURE
                bucket["mass"] += sum(raw_values)
                bucket["zero"] += 1.0 if max(raw_values) == min(raw_values) else 0.0
                bucket["rank"] += 1.0 if _argmax(totals) != _argmax(bases) else 0.0
                bucket["clip"] += sum(1.0 for place_id in PLACE_IDS if scores[place_id] != unclipped[place_id])
                if does_reconsider:
                    bucket["reconsidered"] += 1.0
                    destination, _social_probability = _select(list(PLACE_IDS), [totals[place_id] for place_id in PLACE_IDS], draw)
                    base_destination, _base_probability = _select(list(PLACE_IDS), [bases[place_id] for place_id in PLACE_IDS], draw)
                    if destination != base_destination:
                        bucket["dest"] += 1.0
                bucket["decisions"] += 1.0
        destinations = series[timestep + 1]
        for name in memories:
            memories[name] = _commit(memories[name], destinations)
    rows = []
    for cell_id, bucket in buckets.items():
        decisions = bucket["decisions"]
        rows.append(
            {
                "cell_id": cell_id,
                "choice_seed": choice_seed_value,
                "q": bucket["q"] / decisions,
                "score_sd": bucket["sd"] / decisions,
                "score_range": bucket["range"] / decisions,
                "social_utility_sd_over_temperature": bucket["social_sd"] / decisions,
                "memory_mass": bucket["mass"] / decisions,
                "zero_variance_rate": bucket["zero"] / decisions,
                "ranking_change_rate": bucket["rank"] / decisions,
                "destination_change_rate": None if bucket["reconsidered"] == 0 else bucket["dest"] / bucket["reconsidered"],
                "reconsidered_count": bucket["reconsidered"],
                "decision_count": decisions,
                "clip_change_count": bucket["clip"],
                "llm_calls": 0,
            }
        )
    return rows


def _empty_layer_bucket() -> dict[str, float]:
    return {"q": 0.0, "sd": 0.0, "range": 0.0, "social_sd": 0.0, "mass": 0.0, "zero": 0.0, "rank": 0.0, "dest": 0.0, "reconsidered": 0.0, "decisions": 0.0, "clip": 0.0}


def load_saved_s0_series(path: Path, expected_sha256: str, seeds: tuple[int, ...]) -> dict[int, list[dict[str, str]]]:
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise RuntimeError("Experiment 004 world_states.jsonl hash does not match the frozen manifest")
    grouped: dict[int, dict[int, dict[str, str]]] = {seed: {} for seed in seeds}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if '"condition_id":"S0"' not in line:
                continue
            row = json.loads(line)
            if row.get("condition_id") != "S0":
                continue
            seed = int(row["choice_seed"])
            if seed not in grouped:
                continue
            timestep = int(row["timestep"])
            if timestep in grouped[seed]:
                raise RuntimeError(f"duplicate S0 timestep {timestep} for seed {seed}")
            locations = {str(agent_id): str(place_id) for agent_id, place_id in row["locations"].items()}
            grouped[seed][timestep] = locations
    loaded = {}
    expected_steps = set(range(TIMESTEPS + 1))
    for seed in seeds:
        observed = set(grouped[seed])
        if observed != expected_steps:
            raise RuntimeError(f"S0 seed {seed} does not contain exactly timesteps 0 through 500")
        series = [grouped[seed][timestep] for timestep in range(TIMESTEPS + 1)]
        agents = set(series[0])
        if len(agents) != 20:
            raise RuntimeError("S0 trajectory does not contain 20 agents")
        for step in series:
            if set(step) != agents or any(place_id not in PLACE_IDS for place_id in step.values()):
                raise RuntimeError("S0 trajectory agent or place set is incomplete")
        loaded[seed] = series
    return loaded


def seed_contrasts(values: dict[str, float]) -> dict[str, float]:
    delta = {memory: values[f"{memory}_ZRAW"] - values[f"{memory}_LOGMAX"] for memory in ("CUM", "RW50", "EXP99")}
    contrasts = {f"delta_t_{memory}": delta[memory] for memory in delta}
    for transform in ("LOGMAX", "ZRAW"):
        contrasts[f"RW50_minus_CUM_{transform}"] = values[f"RW50_{transform}"] - values[f"CUM_{transform}"]
        contrasts[f"EXP99_minus_CUM_{transform}"] = values[f"EXP99_{transform}"] - values[f"CUM_{transform}"]
    contrasts["interaction_RW50"] = delta["RW50"] - delta["CUM"]
    contrasts["interaction_EXP99"] = delta["EXP99"] - delta["CUM"]
    return contrasts


def bootstrap_summaries(
    seed_ids: tuple[int, ...],
    table: dict[int, dict[str, float]],
    *,
    rng_seed: int,
    draws: int = BOOTSTRAP_DRAWS,
) -> dict[str, Any]:
    ordered = list(seed_ids)
    indexes = paired_seed_indexes(len(ordered), draws, rng_seed)
    names = sorted({name for row in table.values() for name in row})
    point = {name: statistics.fmean(table[seed][name] for seed in ordered) for name in names}
    replicates = {name: [] for name in names}
    for row in indexes:
        for name in names:
            replicates[name].append(statistics.fmean(table[ordered[index]][name] for index in row))
    low_index, high_index = percentile_indexes(draws)
    summaries = {}
    for name in names:
        samples = [table[seed][name] for seed in ordered]
        ordered_replicates = sorted(replicates[name])
        summaries[name] = {
            "mean": point[name],
            "median": statistics.median(samples),
            "stdev": statistics.stdev(samples) if len(samples) > 1 else 0.0,
            "minimum": min(samples),
            "maximum": max(samples),
            "q25": statistics.quantiles(samples, n=4, method="inclusive")[0],
            "q75": statistics.quantiles(samples, n=4, method="inclusive")[2],
            "ci_low": ordered_replicates[low_index],
            "ci_high": ordered_replicates[high_index],
            "n": len(samples),
        }
    return {
        "draws": draws,
        "analysis_rng_seed": rng_seed,
        "percentile_indexes": [low_index, high_index],
        "interval": "pointwise",
        "index_meaning": "position in the fixed seed list, not the choice-seed number",
        "seed_list_head": ordered[0],
        "summaries": summaries,
    }


def audit_choice_seeds(data_root: Path | None = None) -> dict[str, Any]:
    root = data_root or (PROJECT_ROOT / "data")
    proposed = set(COHORT_B_SEEDS)
    sources = []
    overlaps = []
    unchecked = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in {".json", ".jsonl"}:
            continue
        if "world_v0_17" in path.parts:
            continue
        size = path.stat().st_size
        if path.suffix == ".jsonl" and size > TRACE_SCAN_LIMIT:
            unchecked.append({"path": _display(path, root), "bytes": size, "reason": "large trace; choice seeds are taken from configs and manifests"})
            continue
        if path.suffix == ".json" and size > 20_000_000:
            unchecked.append({"path": _display(path, root), "bytes": size, "reason": "json above the index scan limit"})
            overlaps.append(_display(path, root))
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
                    if isinstance(row, dict) and isinstance(row.get("choice_seed"), int):
                        choice_seeds.add(int(row["choice_seed"]))
        relevant = choice_seeds or trajectory_seeds
        if not relevant and path.suffix == ".jsonl":
            continue
        if not relevant:
            continue
        hit = sorted((choice_seeds | trajectory_seeds) & proposed)
        record = {
            "path": _display(path, root),
            "choice_seed_min": min(choice_seeds) if choice_seeds else None,
            "choice_seed_max": max(choice_seeds) if choice_seeds else None,
            "trajectory_seed_min": min(trajectory_seeds) if trajectory_seeds else None,
            "trajectory_seed_max": max(trajectory_seeds) if trajectory_seeds else None,
        }
        if path.parent.name == "world_v0_6":
            record["trajectory_note"] = "world_v0_6 keys are architecture:social_weight:choice_seed"
        if hit:
            record["overlap"] = hit
            overlaps.append(record["path"])
        sources.append(record)
    return {
        "experiment_id": "experiment_017_memory_by_transformation",
        "proposed_choice_seeds": "200-699",
        "proposed_count": len(proposed),
        "passed": not overlaps,
        "overlap_sources": overlaps,
        "sources": sources,
        "unchecked_large_traces": unchecked,
        "scanned_documents": False,
        "includes_world_v0_17_outputs": False,
        "judgment": (
            "Seeds 200-699 do not appear as choice seeds in the checked configs, manifests, or small indexes."
            if not overlaps
            else "The proposed seeds collide with an existing choice-seed index, or a required index could not be read."
        ),
        "llm_calls": 0,
    }


def historical_hashes() -> dict[str, dict[int, str]]:
    manifest_004 = json.loads((WORLD_V04_DIR / "run_manifest.json").read_text(encoding="utf-8"))
    manifest_005 = json.loads((WORLD_V05_DIR / "run_manifest.json").read_text(encoding="utf-8"))
    hashes_004 = manifest_004["trajectory_sha256"]
    hashes_005 = manifest_005["trajectory_sha256"]
    expected = {
        "CUM_LOGMAX": {},
        "RW50_ZRAW": {},
        "EXP99_ZRAW": {},
        "S0": {},
        "S150": {},
    }
    for seed in COHORT_A_SEEDS:
        expected["CUM_LOGMAX"][seed] = hashes_005[f"CUM:{seed}"]
        expected["S150"][seed] = hashes_004[f"S150:{seed}"]
        expected["RW50_ZRAW"][seed] = hashes_005[f"RW50:{seed}"]
        expected["EXP99_ZRAW"][seed] = hashes_005[f"EXP99:{seed}"]
        expected["S0"][seed] = hashes_004[f"S0:{seed}"]
        if expected["CUM_LOGMAX"][seed] != expected["S150"][seed]:
            raise RuntimeError(f"stored CUM and S150 hashes differ at seed {seed}")
    return expected


def lineage_failures(records: list[dict[str, Any]], expected: dict[str, dict[int, str]]) -> list[dict[str, Any]]:
    failures = []
    for record in records:
        cell_id = record["cell_id"]
        seed = int(record["choice_seed"])
        if cell_id not in LINEAGE_CELL_IDS:
            continue
        actual = record["trajectory_sha256"]
        target = expected[cell_id][seed]
        if actual != target:
            failures.append({"cell_id": cell_id, "choice_seed": seed, "actual": actual, "expected": target})
    return failures


def build_config(audit_sha256: str, world_states_sha256: str) -> dict[str, Any]:
    expected = historical_hashes()
    payload = {
        "world_id": "world_v0_17",
        "experiment_id": "experiment_017_memory_by_transformation",
        "status_at_config_freeze": "CONFIG_FROZEN_BEFORE_RESULTS",
        "design_record": "docs/experiment_017_design/protocol_draft.json",
        "design_record_remains": "DESIGN_ONLY / NOT_RUN",
        "scale_change_from_design": "Cohort B adds held-out choice seeds 200-699. Cohort A remains seeds 0-49. No formula, threshold, window, weight, or contrast rule changes.",
        "utility_source": "utility_atlas_v1",
        "utility_atlas_sha256": EXPECTED_ATLAS_SHA256,
        "choice_temperature": REFERENCE_CHOICE_TEMPERATURE,
        "reconsideration_rate": REFERENCE_RECONSIDERATION_RATE,
        "familiarity_weight": FAMILIARITY_WEIGHT,
        "social_weight": SOCIAL_CELL_WEIGHT,
        "s0_social_weight": S0_WEIGHT,
        "timesteps": TIMESTEPS,
        "late_window": "101-500 cumulative presence, then share, then HHI",
        "agents": 20,
        "places": list(PLACE_IDS),
        "llm_calls": 0,
        "cohorts": {
            "cohort_a": {"seeds": list(COHORT_A_SEEDS), "n": 50, "analysis_rng_seed": COHORT_A_ANALYSIS_SEED, "role": "historical_bridge"},
            "cohort_b": {"seeds": list(COHORT_B_SEEDS), "n": 500, "analysis_rng_seed": COHORT_B_ANALYSIS_SEED, "role": "held_out_stochastic_replication"},
        },
        "cells": [dict(spec) for spec in CELLS],
        "primary_outcomes": {"layer_a": "Q", "layer_b": "late_visit_share_hhi"},
        "contrasts": ["delta_t", "RW50_minus_CUM", "EXP99_minus_CUM", "interaction_RW50", "interaction_EXP99"],
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "percentile_indexes": list(percentile_indexes(BOOTSTRAP_DRAWS)),
            "interval": "pointwise",
            "unit": "choice_seed",
            "sampling": "with_replacement_row_of_n",
        },
        "lineage_cells": list(LINEAGE_CELL_IDS),
        "lineage_seeds": list(COHORT_A_SEEDS),
        "historical_trajectory_sha256": {cell_id: {str(seed): value for seed, value in rows.items()} for cell_id, rows in expected.items()},
        "historical_subset": {"subset_of": "lineage", "seeds": [0, 49], "count": 8, "additional_count": 0},
        "replay": {"cohort_a": [{"cell_id": cell_id, "choice_seed": seed} for cell_id, seed in REPLAY_A], "cohort_b": [{"cell_id": cell_id, "choice_seed": seed} for cell_id, seed in REPLAY_B]},
        "counts": {
            "logical_conditions": 3850,
            "historical_conditions": 200,
            "new_conditions": 3650,
            "extra_replays": 20,
            "formal_worlds_before_run": 0,
        },
        "world_v0_4_states_sha256": world_states_sha256,
        "seed_audit_sha256": audit_sha256,
        "source_sha256": {
            "world/experiment_004.py": file_sha256(PROJECT_ROOT / "world" / "experiment_004.py"),
            "world/experiment_005.py": file_sha256(PROJECT_ROOT / "world" / "experiment_005.py"),
            "world/experiment_017.py": file_sha256(Path(__file__).resolve()),
        },
        "python": sys.version,
        "platform": platform.platform(),
        "social_weight_module_default": SOCIAL_WEIGHT,
        "budget_start": BUDGET_START.isoformat(),
        "budget_seconds": BUDGET_SECONDS,
        "reserve_seconds": RESERVE_SECONDS,
        "incomplete_rule": "Stop only at a batch boundary, keep completed unit files, mark INCOMPLETE, and do not infer from a partial cohort.",
    }
    payload["config_sha256"] = canonical_sha256(payload)
    return payload


def run_experiment(output_dir: Path | None = None, *, resume: bool = False, workers: int | None = None) -> Path:
    destination = output_dir or OUTPUT_DIR
    _refuse_frozen(destination)
    started = time.perf_counter()
    deadline = BUDGET_START.timestamp() + BUDGET_SECONDS - RESERVE_SECONDS
    if not resume:
        refuse_nonempty_output(destination)
        audit = audit_choice_seeds()
        if not audit["passed"]:
            raise RuntimeError("seed audit failed; formal execution stopped without choosing replacement seeds")
        destination.mkdir(parents=True)
        write_json(destination / "seed_audit.json", audit)
        manifest_004 = json.loads((WORLD_V04_DIR / "run_manifest.json").read_text(encoding="utf-8"))
        config = build_config(file_sha256(destination / "seed_audit.json"), manifest_004["files"]["world_states.jsonl"])
        write_json(destination / "world_v0_17_config.json", config)
    else:
        config = _load_resume_config(destination)
        audit = json.loads((destination / "seed_audit.json").read_text(encoding="utf-8"))
    _require_atlas()
    if datetime.now().timestamp() > deadline:
        _write_status(destination, "INCOMPLETE", "budget exhausted before simulation", started)
        return destination
    expected = {cell_id: {int(seed): value for seed, value in rows.items()} for cell_id, rows in config["historical_trajectory_sha256"].items()}
    s0_series = load_saved_s0_series(WORLD_V04_DIR / "world_states.jsonl", config["world_v0_4_states_sha256"], COHORT_A_SEEDS)
    for seed, series in s0_series.items():
        if trajectory_hash(series) != expected["S0"][seed]:
            raise RuntimeError(f"loaded S0 series hash differs from the manifest at seed {seed}")
    worker_count = workers if workers is not None else min(12, max(1, (os.cpu_count() or 2) - 1))
    lineage_tasks = [("cohort_a", cell_id, seed) for cell_id in LINEAGE_CELL_IDS for seed in COHORT_A_SEEDS]
    _run_missing(destination, lineage_tasks, worker_count, deadline, started)
    lineage_records = _read_units(destination, lineage_tasks)
    if len(lineage_records) != len(lineage_tasks):
        _write_status(destination, "INCOMPLETE", "lineage batch stopped before 200 trajectories", started)
        return destination
    failures = lineage_failures(lineage_records, expected)
    subset = [item for item in failures if item["choice_seed"] in {0, 49}]
    lineage_audit = {
        "passed": not failures,
        "compared": len(lineage_records),
        "failures": failures,
        "subset_seeds": [0, 49],
        "subset_failure_count": len(subset),
        "subset_additional_executions": 0,
        "old_manifest_comparison_used_as_substitute": False,
        "llm_calls": 0,
    }
    write_json(destination / "lineage_audit.json", lineage_audit)
    if failures:
        _write_status(destination, "INCOMPLETE", "historical lineage hash mismatch", started)
        return destination
    remaining = [("cohort_a", cell_id, seed) for cell_id in NEW_CROSS_CELL_IDS for seed in COHORT_A_SEEDS]
    remaining.extend(("cohort_b", cell_id, seed) for cell_id in tuple(CELL_BY_ID) for seed in COHORT_B_SEEDS)
    completed = _run_missing(destination, remaining, worker_count, deadline, started)
    if completed is False:
        _write_status(destination, "INCOMPLETE", "budget exhausted during closed-loop worlds", started)
        return destination
    if not _units_complete(destination, remaining):
        _write_status(destination, "INCOMPLETE", "closed-loop units incomplete", started)
        return destination
    _write_layer_metrics(destination, s0_series, worker_count)
    _write_analysis(destination)
    replay_ok = _run_replays(destination, worker_count)
    _write_diagnostics(destination)
    _write_figures(destination)
    _write_status(destination, "COMPLETE" if replay_ok else "INCOMPLETE", "verified" if replay_ok else "replay mismatch", started)
    write_report(destination)
    _write_source_manifest(destination)
    _write_manifest(destination, replay_ok)
    _update_indexes()
    return destination


def _run_missing(destination: Path, tasks: list[tuple[str, str, int]], workers: int, deadline: float, started: float) -> bool:
    pending = [task for task in tasks if not _unit_path(destination, *task).exists()]
    batch_size = 24
    for offset in range(0, len(pending), batch_size):
        if datetime.now().timestamp() > deadline:
            return False
        batch = pending[offset : offset + batch_size]
        for record, cohort, cell_id, seed in _execute_batch(batch, workers):
            path = _unit_path(destination, cohort, cell_id, seed)
            path.parent.mkdir(parents=True, exist_ok=True)
            write_json(path, record)
            _append_index(destination, cohort, cell_id, seed, file_sha256(path), record["trajectory_sha256"])
        done = offset + len(batch)
        elapsed = time.perf_counter() - started
        print(f"closed-loop {done}/{len(pending)} batch elapsed={elapsed:.1f}s lineage_gate_pending={not (destination / 'lineage_audit.json').exists()}", flush=True)
    return True


def _execute_batch(batch: list[tuple[str, str, int]], workers: int) -> list[tuple[dict[str, Any], str, str, int]]:
    if workers <= 1 or len(batch) == 1:
        output = []
        for cohort, cell_id, seed in batch:
            output.append((_run_task((cohort, cell_id, seed)), cohort, cell_id, seed))
        return output
    pool = _shared_pool(workers)
    futures = {pool.submit(_run_task, task): task for task in batch}
    output = []
    for future in as_completed(futures):
        cohort, cell_id, seed = futures[future]
        output.append((future.result(timeout=180), cohort, cell_id, seed))
    return output


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


def _init_worker() -> None:
    _WORKER["atlas"] = load_frozen_atlas()
    _WORKER["locations"] = load_initial_locations()
    _WORKER["matrix"] = distance_matrix()
    _WORKER["places"] = place_index()


def _run_task(task: tuple[str, str, int]) -> dict[str, Any]:
    if "atlas" not in _WORKER:
        _init_worker()
    cohort, cell_id, seed = task
    record = simulate_cell(
        _WORKER["atlas"],
        _WORKER["locations"],
        cell_id,
        seed,
        matrix=_WORKER["matrix"],
        places=_WORKER["places"],
        keep_series=cell_id == "S0" and cohort == "cohort_b",
    )
    record["cohort"] = cohort
    record["llm_calls"] = 0
    return record


def _write_layer_metrics(destination: Path, cohort_a_s0: dict[int, list[dict[str, str]]], workers: int) -> None:
    for cohort, seeds, series_source in (
        ("cohort_a", COHORT_A_SEEDS, cohort_a_s0),
        ("cohort_b", COHORT_B_SEEDS, _load_cohort_b_s0(destination)),
    ):
        layer_a_path = destination / cohort / "layer_a_seed_metrics.jsonl"
        layer_b_path = destination / cohort / "layer_b_seed_metrics.jsonl"
        layer_a_path.parent.mkdir(parents=True, exist_ok=True)
        if _line_count(layer_a_path) != len(seeds) * len(SOCIAL_CELL_IDS):
            _write_layer_a(layer_a_path, cohort, seeds, series_source, workers)
        if _line_count(layer_b_path) != len(seeds) * len(CELL_BY_ID):
            with layer_b_path.open("w", encoding="utf-8", newline="\n") as handle:
                for cell_id in tuple(CELL_BY_ID):
                    for seed in seeds:
                        record = json.loads(_unit_path(destination, cohort, cell_id, seed).read_text(encoding="utf-8"))
                        record.pop("series_packed", None)
                        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
        print(f"{cohort} layer metrics written", flush=True)


def _write_layer_a(path: Path, cohort: str, seeds: tuple[int, ...], series_source: dict[int, list[dict[str, str]]], workers: int) -> None:
    tasks = [(cohort, seed, series_source[seed]) for seed in seeds]
    if workers <= 1:
        grouped = {seed: _layer_a_task((cohort, seed, series_source[seed])) for seed in seeds}
    else:
        grouped = {}
        pool = _shared_pool(workers)
        futures = {pool.submit(_layer_a_task, task): task[1] for task in tasks}
        for future in as_completed(futures):
            grouped[futures[future]] = future.result(timeout=180)
    temporary = path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for seed in seeds:
            for row in grouped[seed]:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
    temporary.replace(path)
    print(f"{cohort} layer A wrote {len(seeds)} seeds", flush=True)


def _layer_a_task(task: tuple[str, int, list[dict[str, str]]]) -> list[dict[str, Any]]:
    if "atlas" not in _WORKER:
        _init_worker()
    cohort, seed, series = task
    rows = layer_a_metrics(_WORKER["atlas"], series, seed, matrix=_WORKER["matrix"], places=_WORKER["places"])
    for row in rows:
        row["cohort"] = cohort
    return rows


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _write_analysis(destination: Path) -> None:
    analysis = {}
    for cohort, seeds, rng_seed in (
        ("cohort_a", COHORT_A_SEEDS, COHORT_A_ANALYSIS_SEED),
        ("cohort_b", COHORT_B_SEEDS, COHORT_B_ANALYSIS_SEED),
    ):
        analysis[cohort] = {}
        for layer, metric in (("layer_a", "q"), ("layer_b", "late_hhi")):
            rows = _read_jsonl(destination / cohort / f"{layer}_seed_metrics.jsonl")
            table: dict[int, dict[str, float]] = {seed: {} for seed in seeds}
            for row in rows:
                if row["cell_id"] in SOCIAL_CELL_IDS or (layer == "layer_b" and row["cell_id"] == "S0"):
                    table[int(row["choice_seed"])][row["cell_id"]] = float(row[metric])
            contrast_table = {seed: seed_contrasts({cell_id: table[seed][cell_id] for cell_id in SOCIAL_CELL_IDS}) for seed in seeds}
            combined = {seed: {**table[seed], **contrast_table[seed]} for seed in seeds}
            analysis[cohort][layer] = bootstrap_summaries(seeds, combined, rng_seed=rng_seed)
    write_json(destination / "uncertainty_analysis.json", analysis)


def _run_replays(destination: Path, workers: int) -> bool:
    tasks = [("cohort_a", cell_id, seed) for cell_id, seed in REPLAY_A]
    tasks.extend(("cohort_b", cell_id, seed) for cell_id, seed in REPLAY_B)
    results = []
    passed = True
    for record, cohort, cell_id, seed in _execute_batch(tasks, workers):
        saved = json.loads(_unit_path(destination, cohort, cell_id, seed).read_text(encoding="utf-8"))
        match = record["trajectory_sha256"] == saved["trajectory_sha256"]
        passed = passed and match
        results.append(
            {
                "cohort": cohort,
                "cell_id": cell_id,
                "choice_seed": seed,
                "match": match,
                "replay_trajectory_sha256": record["trajectory_sha256"],
                "saved_trajectory_sha256": saved["trajectory_sha256"],
                "saved_file_rewritten": False,
                "llm_calls": 0,
            }
        )
    saved_after = file_sha256(_unit_path(destination, tasks[0][0], tasks[0][1], tasks[0][2]))
    write_json(
        destination / "replay_audit.json",
        {"passed": passed, "count": len(results), "results": results, "first_saved_unit_sha256": saved_after, "llm_calls": 0},
    )
    return passed


def _write_diagnostics(destination: Path) -> None:
    numeric = []
    safety = []
    for cohort in ("cohort_a", "cohort_b"):
        for row in _read_jsonl(destination / cohort / "layer_b_seed_metrics.jsonl"):
            numeric.append(
                {
                    "cohort": cohort,
                    "cell_id": row["cell_id"],
                    "choice_seed": row["choice_seed"],
                    "clip_change_count": row["clip_change_count"],
                    "max_abs_raw": row["max_abs_raw"],
                    "max_abs_unclipped": row["max_abs_unclipped"],
                    "nonfinite_count": row["nonfinite_count"],
                }
            )
            safety.append(
                {
                    "cohort": cohort,
                    "cell_id": row["cell_id"],
                    "choice_seed": row["choice_seed"],
                    "saturation_frequency": row["saturation_frequency"],
                    "empty_frequency": row["empty_frequency"],
                    "alternation_rate": row["alternation_rate"],
                    "consecutive_occupancy_tv": row["consecutive_occupancy_tv"],
                    "lag1_site_01": row["lag1_site_01"],
                    "lag1_site_02": row["lag1_site_02"],
                }
            )
    write_json(destination / "numerical_diagnostics.json", {"rows_are_seed_level": True, "rows": numeric, "llm_calls": 0})
    write_json(destination / "safety_diagnostics.json", {"rows_are_seed_level": True, "note": "Diagnostic values only. They do not assign Experiment 005 architecture classes.", "rows": safety, "llm_calls": 0})


def _write_figures(destination: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        write_json(destination / "figure_status.json", {"written": False, "reason": str(exc), "llm_calls": 0})
        return
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    figure_dir = destination / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    analysis = json.loads((destination / "uncertainty_analysis.json").read_text(encoding="utf-8"))
    _box_figure(plt, destination, figure_dir / "layer_a_q.png", "layer_a", "q", "Layer A Q")
    _box_figure(plt, destination, figure_dir / "layer_b_hhi.png", "layer_b", "late_hhi", "Layer B late HHI")
    _contrast_figure(plt, analysis, figure_dir / "contrasts.png")
    write_json(destination / "figure_status.json", {"written": True, "font": "Microsoft JhengHei", "llm_calls": 0})


def _box_figure(plt: Any, destination: Path, path: Path, layer: str, metric: str, title: str) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for axis, cohort in zip(axes, ("cohort_a", "cohort_b"), strict=True):
        rows = [row for row in _read_jsonl(destination / cohort / f"{layer}_seed_metrics.jsonl") if row["cell_id"] in SOCIAL_CELL_IDS]
        groups = [[float(row[metric]) for row in rows if row["cell_id"] == cell_id] for cell_id in SOCIAL_CELL_IDS]
        axis.boxplot(groups, tick_labels=list(SOCIAL_CELL_IDS), showfliers=False)
        axis.set_title(cohort)
        axis.tick_params(axis="x", rotation=45)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _contrast_figure(plt: Any, analysis: dict[str, Any], path: Path) -> None:
    names = ["delta_t_CUM", "delta_t_RW50", "delta_t_EXP99", "interaction_RW50", "interaction_EXP99"]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = (("cohort_a", "layer_a"), ("cohort_b", "layer_a"), ("cohort_a", "layer_b"), ("cohort_b", "layer_b"))
    for axis, (cohort, layer) in zip(axes.ravel(), panels, strict=True):
        summaries = analysis[cohort][layer]["summaries"]
        means = [summaries[name]["mean"] for name in names]
        low = [summaries[name]["mean"] - summaries[name]["ci_low"] for name in names]
        high = [summaries[name]["ci_high"] - summaries[name]["mean"] for name in names]
        axis.errorbar(list(range(len(names))), means, yerr=[low, high], fmt="o")
        axis.axhline(0.0, color="gray", linewidth=0.8)
        axis.set_xticks(list(range(len(names))), names, rotation=45, ha="right")
        axis.set_title(f"{cohort} {layer}")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def write_report(destination: Path) -> None:
    analysis = json.loads((destination / "uncertainty_analysis.json").read_text(encoding="utf-8"))
    lineage = json.loads((destination / "lineage_audit.json").read_text(encoding="utf-8"))
    replay = json.loads((destination / "replay_audit.json").read_text(encoding="utf-8"))
    status = json.loads((destination / "completion_status.json").read_text(encoding="utf-8")) if (destination / "completion_status.json").exists() else {"status": "IN_PROGRESS"}
    lines = [
        "# Experiment 017 報告",
        "",
        f"狀態：{status.get('status', 'IN_PROGRESS')}。這是 memory representation 與 score transformation 的交叉模擬。Cohort A 是 seeds 0–49 的歷史橋接。Cohort B 是 seeds 200–699 的 held-out stochastic replication。兩組分開估計，沒有合併成單一主要平均。",
        "",
        "相對於設計稿，唯一的研究規模變更是加入 Cohort B。公式、權重 150、500 timesteps、late window t=101–500、contrasts 與 Cohort A 的 bootstrap seed 20260928 沒有因結果而改動。設計稿 `protocol_draft.json` 仍是 DESIGN_ONLY / NOT_RUN。",
        "",
        f"歷史 lineage：{lineage['compared']} 條，passed={lineage['passed']}。指定子集 seeds 0 與 49 的失敗數是 {lineage['subset_failure_count']}，沒有另計執行。Replay passed={replay['passed']}，共 {replay['count']} 條。LLM calls = 0。",
        "",
        "Q 是共同 S0 歷史上的條件選擇機率差異。late HHI 是各格子閉環、t=101–500 累積 visit share 的 HHI。兩者不相減，也不使用 0.025 persistence 門檻。HHI 差異不是路徑依賴證明。",
        "",
        "## Cohort A Layer A Q",
        "",
        _summary_table(analysis["cohort_a"]["layer_a"]["summaries"], SOCIAL_CELL_IDS),
        "",
        "## Cohort B Layer A Q",
        "",
        _summary_table(analysis["cohort_b"]["layer_a"]["summaries"], SOCIAL_CELL_IDS),
        "",
        "## Cohort A Layer B late HHI",
        "",
        _summary_table(analysis["cohort_a"]["layer_b"]["summaries"], SOCIAL_CELL_IDS + ("S0",)),
        "",
        "## Cohort B Layer B late HHI",
        "",
        _summary_table(analysis["cohort_b"]["layer_b"]["summaries"], SOCIAL_CELL_IDS + ("S0",)),
        "",
        "## Contrasts",
        "",
        "區間是 pointwise percentile interval，索引 249 與 9749。區間包含 0 或兩個區間重疊，都不構成等價證明。這裡沒有整體 replication 成功標籤。",
        "",
        _contrast_section(analysis),
        "",
        "## 機制範圍",
        "",
        "這個設計能分開的是：在 CUM、RW50、EXP99 與 LOGMAX、ZRAW 的交叉下，轉換對照、記憶對照與 interaction。它不能把 LOGMAX 對 ZRAW 收成純尺度效果，也不能把記憶對照收成純時間尺度。它不解釋 Experiment 016 的 H2000 onset，也不提供外部或真人行為驗證。",
        "",
        "ZRAW 的 score SD 接近 1、五場所 |z| 不超過 2，以及 near-boundary 計數，都是公式性質或舊診斷，不是本輪的行為發現。clip 改寫次數見 numerical diagnostics；精確算術下這個 clip 不應改寫 z。",
        "",
    ]
    text = "\n".join(lines)
    REPORT_PATH.write_text(text, encoding="utf-8", newline="\n")
    (destination / "EXPERIMENT_017_REPORT.md").write_text(text, encoding="utf-8", newline="\n")


def _summary_table(summaries: dict[str, Any], names: tuple[str, ...]) -> str:
    header = "| cell | mean | median | sd | q25 | q75 | CI |"
    rows = [header, "| --- | --- | --- | --- | --- | --- | --- |"]
    for name in names:
        item = summaries[name]
        rows.append(
            f"| {name} | {item['mean']:.6f} | {item['median']:.6f} | {item['stdev']:.6f} | {item['q25']:.6f} | {item['q75']:.6f} | [{item['ci_low']:.6f}, {item['ci_high']:.6f}] |"
        )
    return "\n".join(rows)


def _contrast_section(analysis: dict[str, Any]) -> str:
    names = [
        "delta_t_CUM",
        "delta_t_RW50",
        "delta_t_EXP99",
        "RW50_minus_CUM_LOGMAX",
        "RW50_minus_CUM_ZRAW",
        "EXP99_minus_CUM_LOGMAX",
        "EXP99_minus_CUM_ZRAW",
        "interaction_RW50",
        "interaction_EXP99",
    ]
    lines = ["| cohort | layer | contrast | mean | CI | median |", "| --- | --- | --- | --- | --- | --- |"]
    for cohort in ("cohort_a", "cohort_b"):
        for layer in ("layer_a", "layer_b"):
            summaries = analysis[cohort][layer]["summaries"]
            for name in names:
                item = summaries[name]
                lines.append(
                    f"| {cohort} | {layer} | {name} | {item['mean']:.6f} | [{item['ci_low']:.6f}, {item['ci_high']:.6f}] | {item['median']:.6f} |"
                )
    return "\n".join(lines)


def _write_source_manifest(destination: Path) -> None:
    files = [
        PROJECT_ROOT / "world" / "experiment_017.py",
        PROJECT_ROOT / "world" / "experiment_004.py",
        PROJECT_ROOT / "world" / "experiment_005.py",
        WORLD_V04_DIR / "run_manifest.json",
        WORLD_V05_DIR / "run_manifest.json",
        WORLD_V04_DIR / "world_v0_4_config.json",
        WORLD_V05_DIR / "world_v0_5_config.json",
        destination / "world_v0_17_config.json",
        destination / "seed_audit.json",
    ]
    write_json(
        destination / "source_manifest.json",
        {
            "record_type": "experiment_017_source_manifest",
            "not_an_experiment_run_manifest": True,
            "files": [{"path": path.relative_to(PROJECT_ROOT).as_posix(), "sha256": file_sha256(path)} for path in files],
            "llm_calls": 0,
        },
    )


def _write_manifest(destination: Path, replay_ok: bool) -> None:
    files = {}
    for path in sorted(destination.rglob("*")):
        if path.is_file() and path.name != "run_manifest.json":
            files[path.relative_to(destination).as_posix()] = file_sha256(path)
    write_json(
        destination / "run_manifest.json",
        {
            "experiment_id": "experiment_017_memory_by_transformation",
            "llm_calls": 0,
            "logical_conditions": 3850,
            "historical_lineage_executions": 200,
            "new_conditions": 3650,
            "extra_replays": 20,
            "replay_passed": replay_ok,
            "config_sha256": json.loads((destination / "world_v0_17_config.json").read_text(encoding="utf-8"))["config_sha256"],
            "files": files,
        },
    )


def _write_status(destination: Path, status: str, reason: str, started: float) -> None:
    _shutdown_pool()
    write_json(
        destination / "completion_status.json",
        {
            "status": status,
            "reason": reason,
            "elapsed_seconds": time.perf_counter() - started,
            "logical_conditions": 3850,
            "llm_calls": 0,
        },
    )


def _update_indexes() -> None:
    readme = PROJECT_ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    replacement = "- 017 已完成正式執行。見 [`docs/EXPERIMENT_017_REPORT.md`](docs/EXPERIMENT_017_REPORT.md)。設計稿仍是當時的 DESIGN_ONLY / NOT_RUN 紀錄"
    if "017 設計稿完成" in text:
        text = text.replace(
            "- 017 設計稿完成，尚未實作或執行。見 [`docs/EXPERIMENT_017_DESIGN.md`](docs/EXPERIMENT_017_DESIGN.md)",
            replacement,
        )
        readme.write_text(text, encoding="utf-8", newline="\n")
    experiments = PROJECT_ROOT / "docs" / "EXPERIMENTS.md"
    index = experiments.read_text(encoding="utf-8")
    if "尚未實作或執行" in index:
        index = index.replace(
            "017 設計稿完成，尚未實作或執行。設計見 [`EXPERIMENT_017_DESIGN.md`](EXPERIMENT_017_DESIGN.md)。尚未建立模擬程式或 `world_v0_17` 輸出。",
            "017 已完成正式執行。設計稿見 [`EXPERIMENT_017_DESIGN.md`](EXPERIMENT_017_DESIGN.md)，仍保持撰寫當時的 DESIGN_ONLY / NOT_RUN。結果見 [`EXPERIMENT_017_REPORT.md`](EXPERIMENT_017_REPORT.md)。",
        )
        experiments.write_text(index, encoding="utf-8", newline="\n")


def _load_resume_config(destination: Path) -> dict[str, Any]:
    config_path = destination / "world_v0_17_config.json"
    audit_path = destination / "seed_audit.json"
    if not config_path.exists() or not audit_path.exists():
        raise RuntimeError("resume requires the frozen config and the pre-run seed audit")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if file_sha256(Path(__file__).resolve()) != config["source_sha256"]["world/experiment_017.py"]:
        raise RuntimeError("resume stopped because experiment_017.py no longer matches the frozen config")
    if file_sha256(audit_path) != config["seed_audit_sha256"]:
        raise RuntimeError("resume stopped because the pre-run seed audit hash changed")
    _verify_unit_index(destination)
    return config


def _verify_unit_index(destination: Path) -> None:
    for cohort in ("cohort_a", "cohort_b"):
        index_path = destination / cohort / "unit_index.jsonl"
        if not index_path.exists():
            continue
        seen = set()
        with index_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (row["cohort"], row["cell_id"], int(row["choice_seed"]))
                if key in seen:
                    raise RuntimeError("resume found a duplicate unit index row")
                seen.add(key)
                path = _unit_path(destination, row["cohort"], row["cell_id"], int(row["choice_seed"]))
                if file_sha256(path) != row["file_sha256"]:
                    raise RuntimeError("resume stopped because a completed unit hash changed")
                saved = json.loads(path.read_text(encoding="utf-8"))
                if saved["trajectory_sha256"] != row["trajectory_sha256"]:
                    raise RuntimeError("resume stopped because a trajectory hash does not match its index")


def _load_cohort_b_s0(destination: Path) -> dict[int, list[dict[str, str]]]:
    loaded = {}
    for seed in COHORT_B_SEEDS:
        record = json.loads(_unit_path(destination, "cohort_b", "S0", seed).read_text(encoding="utf-8"))
        series = unpack_series(record["series_packed"])
        if trajectory_hash(series) != record["trajectory_sha256"]:
            raise RuntimeError(f"cohort B S0 packed series does not match its saved hash at seed {seed}")
        loaded[seed] = series
    return loaded


def _read_units(destination: Path, tasks: list[tuple[str, str, int]]) -> list[dict[str, Any]]:
    records = []
    for cohort, cell_id, seed in tasks:
        path = _unit_path(destination, cohort, cell_id, seed)
        if path.exists():
            records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def _units_complete(destination: Path, tasks: list[tuple[str, str, int]]) -> bool:
    return all(_unit_path(destination, cohort, cell_id, seed).exists() for cohort, cell_id, seed in tasks)


def _unit_path(destination: Path, cohort: str, cell_id: str, seed: int) -> Path:
    return destination / cohort / "units" / f"{cell_id}__{seed}.json"


def _append_index(destination: Path, cohort: str, cell_id: str, seed: int, digest: str, trajectory: str) -> None:
    path = destination / cohort / "unit_index.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        {"cohort": cohort, "cell_id": cell_id, "choice_seed": seed, "file_sha256": digest, "trajectory_sha256": trajectory},
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _require_atlas() -> None:
    manifest = json.loads((ATLAS_DIR / "atlas_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("sha256") != EXPECTED_ATLAS_SHA256:
        raise RuntimeError("utility atlas hash is not the frozen experiment atlas")


def _refuse_frozen(destination: Path) -> None:
    frozen = [WORLD_V04_DIR, WORLD_V05_DIR, ATLAS_DIR]
    if destination.resolve() in {path.resolve() for path in frozen}:
        raise RuntimeError("experiment 017 must not write into a frozen source directory")


def _display(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return path.resolve().relative_to(root.resolve()).as_posix()


def _collect_choice_seeds(node: Any, found: set[int]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "choice_seeds" and isinstance(value, list) and all(isinstance(item, int) for item in value):
                found.update(int(item) for item in value)
            elif key == "choice_seed" and isinstance(value, int):
                found.add(value)
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


def main() -> None:
    run_experiment()


if __name__ == "__main__":
    main()
