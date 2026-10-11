"""Experiment 005: social-signal memory architectures.

CUM keeps Experiment 004's cumulative log1p/max scores.
Rolling and exponential memories use a centered score.
Familiarity weight stays 0. This module does not call a model.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
import statistics
from collections import deque
from pathlib import Path
from typing import Any

from agents.persona_generator import write_json
from config.settings import PROJECT_ROOT
from experiments.reproducibility_check import canonical_sha256
from world.choice import agent_reconsiders, choice_seed
from world.dynamics_calibration import OUTPUT_DIR as CALIBRATION_DIR
from world.dynamics_calibration import load_initial_locations, measure_world
from world.experiment_002 import REFERENCE_CHOICE_TEMPERATURE, REFERENCE_RECONSIDERATION_RATE
from world.experiment_004 import (
    OUTPUT_DIR as WORLD_V04_DIR,
    _argmax,
    _entropy,
    _hhi,
    _js_distance,
    _select,
    _share,
    _tv,
    _unique_leader,
    simulate_world as simulate_cumulative,
    trajectory_hash,
)
from world.geography import PLACE_IDS, distance_matrix, occupancy_of, place_index, travel_distance
from world.simulation import OUTPUT_DIR as WORLD_V01_DIR
from world.utility_atlas import OUTPUT_DIR as ATLAS_DIR
from world.utility_atlas import FrozenUtilityAtlas, load_frozen_atlas

OUTPUT_DIR = PROJECT_ROOT / "data" / "worlds" / "world_v0_5"
WORLD_V02_DIR = PROJECT_ROOT / "data" / "worlds" / "world_v0_2"
WORLD_V03_DIR = PROJECT_ROOT / "data" / "worlds" / "world_v0_3"
EXPECTED_ATLAS_SHA256 = "664b511d208f7989e8a671d0e2d6e6433366120bb3d5d6a67db38277b4d5a373"
FAMILIARITY_WEIGHT = 0.0
SOCIAL_WEIGHT = 150.0
SCORE_CLIP = 2.0
COMPRESSED_SCORE_SD = 0.10
EXTREME_LATE_HHI = 0.40
EXTREME_DOMINANT_SHARE = 0.60
EXTREME_SATURATION = 0.10
CHOICE_SEEDS = tuple(range(50))
TIMESTEPS = 500
CUM_NORMALIZATION = "experiment_004_log1p_over_max"
CENTERED_NORMALIZATION = "centered_population_sd_v1_clip_2"
CONDITIONS = (
    {"condition_id": "CUM", "memory": "cumulative", "window": None, "decay": None},
    {"condition_id": "RW20", "memory": "rolling", "window": 20, "decay": None},
    {"condition_id": "RW50", "memory": "rolling", "window": 50, "decay": None},
    {"condition_id": "EXP95", "memory": "exponential", "window": None, "decay": 0.95},
    {"condition_id": "EXP99", "memory": "exponential", "window": None, "decay": 0.99},
)
ANALYSIS_NOTE = (
    "Comparison of social-memory architectures. "
    "CUM uses Experiment 004's log1p/max scores. "
    "Rolling and exponential memories use centered scores, so social_weight 150 is not an equivalent strength across architectures. "
    "These labels are measurement diagnostics. This summary does not claim collective path dependence, "
    "human realism, destination emergence, lock-in, or self-organization."
)
CLASSIFICATION_RULE = (
    "A. signal-compressed: mean within-decision popularity-score SD < 0.10. "
    "B. discriminative-stable: score SD >= 0.10, ranking-change rate and destination-change rate both strictly above CUM, "
    "and the world is outside the unstable region. "
    "C. discriminative-unstable: the same discrimination increases, and the world is unstable. "
    "D. weak-discrimination: score SD >= 0.10, but ranking-change or destination-change is not strictly above CUM. "
    "Unstable means site_01 and site_02 lag-1 are both negative and alternation is at or above the Dynamics Calibration 001 "
    "upper quartile, or late HHI >= 0.40, or late dominant share >= 0.60, or saturation frequency >= 0.10, "
    "or consecutive occupancy TV is at or above that grid's upper quartile. "
    "CUM is not compared with itself; if it is not signal-compressed it is weak-discrimination."
)


class RollingMemory:
    def __init__(self, agent_ids: list[str], window: int) -> None:
        self.window = window
        self.commits: deque[dict[str, str]] = deque()
        self.total = {place_id: 0 for place_id in PLACE_IDS}
        self.own = {agent_id: {place_id: 0 for place_id in PLACE_IDS} for agent_id in agent_ids}

    def loo_raw(self, agent_id: str) -> dict[str, float]:
        return {place_id: float(self.total[place_id] - self.own[agent_id][place_id]) for place_id in PLACE_IDS}

    def commit(self, destinations: dict[str, str]) -> None:
        self.commits.append(dict(destinations))
        for agent_id, place_id in destinations.items():
            self.total[place_id] += 1
            self.own[agent_id][place_id] += 1
        if len(self.commits) > self.window:
            expired = self.commits.popleft()
            for agent_id, place_id in expired.items():
                self.total[place_id] -= 1
                self.own[agent_id][place_id] -= 1


class DecayMemory:
    def __init__(self, agent_ids: list[str], decay: float) -> None:
        self.decay = decay
        self.total = {place_id: 0.0 for place_id in PLACE_IDS}
        self.own = {agent_id: {place_id: 0.0 for place_id in PLACE_IDS} for agent_id in agent_ids}

    def loo_raw(self, agent_id: str) -> dict[str, float]:
        return {place_id: self.total[place_id] - self.own[agent_id][place_id] for place_id in PLACE_IDS}

    def commit(self, destinations: dict[str, str]) -> None:
        occupancy = {place_id: 0.0 for place_id in PLACE_IDS}
        for place_id in destinations.values():
            occupancy[place_id] += 1.0
        self.total = {place_id: self.decay * self.total[place_id] + occupancy[place_id] for place_id in PLACE_IDS}
        updated = {}
        for agent_id in self.own:
            place_id = destinations[agent_id]
            updated[agent_id] = {
                candidate: self.decay * self.own[agent_id][candidate] + (1.0 if candidate == place_id else 0.0)
                for candidate in PLACE_IDS
            }
        self.own = updated


def clip_score(value: float) -> float:
    return min(SCORE_CLIP, max(-SCORE_CLIP, value))


def centered_scores(raw: dict[str, float]) -> tuple[dict[str, float], dict[str, float]]:
    values = [raw[place_id] for place_id in PLACE_IDS]
    if max(values) == min(values):
        zeros = {place_id: 0.0 for place_id in PLACE_IDS}
        return zeros, zeros
    center = statistics.fmean(values)
    scale = math.sqrt(sum((value - center) ** 2 for value in values) / len(values))
    unclipped = {place_id: (raw[place_id] - center) / scale for place_id in PLACE_IDS}
    clipped = {place_id: clip_score(value) for place_id, value in unclipped.items()}
    return unclipped, clipped


def simulate_memory(
    atlas: FrozenUtilityAtlas,
    locations: dict[str, str],
    memory: RollingMemory | DecayMemory,
    *,
    choice_seed_value: int,
    timesteps: int = TIMESTEPS,
    on_step: Any = None,
) -> dict[str, Any]:
    matrix = distance_matrix()
    places = place_index()
    current = {agent_id: locations[agent_id] for agent_id in sorted(locations)}
    series = [dict(current)]
    moves = 0
    bucket = _empty_bucket()
    for timestep in range(timesteps):
        logged = timestep + 1
        occupancy = occupancy_of(current)
        decisions: dict[str, str] = {}
        actions = []
        for agent_id in sorted(current):
            origin = current[agent_id]
            _unclipped, scores = centered_scores(memory.loo_raw(agent_id))
            bases, totals, social_term = _utilities(atlas, matrix, places, agent_id, origin, occupancy, scores)
            does_reconsider = agent_reconsiders(
                choice_seed_value,
                timestep,
                agent_id,
                REFERENCE_RECONSIDERATION_RATE,
            )
            ranking_changed = _argmax(totals) != _argmax(bases)
            if not does_reconsider:
                destination = origin
                probabilities = {place_id: 1.0 if place_id == origin else 0.0 for place_id in PLACE_IDS}
                destination_changed = False
            else:
                draw = random.Random(choice_seed(choice_seed_value, timestep, agent_id)).random()
                destination, probabilities = _select(list(PLACE_IDS), [totals[place_id] for place_id in PLACE_IDS], draw)
                base_destination, _base_probabilities = _select(
                    list(PLACE_IDS),
                    [bases[place_id] for place_id in PLACE_IDS],
                    draw,
                )
                destination_changed = destination != base_destination
            decisions[agent_id] = destination
            actions.append(
                {
                    "timestep": logged,
                    "agent_id": agent_id,
                    "origin": origin,
                    "destination": destination,
                    "stayed": destination == origin,
                    "reconsidered": does_reconsider,
                    "social_weight": SOCIAL_WEIGHT,
                    "familiarity_weight": FAMILIARITY_WEIGHT,
                    "normalization": CENTERED_NORMALIZATION,
                    "base_utility": bases,
                    "popularity_score": scores,
                    "social_utility": social_term,
                    "total_utility": totals,
                    "choice_probabilities": probabilities,
                    "occupancy_before": occupancy,
                    "ranking_changed": ranking_changed,
                    "destination_changed": destination_changed,
                    "llm_calls": 0,
                }
            )
        _absorb(bucket, actions, centered=True)
        moves += sum(1 for agent_id in decisions if decisions[agent_id] != current[agent_id])
        current = {agent_id: decisions[agent_id] for agent_id in sorted(current)}
        memory.commit(decisions)
        series.append(dict(current))
        if on_step is not None:
            on_step(logged, current, memory, actions)
    return {"series": series, "moves": moves, "bucket": bucket, "llm_calls": 0}


def run_experiment(
    atlas: FrozenUtilityAtlas | None = None,
    *,
    output_dir: Path | None = None,
    seeds: tuple[int, ...] = CHOICE_SEEDS,
    timesteps: int = TIMESTEPS,
) -> Path:
    destination = output_dir or OUTPUT_DIR
    _refuse_frozen_directories(destination)
    destination.mkdir(parents=True, exist_ok=True)
    source = load_frozen_atlas() if atlas is None else atlas
    _require_atlas_hash()
    locations = load_initial_locations()
    thresholds = _calibration_thresholds()
    config = _config(seeds, timesteps, thresholds)
    write_json(destination / "world_v0_5_config.json", config)
    state_path = destination / "world_states.jsonl"
    memory_path = destination / "social_memory_states.jsonl"
    action_path = destination / "agent_actions.jsonl"
    for path in (state_path, memory_path, action_path):
        path.write_text("", encoding="utf-8", newline="\n")
    reference = _load_s150_locations()
    worlds = []
    with state_path.open("a", encoding="utf-8", newline="\n") as state_file, memory_path.open(
        "a", encoding="utf-8", newline="\n"
    ) as memory_file, action_path.open("a", encoding="utf-8", newline="\n") as action_file:
        for spec in CONDITIONS:
            for seed in seeds:
                _write_initial(state_file, memory_file, spec, seed, locations)
                if spec["memory"] == "cumulative":
                    bucket = _empty_bucket()
                    result = simulate_cumulative(
                        source,
                        locations,
                        _cumulative_social(list(locations)),
                        social_weight=SOCIAL_WEIGHT,
                        choice_seed_value=seed,
                        timesteps=timesteps,
                        on_step=_log_cumulative(state_file, memory_file, action_file, spec, seed, bucket),
                    )
                    result["bucket"] = bucket
                    if result["series"] != reference[seed]:
                        raise RuntimeError(f"CUM diverges from Experiment 004 S150 at choice seed {seed}")
                else:
                    memory = _make_memory(spec, list(locations))
                    result = simulate_memory(
                        source,
                        locations,
                        memory,
                        choice_seed_value=seed,
                        timesteps=timesteps,
                        on_step=_log_memory(state_file, memory_file, action_file, spec, seed),
                    )
                if result["llm_calls"] != 0:
                    raise RuntimeError("experiment 005 called a model")
                worlds.append(_world_metrics(spec, seed, result, timesteps))
                print(
                    f"{spec['condition_id']} seed={seed} score_sd={worlds[-1]['score_sd']:.3f} late_hhi={worlds[-1]['late_hhi']:.3f}",
                    flush=True,
                )
    replays = {}
    for condition_id in ("CUM", "RW20", "EXP95"):
        spec = next(item for item in CONDITIONS if item["condition_id"] == condition_id)
        replay_hash = _rerun_hash(source, locations, spec, seeds[0], timesteps)
        original = next(world for world in worlds if world["condition_id"] == condition_id and world["choice_seed"] == seeds[0])
        if replay_hash != original["trajectory_sha256"]:
            raise RuntimeError(f"{condition_id} exact replay hash did not match")
        replays[condition_id] = {"choice_seed": seeds[0], "trajectory_sha256": replay_hash, "match": True}
    summary = _summarize(worlds, seeds, timesteps, thresholds, replays)
    write_json(destination / "experiment_005_summary.json", summary)
    manifest = _manifest(destination, config, summary, worlds)
    write_json(destination / "run_manifest.json", manifest)
    return destination


def classify_architecture(
    *,
    score_sd: float,
    ranking_change_rate: float,
    destination_change_rate: float,
    cum_ranking_change_rate: float | None,
    cum_destination_change_rate: float | None,
    unstable: bool,
) -> str:
    if score_sd < COMPRESSED_SCORE_SD:
        return "signal-compressed"
    more_discriminative = (
        cum_ranking_change_rate is not None
        and cum_destination_change_rate is not None
        and ranking_change_rate > cum_ranking_change_rate
        and destination_change_rate > cum_destination_change_rate
    )
    if more_discriminative and unstable:
        return "discriminative-unstable"
    if more_discriminative:
        return "discriminative-stable"
    return "weak-discrimination"


def is_unstable(
    *,
    lag1_site_01: float | None,
    lag1_site_02: float | None,
    alternation: float,
    alternation_cutoff: float,
    late_hhi: float,
    dominant_share: float,
    saturation: float,
    consecutive_tv: float,
    consecutive_tv_cutoff: float,
) -> bool:
    oscillatory = (
        lag1_site_01 is not None
        and lag1_site_02 is not None
        and lag1_site_01 < 0.0
        and lag1_site_02 < 0.0
        and alternation >= alternation_cutoff
    )
    extreme = late_hhi >= EXTREME_LATE_HHI or dominant_share >= EXTREME_DOMINANT_SHARE or saturation >= EXTREME_SATURATION
    return oscillatory or extreme or consecutive_tv >= consecutive_tv_cutoff


def _utilities(
    atlas: FrozenUtilityAtlas,
    matrix: dict[str, dict[str, float]],
    places: dict[str, dict[str, Any]],
    agent_id: str,
    origin: str,
    occupancy: dict[str, int],
    scores: dict[str, float],
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    bases: dict[str, float] = {}
    totals: dict[str, float] = {}
    social_term: dict[str, float] = {}
    for place_id in PLACE_IDS:
        capacity = int(places[place_id]["capacity"])
        distance = travel_distance(matrix, origin, place_id)
        base = float(atlas.lookup(agent_id, place_id, distance, occupancy[place_id], capacity))
        term = SOCIAL_WEIGHT * scores[place_id]
        bases[place_id] = base
        social_term[place_id] = term
        totals[place_id] = base + term
    return bases, totals, social_term


def _world_metrics(spec: dict[str, Any], seed: int, result: dict[str, Any], timesteps: int) -> dict[str, Any]:
    series = result["series"]
    early_counts = _presence(series, 1, 20)
    late_counts = _presence(series, 101, timesteps)
    full_counts = _presence(series, 1, timesteps)
    early_share = _share(early_counts)
    late_share = _share(late_counts)
    full_share = _share(full_counts)
    early_leader = _unique_leader(early_counts)
    late_leader = _unique_leader(late_counts)
    diagnostics = measure_world(series, moves=result["moves"], timesteps=timesteps)
    bucket = result["bucket"]
    decisions = bucket["decisions"]
    return {
        "condition_id": spec["condition_id"],
        "memory": spec["memory"],
        "choice_seed": seed,
        "social_weight": SOCIAL_WEIGHT,
        "familiarity_weight": FAMILIARITY_WEIGHT,
        "trajectory_sha256": trajectory_hash(series),
        "score_sd": bucket["sd"] / decisions,
        "score_range": bucket["range"] / decisions,
        "social_spread": bucket["spread"] / decisions,
        "ranking_change_rate": bucket["rank"] / decisions,
        "destination_change_rate": None if bucket["reconsidered"] == 0 else bucket["dest"] / bucket["reconsidered"],
        "saturation_rate": bucket["saturated"] / bucket["score_n"],
        "late_hhi": _hhi(late_share),
        "late_entropy": _entropy(late_share),
        "late_dominant_share": max(late_share.values()),
        "late_dominant": late_leader,
        "late_share": late_share,
        "full_dominant_share": max(full_share.values()),
        "full_dominant": _unique_leader(full_counts),
        "early_leader_retained": None if early_leader is None else late_leader == early_leader,
        "early_late_tv": _tv(early_share, late_share),
        "early_late_js_distance": _js_distance(early_share, late_share),
        "movement_rate": diagnostics["movement_rate"],
        "mean_dwell": diagnostics["mean_dwell"],
        "occupancy_entropy": diagnostics["entropy_mean"],
        "saturation_frequency": diagnostics["saturation_frequency"],
        "empty_frequency": diagnostics["empty_frequency"],
        "lag1": diagnostics["lag1"],
        "lag2": diagnostics["lag2"],
        "alternation_rate": diagnostics["site_01_site_02_alternation_rate"],
        "consecutive_occupancy_tv": diagnostics["mean_total_variation"],
        "mean_occupancy": diagnostics["mean_occupancy"],
        "llm_calls": 0,
    }


def _summarize(
    worlds: list[dict[str, Any]],
    seeds: tuple[int, ...],
    timesteps: int,
    thresholds: dict[str, float],
    replays: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    by_condition = {
        spec["condition_id"]: [world for world in worlds if world["condition_id"] == spec["condition_id"]]
        for spec in CONDITIONS
    }
    conditions = {}
    for condition_id, rows in by_condition.items():
        conditions[condition_id] = _condition_summary(rows, thresholds)
    cum = conditions["CUM"]
    for condition_id, block in conditions.items():
        block["architecture_class"] = classify_architecture(
            score_sd=float(block["score_sd"]["mean"]),
            ranking_change_rate=float(block["ranking_change_rate"]["mean"]),
            destination_change_rate=float(block["destination_change_rate"]["mean"]),
            cum_ranking_change_rate=None if condition_id == "CUM" else float(cum["ranking_change_rate"]["mean"]),
            cum_destination_change_rate=None if condition_id == "CUM" else float(cum["destination_change_rate"]["mean"]),
            unstable=bool(block["unstable"]),
        )
    return {
        "experiment_id": "experiment_005_social_memory",
        "note": ANALYSIS_NOTE,
        "classification_rule": CLASSIFICATION_RULE,
        "cum_normalization": CUM_NORMALIZATION,
        "centered_normalization": CENTERED_NORMALIZATION,
        "social_weight": SOCIAL_WEIGHT,
        "familiarity_weight": FAMILIARITY_WEIGHT,
        "weights_not_equivalent_across_architectures": True,
        "llm_calls": 0,
        "world_count": len(worlds),
        "timesteps": timesteps,
        "choice_seeds": list(seeds),
        "cum_matches_experiment_004_s150": True,
        "js_definition": "Jensen-Shannon distance = sqrt(JS divergence), with natural log.",
        "calibration_thresholds": thresholds,
        "conditions": conditions,
        "exact_replay": replays,
    }


def _condition_summary(worlds: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
    counts = _dominant_counts([world["full_dominant"] for world in worlds])
    lag1 = {place_id: _mean_sd([world["lag1"][place_id] for world in worlds]) for place_id in ("site_01", "site_02")}
    late_hhi = _mean_sd([world["late_hhi"] for world in worlds])
    dominant_share = _mean_sd([world["late_dominant_share"] for world in worlds])
    saturation = _mean_sd([world["saturation_frequency"] for world in worlds])
    alternation = _mean_sd([world["alternation_rate"] for world in worlds])
    consecutive = _mean_sd([world["consecutive_occupancy_tv"] for world in worlds])
    unstable = is_unstable(
        lag1_site_01=lag1["site_01"]["mean"],
        lag1_site_02=lag1["site_02"]["mean"],
        alternation=float(alternation["mean"]),
        alternation_cutoff=thresholds["alternation_upper_quartile"],
        late_hhi=float(late_hhi["mean"]),
        dominant_share=float(dominant_share["mean"]),
        saturation=float(saturation["mean"]),
        consecutive_tv=float(consecutive["mean"]),
        consecutive_tv_cutoff=thresholds["consecutive_tv_upper_quartile"],
    )
    return {
        "n_seeds": len(worlds),
        "llm_calls": 0,
        "normalization": CUM_NORMALIZATION if worlds[0]["memory"] == "cumulative" else CENTERED_NORMALIZATION,
        "score_sd": _mean_sd([world["score_sd"] for world in worlds]),
        "score_range": _mean_sd([world["score_range"] for world in worlds]),
        "social_utility_spread": _mean_sd([world["social_spread"] for world in worlds]),
        "ranking_change_rate": _mean_sd([world["ranking_change_rate"] for world in worlds]),
        "destination_change_rate": _mean_sd([world["destination_change_rate"] for world in worlds]),
        "saturation_or_compression_rate": _mean_sd([world["saturation_rate"] for world in worlds]),
        "late_hhi": late_hhi,
        "late_entropy": _mean_sd([world["late_entropy"] for world in worlds]),
        "late_dominant_share": dominant_share,
        "full_dominant_share": _mean_sd([world["full_dominant_share"] for world in worlds]),
        "final_dominant_counts": counts,
        "final_dominant_proportions": {key: value / len(worlds) for key, value in counts.items()},
        "final_dominant_entropy": _category_entropy(
            ["tie" if world["full_dominant"] is None else str(world["full_dominant"]) for world in worlds]
        ),
        "early_leader_retention": _mean_sd(
            [None if world["early_leader_retained"] is None else float(world["early_leader_retained"]) for world in worlds]
        ),
        "early_late_tv": _mean_sd([world["early_late_tv"] for world in worlds]),
        "early_late_js_distance": _mean_sd([world["early_late_js_distance"] for world in worlds]),
        "between_seed_late_tv": _between(worlds, _tv),
        "between_seed_late_js_distance": _between(worlds, _js_distance),
        "movement_rate": _mean_sd([world["movement_rate"] for world in worlds]),
        "mean_dwell": _mean_sd([world["mean_dwell"] for world in worlds]),
        "occupancy_entropy": _mean_sd([world["occupancy_entropy"] for world in worlds]),
        "saturation_frequency": saturation,
        "empty_frequency": _mean_sd([world["empty_frequency"] for world in worlds]),
        "alternation_rate": alternation,
        "consecutive_occupancy_tv": consecutive,
        "lag1": {place_id: _mean_sd([world["lag1"][place_id] for world in worlds]) for place_id in PLACE_IDS},
        "lag2": {place_id: _mean_sd([world["lag2"][place_id] for world in worlds]) for place_id in PLACE_IDS},
        "mean_occupancy": {
            place_id: _mean_sd([world["mean_occupancy"][place_id] for world in worlds]) for place_id in PLACE_IDS
        },
        "unstable": unstable,
    }


def _between(worlds: list[dict[str, Any]], distance: Any) -> dict[str, float | int | None]:
    values = [
        distance(left["late_share"], right["late_share"])
        for left, right in itertools.combinations(worlds, 2)
    ]
    quartiles = statistics.quantiles(values, n=4, method="inclusive")
    return {
        "mean": statistics.fmean(values),
        "sd": _stdev(values),
        "median": statistics.median(values),
        "q1": quartiles[0],
        "q3": quartiles[2],
        "n_pairs": len(values),
    }


def _rerun_hash(
    atlas: FrozenUtilityAtlas,
    locations: dict[str, str],
    spec: dict[str, Any],
    seed: int,
    timesteps: int,
) -> str:
    if spec["memory"] == "cumulative":
        result = simulate_cumulative(
            atlas,
            locations,
            _cumulative_social(list(locations)),
            social_weight=SOCIAL_WEIGHT,
            choice_seed_value=seed,
            timesteps=timesteps,
        )
    else:
        result = simulate_memory(
            atlas,
            locations,
            _make_memory(spec, list(locations)),
            choice_seed_value=seed,
            timesteps=timesteps,
        )
    return trajectory_hash(result["series"])


def _make_memory(spec: dict[str, Any], agent_ids: list[str]) -> RollingMemory | DecayMemory:
    if spec["memory"] == "rolling":
        return RollingMemory(agent_ids, int(spec["window"]))
    if spec["memory"] == "exponential":
        return DecayMemory(agent_ids, float(spec["decay"]))
    raise RuntimeError("unknown social memory")


def _cumulative_social(agent_ids: list[str]) -> Any:
    from world.experiment_004 import empty_social

    return empty_social(agent_ids)


def _empty_bucket() -> dict[str, float]:
    return {"sd": 0.0, "range": 0.0, "spread": 0.0, "rank": 0.0, "dest": 0.0, "reconsidered": 0.0, "decisions": 0.0, "saturated": 0.0, "score_n": 0.0}


def _absorb(bucket: dict[str, float], actions: list[dict[str, Any]], *, centered: bool) -> None:
    for action in actions:
        scores = [float(action["popularity_score"][place_id]) for place_id in PLACE_IDS]
        social = [float(action["social_utility"][place_id]) for place_id in PLACE_IDS]
        bucket["sd"] += _population_sd(scores)
        bucket["range"] += max(scores) - min(scores)
        bucket["spread"] += max(social) - min(social)
        bucket["rank"] += 1.0 if action["ranking_changed"] else 0.0
        bucket["decisions"] += 1.0
        if action["reconsidered"]:
            bucket["reconsidered"] += 1.0
            bucket["dest"] += 1.0 if action["destination_changed"] else 0.0
        for score in scores:
            bucket["score_n"] += 1.0
            if centered and abs(score) >= 1.9:
                bucket["saturated"] += 1.0
            if not centered and score > 0.95:
                bucket["saturated"] += 1.0


def _population_sd(values: list[float]) -> float:
    center = statistics.fmean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / len(values))


def _presence(series: list[dict[str, str]], start: int, stop: int) -> dict[str, int]:
    counts = {place_id: 0 for place_id in PLACE_IDS}
    for step in series[start : stop + 1]:
        for place_id in step.values():
            counts[place_id] += 1
    return counts


def _dominant_counts(labels: list[str | None]) -> dict[str, int]:
    counts = {place_id: 0 for place_id in PLACE_IDS}
    counts["tie"] = 0
    for label in labels:
        counts["tie" if label is None else label] += 1
    return counts


def _category_entropy(labels: list[str]) -> float:
    total = len(labels)
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return -sum((count / total) * math.log(count / total) for count in counts.values())


def _calibration_thresholds() -> dict[str, float]:
    payload = json.loads((CALIBRATION_DIR / "summary.json").read_text(encoding="utf-8"))
    alternation = [float(cell["site_01_site_02"]["alternation_rate"]["mean"]) for cell in payload["cells"]]
    consecutive = [float(cell["consecutive_state_total_variation"]["mean"]) for cell in payload["cells"]]
    return {
        "alternation_upper_quartile": statistics.quantiles(alternation, n=4, method="inclusive")[2],
        "consecutive_tv_upper_quartile": statistics.quantiles(consecutive, n=4, method="inclusive")[2],
        "extreme_late_hhi": EXTREME_LATE_HHI,
        "extreme_dominant_share": EXTREME_DOMINANT_SHARE,
        "extreme_saturation": EXTREME_SATURATION,
    }


def _load_s150_locations() -> dict[int, list[dict[str, str]]]:
    grouped: dict[int, list[tuple[int, dict[str, str]]]] = {seed: [] for seed in CHOICE_SEEDS}
    path = WORLD_V04_DIR / "world_states.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if '"condition_id":"S150",' not in line:
                continue
            row = json.loads(line)
            if row.get("condition_id") != "S150":
                continue
            seed = int(row["choice_seed"])
            if seed not in grouped:
                continue
            grouped[seed].append((int(row["timestep"]), {str(key): str(value) for key, value in row["locations"].items()}))
    loaded = {}
    for seed, rows in grouped.items():
        rows.sort(key=lambda item: item[0])
        loaded[seed] = [locations for _timestep, locations in rows[:501]]
        if len(loaded[seed]) != 501:
            raise RuntimeError(f"Experiment 004 S150 seed {seed} does not have 501 states")
    return loaded


def _config(seeds: tuple[int, ...], timesteps: int, thresholds: dict[str, float]) -> dict[str, Any]:
    atlas_manifest = json.loads((ATLAS_DIR / "atlas_manifest.json").read_text(encoding="utf-8"))
    payload = {
        "world_id": "world_v0_5",
        "experiment_id": "experiment_005_social_memory",
        "utility_source": "utility_atlas_v1",
        "utility_atlas_sha256": atlas_manifest["sha256"],
        "prompt_version": "independent_utility_v3",
        "choice_temperature": REFERENCE_CHOICE_TEMPERATURE,
        "reconsideration_rate": REFERENCE_RECONSIDERATION_RATE,
        "social_weight": SOCIAL_WEIGHT,
        "familiarity_weight": FAMILIARITY_WEIGHT,
        "weights_not_equivalent_across_architectures": True,
        "cum_normalization": CUM_NORMALIZATION,
        "centered_normalization": CENTERED_NORMALIZATION,
        "choice_seeds": list(seeds),
        "timesteps": timesteps,
        "conditions": [dict(spec) for spec in CONDITIONS],
        "classification_rule": CLASSIFICATION_RULE,
        "calibration_thresholds": thresholds,
        "llm_calls": 0,
    }
    payload["config_sha256"] = canonical_sha256(payload)
    return payload


def _manifest(destination: Path, config: dict[str, Any], summary: dict[str, Any], worlds: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "experiment_id": "experiment_005_social_memory",
        "llm_calls": 0,
        "world_count": len(worlds),
        "complete": len(worlds) == len(CONDITIONS) * len(config["choice_seeds"]),
        "utility_atlas_sha256": config["utility_atlas_sha256"],
        "config_sha256": config["config_sha256"],
        "cum_matches_experiment_004_s150": True,
        "exact_replay": summary["exact_replay"],
        "architecture_class": {
            condition_id: summary["conditions"][condition_id]["architecture_class"] for condition_id in summary["conditions"]
        },
        "trajectory_sha256": {
            f"{world['condition_id']}:{world['choice_seed']}": world["trajectory_sha256"] for world in worlds
        },
        "files": {
            name: hashlib.sha256((destination / name).read_bytes()).hexdigest()
            for name in (
                "world_v0_5_config.json",
                "social_memory_states.jsonl",
                "world_states.jsonl",
                "agent_actions.jsonl",
                "experiment_005_summary.json",
            )
        },
    }


def _log_cumulative(state_file: Any, memory_file: Any, action_file: Any, spec: dict[str, Any], seed: int, bucket: dict[str, float]) -> Any:
    def write(logged: int, locations: dict[str, str], social: Any, actions: list[dict[str, Any]]) -> None:
        _absorb(bucket, actions, centered=False)
        _dump_state(state_file, spec, seed, logged, locations)
        _dump(
            memory_file,
            {
                "condition_id": spec["condition_id"],
                "choice_seed": seed,
                "timestep": logged,
                "memory": "cumulative",
                "normalization": CUM_NORMALIZATION,
                "raw_total": social.total,
                "raw_own": social.own,
                "llm_calls": 0,
            },
        )
        for action in actions:
            stored = dict(action)
            stored["normalization"] = CUM_NORMALIZATION
            _dump(action_file, {"condition_id": spec["condition_id"], "choice_seed": seed, **stored})

    return write


def _log_memory(state_file: Any, memory_file: Any, action_file: Any, spec: dict[str, Any], seed: int) -> Any:
    def write(logged: int, locations: dict[str, str], memory: RollingMemory | DecayMemory, actions: list[dict[str, Any]]) -> None:
        _dump_state(state_file, spec, seed, logged, locations)
        _dump(
            memory_file,
            {
                "condition_id": spec["condition_id"],
                "choice_seed": seed,
                "timestep": logged,
                "memory": spec["memory"],
                "window": spec["window"],
                "decay": spec["decay"],
                "normalization": CENTERED_NORMALIZATION,
                "raw_total": memory.total,
                "raw_own": memory.own,
                "llm_calls": 0,
            },
        )
        for action in actions:
            _dump(action_file, {"condition_id": spec["condition_id"], "choice_seed": seed, **action})

    return write


def _write_initial(state_file: Any, memory_file: Any, spec: dict[str, Any], seed: int, locations: dict[str, str]) -> None:
    _dump_state(state_file, spec, seed, 0, locations)
    zeros = {place_id: 0 for place_id in PLACE_IDS}
    own = {agent_id: dict(zeros) for agent_id in sorted(locations)}
    _dump(
        memory_file,
        {
            "condition_id": spec["condition_id"],
            "choice_seed": seed,
            "timestep": 0,
            "memory": spec["memory"],
            "window": spec["window"],
            "decay": spec["decay"],
            "raw_total": zeros,
            "raw_own": own,
            "llm_calls": 0,
        },
    )


def _dump_state(handle: Any, spec: dict[str, Any], seed: int, logged: int, locations: dict[str, str]) -> None:
    ordered = {agent_id: locations[agent_id] for agent_id in sorted(locations)}
    _dump(
        handle,
        {
            "condition_id": spec["condition_id"],
            "choice_seed": seed,
            "social_weight": SOCIAL_WEIGHT,
            "timestep": logged,
            "locations": ordered,
            "occupancy": occupancy_of(ordered),
            "llm_calls": 0,
        },
    )


def _mean_sd(values: list[float | None]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if value is not None]
    return {"mean": None if not finite else statistics.fmean(finite), "sd": _stdev(finite), "n": len(finite)}


def _stdev(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return float(statistics.stdev(values))


def _require_atlas_hash() -> None:
    manifest = json.loads((ATLAS_DIR / "atlas_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("sha256") != EXPECTED_ATLAS_SHA256:
        raise RuntimeError("utility atlas hash is not the frozen experiment atlas")


def _refuse_frozen_directories(destination: Path) -> None:
    forbidden = {
        WORLD_V01_DIR.resolve(),
        WORLD_V02_DIR.resolve(),
        WORLD_V03_DIR.resolve(),
        WORLD_V04_DIR.resolve(),
        ATLAS_DIR.resolve(),
        CALIBRATION_DIR.resolve(),
    }
    if destination.resolve() in forbidden:
        raise RuntimeError("experiment 005 must not write into frozen world outputs")


def _dump(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")


def main() -> None:
    run_experiment()


if __name__ == "__main__":
    if __package__ is None:
        raise SystemExit("請在 worldlab 目錄執行：python -m world.experiment_005")
    main()
