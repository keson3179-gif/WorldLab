"""Experiment 003: endogenous familiarity reinforcement.

Every agent starts with the same zero history. Visit counts accumulate only
from moves committed inside the run. Utilities come from the frozen atlas.
This module does not call a model and does not write into earlier world outputs.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Any

from agents.persona_generator import write_json
from config.settings import PROJECT_ROOT
from experiments.experiment_001 import load_base_traits
from experiments.reproducibility_check import canonical_sha256
from world.choice import agent_reconsiders, choice_seed, softmax_choice
from world.dynamics_calibration import load_initial_locations, measure_world
from world.experiment_002 import (
    OUTPUT_DIR as WORLD_V02_DIR,
    REFERENCE_CHOICE_TEMPERATURE,
    REFERENCE_RECONSIDERATION_RATE,
)
from world.geography import PLACE_IDS, distance_matrix, occupancy_of, place_index, travel_distance
from world.history import (
    FAMILIARITY_RULE,
    AgentHistory,
    blank_history,
    clone_population,
    commit_history,
    decision_utility,
    familiarity_scores,
)
from world.simulation import OUTPUT_DIR as WORLD_V01_DIR
from world.utility_atlas import OUTPUT_DIR as ATLAS_DIR
from world.utility_atlas import FrozenUtilityAtlas, load_frozen_atlas

OUTPUT_DIR = PROJECT_ROOT / "data" / "worlds" / "world_v0_3"
CALIBRATION_DIR = PROJECT_ROOT / "data" / "worlds" / "dynamics_calibration_001"
CHOICE_SEEDS = tuple(range(50))
TIMESTEPS = 500
RETURN_WINDOW = 5
FORK_AGENT = "agent_0000"
FORK_TIMESTEP = 20
FORK_SEEDS = tuple(range(20))
CONDITIONS = (
    {"condition_id": "R0", "reinforcement_weight": 0.0},
    {"condition_id": "R50", "reinforcement_weight": 50.0},
    {"condition_id": "R150", "reinforcement_weight": 150.0},
)
ANALYSIS_NOTE = (
    "Descriptive comparison of endogenous familiarity reinforcement. "
    "This summary does not claim human realism or emergence."
)
FORK_NOTE = (
    "Single-event perturbation persistence. "
    "A difference after one forced destination change is not a butterfly-effect claim."
)
AMPLIFICATION_RULE = (
    "history reinforcement amplified path dependence only when, from R0 to R50 to R150, "
    "early-leader retention rises, final personal concentration rises, final personal visit entropy falls, "
    "and between-seed location disagreement over t=101–500 rises."
)
NEUTRAL_NOTE = (
    "Every agent-place history starts at visit_count 0, cumulative_dwell 0, "
    "and time_since_last_visit null. The initial location is not a visit."
)


def neutral_population(traits: dict[str, Any], locations: dict[str, str]) -> dict[str, AgentHistory]:
    return {
        agent_id: blank_history(agent_id, traits[agent_id], locations[agent_id])
        for agent_id in sorted(locations)
    }


def alternate_destination(utilities: dict[str, float], destination: str) -> str:
    ranked = sorted(PLACE_IDS, key=lambda place_id: (-utilities[place_id], place_id))
    if ranked[1] != destination:
        return ranked[1]
    return ranked[0]


def simulate_world(
    atlas: FrozenUtilityAtlas,
    states: dict[str, AgentHistory],
    *,
    reinforcement_weight: float,
    choice_seed_value: int,
    timesteps: int = TIMESTEPS,
    start_index: int = 0,
    on_step: Any = None,
) -> dict[str, Any]:
    matrix = distance_matrix()
    places = place_index()
    current = clone_population(states)
    series = [{agent_id: current[agent_id].current_location for agent_id in sorted(current)}]
    entries = {agent_id: 0 for agent_id in current}
    repeats = {agent_id: 0 for agent_id in current}
    snapshots = {0: _visit_snapshot(current)} if start_index == 0 else {}
    moves = 0
    for offset in range(timesteps):
        timestep = start_index + offset
        destinations, actions = decide_timestep(
            atlas,
            current,
            matrix,
            places,
            reinforcement_weight=reinforcement_weight,
            choice_seed_value=choice_seed_value,
            timestep=timestep,
        )
        for action in actions:
            if action["destination"] == action["origin"]:
                continue
            entries[action["agent_id"]] += 1
            if action["visit_count_before"][action["destination"]] > 0:
                repeats[action["agent_id"]] += 1
        moves += sum(1 for agent_id, destination in destinations.items() if destination != series[-1][agent_id])
        current = commit_history(current, destinations)
        series.append({agent_id: current[agent_id].current_location for agent_id in sorted(current)})
        logged = timestep + 1
        if logged in {20, 100, 500} or logged == start_index + timesteps:
            snapshots[logged] = _visit_snapshot(current)
        if on_step is not None:
            on_step(logged, series[-1], current, actions)
    return {
        "series": series,
        "snapshots": snapshots,
        "entries": entries,
        "repeats": repeats,
        "moves": moves,
        "final_states": current,
        "committed_steps": timesteps,
        "llm_calls": 0,
    }


def decide_timestep(
    atlas: FrozenUtilityAtlas,
    states: dict[str, AgentHistory],
    matrix: dict[str, dict[str, float]],
    places: dict[str, dict[str, Any]],
    *,
    reinforcement_weight: float,
    choice_seed_value: int,
    timestep: int,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    occupancy = occupancy_of({agent_id: states[agent_id].current_location for agent_id in states})
    decisions: dict[str, str] = {}
    actions = []
    for agent_id in sorted(states):
        state = states[agent_id]
        origin = state.current_location
        reconsidered = agent_reconsiders(
            choice_seed_value,
            timestep,
            agent_id,
            REFERENCE_RECONSIDERATION_RATE,
        )
        scores = familiarity_scores(state.visit_count)
        bases: dict[str, float] = {}
        totals: dict[str, float] = {}
        for place_id in PLACE_IDS:
            capacity = int(places[place_id]["capacity"])
            distance = travel_distance(matrix, origin, place_id)
            base = float(atlas.lookup(agent_id, place_id, distance, occupancy[place_id], capacity))
            bases[place_id] = base
            totals[place_id] = decision_utility(base, scores[place_id], reinforcement_weight)
        if not reconsidered:
            destination = origin
            probabilities = {place_id: 1.0 if place_id == origin else 0.0 for place_id in PLACE_IDS}
        else:
            destination, probabilities = softmax_choice(
                list(PLACE_IDS),
                [totals[place_id] for place_id in PLACE_IDS],
                REFERENCE_CHOICE_TEMPERATURE,
                choice_seed(choice_seed_value, timestep, agent_id),
            )
        decisions[agent_id] = destination
        actions.append(
            {
                "timestep": timestep + 1,
                "agent_id": agent_id,
                "origin": origin,
                "destination": destination,
                "stayed": destination == origin,
                "reconsidered": reconsidered,
                "reinforcement_weight": reinforcement_weight,
                "base_utility": bases,
                "familiarity_score": scores,
                "total_utility": totals,
                "choice_probabilities": {place_id: probabilities[place_id] for place_id in PLACE_IDS},
                "occupancy_before": occupancy,
                "visit_count_before": dict(state.visit_count),
                "llm_calls": 0,
            }
        )
    return decisions, actions


def run_fork(
    atlas: FrozenUtilityAtlas,
    states: dict[str, AgentHistory],
    *,
    choice_seed_value: int,
    timesteps: int = TIMESTEPS,
    fork_timestep: int = FORK_TIMESTEP,
    agent_id: str = FORK_AGENT,
) -> dict[str, Any]:
    matrix = distance_matrix()
    places = place_index()
    current = clone_population(states)
    shared = [{agent_id_: current[agent_id_].current_location for agent_id_ in sorted(current)}]
    for timestep in range(fork_timestep - 1):
        destinations, _actions = decide_timestep(
            atlas,
            current,
            matrix,
            places,
            reinforcement_weight=150.0,
            choice_seed_value=choice_seed_value,
            timestep=timestep,
        )
        current = commit_history(current, destinations)
        shared.append({agent_id_: current[agent_id_].current_location for agent_id_ in sorted(current)})
    destinations, actions = decide_timestep(
        atlas,
        current,
        matrix,
        places,
        reinforcement_weight=150.0,
        choice_seed_value=choice_seed_value,
        timestep=fork_timestep - 1,
    )
    target = next(action for action in actions if action["agent_id"] == agent_id)
    replacement = alternate_destination(target["total_utility"], target["destination"])
    if replacement == target["destination"]:
        raise RuntimeError("fork replacement did not change the destination")
    branch_a_dest = dict(destinations)
    branch_b_dest = dict(destinations)
    branch_b_dest[agent_id] = replacement
    branch_a = commit_history(current, branch_a_dest)
    branch_b = commit_history(current, branch_b_dest)
    rest = timesteps - fork_timestep
    continued_a = simulate_world(
        atlas,
        branch_a,
        reinforcement_weight=150.0,
        choice_seed_value=choice_seed_value,
        timesteps=rest,
        start_index=fork_timestep,
    )
    continued_b = simulate_world(
        atlas,
        branch_b,
        reinforcement_weight=150.0,
        choice_seed_value=choice_seed_value,
        timesteps=rest,
        start_index=fork_timestep,
    )
    series_a = shared + continued_a["series"]
    series_b = shared + continued_b["series"]
    return {
        "choice_seed": choice_seed_value,
        "fork_timestep": fork_timestep,
        "agent_id": agent_id,
        "original_destination": target["destination"],
        "replacement_destination": replacement,
        "utilities": target["total_utility"],
        "series_a": series_a,
        "series_b": series_b,
        "final_a": continued_a["final_states"],
        "final_b": continued_b["final_states"],
        "llm_calls": 0,
    }


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
    locations = load_initial_locations()
    bases = load_base_traits(PROJECT_ROOT / "data" / "personas" / "persona_set_v1", len(locations))
    traits = {base["base_agent_id"]: base["traits"] for base in bases}
    population = neutral_population(traits, locations)
    history_document = _history_document(population, locations)
    write_json(destination / "neutral_history_initialization.json", history_document)
    config = _config(history_document["sha256"], seeds, timesteps)
    write_json(destination / "world_v0_3_config.json", config)
    state_path = destination / "world_states.jsonl"
    history_path = destination / "history_states.jsonl"
    action_path = destination / "agent_actions.jsonl"
    for path in (state_path, history_path, action_path):
        path.write_text("", encoding="utf-8", newline="\n")
    worlds = []
    with state_path.open("a", encoding="utf-8", newline="\n") as state_file, history_path.open(
        "a", encoding="utf-8", newline="\n"
    ) as history_file, action_path.open("a", encoding="utf-8", newline="\n") as action_file:
        for spec in CONDITIONS:
            for seed in seeds:
                _write_initial_state(state_file, history_file, spec, seed, population)
                result = simulate_world(
                    source,
                    population,
                    reinforcement_weight=float(spec["reinforcement_weight"]),
                    choice_seed_value=seed,
                    timesteps=timesteps,
                    on_step=_log_step(state_file, history_file, action_file, spec, seed),
                )
                if result["llm_calls"] != 0:
                    raise RuntimeError("experiment 003 called a model")
                worlds.append(_world_metrics(spec, seed, result, timesteps))
                print(
                    f"{spec['condition_id']} seed={seed} max_share={worlds[-1]['max_visit_share_mean']:.3f}",
                    flush=True,
                )
            if spec["condition_id"] == "R0":
                _require_r0_matches_h0(worlds, timesteps)
    replay = simulate_world(
        source,
        population,
        reinforcement_weight=150.0,
        choice_seed_value=seeds[0],
        timesteps=timesteps,
    )
    original = next(world for world in worlds if world["condition_id"] == "R150" and world["choice_seed"] == seeds[0])
    replay_hash = trajectory_hash(replay["series"])
    if replay_hash != original["trajectory_sha256"]:
        raise RuntimeError("R150 exact replay hash did not match")
    forks = [
        run_fork(source, population, choice_seed_value=seed, timesteps=timesteps)
        for seed in FORK_SEEDS
        if timesteps >= FORK_TIMESTEP
    ]
    for fork in forks:
        matched = next(
            world for world in worlds if world["condition_id"] == "R150" and world["choice_seed"] == fork["choice_seed"]
        )
        if trajectory_hash(fork["series_a"]) != matched["trajectory_sha256"]:
            raise RuntimeError("fork branch A diverged from the recorded R150 trajectory")
    summary = _summarize(worlds, seeds, timesteps, seeds[0], replay_hash)
    fork_summary = _summarize_forks(forks, timesteps)
    write_json(destination / "experiment_003_summary.json", summary)
    write_json(destination / "fork_summary.json", fork_summary)
    manifest = _manifest(destination, config, history_document, summary, worlds, fork_summary)
    write_json(destination / "run_manifest.json", manifest)
    return destination


def trajectory_hash(series: list[dict[str, str]]) -> str:
    ordered = [{agent_id: step[agent_id] for agent_id in sorted(step)} for step in series]
    return canonical_sha256(ordered)


def unique_favorite(visit_count: dict[str, int]) -> str | None:
    total = sum(visit_count.values())
    if total == 0:
        return None
    best = max(visit_count.values())
    winners = [place_id for place_id in PLACE_IDS if visit_count[place_id] == best]
    if len(winners) != 1:
        return None
    return winners[0]


def visit_shares(visit_count: dict[str, int]) -> dict[str, float] | None:
    total = sum(visit_count.values())
    if total == 0:
        return None
    return {place_id: visit_count[place_id] / total for place_id in PLACE_IDS}


def personal_entropy(visit_count: dict[str, int]) -> float | None:
    shares = visit_shares(visit_count)
    if shares is None:
        return None
    return -sum(share * math.log(share) for share in shares.values() if share > 0.0)


def max_visit_share(visit_count: dict[str, int]) -> float | None:
    shares = visit_shares(visit_count)
    if shares is None:
        return None
    return max(shares.values())


def _world_metrics(spec: dict[str, Any], seed: int, result: dict[str, Any], timesteps: int) -> dict[str, Any]:
    final_counts = _visit_snapshot(result["final_states"])
    diagnostics = measure_world(result["series"], moves=result["moves"], timesteps=timesteps)
    favorites = {agent_id: unique_favorite(final_counts[agent_id]) for agent_id in sorted(final_counts)}
    shares = [max_visit_share(final_counts[agent_id]) for agent_id in sorted(final_counts)]
    entropies = [personal_entropy(final_counts[agent_id]) for agent_id in sorted(final_counts)]
    repeat_rates = [
        None if result["entries"][agent_id] == 0 else result["repeats"][agent_id] / result["entries"][agent_id]
        for agent_id in sorted(result["entries"])
    ]
    return {
        "condition_id": spec["condition_id"],
        "reinforcement_weight": spec["reinforcement_weight"],
        "choice_seed": seed,
        "trajectory_sha256": trajectory_hash(result["series"]),
        "series": result["series"],
        "snapshots": result["snapshots"],
        "favorites": favorites,
        "max_visit_share_mean": _finite_mean(shares),
        "personal_entropy_mean": _finite_mean(entropies),
        "repeat_visit_probability": _finite_mean(repeat_rates),
        "return_probability_k5": _return_probability(result["series"], favorites),
        "dwell_share_by_place": _dwell_shares(result["final_states"], timesteps),
        "visit_share_by_place": _population_visit_shares(final_counts),
        "early_leader_retention": _retention(result["snapshots"], timesteps),
        "window_concentration": _window_stat(result["snapshots"], timesteps, max_visit_share),
        "window_entropy": _window_stat(result["snapshots"], timesteps, personal_entropy),
        "early_final_correlation": _share_correlation(result["snapshots"], timesteps, late_only=False),
        "early_late_correlation": _share_correlation(result["snapshots"], timesteps, late_only=True),
        "movement_rate": diagnostics["movement_rate"],
        "mean_dwell": diagnostics["mean_dwell"],
        "entropy_mean": diagnostics["entropy_mean"],
        "saturation_frequency": diagnostics["saturation_frequency"],
        "empty_frequency": diagnostics["empty_frequency"],
        "lag1": diagnostics["lag1"],
        "lag2": diagnostics["lag2"],
        "mean_occupancy": diagnostics["mean_occupancy"],
        "alternation_rate": diagnostics["site_01_site_02_alternation_rate"],
        "llm_calls": 0,
    }


def _summarize(
    worlds: list[dict[str, Any]],
    seeds: tuple[int, ...],
    timesteps: int,
    replay_seed: int,
    replay_hash: str,
) -> dict[str, Any]:
    by_condition = {
        spec["condition_id"]: [world for world in worlds if world["condition_id"] == spec["condition_id"]]
        for spec in CONDITIONS
    }
    between = {condition_id: _between_seed(rows) for condition_id, rows in by_condition.items()}
    amplification = _amplification(by_condition, between)
    return {
        "experiment_id": "experiment_003_endogenous_reinforcement",
        "note": ANALYSIS_NOTE,
        "amplification_rule": AMPLIFICATION_RULE,
        "llm_calls": 0,
        "world_count": len(worlds),
        "timesteps": timesteps,
        "choice_seeds": list(seeds),
        "r0_matches_reference_h0_first_100": True,
        "reference_h0_seeds_compared": 20,
        "familiarity_rule": FAMILIARITY_RULE,
        "neutral_history": NEUTRAL_NOTE,
        "definitions": {
            "max_visit_share": "max visit_count / sum of visit_count. Ties in the favorite are not assigned.",
            "personal_visit_entropy": "Shannon entropy in nats of the agent's visit-count shares.",
            "repeat_visit_probability": "Entries whose destination already had visit_count > 0, divided by entries.",
            "return_probability_k5": (
                "After leaving the post-hoc unique final favorite, the favorite appears in the next 5 locations. "
                "Tied favorites and departures without 5 future records are excluded."
            ),
            "early_leader": "Unique maximum visit_count over commits t=1–20. Ties stay unassigned.",
            "late_window": "Visit counts accumulated on commits t=101–500.",
        },
        "conditions": {condition_id: _condition_summary(rows) for condition_id, rows in by_condition.items()},
        "between_seed": between,
        "amplification": amplification,
        "exact_replay": {
            "condition_id": "R150",
            "choice_seed": replay_seed,
            "trajectory_sha256": replay_hash,
            "match": True,
        },
    }


def _condition_summary(worlds: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_seeds": len(worlds),
        "llm_calls": 0,
        "max_visit_share": _mean_sd([world["max_visit_share_mean"] for world in worlds]),
        "personal_visit_entropy": _mean_sd([world["personal_entropy_mean"] for world in worlds]),
        "repeat_visit_probability": _mean_sd([world["repeat_visit_probability"] for world in worlds]),
        "return_probability_k5": _mean_sd([world["return_probability_k5"] for world in worlds]),
        "early_leader_retention": _mean_sd([world["early_leader_retention"]["rate"] for world in worlds]),
        "early_leader_eligible_agents": _mean_sd(
            [float(world["early_leader_retention"]["eligible"]) for world in worlds]
        ),
        "late_share_of_early_leader": _mean_sd(
            [world["early_leader_retention"]["late_share"] for world in worlds]
        ),
        "early_vs_final_share_correlation": _mean_sd([world["early_final_correlation"] for world in worlds]),
        "early_vs_late_share_correlation": _mean_sd([world["early_late_correlation"] for world in worlds]),
        "concentration_by_window": {
            name: _mean_sd([world["window_concentration"][name] for world in worlds])
            for name in ("t01_20", "t21_100", "t101_500")
        },
        "entropy_by_window": {
            name: _mean_sd([world["window_entropy"][name] for world in worlds])
            for name in ("t01_20", "t21_100", "t101_500")
        },
        "visit_share_by_place": {
            place_id: _mean_sd([world["visit_share_by_place"][place_id] for world in worlds]) for place_id in PLACE_IDS
        },
        "place_visit_concentration": _mean_sd([max(world["visit_share_by_place"].values()) for world in worlds]),
        "dwell_share_by_place": {
            place_id: _mean_sd([world["dwell_share_by_place"][place_id] for world in worlds]) for place_id in PLACE_IDS
        },
        "movement_rate": _mean_sd([world["movement_rate"] for world in worlds]),
        "mean_dwell": _mean_sd([world["mean_dwell"] for world in worlds]),
        "occupancy_entropy": _mean_sd([world["entropy_mean"] for world in worlds]),
        "saturation_frequency": _mean_sd([world["saturation_frequency"] for world in worlds]),
        "empty_frequency": _mean_sd([world["empty_frequency"] for world in worlds]),
        "alternation_rate": _mean_sd([world["alternation_rate"] for world in worlds]),
        "mean_occupancy": {
            place_id: _mean_sd([world["mean_occupancy"][place_id] for world in worlds]) for place_id in PLACE_IDS
        },
        "lag1": {place_id: _mean_sd([world["lag1"][place_id] for world in worlds]) for place_id in PLACE_IDS},
        "lag2": {place_id: _mean_sd([world["lag2"][place_id] for world in worlds]) for place_id in PLACE_IDS},
        "final_favorite_counts": _favorite_counts(worlds),
        "between_seed_sd": {
            "max_visit_share": _stdev([world["max_visit_share_mean"] for world in worlds]),
            "personal_visit_entropy": _stdev([world["personal_entropy_mean"] for world in worlds]),
            "movement_rate": _stdev([world["movement_rate"] for world in worlds]),
            "occupancy_entropy": _stdev([world["entropy_mean"] for world in worlds]),
        },
    }


def _between_seed(worlds: list[dict[str, Any]]) -> dict[str, Any]:
    packed = [_pack(world["series"]) for world in worlds]
    occupancies = [[occupancy_of(step) for step in world["series"]] for world in worlds]
    pair_disagreement = []
    pair_tv = []
    long_disagreement = []
    long_tv = []
    final_disagreement = []
    for left, right in itertools.combinations(range(len(worlds)), 2):
        pair_disagreement.append(_tuple_disagreement(packed[left], packed[right], 1, len(packed[left]) - 1))
        pair_tv.append(_occupancy_tv(occupancies[left], occupancies[right], 1, len(occupancies[left]) - 1))
        long_disagreement.append(_tuple_disagreement(packed[left], packed[right], 101, len(packed[left]) - 1))
        long_tv.append(_occupancy_tv(occupancies[left], occupancies[right], 101, len(occupancies[left]) - 1))
        final_disagreement.append(_tuple_disagreement(packed[left], packed[right], len(packed[left]) - 1, len(packed[left]) - 1))
    return {
        "pairwise_location_disagreement": _mean_sd(pair_disagreement),
        "pairwise_occupancy_tv": _mean_sd(pair_tv),
        "long_run_location_disagreement_t101_500": _mean_sd(long_disagreement),
        "long_run_occupancy_tv_t101_500": _mean_sd(long_tv),
        "final_location_disagreement": _mean_sd(final_disagreement),
        "unique_trajectory_hashes": len({world["trajectory_sha256"] for world in worlds}),
        "unique_final_occupancy": len({tuple(occupancy_of(world["series"][-1])[place] for place in PLACE_IDS) for world in worlds}),
        "unique_final_location_vectors": len({packed_world[-1] for packed_world in packed}),
        "place_visit_share_sd": {
            place_id: _stdev([world["visit_share_by_place"][place_id] for world in worlds]) for place_id in PLACE_IDS
        },
        "final_favorite_entropy_across_seeds": _favorite_entropy(worlds),
        "max_visit_share_sd": _stdev([world["max_visit_share_mean"] for world in worlds]),
        "personal_entropy_sd": _stdev([world["personal_entropy_mean"] for world in worlds]),
        "n_pairs": len(pair_disagreement),
    }


def _amplification(by_condition: dict[str, list[dict[str, Any]]], between: dict[str, dict[str, Any]]) -> dict[str, Any]:
    retention = {key: _finite_mean([world["early_leader_retention"]["rate"] for world in rows]) for key, rows in by_condition.items()}
    concentration = {key: _finite_mean([world["max_visit_share_mean"] for world in rows]) for key, rows in by_condition.items()}
    entropy = {key: _finite_mean([world["personal_entropy_mean"] for world in rows]) for key, rows in by_condition.items()}
    divergence = {
        key: between[key]["long_run_location_disagreement_t101_500"]["mean"] for key in by_condition
    }
    checks = {
        "early_leader_retention_rises": _strictly_increasing([retention["R0"], retention["R50"], retention["R150"]]),
        "personal_concentration_rises": _strictly_increasing(
            [concentration["R0"], concentration["R50"], concentration["R150"]]
        ),
        "personal_entropy_falls": _strictly_decreasing([entropy["R0"], entropy["R50"], entropy["R150"]]),
        "between_seed_long_run_divergence_rises": _strictly_increasing(
            [divergence["R0"], divergence["R50"], divergence["R150"]]
        ),
    }
    return {
        "rule": AMPLIFICATION_RULE,
        "history_reinforcement_amplified_path_dependence": all(checks.values()),
        "checks": checks,
        "early_leader_retention": retention,
        "final_max_visit_share": concentration,
        "final_personal_entropy": entropy,
        "long_run_location_disagreement": divergence,
    }


def _summarize_forks(forks: list[dict[str, Any]], timesteps: int) -> dict[str, Any]:
    rows = [_fork_metrics(fork, timesteps) for fork in forks]
    windows = ("t21_50", "t51_100", "t101_250", "t251_500")
    return {
        "note": FORK_NOTE,
        "reinforcement_weight": 150.0,
        "fork_timestep": FORK_TIMESTEP,
        "agent_id": FORK_AGENT,
        "n_pairs": len(rows),
        "llm_calls": 0,
        "replacement_rule": (
            "The replacement is the second-highest total utility when that place is not the drawn destination; "
            "otherwise it is the highest-utility place. Place-id order breaks ties. The replacement draws no random number."
        ),
        "agent_0000_disagreement_rate": _mean_sd([row["agent_disagreement_rate"] for row in rows]),
        "occupancy_tv": _mean_sd([row["mean_occupancy_tv"] for row in rows]),
        "windows": {
            name: {
                "agent_0000_disagreement": _mean_sd([row["windows"][name]["agent_disagreement"] for row in rows]),
                "occupancy_tv": _mean_sd([row["windows"][name]["occupancy_tv"] for row in rows]),
            }
            for name in windows
        },
        "reconvergence_count": sum(1 for row in rows if row["reconvergence_timestep"] is not None),
        "reconvergence_timestep": _mean_sd(
            [float(row["reconvergence_timestep"]) for row in rows if row["reconvergence_timestep"] is not None]
        ),
        "final_favorite_differs_count": sum(1 for row in rows if row["final_favorite_differs"] is True),
        "final_favorite_comparable_count": sum(1 for row in rows if row["final_favorite_differs"] is not None),
        "trajectory_hash_equal_count": sum(1 for row in rows if row["trajectory_hash_equal"]),
        "agent_disagreement_by_timestep_mean": [
            statistics.fmean([row["agent_disagreement_by_timestep"][index] for row in rows])
            for index in range(len(rows[0]["agent_disagreement_by_timestep"]))
        ]
        if rows
        else [],
        "occupancy_tv_by_timestep_mean": [
            statistics.fmean([row["occupancy_tv_by_timestep"][index] for row in rows])
            for index in range(len(rows[0]["occupancy_tv_by_timestep"]))
        ]
        if rows
        else [],
        "pairs": [
            {key: value for key, value in row.items() if key not in {"agent_disagreement_by_timestep", "occupancy_tv_by_timestep"}}
            for row in rows
        ],
    }


def _fork_metrics(fork: dict[str, Any], timesteps: int) -> dict[str, Any]:
    series_a = fork["series_a"]
    series_b = fork["series_b"]
    agent_curve = [
        1.0 if series_a[index][FORK_AGENT] != series_b[index][FORK_AGENT] else 0.0 for index in range(1, timesteps + 1)
    ]
    tv_curve = [
        _one_tv(occupancy_of(series_a[index]), occupancy_of(series_b[index]), len(series_a[0]))
        for index in range(1, timesteps + 1)
    ]
    favorite_a = unique_favorite(fork["final_a"][FORK_AGENT].visit_count)
    favorite_b = unique_favorite(fork["final_b"][FORK_AGENT].visit_count)
    differs: bool | None
    if favorite_a is None or favorite_b is None:
        differs = None
    else:
        differs = favorite_a != favorite_b
    windows = {
        "t21_50": (21, 50),
        "t51_100": (51, 100),
        "t101_250": (101, 250),
        "t251_500": (251, 500),
    }
    return {
        "choice_seed": fork["choice_seed"],
        "original_destination": fork["original_destination"],
        "replacement_destination": fork["replacement_destination"],
        "trajectory_sha256_branch_a": trajectory_hash(series_a),
        "trajectory_sha256_branch_b": trajectory_hash(series_b),
        "trajectory_hash_equal": trajectory_hash(series_a) == trajectory_hash(series_b),
        "agent_disagreement_rate": statistics.fmean(agent_curve[FORK_TIMESTEP:]),
        "mean_occupancy_tv": statistics.fmean(tv_curve[FORK_TIMESTEP:]),
        "reconvergence_timestep": _reconvergence(series_a, series_b),
        "final_favorite_branch_a": favorite_a,
        "final_favorite_branch_b": favorite_b,
        "final_favorite_differs": differs,
        "windows": {
            name: {
                "agent_disagreement": statistics.fmean(agent_curve[start - 1 : stop]),
                "occupancy_tv": statistics.fmean(tv_curve[start - 1 : stop]),
            }
            for name, (start, stop) in windows.items()
            if stop <= timesteps
        },
        "agent_disagreement_by_timestep": agent_curve,
        "occupancy_tv_by_timestep": tv_curve,
        "llm_calls": 0,
    }


def _reconvergence(series_a: list[dict[str, str]], series_b: list[dict[str, str]]) -> int | None:
    agents = sorted(series_a[0])
    last_difference = None
    for index in range(FORK_TIMESTEP + 1, len(series_a)):
        if any(series_a[index][agent_id] != series_b[index][agent_id] for agent_id in agents):
            last_difference = index
    if last_difference is None:
        return FORK_TIMESTEP + 1
    if last_difference == len(series_a) - 1:
        return None
    return last_difference + 1


def _retention(snapshots: dict[int, dict[str, dict[str, int]]], timesteps: int) -> dict[str, Any]:
    if 20 not in snapshots or timesteps not in snapshots or 100 not in snapshots:
        return {"rate": None, "eligible": 0, "retained": 0, "late_share": None}
    early = snapshots[20]
    final = snapshots[timesteps]
    late_counts = _subtract_counts(snapshots[timesteps], snapshots[100])
    eligible = 0
    retained = 0
    late_shares = []
    for agent_id in sorted(early):
        leader = unique_favorite(early[agent_id])
        if leader is None:
            continue
        eligible += 1
        if unique_favorite(final[agent_id]) == leader:
            retained += 1
        late = visit_shares(late_counts[agent_id])
        if late is not None:
            late_shares.append(late[leader])
    return {
        "rate": None if eligible == 0 else retained / eligible,
        "eligible": eligible,
        "retained": retained,
        "late_share": _finite_mean(late_shares),
    }


def _window_stat(snapshots: dict[int, dict[str, dict[str, int]]], timesteps: int, metric: Any) -> dict[str, float | None]:
    if not {20, 100, timesteps}.issubset(snapshots):
        return {"t01_20": None, "t21_100": None, "t101_500": None}
    windows = {
        "t01_20": snapshots[20],
        "t21_100": _subtract_counts(snapshots[100], snapshots[20]),
        "t101_500": _subtract_counts(snapshots[timesteps], snapshots[100]),
    }
    return {
        name: _finite_mean([metric(counts[agent_id]) for agent_id in sorted(counts)])
        for name, counts in windows.items()
    }


def _share_correlation(snapshots: dict[int, dict[str, dict[str, int]]], timesteps: int, *, late_only: bool) -> float | None:
    if 20 not in snapshots or timesteps not in snapshots:
        return None
    early = snapshots[20]
    target_counts = _subtract_counts(snapshots[timesteps], snapshots[100]) if late_only else snapshots[timesteps]
    if late_only and 100 not in snapshots:
        return None
    left = []
    right = []
    for agent_id in sorted(early):
        early_share = visit_shares(early[agent_id])
        final_share = visit_shares(target_counts[agent_id])
        if early_share is None or final_share is None:
            continue
        for place_id in PLACE_IDS:
            left.append(early_share[place_id])
            right.append(final_share[place_id])
    return _pearson(left, right)


def _return_probability(series: list[dict[str, str]], favorites: dict[str, str | None]) -> float | None:
    returned = 0
    events = 0
    for agent_id, favorite in favorites.items():
        if favorite is None:
            continue
        locations = [step[agent_id] for step in series]
        for index in range(len(locations) - 1):
            if locations[index] != favorite or locations[index + 1] == favorite:
                continue
            future = locations[index + 2 : index + 2 + RETURN_WINDOW]
            if len(future) < RETURN_WINDOW:
                continue
            events += 1
            if favorite in future:
                returned += 1
    if events == 0:
        return None
    return returned / events


def _require_r0_matches_h0(worlds: list[dict[str, Any]], timesteps: int) -> None:
    reference = _load_h0_locations()
    compared = 0
    for world in worlds:
        if world["condition_id"] != "R0" or world["choice_seed"] not in reference:
            continue
        expected = reference[world["choice_seed"]]
        actual = world["series"][:101]
        if len(expected) < 101 or timesteps < 100:
            raise RuntimeError("R0/H0 comparison needs 100 timesteps")
        for timestep, (left, right) in enumerate(zip(actual, expected, strict=True)):
            if left != right:
                raise RuntimeError(
                    f"R0 diverges from World v0.2 H0 at choice seed {world['choice_seed']} timestep {timestep}"
                )
        compared += 1
    if compared != 20:
        raise RuntimeError("R0/H0 comparison did not cover choice seeds 0–19")


def _load_h0_locations() -> dict[int, list[dict[str, str]]]:
    grouped: dict[int, list[tuple[int, dict[str, str]]]] = {seed: [] for seed in range(20)}
    path = WORLD_V02_DIR / "world_states.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("condition_id") != "H0" or int(row["choice_seed"]) not in grouped:
                continue
            grouped[int(row["choice_seed"])].append((int(row["timestep"]), {str(key): str(value) for key, value in row["locations"].items()}))
    loaded = {}
    for seed, rows in grouped.items():
        rows.sort(key=lambda item: item[0])
        loaded[seed] = [locations for _timestep, locations in rows[:101]]
    return loaded


def _history_document(states: dict[str, AgentHistory], locations: dict[str, str]) -> dict[str, Any]:
    payload = {
        "algorithm": "worldlab.neutral_history.v1",
        "note": NEUTRAL_NOTE,
        "visit_count": 0,
        "cumulative_dwell": 0,
        "time_since_last_visit": None,
        "previous_location": None,
        "initial_locations": locations,
        "agents": {
            agent_id: {
                "current_location": states[agent_id].current_location,
                "previous_location": states[agent_id].previous_location,
                "visit_count": states[agent_id].visit_count,
                "cumulative_dwell": states[agent_id].cumulative_dwell,
                "time_since_last_visit": states[agent_id].time_since_last_visit,
            }
            for agent_id in sorted(states)
        },
    }
    payload["sha256"] = canonical_sha256(payload)
    return payload


def _config(history_sha256: str, seeds: tuple[int, ...], timesteps: int) -> dict[str, Any]:
    atlas_manifest = json.loads((ATLAS_DIR / "atlas_manifest.json").read_text(encoding="utf-8"))
    payload = {
        "world_id": "world_v0_3",
        "experiment_id": "experiment_003_endogenous_reinforcement",
        "utility_source": "utility_atlas_v1",
        "utility_atlas_sha256": atlas_manifest["sha256"],
        "prompt_version": "independent_utility_v3",
        "choice_temperature": REFERENCE_CHOICE_TEMPERATURE,
        "reconsideration_rate": REFERENCE_RECONSIDERATION_RATE,
        "choice_seeds": list(seeds),
        "timesteps": timesteps,
        "conditions": [dict(spec) for spec in CONDITIONS],
        "familiarity_rule": FAMILIARITY_RULE,
        "neutral_history_sha256": history_sha256,
        "initial_state_source": str(WORLD_V01_DIR / "initial_state.json"),
        "fork": {
            "agent_id": FORK_AGENT,
            "timestep": FORK_TIMESTEP,
            "seeds": list(FORK_SEEDS),
            "reinforcement_weight": 150.0,
        },
        "llm_calls": 0,
        "decision_signal": "overall_utility_plus_familiarity",
    }
    payload["config_sha256"] = canonical_sha256(payload)
    return payload


def _manifest(
    destination: Path,
    config: dict[str, Any],
    history_document: dict[str, Any],
    summary: dict[str, Any],
    worlds: list[dict[str, Any]],
    fork_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "experiment_id": "experiment_003_endogenous_reinforcement",
        "llm_calls": 0,
        "world_count": len(worlds),
        "complete": len(worlds) == len(CONDITIONS) * len(config["choice_seeds"]),
        "utility_atlas_sha256": config["utility_atlas_sha256"],
        "config_sha256": config["config_sha256"],
        "neutral_history_sha256": history_document["sha256"],
        "exact_replay": summary["exact_replay"],
        "fork_pairs": fork_summary["n_pairs"],
        "trajectory_sha256": {
            f"{world['condition_id']}:{world['choice_seed']}": world["trajectory_sha256"] for world in worlds
        },
        "files": {
            name: hashlib.sha256((destination / name).read_bytes()).hexdigest()
            for name in (
                "world_v0_3_config.json",
                "neutral_history_initialization.json",
                "world_states.jsonl",
                "history_states.jsonl",
                "agent_actions.jsonl",
                "experiment_003_summary.json",
                "fork_summary.json",
            )
        },
    }


def _log_step(state_file: Any, history_file: Any, action_file: Any, spec: dict[str, Any], seed: int) -> Any:
    def write(logged: int, locations: dict[str, str], states: dict[str, AgentHistory], actions: list[dict[str, Any]]) -> None:
        _dump(
            state_file,
            {
                "condition_id": spec["condition_id"],
                "choice_seed": seed,
                "reinforcement_weight": spec["reinforcement_weight"],
                "timestep": logged,
                "locations": locations,
                "occupancy": occupancy_of(locations),
                "llm_calls": 0,
            },
        )
        _dump(
            history_file,
            {
                "condition_id": spec["condition_id"],
                "choice_seed": seed,
                "timestep": logged,
                "agents": _history_snapshot(states),
                "llm_calls": 0,
            },
        )
        for action in actions:
            stored = {key: value for key, value in action.items() if key != "visit_count_before"}
            _dump(action_file, {"condition_id": spec["condition_id"], "choice_seed": seed, **stored})

    return write


def _write_initial_state(state_file: Any, history_file: Any, spec: dict[str, Any], seed: int, states: dict[str, AgentHistory]) -> None:
    locations = {agent_id: states[agent_id].current_location for agent_id in sorted(states)}
    _dump(
        state_file,
        {
            "condition_id": spec["condition_id"],
            "choice_seed": seed,
            "reinforcement_weight": spec["reinforcement_weight"],
            "timestep": 0,
            "locations": locations,
            "occupancy": occupancy_of(locations),
            "llm_calls": 0,
        },
    )
    _dump(
        history_file,
        {
            "condition_id": spec["condition_id"],
            "choice_seed": seed,
            "timestep": 0,
            "agents": _history_snapshot(states),
            "llm_calls": 0,
        },
    )


def _history_snapshot(states: dict[str, AgentHistory]) -> dict[str, Any]:
    return {
        agent_id: {
            "current_location": states[agent_id].current_location,
            "previous_location": states[agent_id].previous_location,
            "visit_count": states[agent_id].visit_count,
            "cumulative_dwell": states[agent_id].cumulative_dwell,
            "time_since_last_visit": states[agent_id].time_since_last_visit,
        }
        for agent_id in sorted(states)
    }


def _visit_snapshot(states: dict[str, AgentHistory]) -> dict[str, dict[str, int]]:
    return {agent_id: dict(states[agent_id].visit_count) for agent_id in sorted(states)}


def _dwell_shares(states: dict[str, AgentHistory], timesteps: int) -> dict[str, float]:
    agents = list(states)
    return {
        place_id: statistics.fmean([states[agent_id].cumulative_dwell[place_id] / timesteps for agent_id in agents])
        for place_id in PLACE_IDS
    }


def _population_visit_shares(counts: dict[str, dict[str, int]]) -> dict[str, float]:
    totals = {place_id: sum(agent[place_id] for agent in counts.values()) for place_id in PLACE_IDS}
    total = sum(totals.values())
    if total == 0:
        return {place_id: 0.0 for place_id in PLACE_IDS}
    return {place_id: totals[place_id] / total for place_id in PLACE_IDS}


def _subtract_counts(
    later: dict[str, dict[str, int]],
    earlier: dict[str, dict[str, int]],
) -> dict[str, dict[str, int]]:
    return {
        agent_id: {place_id: later[agent_id][place_id] - earlier[agent_id][place_id] for place_id in PLACE_IDS}
        for agent_id in later
    }


def _favorite_counts(worlds: list[dict[str, Any]]) -> dict[str, int]:
    counts = {place_id: 0 for place_id in PLACE_IDS}
    counts["tie"] = 0
    for world in worlds:
        for favorite in world["favorites"].values():
            counts["tie" if favorite is None else favorite] += 1
    return counts


def _favorite_entropy(worlds: list[dict[str, Any]]) -> float | None:
    agents = sorted(worlds[0]["favorites"])
    entropies = []
    for agent_id in agents:
        labels = ["tie" if world["favorites"][agent_id] is None else str(world["favorites"][agent_id]) for world in worlds]
        entropies.append(_category_entropy(labels))
    return _finite_mean(entropies)


def _category_entropy(labels: list[str]) -> float:
    total = len(labels)
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return -sum((count / total) * math.log(count / total) for count in counts.values())


def _pack(series: list[dict[str, str]]) -> list[tuple[str, ...]]:
    agents = sorted(series[0])
    return [tuple(step[agent_id] for agent_id in agents) for step in series]


def _tuple_disagreement(left: list[tuple[str, ...]], right: list[tuple[str, ...]], start: int, stop: int) -> float:
    compared = 0
    mismatches = 0
    for lhs, rhs in zip(left[start : stop + 1], right[start : stop + 1], strict=True):
        compared += len(lhs)
        mismatches += sum(1 for item, other in zip(lhs, rhs, strict=True) if item != other)
    return mismatches / compared


def _occupancy_tv(left: list[dict[str, int]], right: list[dict[str, int]], start: int, stop: int) -> float:
    values = [
        _one_tv(lhs, rhs, sum(lhs.values()))
        for lhs, rhs in zip(left[start : stop + 1], right[start : stop + 1], strict=True)
    ]
    return statistics.fmean(values)


def _one_tv(left: dict[str, int], right: dict[str, int], population: int) -> float:
    change = sum(abs(right[place_id] - left[place_id]) for place_id in PLACE_IDS)
    return 0.5 * change / population


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    mean_left = statistics.fmean(left)
    mean_right = statistics.fmean(right)
    numerator = sum((item - mean_left) * (other - mean_right) for item, other in zip(left, right, strict=True))
    left_scale = math.sqrt(sum((item - mean_left) ** 2 for item in left))
    right_scale = math.sqrt(sum((item - mean_right) ** 2 for item in right))
    if left_scale == 0.0 or right_scale == 0.0:
        return None
    return numerator / (left_scale * right_scale)


def _strictly_increasing(values: list[float | None]) -> bool:
    if any(value is None for value in values):
        return False
    return bool(values[0] < values[1] < values[2])


def _strictly_decreasing(values: list[float | None]) -> bool:
    if any(value is None for value in values):
        return False
    return bool(values[0] > values[1] > values[2])


def _finite_mean(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    if not finite:
        return None
    return statistics.fmean(finite)


def _mean_sd(values: list[float | None]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if value is not None]
    return {"mean": None if not finite else statistics.fmean(finite), "sd": _stdev(finite), "n": len(finite)}


def _stdev(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return float(statistics.stdev(values))


def _refuse_frozen_directories(destination: Path) -> None:
    forbidden = {
        WORLD_V01_DIR.resolve(),
        WORLD_V02_DIR.resolve(),
        ATLAS_DIR.resolve(),
        CALIBRATION_DIR.resolve(),
    }
    if destination.resolve() in forbidden:
        raise RuntimeError("experiment 003 must not write into frozen world outputs")


def _dump(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")


def main() -> None:
    run_experiment()


if __name__ == "__main__":
    if __package__ is None:
        raise SystemExit("請在 worldlab 目錄執行：python -m world.experiment_003")
    main()
