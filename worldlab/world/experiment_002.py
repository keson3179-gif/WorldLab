"""Experiment 002: history dependence under a fixed reference regime.

Utilities come from the frozen Utility Atlas. Familiarity is added in Python.
This module does not call a model and does not write into World v0.1,
the atlas, or Dynamics Calibration 001.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from agents.persona_generator import write_json
from config.settings import PROJECT_ROOT
from experiments.experiment_001 import load_base_traits
from experiments.reproducibility_check import canonical_sha256
from world.choice import agent_reconsiders, choice_seed, softmax_choice
from world.dynamics_calibration import load_initial_locations, measure_world
from world.geography import PLACE_IDS, distance_matrix, occupancy_of, place_index, travel_distance
from world.history import (
    FAMILIARITY_RULE,
    FAVORITE_ALGORITHM,
    FAVORITE_CUMULATIVE_DWELL,
    FAVORITE_SEED,
    FAVORITE_TIME_SINCE,
    FAVORITE_VISIT_COUNT,
    MANIPULATION_NOTE,
    OTHER_CUMULATIVE_DWELL,
    OTHER_TIME_SINCE,
    OTHER_VISIT_COUNT,
    SHUFFLE_ALGORITHM,
    SHUFFLE_SEED,
    AgentHistory,
    assign_favorites,
    clone_population,
    commit_history,
    decision_utility,
    familiarity_scores,
    initial_history,
    shuffle_favorites,
)
from world.simulation import OUTPUT_DIR as WORLD_V01_DIR
from world.utility_atlas import OUTPUT_DIR as ATLAS_DIR
from world.utility_atlas import FrozenUtilityAtlas, load_frozen_atlas

OUTPUT_DIR = PROJECT_ROOT / "data" / "worlds" / "world_v0_2"
CALIBRATION_DIR = PROJECT_ROOT / "data" / "worlds" / "dynamics_calibration_001"
REFERENCE_CHOICE_TEMPERATURE = 200.0
REFERENCE_RECONSIDERATION_RATE = 0.5
CHOICE_SEEDS = tuple(range(20))
TIMESTEPS = 100
RETURN_WINDOW = 5
CONDITIONS = (
    {"condition_id": "H0", "familiarity_weight": 0.0, "history": "preloaded"},
    {"condition_id": "H1", "familiarity_weight": 50.0, "history": "preloaded"},
    {"condition_id": "H2", "familiarity_weight": 150.0, "history": "preloaded"},
    {"condition_id": "H1_shuffled", "familiarity_weight": 50.0, "history": "shuffled"},
)
ANALYSIS_NOTE = (
    "Descriptive effect sizes for a controlled history manipulation. "
    "This summary does not claim human realism or emergence."
)


def simulate_condition(
    atlas: FrozenUtilityAtlas,
    states: dict[str, AgentHistory],
    favorites: dict[str, str],
    *,
    familiarity_weight: float,
    choice_seed_value: int,
    timesteps: int = TIMESTEPS,
) -> dict[str, Any]:
    matrix = distance_matrix()
    places = place_index()
    current = clone_population(states)
    series = [{agent_id: current[agent_id].current_location for agent_id in sorted(current)}]
    history_series = [_history_snapshot(current)]
    actions = []
    moves = 0
    for timestep in range(timesteps):
        occupancy = occupancy_of({agent_id: current[agent_id].current_location for agent_id in current})
        decisions: dict[str, str] = {}
        step_actions = []
        for agent_id in sorted(current):
            state = current[agent_id]
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
                totals[place_id] = decision_utility(base, scores[place_id], familiarity_weight)
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
            step_actions.append(
                {
                    "timestep": timestep + 1,
                    "agent_id": agent_id,
                    "origin": origin,
                    "destination": destination,
                    "stayed": destination == origin,
                    "reconsidered": reconsidered,
                    "historical_favorite": favorites[agent_id],
                    "familiarity_weight": familiarity_weight,
                    "base_utility": bases,
                    "familiarity_score": scores,
                    "total_utility": totals,
                    "choice_probabilities": {place_id: probabilities[place_id] for place_id in PLACE_IDS},
                    "occupancy_before": occupancy,
                    "llm_calls": 0,
                }
            )
        current = commit_history(current, decisions)
        moves += sum(1 for agent_id in decisions if decisions[agent_id] != series[-1][agent_id])
        series.append({agent_id: current[agent_id].current_location for agent_id in sorted(current)})
        history_series.append(_history_snapshot(current))
        actions.extend(step_actions)
    return {
        "series": series,
        "history": history_series,
        "actions": actions,
        "moves": moves,
        "final_states": current,
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
    favorites = assign_favorites(list(locations), FAVORITE_SEED)
    shuffled = shuffle_favorites(favorites, SHUFFLE_SEED)
    preloaded = _population(traits, locations, favorites)
    shuffled_population = _population(traits, locations, shuffled)
    history_document = _history_document(favorites, shuffled, preloaded, locations)
    write_json(destination / "history_initialization.json", history_document)
    config = _config(history_document["sha256"], seeds, timesteps)
    write_json(destination / "world_v0_2_config.json", config)
    worlds = []
    state_path = destination / "world_states.jsonl"
    history_path = destination / "history_states.jsonl"
    action_path = destination / "agent_actions.jsonl"
    for path in (state_path, history_path, action_path):
        path.write_text("", encoding="utf-8", newline="\n")
    for spec in CONDITIONS:
        population = preloaded if spec["history"] == "preloaded" else shuffled_population
        favorite_map = favorites if spec["history"] == "preloaded" else shuffled
        for seed in seeds:
            result = simulate_condition(
                source,
                population,
                favorite_map,
                familiarity_weight=float(spec["familiarity_weight"]),
                choice_seed_value=seed,
                timesteps=timesteps,
            )
            if result["llm_calls"] != 0:
                raise RuntimeError("experiment 002 called a model")
            record = _world_record(spec, seed, favorite_map, result, timesteps)
            worlds.append(record)
            _append_jsonl(state_path, _state_lines(spec, seed, result))
            _append_jsonl(history_path, _history_lines(spec, seed, result))
            _append_jsonl(action_path, _action_lines(spec, seed, result["actions"]))
            print(
                f"{spec['condition_id']} seed={seed} favorite_choice={record['favorite_choice_rate']:.3f}",
                flush=True,
            )
    replay_seed = seeds[0]
    replay = simulate_condition(
        source,
        preloaded,
        favorites,
        familiarity_weight=50.0,
        choice_seed_value=replay_seed,
        timesteps=timesteps,
    )
    original = next(world for world in worlds if world["condition_id"] == "H1" and world["choice_seed"] == replay_seed)
    replay_hash = _trajectory_hash(replay["series"])
    replay_match = replay_hash == original["trajectory_sha256"]
    if not replay_match:
        raise RuntimeError("H1 exact replay hash did not match")
    summary = _summarize(worlds, seeds, timesteps, replay_seed, replay_hash, replay_match)
    write_json(destination / "experiment_002_summary.json", summary)
    manifest = _manifest(destination, config, history_document, summary, worlds)
    write_json(destination / "run_manifest.json", manifest)
    return destination


def favorite_choice_rate(series: list[dict[str, str]], favorites: dict[str, str]) -> float:
    chosen = series[1:]
    hits = sum(1 for step in chosen for agent_id, place_id in step.items() if place_id == favorites[agent_id])
    return hits / (len(chosen) * len(series[0]))


def favorite_dwell_share(series: list[dict[str, str]], favorites: dict[str, str]) -> float:
    return favorite_choice_rate(series, favorites)


def return_probability(series: list[dict[str, str]], favorites: dict[str, str], window: int = RETURN_WINDOW) -> float | None:
    returned = 0
    events = 0
    for agent_id in sorted(series[0]):
        favorite = favorites[agent_id]
        locations = [step[agent_id] for step in series]
        for index in range(len(locations) - 1):
            if locations[index] != favorite or locations[index + 1] == favorite:
                continue
            future = locations[index + 2 : index + 2 + window]
            if len(future) < window:
                continue
            events += 1
            if favorite in future:
                returned += 1
    if events == 0:
        return None
    return returned / events


def trajectory_divergence(left: list[dict[str, str]], right: list[dict[str, str]]) -> dict[str, Any]:
    agents = sorted(left[0])
    compared = 0
    disagreements = 0
    tvs = []
    for left_step, right_step in zip(left[1:], right[1:], strict=True):
        for agent_id in agents:
            compared += 1
            if left_step[agent_id] != right_step[agent_id]:
                disagreements += 1
        tvs.append(_total_variation(occupancy_of(left_step), occupancy_of(right_step), len(agents)))
    return {
        "location_disagreement_rate": disagreements / compared,
        "mean_occupancy_tv": statistics.fmean(tvs),
        "cumulative_occupancy_tv": sum(tvs),
        "occupancy_tv_by_timestep": tvs,
        "trajectory_hash_equal": _trajectory_hash(left) == _trajectory_hash(right),
    }


def window_metrics(
    series: list[dict[str, str]],
    favorites: dict[str, str],
    other: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    windows = {"t01_20": (1, 20), "t21_50": (21, 50), "t51_100": (51, 100)}
    report = {}
    for name, (start, stop) in windows.items():
        chunk = [series[0], *series[start : stop + 1]]
        item: dict[str, Any] = {
            "favorite_choice_rate": favorite_choice_rate(chunk, favorites),
            "favorite_dwell_share": favorite_dwell_share(chunk, favorites),
        }
        if other is not None:
            item["occupancy_tv"] = statistics.fmean(
                [
                    _total_variation(occupancy_of(series[index]), occupancy_of(other[index]), len(series[0]))
                    for index in range(start, stop + 1)
                ]
            )
        report[name] = item
    return report


def _population(traits: dict[str, Any], locations: dict[str, str], favorites: dict[str, str]) -> dict[str, AgentHistory]:
    return {
        agent_id: initial_history(agent_id, traits[agent_id], locations[agent_id], favorites[agent_id])
        for agent_id in sorted(locations)
    }


def _world_record(
    spec: dict[str, Any],
    seed: int,
    favorites: dict[str, str],
    result: dict[str, Any],
    timesteps: int,
) -> dict[str, Any]:
    series = result["series"]
    diagnostics = measure_world(series, moves=result["moves"], timesteps=timesteps)
    return {
        "condition_id": spec["condition_id"],
        "familiarity_weight": spec["familiarity_weight"],
        "history": spec["history"],
        "choice_seed": seed,
        "favorite_choice_rate": favorite_choice_rate(series, favorites),
        "favorite_dwell_share": favorite_dwell_share(series, favorites),
        "return_probability": return_probability(series, favorites),
        "trajectory_sha256": _trajectory_hash(series),
        "series": series,
        "favorites": favorites,
        "movement_rate": diagnostics["movement_rate"],
        "mean_dwell": diagnostics["mean_dwell"],
        "entropy_mean": diagnostics["entropy_mean"],
        "saturation_frequency": diagnostics["saturation_frequency"],
        "empty_frequency": diagnostics["empty_frequency"],
        "lag1": diagnostics["lag1"],
        "lag2": diagnostics["lag2"],
        "alternation_rate": diagnostics["site_01_site_02_alternation_rate"],
        "llm_calls": 0,
    }


def _summarize(
    worlds: list[dict[str, Any]],
    seeds: tuple[int, ...],
    timesteps: int,
    replay_seed: int,
    replay_hash: str,
    replay_match: bool,
) -> dict[str, Any]:
    by_condition = {spec["condition_id"]: [world for world in worlds if world["condition_id"] == spec["condition_id"]] for spec in CONDITIONS}
    pairs = (("H0", "H1"), ("H0", "H2"), ("H1", "H2"), ("H1", "H1_shuffled"))
    divergences = {f"{left}_vs_{right}": _paired_divergence(by_condition[left], by_condition[right]) for left, right in pairs}
    windows = {
        spec["condition_id"]: _paired_windows(by_condition["H0"], by_condition[spec["condition_id"]])
        for spec in CONDITIONS
        if spec["condition_id"] != "H0"
    }
    return {
        "experiment_id": "experiment_002_familiarity",
        "note": ANALYSIS_NOTE,
        "llm_calls": 0,
        "world_count": len(worlds),
        "timesteps": timesteps,
        "choice_seeds": list(seeds),
        "reference_regime": {
            "choice_temperature": REFERENCE_CHOICE_TEMPERATURE,
            "reconsideration_rate": REFERENCE_RECONSIDERATION_RATE,
        },
        "familiarity_rule": FAMILIARITY_RULE,
        "return_window": RETURN_WINDOW,
        "return_definition": (
            "A departure is a transition from the historical favorite to another place. "
            "A return is the favorite appearing in the next 5 location records after that first away timestep. "
            "Departures without 5 future records are excluded."
        ),
        "dwell_definition": (
            "Favorite dwell share uses the committed location after each timestep. "
            "That location is also the recorded choice, so choice rate and dwell share match."
        ),
        "conditions": {condition_id: _condition_summary(rows) for condition_id, rows in by_condition.items()},
        "divergence": divergences,
        "divergence_by_window": windows,
        "shuffled_control": _shuffled_control(by_condition),
        "exact_replay": {"condition_id": "H1", "choice_seed": replay_seed, "trajectory_sha256": replay_hash, "match": replay_match},
    }


def _condition_summary(worlds: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_seeds": len(worlds),
        "llm_calls": 0,
        "favorite_choice_rate": _mean_sd([world["favorite_choice_rate"] for world in worlds]),
        "favorite_dwell_share": _mean_sd([world["favorite_dwell_share"] for world in worlds]),
        "return_probability": _mean_sd([world["return_probability"] for world in worlds]),
        "movement_rate": _mean_sd([world["movement_rate"] for world in worlds]),
        "mean_dwell": _mean_sd([world["mean_dwell"] for world in worlds]),
        "entropy_mean": _mean_sd([world["entropy_mean"] for world in worlds]),
        "saturation_frequency": _mean_sd([world["saturation_frequency"] for world in worlds]),
        "empty_frequency": _mean_sd([world["empty_frequency"] for world in worlds]),
        "alternation_rate": _mean_sd([world["alternation_rate"] for world in worlds]),
        "lag1": {place_id: _mean_sd([world["lag1"][place_id] for world in worlds]) for place_id in PLACE_IDS},
        "lag2": {place_id: _mean_sd([world["lag2"][place_id] for world in worlds]) for place_id in PLACE_IDS},
        "between_seed_sd": {
            "favorite_choice_rate": _stdev([world["favorite_choice_rate"] for world in worlds]),
            "return_probability": _stdev([world["return_probability"] for world in worlds if world["return_probability"] is not None]),
            "movement_rate": _stdev([world["movement_rate"] for world in worlds]),
            "entropy_mean": _stdev([world["entropy_mean"] for world in worlds]),
        },
    }


def _paired_divergence(left_worlds: list[dict[str, Any]], right_worlds: list[dict[str, Any]]) -> dict[str, Any]:
    paired = []
    for left in left_worlds:
        right = next(world for world in right_worlds if world["choice_seed"] == left["choice_seed"])
        item = trajectory_divergence(left["series"], right["series"])
        item["choice_seed"] = left["choice_seed"]
        item["windows"] = window_metrics(left["series"], left["favorites"], right["series"])
        paired.append(item)
    return {
        "location_disagreement_rate": _mean_sd([item["location_disagreement_rate"] for item in paired]),
        "mean_occupancy_tv": _mean_sd([item["mean_occupancy_tv"] for item in paired]),
        "cumulative_occupancy_tv": _mean_sd([item["cumulative_occupancy_tv"] for item in paired]),
        "trajectory_hash_equal_count": sum(1 for item in paired if item["trajectory_hash_equal"]),
        "n_seeds": len(paired),
        "occupancy_tv_by_timestep_mean": [
            statistics.fmean([item["occupancy_tv_by_timestep"][index] for item in paired])
            for index in range(len(paired[0]["occupancy_tv_by_timestep"]))
        ],
    }


def _paired_windows(baseline: list[dict[str, Any]], comparison: list[dict[str, Any]]) -> dict[str, Any]:
    names = ("t01_20", "t21_50", "t51_100")
    collected: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
    for left in comparison:
        right = next(world for world in baseline if world["choice_seed"] == left["choice_seed"])
        windows = window_metrics(left["series"], left["favorites"], right["series"])
        for name in names:
            collected[name].append(windows[name])
    return {
        name: {
            "favorite_choice_rate": _mean_sd([item["favorite_choice_rate"] for item in rows]),
            "favorite_dwell_share": _mean_sd([item["favorite_dwell_share"] for item in rows]),
            "occupancy_tv_versus_h0": _mean_sd([item["occupancy_tv"] for item in rows]),
        }
        for name, rows in collected.items()
    }


def _shuffled_control(by_condition: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    h0 = statistics.fmean([world["favorite_choice_rate"] for world in by_condition["H0"]])
    h1 = statistics.fmean([world["favorite_choice_rate"] for world in by_condition["H1"]])
    shuffled = statistics.fmean([world["favorite_choice_rate"] for world in by_condition["H1_shuffled"]])
    original_under_shuffle = []
    for world in by_condition["H1_shuffled"]:
        matched = next(item for item in by_condition["H1"] if item["choice_seed"] == world["choice_seed"])
        original_under_shuffle.append(favorite_choice_rate(world["series"], matched["favorites"]))
    return {
        "h1_favorite_choice_rate": h1,
        "h1_shuffled_favorite_choice_rate": shuffled,
        "h0_favorite_choice_rate": h0,
        "h1_minus_h0": h1 - h0,
        "shuffled_minus_h0": shuffled - h0,
        "shuffled_choice_rate_for_original_favorites": _mean_sd(original_under_shuffle),
        "interpretation": (
            "H1_shuffled keeps each agent's visit-count shape and permutes which place carries it. "
            "The choice rate is computed against that agent's own shuffled favorite. "
            "The original-favorite rate asks whether the boost stays on the unshuffled place."
        ),
    }


def _history_document(
    favorites: dict[str, str],
    shuffled: dict[str, str],
    states: dict[str, AgentHistory],
    locations: dict[str, str],
) -> dict[str, Any]:
    payload = {
        "algorithm": FAVORITE_ALGORITHM,
        "shuffle_algorithm": SHUFFLE_ALGORITHM,
        "favorite_seed": FAVORITE_SEED,
        "shuffle_seed": SHUFFLE_SEED,
        "manipulation_note": MANIPULATION_NOTE,
        "favorite_place": {
            "visit_count": FAVORITE_VISIT_COUNT,
            "cumulative_dwell": FAVORITE_CUMULATIVE_DWELL,
            "time_since_last_visit": FAVORITE_TIME_SINCE,
        },
        "other_places": {
            "visit_count": OTHER_VISIT_COUNT,
            "cumulative_dwell": OTHER_CUMULATIVE_DWELL,
            "time_since_last_visit": OTHER_TIME_SINCE,
        },
        "favorites": favorites,
        "shuffled_favorites": shuffled,
        "initial_locations": locations,
        "previous_location": None,
        "agents": {agent_id: _history_snapshot(states)[agent_id] for agent_id in sorted(states)},
    }
    payload["sha256"] = canonical_sha256(payload)
    return payload


def _config(history_sha256: str, seeds: tuple[int, ...], timesteps: int) -> dict[str, Any]:
    atlas_manifest = json.loads((ATLAS_DIR / "atlas_manifest.json").read_text(encoding="utf-8"))
    payload = {
        "world_id": "world_v0_2",
        "experiment_id": "experiment_002_familiarity",
        "utility_source": "utility_atlas_v1",
        "utility_atlas_sha256": atlas_manifest["sha256"],
        "prompt_version": "independent_utility_v3",
        "choice_temperature": REFERENCE_CHOICE_TEMPERATURE,
        "reconsideration_rate": REFERENCE_RECONSIDERATION_RATE,
        "choice_seeds": list(seeds),
        "timesteps": timesteps,
        "conditions": [dict(spec) for spec in CONDITIONS],
        "familiarity_rule": FAMILIARITY_RULE,
        "favorite_seed": FAVORITE_SEED,
        "shuffle_seed": SHUFFLE_SEED,
        "history_initialization_sha256": history_sha256,
        "initial_state_source": str(WORLD_V01_DIR / "initial_state.json"),
        "llm_calls": 0,
        "decision_signal": "overall_utility_plus_familiarity",
        "component_scores_not_used": True,
    }
    payload["config_sha256"] = canonical_sha256(payload)
    return payload


def _manifest(
    destination: Path,
    config: dict[str, Any],
    history_document: dict[str, Any],
    summary: dict[str, Any],
    worlds: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "experiment_id": "experiment_002_familiarity",
        "llm_calls": 0,
        "world_count": len(worlds),
        "complete": len(worlds) == len(CONDITIONS) * len(config["choice_seeds"]),
        "utility_atlas_sha256": config["utility_atlas_sha256"],
        "config_sha256": config["config_sha256"],
        "history_initialization_sha256": history_document["sha256"],
        "favorite_seed": FAVORITE_SEED,
        "shuffle_seed": SHUFFLE_SEED,
        "exact_replay": summary["exact_replay"],
        "trajectory_sha256": {
            f"{world['condition_id']}:{world['choice_seed']}": world["trajectory_sha256"] for world in worlds
        },
        "files": {
            name: hashlib.sha256((destination / name).read_bytes()).hexdigest()
            for name in (
                "world_v0_2_config.json",
                "history_initialization.json",
                "world_states.jsonl",
                "history_states.jsonl",
                "agent_actions.jsonl",
                "experiment_002_summary.json",
            )
        },
    }


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


def _state_lines(spec: dict[str, Any], seed: int, result: dict[str, Any]) -> list[dict[str, Any]]:
    lines = []
    for index, locations in enumerate(result["series"]):
        occupancy = occupancy_of(locations)
        lines.append(
            {
                "condition_id": spec["condition_id"],
                "choice_seed": seed,
                "familiarity_weight": spec["familiarity_weight"],
                "timestep": index,
                "locations": locations,
                "occupancy": occupancy,
                "llm_calls": 0,
            }
        )
    return lines


def _history_lines(spec: dict[str, Any], seed: int, result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "condition_id": spec["condition_id"],
            "choice_seed": seed,
            "timestep": index,
            "agents": snapshot,
            "llm_calls": 0,
        }
        for index, snapshot in enumerate(result["history"])
    ]


def _action_lines(spec: dict[str, Any], seed: int, actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"condition_id": spec["condition_id"], "choice_seed": seed, **action} for action in actions]


def _trajectory_hash(series: list[dict[str, str]]) -> str:
    ordered = [{agent_id: step[agent_id] for agent_id in sorted(step)} for step in series]
    return canonical_sha256(ordered)


def _total_variation(left: dict[str, int], right: dict[str, int], population: int) -> float:
    change = sum(abs(right[place_id] - left[place_id]) for place_id in PLACE_IDS)
    return 0.5 * change / population


def _mean_sd(values: list[float | None]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if value is not None]
    return {"mean": None if not finite else statistics.fmean(finite), "sd": _stdev(finite), "n": len(finite)}


def _stdev(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return float(statistics.stdev(values))


def _refuse_frozen_directories(destination: Path) -> None:
    forbidden = {WORLD_V01_DIR.resolve(), ATLAS_DIR.resolve(), CALIBRATION_DIR.resolve()}
    if destination.resolve() in forbidden:
        raise RuntimeError("experiment 002 must not write into frozen world outputs")


def _append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    lines = [json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) for record in records]
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + ("\n" if lines else ""))


def main() -> None:
    run_experiment()


if __name__ == "__main__":
    if __package__ is None:
        raise SystemExit("請在 worldlab 目錄執行：python -m world.experiment_002")
    main()
