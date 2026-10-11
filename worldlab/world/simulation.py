"""Synchronous World v0.1 runner.

Phase 1 scores every candidate from the shared state. Phase 2 commits every
move afterward. A content-addressed cache does not change those scores.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from agents.persona_generator import append_jsonl, write_json
from config.settings import PROJECT_ROOT, Settings, get_settings
from experiments.experiment_001 import load_base_traits
from experiments.experiment_001e import PROMPT_VERSION, PlaceUtility, build_utility_messages, prompt_sha256
from experiments.reproducibility_check import canonical_sha256
from llm.client import LLMCallError, LLMClient, StructuredLLMClient
from world.analysis import summarize_run
from world.choice import CHOICE_ALGORITHM, PLACEMENT_ALGORITHM, choice_seed, initial_locations, softmax_choice
from world.geography import (
    CROWDING_FORMULA,
    PLACE_IDS,
    PLACES,
    candidate_attributes,
    crowding_of,
    distance_matrix,
    occupancy_of,
    place_index,
)

WORLD_ID = "world_v0_1"
OUTPUT_DIR = PROJECT_ROOT / "data" / "worlds" / WORLD_ID
COMPONENT_NOTE = "Component scores are diagnostic outputs, not independent construct measures."


def run_world(
    settings: Settings | None = None,
    client: LLMClient | None = None,
    *,
    persona_dir: Path | None = None,
    output_dir: Path | None = None,
    cache_path: Path | None = None,
    timesteps: int | None = None,
) -> Path:
    active = get_settings() if settings is None else settings
    _require_world_settings(active)
    destination = output_dir or OUTPUT_DIR
    destination.mkdir(parents=True, exist_ok=True)
    cache_file = cache_path or (destination / "utility_cache.jsonl")
    source = persona_dir or (active.personas_dir / "persona_set_v1")
    bases = load_base_traits(source, active.world_agent_count)
    horizon = active.world_timesteps if timesteps is None else timesteps
    matrix = distance_matrix()
    places = place_index()
    configuration = _model_configuration(active)
    cache = _load_cache(cache_file)
    completed = _completed_timesteps(destination / "world_states.jsonl")
    if completed == 0:
        locations = initial_locations([base["base_agent_id"] for base in bases], PLACE_IDS, active.world_seed)
        _write_static_files(destination, active, configuration, bases, locations, matrix, horizon)
    else:
        locations = _locations_from_last_state(destination / "world_states.jsonl")
    traits = {base["base_agent_id"]: base["traits"] for base in bases}
    llm = client if client is not None else StructuredLLMClient(active)
    totals = _count_existing(destination / "utility_evaluations.jsonl")
    for timestep in range(completed, horizon):
        print(f"timestep {timestep + 1}/{horizon} evaluating", flush=True)
        snapshot, evaluations, actions = step_world(
            locations,
            traits,
            matrix,
            places,
            timestep=timestep,
            world_seed=active.world_seed,
            choice_temperature=active.choice_temperature,
            settings=active,
            client=llm,
            cache=cache,
            cache_path=cache_file,
            configuration=configuration,
        )
        append_jsonl(destination / "world_states.jsonl", snapshot)
        for action in actions:
            append_jsonl(destination / "agent_actions.jsonl", action)
        for evaluation in evaluations:
            append_jsonl(destination / "utility_evaluations.jsonl", evaluation)
        locations = dict(snapshot["locations_after"])
        totals["slots"] += len(evaluations)
        totals["cache_hits"] += sum(1 for row in evaluations if row["cache_hit"])
        totals["llm_calls"] += sum(1 for row in evaluations if not row["cache_hit"])
        print(
            f"timestep {timestep + 1}/{horizon} llm_calls={totals['llm_calls']} cache_hits={totals['cache_hits']}",
            flush=True,
        )
    snapshots = _read_jsonl(destination / "world_states.jsonl")
    evaluations = _read_jsonl(destination / "utility_evaluations.jsonl")
    initial = json.loads((destination / "initial_state.json").read_text(encoding="utf-8"))
    analysis = summarize_run(
        initial_locations=initial["locations"],
        snapshots=snapshots,
        evaluations=evaluations,
    )
    write_json(destination / "analysis.json", analysis)
    manifest = _manifest(destination, configuration, totals, snapshots, initial["locations"])
    write_json(destination / "run_manifest.json", manifest)
    return destination


def step_world(
    locations: dict[str, str],
    traits: dict[str, Any],
    matrix: dict[str, dict[str, float]],
    places: dict[str, dict[str, Any]],
    *,
    timestep: int,
    world_seed: int,
    choice_temperature: float,
    settings: Settings,
    client: LLMClient,
    cache: dict[str, dict[str, int]],
    cache_path: Path | None,
    configuration: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    del places
    occupancy = occupancy_of(locations)
    crowding = crowding_of(occupancy)
    evaluations: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    choices: dict[str, str] = {}
    for agent_id in sorted(locations):
        origin = locations[agent_id]
        utilities: list[int] = []
        per_place: dict[str, dict[str, int]] = {}
        for place_id in PLACE_IDS:
            attributes = candidate_attributes(place_id, origin, occupancy, matrix)
            scores, digest, hit = evaluate_place(
                traits[agent_id],
                attributes,
                settings=settings,
                client=client,
                cache=cache,
                cache_path=cache_path,
            )
            per_place[place_id] = scores
            utilities.append(scores["overall_utility"])
            evaluations.append(
                {
                    "timestep": timestep,
                    "agent_id": agent_id,
                    "place_id": place_id,
                    "origin": origin,
                    "prompt_sha256": digest,
                    "cache_hit": hit,
                    "crowding": attributes["crowding"],
                    "visual_attractiveness": attributes["visual_attractiveness"],
                    "novelty": attributes["novelty"],
                    "distance": attributes["distance"],
                    **scores,
                    "decision_signal": "overall_utility",
                    "component_scores_are_diagnostics_only": True,
                    "model_configuration": configuration,
                }
            )
        seed = choice_seed(world_seed, timestep, agent_id)
        destination, probabilities = softmax_choice(list(PLACE_IDS), utilities, choice_temperature, seed)
        choices[agent_id] = destination
        actions.append(
            {
                "timestep": timestep,
                "agent_id": agent_id,
                "origin": origin,
                "destination": destination,
                "stayed": destination == origin,
                "overall_utilities": {place_id: per_place[place_id]["overall_utility"] for place_id in PLACE_IDS},
                "choice_probabilities": {place_id: probabilities[place_id] for place_id in PLACE_IDS},
                "choice_seed": seed,
                "choice_algorithm": CHOICE_ALGORITHM,
                "world_seed": world_seed,
                "choice_temperature": choice_temperature,
            }
        )
    updated = {agent_id: choices[agent_id] for agent_id in sorted(locations)}
    updated_occupancy = occupancy_of(updated)
    snapshot = {
        "timestep": timestep,
        "world_seed": world_seed,
        "choice_temperature": choice_temperature,
        "occupancy_before": occupancy,
        "crowding_before": crowding,
        "locations_before": {agent_id: locations[agent_id] for agent_id in sorted(locations)},
        "destinations": choices,
        "occupancy_after": updated_occupancy,
        "crowding_after": crowding_of(updated_occupancy),
        "locations_after": updated,
    }
    return snapshot, evaluations, actions


def evaluate_place(
    traits: Any,
    attributes: dict[str, float],
    *,
    settings: Settings,
    client: LLMClient,
    cache: dict[str, dict[str, int]],
    cache_path: Path | None,
) -> tuple[dict[str, int], str, bool]:
    messages = build_utility_messages(traits, attributes, prompt_version=PROMPT_VERSION)
    digest = prompt_sha256(messages)
    cached = cache.get(digest)
    if cached is not None:
        return dict(cached), digest, True
    parsed = _complete(settings, client, messages)
    scores = {
        "overall_utility": int(parsed.overall_utility),
        "crowding_fit": int(parsed.crowding_fit),
        "visual_fit": int(parsed.visual_fit),
        "distance_fit": int(parsed.distance_fit),
        "novelty_fit": int(parsed.novelty_fit),
    }
    cache[digest] = scores
    if cache_path is not None:
        append_jsonl(cache_path, {"prompt_sha256": digest, "scores": scores})
    return scores, digest, False


def _complete(settings: Settings, client: LLMClient, messages: list[dict[str, str]]) -> PlaceUtility:
    error = ""
    for _attempt in range(settings.decision_max_attempts):
        try:
            result = client.complete(messages=messages, response_model=PlaceUtility)
        except (LLMCallError, ValidationError) as exc:
            error = str(exc)
            continue
        if isinstance(result.parsed, PlaceUtility):
            return result.parsed
        error = "parsed object was not a PlaceUtility"
    raise RuntimeError(error or "utility evaluation failed")


def _write_static_files(
    destination: Path,
    settings: Settings,
    configuration: dict[str, Any],
    bases: list[dict[str, Any]],
    locations: dict[str, str],
    matrix: dict[str, dict[str, float]],
    horizon: int,
) -> None:
    occupancy = occupancy_of(locations)
    config = {
        "world_id": WORLD_ID,
        "persona_set": "persona_set_v1",
        "prompt_version": PROMPT_VERSION,
        "decision_signal": "overall_utility",
        "component_score_note": COMPONENT_NOTE,
        "crowding_formula": CROWDING_FORMULA,
        "placement_algorithm": PLACEMENT_ALGORITHM,
        "choice_algorithm": CHOICE_ALGORITHM,
        "world_seed": settings.world_seed,
        "choice_temperature": settings.choice_temperature,
        "llm_temperature": settings.temperature,
        "llm_seed": settings.llm_seed,
        "timesteps": horizon,
        "agent_count": len(bases),
        "place_count": len(PLACE_IDS),
        "model_configuration": configuration,
    }
    config["config_sha256"] = canonical_sha256(config)
    write_json(destination / "world_config.json", config)
    write_json(
        destination / "places.json",
        {"crowding_formula": CROWDING_FORMULA, "places": [dict(place) for place in PLACES]},
    )
    write_json(
        destination / "distance_matrix.json",
        {"place_ids": list(PLACE_IDS), "stay_distance": 0.0, "matrix": [[matrix[origin][destination] for destination in PLACE_IDS] for origin in PLACE_IDS]},
    )
    write_json(
        destination / "initial_state.json",
        {
            "world_seed": settings.world_seed,
            "placement_algorithm": PLACEMENT_ALGORITHM,
            "agent_ids": [base["base_agent_id"] for base in bases],
            "locations": {agent_id: locations[agent_id] for agent_id in sorted(locations)},
            "occupancy": occupancy,
            "crowding": crowding_of(occupancy),
        },
    )


def _manifest(
    destination: Path,
    configuration: dict[str, Any],
    totals: dict[str, int],
    snapshots: list[dict[str, Any]],
    initial_locations_map: dict[str, str],
) -> dict[str, Any]:
    trajectory = [initial_locations_map]
    trajectory.extend(snapshot["locations_after"] for snapshot in snapshots)
    slots = totals["slots"]
    payload = {
        "world_id": WORLD_ID,
        "complete": len(snapshots) == int(json.loads((destination / "world_config.json").read_text(encoding="utf-8"))["timesteps"]),
        "timesteps": len(snapshots),
        "utility_slots": slots,
        "llm_calls": totals["llm_calls"],
        "cache_hits": totals["cache_hits"],
        "cache_hit_rate": 0.0 if slots == 0 else totals["cache_hits"] / slots,
        "model_configuration": configuration,
        "trajectory_sha256": canonical_sha256(trajectory),
        "files": {
            name: _file_sha256(destination / name)
            for name in (
                "world_config.json",
                "initial_state.json",
                "places.json",
                "distance_matrix.json",
                "world_states.jsonl",
                "agent_actions.jsonl",
                "utility_evaluations.jsonl",
                "analysis.json",
            )
        },
    }
    return payload


def _require_world_settings(settings: Settings) -> None:
    if settings.temperature != 0.0:
        raise RuntimeError("world v0.1 keeps the LLM temperature at 0")
    if settings.reasoning_effort != "none":
        raise RuntimeError("world v0.1 requires reasoning_effort none")
    if settings.llm_seed != 42:
        raise RuntimeError("world v0.1 requires llm seed 42")
    if settings.choice_temperature <= 0.0:
        raise RuntimeError("choice temperature must be positive")


def _model_configuration(settings: Settings) -> dict[str, Any]:
    return {
        "model": settings.model,
        "temperature": settings.temperature,
        "seed": settings.llm_seed,
        "reasoning_effort": settings.reasoning_effort,
        "api_endpoint": settings.api_endpoint,
        "decision_prompt_version": PROMPT_VERSION,
        "choice_temperature": settings.choice_temperature,
        "world_seed": settings.world_seed,
    }


def _load_cache(path: Path) -> dict[str, dict[str, int]]:
    cache: dict[str, dict[str, int]] = {}
    for row in _read_jsonl(path):
        digest = str(row["prompt_sha256"])
        scores = {key: int(value) for key, value in row["scores"].items()}
        previous = cache.get(digest)
        if previous is not None and previous != scores:
            raise RuntimeError("utility cache has conflicting scores for one prompt")
        cache[digest] = scores
    return cache


def _completed_timesteps(path: Path) -> int:
    return len(_read_jsonl(path))


def _locations_from_last_state(path: Path) -> dict[str, str]:
    rows = _read_jsonl(path)
    return {str(agent_id): str(place_id) for agent_id, place_id in rows[-1]["locations_after"].items()}


def _count_existing(path: Path) -> dict[str, int]:
    rows = _read_jsonl(path)
    hits = sum(1 for row in rows if row.get("cache_hit"))
    return {"slots": len(rows), "cache_hits": hits, "llm_calls": len(rows) - hits}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _file_sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    run_world()


if __name__ == "__main__":
    if __package__ is None:
        raise SystemExit("請在 worldlab 目錄執行：python -m world.simulation")
    main()
