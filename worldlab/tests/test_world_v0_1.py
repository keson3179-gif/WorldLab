from __future__ import annotations

import json

from agents.persona_generator import generate_traits
from config.settings import get_settings
from experiments.experiment_001 import load_base_traits
from experiments.experiment_001e import PROMPT_VERSION, build_utility_messages, prompt_sha256
from experiments.tradeoff_scenarios import PROMPT_FIELDS
from llm.client import LLMCallResult
from world.choice import choice_seed, initial_locations, softmax_choice
from world.geography import PLACE_IDS, candidate_attributes, crowding, distance_matrix, occupancy_of
from world.simulation import evaluate_place, run_world, step_world


def test_geography_bounds_and_conservation() -> None:
    matrix = distance_matrix()
    for origin in PLACE_IDS:
        for destination in PLACE_IDS:
            value = matrix[origin][destination]
            assert 0.0 <= value <= 1.0
            assert value == matrix[destination][origin]
        assert matrix[origin][origin] == 0.0
    assert crowding(0, 8) == 0.0
    assert crowding(8, 8) == 1.0
    assert crowding(12, 8) == 1.0
    locations = initial_locations([f"agent_{index:04d}" for index in range(20)], PLACE_IDS, get_settings().world_seed)
    occupancy = occupancy_of(locations)
    assert sum(occupancy.values()) == 20
    assert set(locations.values()) <= set(PLACE_IDS)


def test_same_state_and_seed_repeat_the_choice() -> None:
    utilities = [800, 640, 510, 500, 120]
    seed = choice_seed(42, 3, "agent_0000")
    first, first_probabilities = softmax_choice(list(PLACE_IDS), utilities, 100.0, seed)
    second, second_probabilities = softmax_choice(list(PLACE_IDS), utilities, 100.0, seed)
    assert first == second
    assert first_probabilities == second_probabilities
    assert abs(sum(first_probabilities.values()) - 1.0) < 1e-12
    draws = {softmax_choice(list(PLACE_IDS), [500, 500, 500, 500, 500], 100.0, choice_seed(42, step, "agent_0000"))[0] for step in range(30)}
    assert len(draws) > 1


def test_prompt_has_no_place_identity_or_candidate_position() -> None:
    traits, _ = generate_traits(42, 0)
    occupancy = {place_id: 4 for place_id in PLACE_IDS}
    matrix = distance_matrix()
    attributes = candidate_attributes("site_01", "site_02", occupancy, matrix)
    messages = build_utility_messages(traits, attributes, prompt_version=PROMPT_VERSION)
    text = "\n".join(message["content"] for message in messages)
    rendered = {field: attributes[field] for field in PROMPT_FIELDS}
    other = {field: candidate_attributes("site_03", "site_02", occupancy, matrix)[field] for field in PROMPT_FIELDS}
    assert json.dumps(rendered, ensure_ascii=False, indent=2, allow_nan=False) in text
    assert json.dumps(other, ensure_ascii=False, indent=2, allow_nan=False) not in text
    lowered = text.lower()
    assert "site_" not in lowered
    assert "place_id" not in lowered
    assert "option a" not in lowered
    assert "option b" not in lowered
    assert "candidate" not in lowered


def test_cache_repeats_scores_without_another_call() -> None:
    settings = get_settings()
    client = _ScriptedUtility()
    traits, _ = generate_traits(42, 0)
    attributes = candidate_attributes("site_01", "site_01", {place_id: 0 for place_id in PLACE_IDS}, distance_matrix())
    cache: dict[str, dict[str, int]] = {}
    first, digest, hit = evaluate_place(traits, attributes, settings=settings, client=client, cache=cache, cache_path=None)
    second, again, second_hit = evaluate_place(traits, attributes, settings=settings, client=client, cache=cache, cache_path=None)
    assert hit is False
    assert second_hit is True
    assert digest == again
    assert first == second
    assert client.calls == 1


def test_world_step_is_synchronous_and_order_invariant() -> None:
    settings = get_settings()
    bases = load_base_traits(settings.personas_dir / "persona_set_v1", settings.world_agent_count)
    traits = {base["base_agent_id"]: base["traits"] for base in bases}
    locations = initial_locations(list(traits), PLACE_IDS, settings.world_seed)
    before = dict(locations)
    snapshot, evaluations, actions = step_world(
        locations,
        traits,
        distance_matrix(),
        {},
        timestep=0,
        world_seed=settings.world_seed,
        choice_temperature=settings.choice_temperature,
        settings=settings,
        client=_ScriptedUtility(),
        cache={},
        cache_path=None,
        configuration={},
    )
    assert locations == before
    assert snapshot["locations_before"] == before
    assert sum(snapshot["occupancy_before"].values()) == 20
    assert sum(snapshot["occupancy_after"].values()) == 20
    assert set(snapshot["locations_after"]) == set(before)
    assert all(place_id in PLACE_IDS for place_id in snapshot["locations_after"].values())
    for row in evaluations:
        assert row["crowding"] == snapshot["crowding_before"][row["place_id"]]
        assert 0.0 <= row["crowding"] <= 1.0
        assert 0.0 <= row["distance"] <= 1.0
    assert all("crowding_fit" not in action for action in actions)
    reversed_locations = {agent_id: before[agent_id] for agent_id in reversed(list(before))}
    other, _, _ = step_world(
        reversed_locations,
        traits,
        distance_matrix(),
        {},
        timestep=0,
        world_seed=settings.world_seed,
        choice_temperature=settings.choice_temperature,
        settings=settings,
        client=_ScriptedUtility(),
        cache={},
        cache_path=None,
        configuration={},
    )
    assert other["destinations"] == snapshot["destinations"]
    assert other["locations_after"] == snapshot["locations_after"]


def test_same_world_seed_replays_and_writes_the_log(tmp_path) -> None:
    settings = get_settings()
    assert settings.world_seed == 42
    assert settings.temperature == 0.0
    assert settings.choice_temperature > 0.0
    first = run_world(client=_ScriptedUtility(), output_dir=tmp_path / "first", timesteps=2)
    second = run_world(client=_ScriptedUtility(), output_dir=tmp_path / "second", timesteps=2)
    first_manifest = json.loads((first / "run_manifest.json").read_text(encoding="utf-8"))
    second_manifest = json.loads((second / "run_manifest.json").read_text(encoding="utf-8"))
    assert first_manifest["trajectory_sha256"] == second_manifest["trajectory_sha256"]
    assert first_manifest["complete"] is True
    assert first_manifest["utility_slots"] == 20 * 5 * 2
    for name in (
        "world_config.json",
        "initial_state.json",
        "places.json",
        "distance_matrix.json",
        "world_states.jsonl",
        "agent_actions.jsonl",
        "utility_evaluations.jsonl",
        "run_manifest.json",
    ):
        assert (first / name).is_file()
    states = [json.loads(line) for line in (first / "world_states.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(states) == 2
    for state in states:
        assert len(state["locations_after"]) == 20
        assert sum(state["occupancy_after"].values()) == 20
    analysis = json.loads((first / "analysis.json").read_text(encoding="utf-8"))
    assert "does not claim emergence" in analysis["note"]


class _ScriptedUtility:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, *, messages, response_model):
        self.calls += 1
        value = int(prompt_sha256(messages)[:8], 16) % 1001
        payload = {name: value for name in ("overall_utility", "crowding_fit", "visual_fit", "distance_fit", "novelty_fit")}
        parsed = response_model.model_validate(payload)
        return LLMCallResult(
            raw_text=parsed.model_dump_json(),
            parsed=parsed,
            model="scripted",
            temperature=0.0,
            llm_seed=42,
            reasoning_effort="none",
            api_endpoint="scripted",
        )
