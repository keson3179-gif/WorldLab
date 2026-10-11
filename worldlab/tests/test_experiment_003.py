"""World v0.3 endogenous reinforcement tests. No model calls."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.experiment_001 import load_base_traits
from world.dynamics_calibration import load_initial_locations
from world.experiment_003 import (
    FORK_AGENT,
    alternate_destination,
    decide_timestep,
    familiarity_scores,
    neutral_population,
    run_fork,
    simulate_world,
    trajectory_hash,
)
from world.geography import PLACE_IDS, distance_matrix, occupancy_of, place_index
from world.history import blank_history, commit_history, decision_utility, familiarity_scores as history_scores
from world.utility_atlas import load_frozen_atlas

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def atlas():
    return load_frozen_atlas()


@pytest.fixture(scope="module")
def population():
    locations = load_initial_locations()
    bases = load_base_traits(ROOT / "data" / "personas" / "persona_set_v1", len(locations))
    traits = {base["base_agent_id"]: base["traits"] for base in bases}
    return neutral_population(traits, locations)


def test_neutral_history_is_zero_and_initial_location_is_not_a_visit(population) -> None:
    state = next(iter(population.values()))
    assert state.previous_location is None
    assert all(count == 0 for count in state.visit_count.values())
    assert all(count == 0 for count in state.cumulative_dwell.values())
    assert all(value is None for value in state.time_since_last_visit.values())
    stayed = commit_history({"agent_0000": state}, {"agent_0000": state.current_location})["agent_0000"]
    assert all(count == 0 for count in stayed.visit_count.values())
    assert stayed.cumulative_dwell[state.current_location] == 1
    other = next(place_id for place_id in PLACE_IDS if place_id != state.current_location)
    moved = commit_history({"agent_0000": stayed}, {"agent_0000": other})["agent_0000"]
    assert moved.visit_count[other] == 1
    assert moved.visit_count[state.current_location] == 0
    assert familiarity_scores(state.visit_count) == {place_id: 0.0 for place_id in PLACE_IDS}
    assert familiarity_scores(stayed.visit_count) == {place_id: 0.0 for place_id in PLACE_IDS}
    assert familiarity_scores(moved.visit_count)[other] == 1.0


def test_familiarity_rule_matches_experiment_002() -> None:
    assert familiarity_scores is history_scores
    counts = {place_id: 2 for place_id in PLACE_IDS}
    counts["site_01"] = 20
    scores = familiarity_scores(counts)
    assert scores["site_01"] == 1.0
    assert all(0.0 <= score <= 1.0 for score in scores.values())
    assert decision_utility(640, scores["site_02"], 0) == 640
    assert decision_utility(640, scores["site_02"], 50) == 640 + 50 * scores["site_02"]


def test_weight_zero_ignores_history_and_matches_h0(atlas, population) -> None:
    boosted = {
        agent_id: blank_history(agent_id, state.traits, state.current_location) for agent_id, state in population.items()
    }
    for state in boosted.values():
        state.visit_count["site_01"] = 40
    plain = simulate_world(atlas, population, reinforcement_weight=0, choice_seed_value=0, timesteps=100)
    loaded = simulate_world(atlas, boosted, reinforcement_weight=0, choice_seed_value=0, timesteps=4)
    reference = simulate_world(atlas, population, reinforcement_weight=0, choice_seed_value=0, timesteps=4)
    assert loaded["series"] == reference["series"]
    _destinations, actions = decide_timestep(
        atlas,
        boosted,
        distance_matrix(),
        place_index(),
        reinforcement_weight=0,
        choice_seed_value=0,
        timestep=0,
    )
    for action in actions:
        assert action["total_utility"] == action["base_utility"]
    h0 = _h0_locations(0)
    assert plain["series"] == h0
    for seed in range(1, 20):
        result = simulate_world(atlas, population, reinforcement_weight=0, choice_seed_value=seed, timesteps=100)
        assert result["series"] == _h0_locations(seed)


def test_history_conservation_synchrony_and_order(atlas, population) -> None:
    result = simulate_world(atlas, population, reinforcement_weight=50, choice_seed_value=2, timesteps=6)
    for agent_id, state in result["final_states"].items():
        assert sum(state.cumulative_dwell.values()) == 6
        assert sum(state.visit_count.values()) == result["entries"][agent_id]
        assert all(value >= 0 for value in state.visit_count.values())
        assert all(value >= 0 for value in state.cumulative_dwell.values())
    _destinations, actions = decide_timestep(
        atlas,
        population,
        distance_matrix(),
        place_index(),
        reinforcement_weight=50,
        choice_seed_value=2,
        timestep=0,
    )
    occupancies = [tuple(sorted(action["occupancy_before"].items())) for action in actions]
    assert len(set(occupancies)) == 1
    origins = {agent_id: population[agent_id].current_location for agent_id in population}
    assert actions[0]["occupancy_before"] == occupancy_of(origins)
    reversed_states = {agent_id: population[agent_id] for agent_id in reversed(list(population))}
    other = simulate_world(atlas, reversed_states, reinforcement_weight=50, choice_seed_value=2, timesteps=6)
    assert result["series"] == other["series"]
    assert result["snapshots"][6] == other["snapshots"][6]


def test_fork_changes_one_destination_without_an_extra_draw(atlas, population) -> None:
    import world.choice as choice

    calls: list[int] = []
    original = choice.random.Random

    class Tracking:
        def __init__(self, seed: int) -> None:
            calls.append(seed)
            self._rng = original(seed)

        def random(self) -> float:
            return self._rng.random()

    utilities = {place_id: float(index) for index, place_id in enumerate(PLACE_IDS)}
    choice.random.Random = Tracking
    try:
        before = len(calls)
        replacement = alternate_destination(utilities, "site_05")
        assert len(calls) == before
    finally:
        choice.random.Random = original
    assert replacement == "site_04"
    normal = simulate_world(atlas, population, reinforcement_weight=150, choice_seed_value=0, timesteps=25)
    fork = run_fork(atlas, population, choice_seed_value=0, timesteps=25)
    assert fork["series_a"] == normal["series"]
    assert fork["series_a"][19] == fork["series_b"][19]
    assert fork["series_b"][20][FORK_AGENT] != fork["series_a"][20][FORK_AGENT]
    assert fork["replacement_destination"] != fork["original_destination"]
    choice.random.Random = Tracking
    try:
        calls.clear()
        simulate_world(atlas, population, reinforcement_weight=150, choice_seed_value=4, timesteps=3)
        first = list(calls)
        calls.clear()
        shifted = {
            agent_id: blank_history(
                agent_id,
                state.traits,
                PLACE_IDS[(PLACE_IDS.index(state.current_location) + 1) % len(PLACE_IDS)],
            )
            for agent_id, state in population.items()
        }
        for state in shifted.values():
            state.visit_count[state.current_location] = 8
        simulate_world(atlas, shifted, reinforcement_weight=150, choice_seed_value=4, timesteps=3)
        second = list(calls)
    finally:
        choice.random.Random = original
    assert first == second
    replay = simulate_world(atlas, population, reinforcement_weight=150, choice_seed_value=0, timesteps=25)
    assert trajectory_hash(replay["series"]) == trajectory_hash(normal["series"])


def test_zero_llm_calls_and_place_ids_stay_out_of_the_utility_model() -> None:
    source = (ROOT / "world" / "experiment_003.py").read_text(encoding="utf-8")
    assert "StructuredLLMClient" not in source
    assert "build_utility_messages" not in source
    assert "PlaceUtility" not in source
    assert "option a" not in source.lower()


def _h0_locations(seed: int) -> list[dict[str, str]]:
    rows = []
    path = ROOT / "data" / "worlds" / "world_v0_2" / "world_states.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if f'"condition_id":"H0"' not in line or f'"choice_seed":{seed},' not in line:
                continue
            import json

            row = json.loads(line)
            if row["condition_id"] == "H0" and int(row["choice_seed"]) == seed:
                rows.append((int(row["timestep"]), row["locations"]))
    rows.sort(key=lambda item: item[0])
    return [locations for _timestep, locations in rows[:101]]
