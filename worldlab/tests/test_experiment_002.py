"""World v0.2 history state and familiarity tests. No model calls."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.experiment_001 import load_base_traits
from world.dynamics_calibration import load_initial_locations
from world.experiment_002 import (
    OUTPUT_DIR,
    favorite_choice_rate,
    return_probability,
    simulate_condition,
    trajectory_divergence,
)
from world.geography import PLACE_IDS
from world.history import (
    AgentHistory,
    assign_favorites,
    blank_history,
    commit_history,
    decision_utility,
    familiarity_scores,
    initial_history,
    shuffle_favorites,
)
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
    favorites = assign_favorites(list(locations))
    states = {
        agent_id: initial_history(agent_id, traits[agent_id], locations[agent_id], favorites[agent_id])
        for agent_id in sorted(locations)
    }
    return states, favorites, traits, locations


def test_history_initialization_and_favorites_are_deterministic() -> None:
    agents = [f"agent_{index:04d}" for index in range(20)]
    first = assign_favorites(agents, 42)
    second = assign_favorites(list(reversed(agents)), 42)
    assert first == second
    assert sorted(first) == agents
    counts = {place_id: list(first.values()).count(place_id) for place_id in PLACE_IDS}
    assert counts == {place_id: 4 for place_id in PLACE_IDS}


def test_history_counters_and_dwell_conservation() -> None:
    state = initial_history("agent_0000", {"crowd_aversion": 0.2}, "site_01", "site_02")
    before = sum(state.cumulative_dwell.values())
    updated = commit_history({"agent_0000": state}, {"agent_0000": "site_03"})["agent_0000"]
    assert updated.visit_count["site_03"] == state.visit_count["site_03"] + 1
    assert updated.visit_count["site_01"] == state.visit_count["site_01"]
    assert updated.cumulative_dwell["site_03"] == state.cumulative_dwell["site_03"] + 1
    assert sum(updated.cumulative_dwell.values()) == before + 1
    assert updated.time_since_last_visit["site_03"] == 0
    assert updated.time_since_last_visit["site_02"] == 1
    assert updated.previous_location == "site_01"
    assert updated.current_location == "site_03"
    assert updated.traits == state.traits
    stayed = commit_history({"agent_0000": updated}, {"agent_0000": "site_03"})["agent_0000"]
    assert stayed.visit_count["site_03"] == updated.visit_count["site_03"]
    assert sum(stayed.cumulative_dwell.values()) == before + 2
    assert all(value >= 0 for value in stayed.visit_count.values())
    assert all(value >= 0 for value in stayed.cumulative_dwell.values())


def test_null_time_since_last_visit_stays_null_until_entry() -> None:
    state = blank_history("agent_0000", None, "site_01")
    state.time_since_last_visit["site_01"] = 0
    state.visit_count["site_01"] = 1
    away = commit_history({"agent_0000": state}, {"agent_0000": "site_02"})["agent_0000"]
    assert away.time_since_last_visit["site_03"] is None
    assert away.time_since_last_visit["site_02"] == 0
    assert away.visit_count["site_02"] == 1
    later = commit_history({"agent_0000": away}, {"agent_0000": "site_02"})["agent_0000"]
    assert later.time_since_last_visit["site_03"] is None
    assert later.time_since_last_visit["site_01"] == 2


def test_familiarity_is_bounded_and_utility_adds_the_history_term() -> None:
    scores = familiarity_scores({"site_01": 20, "site_02": 2, "site_03": 2, "site_04": 2, "site_05": 2})
    assert all(0.0 <= score <= 1.0 for score in scores.values())
    assert scores["site_01"] == 1.0
    assert familiarity_scores({place_id: 0 for place_id in PLACE_IDS}) == {place_id: 0.0 for place_id in PLACE_IDS}
    assert decision_utility(640, scores["site_02"], 50) == 640 + 50 * scores["site_02"]
    assert decision_utility(640, scores["site_01"], 0) == 640


def test_weight_zero_matches_a_history_free_world(atlas, population) -> None:
    states, favorites, traits, locations = population
    blank = {agent_id: blank_history(agent_id, traits[agent_id], locations[agent_id]) for agent_id in locations}
    weighted = simulate_condition(atlas, states, favorites, familiarity_weight=0, choice_seed_value=0, timesteps=4)
    empty = simulate_condition(atlas, blank, favorites, familiarity_weight=0, choice_seed_value=0, timesteps=4)
    assert weighted["series"] == empty["series"]
    assert weighted["llm_calls"] == 0
    for action in weighted["actions"]:
        for place_id in PLACE_IDS:
            assert action["total_utility"][place_id] == action["base_utility"][place_id]


def test_synchronous_update_and_iteration_order(atlas, population) -> None:
    states, favorites, _traits, _locations = population
    forward = simulate_condition(atlas, states, favorites, familiarity_weight=50, choice_seed_value=3, timesteps=3)
    reversed_states = {agent_id: states[agent_id] for agent_id in reversed(list(states))}
    backward = simulate_condition(atlas, reversed_states, favorites, familiarity_weight=50, choice_seed_value=3, timesteps=3)
    assert forward["series"] == backward["series"]
    assert forward["history"] == backward["history"]
    for timestep in (1, 2, 3):
        rows = [action for action in forward["actions"] if action["timestep"] == timestep]
        occupancies = [tuple(sorted(action["occupancy_before"].items())) for action in rows]
        assert len(set(occupancies)) == 1
        before = rows[0]["occupancy_before"]
        origins = forward["series"][timestep - 1]
        assert before == {place_id: list(origins.values()).count(place_id) for place_id in PLACE_IDS}
        assert forward["series"][timestep] == {action["agent_id"]: action["destination"] for action in rows}


def test_exact_replay_and_shuffled_mapping(atlas, population) -> None:
    states, favorites, _traits, _locations = population
    first = simulate_condition(atlas, states, favorites, familiarity_weight=50, choice_seed_value=1, timesteps=5)
    second = simulate_condition(atlas, states, favorites, familiarity_weight=50, choice_seed_value=1, timesteps=5)
    assert trajectory_divergence(first["series"], second["series"])["trajectory_hash_equal"] is True
    shuffled_a = shuffle_favorites(favorites, 43)
    shuffled_b = shuffle_favorites(dict(reversed(list(favorites.items()))), 43)
    assert shuffled_a == shuffled_b
    assert shuffled_a != favorites
    assert sorted(shuffled_a.values()) == sorted(favorites.values())


def test_return_probability_uses_a_full_five_step_window() -> None:
    favorite = {"agent_0000": "site_01"}
    series = [
        {"agent_0000": "site_01"},
        {"agent_0000": "site_02"},
        {"agent_0000": "site_02"},
        {"agent_0000": "site_02"},
        {"agent_0000": "site_02"},
        {"agent_0000": "site_02"},
        {"agent_0000": "site_01"},
    ]
    assert return_probability(series, favorite) == 1.0
    short = series[:6]
    assert return_probability(short, favorite) is None
    assert favorite_choice_rate(series, favorite) == 1 / 6


def test_zero_llm_calls_and_frozen_outputs_are_untouched() -> None:
    source = (ROOT / "world" / "experiment_002.py").read_text(encoding="utf-8")
    history = (ROOT / "world" / "history.py").read_text(encoding="utf-8")
    assert "StructuredLLMClient" not in source
    assert "StructuredLLMClient" not in history
    assert OUTPUT_DIR.name == "world_v0_2"
    for name in ("world_v0_1", "utility_atlas_v1", "dynamics_calibration_001"):
        assert name not in str(OUTPUT_DIR)
