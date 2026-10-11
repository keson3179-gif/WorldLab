"""World v0.4 social popularity tests. No model calls."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.experiment_001 import load_base_traits
from world.choice import agent_reconsiders, choice_seed, softmax_choice
from world.dynamics_calibration import load_initial_locations
from world.experiment_002 import REFERENCE_CHOICE_TEMPERATURE, REFERENCE_RECONSIDERATION_RATE
from world.experiment_004 import (
    FAMILIARITY_WEIGHT,
    classify_result,
    commit_social,
    empty_social,
    popularity_scores,
    simulate_world,
    trajectory_hash,
)
from world.geography import PLACE_IDS
from world.utility_atlas import load_frozen_atlas

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def atlas():
    return load_frozen_atlas()


@pytest.fixture(scope="module")
def locations():
    return load_initial_locations()


def test_social_history_starts_at_zero_and_excludes_the_initial_location(locations) -> None:
    social = empty_social(list(locations))
    assert all(value == 0 for value in social.total.values())
    assert all(value is not None for value in locations.values())
    for agent_id in locations:
        assert all(value == 0 for value in social.own[agent_id].values())
        assert popularity_scores(social, agent_id) == {place_id: 0.0 for place_id in PLACE_IDS}
    stayed = {agent_id: locations[agent_id] for agent_id in locations}
    after = commit_social(social, stayed)
    assert social.total == {place_id: 0 for place_id in PLACE_IDS}
    assert sum(after.total.values()) == len(locations)
    assert after.total[locations["agent_0000"]] >= 1


def test_focal_agent_is_excluded_and_scores_use_only_earlier_commits() -> None:
    social = empty_social(["agent_0000", "agent_0001"])
    first = commit_social(social, {"agent_0000": "site_01", "agent_0001": "site_01"})
    second = commit_social(first, {"agent_0000": "site_01", "agent_0001": "site_02"})
    own = popularity_scores(second, "agent_0000")
    other = popularity_scores(second, "agent_0001")
    assert second.total["site_01"] - second.own["agent_0000"]["site_01"] == 1
    assert second.total["site_01"] - second.own["agent_0001"]["site_01"] == 2
    assert other["site_02"] == 0.0
    assert all(0.0 <= score <= 1.0 for score in own.values())
    assert max(own.values()) == 1.0
    reversed_commit = commit_social(
        social,
        {"agent_0001": "site_02", "agent_0000": "site_01"},
    )
    forward = commit_social(social, {"agent_0000": "site_01", "agent_0001": "site_02"})
    assert reversed_commit.total == forward.total


def test_weight_zero_matches_softmax_and_ignores_social_state(atlas, locations) -> None:
    assert FAMILIARITY_WEIGHT == 0.0
    plain = simulate_world(
        atlas,
        locations,
        empty_social(list(locations)),
        social_weight=0,
        choice_seed_value=0,
        timesteps=4,
    )
    boosted = empty_social(list(locations))
    for agent_id in boosted.own:
        for place_id in PLACE_IDS:
            boosted.own[agent_id][place_id] = 30
            boosted.total[place_id] += 30
    ignored = simulate_world(atlas, locations, boosted, social_weight=0, choice_seed_value=0, timesteps=4)
    assert plain["series"] == ignored["series"]
    assert plain["destination_change_rate"] == 0.0
    utilities = [100.0, 250.0, 250.0, 10.0, 90.0]
    for seed in (1, 7, 99):
        draw = __import__("random").Random(seed).random()
        selected, probabilities = world_select(utilities, draw)
        reference, reference_probabilities = softmax_choice(list(PLACE_IDS), utilities, REFERENCE_CHOICE_TEMPERATURE, seed)
        assert selected == reference
        assert probabilities == reference_probabilities


def test_s0_reproduces_r0_and_exogenous_signal_is_not_own_history(atlas, locations) -> None:
    social = empty_social(list(locations))
    for seed in range(20):
        result = simulate_world(atlas, locations, social, social_weight=0, choice_seed_value=seed, timesteps=500)
        assert result["series"] == _r0_locations(seed)
        assert result["llm_calls"] == 0
    reference = {logged: _zero_scores(locations) for logged in range(1, 9)}
    forced = simulate_world(
        atlas,
        locations,
        social,
        social_weight=150,
        choice_seed_value=3,
        timesteps=8,
        score_source=reference,
    )
    endogenous = simulate_world(atlas, locations, social, social_weight=150, choice_seed_value=3, timesteps=8)
    assert forced["used_scores"][8]["agent_0000"] == {place_id: 0.0 for place_id in PLACE_IDS}
    assert max(popularity_scores(forced["final_social"], "agent_0000").values()) > 0.0
    assert endogenous["used_scores"][8]["agent_0000"] != forced["used_scores"][8]["agent_0000"]


def test_synchronous_update_order_replay_and_rng_logging(atlas, locations) -> None:
    seen = []

    def capture(logged, _locations, social, actions):
        if logged == 1:
            seen.append((social.total, actions))

    social = empty_social(list(locations))
    simulate_world(atlas, locations, social, social_weight=50, choice_seed_value=1, timesteps=2, on_step=capture)
    total, actions = seen[0]
    assert len({tuple(sorted(action["occupancy_before"].items())) for action in actions}) == 1
    assert len({tuple(sorted(action["social_total_before"].items())) for action in actions}) == 1
    assert actions[0]["social_total_before"] == {place_id: 0 for place_id in PLACE_IDS}
    assert sum(total.values()) == len(locations)
    reversed_locations = {agent_id: locations[agent_id] for agent_id in reversed(list(locations))}
    forward = simulate_world(atlas, locations, social, social_weight=50, choice_seed_value=1, timesteps=6)
    backward = simulate_world(atlas, reversed_locations, social, social_weight=50, choice_seed_value=1, timesteps=6)
    assert forward["series"] == backward["series"]
    assert trajectory_hash(forward["series"]) == trajectory_hash(backward["series"])
    repeated = simulate_world(atlas, locations, social, social_weight=150, choice_seed_value=4, timesteps=6)
    assert trajectory_hash(repeated["series"]) == trajectory_hash(
        simulate_world(atlas, locations, social, social_weight=150, choice_seed_value=4, timesteps=6)["series"]
    )
    import world.experiment_004 as experiment

    recorded = []
    original = experiment.random.Random

    class Tracking:
        def __init__(self, seed: int) -> None:
            recorded.append(seed)
            self._rng = original(seed)

        def random(self) -> float:
            return self._rng.random()

    experiment.random.Random = Tracking
    try:
        recorded.clear()
        simulate_world(atlas, locations, social, social_weight=50, choice_seed_value=5, timesteps=3)
        without_log = list(recorded)
        recorded.clear()
        simulate_world(
            atlas,
            locations,
            social,
            social_weight=50,
            choice_seed_value=5,
            timesteps=3,
            on_step=lambda *_args: None,
        )
        with_log = list(recorded)
        recorded.clear()
        before = len(recorded)
        popularity_scores(social, "agent_0000")
        assert len(recorded) == before
    finally:
        experiment.random.Random = original
    assert without_log == with_log
    expected = 0
    for timestep in range(3):
        for agent_id in sorted(locations):
            expected += 1
            if agent_reconsiders(5, timestep, agent_id, REFERENCE_RECONSIDERATION_RATE):
                expected += 1
                assert choice_seed(5, timestep, agent_id) in without_log
    assert len(without_log) == expected


def test_classification_rule_is_fixed() -> None:
    base = _bundle(0.22, 1.58, 0.20, 0.04, "site_02", 0.30, 0.22, 0.30, 0.20)
    amplified = {
        "S0": base,
        "S50": _bundle(0.24, 1.50, 0.30, 0.06, "site_02", 0.40, 0.26, 0.40, 0.30),
        "S150": _bundle(0.28, 1.40, 0.45, 0.09, "site_02", 0.55, 0.34, 0.60, 0.45),
        "S150_X": _bundle(0.23, 1.56, 0.22, 0.045, "site_02", 0.32, 0.23, 0.32, 0.22),
    }
    assert classify_result(amplified)["label"] == "endogenous social feedback amplified collective path dependence"
    centralized = {
        "S0": base,
        "S50": _bundle(0.25, 1.50, 0.22, 0.03, "site_02", 0.40, 0.28, 0.45, 0.35),
        "S150": _bundle(0.30, 1.40, 0.24, 0.02, "site_02", 0.60, 0.36, 0.70, 0.50),
        "S150_X": _bundle(0.30, 1.40, 0.24, 0.02, "site_02", 0.60, 0.36, 0.70, 0.50),
    }
    assert classify_result(centralized)["label"] == "structural centralization"
    flat = {name: dict(base) for name in ("S0", "S50", "S150", "S150_X")}
    assert classify_result(flat)["label"] == "no durable collective social effect"


def test_zero_llm_calls_and_place_ids_stay_out_of_the_utility_model() -> None:
    source = (ROOT / "world" / "experiment_004.py").read_text(encoding="utf-8")
    assert "StructuredLLMClient" not in source
    assert "build_utility_messages" not in source
    assert "PlaceUtility" not in source
    assert "visual_attractiveness" not in source
    assert "atlas.lookup(agent_id, place_id, distance, occupancy[place_id], capacity)" in source


def world_select(utilities, draw):
    from world.experiment_004 import _select

    return _select(list(PLACE_IDS), utilities, draw)


def _zero_scores(locations):
    return {agent_id: {place_id: 0.0 for place_id in PLACE_IDS} for agent_id in locations}


def _bundle(hhi, entropy, retention, between_tv, modal, proportion, share, frequency, duration):
    return {
        "late_hhi": hhi,
        "late_entropy": entropy,
        "retention": retention,
        "between_tv": between_tv,
        "modal_place": modal,
        "modal_proportion": proportion,
        "site_02_share": share,
        "site_02_dominance_frequency": frequency,
        "site_02_dominance_duration": duration,
    }


def _r0_locations(seed: int) -> list[dict[str, str]]:
    rows = []
    path = ROOT / "data" / "worlds" / "world_v0_3" / "world_states.jsonl"
    needle = f'"choice_seed":{seed},'
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if '"condition_id":"R0"' not in line or needle not in line:
                continue
            import json

            row = json.loads(line)
            if row["condition_id"] == "R0" and int(row["choice_seed"]) == seed:
                rows.append((int(row["timestep"]), row["locations"]))
    rows.sort(key=lambda item: item[0])
    return [item[1] for item in rows[:501]]
