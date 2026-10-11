"""World v0.5 social-memory architecture tests. No model calls."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from world.choice import agent_reconsiders, choice_seed
from world.dynamics_calibration import load_initial_locations
from world.experiment_002 import REFERENCE_RECONSIDERATION_RATE
from world.experiment_004 import simulate_world as simulate_cumulative
from world.experiment_004 import trajectory_hash
from world.experiment_005 import (
    COMPRESSED_SCORE_SD,
    DecayMemory,
    RollingMemory,
    centered_scores,
    classify_architecture,
    clip_score,
    simulate_memory,
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


def test_rolling_window_length_lag_and_focal_exclusion() -> None:
    agents = ["agent_0000", "agent_0001"]
    memory = RollingMemory(agents, 20)
    assert memory.loo_raw("agent_0000") == {place_id: 0.0 for place_id in PLACE_IDS}
    before_commit = None
    for step in range(25):
        before_commit = dict(memory.total)
        memory.commit({"agent_0000": "site_01", "agent_0001": "site_02"})
        assert memory.loo_raw("agent_0000")["site_01"] == 0.0
        if step < 20:
            assert sum(memory.total.values()) == sum(before_commit.values()) + 2
        else:
            assert sum(memory.total.values()) == sum(before_commit.values())
    assert len(memory.commits) == 20
    assert memory.total["site_01"] == 20
    assert memory.total["site_02"] == 20
    assert memory.loo_raw("agent_0000")["site_01"] == 0.0
    assert memory.loo_raw("agent_0000")["site_02"] == 20.0
    assert memory.loo_raw("agent_0001")["site_01"] == 20.0


def test_decay_is_deterministic_and_excludes_own_contribution() -> None:
    agents = ["agent_0000", "agent_0001"]
    first = DecayMemory(agents, 0.95)
    second = DecayMemory(agents, 0.95)
    commits = [
        {"agent_0000": "site_01", "agent_0001": "site_01"},
        {"agent_0000": "site_01", "agent_0001": "site_02"},
        {"agent_0000": "site_03", "agent_0001": "site_01"},
    ]
    for destinations in commits:
        first.commit(destinations)
        second.commit(dict(reversed(list(destinations.items()))))
    assert first.total == second.total
    assert first.own == second.own
    assert first.loo_raw("agent_0000")["site_03"] == pytest.approx(first.total["site_03"] - first.own["agent_0000"]["site_03"])
    alone = DecayMemory(["agent_0000"], 0.95)
    alone.commit({"agent_0000": "site_01"})
    alone.commit({"agent_0000": "site_01"})
    assert alone.loo_raw("agent_0000")["site_01"] == pytest.approx(0.0)
    assert alone.total["site_01"] == pytest.approx(0.95 + 1.0)


def test_centered_scores_are_zero_mean_and_clipped() -> None:
    zeros, _clipped = centered_scores({place_id: 4.0 for place_id in PLACE_IDS})
    assert zeros == {place_id: 0.0 for place_id in PLACE_IDS}
    raw = {place_id: float(index) for index, place_id in enumerate(PLACE_IDS)}
    unclipped, clipped = centered_scores(raw)
    assert sum(unclipped.values()) == pytest.approx(0.0)
    assert all(-2.0 <= value <= 2.0 for value in clipped.values())
    assert clip_score(3.5) == 2.0
    assert clip_score(-3.5) == -2.0
    extreme = {place_id: 0.0 for place_id in PLACE_IDS}
    extreme["site_01"] = 100.0
    _unclipped, bounded = centered_scores(extreme)
    assert max(bounded.values()) == pytest.approx(2.0)
    assert min(bounded.values()) >= -2.0


def test_cum_matches_s150_and_memory_rng_is_single_draw(atlas, locations) -> None:
    from world.experiment_004 import empty_social

    result = simulate_cumulative(
        atlas,
        locations,
        empty_social(list(locations)),
        social_weight=150,
        choice_seed_value=0,
        timesteps=500,
    )
    assert result["series"] == _s150_locations(0)
    import world.experiment_005 as experiment

    recorded: list[int] = []
    original = experiment.random.Random

    class Tracking:
        def __init__(self, seed: int) -> None:
            recorded.append(seed)
            self._rng = original(seed)

        def random(self) -> float:
            return self._rng.random()

    memory = RollingMemory(list(locations), 20)
    experiment.random.Random = Tracking
    try:
        simulate_memory(atlas, locations, memory, choice_seed_value=5, timesteps=3)
    finally:
        experiment.random.Random = original
    expected = 0
    for timestep in range(3):
        for agent_id in sorted(locations):
            expected += 1
            if agent_reconsiders(5, timestep, agent_id, REFERENCE_RECONSIDERATION_RATE):
                expected += 1
                assert choice_seed(5, timestep, agent_id) in recorded
    assert len(recorded) == expected


def test_synchronous_order_and_replay(atlas, locations) -> None:
    seen = []

    def capture(logged, _locations, memory, actions):
        if logged == 1:
            seen.append((dict(memory.total), actions))

    memory = RollingMemory(list(locations), 20)
    simulate_memory(atlas, locations, memory, choice_seed_value=1, timesteps=2, on_step=capture)
    total, actions = seen[0]
    assert len({tuple(sorted(action["occupancy_before"].items())) for action in actions}) == 1
    assert all(score == 0.0 for action in actions for score in action["popularity_score"].values())
    assert sum(total.values()) == len(locations)
    forward = simulate_memory(atlas, locations, RollingMemory(list(locations), 20), choice_seed_value=2, timesteps=5)
    backward = simulate_memory(
        atlas,
        {agent_id: locations[agent_id] for agent_id in reversed(list(locations))},
        RollingMemory(list(reversed(list(locations))), 20),
        choice_seed_value=2,
        timesteps=5,
    )
    assert forward["series"] == backward["series"]
    assert trajectory_hash(forward["series"]) == trajectory_hash(
        simulate_memory(atlas, locations, RollingMemory(list(locations), 20), choice_seed_value=2, timesteps=5)["series"]
    )


def test_architecture_classes_follow_the_pre_specified_rule() -> None:
    assert COMPRESSED_SCORE_SD == 0.10
    assert classify_architecture(
        score_sd=0.05,
        ranking_change_rate=0.2,
        destination_change_rate=0.2,
        cum_ranking_change_rate=0.01,
        cum_destination_change_rate=0.01,
        unstable=True,
    ) == "signal-compressed"
    assert classify_architecture(
        score_sd=0.4,
        ranking_change_rate=0.2,
        destination_change_rate=0.1,
        cum_ranking_change_rate=0.01,
        cum_destination_change_rate=0.01,
        unstable=False,
    ) == "discriminative-stable"
    assert classify_architecture(
        score_sd=0.4,
        ranking_change_rate=0.2,
        destination_change_rate=0.1,
        cum_ranking_change_rate=0.01,
        cum_destination_change_rate=0.01,
        unstable=True,
    ) == "discriminative-unstable"
    assert classify_architecture(
        score_sd=0.4,
        ranking_change_rate=0.01,
        destination_change_rate=0.01,
        cum_ranking_change_rate=0.02,
        cum_destination_change_rate=0.0,
        unstable=False,
    ) == "weak-discrimination"


def test_zero_llm_calls_and_no_utility_prompt() -> None:
    source = (ROOT / "world" / "experiment_005.py").read_text(encoding="utf-8")
    assert "StructuredLLMClient" not in source
    assert "build_utility_messages" not in source
    assert "PlaceUtility" not in source


def _s150_locations(seed: int) -> list[dict[str, str]]:
    rows = []
    path = ROOT / "data" / "worlds" / "world_v0_4" / "world_states.jsonl"
    needle = f'"condition_id":"S150","choice_seed":{seed},'
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if needle not in line:
                continue
            row = json.loads(line)
            if row["condition_id"] == "S150" and int(row["choice_seed"]) == seed:
                rows.append((int(row["timestep"]), row["locations"]))
    rows.sort(key=lambda item: item[0])
    return [locations for _timestep, locations in rows[:501]]
