"""World v0.6 social-weight by memory-timescale tests. No model calls."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import world.experiment_005 as memory_experiment
import world.experiment_006 as experiment
from world.choice import agent_reconsiders, choice_seed
from world.dynamics_calibration import load_initial_locations
from world.experiment_002 import REFERENCE_RECONSIDERATION_RATE
from world.experiment_004 import trajectory_hash
from world.experiment_005 import (
    CENTERED_NORMALIZATION,
    DecayMemory,
    RollingMemory,
    centered_scores,
    clip_score,
    simulate_memory,
)
from world.experiment_006 import (
    ALTERNATION_UPPER_QUARTILE,
    ARCHITECTURES,
    CONSECUTIVE_TV_UPPER_QUARTILE,
    EXTREME_DOMINANT_SHARE,
    EXTREME_LATE_HHI,
    EXTREME_SATURATION,
    WEIGHTS,
    classify_regime,
    dose_shape,
    select_reference,
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


def test_centered_normalization_is_shared_and_weight_scales_the_score(atlas, locations) -> None:
    assert [item["condition_id"] for item in ARCHITECTURES] == ["RW20", "RW50", "EXP95", "EXP99"]
    assert experiment.CENTERED_NORMALIZATION == CENTERED_NORMALIZATION
    assert experiment.simulate_memory is simulate_memory
    assert "CUM" not in {item["condition_id"] for item in ARCHITECTURES}
    scores = {place_id: float(index - 2) for index, place_id in enumerate(PLACE_IDS)}
    low = _utilities_scale(atlas, locations, scores, 25.0)
    high = _utilities_scale(atlas, locations, scores, 150.0)
    assert low["scores"] == high["scores"] == scores
    for place_id in PLACE_IDS:
        assert low["social"][place_id] == pytest.approx(25.0 * scores[place_id])
        assert high["social"][place_id] == pytest.approx(6.0 * low["social"][place_id])
        assert high["totals"][place_id] == pytest.approx(high["bases"][place_id] + high["social"][place_id])
    assert memory_experiment.SOCIAL_WEIGHT == 150.0


def test_weight_does_not_change_memory_under_a_matched_trajectory() -> None:
    agents = ["agent_0000", "agent_0001"]
    commits = [
        {"agent_0000": "site_01", "agent_0001": "site_02"},
        {"agent_0000": "site_01", "agent_0001": "site_01"},
        {"agent_0000": "site_04", "agent_0001": "site_03"},
    ]
    rolling = [RollingMemory(agents, 20) for _weight in WEIGHTS]
    decay = [DecayMemory(agents, 0.95) for _weight in WEIGHTS]
    for destinations in commits:
        for memory in rolling + decay:
            memory.commit(dict(destinations))
    assert all(memory.total == rolling[0].total and memory.own == rolling[0].own for memory in rolling)
    assert all(memory.total == decay[0].total and memory.own == decay[0].own for memory in decay)


def test_window_decay_exclusion_lag_and_clip() -> None:
    agents = ["agent_0000", "agent_0001"]
    rolling = RollingMemory(agents, 20)
    assert rolling.loo_raw("agent_0000") == {place_id: 0.0 for place_id in PLACE_IDS}
    for _step in range(25):
        rolling.commit({"agent_0000": "site_01", "agent_0001": "site_02"})
    assert len(rolling.commits) == 20
    assert rolling.loo_raw("agent_0000")["site_01"] == 0.0
    assert rolling.loo_raw("agent_0001")["site_01"] == 20.0
    decay = DecayMemory(agents, 0.99)
    decay.commit({"agent_0000": "site_01", "agent_0001": "site_01"})
    assert decay.loo_raw("agent_0000")["site_01"] == pytest.approx(decay.total["site_01"] - decay.own["agent_0000"]["site_01"])
    alone = DecayMemory(["agent_0000"], 0.95)
    alone.commit({"agent_0000": "site_01"})
    alone.commit({"agent_0000": "site_01"})
    assert alone.loo_raw("agent_0000")["site_01"] == pytest.approx(0.0)
    _unclipped, clipped = centered_scores({place_id: 3.0 for place_id in PLACE_IDS})
    assert clipped == {place_id: 0.0 for place_id in PLACE_IDS}
    assert clip_score(4.0) == 2.0
    assert clip_score(-4.0) == -2.0
    extreme = {place_id: 0.0 for place_id in PLACE_IDS}
    extreme["site_01"] = 50.0
    assert max(centered_scores(extreme)[1].values()) == pytest.approx(2.0)


def test_matched_rng_sync_order_and_replay(atlas, locations) -> None:
    recorded: list[int] = []
    original = memory_experiment.random.Random

    class Tracking:
        def __init__(self, seed: int) -> None:
            recorded.append(seed)
            self._rng = original(seed)

        def random(self) -> float:
            return self._rng.random()

    seen = []

    def capture(logged, _locations, memory, actions):
        if logged == 1:
            seen.append((dict(memory.total), actions))

    memory_experiment.random.Random = Tracking
    try:
        with experiment._weight(50.0):
            logged = simulate_memory(
                atlas,
                locations,
                RollingMemory(list(locations), 20),
                choice_seed_value=4,
                timesteps=3,
                on_step=capture,
            )
            silent = simulate_memory(
                atlas,
                locations,
                RollingMemory(list(locations), 20),
                choice_seed_value=4,
                timesteps=3,
            )
    finally:
        memory_experiment.random.Random = original
    expected = 0
    for timestep in range(3):
        for agent_id in sorted(locations):
            expected += 1
            if agent_reconsiders(4, timestep, agent_id, REFERENCE_RECONSIDERATION_RATE):
                expected += 1
                assert choice_seed(4, timestep, agent_id) in recorded
    assert len(recorded) == expected * 2
    assert logged["series"] == silent["series"]
    total, actions = seen[0]
    assert all(score == 0.0 for action in actions for score in action["popularity_score"].values())
    assert sum(total.values()) == len(locations)
    assert len({tuple(sorted(action["occupancy_before"].items())) for action in actions}) == 1
    with experiment._weight(25.0):
        forward = simulate_memory(atlas, locations, RollingMemory(list(locations), 20), choice_seed_value=3, timesteps=4)
        backward = simulate_memory(
            atlas,
            {agent_id: locations[agent_id] for agent_id in reversed(list(locations))},
            RollingMemory(list(reversed(list(locations))), 20),
            choice_seed_value=3,
            timesteps=4,
        )
        replay = simulate_memory(atlas, locations, RollingMemory(list(locations), 20), choice_seed_value=3, timesteps=4)
    assert forward["series"] == backward["series"]
    assert trajectory_hash(forward["series"]) == trajectory_hash(replay["series"])
    assert memory_experiment.SOCIAL_WEIGHT == 150.0


def test_regime_dose_and_reference_rules_are_fixed() -> None:
    assert ALTERNATION_UPPER_QUARTILE == 0.694375
    assert CONSECUTIVE_TV_UPPER_QUARTILE == 0.36148125
    assert EXTREME_LATE_HHI == 0.40
    assert EXTREME_DOMINANT_SHARE == 0.60
    assert EXTREME_SATURATION == 0.10
    assert classify_regime(0.05, False) == "weak influence"
    assert classify_regime(0.20, False) == "moderate influence"
    assert classify_regime(0.40, False) == "strong-stable influence"
    assert classify_regime(0.05, True) == "unstable influence"
    assert dose_shape([0.1, 0.2, 0.3, 0.4], 0.01) == "monotonic increasing"
    assert dose_shape([0.4, 0.3, 0.2, 0.1], 0.01) == "monotonic decreasing"
    assert dose_shape([0.1, 0.4, 0.2, 0.5], 0.01) == "non-monotonic"
    assert dose_shape([0.10, 0.101, 0.100, 0.102], 0.01) == "approximately flat"
    near = _cell("RW20", 50, destination=0.24, clip=0.02, tv=0.05)
    other = _cell("EXP99", 150, destination=0.31, clip=0.01, tv=0.20)
    assert select_reference([near, other], 0.02)["selection"]["cell_id"] == "RW20_w50"
    assert select_reference([_cell("RW20", 150, destination=0.50)], 0.02)["selection"] == "none"
    assert select_reference([_cell("RW20", 50, unstable=True)], 0.02)["selection"] == "none"


def test_zero_llm_calls_and_frozen_atlas_lookup() -> None:
    source = (ROOT / "world" / "experiment_006.py").read_text(encoding="utf-8")
    assert "StructuredLLMClient" not in source
    assert "build_utility_messages" not in source
    assert "PlaceUtility" not in source
    assert "math.log1p" not in source
    manifest = json.loads((ROOT / "data" / "worlds" / "utility_atlas_v1" / "atlas_manifest.json").read_text(encoding="utf-8"))
    assert manifest["sha256"] == "664b511d208f7989e8a671d0e2d6e6433366120bb3d5d6a67db38277b4d5a373"


@pytest.mark.slow
def test_weight_150_reproduces_experiment_005_seeds(atlas, locations) -> None:
    manifest = json.loads((ROOT / "data" / "worlds" / "world_v0_5" / "run_manifest.json").read_text(encoding="utf-8"))
    for spec in ARCHITECTURES:
        for seed in range(50):
            with experiment._weight(150.0):
                result = simulate_memory(
                    atlas,
                    locations,
                    memory_experiment._make_memory(spec, list(locations)),
                    choice_seed_value=seed,
                    timesteps=500,
                )
            assert trajectory_hash(result["series"]) == manifest["trajectory_sha256"][f"{spec['condition_id']}:{seed}"]
    assert memory_experiment.SOCIAL_WEIGHT == 150.0


def _utilities_scale(atlas, locations, scores, weight):
    from world.experiment_005 import _utilities
    from world.geography import distance_matrix, occupancy_of, place_index

    agent_id = sorted(locations)[0]
    with experiment._weight(weight):
        bases, totals, social = _utilities(
            atlas,
            distance_matrix(),
            place_index(),
            agent_id,
            locations[agent_id],
            occupancy_of(locations),
            scores,
        )
    return {"bases": bases, "totals": totals, "social": social, "scores": scores}


def _cell(architecture, weight, *, destination=0.25, ranking=0.40, clip=0.01, tv=0.05, unstable=False, places=None):
    return {
        "cell_id": f"{architecture}_w{weight:g}",
        "architecture": architecture,
        "social_weight": float(weight),
        "unstable": unstable,
        "destination_change_rate": destination,
        "ranking_change_rate": ranking,
        "clipping_rate": clip,
        "between_seed_late_tv": tv,
        "final_dominant_places": ["site_01", "site_02", "site_04"] if places is None else places,
    }
