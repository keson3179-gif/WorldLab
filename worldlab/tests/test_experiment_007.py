"""World v0.7 early-perturbation tests. No model calls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import world.experiment_007 as experiment
from world.choice import agent_reconsiders, choice_seed
from world.dynamics_calibration import load_initial_locations
from world.experiment_002 import REFERENCE_RECONSIDERATION_RATE
from world.experiment_004 import _tv, trajectory_hash
from world.experiment_007 import (
    ALTERNATION_UPPER_QUARTILE,
    CHOICE_SEEDS,
    CONSECUTIVE_TV_UPPER_QUARTILE,
    EXTREME_DOMINANT_SHARE,
    EXTREME_LATE_HHI,
    EXTREME_SATURATION,
    INITIAL_TV,
    INTERVENTION_STEP,
    REGIMES,
    TIMESTEPS,
    classify_persistence,
    intervention_hash,
    overall_conclusion,
    plan_intervention,
    ranked_places,
    simulate_branch,
)
from world.geography import PLACE_IDS, occupancy_of
from world.utility_atlas import load_frozen_atlas

ROOT = Path(__file__).resolve().parents[1]
FROZEN_SHA256 = {
    "data/worlds/world_v0_4/run_manifest.json": "e19c6354ac9cd72ea46889f2761ae7b839af92061b9cae442cd8e1ee1a9c56a8",
    "data/worlds/world_v0_4/experiment_004_summary.json": "3b138dc7d492451f70d544551f933771d1b85a821e34f13fe8b8f31a3044072d",
    "data/worlds/world_v0_5/run_manifest.json": "23fcd69024fb37690362d43540aac646f683a6098b609680e80db1e801734a27",
    "data/worlds/world_v0_5/experiment_005_summary.json": "d76ff4c7a8ce303b8203af78df87e48c278bcb44d75319595065b50771891041",
    "data/worlds/world_v0_6/run_manifest.json": "d038d8d3ab07d1dac5473285183d42f663d6e0820aa7444bb65d4c52e040bcd5",
    "data/worlds/world_v0_6/experiment_006_summary.json": "f8b5cb55300eb093fa04d0d0866904d7c113b39aa3e9c245753599fc5705ea6e",
    "data/worlds/utility_atlas_v1/atlas_manifest.json": "90ff0feb2f846cf521d4a8d9e1b24e0e8f799e4a409a51c84daea374f744f9f1",
    "data/personas/persona_set_v1/manifest.json": "27b121d100c95a40adc4f1cbf282dc3e17f72a8b5f7d5c5b7926866e1cee2c15",
}


@pytest.fixture(scope="module")
def atlas():
    return load_frozen_atlas()


@pytest.fixture(scope="module")
def locations():
    return load_initial_locations()


def test_preregistered_design_and_frozen_inputs_are_unchanged() -> None:
    assert TIMESTEPS == 500
    assert CHOICE_SEEDS == tuple(range(50))
    assert len(REGIMES) * len(CHOICE_SEEDS) * 2 == 300
    assert [item["regime_id"] for item in REGIMES] == ["S0", "RW50", "EXP99"]
    assert ALTERNATION_UPPER_QUARTILE == 0.694375
    assert CONSECUTIVE_TV_UPPER_QUARTILE == 0.36148125
    assert EXTREME_LATE_HHI == 0.40
    assert EXTREME_DOMINANT_SHARE == 0.60
    assert EXTREME_SATURATION == 0.10
    for relative, digest in FROZEN_SHA256.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest
    source = (ROOT / "world" / "experiment_007.py").read_text(encoding="utf-8")
    assert "StructuredLLMClient" not in source
    assert "build_utility_messages" not in source
    manifest = json.loads((ROOT / "data" / "worlds" / "utility_atlas_v1" / "atlas_manifest.json").read_text(encoding="utf-8"))
    assert manifest["sha256"] == "664b511d208f7989e8a671d0e2d6e6433366120bb3d5d6a67db38277b4d5a373"


def test_target_tie_break_and_agent_selection_are_deterministic() -> None:
    tied = {place_id: 4 for place_id in PLACE_IDS}
    assert ranked_places(tied) == list(PLACE_IDS)
    assert plan_intervention("RW50", 3, tied, {f"agent_{index:04d}": "site_01" for index in range(20)})["target_place"] == "site_02"
    occupancy = {"site_01": 2, "site_02": 2, "site_03": 0, "site_04": 0, "site_05": 10}
    assert ranked_places(occupancy)[1] == "site_01"
    proposed = {f"agent_{index:04d}": "site_03" for index in range(20)}
    first = plan_intervention("EXP99", 7, occupancy, proposed)
    second = plan_intervention("EXP99", 7, occupancy, proposed)
    assert first["selected_agents"] == second["selected_agents"]
    all_ordered = sorted(proposed, key=lambda agent_id: intervention_hash("EXP99", 7, agent_id))
    assert first["selected_agents"] == all_ordered[:2]
    with pytest.raises(RuntimeError):
        plan_intervention("S0", 1, occupancy, {agent_id: "site_01" for agent_id in proposed})


def test_intervention_is_matched_and_changes_only_t21(atlas, locations) -> None:
    spec = REGIMES[1]
    recorded: list[int] = []
    original = experiment.random.Random

    class Tracking:
        def __init__(self, seed: int) -> None:
            recorded.append(seed)
            self._rng = original(seed)

        def random(self) -> float:
            return self._rng.random()

    experiment.random.Random = Tracking
    try:
        control_seeds = _recorded_run(atlas, locations, spec, recorded, apply=False)
        perturbed_seeds = _recorded_run(atlas, locations, spec, recorded, apply=True)
    finally:
        experiment.random.Random = original
    assert control_seeds == perturbed_seeds
    expected = 0
    for timestep in range(30):
        for agent_id in sorted(locations):
            expected += 1
            if agent_reconsiders(4, timestep, agent_id, REFERENCE_RECONSIDERATION_RATE):
                expected += 1
                assert choice_seed(4, timestep, agent_id) in control_seeds
    assert len(control_seeds) == expected
    control = simulate_branch(atlas, locations, spec, choice_seed_value=4, apply_intervention=False, timesteps=30)
    perturbed = simulate_branch(atlas, locations, spec, choice_seed_value=4, apply_intervention=True, timesteps=30)
    assert control["series"][:INTERVENTION_STEP] == perturbed["series"][:INTERVENTION_STEP]
    assert control["override_steps"] == []
    assert perturbed["override_steps"] == [INTERVENTION_STEP]
    assert perturbed["intervention"]["effective_overrides"] == 2
    target = ranked_places(occupancy_of(control["series"][20]))[1]
    assert perturbed["intervention"]["target_place"] == target
    assert perturbed["intervention"]["selected_agents"] == control["intervention"]["selected_agents"]
    tv = _tv(_shares(control["series"][21]), _shares(perturbed["series"][21]))
    assert tv == pytest.approx(INITIAL_TV)
    changed = [
        agent_id
        for agent_id in control["series"][21]
        if control["series"][21][agent_id] != perturbed["series"][21][agent_id]
    ]
    assert changed == perturbed["intervention"]["selected_agents"] or set(changed) == set(perturbed["intervention"]["selected_agents"])
    forward = simulate_branch(atlas, locations, spec, choice_seed_value=4, apply_intervention=True, timesteps=30)
    backward = simulate_branch(
        atlas,
        {agent_id: locations[agent_id] for agent_id in reversed(list(locations))},
        spec,
        choice_seed_value=4,
        apply_intervention=True,
        timesteps=30,
    )
    replay = simulate_branch(atlas, locations, spec, choice_seed_value=4, apply_intervention=True, timesteps=30)
    assert forward["series"] == backward["series"]
    assert trajectory_hash(forward["series"]) == trajectory_hash(replay["series"])
    seen = []

    def capture(logged, _current, _memory, actions, _plan):
        if logged == 1:
            seen.append(actions)

    simulate_branch(atlas, locations, spec, choice_seed_value=4, apply_intervention=False, timesteps=30, on_step=capture)
    assert len({tuple(sorted(action["occupancy_before"].items())) for action in seen[0]}) == 1


def test_persistence_classes_follow_the_preregistered_cutoffs() -> None:
    assert classify_persistence(0.024) == "attenuated"
    assert classify_persistence(0.025) == "persistent residual"
    assert classify_persistence(0.099) == "persistent residual"
    assert classify_persistence(0.10) == "amplified divergence"
    assert overall_conclusion({"S0": "attenuated", "RW50": "attenuated", "EXP99": "attenuated"}).startswith(
        "small early perturbations"
    )
    assert overall_conclusion({"S0": "attenuated", "RW50": "persistent residual", "EXP99": "attenuated"}).startswith(
        "early perturbations leave"
    )
    assert overall_conclusion({"S0": "attenuated", "RW50": "amplified divergence", "EXP99": "persistent residual"}).startswith(
        "early perturbation divergence can amplify"
    )


def test_s0_and_reference_controls_match_frozen_trajectories(atlas, locations) -> None:
    s0 = json.loads((ROOT / "data" / "worlds" / "world_v0_4" / "run_manifest.json").read_text(encoding="utf-8"))
    later = json.loads((ROOT / "data" / "worlds" / "world_v0_6" / "run_manifest.json").read_text(encoding="utf-8"))
    for seed in CHOICE_SEEDS:
        result = simulate_branch(
            atlas, locations, REGIMES[0], choice_seed_value=seed, apply_intervention=False, timesteps=500
        )
        assert trajectory_hash(result["series"]) == s0["trajectory_sha256"][f"S0:{seed}"]
    assert s0["trajectory_sha256"]["S0:0"] == "610cb9dfff650d135bf2520cac4247e7b2707073fae6e0fa52f094a040175955"
    for spec, key in ((REGIMES[1], "RW50:100:0"), (REGIMES[2], "EXP99:100:0")):
        result = simulate_branch(atlas, locations, spec, choice_seed_value=0, apply_intervention=False, timesteps=500)
        assert trajectory_hash(result["series"]) == later["trajectory_sha256"][key]


def _recorded_run(atlas, locations, spec, recorded, *, apply: bool) -> list[int]:
    recorded.clear()
    simulate_branch(atlas, locations, spec, choice_seed_value=4, apply_intervention=apply, timesteps=30)
    return list(recorded)


def _shares(locations: dict[str, str]) -> dict[str, float]:
    counts = occupancy_of(locations)
    population = len(locations)
    return {place_id: counts[place_id] / population for place_id in PLACE_IDS}
