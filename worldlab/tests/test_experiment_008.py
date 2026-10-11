"""World v0.8 symmetric alternative-history tests. No model calls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import world.experiment_008 as experiment
from world.choice import agent_reconsiders, choice_seed
from world.dynamics_calibration import load_initial_locations
from world.experiment_002 import REFERENCE_RECONSIDERATION_RATE
from world.experiment_004 import _tv, trajectory_hash
from world.experiment_008 import (
    ALTERNATION_UPPER_QUARTILE,
    BRANCHES,
    CHOICE_SEEDS,
    CONSECUTIVE_TV_UPPER_QUARTILE,
    DIRECTIONAL_THRESHOLD,
    EXTREME_DOMINANT_SHARE,
    EXTREME_LATE_HHI,
    EXTREME_SATURATION,
    INITIAL_TV,
    INTERVENTION_STEP,
    REGIMES,
    TIMESTEPS,
    _bootstrap_ci,
    _paired,
    classify_direction,
    classify_outcome,
    directional_history_index,
    intervention_hash,
    overall_conclusion,
    select_agents,
    select_history_pair,
    simulate_branch,
)
from world.geography import PLACE_IDS, occupancy_of
from world.utility_atlas import load_frozen_atlas

ROOT = Path(__file__).resolve().parents[1]
FROZEN_SHA256 = {
    "data/worlds/world_v0_4/run_manifest.json": "e19c6354ac9cd72ea46889f2761ae7b839af92061b9cae442cd8e1ee1a9c56a8",
    "data/worlds/world_v0_5/run_manifest.json": "23fcd69024fb37690362d43540aac646f683a6098b609680e80db1e801734a27",
    "data/worlds/world_v0_6/run_manifest.json": "d038d8d3ab07d1dac5473285183d42f663d6e0820aa7444bb65d4c52e040bcd5",
    "data/worlds/world_v0_7/run_manifest.json": "dc1e1f26e37c02f96c3331282a9438d6ef9bfa4cae6cf629d566093079a660fb",
    "data/worlds/world_v0_7/experiment_007_summary.json": "b9c69a0adfbfb518337c993325009fbbb5f070967c4010a1e27ae9fbf3556cc7",
    "data/worlds/utility_atlas_v1/atlas_manifest.json": "90ff0feb2f846cf521d4a8d9e1b24e0e8f799e4a409a51c84daea374f744f9f1",
    "data/personas/persona_set_v1/manifest.json": "27b121d100c95a40adc4f1cbf282dc3e17f72a8b5f7d5c5b7926866e1cee2c15",
}


@pytest.fixture(scope="module")
def atlas():
    return load_frozen_atlas()


@pytest.fixture(scope="module")
def locations():
    return load_initial_locations()


def test_preregistered_design_and_frozen_inputs() -> None:
    assert TIMESTEPS == 500
    assert CHOICE_SEEDS == tuple(range(50))
    assert BRANCHES == ("CONTROL", "HISTORY_A", "HISTORY_B")
    assert len(REGIMES) * len(CHOICE_SEEDS) * len(BRANCHES) == 450
    assert DIRECTIONAL_THRESHOLD == 0.01
    assert ALTERNATION_UPPER_QUARTILE == 0.694375
    assert CONSECUTIVE_TV_UPPER_QUARTILE == 0.36148125
    assert EXTREME_LATE_HHI == 0.40
    assert EXTREME_DOMINANT_SHARE == 0.60
    assert EXTREME_SATURATION == 0.10
    for relative, digest in FROZEN_SHA256.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest
    source = (ROOT / "world" / "experiment_008.py").read_text(encoding="utf-8")
    assert "StructuredLLMClient" not in source
    assert "build_utility_messages" not in source
    manifest = json.loads((ROOT / "data" / "worlds" / "utility_atlas_v1" / "atlas_manifest.json").read_text(encoding="utf-8"))
    assert manifest["sha256"] == "664b511d208f7989e8a671d0e2d6e6433366120bb3d5d6a67db38277b4d5a373"


def test_history_pair_and_agent_selection_rules() -> None:
    occupancy = {"site_01": 10, "site_02": 4, "site_03": 3, "site_04": 2, "site_05": 1}
    mass = {"site_01": 8.0, "site_02": 3.0, "site_03": 3.2, "site_04": 4.0, "site_05": 1.8}
    chosen = select_history_pair(occupancy, mass)
    assert chosen["current_dominant"] == "site_01"
    assert {chosen["target_A"], chosen["target_B"]} == {"site_02", "site_03"}
    assert chosen["target_A"] < chosen["target_B"]
    tied_mass = {"site_01": 1.0, "site_02": 2.0, "site_03": 2.0, "site_04": 5.0, "site_05": 5.0}
    tied_occupancy = {"site_01": 1, "site_02": 1, "site_03": 1, "site_04": 1, "site_05": 16}
    tied = select_history_pair(tied_occupancy, tied_mass)
    assert (tied["target_A"], tied["target_B"]) == ("site_02", "site_03")
    proposed = {f"agent_{index:04d}": "site_05" for index in range(20)}
    first = select_agents("RW50", 4, proposed, "site_02", "site_04")
    assert first == select_agents("RW50", 4, proposed, "site_02", "site_04")
    ordered = sorted(proposed, key=lambda agent_id: intervention_hash("RW50", 4, agent_id))
    assert first == ordered[:2]
    assert all(proposed[agent_id] not in {"site_02", "site_04"} for agent_id in first)
    blocked = {agent_id: "site_02" if index < 19 else "site_04" for index, agent_id in enumerate(proposed)}
    with pytest.raises(RuntimeError):
        select_agents("S0", 1, blocked, "site_02", "site_04")


def test_symmetric_fork_is_matched_and_confined(atlas, locations) -> None:
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
        draws = [_recorded(atlas, locations, spec, recorded, branch) for branch in BRANCHES]
    finally:
        experiment.random.Random = original
    assert draws[0] == draws[1] == draws[2]
    expected = 0
    for timestep in range(30):
        for agent_id in sorted(locations):
            expected += 1
            if agent_reconsiders(8, timestep, agent_id, REFERENCE_RECONSIDERATION_RATE):
                expected += 1
    assert len(draws[0]) == expected
    runs = {
        branch: simulate_branch(atlas, locations, spec, choice_seed_value=8, branch=branch, timesteps=30)
        for branch in BRANCHES
    }
    assert runs["CONTROL"]["series"][:INTERVENTION_STEP] == runs["HISTORY_A"]["series"][:INTERVENTION_STEP]
    assert runs["CONTROL"]["series"][:INTERVENTION_STEP] == runs["HISTORY_B"]["series"][:INTERVENTION_STEP]
    assert runs["CONTROL"]["override_steps"] == []
    assert runs["HISTORY_A"]["override_steps"] == [INTERVENTION_STEP]
    assert runs["HISTORY_B"]["override_steps"] == [INTERVENTION_STEP]
    assert runs["HISTORY_A"]["plan"]["selected_agents"] == runs["HISTORY_B"]["plan"]["selected_agents"]
    assert runs["HISTORY_A"]["plan"]["effective_overrides"] == 2
    assert runs["HISTORY_B"]["plan"]["effective_overrides"] == 2
    plan = runs["CONTROL"]["plan"]
    assert plan["target_A"] != plan["target_B"]
    assert plan["current_dominant"] not in {plan["target_A"], plan["target_B"]}
    assert set(plan["control_proposed_destinations"]).isdisjoint({plan["target_A"], plan["target_B"]})
    gaps = [
        abs(plan["expected_choice_mass"][left] - plan["expected_choice_mass"][right])
        for left in PLACE_IDS
        if left != plan["current_dominant"]
        for right in PLACE_IDS
        if right > left and right != plan["current_dominant"]
    ]
    assert plan["absolute_gap"] == min(gaps)
    for left, right in (
        (runs["CONTROL"], runs["HISTORY_A"]),
        (runs["CONTROL"], runs["HISTORY_B"]),
        (runs["HISTORY_A"], runs["HISTORY_B"]),
    ):
        assert _tv(_shares(left["series"][21]), _shares(right["series"][21])) == pytest.approx(INITIAL_TV)
    seen = []

    def capture(logged, _current, _memory, actions, _plan):
        if logged == 1:
            seen.append(actions)

    simulate_branch(atlas, locations, spec, choice_seed_value=8, branch="CONTROL", timesteps=30, on_step=capture)
    assert len({tuple(sorted(action["occupancy_before"].items())) for action in seen[0]}) == 1
    forward = simulate_branch(atlas, locations, spec, choice_seed_value=8, branch="HISTORY_A", timesteps=30)
    backward = simulate_branch(
        atlas,
        {agent_id: locations[agent_id] for agent_id in reversed(list(locations))},
        spec,
        choice_seed_value=8,
        branch="HISTORY_A",
        timesteps=30,
    )
    assert trajectory_hash(forward["series"]) == trajectory_hash(backward["series"])


def test_dhi_taxonomy_and_paired_bootstrap() -> None:
    assert directional_history_index(0.4, 0.2, 0.5, 0.1) == pytest.approx(0.3)
    assert classify_direction(-0.01, 0.1) == "no directional retention"
    assert classify_direction(0.005, 0.001) == "weak directional retention"
    assert classify_direction(0.02, 0.01) == "directionally consistent"
    assert classify_direction(0.02, -0.001) == "directional signal with uncertainty"
    assert classify_outcome(0.02, 0.05, 0.01) == "type_0"
    assert classify_outcome(0.05, 0.0, -0.01) == "type_1"
    assert classify_outcome(0.05, 0.02, 0.01) == "type_2"
    assert classify_outcome(0.10, 0.02, 0.01) == "type_3"
    assert overall_conclusion({"S0": "type_0", "RW50": "type_0", "EXP99": "type_0"}).startswith("symmetric alternative")
    assert overall_conclusion({"S0": "type_0", "RW50": "type_1", "EXP99": "type_0"}).startswith("symmetric alternative early histories produce persistent but")
    assert overall_conclusion({"S0": "type_0", "RW50": "type_2", "EXP99": "type_1"}).startswith("experimentally assigned alternative early histories produce persistent directional")
    assert "amplified" in overall_conclusion({"S0": "type_0", "RW50": "type_3", "EXP99": "type_2"})
    left = [{"choice_seed": 0, "late_tv": 0.0, "persistence_ratio": 0.0, "late_dhi": 0.0, "final_dominant_disagreement": False}]
    right = [{"choice_seed": 0, "late_tv": 0.1, "persistence_ratio": 1.0, "late_dhi": 0.02, "final_dominant_disagreement": True}]
    paired = _paired(left, right)
    assert paired["late_tv"]["mean_difference"] == pytest.approx(0.1)
    low, high = _bootstrap_ci([0.1, 0.2, 0.3, 0.4])
    assert low <= statistics_mean([0.1, 0.2, 0.3, 0.4]) <= high


def test_recorded_run_keeps_rules_and_replays(atlas, locations) -> None:
    output = ROOT / "data" / "worlds" / "world_v0_8"
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "experiment_008_summary.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is True
    assert manifest["world_count"] == 450
    assert manifest["llm_calls"] == 0
    assert summary["outcome_type"] == {"S0": "type_0", "RW50": "type_1", "EXP99": "type_1"}
    assert summary["conclusion"] == overall_conclusion(summary["outcome_type"])
    records = [
        json.loads(line)
        for line in (output / "alternative_history_interventions.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(records) == 150
    assert {record["choice_seed"] for record in records} == set(range(50))
    for record in records:
        assert record["target_A"] != record["target_B"]
        assert record["current_dominant"] not in {record["target_A"], record["target_B"]}
        assert record["override_count_A"] == record["override_count_B"] == 2
        assert len(record["selected_agents"]) == 2
        assert set(record["control_proposed_destinations"]).isdisjoint({record["target_A"], record["target_B"]})
        assert record["initial_tv_control_A"] == pytest.approx(0.10)
        assert record["initial_tv_control_B"] == pytest.approx(0.10)
        assert record["initial_tv_A_B"] == pytest.approx(0.10)
        assert record["absolute_gap"] == pytest.approx(abs(record["expected_mass_A"] - record["expected_mass_B"]))
    requested = (("S0", "CONTROL", 0), ("RW50", "HISTORY_A", 0), ("EXP99", "HISTORY_B", 0))
    for regime_id, branch, seed in requested:
        spec = next(item for item in REGIMES if item["regime_id"] == regime_id)
        result = simulate_branch(atlas, locations, spec, choice_seed_value=seed, branch=branch)
        assert result["trajectory_sha256"] == manifest["trajectory_sha256"][f"{regime_id}:{seed}:{branch}"]
        assert result["llm_calls"] == 0


def test_controls_match_frozen_lineage(atlas, locations) -> None:
    s0 = json.loads((ROOT / "data" / "worlds" / "world_v0_4" / "run_manifest.json").read_text(encoding="utf-8"))
    later = json.loads((ROOT / "data" / "worlds" / "world_v0_6" / "run_manifest.json").read_text(encoding="utf-8"))
    for seed in CHOICE_SEEDS:
        result = simulate_branch(atlas, locations, REGIMES[0], choice_seed_value=seed, branch="CONTROL")
        assert result["trajectory_sha256"] == s0["trajectory_sha256"][f"S0:{seed}"]
    assert s0["trajectory_sha256"]["S0:0"] == "610cb9dfff650d135bf2520cac4247e7b2707073fae6e0fa52f094a040175955"
    for spec, key in ((REGIMES[1], "RW50:100:0"), (REGIMES[2], "EXP99:100:0")):
        result = simulate_branch(atlas, locations, spec, choice_seed_value=0, branch="CONTROL")
        assert result["trajectory_sha256"] == later["trajectory_sha256"][key]


def _recorded(atlas, locations, spec, recorded, branch) -> list[int]:
    recorded.clear()
    simulate_branch(atlas, locations, spec, choice_seed_value=8, branch=branch, timesteps=30)
    return list(recorded)


def _shares(locations: dict[str, str]) -> dict[str, float]:
    counts = occupancy_of(locations)
    population = len(locations)
    return {place_id: counts[place_id] / population for place_id in PLACE_IDS}


def statistics_mean(values: list[float]) -> float:
    return sum(values) / len(values)
