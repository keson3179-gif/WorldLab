"""World v0.9 state-carrier tests. No model calls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import world.experiment_009 as experiment
from world.choice import agent_reconsiders, choice_seed
from world.dynamics_calibration import load_initial_locations
from world.experiment_002 import REFERENCE_RECONSIDERATION_RATE
from world.experiment_004 import trajectory_hash
from world.experiment_005 import DecayMemory, RollingMemory
from world.experiment_009 import (
    ALTERNATION_UPPER_QUARTILE,
    BRANCHES,
    CHOICE_SEEDS,
    CONSECUTIVE_TV_UPPER_QUARTILE,
    EXTREME_DOMINANT_SHARE,
    EXTREME_LATE_HHI,
    EXTREME_SATURATION,
    REGIMES,
    TIMESTEPS,
    _bootstrap_ci,
    build_decision_state_inventory,
    classify_carrier,
    classify_persistence,
    install_memory,
    load_intervention_plans,
    memory_payload,
    overall_conclusion,
    retention_fraction,
    simulate_branch,
    state_hashes,
)
from world.utility_atlas import load_frozen_atlas

ROOT = Path(__file__).resolve().parents[1]
FROZEN_SHA256 = {
    "data/worlds/world_v0_4/run_manifest.json": "e19c6354ac9cd72ea46889f2761ae7b839af92061b9cae442cd8e1ee1a9c56a8",
    "data/worlds/world_v0_5/run_manifest.json": "23fcd69024fb37690362d43540aac646f683a6098b609680e80db1e801734a27",
    "data/worlds/world_v0_6/run_manifest.json": "d038d8d3ab07d1dac5473285183d42f663d6e0820aa7444bb65d4c52e040bcd5",
    "data/worlds/world_v0_7/run_manifest.json": "dc1e1f26e37c02f96c3331282a9438d6ef9bfa4cae6cf629d566093079a660fb",
    "data/worlds/world_v0_7/experiment_007_summary.json": "b9c69a0adfbfb518337c993325009fbbb5f070967c4010a1e27ae9fbf3556cc7",
    "data/worlds/world_v0_7/world_v0_7_config.json": "7fd2d130a6709013ff66d07cdbaa6e8b402afa4d947c63be6fc3f7085b334e3d",
    "data/worlds/world_v0_8/run_manifest.json": "b98b35b7d3ee8e9b8e26bffd90b1f47e6a69d8a58e06c89588637ee8124f0aed",
    "data/worlds/world_v0_8/experiment_008_summary.json": "f72f84a62aa01fa78a0787f6a3cc28e16cf807e897646bb60e9dee7223c66959",
    "data/worlds/world_v0_8/world_v0_8_config.json": "23797ae305364778358e42cc660a7d3a09a4d5e7203c863b74f65e13395b4faa",
    "data/worlds/utility_atlas_v1/atlas_manifest.json": "90ff0feb2f846cf521d4a8d9e1b24e0e8f799e4a409a51c84daea374f744f9f1",
    "data/personas/persona_set_v1/manifest.json": "27b121d100c95a40adc4f1cbf282dc3e17f72a8b5f7d5c5b7926866e1cee2c15",
}


@pytest.fixture(scope="module")
def atlas():
    return load_frozen_atlas()


@pytest.fixture(scope="module")
def locations():
    return load_initial_locations()


@pytest.fixture(scope="module")
def plans():
    return load_intervention_plans()


def test_inventory_and_frozen_inputs() -> None:
    assert TIMESTEPS == 500
    assert CHOICE_SEEDS == tuple(range(50))
    assert len(BRANCHES) == 5
    assert len(REGIMES) * 50 * 5 == 750
    assert ALTERNATION_UPPER_QUARTILE == 0.694375
    assert CONSECUTIVE_TV_UPPER_QUARTILE == 0.36148125
    assert (EXTREME_LATE_HHI, EXTREME_DOMINANT_SHARE, EXTREME_SATURATION) == (0.40, 0.60, 0.10)
    inventory = build_decision_state_inventory()
    partitions = {field["partition"] for field in inventory["fields"]}
    assert partitions <= set(inventory["partitions"])
    assert all(field["partition"] in inventory["partitions"] for field in inventory["fields"])
    names = {field["field_name"] for field in inventory["fields"]}
    assert {"current_location", "rolling_commits", "rolling_total", "rolling_own", "decay_total", "decay_own"} <= names
    by_name = {field["field_name"]: field for field in inventory["fields"]}
    assert by_name["current_location"]["partition"] == "non_social_decision_state"
    assert by_name["rolling_commits"]["partition"] == "social_memory_state"
    assert by_name["decay_total"]["partition"] == "social_memory_state"
    assert by_name["reconsideration_draw"]["partition"] == "RNG_state"
    assert by_name["centered_social_score"]["partition"] == "derived_recomputable_state"
    assert "not decision-relevant" in by_name["visit_count"]["decision_relevance"]
    assert inventory["copy_depth"]["forbidden"] == "copying only the current centered score"
    for relative, digest in FROZEN_SHA256.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest
    source = (ROOT / "world" / "experiment_009.py").read_text(encoding="utf-8")
    assert "StructuredLLMClient" not in source
    assert "plan_intervention" not in source
    manifest = json.loads((ROOT / "data" / "worlds" / "utility_atlas_v1" / "atlas_manifest.json").read_text(encoding="utf-8"))
    assert manifest["sha256"] == "664b511d208f7989e8a671d0e2d6e6433366120bb3d5d6a67db38277b4d5a373"


def test_memory_copy_keeps_the_full_internal_state() -> None:
    rolling = RollingMemory(["agent_0001", "agent_0002"], 50)
    rolling.commit({"agent_0001": "site_01", "agent_0002": "site_02"})
    rolling.commit({"agent_0001": "site_03", "agent_0002": "site_01"})
    payload = memory_payload(rolling)
    assert payload["commits"][0]["agent_0001"] == "site_01"
    assert "total" in payload and "own" in payload
    assert "scores" not in payload
    restored = RollingMemory(["agent_0001", "agent_0002"], 50)
    install_memory(restored, payload)
    assert state_hashes({"agent_0001": "site_01", "agent_0002": "site_02"}, restored, 0)["social_memory_sha256"] == state_hashes(
        {"agent_0001": "site_01", "agent_0002": "site_02"}, rolling, 0
    )["social_memory_sha256"]
    decay = DecayMemory(["agent_0001"], 0.99)
    decay.commit({"agent_0001": "site_04"})
    decay_payload = memory_payload(decay)
    assert set(decay_payload) == {"architecture", "decay", "total", "own"}
    copied = DecayMemory(["agent_0001"], 0.99)
    install_memory(copied, decay_payload)
    assert copied.total == decay.total
    assert copied.own == decay.own


def test_carrier_rules_and_paired_bootstrap() -> None:
    assert classify_persistence(0.02) == "attenuated"
    assert classify_persistence(0.05) == "persistent residual"
    assert classify_persistence(0.10) == "amplified divergence"
    assert classify_carrier("persistent residual", "attenuated", "attenuated") == "A"
    assert classify_carrier("persistent residual", "attenuated", "persistent residual") == "B"
    assert classify_carrier("persistent residual", "amplified divergence", "attenuated") == "C"
    assert classify_carrier("persistent residual", "persistent residual", "persistent residual") == "D"
    assert retention_fraction(0.02, 0.0) is None
    assert retention_fraction(0.04, 0.08) == pytest.approx(0.5)
    carriers = {"RW50": "A", "EXP99": "A"}
    assert overall_conclusion(carriers, gates_ok=True).startswith("persistent collective residuals require coupled")
    assert overall_conclusion({"RW50": "B", "EXP99": "B"}, gates_ok=True).startswith("perturbed social-memory state")
    assert overall_conclusion({"RW50": "C", "EXP99": "C"}, gates_ok=True).startswith("perturbed non-social")
    assert overall_conclusion({"RW50": "D", "EXP99": "D"}, gates_ok=True).startswith("persistent collective divergence can be carried")
    assert "social-memory architecture" in overall_conclusion({"RW50": "A", "EXP99": "B"}, gates_ok=True)
    assert overall_conclusion(carriers, gates_ok=False).startswith("Experiment 009 mechanism classification is invalid")
    low, high = _bootstrap_ci([0.1, 0.2, 0.3, 0.4])
    assert low <= 0.25 <= high


def test_surgery_partitions_and_exact_gates(atlas, locations, plans) -> None:
    for spec in REGIMES:
        runs = _five(atlas, locations, spec, plans, seed=0, timesteps=40)
        prefix = runs["CONTROL"]["series"][:21]
        for branch, result in runs.items():
            assert result["series"][:21] == prefix
            assert result["surgery"]["checkpoint"] == "POST_T21_PRE_T22"
            assert result["surgery"]["surgery_rng_draws"] == 0
            assert result["surgery"]["selected_agents"] == plans[(spec["regime_id"], 0)]["selected_agents"]
            assert result["surgery"]["target_place"] == plans[(spec["regime_id"], 0)]["target_place"]
        assert runs["CONTROL"]["override_steps"] == []
        for branch in ("PERTURBED", "NON_SOCIAL_ONLY", "SOCIAL_MEMORY_ONLY", "FULL_RESET"):
            assert runs[branch]["override_steps"] == [21]
        control, perturbed = runs["CONTROL"]["checkpoint"], runs["PERTURBED"]["checkpoint"]
        non_social, memory_only, reset = (runs[name]["checkpoint"] for name in ("NON_SOCIAL_ONLY", "SOCIAL_MEMORY_ONLY", "FULL_RESET"))
        assert non_social["non_social_sha256"] == perturbed["non_social_sha256"]
        assert non_social["social_memory_sha256"] == control["social_memory_sha256"]
        assert memory_only["non_social_sha256"] == control["non_social_sha256"]
        assert memory_only["social_memory_sha256"] == perturbed["social_memory_sha256"]
        assert memory_only["social_memory_sha256"] != control["social_memory_sha256"]
        assert reset["decision_state_sha256"] == control["decision_state_sha256"]
        assert runs["SOCIAL_MEMORY_ONLY"]["surgery"]["occupancy_tv_vs_control"] == 0.0
        assert runs["SOCIAL_MEMORY_ONLY"]["surgery"]["agent_disagreement_vs_control"] == 0.0
        assert runs["NON_SOCIAL_ONLY"]["surgery"]["occupancy_tv_vs_control"] == pytest.approx(0.10)
        assert runs["FULL_RESET"]["canonical_dynamics_sha256"] == runs["CONTROL"]["canonical_dynamics_sha256"]
        assert runs["FULL_RESET"]["series"][21:] == runs["CONTROL"]["series"][21:]
        if spec["regime_id"] == "S0":
            assert runs["SOCIAL_MEMORY_ONLY"]["decision_dynamics_sha256"] == runs["CONTROL"]["decision_dynamics_sha256"]
            assert runs["NON_SOCIAL_ONLY"]["decision_dynamics_sha256"] == runs["PERTURBED"]["decision_dynamics_sha256"]
        assert "commits" in runs["CONTROL"]["checkpoint"]["social_memory"] or runs["CONTROL"]["checkpoint"]["social_memory"]["architecture"] == "exponential"
        if spec["memory"] == "rolling":
            assert len(runs["PERTURBED"]["checkpoint"]["social_memory"]["commits"]) == 21


def test_matched_rng_sync_and_iteration_order(atlas, locations, plans) -> None:
    spec = REGIMES[1]
    recorded: list[int] = []
    original = experiment.random.Random

    class Tracking:
        def __init__(self, seed: int) -> None:
            recorded.append(seed)
            self._rng = original(seed)

        def random(self) -> float:
            return self._rng.random()

    sources = {
        "CONTROL": simulate_branch(
            atlas, locations, spec, choice_seed_value=3, branch="CONTROL", plan=plans[("RW50", 3)], timesteps=30
        )["checkpoint"]
    }
    sources["PERTURBED"] = simulate_branch(
        atlas,
        locations,
        spec,
        choice_seed_value=3,
        branch="PERTURBED",
        plan=plans[("RW50", 3)],
        timesteps=30,
        sources=sources,
    )["checkpoint"]
    experiment.random.Random = Tracking
    try:
        draws = []
        for branch in BRANCHES:
            recorded.clear()
            simulate_branch(
                atlas,
                locations,
                spec,
                choice_seed_value=3,
                branch=branch,
                plan=plans[("RW50", 3)],
                timesteps=30,
                sources=None if branch == "CONTROL" else sources,
            )
            draws.append(list(recorded))
    finally:
        experiment.random.Random = original
    assert draws[0] == draws[1] == draws[2] == draws[3] == draws[4]
    expected = 0
    for timestep in range(30):
        for agent_id in sorted(locations):
            expected += 1
            if agent_reconsiders(3, timestep, agent_id, REFERENCE_RECONSIDERATION_RATE):
                expected += 1
    assert len(draws[0]) == expected
    seen = []

    def capture(logged, _current, _memory, actions, _surgery):
        if logged == 1:
            seen.append(actions)

    _one(atlas, locations, spec, plans, "CONTROL", timesteps=30, on_step=capture)
    assert len({tuple(sorted(action["occupancy_before"].items())) for action in seen[0]}) == 1
    forward = _one(atlas, locations, spec, plans, "PERTURBED", timesteps=30)
    backward = simulate_branch(
        atlas,
        {agent_id: locations[agent_id] for agent_id in reversed(list(locations))},
        spec,
        choice_seed_value=3,
        branch="PERTURBED",
        plan=plans[("RW50", 3)],
        timesteps=30,
    )
    assert trajectory_hash(forward["series"]) == trajectory_hash(backward["series"])


@pytest.mark.slow
def test_control_and_perturbed_match_experiment_007(atlas, locations, plans) -> None:
    references = json.loads((ROOT / "data" / "worlds" / "world_v0_7" / "run_manifest.json").read_text(encoding="utf-8"))
    hashes = references["trajectory_sha256"]
    assert hashes["S0:0:CONTROL"] == "610cb9dfff650d135bf2520cac4247e7b2707073fae6e0fa52f094a040175955"
    for spec in REGIMES:
        for seed in CHOICE_SEEDS:
            for branch in ("CONTROL", "PERTURBED"):
                result = simulate_branch(
                    atlas,
                    locations,
                    spec,
                    choice_seed_value=seed,
                    branch=branch,
                    plan=plans[(spec["regime_id"], seed)],
                )
                assert result["trajectory_sha256"] == hashes[f"{spec['regime_id']}:{seed}:{branch}"]
                assert result["llm_calls"] == 0


def test_recorded_run_replays_selected_worlds(atlas, locations, plans) -> None:
    output = ROOT / "data" / "worlds" / "world_v0_9"
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "experiment_009_summary.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is True
    assert manifest["world_count"] == 750
    assert manifest["llm_calls"] == 0
    assert summary["gates"]["full_reset_exact"] is True
    assert summary["gates"]["pairs_checked"] == 150
    assert summary["conclusion"] == overall_conclusion(summary["carrier_type"], gates_ok=True)
    requested = (("S0", "CONTROL", 0), ("RW50", "PERTURBED", 0), ("EXP99", "NON_SOCIAL_ONLY", 0))
    for regime_id, branch, seed in requested:
        spec = next(item for item in REGIMES if item["regime_id"] == regime_id)
        sources = None
        if branch == "NON_SOCIAL_ONLY":
            sources = {
                name: simulate_branch(
                    atlas, locations, spec, choice_seed_value=seed, branch=name, plan=plans[(regime_id, seed)]
                )["checkpoint"]
                for name in ("CONTROL", "PERTURBED")
            }
        result = simulate_branch(
            atlas, locations, spec, choice_seed_value=seed, branch=branch, plan=plans[(regime_id, seed)], sources=sources
        )
        assert result["trajectory_sha256"] == manifest["trajectory_sha256"][f"{regime_id}:{seed}:{branch}"]


def _five(atlas, locations, spec, plans, seed: int, timesteps: int) -> dict[str, dict]:
    runs = {}
    sources = {}
    for branch in BRANCHES:
        runs[branch] = simulate_branch(
            atlas,
            locations,
            spec,
            choice_seed_value=seed,
            branch=branch,
            plan=plans[(spec["regime_id"], seed)],
            timesteps=timesteps,
            sources=None if branch == "CONTROL" else sources,
        )
        if branch in {"CONTROL", "PERTURBED"}:
            sources[branch] = runs[branch]["checkpoint"]
    return runs


def _one(atlas, locations, spec, plans, branch: str, timesteps: int, on_step=None):
    sources = None
    if branch != "CONTROL":
        sources = {
            "CONTROL": simulate_branch(
                atlas,
                locations,
                spec,
                choice_seed_value=3,
                branch="CONTROL",
                plan=plans[(spec["regime_id"], 3)],
                timesteps=timesteps,
            )["checkpoint"]
        }
        if branch != "PERTURBED":
            sources["PERTURBED"] = simulate_branch(
                atlas,
                locations,
                spec,
                choice_seed_value=3,
                branch="PERTURBED",
                plan=plans[(spec["regime_id"], 3)],
                timesteps=timesteps,
                sources=sources,
            )["checkpoint"]
    return simulate_branch(
        atlas,
        locations,
        spec,
        choice_seed_value=3,
        branch=branch,
        plan=plans[(spec["regime_id"], 3)],
        timesteps=timesteps,
        sources=sources,
        on_step=on_step,
    )


def _recorded(atlas, locations, spec, plans, recorded, branch: str) -> list[int]:
    recorded.clear()
    _one(atlas, locations, spec, plans, branch, timesteps=30)
    return list(recorded)
