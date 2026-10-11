"""World v0.11 symmetric social-environment tests. No model calls."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

import pytest

import world.experiment_011 as experiment
from world.choice import agent_reconsiders
from world.dynamics_calibration import load_initial_locations
from world.experiment_002 import REFERENCE_RECONSIDERATION_RATE
from world.experiment_004 import trajectory_hash
from world.experiment_005 import RollingMemory, centered_scores, clip_score
from world.experiment_009 import load_intervention_plans
from world.experiment_011 import (
    ALTERNATION_UPPER_QUARTILE,
    ASYMMETRIC_REINTERPRETATION,
    BOOTSTRAP_DRAWS,
    BRANCHES,
    CHOICE_SEEDS,
    CONSECUTIVE_TV_UPPER_QUARTILE,
    ENVIRONMENTS,
    EXTREME_DOMINANT_SHARE,
    EXTREME_LATE_HHI,
    EXTREME_SATURATION,
    REGIMES,
    RELATIONSHIP,
    TIMESTEPS,
    _bootstrap_ci,
    classify_off,
    classify_sync,
    overall_conclusion,
    reconvergence_time,
    simulate_branch,
    stable_attenuation_time,
)
from world.geography import PLACE_IDS
from world.utility_atlas import load_frozen_atlas

ROOT = Path(__file__).resolve().parents[1]
FROZEN_SHA256 = {
    "data/worlds/world_v0_7/run_manifest.json": "dc1e1f26e37c02f96c3331282a9438d6ef9bfa4cae6cf629d566093079a660fb",
    "data/worlds/world_v0_8/run_manifest.json": "b98b35b7d3ee8e9b8e26bffd90b1f47e6a69d8a58e06c89588637ee8124f0aed",
    "data/worlds/world_v0_8/experiment_008_summary.json": "f72f84a62aa01fa78a0787f6a3cc28e16cf807e897646bb60e9dee7223c66959",
    "data/worlds/world_v0_9/run_manifest.json": "84840c8f74b7297c251007a05073dda07d053f6c964de39deb8de770806a03d4",
    "data/worlds/world_v0_9/experiment_009_summary.json": "a1dba69db00c143c5c54f482320117f1612a2cabede413d6757d4243a65a7cd2",
    "data/worlds/world_v0_9/world_v0_9_config.json": "8a5e2ea01683cb72b193ec3af5fa857c2f637833d6b4eb800dd56f2de814cfaf",
    "data/worlds/world_v0_10/run_manifest.json": "81cdc4894894a19843e45263164da42d957b855778c4b05b732a83be5f089318",
    "data/worlds/world_v0_10/experiment_010_summary.json": "c4ef717ad6baef3bdee68fe6d8c02cc876f187a63265533ee5bba9d506b2f59d",
    "data/worlds/world_v0_10/world_v0_10_config.json": "2bf5ca0bd62dbeaf91027500f6808783e026788a5a62d1a16ea99d988e99966c",
    "data/worlds/utility_atlas_v1/atlas_manifest.json": "90ff0feb2f846cf521d4a8d9e1b24e0e8f799e4a409a51c84daea374f744f9f1",
    "data/personas/persona_set_v1/manifest.json": "27b121d100c95a40adc4f1cbf282dc3e17f72a8b5f7d5c5b7926866e1cee2c15",
}


def test_frozen_inputs_and_classification_rules() -> None:
    assert TIMESTEPS == 500
    assert CHOICE_SEEDS == tuple(range(50))
    assert len(BRANCHES) == 8
    assert len(ENVIRONMENTS) == 4
    assert len(REGIMES) * 50 * 8 == 1200
    assert BOOTSTRAP_DRAWS == 10000
    assert (ALTERNATION_UPPER_QUARTILE, CONSECUTIVE_TV_UPPER_QUARTILE) == (0.694375, 0.36148125)
    assert (EXTREME_LATE_HHI, EXTREME_DOMINANT_SHARE, EXTREME_SATURATION) == (0.40, 0.60, 0.10)
    assert classify_off(0.024999) == "OFF-A"
    assert classify_off(0.025) == "OFF-B"
    assert classify_off(0.099999) == "OFF-B"
    assert classify_off(0.10) == "OFF-C"
    assert classify_sync(0.0, 0.0) == "S1"
    assert classify_sync(0.0, 0.025) == "S2"
    assert classify_sync(0.025, 0.0) == "S3"
    assert classify_sync(0.025, 0.025) == "S4"
    both = overall_conclusion({"RW50": "OFF-A", "EXP99": "OFF-A"}, {"RW50": "S1", "EXP99": "S1"})
    assert "removed symmetrically" in both
    assert "independently of donor history" in both
    assert ASYMMETRIC_REINTERPRETATION in both
    assert RELATIONSHIP in both
    assert "donor-field content" in overall_conclusion({"RW50": "OFF-B", "EXP99": "OFF-B"}, {"RW50": "S2", "EXP99": "S2"})
    assert "insufficient" in overall_conclusion({"RW50": "OFF-C", "EXP99": "OFF-C"}, {"RW50": "S4", "EXP99": "S4"})
    assert "RW50:" in overall_conclusion({"RW50": "OFF-A", "EXP99": "OFF-C"}, {"RW50": "S1", "EXP99": "S4"})
    assert stable_attenuation_time([0.1] * 22 + [0.01] * 20) == 41
    assert reconvergence_time([0.1] * 21 + [0.0] * 20) == 40
    assert reconvergence_time([0.0] * 21 + [0.1] * 20) is None
    low, high = _bootstrap_ci([0.1, 0.2, 0.3, 0.4])
    assert low <= 0.25 <= high
    for relative, digest in FROZEN_SHA256.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest
    source = (ROOT / "world" / "experiment_011.py").read_text(encoding="utf-8")
    assert "StructuredLLMClient" not in source
    assert "plan_intervention" not in source
    assert "closure_type_exp010" in source
    assert "off_class_exp011" in source and "synchronization_type_exp011" in source
    raw = {place_id: float(index) for index, place_id in enumerate(PLACE_IDS)}
    unclipped, clipped = centered_scores(raw)
    assert abs(statistics.fmean(unclipped.values())) < 1e-12
    scale = statistics.pstdev(list(unclipped.values()))
    assert abs(scale - 1.0) < 1e-12
    assert all(-2.0 <= clipped[place_id] <= 2.0 for place_id in PLACE_IDS)
    assert clip_score(3.5) == 2.0 and clip_score(-3.5) == -2.0
    memory = RollingMemory(["agent_a", "agent_b"], 50)
    memory.commit({"agent_a": "site_01", "agent_b": "site_02"})
    assert memory.loo_raw("agent_a")["site_01"] == 0.0
    assert memory.loo_raw("agent_b")["site_01"] == 1.0


def test_symmetric_environment_gates_on_short_runs(atlas, locations, plans) -> None:
    for spec in REGIMES:
        runs = _eight(atlas, locations, spec, plans, seed=0, timesteps=40)
        prefix = runs["CONTROL_ENDOGENOUS"]["series"][:21]
        for branch, result in runs.items():
            assert result["series"][:21] == prefix
            assert result["llm_calls"] == 0
            assert result["rng_draw_counts"] == runs["CONTROL_ENDOGENOUS"]["rng_draw_counts"]
        for branch, result in runs.items():
            if branch.startswith("CONTROL_"):
                assert result["override_count"] == 0
                assert result["override_steps"] == []
                assert result["series"][:22] == runs["CONTROL_ENDOGENOUS"]["series"][:22]
            else:
                assert result["override_count"] == 2
                assert result["override_steps"] == [21]
                assert result["series"][:22] == runs["PERTURBED_ENDOGENOUS"]["series"][:22]
        control_field = runs["CONTROL_ENDOGENOUS"]["social_field"]
        perturbed_field = runs["PERTURBED_ENDOGENOUS"]["social_field"]
        assert set(control_field) == set(range(22, 41))
        assert set(perturbed_field) == set(range(22, 41))
        assert set(control_field[22]) == set(locations)
        for scores in control_field[30].values():
            assert set(scores) == set(PLACE_IDS)
            assert all(-2.0 <= value <= 2.0 for value in scores.values())
        assert runs["CONTROL_COMMON_CONTROL_FIELD"]["social_field"] == {}
        assert runs["PERTURBED_COMMON_PERTURBED_FIELD"]["social_field"] == {}
        assert runs["CONTROL_COMMON_CONTROL_FIELD"]["field_hashes"] == runs["PERTURBED_COMMON_CONTROL_FIELD"]["field_hashes"]
        assert runs["CONTROL_COMMON_CONTROL_FIELD"]["field_hashes"] == runs["CONTROL_ENDOGENOUS"]["field_hashes"]
        assert runs["CONTROL_COMMON_PERTURBED_FIELD"]["field_hashes"] == runs["PERTURBED_COMMON_PERTURBED_FIELD"]["field_hashes"]
        assert runs["CONTROL_COMMON_PERTURBED_FIELD"]["field_hashes"] == runs["PERTURBED_ENDOGENOUS"]["field_hashes"]
        assert runs["PERTURBED_COMMON_CONTROL_FIELD"]["shadow_scores_differed"] is True
        assert runs["CONTROL_SOCIAL_OFF"]["social_utility_absmax"] == 0.0
        assert runs["PERTURBED_SOCIAL_OFF"]["social_utility_absmax"] == 0.0
        if spec["regime_id"] == "S0":
            control_hash = runs["CONTROL_ENDOGENOUS"]["canonical_dynamics_sha256"]
            perturbed_hash = runs["PERTURBED_ENDOGENOUS"]["canonical_dynamics_sha256"]
            for branch, result in runs.items():
                expected = control_hash if branch.startswith("CONTROL_") else perturbed_hash
                assert result["canonical_dynamics_sha256"] == expected


def test_social_off_keeps_rng_and_iteration_order(atlas, locations, plans) -> None:
    spec = REGIMES[1]
    plan = plans[("RW50", 4)]
    donors = _donors(atlas, locations, spec, plan, seed=4, timesteps=30)
    seen = []

    def capture(logged, _current, _memory, actions, _field):
        if logged >= 22:
            seen.extend(actions)

    for branch in ("CONTROL_SOCIAL_OFF", "PERTURBED_SOCIAL_OFF"):
        seen.clear()
        simulate_branch(
            atlas,
            locations,
            spec,
            choice_seed_value=4,
            branch=branch,
            plan=plan,
            timesteps=30,
            on_step=capture,
        )
        assert seen
        assert all(action["effective_social_weight"] == 0.0 for action in seen)
        assert all(value == 0.0 for action in seen for value in action["social_utility"].values())
        assert all(action["total_utility"] == action["base_utility"] for action in seen)
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
        draws = []
        for branch in BRANCHES:
            recorded.clear()
            donor = None
            if branch.endswith("COMMON_CONTROL_FIELD"):
                donor = donors["CONTROL_ENDOGENOUS"]
            elif branch.endswith("COMMON_PERTURBED_FIELD"):
                donor = donors["PERTURBED_ENDOGENOUS"]
            simulate_branch(
                atlas,
                locations,
                spec,
                choice_seed_value=4,
                branch=branch,
                plan=plan,
                timesteps=30,
                donor_field=donor,
            )
            draws.append(list(recorded))
    finally:
        experiment.random.Random = original
    assert draws[0] == draws[1] == draws[2] == draws[3] == draws[4] == draws[5] == draws[6] == draws[7]
    expected = 0
    for timestep in range(30):
        for agent_id in sorted(locations):
            expected += 1
            if agent_reconsiders(4, timestep, agent_id, REFERENCE_RECONSIDERATION_RATE):
                expected += 1
    assert len(draws[0]) == expected
    forward = simulate_branch(
        atlas, locations, spec, choice_seed_value=4, branch="PERTURBED_ENDOGENOUS", plan=plan, timesteps=30
    )
    backward = simulate_branch(
        atlas,
        {agent_id: locations[agent_id] for agent_id in reversed(list(locations))},
        spec,
        choice_seed_value=4,
        branch="PERTURBED_ENDOGENOUS",
        plan=plan,
        timesteps=30,
    )
    assert trajectory_hash(forward["series"]) == trajectory_hash(backward["series"])


@pytest.mark.slow
def test_endogenous_branches_match_experiment_007(atlas, locations, plans) -> None:
    references = json.loads((ROOT / "data" / "worlds" / "world_v0_7" / "run_manifest.json").read_text(encoding="utf-8"))
    hashes = references["trajectory_sha256"]
    for spec in REGIMES:
        for seed in CHOICE_SEEDS:
            plan = plans[(spec["regime_id"], seed)]
            for branch, label in (("CONTROL_ENDOGENOUS", "CONTROL"), ("PERTURBED_ENDOGENOUS", "PERTURBED")):
                result = simulate_branch(atlas, locations, spec, choice_seed_value=seed, branch=branch, plan=plan)
                assert result["trajectory_sha256"] == hashes[f"{spec['regime_id']}:{seed}:{label}"]


def test_recorded_run_replays_selected_worlds(atlas, locations, plans) -> None:
    output = ROOT / "data" / "worlds" / "world_v0_11"
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "experiment_011_summary.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is True
    assert manifest["world_count"] == 1200
    assert manifest["llm_calls"] == 0
    assert summary["gates"]["s0_equivalence"] is True
    assert summary["gates"]["common_control_field"] is True
    assert summary["gates"]["common_perturbed_field"] is True
    assert summary["gates"]["social_off_zero"] is True
    assert summary["closure_type_exp010"] == {"RW50": "C", "EXP99": "C"}
    assert summary["conclusion"] == overall_conclusion(summary["off_class_exp011"], summary["synchronization_type_exp011"])
    requested = (
        ("S0", "CONTROL_ENDOGENOUS", None),
        ("RW50", "PERTURBED_ENDOGENOUS", None),
        ("EXP99", "PERTURBED_COMMON_CONTROL_FIELD", "CONTROL_ENDOGENOUS"),
    )
    for regime_id, branch, donor_branch in requested:
        spec = next(item for item in REGIMES if item["regime_id"] == regime_id)
        plan = plans[(regime_id, 0)]
        donor = None
        if donor_branch is not None:
            donor = simulate_branch(atlas, locations, spec, choice_seed_value=0, branch=donor_branch, plan=plan)["social_field"]
        result = simulate_branch(atlas, locations, spec, choice_seed_value=0, branch=branch, plan=plan, donor_field=donor)
        assert result["trajectory_sha256"] == manifest["trajectory_sha256"][f"{regime_id}:0:{branch}"]


def _donors(atlas, locations, spec, plan, seed: int, timesteps: int) -> dict[str, dict]:
    donors = {}
    for branch in ("CONTROL_ENDOGENOUS", "PERTURBED_ENDOGENOUS"):
        donors[branch] = simulate_branch(
            atlas, locations, spec, choice_seed_value=seed, branch=branch, plan=plan, timesteps=timesteps
        )["social_field"]
    return donors


def _eight(atlas, locations, spec, plans, seed: int, timesteps: int) -> dict[str, dict]:
    plan = plans[(spec["regime_id"], seed)]
    runs = {}
    for branch in ("CONTROL_ENDOGENOUS", "PERTURBED_ENDOGENOUS"):
        runs[branch] = simulate_branch(
            atlas, locations, spec, choice_seed_value=seed, branch=branch, plan=plan, timesteps=timesteps
        )
    donor = {
        "common_control": runs["CONTROL_ENDOGENOUS"]["social_field"],
        "common_perturbed": runs["PERTURBED_ENDOGENOUS"]["social_field"],
        "social_off": None,
    }
    for branch in BRANCHES:
        environment = experiment.environment_of(branch)
        if environment == "endogenous":
            continue
        runs[branch] = simulate_branch(
            atlas,
            locations,
            spec,
            choice_seed_value=seed,
            branch=branch,
            plan=plan,
            timesteps=timesteps,
            donor_field=donor[environment],
        )
    return runs


@pytest.fixture(scope="module")
def atlas():
    return load_frozen_atlas()


@pytest.fixture(scope="module")
def locations():
    return load_initial_locations()


@pytest.fixture(scope="module")
def plans():
    return load_intervention_plans()
