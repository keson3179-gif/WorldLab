"""World v0.10 feedback-closure tests. No model calls."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

import pytest

import world.experiment_010 as experiment
from world.choice import agent_reconsiders
from world.dynamics_calibration import load_initial_locations
from world.experiment_002 import REFERENCE_RECONSIDERATION_RATE
from world.experiment_004 import trajectory_hash
from world.experiment_005 import RollingMemory, centered_scores, clip_score
from world.experiment_009 import load_intervention_plans
from world.experiment_010 import (
    ALTERNATION_UPPER_QUARTILE,
    BRANCHES,
    CHOICE_SEEDS,
    CONSECUTIVE_TV_UPPER_QUARTILE,
    EXTREME_DOMINANT_SHARE,
    EXTREME_LATE_HHI,
    EXTREME_SATURATION,
    PERTURBED_BRANCHES,
    REGIMES,
    TIMESTEPS,
    _bootstrap_ci,
    classify_closure,
    overall_conclusion,
    reconvergence_time,
    retention_fraction,
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


def test_frozen_inputs_and_closure_rules() -> None:
    assert TIMESTEPS == 500
    assert CHOICE_SEEDS == tuple(range(50))
    assert len(BRANCHES) == 4
    assert len(REGIMES) * 50 * 4 == 600
    assert (ALTERNATION_UPPER_QUARTILE, CONSECUTIVE_TV_UPPER_QUARTILE) == (0.694375, 0.36148125)
    assert (EXTREME_LATE_HHI, EXTREME_DOMINANT_SHARE, EXTREME_SATURATION) == (0.40, 0.60, 0.10)
    assert classify_closure("persistent residual", 0.01, 0.01) == "A"
    assert classify_closure("persistent residual", 0.05, 0.01) == "B"
    assert classify_closure("amplified divergence", 0.01, 0.08) == "C"
    assert classify_closure("persistent residual", 0.04, 0.09) == "D"
    assert classify_closure("attenuated", 0.0, 0.0) is None
    assert retention_fraction(0.02, 0.0) is None
    assert overall_conclusion({"RW50": "A", "EXP99": "A"}, gates_ok=True).startswith("persistent collective residuals require")
    assert overall_conclusion({"RW50": "B", "EXP99": "B"}, gates_ok=True).startswith("ongoing social influence sustains")
    assert overall_conclusion({"RW50": "D", "EXP99": "D"}, gates_ok=True).startswith("persistent collective divergence survives")
    assert "social-memory architecture" in overall_conclusion({"RW50": "A", "EXP99": "D"}, gates_ok=True)
    assert overall_conclusion({"RW50": "A", "EXP99": "A"}, gates_ok=False).startswith("Experiment 010 closure classification is invalid")
    assert overall_conclusion({"RW50": None, "EXP99": "A"}, gates_ok=True).startswith("Experiment 010 mechanism classification invalid")
    values = [0.1] * 22 + [0.01] * 20
    assert stable_attenuation_time(values) == 41
    zeros = [0.1] * 21 + [0.0] * 20
    assert reconvergence_time(zeros) == 40
    low, high = _bootstrap_ci([0.1, 0.2, 0.3, 0.4])
    assert low <= 0.25 <= high
    for relative, digest in FROZEN_SHA256.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest
    source = (ROOT / "world" / "experiment_010.py").read_text(encoding="utf-8")
    assert "StructuredLLMClient" not in source
    assert "plan_intervention" not in source
    assert "carrier_type_exp009" in source and "closure_type_exp010" in source
    raw = {place_id: float(index) for index, place_id in enumerate(PLACE_IDS)}
    unclipped, clipped = centered_scores(raw)
    assert abs(statistics.fmean(unclipped.values())) < 1e-12
    assert all(-2.0 <= clipped[place_id] <= 2.0 for place_id in PLACE_IDS)
    assert clip_score(3.5) == 2.0 and clip_score(-3.5) == -2.0
    memory = RollingMemory(["agent_a", "agent_b"], 50)
    memory.commit({"agent_a": "site_01", "agent_b": "site_02"})
    assert memory.loo_raw("agent_a")["site_01"] == 0.0
    assert memory.loo_raw("agent_b")["site_01"] == 1.0


def test_ablation_gates_on_short_runs(atlas, locations, plans) -> None:
    for spec in REGIMES:
        runs = _four(atlas, locations, spec, plans, seed=0, timesteps=40)
        prefix = runs["CONTROL_CLOSED"]["series"][:21]
        perturbed = runs["PERTURBED_CLOSED"]["series"][:22]
        for branch, result in runs.items():
            assert result["series"][:21] == prefix
            assert result["llm_calls"] == 0
        for branch in PERTURBED_BRANCHES:
            assert runs[branch]["series"][:22] == perturbed
            assert runs[branch]["override_steps"] == [21]
        plan = plans[(spec["regime_id"], 0)]
        assert runs["CONTROL_CLOSED"]["override_steps"] == []
        field = runs["CONTROL_CLOSED"]["social_field"]
        assert set(field) == set(range(22, 41))
        assert set(field[22]) == set(locations)
        for agent_id, scores in field[30].items():
            assert set(scores) == set(PLACE_IDS)
            assert all(-2.0 <= value <= 2.0 for value in scores.values())
        for row in runs["PERTURBED_CONTROL_FIELD"]["audits"]:
            assert row["field_match"] is True
            assert row["timestep"] >= 22
        assert max(runs["PERTURBED_CONTROL_FIELD"]["decision_field_gap"][22:]) == 0.0
        assert runs["PERTURBED_CONTROL_FIELD"]["memory_trace"][21] != runs["CONTROL_CLOSED"]["memory_trace"][21]
        if spec["regime_id"] == "S0":
            closed = runs["PERTURBED_CLOSED"]["canonical_dynamics_sha256"]
            assert runs["PERTURBED_CONTROL_FIELD"]["canonical_dynamics_sha256"] == closed
            assert runs["PERTURBED_SOCIAL_OFF"]["canonical_dynamics_sha256"] == closed
        assert plan["selected_agents"]


def test_social_off_is_zero_and_rng_stays_matched(atlas, locations, plans) -> None:
    spec = REGIMES[1]
    seen = []

    def capture(logged, _current, _memory, actions, _field):
        if logged >= 22:
            seen.extend(actions)

    field = simulate_branch(
        atlas, locations, spec, choice_seed_value=4, branch="CONTROL_CLOSED", plan=plans[("RW50", 4)], timesteps=30
    )["social_field"]
    off = simulate_branch(
        atlas,
        locations,
        spec,
        choice_seed_value=4,
        branch="PERTURBED_SOCIAL_OFF",
        plan=plans[("RW50", 4)],
        timesteps=30,
        social_field=field,
        on_step=capture,
    )
    assert seen
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
            simulate_branch(
                atlas,
                locations,
                spec,
                choice_seed_value=4,
                branch=branch,
                plan=plans[("RW50", 4)],
                timesteps=30,
                social_field=None if branch == "CONTROL_CLOSED" else field,
            )
            draws.append(list(recorded))
    finally:
        experiment.random.Random = original
    assert draws[0] == draws[1] == draws[2] == draws[3]
    expected = 0
    for timestep in range(30):
        for agent_id in sorted(locations):
            expected += 1
            if agent_reconsiders(4, timestep, agent_id, REFERENCE_RECONSIDERATION_RATE):
                expected += 1
    assert len(draws[0]) == expected
    assert off["override_steps"] == [21]
    forward = simulate_branch(
        atlas, locations, spec, choice_seed_value=4, branch="PERTURBED_CLOSED", plan=plans[("RW50", 4)], timesteps=30
    )
    backward = simulate_branch(
        atlas,
        {agent_id: locations[agent_id] for agent_id in reversed(list(locations))},
        spec,
        choice_seed_value=4,
        branch="PERTURBED_CLOSED",
        plan=plans[("RW50", 4)],
        timesteps=30,
    )
    assert trajectory_hash(forward["series"]) == trajectory_hash(backward["series"])


@pytest.mark.slow
def test_closed_branches_match_experiment_007(atlas, locations, plans) -> None:
    references = json.loads((ROOT / "data" / "worlds" / "world_v0_7" / "run_manifest.json").read_text(encoding="utf-8"))
    hashes = references["trajectory_sha256"]
    assert hashes["S0:0:CONTROL"] == "610cb9dfff650d135bf2520cac4247e7b2707073fae6e0fa52f094a040175955"
    for spec in REGIMES:
        for seed in CHOICE_SEEDS:
            for branch, label in (("CONTROL_CLOSED", "CONTROL"), ("PERTURBED_CLOSED", "PERTURBED")):
                result = simulate_branch(
                    atlas,
                    locations,
                    spec,
                    choice_seed_value=seed,
                    branch=branch,
                    plan=plans[(spec["regime_id"], seed)],
                )
                assert result["trajectory_sha256"] == hashes[f"{spec['regime_id']}:{seed}:{label}"]


def test_recorded_run_replays_selected_worlds(atlas, locations, plans) -> None:
    output = ROOT / "data" / "worlds" / "world_v0_10"
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "experiment_010_summary.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is True
    assert manifest["world_count"] == 600
    assert manifest["llm_calls"] == 0
    assert summary["gates"]["control_field_replay"] is True
    assert summary["gates"]["s0_equivalence"] is True
    assert summary["carrier_type_exp009"] == {"RW50": "D", "EXP99": "D"}
    assert set(summary["closure_type_exp010"]) == {"RW50", "EXP99"}
    assert summary["conclusion"] == overall_conclusion(summary["closure_type_exp010"], gates_ok=True)
    requested = (("S0", "CONTROL_CLOSED", 0), ("RW50", "PERTURBED_CLOSED", 0), ("EXP99", "PERTURBED_CONTROL_FIELD", 0))
    for regime_id, branch, seed in requested:
        spec = next(item for item in REGIMES if item["regime_id"] == regime_id)
        plan = plans[(regime_id, seed)]
        field = None
        if branch == "PERTURBED_CONTROL_FIELD":
            field = simulate_branch(
                atlas, locations, spec, choice_seed_value=seed, branch="CONTROL_CLOSED", plan=plan
            )["social_field"]
        result = simulate_branch(
            atlas, locations, spec, choice_seed_value=seed, branch=branch, plan=plan, social_field=field
        )
        assert result["trajectory_sha256"] == manifest["trajectory_sha256"][f"{regime_id}:{seed}:{branch}"]


def _four(atlas, locations, spec, plans, seed: int, timesteps: int) -> dict[str, dict]:
    plan = plans[(spec["regime_id"], seed)]
    control = simulate_branch(
        atlas, locations, spec, choice_seed_value=seed, branch="CONTROL_CLOSED", plan=plan, timesteps=timesteps
    )
    runs = {"CONTROL_CLOSED": control}
    for branch in PERTURBED_BRANCHES:
        runs[branch] = simulate_branch(
            atlas,
            locations,
            spec,
            choice_seed_value=seed,
            branch=branch,
            plan=plan,
            timesteps=timesteps,
            social_field=control["social_field"],
        )
    return runs
