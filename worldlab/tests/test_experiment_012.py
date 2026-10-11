"""World v0.12 feedback-autonomy tests. No model calls."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

import pytest

import world.experiment_011 as experiment_011
import world.experiment_012 as experiment
from world.choice import agent_reconsiders
from world.dynamics_calibration import load_initial_locations
from world.experiment_002 import REFERENCE_RECONSIDERATION_RATE
from world.experiment_004 import trajectory_hash
from world.experiment_009 import load_intervention_plans
from world.experiment_012 import (
    ALTERNATION_UPPER_QUARTILE,
    BLEND_TOLERANCE,
    BOOTSTRAP_DRAWS,
    BOUNDARY_SENTENCE,
    CHOICE_SEEDS,
    CONSECUTIVE_TV_UPPER_QUARTILE,
    EXTREME_DOMINANT_SHARE,
    EXTREME_LATE_HHI,
    EXTREME_SATURATION,
    HISTORIES,
    LAMBDAS,
    MIDPOINT_NOTE,
    REGIMES,
    TIMESTEPS,
    _bootstrap_ci,
    blend_scores,
    classify_autonomy,
    load_donor_store,
    midpoint_scores,
    overall_conclusion,
    reconvergence_time,
    simulate_history,
    stable_attenuation_time,
    trajectory_shape,
)
from world.geography import PLACE_IDS
from world.utility_atlas import load_frozen_atlas

ROOT = Path(__file__).resolve().parents[1]
FROZEN_SHA256 = {
    "data/worlds/world_v0_7/run_manifest.json": "dc1e1f26e37c02f96c3331282a9438d6ef9bfa4cae6cf629d566093079a660fb",
    "data/worlds/world_v0_8/run_manifest.json": "b98b35b7d3ee8e9b8e26bffd90b1f47e6a69d8a58e06c89588637ee8124f0aed",
    "data/worlds/world_v0_9/run_manifest.json": "84840c8f74b7297c251007a05073dda07d053f6c964de39deb8de770806a03d4",
    "data/worlds/world_v0_10/run_manifest.json": "81cdc4894894a19843e45263164da42d957b855778c4b05b732a83be5f089318",
    "data/worlds/world_v0_10/experiment_010_summary.json": "c4ef717ad6baef3bdee68fe6d8c02cc876f187a63265533ee5bba9d506b2f59d",
    "data/worlds/world_v0_11/run_manifest.json": "cff55b962e9cb7a9edd7ce0bd9d13efbe0d69ebeca02c857bfd2b7537a66cd45",
    "data/worlds/world_v0_11/experiment_011_summary.json": "5765b3a8643c237c5dd0d798edf06e28909f59f97e5442ca374afcb4174df779",
    "data/worlds/world_v0_11/world_v0_11_config.json": "272654267afc8db07e734d6487245e34fa6b7d30a30fc2304f7e676069dd0895",
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


def test_frozen_inputs_and_onset_rules() -> None:
    assert TIMESTEPS == 500
    assert CHOICE_SEEDS == tuple(range(50))
    assert LAMBDAS == (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    assert HISTORIES == ("CONTROL", "PERTURBED")
    assert len(REGIMES) * 50 * 6 * 2 == 1800
    assert BOOTSTRAP_DRAWS == 10000
    assert BLEND_TOLERANCE == 1e-12
    assert (ALTERNATION_UPPER_QUARTILE, CONSECUTIVE_TV_UPPER_QUARTILE) == (0.694375, 0.36148125)
    assert (EXTREME_LATE_HHI, EXTREME_DOMINANT_SHARE, EXTREME_SATURATION) == (0.40, 0.60, 0.10)
    boundary = classify_autonomy([0.0, 0.0, 0.0, 0.0, 0.0, 0.0957])
    assert boundary["autonomy_pattern"] == "FULL_AUTONOMY_BOUNDARY"
    assert boundary["persistence_onset_lambda"] == 1.0
    assert boundary["persistence_onset_band"]["label"] == "(0.80, 1.00]"
    graded = classify_autonomy([0.0, 0.01, 0.03, 0.04, 0.05, 0.06])
    assert graded["autonomy_pattern"] == "GRADED_ONSET"
    assert graded["persistence_onset_lambda"] == 0.4
    assert graded["persistence_onset_band"]["label"] == "(0.20, 0.40]"
    assert classify_autonomy([0.0, 0.03, 0.01, 0.04, 0.05, 0.06])["autonomy_pattern"] == "nonmonotonic"
    assert classify_autonomy([0.03, 0.0, 0.0, 0.0, 0.0, 0.0])["autonomy_pattern"] == "COMMON_MIDPOINT_PERSISTENCE"
    assert overall_conclusion({"RW50": "FULL_AUTONOMY_BOUNDARY", "EXP99": "FULL_AUTONOMY_BOUNDARY"}, {"RW50": None, "EXP99": None}) == BOUNDARY_SENTENCE
    assert "social-memory architecture" in overall_conclusion(
        {"RW50": "GRADED_ONSET", "EXP99": "GRADED_ONSET"}, {"RW50": "(0.40, 0.60]", "EXP99": "(0.80, 1.00]"}
    )
    assert overall_conclusion({"RW50": "nonmonotonic", "EXP99": "GRADED_ONSET"}, {"RW50": None, "EXP99": "(0.40, 0.60]"}).startswith(
        "history persistence depends non-monotonically"
    )
    assert trajectory_shape([0.0, 0.0, 0.0, 0.0, 0.0, 0.09]) == "threshold-like / nonlinear pattern"
    assert stable_attenuation_time([0.1] * 22 + [0.01] * 20) == 41
    assert reconvergence_time([0.1] * 21 + [0.0] * 20) == 40
    low, high = _bootstrap_ci([0.1, 0.2, 0.3, 0.4])
    assert low <= 0.25 <= high
    assert midpoint_scores({"site_01": 2.0, "site_02": -2.0, "site_03": 1.0, "site_04": 1.0, "site_05": 0.0}, {"site_01": 2.0, "site_02": 2.0, "site_03": -1.0, "site_04": 1.0, "site_05": 0.0})[
        "site_01"
    ] == 2.0
    blended = blend_scores({"site_01": 1.0, "site_02": 0.0, "site_03": 0.0, "site_04": 0.0, "site_05": 0.0}, {"site_01": 0.0, "site_02": 0.0, "site_03": 0.0, "site_04": 0.0, "site_05": 0.0}, 0.4)
    assert blended["site_01"] == pytest.approx(0.6)
    assert blend_scores({"site_01": 1.5, "site_02": 0.0, "site_03": 0.0, "site_04": 0.0, "site_05": 0.0}, {"site_01": -1.5, "site_02": 0.0, "site_03": 0.0, "site_04": 0.0, "site_05": 0.0}, 0.0)[
        "site_01"
    ] == 1.5
    assert blend_scores({"site_01": 1.5, "site_02": 0.0, "site_03": 0.0, "site_04": 0.0, "site_05": 0.0}, {"site_01": -0.25, "site_02": 0.0, "site_03": 0.0, "site_04": 0.0, "site_05": 0.0}, 1.0)[
        "site_01"
    ] == -0.25
    for relative, digest in FROZEN_SHA256.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest
    source = (ROOT / "world" / "experiment_012.py").read_text(encoding="utf-8")
    assert "StructuredLLMClient" not in source
    assert "plan_intervention" not in source
    assert "clip_score" not in source
    assert MIDPOINT_NOTE in source
    assert "not a naturally observed social field" in source


def test_blend_gates_on_short_runs(atlas, locations, plans) -> None:
    for spec in REGIMES:
        donors = _short_donors(atlas, locations, spec, plans, seed=0, timesteps=40)
        runs = {}
        for lambda_value in (0.0, 0.4, 1.0):
            for history in HISTORIES:
                runs[(lambda_value, history)] = simulate_history(
                    atlas,
                    locations,
                    spec,
                    choice_seed_value=0,
                    history=history,
                    lambda_value=lambda_value,
                    plan=plans[(spec["regime_id"], 0)],
                    donors=donors,
                    timesteps=40,
                )
        prefix = runs[(1.0, "CONTROL")]["series"][:21]
        for result in runs.values():
            assert result["series"][:21] == prefix
            assert result["max_blend_error"] <= BLEND_TOLERANCE
            assert result["llm_calls"] == 0
            assert result["rng_draw_counts"] == runs[(0.0, "CONTROL")]["rng_draw_counts"]
        for lambda_value in (0.0, 0.4, 1.0):
            assert runs[(lambda_value, "CONTROL")]["override_count"] == 0
            assert runs[(lambda_value, "PERTURBED")]["override_count"] == 2
            assert runs[(lambda_value, "CONTROL")]["series"][:22] == runs[(1.0, "CONTROL")]["series"][:22]
            assert runs[(lambda_value, "PERTURBED")]["series"][:22] == runs[(1.0, "PERTURBED")]["series"][:22]
        assert runs[(0.0, "CONTROL")]["used_trace"][22:] == runs[(0.0, "PERTURBED")]["used_trace"][22:]
        assert runs[(1.0, "CONTROL")]["used_trace"] == runs[(1.0, "CONTROL")]["endo_trace"]
        assert runs[(1.0, "PERTURBED")]["used_trace"] == runs[(1.0, "PERTURBED")]["endo_trace"]
        if spec["regime_id"] != "S0":
            assert runs[(0.0, "PERTURBED")]["shadow_scores_differed"] is True
        if spec["regime_id"] == "S0":
            for history in HISTORIES:
                expected = runs[(1.0, history)]["canonical_dynamics_sha256"]
                assert runs[(0.0, history)]["canonical_dynamics_sha256"] == expected
                assert runs[(0.4, history)]["canonical_dynamics_sha256"] == expected
        for history, label in (("CONTROL", "CONTROL_ENDOGENOUS"), ("PERTURBED", "PERTURBED_ENDOGENOUS")):
            reference = experiment_011.simulate_branch(
                atlas,
                locations,
                spec,
                choice_seed_value=0,
                branch=label,
                plan=plans[(spec["regime_id"], 0)],
                timesteps=40,
            )
            assert runs[(1.0, history)]["trajectory_sha256"] == reference["trajectory_sha256"]


def test_midpoint_reuses_experiment_011_donors(locations) -> None:
    store = load_donor_store(sorted(locations), wanted={("S0", 0)})
    control = store.field("S0", 0, "CONTROL")
    perturbed = store.field("S0", 0, "PERTURBED")
    agent_id = sorted(locations)[0]
    scores = midpoint_scores(control[22][agent_id], perturbed[22][agent_id])
    for place_id in PLACE_IDS:
        assert scores[place_id] == 0.5 * (control[22][agent_id][place_id] + perturbed[22][agent_id][place_id])
        assert -2.0 <= control[22][agent_id][place_id] <= 2.0
        assert -2.0 <= scores[place_id] <= 2.0


def test_rng_is_matched_and_iteration_order_is_invariant(atlas, locations, plans) -> None:
    spec = REGIMES[1]
    plan = plans[("RW50", 4)]
    donors = _short_donors(atlas, locations, spec, plans, seed=4, timesteps=30)
    recorded: list[int] = []
    original = experiment.Random

    class Tracking:
        def __init__(self, seed: int) -> None:
            recorded.append(seed)
            self._rng = original(seed)

        def random(self) -> float:
            return self._rng.random()

    draws = []
    experiment.Random = Tracking
    try:
        for lambda_value in (0.0, 1.0):
            for history in HISTORIES:
                recorded.clear()
                simulate_history(
                    atlas,
                    locations,
                    spec,
                    choice_seed_value=4,
                    history=history,
                    lambda_value=lambda_value,
                    plan=plan,
                    donors=donors,
                    timesteps=30,
                )
                draws.append(list(recorded))
    finally:
        experiment.Random = original
    assert draws[0] == draws[1] == draws[2] == draws[3]
    expected = 0
    for timestep in range(30):
        for agent_id in sorted(locations):
            if agent_reconsiders(4, timestep, agent_id, REFERENCE_RECONSIDERATION_RATE):
                expected += 1
    assert len(draws[0]) == expected
    forward = simulate_history(
        atlas, locations, spec, choice_seed_value=4, history="PERTURBED", lambda_value=1.0, plan=plan, donors=donors, timesteps=30
    )
    backward = simulate_history(
        atlas,
        {agent_id: locations[agent_id] for agent_id in reversed(list(locations))},
        spec,
        choice_seed_value=4,
        history="PERTURBED",
        lambda_value=1.0,
        plan=plan,
        donors=donors,
        timesteps=30,
    )
    assert trajectory_hash(forward["series"]) == trajectory_hash(backward["series"])


@pytest.mark.slow
def test_lambda1_matches_experiment_011_lineage(atlas, locations, plans) -> None:
    references = json.loads((ROOT / "data" / "worlds" / "world_v0_11" / "run_manifest.json").read_text(encoding="utf-8"))["trajectory_sha256"]
    store = load_donor_store(sorted(locations))
    for spec in REGIMES:
        for seed in CHOICE_SEEDS:
            donors = {
                "CONTROL": store.field(spec["regime_id"], seed, "CONTROL"),
                "PERTURBED": store.field(spec["regime_id"], seed, "PERTURBED"),
            }
            plan = plans[(spec["regime_id"], seed)]
            for history, label in (("CONTROL", "CONTROL_ENDOGENOUS"), ("PERTURBED", "PERTURBED_ENDOGENOUS")):
                result = simulate_history(
                    atlas,
                    locations,
                    spec,
                    choice_seed_value=seed,
                    history=history,
                    lambda_value=1.0,
                    plan=plan,
                    donors=donors,
                )
                assert result["trajectory_sha256"] == references[f"{spec['regime_id']}:{seed}:{label}"]


def test_recorded_run_replays_selected_worlds(atlas, locations, plans) -> None:
    output = ROOT / "data" / "worlds" / "world_v0_12"
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "experiment_012_summary.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is True
    assert manifest["world_count"] == 1800
    assert manifest["llm_calls"] == 0
    assert summary["gates"]["lambda1_lineage"] is True
    assert summary["gates"]["s0_lambda_equivalence"] is True
    assert summary["gates"]["blend_identity"] is True
    assert summary["conclusion"] == overall_conclusion(summary["autonomy_pattern"], summary["persistence_onset_band"])
    assert summary["experiment_011_reference"]["off_class_exp011"] == {"RW50": "OFF-A", "EXP99": "OFF-A"}
    store = load_donor_store(sorted(locations), wanted={("S0", 0), ("RW50", 0), ("EXP99", 0)})
    requested = (("S0", "CONTROL", 1.0), ("RW50", "PERTURBED", 1.0), ("EXP99", "CONTROL", 0.4))
    for regime_id, history, lambda_value in requested:
        spec = next(item for item in REGIMES if item["regime_id"] == regime_id)
        result = simulate_history(
            atlas,
            locations,
            spec,
            choice_seed_value=0,
            history=history,
            lambda_value=lambda_value,
            plan=plans[(regime_id, 0)],
            donors={
                "CONTROL": store.field(regime_id, 0, "CONTROL"),
                "PERTURBED": store.field(regime_id, 0, "PERTURBED"),
            },
        )
        assert result["trajectory_sha256"] == manifest["trajectory_sha256"][f"{regime_id}:0:{history}:{lambda_value:.2f}"]


def _short_donors(atlas, locations, spec, plans, seed: int, timesteps: int) -> dict[str, dict]:
    plan = plans[(spec["regime_id"], seed)]
    donors = {}
    for branch, history in (("CONTROL_ENDOGENOUS", "CONTROL"), ("PERTURBED_ENDOGENOUS", "PERTURBED")):
        donors[history] = experiment_011.simulate_branch(
            atlas, locations, spec, choice_seed_value=seed, branch=branch, plan=plan, timesteps=timesteps
        )["social_field"]
    return donors
