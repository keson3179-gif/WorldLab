"""World v0.13 held-out autonomy-onset tests. No model calls."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import world.experiment_007 as intervention_experiment
import world.experiment_013 as experiment
from world.choice import agent_reconsiders
from world.dynamics_calibration import load_initial_locations
from world.experiment_002 import REFERENCE_RECONSIDERATION_RATE
from world.experiment_004 import trajectory_hash
from world.experiment_007 import plan_intervention
from world.experiment_013 import (
    ALTERNATION_UPPER_QUARTILE,
    BLEND_TOLERANCE,
    BOOTSTRAP_DRAWS,
    BOTH_REPLICATED,
    CONSECUTIVE_TV_UPPER_QUARTILE,
    EXTREME_DOMINANT_SHARE,
    EXTREME_LATE_HHI,
    EXTREME_SATURATION,
    HELDOUT_SEEDS,
    HISTORIES,
    LAMBDAS,
    NONE_REPLICATED,
    ONE_REPLICATED,
    PRIOR_SEEDS,
    REFINEMENT_LAMBDAS,
    REGIMES,
    TIMESTEPS,
    audit_experiment_012_seeds,
    build_intervention_record,
    classify_regime,
    coarse_status,
    midpoint_scores,
    overall_conclusion,
    post_t22_redivergence_onset,
    refine_onset,
    simulate_history,
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
    "data/worlds/world_v0_12/run_manifest.json": "c6db41cc01effac751991a7ead5af595dc8fb2f9c214a2bfc468d9eea4c7c2fb",
    "data/worlds/world_v0_12/experiment_012_summary.json": "c2b1eceef03e47a07cc3191fa8df760ae5761e30d69bf186ce72b797cc9d6138",
    "data/worlds/world_v0_12/world_v0_12_config.json": "490cd53b9f19a4b9629e2061e4dc57c9e85a7d1d197eccd3041c850b605a6b7e",
    "data/worlds/utility_atlas_v1/atlas_manifest.json": "90ff0feb2f846cf521d4a8d9e1b24e0e8f799e4a409a51c84daea374f744f9f1",
    "data/personas/persona_set_v1/manifest.json": "27b121d100c95a40adc4f1cbf282dc3e17f72a8b5f7d5c5b7926866e1cee2c15",
}


@pytest.fixture(scope="module")
def atlas():
    return load_frozen_atlas()


@pytest.fixture(scope="module")
def locations():
    return load_initial_locations()


def _donors(atlas, locations, spec, plan, seed: int, timesteps: int) -> dict:
    donors = {}
    for history in HISTORIES:
        donors[history] = simulate_history(
            atlas,
            locations,
            spec,
            choice_seed_value=seed,
            history=history,
            lambda_value=1.0,
            plan=plan,
            donors=None,
            timesteps=timesteps,
            require_donors=False,
        )["endogenous_field"]
    return donors


def test_frozen_inputs_and_preregistered_rules() -> None:
    assert TIMESTEPS == 500
    assert HELDOUT_SEEDS == tuple(range(50, 100))
    assert PRIOR_SEEDS == tuple(range(50))
    assert LAMBDAS == (0.0, 0.2, 0.25, 0.3, 0.35, 0.4, 1.0)
    assert REFINEMENT_LAMBDAS == (0.2, 0.25, 0.3, 0.35, 0.4)
    assert len(REGIMES) * 50 * 7 * 2 == 2100
    assert BOOTSTRAP_DRAWS == 10000
    assert BLEND_TOLERANCE == 1e-12
    assert (ALTERNATION_UPPER_QUARTILE, CONSECUTIVE_TV_UPPER_QUARTILE) == (0.694375, 0.36148125)
    assert (EXTREME_LATE_HHI, EXTREME_DOMINANT_SHARE, EXTREME_SATURATION) == (0.40, 0.60, 0.10)
    assert plan_intervention is intervention_experiment.plan_intervention
    audit = audit_experiment_012_seeds()
    assert audit["passed"] is True
    assert audit["seeds_50_to_99_in_experiment_012"] is False
    assert audit["experiment_012_metric_seeds"] == list(range(50))
    for path, digest in FROZEN_SHA256.items():
        payload = (ROOT / path).read_bytes()
        import hashlib

        assert hashlib.sha256(payload).hexdigest() == digest
    replicated = coarse_status(0.00327, 0.02877, 0.0957)
    assert replicated == "COARSE_REPLICATED"
    assert coarse_status(0.03, 0.04, 0.09) == "SHIFTED_BELOW"
    assert coarse_status(0.01, 0.02, 0.09) == "SHIFTED_ABOVE"
    assert coarse_status(0.01, 0.04, 0.01) == "GLOBAL_NONMONOTONIC"
    refined = refine_onset({0.2: 0.01, 0.25: 0.02, 0.3: 0.03, 0.35: 0.04, 0.4: 0.05})
    assert refined["refinement_pattern"] == "REFINED_GRADED_ONSET"
    assert refined["refined_onset_lambda"] == 0.3
    assert refined["refined_onset_band"]["label"] == "(0.25, 0.30]"
    edge = refine_onset({0.2: 0.0, 0.25: 0.0, 0.3: 0.0, 0.35: 0.0, 0.4: 0.03})
    assert edge["refined_onset_band"]["label"] == "(0.35, 0.40]"
    early = refine_onset({0.2: 0.0, 0.25: 0.03, 0.3: 0.04, 0.35: 0.04, 0.4: 0.05})
    assert early["refined_onset_band"]["label"] == "(0.20, 0.25]"
    assert refine_onset({0.2: 0.0, 0.25: 0.03, 0.3: 0.01, 0.35: 0.04, 0.4: 0.05})["refinement_pattern"] == "NONMONOTONIC"
    blocked = classify_regime({0.0: 0.0, 0.2: 0.0, 0.25: 0.03, 0.3: 0.04, 0.35: 0.04, 0.4: 0.05, 1.0: 0.01})
    assert blocked["coarse_replication_status"] == "GLOBAL_NONMONOTONIC"
    assert blocked["refined_onset_claim"] is None
    shifted = classify_regime({0.0: 0.0, 0.2: 0.03, 0.25: 0.04, 0.3: 0.04, 0.35: 0.04, 0.4: 0.05, 1.0: 0.08})
    assert shifted["coarse_replication_status"] == "SHIFTED_BELOW"
    assert shifted["refined_onset_claim"] is None
    same = {
        "RW50": classify_regime({0.0: 0.0, 0.2: 0.01, 0.25: 0.02, 0.3: 0.03, 0.35: 0.04, 0.4: 0.05, 1.0: 0.09}),
        "EXP99": classify_regime({0.0: 0.0, 0.2: 0.01, 0.25: 0.04, 0.3: 0.05, 0.35: 0.06, 0.4: 0.07, 1.0: 0.08}),
    }
    conclusion = overall_conclusion(same)
    assert conclusion.startswith(BOTH_REPLICATED)
    assert "(0.25, 0.30]" in conclusion
    assert "(0.20, 0.25]" in conclusion
    assert "social-memory architectures" in conclusion
    only = dict(same)
    only["EXP99"] = classify_regime({0.0: 0.0, 0.2: 0.04, 0.25: 0.04, 0.3: 0.04, 0.35: 0.04, 0.4: 0.05, 1.0: 0.08})
    assert overall_conclusion(only) == ONE_REPLICATED
    neither = {
        name: classify_regime({0.0: 0.0, 0.2: 0.0, 0.25: 0.0, 0.3: 0.0, 0.35: 0.0, 0.4: 0.01, 1.0: 0.09})
        for name in ("RW50", "EXP99")
    }
    assert overall_conclusion(neither) == NONE_REPLICATED
    assert post_t22_redivergence_onset([0.1] * 30 + [0.0] * 20 + [0.03] * 5) == 50
    assert post_t22_redivergence_onset([0.1] * 500) is None
    assert post_t22_redivergence_onset([0.1] * 30 + [0.0] * 40) is None
    scores = midpoint_scores(
        {"site_01": 1.0, "site_02": -1.0, "site_03": 0.5, "site_04": 0.0, "site_05": 2.0},
        {"site_01": -1.0, "site_02": 1.0, "site_03": 0.5, "site_04": 0.0, "site_05": -2.0},
    )
    assert scores["site_01"] == 0.0
    assert scores["site_05"] == 0.0


def test_heldout_intervention_is_deterministic_and_reuses_experiment_007(atlas, locations) -> None:
    spec = REGIMES[1]
    first = build_intervention_record(atlas, locations, spec, 50)
    second = build_intervention_record(atlas, locations, spec, 50)
    assert first["registry_sha256"] == second["registry_sha256"]
    assert first["override_count"] == 2
    assert first["intervention_timestep"] == 21
    assert first["override_consumes_rng"] == 0
    assert first["generator"] == "world.experiment_007.plan_intervention"
    assert len(first["selected_agents"]) == 2
    preview = simulate_history(
        atlas,
        locations,
        spec,
        choice_seed_value=50,
        history="PERTURBED",
        lambda_value=1.0,
        plan=first,
        donors=None,
        timesteps=21,
        require_donors=False,
    )
    live = preview["captured_plan"]
    assert live["target_place"] == first["target_place"]
    assert live["selected_agents"] == first["selected_agents"]
    assert preview["override_count"] == 2
    assert preview["series"][21] != preview["series"][20]
    control = simulate_history(
        atlas,
        locations,
        spec,
        choice_seed_value=50,
        history="CONTROL",
        lambda_value=1.0,
        plan=first,
        donors=None,
        timesteps=21,
        require_donors=False,
    )
    assert control["override_count"] == 0
    assert control["series"][:21] == preview["series"][:21]
    assert control["rng_draw_counts"] == preview["rng_draw_counts"]
    manual = plan_intervention("RW50", 50, live["occupancy"], live["proposed_destinations"])
    assert manual["selected_agents"] == first["selected_agents"]
    assert manual["target_place"] == first["target_place"]
    assert all(live["proposed_destinations"][agent_id] != first["target_place"] for agent_id in first["selected_agents"])


def test_blend_gates_on_heldout_short_run(atlas, locations) -> None:
    spec = REGIMES[1]
    plan = build_intervention_record(atlas, locations, spec, 51)
    donors = _donors(atlas, locations, spec, plan, 51, 40)
    runs = {}
    for lambda_value in (0.0, 0.3, 1.0):
        for history in HISTORIES:
            runs[(lambda_value, history)] = simulate_history(
                atlas,
                locations,
                spec,
                choice_seed_value=51,
                history=history,
                lambda_value=lambda_value,
                plan=plan,
                donors=donors,
                timesteps=40,
            )
    assert runs[(0.0, "CONTROL")]["used_trace"] == runs[(0.0, "PERTURBED")]["used_trace"]
    assert runs[(1.0, "CONTROL")]["used_trace"] == runs[(1.0, "CONTROL")]["endo_trace"]
    assert runs[(1.0, "PERTURBED")]["max_blend_error"] <= BLEND_TOLERANCE
    assert runs[(0.3, "CONTROL")]["max_blend_error"] <= BLEND_TOLERANCE
    assert runs[(0.0, "CONTROL")]["rng_draw_counts"] == runs[(1.0, "PERTURBED")]["rng_draw_counts"]
    assert runs[(0.0, "CONTROL")]["series"][:21] == runs[(1.0, "PERTURBED")]["series"][:21]
    agent_id = sorted(locations)[0]
    midpoint = midpoint_scores(donors["CONTROL"][22][agent_id], donors["PERTURBED"][22][agent_id])
    for place_id in PLACE_IDS:
        assert midpoint[place_id] == 0.5 * (donors["CONTROL"][22][agent_id][place_id] + donors["PERTURBED"][22][agent_id][place_id])
        assert -2.0 <= midpoint[place_id] <= 2.0
    s0 = REGIMES[0]
    s0_plan = build_intervention_record(atlas, locations, s0, 52)
    s0_donors = _donors(atlas, locations, s0, s0_plan, 52, 36)
    hashes = []
    for lambda_value in LAMBDAS:
        result = simulate_history(
            atlas,
            locations,
            s0,
            choice_seed_value=52,
            history="CONTROL",
            lambda_value=lambda_value,
            plan=s0_plan,
            donors=s0_donors,
            timesteps=36,
        )
        hashes.append(result["canonical_dynamics_sha256"])
    assert len(set(hashes)) == 1


def test_rng_is_matched_and_iteration_order_is_invariant(atlas, locations) -> None:
    spec = REGIMES[1]
    plan = build_intervention_record(atlas, locations, spec, 53)
    donors = _donors(atlas, locations, spec, plan, 53, 30)
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
                    choice_seed_value=53,
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
            if agent_reconsiders(53, timestep, agent_id, REFERENCE_RECONSIDERATION_RATE):
                expected += 1
    assert len(draws[0]) == expected
    forward = simulate_history(
        atlas, locations, spec, choice_seed_value=53, history="PERTURBED", lambda_value=1.0, plan=plan, donors=donors, timesteps=30
    )
    backward = simulate_history(
        atlas,
        {agent_id: locations[agent_id] for agent_id in reversed(list(locations))},
        spec,
        choice_seed_value=53,
        history="PERTURBED",
        lambda_value=1.0,
        plan=plan,
        donors=donors,
        timesteps=30,
    )
    assert trajectory_hash(forward["series"]) == trajectory_hash(backward["series"])


def test_recorded_run_replays_selected_worlds(atlas, locations) -> None:
    output = ROOT / "data" / "worlds" / "world_v0_13"
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "experiment_013_summary.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is True
    assert manifest["world_count"] == 2100
    assert manifest["llm_calls"] == 0
    assert summary["gates"]["heldout_seed_audit"] is True
    assert summary["gates"]["s0_lambda_equivalence"] is True
    assert summary["gates"]["blend_identity"] is True
    assert summary["gates"]["matched_experiment_011_lineage"] is False
    assert summary["conclusion"] == overall_conclusion(
        {name: summary["regimes"][name] for name in ("RW50", "EXP99")}
    )
    requested = (("S0", "CONTROL", 1.0, 50), ("RW50", "PERTURBED", 0.3, 50), ("EXP99", "CONTROL", 0.35, 50), ("RW50", "CONTROL", 1.0, 99))
    for regime_id, history, lambda_value, seed in requested:
        spec = next(item for item in REGIMES if item["regime_id"] == regime_id)
        plan = build_intervention_record(atlas, locations, spec, seed)
        donors = _donors(atlas, locations, spec, plan, seed, TIMESTEPS)
        result = simulate_history(
            atlas,
            locations,
            spec,
            choice_seed_value=seed,
            history=history,
            lambda_value=lambda_value,
            plan=plan,
            donors=donors,
        )
        assert result["trajectory_sha256"] == manifest["trajectory_sha256"][f"{regime_id}:{seed}:{history}:{lambda_value:.2f}"]
