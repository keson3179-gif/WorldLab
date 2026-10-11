"""World v0.16 independent-seed replication tests. No model calls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import world.experiment_005 as memory_experiment
import world.experiment_013 as experiment_013
import world.experiment_015 as experiment_015
import world.experiment_016 as experiment
from world.dynamics_calibration import load_initial_locations
from world.experiment_012 import HISTORIES, REGIMES
from world.experiment_013 import simulate_history
from world.experiment_014 import load_intervention_registry
from world.utility_atlas import load_frozen_atlas

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def atlas():
    return load_frozen_atlas()


@pytest.fixture(scope="module")
def locations():
    return load_initial_locations()


@pytest.fixture(scope="module")
def plans():
    return load_intervention_registry()


def test_replication_taxonomy_and_uncertainty() -> None:
    assert experiment.FORMAL_SEEDS == tuple(range(100, 200))
    assert experiment.LINEAGE_SEEDS == (50, 75, 99)
    assert set(experiment.FORMAL_SEEDS).isdisjoint(range(100))
    assert experiment.formal_world_count() == 2000
    assert len(experiment.lineage_keys()) == 60
    replicated = {0.4: 0.0, 0.6: 0.03, 0.8: 0.05, 1.0: 0.08}
    assert experiment.onset_replication(replicated) == "ONSET_REPLICATED"
    assert experiment.onset_replication({0.4: 0.03, 0.6: 0.04, 0.8: 0.05, 1.0: 0.08}) == "ONSET_NOT_REPLICATED"
    assert experiment.overall_onset_replication({"RW50": "ONSET_REPLICATED", "EXP99": "ONSET_REPLICATED"}) == "FULL_ONSET_REPLICATION"
    assert experiment.overall_onset_replication({"RW50": "ONSET_REPLICATED", "EXP99": "ONSET_NOT_REPLICATED"}) == "PARTIAL_ONSET_REPLICATION"
    assert experiment.overall_onset_replication({"RW50": "ONSET_NOT_REPLICATED", "EXP99": "ONSET_NOT_REPLICATED"}) == "ONSET_NOT_REPLICATED"
    assert experiment.class_pattern_replicated({0.4: "attenuated", 0.6: "persistent residual", 0.8: "persistent residual", 1.0: "persistent residual"})
    assert not experiment.class_pattern_replicated({0.4: "attenuated", 0.6: "persistent residual", 0.8: "persistent residual", 1.0: "amplified divergence"})
    assert experiment.adjacent_direction([0.01, 0.02, 0.03]) == "ALL_ADJACENT_DIRECTIONS_REPLICATED"
    assert experiment.adjacent_direction([0.01, -0.02, 0.03]) == "PARTIAL_ADJACENT_DIRECTIONS"
    assert experiment.adjacent_direction([0.0, -0.01, -0.02]) == "NO_POSITIVE_ADJACENT_DIRECTION"
    assert experiment.horizon_onset({0.4: 0.0, 0.6: 0.03, 0.8: 0.04, 1.0: 0.08})["horizon_onset_band"]["label"] == "(0.40, 0.60]"
    assert experiment.horizon_onset({0.4: 0.03, 0.6: 0.04, 0.8: 0.05, 1.0: 0.08})["status"] == "MINIMUM_TESTED_PERSISTENT"
    assert experiment.horizon_onset({0.4: 0.03, 0.6: 0.01, 0.8: 0.04, 1.0: 0.05})["status"] == "NONMONOTONIC"
    assert experiment.horizon_onset({0.4: 0.0, 0.6: 0.0, 0.8: 0.0, 1.0: 0.01})["status"] == "NONE"
    assert experiment.replacement_reproduced({"AA": 1, "AP": 2, "PA": 3, "PP": 4})
    assert not experiment.replacement_reproduced({"AA": 1, "AP": 2, "PA": 0, "PP": 4})
    empty = experiment.identity_reference([False, False], [False, False])
    assert empty["jaccard"] is None
    assert empty["hypergeometric_central_95"] == [0, 0]
    reference = experiment.identity_reference([True, True, False, False, False], [True, False, True, False, False])
    assert reference["expected_overlap"] == pytest.approx(0.8)
    assert reference["observed_overlap"] == 1
    assert reference["hypergeometric_central_95"] == [0, 2]
    indexes = experiment._independent_indexes(2, 3, draws=5, seed=experiment.ANALYSIS_BOOTSTRAP_SEED)
    assert all(len(old) == 2 and len(new) == 3 for old, new in indexes)
    scaled = experiment.independent_mean_differences(
        [[1.0, 2.0], [2.0, 4.0]],
        [[10.0, 20.0, 30.0], [20.0, 40.0, 60.0]],
        draws=30,
    )
    assert scaled[1]["ci95_low"] == pytest.approx(scaled[0]["ci95_low"] * 2)
    assert scaled[1]["mean_difference"] == pytest.approx(scaled[0]["mean_difference"] * 2)
    zero = experiment.paired_mean_intervals([[0.0] * 100], draws=20)
    assert zero[0]["degenerate_zero"] is True
    assert (zero[0]["ci95_low"], zero[0]["ci95_high"]) == (0.0, 0.0)
    low, high = experiment.wilson_interval(0, 100)
    assert low == pytest.approx(0.0)
    assert high < 0.05
    assert experiment.wilson_interval(100, 100)[1] == pytest.approx(1.0)
    rate_low, rate_high = experiment.wilson_interval(22, 100)
    assert rate_low < 0.22 < rate_high


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_saved_seed_audit_is_accepted_from_frozen_artifacts() -> None:
    output = experiment.OUTPUT_DIR
    audit_path = output / "seed_audit.json"
    config_path = output / "world_v0_16_config.json"
    manifest_path = output / "run_manifest.json"
    before = {path: _sha256(path) for path in (audit_path, config_path, manifest_path)}
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert audit["passed"] is True
    assert audit["overlap_sources"] == []
    assert audit["design_fixed_before_new_seed_results"] is True
    assert audit["proposed_choice_seeds"] == list(range(100, 200))
    assert config["formal_choice_seeds"] == list(range(100, 200))
    assert config["seed_audit_passed"] is True
    assert manifest["gates"]["seed_audit"] is True
    assert manifest["seed_count"] == 100
    assert manifest["config_sha256"] == config["config_sha256"]
    assert manifest["files"]["seed_audit.json"] == before[audit_path]
    assert manifest["files"]["world_v0_16_config.json"] == before[config_path]
    keys = experiment.lineage_keys()
    reference = json.loads((experiment.WORLD_V15_DIR / "run_manifest.json").read_text(encoding="utf-8"))
    assert set(keys) <= set(reference["trajectory_sha256"])
    assert all(int(key.split(":")[1]) in experiment.LINEAGE_SEEDS for key in keys)
    assert {path: _sha256(path) for path in before} == before


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_seed_audit_detects_overlap_without_misreading_world_v0_6(tmp_path) -> None:
    root = tmp_path / "isolated_data"
    _write_json(
        root / "worlds" / "world_v0_6" / "run_manifest.json",
        {"trajectory_sha256": {"RW20:100:12": "weight-not-seed", "RW20:150:49": "weight-not-seed"}},
    )
    _write_json(root / "worlds" / "world_v0_2" / "config.json", {"choice_seeds": list(range(20))})
    clear = experiment.audit_seeds(root)
    assert clear["passed"] is True
    assert clear["overlap_sources"] == []
    noted = next(item for item in clear["sources"] if Path(item["path"]).parent.name == "world_v0_6")
    assert "social_weight" in noted["trajectory_note"]
    assert noted["trajectory_seed_min"] == 12
    assert noted["trajectory_seed_max"] == 49
    later = root / "worlds" / "world_v0_17" / "config.json"
    _write_json(later, {"choice_seeds": [150]})
    collided = experiment.audit_seeds(root)
    assert collided["passed"] is False
    assert any(Path(path).parent.name == "world_v0_17" for path in collided["overlap_sources"])
    assert all(not Path(path).is_absolute() for path in collided["overlap_sources"])
    saved = _sha256(experiment.OUTPUT_DIR / "seed_audit.json")
    assert json.loads((experiment.OUTPUT_DIR / "seed_audit.json").read_text(encoding="utf-8"))["passed"] is True
    assert _sha256(experiment.OUTPUT_DIR / "seed_audit.json") == saved


def test_nonempty_and_frozen_outputs_are_refused(tmp_path) -> None:
    target = tmp_path / "world_v0_16"
    target.mkdir()
    sentinel = target / "experiment_016_summary.json"
    sentinel.write_bytes(b'{"frozen": true}\n')
    before = hashlib.sha256(sentinel.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError, match="non-empty output directory"):
        experiment.run_experiment(output_dir=target)
    assert hashlib.sha256(sentinel.read_bytes()).hexdigest() == before
    frozen = experiment.WORLD_V15_DIR / "world_v0_15_config.json"
    frozen_before = hashlib.sha256(frozen.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError, match="non-empty output directory"):
        experiment.run_experiment(output_dir=experiment.WORLD_V15_DIR)
    assert hashlib.sha256(frozen.read_bytes()).hexdigest() == frozen_before


def test_context_managers_restore_after_exceptions(atlas, locations, plans) -> None:
    original = experiment_013.LAMBDAS
    with pytest.raises(RuntimeError, match="boom"):
        with experiment.dose_grid():
            raise RuntimeError("boom")
    assert experiment_013.LAMBDAS == original
    before = memory_experiment.SOCIAL_WEIGHT
    bad = dict(plans[("RW50", 50)])
    bad["target_place"] = "missing-place"
    with pytest.raises(RuntimeError):
        simulate_history(
            atlas,
            locations,
            next(item for item in REGIMES if item["regime_id"] == "RW50"),
            choice_seed_value=50,
            history="PERTURBED",
            lambda_value=1.0,
            plan=bad,
            donors=None,
            timesteps=30,
            require_donors=False,
        )
    assert memory_experiment.SOCIAL_WEIGHT == before


def test_new_registry_rebuilds_and_short_run_keeps_invariants(atlas, locations, plans) -> None:
    spec = next(item for item in REGIMES if item["regime_id"] == "RW50")
    first = experiment_013.build_intervention_record(atlas, locations, spec, 100)
    second = experiment_013.build_intervention_record(atlas, locations, spec, 100)
    assert first["registry_sha256"] == second["registry_sha256"]
    assert first["override_count"] == 2
    assert first["override_consumes_rng"] == 0
    assert first["choice_seed"] == 100
    s0 = next(item for item in REGIMES if item["regime_id"] == "S0")
    plan = plans[("S0", 50)]
    social_plan = plans[("RW50", 50)]
    with experiment.dose_grid():
        donors = {}
        canonical = {}
        for history in HISTORIES:
            result = simulate_history(
                atlas,
                locations,
                s0,
                choice_seed_value=50,
                history=history,
                lambda_value=1.0,
                plan=plan,
                donors=None,
                timesteps=30,
                require_donors=False,
            )
            donors[history] = result["endogenous_field"]
            canonical[history] = result["canonical_dynamics_sha256"]
            rng = result["rng_draw_counts"]
        for history in HISTORIES:
            blended = simulate_history(
                atlas,
                locations,
                s0,
                choice_seed_value=50,
                history=history,
                lambda_value=0.0,
                plan=plan,
                donors=donors,
                timesteps=30,
            )
            assert blended["canonical_dynamics_sha256"] == canonical[history]
            assert blended["rng_draw_counts"] == rng
        social_donors = {}
        social_rng = None
        for history in HISTORIES:
            result = simulate_history(
                atlas,
                locations,
                spec,
                choice_seed_value=50,
                history=history,
                lambda_value=1.0,
                plan=social_plan,
                donors=None,
                timesteps=30,
                require_donors=False,
            )
            social_donors[history] = result["endogenous_field"]
            social_rng = result["rng_draw_counts"]
        dose = simulate_history(
            atlas,
            locations,
            spec,
            choice_seed_value=50,
            history="PERTURBED",
            lambda_value=0.6,
            plan=social_plan,
            donors=social_donors,
            timesteps=30,
        )
        assert dose["max_blend_error"] <= experiment_013.BLEND_TOLERANCE
        assert dose["rng_draw_counts"] == social_rng
        reversed_locations = {key: locations[key] for key in reversed(tuple(locations))}
        reordered = simulate_history(
            atlas,
            reversed_locations,
            spec,
            choice_seed_value=50,
            history="CONTROL",
            lambda_value=1.0,
            plan=social_plan,
            donors=None,
            timesteps=30,
            require_donors=False,
        )
        forward = simulate_history(
            atlas,
            locations,
            spec,
            choice_seed_value=50,
            history="CONTROL",
            lambda_value=1.0,
            plan=social_plan,
            donors=None,
            timesteps=30,
            require_donors=False,
        )
        assert reordered["trajectory_sha256"] == forward["trajectory_sha256"]
    assert experiment_013.LAMBDAS == (0.0, 0.2, 0.25, 0.3, 0.35, 0.4, 1.0)
    assert memory_experiment.SOCIAL_WEIGHT == 150.0


def test_recorded_lineage_and_formal_counts() -> None:
    output = ROOT / "data" / "worlds" / "world_v0_16"
    audit = json.loads((output / "implementation_lineage_audit.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "experiment_016_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    seed_audit = json.loads((output / "seed_audit.json").read_text(encoding="utf-8"))
    reference = json.loads((experiment.WORLD_V15_DIR / "run_manifest.json").read_text(encoding="utf-8"))
    assert seed_audit["passed"] is True
    assert audit["anchor_count"] == 60
    assert audit["matched"] == 60
    assert audit["mismatched"] == 0
    assert all(row["match"] and row["actual"] == row["expected"] == reference["trajectory_sha256"][row["key"]] for row in audit["rows"])
    assert summary["world_count"] == 2000
    assert summary["seed_count"] == 100
    assert summary["lineage_world_count"] == 60
    assert summary["llm_calls"] == 0
    assert summary["gates"]["complete"] is True
    assert manifest["complete"] is True
    assert len(manifest["trajectory_sha256"]) == 2000
    assert {int(key.split(":")[1]) for key in manifest["trajectory_sha256"]} == set(range(100, 200))


@pytest.mark.slow
def test_recorded_replay_matches_selected_worlds(atlas, locations) -> None:
    output = ROOT / "data" / "worlds" / "world_v0_16"
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    registry = {}
    with (output / "replication_interventions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                registry[(row["regime"], int(row["choice_seed"]))] = row
    requested = (("RW50", "PERTURBED", 0.4, 100), ("EXP99", "CONTROL", 1.0, 100), ("S0", "CONTROL", 0.0, 100))
    with experiment.dose_grid():
        for regime_id, history, lambda_value, seed in requested:
            spec = next(item for item in REGIMES if item["regime_id"] == regime_id)
            donors = {}
            for donor_history in HISTORIES:
                donors[donor_history] = simulate_history(
                    atlas,
                    locations,
                    spec,
                    choice_seed_value=seed,
                    history=donor_history,
                    lambda_value=1.0,
                    plan=registry[(regime_id, seed)],
                    donors=None,
                    timesteps=2000,
                    require_donors=False,
                )["endogenous_field"]
            result = simulate_history(
                atlas,
                locations,
                spec,
                choice_seed_value=seed,
                history=history,
                lambda_value=lambda_value,
                plan=registry[(regime_id, seed)],
                donors=None if lambda_value == 1.0 else donors,
                timesteps=2000,
                require_donors=lambda_value != 1.0,
            )
            key = f"{regime_id}:{seed}:{history}:{lambda_value:.2f}"
            assert result["trajectory_sha256"] == manifest["trajectory_sha256"][key]
            assert result["llm_calls"] == 0
