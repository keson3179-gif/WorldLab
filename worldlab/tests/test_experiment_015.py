"""World v0.15 long-horizon dose-response tests. No model calls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import world.experiment_013 as experiment_013
import world.experiment_014 as experiment_014
import world.experiment_015 as experiment
from world.dynamics_calibration import load_initial_locations
from world.experiment_009 import classify_persistence
from world.experiment_012 import HISTORIES, REGIMES
from world.experiment_013 import simulate_history
from world.experiment_014 import long_horizon_class
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
    return experiment_014.load_intervention_registry()


def test_onset_grid_mean_class_and_uncertainty() -> None:
    assert experiment.SOCIAL_LAMBDAS == (0.4, 0.6, 0.8, 1.0)
    assert experiment.S0_LAMBDAS == (0.0, 1.0)
    assert experiment.TIMESTEPS == 2000
    assert experiment.HELDOUT_SEEDS == tuple(range(50, 100))
    assert experiment.long_horizon_class is long_horizon_class
    assert experiment.horizon_onset is not experiment_014.horizon_onset
    with pytest.raises(RuntimeError, match="0.40, 0.60, 0.80, and 1.00"):
        experiment.horizon_onset({0.3: 0.0, 0.35: 0.0, 0.4: 0.03, 1.0: 0.08})
    minimum = experiment.horizon_onset({0.4: 0.03, 0.6: 0.04, 0.8: 0.05, 1.0: 0.08})
    assert minimum["status"] == "MINIMUM_TESTED_PERSISTENT"
    assert minimum["horizon_onset_lambda"] is None
    assert minimum["horizon_onset_band"] is None
    assert minimum["minimum_tested_already_persistent"] is True
    band = experiment.horizon_onset({0.4: 0.02, 0.6: 0.03, 0.8: 0.04, 1.0: 0.08})
    assert band["status"] == "ONSET"
    assert band["horizon_onset_band"]["label"] == "(0.40, 0.60]"
    assert experiment.horizon_onset({0.4: 0.0, 0.6: 0.01, 0.8: 0.03, 1.0: 0.08})["horizon_onset_band"]["label"] == "(0.60, 0.80]"
    assert experiment.horizon_onset({0.4: 0.0, 0.6: 0.01, 0.8: 0.02, 1.0: 0.08})["horizon_onset_band"]["label"] == "(0.80, 1.00]"
    assert experiment.horizon_onset({0.4: 0.03, 0.6: 0.01, 0.8: 0.04, 1.0: 0.08})["status"] == "NONMONOTONIC"
    assert experiment.horizon_onset({0.4: 0.0, 0.6: 0.01, 0.8: 0.02, 1.0: 0.024})["status"] == "NONE"
    assert classify_persistence(0.02) == "attenuated"
    assert classify_persistence(0.025) == "persistent residual"
    cell = {"persistence_class": classify_persistence(0.01547), "seed_persistent_count": 11}
    assert cell["persistence_class"] == "attenuated"
    assert cell["seed_persistent_count"] == 11
    samples = experiment._bootstrap_means([[1.0, 100.0], [100.0, 1.0]], draws=30, seed=experiment.ANALYSIS_BOOTSTRAP_SEED)
    assert all(left + right == pytest.approx(101.0) for left, right in zip(samples[0], samples[1], strict=True))
    zero = experiment.paired_mean_intervals([[0.0] * 50], draws=100)
    assert zero[0]["degenerate_zero"] is True
    assert (zero[0]["ci95_low"], zero[0]["ci95_high"]) == (0.0, 0.0)
    assert "unobserved tail" in zero[0]["note"]
    assert experiment.wilson_interval(0, 50)[0] == pytest.approx(0.0)
    assert experiment.wilson_interval(50, 50)[1] == pytest.approx(1.0)
    low, high = experiment.wilson_interval(11, 50)
    assert low < 0.22 < high
    empty = experiment.set_relation([False, False], [False, False])
    assert empty["jaccard"] is None
    assert empty["relation"] == "empty"
    assert experiment.jaccard(set(), set()) is None
    replaced = experiment.set_relation([True, False], [False, True])
    assert replaced["relation"] == "replaced"
    assert replaced["counts"] == {"AA": 0, "AP": 1, "PA": 1, "PP": 0}


def test_anchor_mismatch_blocks_new_lambdas() -> None:
    keys = experiment.anchor_keys()
    assert len(keys) == 600
    assert len(set(keys)) == 600
    manifest = json.loads((experiment.WORLD_V14_DIR / "run_manifest.json").read_text(encoding="utf-8"))
    assert set(keys) <= set(manifest["trajectory_sha256"])
    rows = [
        {"key": key, "expected": "a", "actual": "a" if index else "b", "match": bool(index)}
        for index, key in enumerate(keys)
    ]
    audit = experiment.build_anchor_audit(rows)
    assert audit["anchor_count"] == 600
    assert audit["mismatched"] == 1
    assert experiment.new_lambdas_allowed(audit) is False
    matched = experiment.build_anchor_audit([{**row, "actual": "a", "match": True} for row in rows])
    assert experiment.new_lambdas_allowed(matched) is True


def test_nonempty_output_is_refused_without_changing_files(tmp_path) -> None:
    target = tmp_path / "world_v0_15"
    target.mkdir()
    sentinel = target / "experiment_015_summary.json"
    sentinel.write_bytes(b'{"frozen": true}\n')
    before = hashlib.sha256(sentinel.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError, match="non-empty output directory"):
        experiment.run_experiment(output_dir=target)
    assert hashlib.sha256(sentinel.read_bytes()).hexdigest() == before
    assert sentinel.read_bytes() == b'{"frozen": true}\n'
    assert list(target.iterdir()) == [sentinel]


def test_completed_v014_directory_is_refused() -> None:
    target = experiment.WORLD_V14_DIR
    config = target / "world_v0_14_config.json"
    before = hashlib.sha256(config.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError, match="non-empty output directory"):
        experiment.run_experiment(output_dir=target)
    assert hashlib.sha256(config.read_bytes()).hexdigest() == before


def test_dose_grid_restores_the_013_lambda_guard() -> None:
    original = experiment_013.LAMBDAS
    with experiment.dose_grid():
        assert 0.6 in experiment_013.LAMBDAS
        assert 0.8 in experiment_013.LAMBDAS
    assert experiment_013.LAMBDAS == original


def test_short_run_keeps_s0_blend_rng_and_agent_order(atlas, locations, plans) -> None:
    s0 = next(item for item in REGIMES if item["regime_id"] == "S0")
    social = next(item for item in REGIMES if item["regime_id"] == "RW50")
    plan = plans[("S0", 50)]
    social_plan = plans[("RW50", 50)]
    with experiment.dose_grid():
        donors = {}
        canonical = {}
        rng = {}
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
            rng[history] = result["rng_draw_counts"]
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
            assert blended["rng_draw_counts"] == rng[history]
            assert blended["max_blend_error"] <= experiment_013.BLEND_TOLERANCE
        social_donors = {}
        social_rng = None
        for history in HISTORIES:
            result = simulate_history(
                atlas,
                locations,
                social,
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
            assert result["used_trace"] == result["endo_trace"]
        dose = simulate_history(
            atlas,
            locations,
            social,
            choice_seed_value=50,
            history="PERTURBED",
            lambda_value=0.6,
            plan=social_plan,
            donors=social_donors,
            timesteps=30,
        )
        assert dose["max_blend_error"] <= experiment_013.BLEND_TOLERANCE
        assert dose["rng_draw_counts"] == social_rng
        assert dose["override_count"] == 2
        reversed_locations = {key: locations[key] for key in reversed(tuple(locations))}
        reordered = simulate_history(
            atlas,
            reversed_locations,
            social,
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
            social,
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


def test_recorded_anchors_match_v014() -> None:
    output = ROOT / "data" / "worlds" / "world_v0_15"
    audit = json.loads((output / "anchor_lineage_audit.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "experiment_015_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    reference = json.loads((experiment.WORLD_V14_DIR / "run_manifest.json").read_text(encoding="utf-8"))
    assert audit["anchor_count"] == 600
    assert audit["matched"] == 600
    assert audit["mismatched"] == 0
    assert len(audit["rows"]) == 600
    assert all(row["match"] and row["actual"] == row["expected"] == reference["trajectory_sha256"][row["key"]] for row in audit["rows"])
    assert summary["world_count"] == 1000
    assert summary["llm_calls"] == 0
    assert summary["gates"]["complete"] is True
    assert manifest["complete"] is True
    assert manifest["llm_calls"] == 0
    assert len(manifest["trajectory_sha256"]) == 1000


@pytest.mark.slow
def test_recorded_replay_matches_selected_worlds(atlas, locations, plans) -> None:
    output = ROOT / "data" / "worlds" / "world_v0_15"
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    requested = (("RW50", "PERTURBED", 0.4, 50), ("EXP99", "CONTROL", 0.8, 99), ("S0", "CONTROL", 0.0, 50))
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
                    plan=plans[(regime_id, seed)],
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
                plan=plans[(regime_id, seed)],
                donors=None if lambda_value == 1.0 else donors,
                timesteps=2000,
                require_donors=lambda_value != 1.0,
            )
            key = f"{regime_id}:{seed}:{history}:{lambda_value:.2f}"
            assert result["trajectory_sha256"] == manifest["trajectory_sha256"][key]
            assert result["llm_calls"] == 0
