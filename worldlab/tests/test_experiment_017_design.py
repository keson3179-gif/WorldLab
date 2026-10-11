"""Formula checks for the Experiment 017 design. No world run and no model calls."""

from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _checks():
    path = ROOT / "docs" / "experiment_017_design" / "formula_checks.py"
    spec = importlib.util.spec_from_file_location("experiment_017_formula_checks", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def design():
    return _checks()


@pytest.fixture(scope="module")
def report(design):
    return design.run_checks()


def test_softmax_shift_and_equal_inputs_share_choice_probabilities(report) -> None:
    shift = report["softmax_shift_invariance"]
    equal = report["equal_inputs"]
    assert shift["unchanged"] is True
    assert shift["max_abs_probability_change"] == 0.0
    assert equal["all_zero_logmax"] == equal["all_zero_zraw"]
    assert set(equal["equal_positive_logmax"].values()) == {1.0}
    assert set(equal["equal_positive_zraw"].values()) == {0.0}
    assert equal["choice_probability_max_abs_gap"] == 0.0
    assert equal["q_against_base"] == 0.0
    assert report["q_definition"]["value"] > 0.0
    assert report["q_definition"]["includes_stay_put_one_hot"] is False
    assert report["q_definition"]["persistence_threshold_applied"] is False


def test_scale_changes_logmax_and_leaves_zraw_unchanged(report) -> None:
    scale = report["scale"]
    assert scale["logmax_max_abs_change"] > 0.0
    assert scale["zraw_max_abs_change"] < 1e-12
    assert scale["zraw_uses_population_sd"] is True


def test_five_place_population_z_stays_inside_clip(report) -> None:
    bound = report["five_place_bound"]
    assert bound["analytic_bound"] == pytest.approx(2.0)
    assert bound["grid_max_abs_unclipped_z"] <= 2.0 + 1e-12
    assert bound["grid_max_abs_clip_change"] < 1e-12
    assert bound["near_boundary_scores_unchanged_by_clip"] > 0
    assert bound["spike_unclipped_site_01"] == pytest.approx(2.0)
    assert bound["spike_clip_identity"] is True
    assert bound["population_sd_of_spike_z"] == pytest.approx(1.0)


def test_existing_memory_functions_follow_commit_order(report) -> None:
    memory = report["historical_memory"]
    assert set(memory["cumulative_before_commit"].values()) == {0.0}
    assert memory["cumulative_after_one_commit"]["site_02"] == pytest.approx(1.0)
    assert memory["cumulative_after_one_commit"]["site_01"] == 0.0
    assert set(memory["rolling_before_commit"].values()) == {0.0}
    assert memory["rolling_after_one_commit"]["site_02"] == 1.0
    assert memory["rolling_after_one_commit"]["site_01"] == 0.0
    assert memory["rolling_window_2_after_expiration"]["site_02"] == 1.0
    assert memory["rolling_window_2_after_expiration"]["site_03"] == 1.0
    assert memory["rolling_window_2_after_expiration"]["site_01"] == 0.0
    assert memory["rolling_50_length_before_overflow"] == 50
    assert memory["rolling_50_length_after_51st"] == 50
    assert memory["rolling_50_oldest_removed"] is True
    assert set(memory["decay_before_commit"].values()) == {0.0}
    assert memory["decay_first_total_site_01"] == pytest.approx(2.0)
    assert memory["decay_second_loo"]["site_01"] == pytest.approx(1.99)
    assert memory["decay_second_loo"]["site_02"] == pytest.approx(0.0)
    assert memory["negative_loo_raises"] is True


def test_weight_zero_uses_experiment_004_utility_path(report) -> None:
    gate = report["weight_zero_gate"]
    assert gate["experiment_004_weight_zero_totals_match_base"] is True
    assert gate["experiment_004_weight_zero_social_term"] is True
    assert gate["experiment_005_utility_uses_module_weight"] is True


def test_stored_cum_hashes_match_s150(report) -> None:
    anchors = report["stored_anchor_hashes"]
    assert anchors["comparison"] == "stored_manifest_hashes_only"
    assert anchors["future_runner_lineage_executed"] is False
    assert anchors["cum_hash_equals_s150_hash"] is True
    assert anchors["CUM:0"] == anchors["S150:0"]
    assert anchors["CUM:49"] == anchors["S150:49"]


def test_bootstrap_matrix_has_one_row_per_draw(design) -> None:
    small = design.paired_seed_indexes(3, 4, 11)
    assert len(small) == 4
    assert all(isinstance(row, list) and len(row) == 3 for row in small)
    assert all(isinstance(index, int) for row in small for index in row)
    formal = design.paired_seed_indexes(50, 10000, 20260928)
    repeated = design.paired_seed_indexes(50, 10000, 20260928)
    assert formal == repeated
    assert len(formal) == 10000
    assert all(len(row) == 50 for row in formal)
    assert all(0 <= index < 50 for row in formal for index in row)
    assert any(len(set(row)) < len(row) for row in formal)
    state = random.getstate()
    design.paired_seed_indexes(50, 20, 5)
    assert random.getstate() == state
    assert design.percentile_indexes(10000) == (249, 9749)


def test_paired_resample_preserves_a_constant_gap(design) -> None:
    values = [float(seed) for seed in range(50)]
    assert len(set(values)) > 1
    constant = 3.5
    indexes = design.paired_seed_indexes(50, 10000, 20260928)
    shifted = [value + constant for value in values]
    gaps = [
        design.resampled_mean(shifted, row) - design.resampled_mean(values, row)
        for row in indexes
    ]
    assert len(gaps) == 10000
    assert all(len(row) == 50 for row in indexes)
    assert max(abs(gap - constant) for gap in gaps) < 1e-9


def test_resampled_mean_counts_repeated_indexes(design) -> None:
    assert design.resampled_mean([0.0, 1.0, 4.0], [2, 2, 1]) == 3.0


def test_protocol_counts_keep_lineage_and_replay_separate() -> None:
    payload = json.loads((ROOT / "docs" / "experiment_017_design" / "protocol_draft.json").read_text(encoding="utf-8"))
    counts = payload["counts"]
    assert payload["status"] == "DESIGN_ONLY"
    assert payload["run_status"] == "NOT_RUN"
    assert payload["formal_world_executed"] is False
    assert payload["not_a_run_manifest"] is True
    assert counts == {
        "logical_world_conditions": 350,
        "existing_historical_conditions": 200,
        "new_closed_loop_conditions": 150,
        "future_historical_lineage_executions": 200,
        "historical_lineage_subset": 8,
        "historical_lineage_subset_counted_inside": 200,
        "new_cell_post_run_replays": 6,
        "layer_a_offline_seed_histories": 50,
        "formal_worlds_executed_this_round": 0,
        "new_worlds_label_rejected": "350 logical conditions are not 350 new worlds",
    }
    subset = payload["lineage"]["historical_lineage_subset"]
    assert subset["subset_of"] == "historical_lineage"
    assert subset["cells"] == ["CUM_LOGMAX", "RW50_ZRAW", "EXP99_ZRAW", "S0"]
    assert subset["seeds"] == [0, 49]
    assert subset["count"] == 8
    assert subset["additional_count"] == 0
    assert subset["sufficient_by_itself"] is False
    replay = payload["lineage"]["new_cell_post_run_replay"]
    assert [(item["cell_id"], item["seed"]) for item in replay["items"]] == [
        ("CUM_ZRAW", 0),
        ("CUM_ZRAW", 49),
        ("RW50_LOGMAX", 0),
        ("RW50_LOGMAX", 49),
        ("EXP99_LOGMAX", 0),
        ("EXP99_LOGMAX", 49),
    ]
    assert replay["count"] == 6
    assert replay["executed_this_round"] is False
    assert payload["lineage"]["historical_lineage"]["count"] == 200
    assert payload["lineage"]["historical_lineage"]["executed_this_round"] is False
    assert payload["lineage"]["s0_dynamic_equivalence"]["required"] is True
    assert payload["lineage"]["s0_dynamic_equivalence"]["extra_execution_count"] == 0
    assert payload["bootstrap"]["index_matrix_shape"] == [10000, 50]
    assert payload["bootstrap"]["percentile_indexes"] == [249, 9749]
    assert payload["bootstrap"]["formal_research_bootstrap_executed"] is False


def test_report_keeps_formal_bootstrap_unrun(report) -> None:
    rule = report["bootstrap_rule"]
    assert rule["synthetic_matrix_shape"] == [10000, 50]
    assert rule["synthetic_sampling_check_executed"] is True
    assert rule["formal_research_bootstrap_executed"] is False
    assert rule["synthetic_known_row_mean"] == 3.0
    assert rule["synthetic_paired_constant_check"]["max_abs_gap_from_constant"] < 1e-9
    assert rule["synthetic_paired_constant_check"]["formal_research_series_used"] is False
    assert report["formal_world_executed"] is False
    assert report["run_status"] == "NOT_RUN"
    assert report["llm_calls"] == 0
