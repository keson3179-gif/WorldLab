"""Focused Experiment 017 tests. Formal cohorts are not executed here."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from world.dynamics_calibration import load_initial_locations
from world.experiment_004 import empty_social, popularity_scores, trajectory_hash
from world.experiment_005 import centered_scores
from world.experiment_017 import (
    COHORT_B_SEEDS,
    CELL_BY_ID,
    LINEAGE_CELL_IDS,
    SOCIAL_CELL_IDS,
    file_sha256,
    layer_a_metrics,
    lineage_failures,
    load_saved_s0_series,
    pack_series,
    paired_seed_indexes,
    percentile_indexes,
    refuse_nonempty_output,
    resampled_mean,
    score_memory,
    seed_contrasts,
    simulate_cell,
    unpack_series,
)
from world.geography import PLACE_IDS
from world.utility_atlas import load_frozen_atlas

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def atlas():
    return load_frozen_atlas()


@pytest.fixture(scope="module")
def locations():
    return load_initial_locations()


def test_score_functions_match_existing_implementations() -> None:
    loo = {place_id: float(index + 1) for index, place_id in enumerate(PLACE_IDS)}
    state = empty_social(["focal", "other"])
    for place_id in PLACE_IDS:
        state.total[place_id] = loo[place_id]
        state.own["focal"][place_id] = 0
        state.own["other"][place_id] = loo[place_id]
    _unclipped, scores, raw = score_memory(state, "focal", "LOGMAX")
    assert scores == popularity_scores(state, "focal")
    rolling = {place_id: float(index + 1) for index, place_id in enumerate(PLACE_IDS)}
    from world.experiment_005 import RollingMemory

    memory = RollingMemory(["focal", "other"], 50)
    memory.total = dict(rolling)
    memory.own["focal"] = {place_id: 0.0 for place_id in PLACE_IDS}
    memory.own["other"] = dict(rolling)
    unclipped, clipped, _raw = score_memory(memory, "focal", "ZRAW")
    expected_unclipped, expected_clipped = centered_scores(rolling)
    assert unclipped == expected_unclipped
    assert clipped == expected_clipped
    assert raw == rolling


def test_initial_memory_is_zero_before_commit(atlas, locations) -> None:
    result = simulate_cell(atlas, locations, "CUM_ZRAW", 0, timesteps=1)
    assert result["zero_variance_rate"] == 1.0
    assert result["decision_count"] == 20
    assert result["llm_calls"] == 0


def test_weight_zero_configurations_share_decisions(atlas, locations) -> None:
    hashes = [
        simulate_cell(atlas, locations, cell_id, 3, timesteps=6, social_weight=0.0)["trajectory_sha256"]
        for cell_id in CELL_BY_ID
    ]
    assert len(set(hashes)) == 1


def test_cross_cells_run_and_keep_late_window(atlas, locations) -> None:
    result = simulate_cell(atlas, locations, "RW50_LOGMAX", 1, timesteps=120)
    assert result["late_hhi"] > 0.0
    assert result["nonfinite_count"] == 0
    assert result["clip_change_count"] == 0


def test_saved_s0_loader_rejects_duplicate_and_missing_steps(tmp_path: Path) -> None:
    path = tmp_path / "world_states.jsonl"
    agents = {f"agent_{index:04d}": "site_01" for index in range(20)}
    lines = []
    for timestep in range(501):
        lines.append({"condition_id": "S0", "choice_seed": 0, "timestep": timestep, "locations": agents})
    lines.append({"condition_id": "S0", "choice_seed": 0, "timestep": 3, "locations": agents})
    path.write_text("".join(json.dumps(line, separators=(",", ":")) + "\n" for line in lines), encoding="utf-8")
    with pytest.raises(RuntimeError, match="duplicate"):
        load_saved_s0_series(path, file_sha256(path), (0,))
    path.write_text("".join(json.dumps(line, separators=(",", ":")) + "\n" for line in lines[:10]), encoding="utf-8")
    with pytest.raises(RuntimeError, match="exactly timesteps"):
        load_saved_s0_series(path, file_sha256(path), (0,))


def test_nonempty_output_is_refused(tmp_path: Path) -> None:
    (tmp_path / "existing.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="non-empty"):
        refuse_nonempty_output(tmp_path)


def test_bootstrap_rows_are_paired_and_cohort_positions_are_not_choice_seeds() -> None:
    import random

    state = random.getstate()
    indexes = paired_seed_indexes(4, 5, 20260928)
    assert random.getstate() == state
    assert len(indexes) == 5
    assert all(len(row) == 4 for row in indexes)
    values = [0.0, 1.0, 4.0, 9.0]
    shifted = [value + 2.0 for value in values]
    gaps = [resampled_mean(shifted, row) - resampled_mean(values, row) for row in indexes]
    assert max(abs(gap - 2.0) for gap in gaps) < 1e-9
    assert resampled_mean([0.0, 1.0, 4.0], [2, 2, 1]) == 3.0
    assert percentile_indexes(10000) == (249, 9749)
    assert COHORT_B_SEEDS[0] == 200
    assert 0 not in COHORT_B_SEEDS
    assert COHORT_B_SEEDS[indexes[0][0]] == 200 + indexes[0][0]


def test_contrasts_use_paired_cell_values() -> None:
    values = {
        "CUM_LOGMAX": 1.0,
        "CUM_ZRAW": 4.0,
        "RW50_LOGMAX": 2.0,
        "RW50_ZRAW": 8.0,
        "EXP99_LOGMAX": 3.0,
        "EXP99_ZRAW": 3.0,
    }
    contrasts = seed_contrasts(values)
    assert contrasts["delta_t_CUM"] == 3.0
    assert contrasts["delta_t_RW50"] == 6.0
    assert contrasts["interaction_RW50"] == 3.0
    assert contrasts["interaction_EXP99"] == -3.0


def test_lineage_mismatch_stops_before_new_cells() -> None:
    expected = {cell_id: {0: "abc"} for cell_id in LINEAGE_CELL_IDS}
    records = [{"cell_id": "CUM_LOGMAX", "choice_seed": 0, "trajectory_sha256": "different"}]
    started = {"new_cells": False}

    def gate() -> None:
        if lineage_failures(records, expected):
            raise RuntimeError("lineage failed")
        started["new_cells"] = True

    with pytest.raises(RuntimeError, match="lineage failed"):
        gate()
    assert started["new_cells"] is False


def test_replay_comparison_does_not_rewrite_saved_hash(atlas, locations, tmp_path: Path) -> None:
    saved = simulate_cell(atlas, locations, "CUM_ZRAW", 4, timesteps=4)
    path = tmp_path / "unit.json"
    path.write_text(json.dumps(saved), encoding="utf-8")
    digest = file_sha256(path)
    replay = simulate_cell(atlas, locations, "CUM_ZRAW", 4, timesteps=4)
    assert replay["trajectory_sha256"] == saved["trajectory_sha256"]
    assert file_sha256(path) == digest


def test_packed_series_round_trip(atlas, locations) -> None:
    result = simulate_cell(atlas, locations, "S0", 2, timesteps=5, keep_series=True)
    restored = unpack_series(result["series_packed"])
    assert trajectory_hash(restored) == result["trajectory_sha256"]
    assert pack_series(restored, sorted(locations)) == result["series_packed"]


def test_historical_cells_match_canonical_hashes_for_seed_zero(atlas, locations) -> None:
    manifest_004 = json.loads((ROOT / "data" / "worlds" / "world_v0_4" / "run_manifest.json").read_text(encoding="utf-8"))
    manifest_005 = json.loads((ROOT / "data" / "worlds" / "world_v0_5" / "run_manifest.json").read_text(encoding="utf-8"))
    expected = {
        "CUM_LOGMAX": manifest_005["trajectory_sha256"]["CUM:0"],
        "RW50_ZRAW": manifest_005["trajectory_sha256"]["RW50:0"],
        "EXP99_ZRAW": manifest_005["trajectory_sha256"]["EXP99:0"],
        "S0": manifest_004["trajectory_sha256"]["S0:0"],
    }
    assert manifest_004["trajectory_sha256"]["S150:0"] == expected["CUM_LOGMAX"]
    for cell_id, digest in expected.items():
        assert simulate_cell(atlas, locations, cell_id, 0)["trajectory_sha256"] == digest


def test_layer_a_uses_supplied_history_without_requiring_closed_loop_output(atlas, locations) -> None:
    reference = simulate_cell(atlas, locations, "S0", 5, timesteps=500, keep_series=True)
    series = unpack_series(reference["series_packed"])
    rows = layer_a_metrics(atlas, series, 5)
    assert [row["cell_id"] for row in rows] == list(SOCIAL_CELL_IDS)
    assert all(row["decision_count"] == 20 * 500 for row in rows)
    assert all(0.0 <= row["q"] <= 1.0 for row in rows)
