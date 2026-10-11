"""Focused Experiment 018 tests. The 25,000-world cohort is not executed here."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from world.dynamics_calibration import load_initial_locations
from world.experiment_017 import layer_a_metrics, paired_seed_indexes, resampled_mean, simulate_cell, unpack_series
from world.experiment_018 import (
    CALIBRATION_SEEDS,
    HELDOUT_SEEDS,
    REPLAY_SEEDS,
    SOCIAL_CELL_IDS,
    WEIGHT_CAP,
    bootstrap_means,
    build_decision_cache,
    cohort_seed_q,
    completion_decision,
    condition_specs,
    fit_targets,
    hhi_contrasts,
    percentile_indexes,
    q_components,
    refuse_nonempty_output,
    require_finite_array,
    require_indexed_trajectory,
    seed_mean_from_cache,
    solve_weight,
)
from world.experiment_018 import _world_task
from world.geography import distance_matrix, place_index
from world.utility_atlas import load_frozen_atlas

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def atlas():
    return load_frozen_atlas()


@pytest.fixture(scope="module")
def locations():
    return load_initial_locations()


@pytest.fixture(scope="module")
def route():
    return distance_matrix(), place_index()


def test_heldout_seeds_are_disjoint_from_calibration() -> None:
    assert HELDOUT_SEEDS[0] == 700
    assert HELDOUT_SEEDS[-1] == 1699
    assert len(HELDOUT_SEEDS) == 1000
    assert 0 not in HELDOUT_SEEDS
    assert set(HELDOUT_SEEDS).isdisjoint(CALIBRATION_SEEDS)
    assert set(REPLAY_SEEDS) <= set(HELDOUT_SEEDS)
    assert percentile_indexes(10_000) == (249, 9749)


def test_condition_count_stays_at_twenty_five() -> None:
    weights = {f"{cell_id}__{target}": float(index + 1) for index, cell_id in enumerate(SOCIAL_CELL_IDS) for target in ("q05", "q15", "q25")}
    rows = condition_specs(weights)
    assert len(rows) == 25
    assert len({row["condition_id"] for row in rows}) == 25
    assert sum(row["role"] == "matched" for row in rows) == 18
    assert sum(row["role"] == "reference_150" for row in rows) == 6
    assert rows[-1]["social_weight"] == 0.0


def test_weight_zero_q_is_exact_and_weight_changes_closed_loop(atlas, locations) -> None:
    series = simulate_cell(atlas, locations, "S0", 4, timesteps=12, social_weight=0.0, keep_series=True)
    unpacked = unpack_series(series["series_packed"])
    matrix, places = distance_matrix(), place_index()
    bases, scores = build_decision_cache(atlas, unpacked, matrix, places)
    assert seed_mean_from_cache(bases, scores[0], 0.0) == 0.0
    low = simulate_cell(atlas, locations, "CUM_ZRAW", 4, timesteps=20, social_weight=10.0)
    high = simulate_cell(atlas, locations, "CUM_ZRAW", 4, timesteps=20, social_weight=150.0)
    assert low["social_weight"] == 10.0
    assert low["trajectory_sha256"] != high["trajectory_sha256"]
    hashes = [simulate_cell(atlas, locations, cell_id, 4, timesteps=6, social_weight=0.0)["trajectory_sha256"] for cell_id in list(SOCIAL_CELL_IDS) + ["S0"]]
    assert len(set(hashes)) == 1


def test_cached_q_matches_historical_layer_a_and_changes_with_weight(atlas, route) -> None:
    matrix, places = route
    payload = json.loads((ROOT / "data" / "worlds" / "world_v0_17" / "cohort_b" / "units" / "S0__200.json").read_text(encoding="utf-8"))
    series = unpack_series(payload["series_packed"])
    historical = {row["cell_id"]: row["q"] for row in layer_a_metrics(atlas, series, 200, matrix=matrix, places=places)}
    bases, scores = build_decision_cache(atlas, series, matrix, places)
    for index, cell_id in enumerate(SOCIAL_CELL_IDS):
        got = seed_mean_from_cache(bases, scores[index], 150.0)
        stacked = cohort_seed_q(bases[None, ...], scores[index][None, ...], 150.0)[0]
        assert abs(got - historical[cell_id]) <= 1e-10
        assert abs(float(stacked) - got) <= 1e-12
        assert seed_mean_from_cache(bases, scores[index], 0.0) == 0.0
    assert seed_mean_from_cache(bases, scores[1], 10.0) != seed_mean_from_cache(bases, scores[1], 150.0)


def test_weight_150_trajectory_matches_experiment_017(atlas, locations) -> None:
    stored = json.loads((ROOT / "data" / "worlds" / "world_v0_17" / "cohort_a" / "units" / "CUM_LOGMAX__0.json").read_text(encoding="utf-8"))
    fresh = simulate_cell(atlas, locations, "CUM_LOGMAX", 0, social_weight=150.0)
    assert fresh["trajectory_sha256"] == stored["trajectory_sha256"]


def test_solver_stops_inside_tolerance_and_keeps_existing_files(tmp_path: Path) -> None:
    (tmp_path / "keep.txt").write_text("keep", encoding="utf-8")

    def increasing(weight: float) -> float:
        return weight / (weight + 20.0)

    found = solve_weight(increasing, 0.05)
    assert found["ok"] is True
    assert abs(found["mean_q"] - 0.05) <= 0.0005
    assert found["weight"] <= WEIGHT_CAP
    missed = solve_weight(lambda weight: 0.0 if weight < 10.0 else 1.0, 0.05)
    assert missed["ok"] is False
    assert missed["reason"] == "tolerance_not_met"
    with pytest.raises(RuntimeError, match="not monotone"):
        solve_weight(lambda weight: 0.2 if weight <= 150.0 else 0.05, 0.25)
    assert (tmp_path / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_fit_targets_does_not_receive_heldout_seeds() -> None:
    def evaluate(cell_id: str, weight: float) -> float:
        assert cell_id in SOCIAL_CELL_IDS
        return weight / (weight + 20.0)

    fitted = fit_targets(evaluate)
    assert fitted["ok"] is True
    assert len(fitted["weight_values"]) == 18
    assert set(fitted["weight_values"]) == {f"{cell_id}__{target}" for cell_id in SOCIAL_CELL_IDS for target in ("q05", "q15", "q25")}


def test_source_trajectory_rejects_duplicate_or_missing_steps() -> None:
    rows = [{"timestep": timestep} for timestep in range(5)]
    with pytest.raises(RuntimeError, match="missing"):
        require_indexed_trajectory(rows, timesteps=5)
    rows.append({"timestep": 2})
    with pytest.raises(RuntimeError, match="duplicate"):
        require_indexed_trajectory(rows, timesteps=5)
    with pytest.raises(RuntimeError, match="nonfinite"):
        require_finite_array(np.array([1.0, np.inf]), "score")


def test_bootstrap_batches_match_the_single_stream() -> None:
    values = np.arange(5, dtype=np.float64)[:, None]
    means = bootstrap_means(values, 20260930, draws=4, batch=2)
    expected = paired_seed_indexes(5, 4, 20260930)
    for got, indexes in zip(means, expected):
        assert abs(float(got[0]) - resampled_mean([0, 1, 2, 3, 4], indexes)) < 1e-12
    contrasts = hhi_contrasts({f"{memory}_{transform}__{suffix}": float(index) for index, ((memory, transform), suffix) in enumerate([((memory, transform), suffix) for memory in ("CUM", "RW50", "EXP99") for transform in ("LOGMAX", "ZRAW") for suffix in ("q05", "q15", "q25", "w150")])})
    assert "D_matched_CUM_q05" in contrasts
    assert contrasts["D_diff_CUM_q05"] == contrasts["D_matched_CUM_q05"] - contrasts["D_150_CUM"]


def test_lineage_or_replay_failure_cannot_be_complete() -> None:
    assert completion_decision(lineage_passed=False, replay_passed=True, worlds_complete=True, calibration_passed=True) == "INCOMPLETE"
    assert completion_decision(lineage_passed=True, replay_passed=False, worlds_complete=True, calibration_passed=True) == "INCOMPLETE"
    assert completion_decision(lineage_passed=True, replay_passed=True, worlds_complete=True, calibration_passed=True) == "COMPLETE"


def test_nonempty_output_is_refused(tmp_path: Path) -> None:
    (tmp_path / "existing.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="non-empty"):
        refuse_nonempty_output(tmp_path)


def test_worker_count_does_not_change_the_trajectory(locations) -> None:
    from concurrent.futures import ProcessPoolExecutor

    from world.experiment_018 import _init_worker

    task = {
        "cohort": "test",
        "condition_id": "CUM_ZRAW__q05",
        "cell_id": "CUM_ZRAW",
        "role": "matched",
        "target_id": "q05",
        "choice_seed": 3,
        "social_weight": 12.5,
        "keep_series": False,
        "layer_a": False,
        "timesteps": 8,
    }
    local = _world_task(task)
    with ProcessPoolExecutor(max_workers=2, initializer=_init_worker) as pool:
        remote = pool.submit(_world_task, task).result(timeout=120)
    assert local["trajectory_sha256"] == remote["trajectory_sha256"]
    assert remote["social_weight"] == 12.5
    assert q_components(np.zeros((2, 2, 5)), np.ones((2, 2, 5)), 0.0).sum() == 0.0
