from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from agents.persona_generator import write_json
from world.dynamics_calibration import load_initial_locations
from world.experiment_017 import CELL_BY_ID, _commit, file_sha256, make_memory, score_memory, unpack_series
from world.experiment_018 import OUTPUT_DIR as WORLD_V018_DIR
from world.experiment_020 import (
    ANALYSIS_SEED,
    ELIGIBILITY_START,
    ELIGIBILITY_STOP,
    FORMAL_SEEDS,
    SWITCH_INDEX,
    BudgetStop,
    _formal_seed,
    _init_worker,
    audit_choice_seeds,
    bootstrap_column,
    completion_decision,
    conditional_means,
    donor_score_cube,
    eligibility_place,
    load_frozen_weights,
    maintenance_record,
    percentile_indexes,
    refuse_nonempty_output,
    run_experiment,
    shared_indexes,
    simulate_recipient,
    source_at,
)
from world.geography import PLACE_IDS, distance_matrix, place_index
from world.utility_atlas import load_frozen_atlas


@pytest.fixture(scope="module")
def world():
    return {
        "atlas": load_frozen_atlas(),
        "locations": load_initial_locations(),
        "matrix": distance_matrix(),
        "places": place_index(),
    }


def _agents(count: int = 20) -> list[str]:
    return [f"a{index:02d}" for index in range(count)]


def _filled(agents: list[str], place_id: str) -> dict[str, str]:
    return {agent_id: place_id for agent_id in agents}


def _mixed(agents: list[str]) -> dict[str, str]:
    return {agent_id: PLACE_IDS[index % 5] for index, agent_id in enumerate(agents)}


def test_donor_score_uses_prior_commits_and_leave_one_out() -> None:
    agents = _agents()
    series = []
    for timestep in range(3):
        series.append(_filled(agents, "site_01" if timestep == 1 else "site_02"))
    cube = donor_score_cube(series, agents, CELL_BY_ID["CUM_LOGMAX"])
    assert float(cube[0].max()) == 0.0
    memory = make_memory(CELL_BY_ID["CUM_LOGMAX"], agents)
    _unclipped, _scores, loo = score_memory(memory, agents[0], "LOGMAX")
    assert all(value == 0.0 for value in loo.values())
    memory = _commit(memory, series[1])
    _unclipped, scores, loo = score_memory(memory, agents[0], "LOGMAX")
    assert loo["site_01"] == 19.0
    assert loo["site_02"] == 0.0
    assert cube[1, 0, 0] == pytest.approx(scores["site_01"])
    assert cube.shape[0] == 2


def test_switch_index_and_source_labels() -> None:
    assert source_at("EE", SWITCH_INDEX - 1) == "E"
    assert source_at("EE", SWITCH_INDEX) == "E"
    assert source_at("EO", SWITCH_INDEX - 1) == "E"
    assert source_at("EO", SWITCH_INDEX) == "O"
    assert source_at("OE", SWITCH_INDEX - 1) == "O"
    assert source_at("OE", SWITCH_INDEX) == "E"
    assert source_at("OO", SWITCH_INDEX) == "O"


def test_formation_maintenance_transfer_and_release() -> None:
    agents = _agents()
    series = [_mixed(agents) for _timestep in range(80)]
    for timestep in range(ELIGIBILITY_START, ELIGIBILITY_STOP + 1):
        series[timestep] = _filled(agents, "site_01")
    assert eligibility_place(series) == "site_01"
    stayed = [dict(step) for step in series]
    for timestep in range(51, 80):
        stayed[timestep] = _filled(agents, "site_01")
    stayed_record = maintenance_record(stayed, "site_01")
    assert stayed_record["maintained_to_horizon"] is True
    assert stayed_record["original_exit"] is None
    assert stayed_record["released"] is False
    assert stayed_record["late_run_other"] is False
    moved = [dict(step) for step in series]
    for timestep in range(51, 80):
        moved[timestep] = _filled(agents, "site_02")
    moved_record = maintenance_record(moved, "site_01")
    assert moved_record["original_exit"] == 51
    assert moved_record["released"] is False
    assert moved_record["late_run_other"] is True
    assert moved_record["maintained_to_horizon"] is False
    dispersed = [dict(step) for step in series]
    for timestep in range(51, 80):
        dispersed[timestep] = _mixed(agents)
    dispersed_record = maintenance_record(dispersed, "site_01")
    assert dispersed_record["original_exit"] == 51
    assert dispersed_record["release_time"] == 51
    assert dispersed_record["late_run_other"] is False
    absent = maintenance_record([_mixed(agents) for _timestep in range(80)], None)
    assert absent["eligible"] is False
    assert absent["maintained_to_horizon"] is None


def test_bootstrap_pairing_and_empty_denominator() -> None:
    assert percentile_indexes(10_000) == (249, 9749)
    assert FORMAL_SEEDS[0] == 1700
    indexes = shared_indexes(4, 30, ANALYSIS_SEED)
    assert np.array_equal(indexes, shared_indexes(4, 30, ANALYSIS_SEED))
    left = np.array([1.0, 2.0, 4.0, 8.0])
    right = np.array([0.25, 0.25, 0.25, 0.25])
    assert np.allclose(bootstrap_column(left - right, indexes), bootstrap_column(left, indexes) - bootstrap_column(right, indexes))
    empty = np.array([False, False, False, False])
    values = np.array([np.nan, np.nan, np.nan, np.nan])
    kept, dropped = conditional_means(values, empty, indexes)
    assert kept.size == 0
    assert dropped == len(indexes)


def test_completion_directory_and_resume(tmp_path: Path) -> None:
    assert completion_decision(lineage_passed=False, gates_passed=True, replay_passed=True, seeds_complete=True) == "INCOMPLETE"
    assert completion_decision(lineage_passed=True, gates_passed=True, replay_passed=False, seeds_complete=True) == "INCOMPLETE"
    assert completion_decision(lineage_passed=True, gates_passed=True, replay_passed=True, seeds_complete=True) == "COMPLETE"
    (tmp_path / "existing.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(RuntimeError, match="non-empty"):
        refuse_nonempty_output(tmp_path)
    code_hash = file_sha256(Path(run_experiment.__code__.co_filename))
    protocol = {"protocol_sha256": "abc", "weights_hex": {"CUM_LOGMAX__q05": "0x1p+1"}, "source_sha256": {"world/experiment_020.py": code_hash}}
    write_json(tmp_path / "experiment_020_protocol.json", protocol)
    write_json(tmp_path / "world_v0_20_config.json", {"protocol_sha256": "abc", "weights_hex": protocol["weights_hex"]})
    write_json(tmp_path / "failure_record.json", {"stage": "lineage", "llm_calls": 0})
    with pytest.raises(RuntimeError, match="failure"):
        run_experiment(tmp_path, resume=True)
    assert issubclass(BudgetStop, RuntimeError)


def test_seed_audit_sees_017_through_019_and_ignores_own_output(tmp_path: Path) -> None:
    for name in ("world_v0_17", "world_v0_18", "world_v0_19"):
        directory = tmp_path / name
        directory.mkdir()
        write_json(directory / "index.json", {"choice_seed": 10})
    write_json(tmp_path / "world_v0_20" / "seed.json", {"choice_seed": 1800})
    clean = audit_choice_seeds(tmp_path)
    assert clean["passed"] is True
    assert clean["includes_world_v0_17_outputs"] is True
    assert clean["includes_world_v0_18_outputs"] is True
    assert clean["includes_world_v0_19_outputs"] is True
    assert clean["includes_world_v0_20_outputs"] is False
    write_json(tmp_path / "world_v0_19" / "overlap.json", {"choice_seed": 1800})
    overlap = audit_choice_seeds(tmp_path)
    assert overlap["passed"] is False
    assert overlap["overlap_sources"]


def test_schedules_share_prefixes_and_switch_memory(world: dict) -> None:
    weights = load_frozen_weights()
    spec = {
        "condition_id": "CUM_LOGMAX__q25",
        "cell_id": "CUM_LOGMAX",
        "target_id": "q25",
        "social_weight": weights["CUM_LOGMAX__q25"],
        "weight_hex": weights["CUM_LOGMAX__q25"].hex(),
    }
    _init_worker()
    payload = _formal_seed({"choice_seed": 1700, "conditions": [spec], "timesteps": 60, "keep_switch_detail": True})
    assert payload["donor_unchanged"] is True
    assert payload["prefix_failures"] == []
    by_schedule = {row["schedule"]: row for row in payload["worlds"]}
    assert by_schedule["OE"]["switch_detail"]["source"] == "E"
    assert by_schedule["OO"]["switch_detail"]["source"] == "O"
    assert by_schedule["OE"]["switch_detail"]["used"] == pytest.approx(by_schedule["OE"]["switch_detail"]["own"])
    donor_series = unpack_series(payload["donor_series_packed"])
    donor_cube = donor_score_cube(donor_series, sorted(world["locations"]), CELL_BY_ID["CUM_LOGMAX"])
    assert by_schedule["OO"]["switch_detail"]["used"] == pytest.approx(donor_cube[SWITCH_INDEX, 0].tolist())
    assert sum(abs(value) for value in by_schedule["OO"]["switch_detail"]["own"]) > 0.0
    recipient = next(item for item in payload["packed"] if item["schedule"] == "OE")
    series = unpack_series(recipient["series_packed"])
    memory = make_memory(CELL_BY_ID["CUM_LOGMAX"], sorted(world["locations"]))
    for timestep in range(SWITCH_INDEX):
        memory = _commit(memory, series[timestep + 1])
    _unclipped, scores, _loo = score_memory(memory, sorted(world["locations"])[0], "LOGMAX")
    assert by_schedule["OE"]["switch_detail"]["own"] == pytest.approx([scores[place_id] for place_id in PLACE_IDS])
    zero = np.zeros((55, len(world["locations"]), 5), dtype=np.float64)
    hashes = []
    for schedule in ("EE", "EO", "OE", "OO"):
        world_row = simulate_recipient(
            world["atlas"],
            world["locations"],
            world["matrix"],
            world["places"],
            "CUM_LOGMAX",
            1700,
            0.0,
            schedule,
            zero,
            timesteps=55,
        )
        assert world_row["q_t1_500"] == 0.0
        hashes.append(world_row["trajectory_sha256"])
    assert len(set(hashes)) == 1


def test_endogenous_schedule_matches_experiment_018(world: dict) -> None:
    weights = load_frozen_weights()
    agents = sorted(world["locations"])
    zeros = np.zeros((500, len(agents), 5), dtype=np.float64)
    expected = {}
    with (WORLD_V018_DIR / "heldout" / "world_metrics.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if int(row["choice_seed"]) == 700 and row["condition_id"] in {"S0", "CUM_LOGMAX__q25"}:
                expected[row["condition_id"]] = row["trajectory_sha256"]
    donor = simulate_recipient(world["atlas"], world["locations"], world["matrix"], world["places"], "S0", 700, 0.0, "EE", zeros)
    recipient = simulate_recipient(
        world["atlas"],
        world["locations"],
        world["matrix"],
        world["places"],
        "CUM_LOGMAX",
        700,
        weights["CUM_LOGMAX__q25"],
        "EE",
        zeros,
    )
    assert donor["trajectory_sha256"] == expected["S0"]
    assert recipient["trajectory_sha256"] == expected["CUM_LOGMAX__q25"]
    assert recipient["field_gap_t1_500"] == 0.0
