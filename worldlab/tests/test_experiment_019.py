from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from agents.persona_generator import write_json
from world.dynamics_calibration import load_initial_locations
from world.experiment_004 import _hhi, _share
from world.experiment_005 import _presence
from world.experiment_017 import _base_utilities, file_sha256, simulate_cell, unpack_series
from world.experiment_018 import OUTPUT_DIR as WORLD_V018_DIR
from world.experiment_019 import (
    ANALYSIS_SEED,
    HHI_TOLERANCE,
    MEMORIES,
    Q_TOLERANCE,
    SETTINGS,
    TRANSFORMS,
    WINDOWS,
    BudgetStop,
    completion_decision,
    concentration_event,
    late_visit_metrics,
    load_frozen_weights,
    paired_contrasts,
    percentile_indexes,
    place_relation,
    probabilities_match_select,
    readout_arrays,
    refuse_nonempty_output,
    resolve_source_path,
    run_experiment,
    score_from_walk,
    shared_indexes,
    bootstrap_column,
    walk_memory,
)
from world.geography import PLACE_IDS, distance_matrix, occupancy_of, place_index
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


def test_concentration_records_start_confirmation_exit_and_absence() -> None:
    agents = _agents()
    series = []
    for timestep in range(40):
        if 5 <= timestep <= 29:
            series.append(_filled(agents, "site_01"))
        elif timestep == 36:
            series.append(_filled(agents, "site_02"))
        else:
            series.append(_mixed(agents))
    event = concentration_event(series, 18)
    assert event["occurred"] is True
    assert event["start"] == 5
    assert event["confirm"] == 24
    assert event["place"] == "site_01"
    assert event["exited"] is True
    held = [_mixed(agents) for _timestep in range(5)] + [_filled(agents, "site_03") for _timestep in range(30)]
    stayed = concentration_event(held, 18)
    assert stayed["start"] == 5
    assert stayed["confirm"] == 24
    assert stayed["exited"] is False
    absent = concentration_event([_mixed(agents) for _timestep in range(40)], 18)
    assert absent == {
        "occurred": False,
        "start": None,
        "confirm": None,
        "place": None,
        "exited": None,
        "threshold": 18,
        "run_length": 20,
    }
    full = concentration_event(series, 20)
    assert full["occurred"] is True
    assert full["place"] == "site_01"


def test_place_relation_keeps_ties_and_disagreements() -> None:
    assert place_relation(["site_01", "site_01", "site_01"]) == "same"
    assert place_relation(["site_01", "site_02", "site_01"]) == "different"
    assert place_relation(["site_01", None, "site_01"]) == "tie"


def test_late_share_matches_the_saved_definition() -> None:
    agents = _agents()
    series = []
    for timestep in range(501):
        if timestep >= 101:
            series.append(_filled(agents, "site_01"))
        else:
            series.append(_mixed(agents))
    got = late_visit_metrics(series)
    counts = _presence(series, 101, 500)
    share = _share(counts)
    assert got["late_hhi"] == pytest.approx(_hhi(share), abs=HHI_TOLERANCE)
    assert got["late_dominant"] == "site_01"
    assert got["tie"] is False
    tied = []
    for _timestep in range(501):
        step = {}
        for index, agent_id in enumerate(agents):
            step[agent_id] = "site_01" if index < 10 else "site_02"
        tied.append(step)
    tie_metrics = late_visit_metrics(tied)
    assert tie_metrics["late_dominant"] is None
    assert tie_metrics["tie"] is True


def test_probabilities_match_select_on_fixed_vectors() -> None:
    for utilities in ([0.0, 1.0, -2.0, 5.0, 3.5], [0.0, 0.0, 0.0, 0.0, 0.0], [10.0, 10.0, 10.0, 10.0, 0.0]):
        assert probabilities_match_select(utilities) <= 1e-12


def test_weight_zero_has_no_social_term_and_readouts_share_one_walk(world: dict) -> None:
    agents = sorted(world["locations"])
    series = []
    for timestep in range(3):
        series.append(_filled(agents, "site_01" if timestep == 1 else "site_02"))
    original = deepcopy(series)
    walked = walk_memory(world["atlas"], series, world["matrix"], world["places"], "CUM")
    assert series == original
    assert float(walked["raw"][0].max()) == 0.0
    assert walked["raw"][1, 0, 0] == 19.0
    assert walked["raw"][1, 0, 1] == 0.0
    occupancy = occupancy_of(series[0])
    base = _base_utilities(world["atlas"], world["matrix"], world["places"], agents[0], series[0][agents[0]], occupancy)
    assert walked["bases"][0, 0, 0] == pytest.approx(base[PLACE_IDS[0]])
    zero = readout_arrays(walked["bases"], walked["LOGMAX"], 0.0)
    zero_z = readout_arrays(walked["bases"], walked["ZRAW"], 0.0)
    assert float(zero["q"].mean()) == 0.0
    assert float(zero_z["q"].mean()) == 0.0
    assert np.all(zero["social"] == 0.0)
    assert np.all(zero_z["social"] == 0.0)
    low = readout_arrays(walked["bases"], walked["LOGMAX"], 10.0)
    high = readout_arrays(walked["bases"], walked["LOGMAX"], 5000.0)
    assert float(high["q"].mean()) > float(low["q"].mean())
    assert "Random" not in walk_memory.__code__.co_names
    assert "agent_reconsiders" not in walk_memory.__code__.co_names


def _row(q_value: float, range_value: float) -> dict[str, float]:
    row = {"q_t101_500": q_value, "logit_range_t101_500": range_value}
    for name, _start, _stop in WINDOWS:
        row[f"q_{name}"] = q_value
    return row


def test_paired_contrasts_use_the_same_window_and_shared_indexes() -> None:
    rows = {}
    level = {"H0": 0.10, "HL": 0.40, "HZ": 0.20}
    for memory in MEMORIES:
        for setting in SETTINGS:
            for history in ("H0", "HL", "HZ"):
                for transform in TRANSFORMS:
                    rows[(memory, setting, history, transform)] = _row(level[history], level[history] / 2)
    contrasts = paired_contrasts(rows)
    assert contrasts["delta_q_CUM_LOGMAX_q25"] == pytest.approx(0.30)
    assert contrasts["delta_q_CUM_ZRAW_q25"] == pytest.approx(0.10)
    assert contrasts["rule_CUM_q25_H0"] == pytest.approx(0.0)
    assert contrasts["interaction_CUM_q25_HL"] == pytest.approx(0.0)
    assert contrasts["q_CUM_q25_HL_LOGMAX"] == pytest.approx(0.40)
    assert percentile_indexes(10_000) == (249, 9749)
    indexes = shared_indexes(4, 30, ANALYSIS_SEED)
    again = shared_indexes(4, 30, ANALYSIS_SEED)
    assert np.array_equal(indexes, again)
    left = np.array([1.0, 2.0, 4.0, 8.0])
    right = np.array([0.5, 0.5, 0.5, 0.5])
    assert np.allclose(bootstrap_column(left - right, indexes), bootstrap_column(left, indexes) - bootstrap_column(right, indexes))


def test_completion_and_directory_guards(tmp_path: Path) -> None:
    assert completion_decision(hashes_passed=False, compatibility_passed=True, replay_passed=True, seeds_complete=True) == "INCOMPLETE"
    assert completion_decision(hashes_passed=True, compatibility_passed=False, replay_passed=True, seeds_complete=True) == "INCOMPLETE"
    assert completion_decision(hashes_passed=True, compatibility_passed=True, replay_passed=False, seeds_complete=True) == "INCOMPLETE"
    assert completion_decision(hashes_passed=True, compatibility_passed=True, replay_passed=True, seeds_complete=False) == "INCOMPLETE"
    assert completion_decision(hashes_passed=True, compatibility_passed=True, replay_passed=True, seeds_complete=True) == "COMPLETE"
    (tmp_path / "existing.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(RuntimeError, match="non-empty"):
        refuse_nonempty_output(tmp_path)
    resolved = resolve_source_path("data\\worlds\\world_v0_18\\calibrated_weights.json")
    assert resolved.is_file()
    assert file_sha256(resolved) == file_sha256(WORLD_V018_DIR / "calibrated_weights.json")
    code_hash = file_sha256(Path(run_experiment.__code__.co_filename))
    protocol = {"protocol_sha256": "abc", "source_sha256": {"world/experiment_019.py": code_hash}}
    write_json(tmp_path / "diagnostic_protocol.json", protocol)
    write_json(tmp_path / "world_v0_19_config.json", {"protocol_sha256": "abc", "source_sha256": protocol["source_sha256"]})
    write_json(tmp_path / "failure_record.json", {"stage": "gate", "llm_calls": 0})
    with pytest.raises(RuntimeError, match="failure"):
        run_experiment(tmp_path, resume=True)
    assert issubclass(BudgetStop, RuntimeError)


def test_h0_q_and_one_replay_match_experiment_018(world: dict) -> None:
    weights = load_frozen_weights()
    unit = json.loads((WORLD_V018_DIR / "heldout" / "units" / "S0__700.json").read_text(encoding="utf-8"))
    series = unpack_series(unit["series_packed"])
    walked = walk_memory(world["atlas"], series, world["matrix"], world["places"], "CUM")
    expected = {}
    with (WORLD_V018_DIR / "heldout" / "layer_a_seed_metrics.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if int(row["choice_seed"]) == 700 and row["condition_id"] in {"CUM_LOGMAX__q25", "CUM_ZRAW__q25", "CUM_LOGMAX__q05"}:
                expected[row["condition_id"]] = float(row["q"])
    scored = score_from_walk(
        walked,
        {"LOGMAX": weights["CUM_LOGMAX__q25"], "ZRAW": weights["CUM_ZRAW__q25"]},
    )
    assert abs(scored["rows"]["LOGMAX"]["q_t1_500"] - expected["CUM_LOGMAX__q25"]) <= Q_TOLERANCE
    assert abs(scored["rows"]["ZRAW"]["q_t1_500"] - expected["CUM_ZRAW__q25"]) <= Q_TOLERANCE
    other = score_from_walk(walked, {"LOGMAX": weights["CUM_LOGMAX__q05"], "ZRAW": weights["CUM_ZRAW__q05"]})
    assert abs(other["rows"]["LOGMAX"]["q_t1_500"] - expected["CUM_LOGMAX__q05"]) <= Q_TOLERANCE
    record = simulate_cell(
        world["atlas"],
        world["locations"],
        "CUM_LOGMAX",
        700,
        social_weight=weights["CUM_LOGMAX__q25"],
        matrix=world["matrix"],
        places=world["places"],
        keep_series=True,
    )
    saved = json.loads((WORLD_V018_DIR / "heldout" / "units" / "CUM_LOGMAX__q25__700.json").read_text(encoding="utf-8"))
    assert record["trajectory_sha256"] == saved["trajectory_sha256"]
    replayed = unpack_series(record["series_packed"])
    metrics = late_visit_metrics(replayed)
    assert abs(metrics["late_hhi"] - saved["late_hhi"]) <= HHI_TOLERANCE
    assert metrics["late_dominant"] == saved["late_dominant"]
