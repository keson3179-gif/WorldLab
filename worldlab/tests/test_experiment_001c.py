from __future__ import annotations

import json
from pathlib import Path

import pytest

from config.settings import get_settings
from experiments.decision_engine import build_decision_messages
from experiments.experiment_001 import OUTPUT_DIR as EXPERIMENT_001_DIR
from experiments.experiment_001b import OUTPUT_DIR as EXPERIMENT_001B_DIR
from experiments.experiment_001b import calibration_trait_definitions
from experiments.experiment_001c import (
    EXPECTED_DECISIONS,
    OPTION_1,
    OPTION_2,
    OUTPUT_DIR,
    build_pair_map,
    fit_presentation_models,
    load_frozen_scenarios,
    order_invariant,
    substantive_id,
    summarize_invariance,
)
from experiments.tradeoff_scenarios import PROMPT_FIELDS, generate_scenario_matrix, prompt_places
from agents.persona_generator import generate_traits


def test_mirror_swaps_positions_and_keeps_attribute_values() -> None:
    matrix = generate_scenario_matrix()
    pair_map = build_pair_map(matrix["scenarios"])
    assert len(pair_map["pairs"]) == 24
    again = build_pair_map(matrix["scenarios"])
    assert pair_map["pair_map_sha256"] == again["pair_map_sha256"]
    low_on_a = 0
    for pair in pair_map["pairs"]:
        original = pair["original"]
        mirrored = pair["mirrored"]
        assert original["place_a"] == mirrored["place_b"]
        assert original["place_b"] == mirrored["place_a"]
        assert original["place_a_substantive_id"] == OPTION_1
        assert mirrored["place_a_substantive_id"] == OPTION_2
        assert set(original["place_a"]) == set(PROMPT_FIELDS)
        if original["presentation_side"] == "low_crowd_is_A":
            low_on_a += 1
            assert mirrored["presentation_side"] == "low_crowd_is_B"
        else:
            assert mirrored["presentation_side"] == "low_crowd_is_A"
    assert low_on_a == 12


def test_invariance_compares_substantive_options_not_letters() -> None:
    assert substantive_id("A", "original") == OPTION_1
    assert substantive_id("B", "mirrored") == OPTION_1
    assert substantive_id("A", "mirrored") == OPTION_2
    assert order_invariant("A", "B") == 1
    assert order_invariant("B", "A") == 1
    assert order_invariant("A", "A") == 0
    assert order_invariant("B", "B") == 0


def test_prompt_for_a_mirrored_show_does_not_name_the_gate() -> None:
    traits, _ = generate_traits(42, 1)
    pair = build_pair_map(generate_scenario_matrix()["scenarios"])["pairs"][0]
    place_a, place_b = prompt_places(pair["mirrored"])
    assert place_a == pair["physical_option_2"]
    messages = build_decision_messages(
        traits,
        place_a,
        place_b,
        decision_prompt_version="place_choice_v2",
        trait_definitions=calibration_trait_definitions(),
    )
    text = "\n".join(message["content"] for message in messages)
    for banned in (
        "mirror",
        "original",
        "physical_option",
        "hypothesis",
        "presentation_side",
        "order_invariant",
        "tradeoff",
        "avoidance_cost",
    ):
        assert banned not in text
    assert "PLACE_A" in text
    assert "PLACE_B" in text


def test_frozen_matrix_rejects_a_different_hash(tmp_path: Path) -> None:
    matrix = generate_scenario_matrix()
    path = tmp_path / "scenarios.json"
    path.write_text(json.dumps(matrix), encoding="utf-8")
    assert load_frozen_scenarios(path)["matrix_sha256"] == matrix["matrix_sha256"]
    tampered = dict(matrix)
    tampered["matrix_sha256"] = "0" * 64
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(RuntimeError):
        load_frozen_scenarios(path)


def test_gate_directories_and_decision_count_stay_separate() -> None:
    settings = get_settings()
    assert settings.temperature == 0.0
    assert settings.llm_seed == 42
    assert settings.reasoning_effort == "none"
    assert EXPECTED_DECISIONS == 4800
    assert len({OUTPUT_DIR, EXPERIMENT_001_DIR, EXPERIMENT_001B_DIR}) == 3


def test_pair_summary_counts_discordance_and_both_side_slopes() -> None:
    letter_rows = _letter_rows()
    summary = summarize_invariance(letter_rows, matrix_sha256="abc", pair_map_sha256="def", expected_decision_count=len(letter_rows))
    assert summary["complete_pair_count"] == 240
    assert summary["order_invariance_rate"] == 0.5
    assert summary["mismatch_count"] == 120
    assert summary["a_choice_rate"]["original"]["rate"] == 1.0
    assert summary["a_choice_rate"]["mirrored"]["rate"] == 0.5
    assert summary["discordance"]["original_option_1_mirror_option_2"] == 120
    assert summary["discordance"]["original_option_2_mirror_option_1"] == 0
    assert summary["discordance"]["both_option_1"] == 120
    assert summary["discordance"]["mcnemar"]["status"] == "ok"
    assert summary["pass_threshold"] is None
    assert summary["presentation_effect"]["physical_option_1"]["as_A"] == 1.0
    assert summary["presentation_effect"]["physical_option_1"]["as_B"] == 0.5

    dose_rows = _dose_rows()
    dose = summarize_invariance(dose_rows, matrix_sha256="abc", pair_map_sha256="def", expected_decision_count=len(dose_rows))
    side_a = [dose["dose_response_by_presentation_side"]["low_crowd_is_A"]["rates"][label]["rate"] for label in ("0.00", "0.25", "0.50", "0.75", "1.00")]
    side_b = [dose["dose_response_by_presentation_side"]["low_crowd_is_B"]["rates"][label]["rate"] for label in ("0.00", "0.25", "0.50", "0.75", "1.00")]
    assert side_a == [1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6]
    assert side_b == [1 / 6, 1 / 6, 2 / 6, 2 / 6, 3 / 6]
    assert dose["dose_response_by_presentation_side"]["low_crowd_is_A"]["strict_increasing"] is True
    assert dose["dose_response_by_presentation_side"]["low_crowd_is_B"]["strict_increasing"] is False
    assert dose["dose_response_by_presentation_side"]["low_crowd_is_B"]["non_strict_increasing"] is True
    assert dose["low_crowd_choice_by_presentation_side"]["low_crowd_is_A"]["rate"] > dose["low_crowd_choice_by_presentation_side"]["low_crowd_is_B"]["rate"]
    regression = fit_presentation_models(dose_rows)
    side = regression["presentation_model"]
    interaction = regression["interaction_model"]
    assert side["status"] == "ok"
    assert interaction["status"] == "ok"
    assert side["focus"]["coefficient"] < 0
    assert side["focus"]["odds_ratio"] < 1
    assert "ci95" in side["focus"]
    assert interaction["focus"]["name"]
    assert regression["analysis_level"] == "pilot"


def _letter_rows() -> list[dict[str, object]]:
    pairs = build_pair_map(generate_scenario_matrix()["scenarios"])["pairs"]
    rows = []
    for agent_index in range(2):
        for level in (0.0, 0.25, 0.5, 0.75, 1.0):
            for pair in pairs:
                number = int(str(pair["scenario_id"]).split("_")[1])
                mirror_choice = "B" if number % 2 == 0 else "A"
                for presentation, choice in (("original", "A"), ("mirrored", mirror_choice)):
                    rows.append(_row(agent_index, level, pair, presentation, choice, chosen=0))
    return rows


def _dose_rows() -> list[dict[str, object]]:
    pairs = build_pair_map(generate_scenario_matrix()["scenarios"])["pairs"]
    rows = []
    for agent_index in range(2):
        for level in (0.0, 0.25, 0.5, 0.75, 1.0):
            for pair in pairs:
                for presentation in ("original", "mirrored"):
                    side = pair[presentation]["presentation_side"]
                    cost = int(pair["avoidance_cost_level"])
                    span = int(level * 4) if side == "low_crowd_is_A" else int(level * 2)
                    chosen = int(cost <= 1 + span)
                    choice = "A" if (chosen == 1) == (side == "low_crowd_is_A") else "B"
                    rows.append(_row(agent_index, level, pair, presentation, choice, chosen))
    return rows


def _row(agent_index: int, level: float, pair: dict, presentation: str, choice: str, chosen: int) -> dict[str, object]:
    view = pair[presentation]
    return {
        "base_agent_id": f"agent_{agent_index:04d}",
        "crowd_aversion": level,
        "scenario_id": pair["scenario_id"],
        "presentation": presentation,
        "presentation_side": view["presentation_side"],
        "tradeoff_block": pair["tradeoff_block"],
        "avoidance_cost_level": pair["avoidance_cost_level"],
        "place_a_substantive_id": view["place_a_substantive_id"],
        "place_b_substantive_id": view["place_b_substantive_id"],
        "choice": choice,
        "chosen_lower_crowding": chosen,
        "schema_valid": True,
        "schema_failure_count": 0,
        "retry_count": 0,
    }
