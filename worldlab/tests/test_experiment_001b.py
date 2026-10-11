from __future__ import annotations

import re

from agents.persona_generator import TRAIT_DEFINITIONS, generate_traits
from config.settings import get_settings
from experiments.decision_engine import build_decision_messages, traits_for_condition
from experiments.experiment_001 import OUTPUT_DIR as EXPERIMENT_001_DIR
from experiments.experiment_001b import (
    CROWD_AVERSION_DEFINITION,
    CROWD_AVERSION_LEVELS,
    OUTPUT_DIR,
    calibration_trait_definitions,
    fit_pilot_logit,
    summarize_calibration,
)
from experiments.reproducibility_check import canonical_sha256
from experiments.tradeoff_scenarios import PROMPT_FIELDS, generate_scenario_matrix, prompt_places


def test_calibration_matrix_is_deterministic_and_balanced() -> None:
    first = generate_scenario_matrix()
    second = generate_scenario_matrix()
    assert first == second
    body = {key: value for key, value in first.items() if key != "matrix_sha256"}
    assert first["matrix_sha256"] == canonical_sha256(body)
    scenarios = first["scenarios"]
    assert len(scenarios) == 24
    assert [scenario["scenario_id"] for scenario in scenarios] == [f"item_{index:02d}" for index in range(1, 25)]
    assert sum(scenario["lower_crowding_side"] == "A" for scenario in scenarios) == 12
    for scenario in scenarios:
        assert re.fullmatch(r"item_\d{2}", scenario["scenario_id"])
        low, high = _low_high(scenario)
        assert low["crowding"] == 0.25
        assert high["crowding"] == 0.75
    for block, check in {
        "A": _attractiveness_only,
        "B": _distance_only,
        "C": _novelty_only,
        "D": _all_three,
    }.items():
        rows = [scenario for scenario in scenarios if scenario["tradeoff_block"] == block]
        assert [scenario["avoidance_cost_level"] for scenario in rows] == [1, 2, 3, 4, 5, 6]
        gaps = [scenario["sacrificed_gap"] for scenario in rows]
        assert gaps == sorted(gaps)
        assert len(set(gaps)) == 6
        assert [scenario["lower_crowding_side"] for scenario in rows].count("A") == 3
        for scenario in rows:
            check(_low_high(scenario), scenario["sacrificed_gap"])
    single = {scenario["avoidance_cost_level"]: scenario["sacrificed_gap"] for scenario in scenarios if scenario["tradeoff_block"] == "A"}
    combined = {scenario["avoidance_cost_level"]: scenario["sacrificed_gap"] for scenario in scenarios if scenario["tradeoff_block"] == "D"}
    assert all(combined[level] < single[level] for level in range(1, 7))
    for level in range(1, 7):
        rows = [scenario for scenario in scenarios if scenario["avoidance_cost_level"] == level]
        assert [scenario["lower_crowding_side"] for scenario in rows].count("A") == 2


def test_calibration_prompt_states_scale_anchors_without_the_hypothesis() -> None:
    traits, _ = generate_traits(42, 0)
    matrix = generate_scenario_matrix()
    place_a, place_b = prompt_places(matrix["scenarios"][0])
    assert set(place_a) == set(PROMPT_FIELDS)
    messages = build_decision_messages(
        traits,
        place_a,
        place_b,
        decision_prompt_version="place_choice_v2",
        trait_definitions=calibration_trait_definitions(),
    )
    text = "\n".join(message["content"] for message in messages)
    assert CROWD_AVERSION_DEFINITION in text
    assert "almost no negative effect" in text
    assert "extremely strong tendency to avoid crowded places" in text
    assert TRAIT_DEFINITIONS["crowd_aversion"] not in text
    for banned in (
        "hypothesis",
        "manipulation",
        "tradeoff",
        "trade-off",
        "avoidance_cost",
        "difficulty",
        "behavioral_profile",
        "crowd_response",
        "behavioral_constraints",
        "priorities",
        "still avoid",
        "item_",
        "sacrificed_gap",
    ):
        assert banned not in text
    unchanged = build_decision_messages(
        traits,
        place_a,
        place_b,
        decision_prompt_version="place_choice_v1",
    )
    unchanged_text = "\n".join(message["content"] for message in unchanged)
    assert TRAIT_DEFINITIONS["crowd_aversion"] in unchanged_text


def test_clones_and_output_directory_stay_away_from_experiment_001() -> None:
    settings = get_settings()
    assert settings.temperature == 0.0
    assert settings.calibration_base_agent_count == 20
    assert settings.calibration_prompt_version == "place_choice_v2"
    assert settings.decision_prompt_version == "place_choice_v1"
    assert OUTPUT_DIR != EXPERIMENT_001_DIR
    traits, _ = generate_traits(42, 4)
    clones = [traits_for_condition(traits, level) for level in CROWD_AVERSION_LEVELS]
    assert [clone.crowd_aversion for clone in clones] == list(CROWD_AVERSION_LEVELS)
    for clone in clones:
        for name in traits.model_dump():
            if name == "crowd_aversion":
                continue
            assert getattr(clone, name) == getattr(traits, name)


def test_calibration_summary_and_pilot_curves() -> None:
    decisions = _synthetic_decisions()
    summary = summarize_calibration(decisions, matrix_sha256="abc", expected_decision_count=len(decisions))
    rates = [summary["choice_rates_by_level"][label]["rate"] for label in ("0.00", "0.25", "0.50", "0.75", "1.00")]
    assert rates == [1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6]
    assert summary["monotonicity"]["holds"] is True
    assert summary["monotonicity"]["strict"] is True
    assert summary["matched_units"]["unchanged"] == 16
    assert summary["matched_units"]["expected_direction"] == 32
    assert summary["matched_units"]["reverse"] == 0
    assert summary["matched_units"]["switch_rate"] == 32 / 48
    assert summary["criteria"]["A_dose_response"]["strict_increasing"] is True
    assert summary["criteria"]["B_tradeoff_sensitivity"]["non_strict_increasing_mean_threshold"] is True
    assert summary["switch_threshold"]["mean_by_level"]["1.00"] > summary["switch_threshold"]["mean_by_level"]["0.00"]
    assert summary["difficulty_by_aversion"]["1"]["0.00"]["rate"] == 1
    assert summary["difficulty_by_aversion"]["6"]["1.00"]["rate"] == 0
    assert summary["ceiling_floor"]["by_level"]["1.00"]["ceiling"] is False
    assert summary["complete"] is True
    regression = fit_pilot_logit(decisions)
    assert regression["analysis_level"] == "pilot"
    assert regression["status"] == "ok"
    assert regression["interaction"]["coefficient"] > 0
    assert regression["interaction"]["odds_ratio"] > 1
    curves = regression["curves_mean_across_blocks"]
    low_easy = _curve(curves, 0.0, 1)
    high_hard = _curve(curves, 1.0, 6)
    high_easy = _curve(curves, 1.0, 1)
    assert high_easy > low_easy
    assert high_easy > high_hard
    assert 0.0 <= low_easy <= 1.0


def _synthetic_decisions() -> list[dict[str, object]]:
    matrix = generate_scenario_matrix()
    decisions = []
    for agent_index in range(2):
        for level in CROWD_AVERSION_LEVELS:
            for scenario in matrix["scenarios"]:
                cost = int(scenario["avoidance_cost_level"])
                decisions.append(
                    {
                        "base_agent_id": f"agent_{agent_index:04d}",
                        "crowd_aversion": level,
                        "scenario_id": scenario["scenario_id"],
                        "tradeoff_block": scenario["tradeoff_block"],
                        "avoidance_cost_level": cost,
                        "schema_valid": True,
                        "schema_failure_count": 0,
                        "retry_count": 0,
                        "chosen_lower_crowding": int(cost <= 1 + int(level * 4)),
                    }
                )
    return decisions


def _curve(curves: list[dict[str, float | int]], level: float, cost: int) -> float:
    for item in curves:
        if item["crowd_aversion"] == level and item["avoidance_cost_level"] == cost:
            return float(item["probability"])
    raise AssertionError("missing curve point")


def _low_high(scenario: dict[str, object]) -> tuple[dict[str, float], dict[str, float]]:
    place_a = scenario["place_a"]
    place_b = scenario["place_b"]
    assert isinstance(place_a, dict)
    assert isinstance(place_b, dict)
    if place_a["crowding"] < place_b["crowding"]:
        return place_a, place_b
    return place_b, place_a


def _attractiveness_only(pair: tuple[dict[str, float], dict[str, float]], gap: float) -> None:
    low, high = pair
    assert round(high["visual_attractiveness"] - low["visual_attractiveness"], 2) == gap
    assert low["distance"] == high["distance"]
    assert low["novelty"] == high["novelty"]


def _distance_only(pair: tuple[dict[str, float], dict[str, float]], gap: float) -> None:
    low, high = pair
    assert round(low["distance"] - high["distance"], 2) == gap
    assert low["visual_attractiveness"] == high["visual_attractiveness"]
    assert low["novelty"] == high["novelty"]


def _novelty_only(pair: tuple[dict[str, float], dict[str, float]], gap: float) -> None:
    low, high = pair
    assert round(high["novelty"] - low["novelty"], 2) == gap
    assert low["visual_attractiveness"] == high["visual_attractiveness"]
    assert low["distance"] == high["distance"]


def _all_three(pair: tuple[dict[str, float], dict[str, float]], gap: float) -> None:
    low, high = pair
    assert round(high["visual_attractiveness"] - low["visual_attractiveness"], 2) == gap
    assert round(low["distance"] - high["distance"], 2) == gap
    assert round(high["novelty"] - low["novelty"], 2) == gap
