from __future__ import annotations

import inspect

import pytest
from pydantic import ValidationError

from agents.persona_generator import generate_traits
from config.settings import get_settings
from experiments.experiment_001 import OUTPUT_DIR as EXPERIMENT_001_DIR
from experiments.experiment_001b import OUTPUT_DIR as EXPERIMENT_001B_DIR
from experiments.experiment_001c import OUTPUT_DIR as EXPERIMENT_001C_DIR
from experiments.experiment_001d import OUTPUT_DIR as EXPERIMENT_001D_DIR
from experiments.experiment_001d import build_design, select_scenarios
from experiments.experiment_001e import (
    EXPECTED_CALLS,
    OPTION_1,
    OPTION_2,
    OUTPUT_DIR,
    PlaceUtility,
    aggregate_units,
    build_utility_messages,
    compare_with_direct_choice,
    prompt_sha256,
    summarize_utilities,
    _tasks,
)
from experiments.tradeoff_scenarios import generate_scenario_matrix
from models.schemas import export_response_schema


def test_prompt_rates_one_place_and_hides_the_other() -> None:
    traits, _ = generate_traits(42, 0)
    pair = build_design(select_scenarios(generate_scenario_matrix()))["pairs"][0]
    messages = build_utility_messages(traits, pair["physical_option_1"], prompt_version="independent_utility_v3")
    text = "\n".join(message["content"] for message in messages)
    assert _dump(pair["physical_option_1"]) in text
    assert _dump(pair["physical_option_2"]) not in text
    assert "almost no negative effect" in text
    assert "Rate this place." in text
    for banned in ("option a", "option b", "place_a", "place_b", "physical_option", "scenario_id", "replicate", "choose a", "choose between", "ranking", "preferred option"):
        assert banned not in text.lower()
    assert "scenario_id" not in inspect.signature(build_utility_messages).parameters
    assert "replicate_id" not in inspect.signature(build_utility_messages).parameters
    again = build_utility_messages(traits, pair["physical_option_1"], prompt_version="independent_utility_v3")
    assert prompt_sha256(messages) == prompt_sha256(again)


def test_utility_schema_is_scores_only() -> None:
    schema = export_response_schema(PlaceUtility)
    assert set(schema["properties"]) == {"overall_utility", "crowding_fit", "visual_fit", "distance_fit", "novelty_fit"}
    PlaceUtility.model_validate({name: 1000 for name in schema["properties"]})
    with pytest.raises(ValidationError):
        PlaceUtility.model_validate({name: 1000 for name in schema["properties"]} | {"overall_utility": 1001})
    with pytest.raises(ValidationError):
        PlaceUtility.model_validate({name: 0 for name in schema["properties"]} | {"choice": "A"})


def test_call_grid_repeats_each_place_twice_with_one_prompt() -> None:
    settings = get_settings()
    assert settings.temperature == 0.0
    assert settings.llm_seed == 42
    assert EXPECTED_CALLS == 1440
    assert len({OUTPUT_DIR, EXPERIMENT_001_DIR, EXPERIMENT_001B_DIR, EXPERIMENT_001C_DIR, EXPERIMENT_001D_DIR}) == 5
    bases = [{"base_agent_id": f"agent_{index:04d}", "traits": generate_traits(42, index)[0]} for index in range(10)]
    pairs = build_design(select_scenarios(generate_scenario_matrix()))["pairs"]
    tasks = _tasks(bases, pairs)
    assert len(tasks) == 1440
    slots = {(task["base_agent_id"], task["crowd_aversion"], task["scenario_id"], task["substantive_place_id"]) for task in tasks}
    assert len(slots) == 720
    for slot in slots:
        matched = [task for task in tasks if (task["base_agent_id"], task["crowd_aversion"], task["scenario_id"], task["substantive_place_id"]) == slot]
        assert len(matched) == 2
        assert matched[0]["prompt_sha256"] == matched[1]["prompt_sha256"]
        assert matched[0]["messages"] == matched[1]["messages"]


def test_ties_stay_out_of_the_primary_choice_rate() -> None:
    pair = {
        "scenario_id": "item_01",
        "tradeoff_block": "A",
        "avoidance_cost_level": 1,
        "lower_crowding_substantive_id": OPTION_1,
        "physical_option_1": {"crowding": 0.25},
        "physical_option_2": {"crowding": 0.75},
    }
    evaluations = [
        _evaluation("agent_0000", 0.0, "item_01", OPTION_1, 1, 40),
        _evaluation("agent_0000", 0.0, "item_01", OPTION_1, 2, 60),
        _evaluation("agent_0000", 0.0, "item_01", OPTION_2, 1, 10),
        _evaluation("agent_0000", 0.0, "item_01", OPTION_2, 2, 10),
        _evaluation("agent_0000", 0.5, "item_01", OPTION_1, 1, 20),
        _evaluation("agent_0000", 0.5, "item_01", OPTION_1, 2, 20),
        _evaluation("agent_0000", 0.5, "item_01", OPTION_2, 1, 20),
        _evaluation("agent_0000", 0.5, "item_01", OPTION_2, 2, 20),
    ]
    units = aggregate_units(evaluations, [pair], tie_seed=42)
    decisive = [unit for unit in units if unit["complete"] and unit["tie"] is False]
    tied = [unit for unit in units if unit["tie"] is True]
    assert decisive[0]["primary_choice"] == OPTION_1
    assert decisive[0]["low_crowd_chosen"] is True
    assert tied[0]["primary_choice"] is None
    assert tied[0]["low_crowd_chosen"] is None
    assert tied[0]["secondary_tie_break_used"] is True
    summary = summarize_utilities(evaluations, units, matrix_sha256="abc", design_sha256="def", expected_call_count=8)
    assert summary["low_crowd_choice_rate_by_aversion"]["0.00"] == {"rate": 1.0, "n": 1}
    assert summary["low_crowd_choice_rate_by_aversion"]["0.50"]["n"] == 0
    assert summary["ties"]["tie_count"] == 1
    assert summary["ties"]["tie_rate"] == 0.5
    assert summary["utility_advantage_by_aversion"]["0.00"]["mean"] == 40
    assert summary["low_crowd_utility_by_aversion"]["0.00"]["mean"] == 50
    assert summary["architecture_invariance"]["label_invariance"] == 1.0
    assert summary["architecture_invariance"]["position_invariance"] == 1.0
    assert "not evidence about the model's psychology" in summary["architecture_invariance"]["interpretation"]
    repeat = summary["repeatability"]["overall_utility"]
    assert repeat["n_pairs"] == 4
    assert repeat["exact_match_rate"] == 0.75
    comparison = compare_with_direct_choice(
        summary,
        {"a_label_advantage": {"difference": 0.5}, "first_position_advantage": {"difference": -0.1}, "enum_order_advantage": {"difference": 0.0}, "substantive_invariance": {"rate": 0.8}},
        [{"schema_valid": True, "crowd_aversion": 0.0, "chosen_lower_crowding": 1}, {"schema_valid": True, "crowd_aversion": 1.0, "chosen_lower_crowding": 0}],
    )
    assert comparison["note"]
    assert comparison["independent_utility_v3"]["technical_artifact_magnitude"]["label"] == 0.0
    assert comparison["direct_choice_v2"]["technical_artifact_magnitude"]["label"] == 0.5


def _evaluation(agent: str, level: float, scenario: str, place: str, replicate: int, utility: int) -> dict[str, object]:
    scores = {"overall_utility": utility, "crowding_fit": utility, "visual_fit": utility, "distance_fit": utility, "novelty_fit": utility}
    return {
        "base_agent_id": agent,
        "crowd_aversion": level,
        "scenario_id": scenario,
        "substantive_place_id": place,
        "replicate_id": replicate,
        "prompt_sha256": f"{agent}|{level}|{scenario}|{place}",
        "schema_valid": True,
        "schema_failure_count": 0,
        "retry_count": 0,
        **scores,
    }


def _dump(place: dict[str, float]) -> str:
    import json

    return json.dumps(place, ensure_ascii=False, indent=2, allow_nan=False)
