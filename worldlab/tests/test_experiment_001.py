from __future__ import annotations

import inspect
import json

import pytest
from pydantic import ValidationError

from agents.persona_generator import generate_traits
from config.settings import get_settings
from experiments.decision_engine import (
    build_decision_messages,
    chosen_lower_crowding,
    traits_for_condition,
)
from experiments.experiment_001 import fit_clustered_logit, summarize_decisions
from experiments.place_scenarios import generate_scenario_matrix
from models.schemas import PlaceChoiceDecision, export_response_schema


def test_scenario_matrix_is_reproducible_and_balanced() -> None:
    settings = get_settings()
    assert settings.scenario_count == 30
    assert settings.crowd_aversion_low == 0.1
    assert settings.crowd_aversion_mid == 0.5
    assert settings.crowd_aversion_high == 0.9
    first = generate_scenario_matrix(
        scenario_seed=settings.scenario_seed,
        scenario_count=settings.scenario_count,
        crowding_min_gap=settings.crowding_min_gap,
    )
    second = generate_scenario_matrix(
        scenario_seed=settings.scenario_seed,
        scenario_count=settings.scenario_count,
        crowding_min_gap=settings.crowding_min_gap,
    )
    assert first == second
    scenarios = first["scenarios"]
    low_on_a = 0
    tradeoff_kinds = {"attractiveness": 0, "distance": 0, "novelty": 0}
    for scenario in scenarios:
        place_a = scenario["place_a"]
        place_b = scenario["place_b"]
        assert abs(place_a["crowding"] - place_b["crowding"]) >= settings.crowding_min_gap
        low, high = (place_a, place_b) if place_a["crowding"] < place_b["crowding"] else (place_b, place_a)
        less_attractive = low["visual_attractiveness"] < high["visual_attractiveness"]
        farther = low["distance"] > high["distance"]
        less_novel = low["novelty"] < high["novelty"]
        assert less_attractive or farther or less_novel
        assert not (low["visual_attractiveness"] > high["visual_attractiveness"] and low["distance"] < high["distance"] and low["novelty"] > high["novelty"] and not (less_attractive or farther or less_novel))
        tradeoff_kinds["attractiveness"] += int(less_attractive)
        tradeoff_kinds["distance"] += int(farther)
        tradeoff_kinds["novelty"] += int(less_novel)
        if place_a["crowding"] < place_b["crowding"]:
            low_on_a += 1
    assert low_on_a == 15
    assert all(count > 0 for count in tradeoff_kinds.values())
    body = {key: value for key, value in first.items() if key != "matrix_sha256"}
    from experiments.reproducibility_check import canonical_sha256

    assert first["matrix_sha256"] == canonical_sha256(body)


def test_clones_change_only_crowd_aversion() -> None:
    traits, _ = generate_traits(42, 0)
    low = traits_for_condition(traits, 0.1)
    high = traits_for_condition(traits, 0.9)
    assert low.crowd_aversion == 0.1
    assert high.crowd_aversion == 0.9
    for name in traits.model_dump():
        if name == "crowd_aversion":
            continue
        assert getattr(low, name) == getattr(traits, name)
        assert getattr(high, name) == getattr(traits, name)


def test_decision_prompt_does_not_include_profile_prose() -> None:
    traits, _ = generate_traits(42, 3)
    messages = build_decision_messages(
        traits,
        {"crowding": 0.2, "visual_attractiveness": 0.4, "distance": 0.6, "novelty": 0.3},
        {"crowding": 0.8, "visual_attractiveness": 0.7, "distance": 0.2, "novelty": 0.5},
        decision_prompt_version="place_choice_v1",
    )
    text = "\n".join(message["content"] for message in messages)
    for banned in (
        "behavioral_profile",
        "crowd_response",
        "behavioral_constraints",
        "priorities",
        "hypothesis",
        "manipulation",
        "The subject displays",
    ):
        assert banned not in text
    assert "TRAITS_JSON" in text
    assert "PLACE_A" in text
    assert "PLACE_B" in text
    assert "behavioral_profile" not in inspect.signature(build_decision_messages).parameters


def test_chosen_lower_crowding_is_computed_from_attributes() -> None:
    place_a = {"crowding": 0.2, "visual_attractiveness": 0.1, "distance": 0.8, "novelty": 0.2}
    place_b = {"crowding": 0.7, "visual_attractiveness": 0.9, "distance": 0.1, "novelty": 0.8}
    assert chosen_lower_crowding("A", place_a, place_b) == 1
    assert chosen_lower_crowding("B", place_a, place_b) == 0


def test_place_choice_schema_rejects_extra_fields_and_is_flat() -> None:
    PlaceChoiceDecision.model_validate({"choice": "A", "reason_codes": ["crowding", "other"]})
    with pytest.raises(ValidationError):
        PlaceChoiceDecision.model_validate({"choice": "A", "reason_codes": ["crowding"], "summary": "no"})
    with pytest.raises(ValidationError):
        PlaceChoiceDecision.model_validate({"choice": "C", "reason_codes": ["crowding"]})
    schema = export_response_schema(PlaceChoiceDecision)
    assert "behavioral_profile" not in schema["properties"]
    assert set(schema["properties"]) == {"choice", "reason_codes"}
    assert PlaceChoiceDecision.response_schema_name == "place_choice"


def test_summary_monotonicity_and_pilot_regression() -> None:
    decisions = []
    thresholds = {"LOW": 1, "MID": 2, "HIGH": 4}
    for agent_index in range(10):
        for condition, level in (("LOW", 0.1), ("MID", 0.5), ("HIGH", 0.9)):
            for scenario_index in range(4):
                attractiveness_gap = (scenario_index - 1.5) * 0.2
                distance_gap = (agent_index - 4.5) * 0.04
                novelty_gap = ((scenario_index + agent_index) % 3 - 1) * 0.15
                decisions.append(
                    {
                        "base_agent_id": f"agent_{agent_index:04d}",
                        "condition": condition,
                        "crowd_aversion": level,
                        "scenario_id": f"scenario_{scenario_index:03d}",
                        "schema_valid": True,
                        "schema_failure_count": 0,
                        "retry_count": 0,
                        "chosen_lower_crowding": int(
                            (scenario_index + agent_index) % 5 < thresholds[condition]
                        ),
                        "place_a": {
                            "crowding": 0.2,
                            "visual_attractiveness": 0.5 + attractiveness_gap,
                            "distance": 0.5 + distance_gap,
                            "novelty": 0.5 + novelty_gap,
                        },
                        "place_b": {
                            "crowding": 0.8,
                            "visual_attractiveness": 0.5,
                            "distance": 0.5,
                            "novelty": 0.5,
                        },
                    }
                )
    summary = summarize_decisions(decisions, matrix_sha256="abc", expected_decision_count=len(decisions))
    assert summary["proportions"]["LOW"]["rate"] < summary["proportions"]["MID"]["rate"]
    assert summary["proportions"]["MID"]["rate"] <= summary["proportions"]["HIGH"]["rate"]
    assert summary["monotonicity"]["holds"] is True
    assert summary["complete"] is True
    regression = fit_clustered_logit(decisions)
    assert regression["analysis_level"] == "pilot"
    assert regression["status"] == "ok"
    coefficient = regression["coefficients"]["crowd_aversion"]
    for key in ("coefficient", "se", "p_value", "ci95", "odds_ratio"):
        assert key in coefficient
    assert coefficient["coefficient"] > 0
    assert coefficient["odds_ratio"] > 1
    assert len(json.dumps(regression)) > 0
