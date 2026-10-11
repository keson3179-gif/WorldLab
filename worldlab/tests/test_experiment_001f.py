from __future__ import annotations

import json

import pytest

from agents.persona_generator import generate_traits
from config.settings import get_settings
from experiments.experiment_001 import OUTPUT_DIR as EXPERIMENT_001_DIR
from experiments.experiment_001b import OUTPUT_DIR as EXPERIMENT_001B_DIR
from experiments.experiment_001c import OUTPUT_DIR as EXPERIMENT_001C_DIR
from experiments.experiment_001d import OUTPUT_DIR as EXPERIMENT_001D_DIR
from experiments.experiment_001e import OUTPUT_DIR as EXPERIMENT_001E_DIR
from experiments.experiment_001f import (
    ATTRIBUTE_LEVELS,
    EXPECTED_CALLS,
    OUTPUT_DIR,
    TRAIT_LEVELS,
    build_tasks,
    classify_component,
    run_experiment,
    summarize_evaluations,
)
from llm.client import LLMCallResult
from models.schemas import TRAIT_NAMES


def test_factorial_is_one_place_and_one_trait_change() -> None:
    settings = get_settings()
    assert settings.temperature == 0.0
    assert settings.llm_seed == 42
    assert EXPECTED_CALLS == 750
    assert len({OUTPUT_DIR, EXPERIMENT_001_DIR, EXPERIMENT_001B_DIR, EXPERIMENT_001C_DIR, EXPERIMENT_001D_DIR, EXPERIMENT_001E_DIR}) == 6
    bases = [{"base_agent_id": f"agent_{index:04d}", "traits": generate_traits(42, index)[0]} for index in range(10)]
    tasks = build_tasks(bases)
    assert len(tasks) == 750
    assert len({(task["construct_id"], task["base_agent_id"], f"{task['trait_level']:.2f}", f"{task['attribute_level']:.2f}") for task in tasks}) == 750
    assert "replicate_id" not in tasks[0]
    crowd = [task for task in tasks if task["construct_id"] == "crowd" and task["base_agent_id"] == "agent_0000"]
    assert len(crowd) == 25
    high = next(task for task in crowd if task["attribute_level"] == ATTRIBUTE_LEVELS[-1] and task["trait_level"] == TRAIT_LEVELS[-1])
    assert high["place"]["crowding"] == 0.9
    assert high["place"]["visual_attractiveness"] == 0.5
    assert high["place"]["distance"] == 0.5
    assert high["place"]["novelty"] == 0.5
    traits = _traits_from_messages(high["messages"])
    original = {name: getattr(bases[0]["traits"], name) for name in TRAIT_NAMES}
    assert traits["crowd_aversion"] == 1.0
    assert all(traits[name] == original[name] for name in TRAIT_NAMES if name != "crowd_aversion")
    text = "\n".join(message["content"] for message in high["messages"])
    assert "construct_id" not in text
    assert "trait_level" not in text
    assert "attribute_level" not in text


def test_classification_boundaries() -> None:
    assert classify_component(100, [0, 10, 25])["class_code"] == "A"
    assert classify_component(100, [40, 0, 0])["class_code"] == "B"
    assert classify_component(100, [100, 0, 0])["class_code"] == "C"
    assert classify_component(0, [0, 0, 0])["class_code"] == "C"


def test_synthetic_surfaces_keep_target_and_leakage_apart() -> None:
    separable = _grid("crowd", lambda trait, attribute, agent: (1000 - int(round(800 * trait * attribute)), 400, 400, 400))
    summary = summarize_evaluations(separable, expected_call_count=len(separable))
    crowd = summary["constructs"][0]
    assert crowd["classification"]["class_code"] == "A"
    assert crowd["observed_direction_matches_stated_expectation"] is True
    assert crowd["model"]["status"] == "ok"
    assert crowd["model"]["coefficients"]["trait_value:attribute_value"]["coefficient"] < 0
    assert crowd["monotonicity"]["by_trait_level"]["1.00"]["strict"] is True
    assert crowd["leakage"]["non_targets"]["distance_fit"]["proportion_exact_unchanged"] == 1.0
    partial = _grid(
        "crowd",
        lambda trait, attribute, agent: (1000 - int(round(800 * trait * attribute)), 500 + int(trait * 200), 400, 400),
    )
    assert summarize_evaluations(partial, expected_call_count=len(partial))["constructs"][0]["classification"]["class_code"] == "B"
    holistic = _grid(
        "crowd",
        lambda trait, attribute, agent: (score := 1000 - int(round(800 * trait * attribute)), score, score, score)[:4],
    )
    assert summarize_evaluations(holistic, expected_call_count=len(holistic))["constructs"][0]["classification"]["class_code"] == "C"


def test_runner_stops_before_earlier_experiment_directories(tmp_path) -> None:
    with pytest.raises(RuntimeError):
        run_experiment(output_dir=EXPERIMENT_001E_DIR)
    destination = run_experiment(client=_ConstantClient(), output_dir=tmp_path / "separability")
    summary = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
    assert summary["complete"] is True
    assert summary["call_count"] == 750
    assert summary["schema_failure_count"] == 0
    assert {item["classification"]["class_code"] for item in summary["constructs"]} == {"C"}


class _ConstantClient:
    def complete(self, *, messages, response_model):
        del messages
        parsed = response_model.model_validate({name: 500 for name in ("overall_utility", "crowding_fit", "visual_fit", "distance_fit", "novelty_fit")})
        return LLMCallResult(raw_text=parsed.model_dump_json(), parsed=parsed, model="scripted", temperature=0.0, llm_seed=42, reasoning_effort="none", api_endpoint="scripted")


def _grid(construct_id: str, score_for) -> list[dict[str, object]]:
    rows = []
    names = ("crowding_fit", "visual_fit", "distance_fit", "novelty_fit")
    for agent in range(10):
        for trait in TRAIT_LEVELS:
            for attribute in ATTRIBUTE_LEVELS:
                scores = score_for(trait, attribute, agent)
                rows.append(
                    {
                        "construct_id": construct_id,
                        "base_agent_id": f"agent_{agent:04d}",
                        "trait_level": trait,
                        "attribute_level": attribute,
                        "schema_valid": True,
                        "schema_failure_count": 0,
                        "retry_count": 0,
                        "overall_utility": scores[0],
                        **dict(zip(names, scores, strict=True)),
                    }
                )
    return rows


def _traits_from_messages(messages: list[dict[str, str]]) -> dict[str, float]:
    text = messages[1]["content"]
    start = text.index("TRAITS_JSON\n") + len("TRAITS_JSON\n")
    end = text.index("\nEND_TRAITS_JSON")
    return json.loads(text[start:end])
