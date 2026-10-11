"""Experiment 001F: trait–attribute separability of independent utility scores.

Each call scores one place. One target trait and one target attribute vary
on a fixed grid. Other traits stay at the base-agent values, and other place
attributes stay at 0.50. This module does not modify earlier experiments and
does not create a world.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import statsmodels.formula.api as smf
from pydantic import ValidationError

from agents.persona_generator import append_jsonl, write_json
from config.settings import PROJECT_ROOT, Settings, get_settings
from experiments.experiment_001 import OUTPUT_DIR as EXPERIMENT_001_DIR
from experiments.experiment_001 import load_base_traits
from experiments.experiment_001b import OUTPUT_DIR as EXPERIMENT_001B_DIR
from experiments.experiment_001c import OUTPUT_DIR as EXPERIMENT_001C_DIR
from experiments.experiment_001d import OUTPUT_DIR as EXPERIMENT_001D_DIR
from experiments.experiment_001e import OUTPUT_DIR as EXPERIMENT_001E_DIR
from experiments.experiment_001e import PROMPT_VERSION, SCORE_FIELDS, PlaceUtility, build_utility_messages, prompt_sha256
from experiments.reproducibility_check import canonical_sha256
from experiments.tradeoff_scenarios import PROMPT_FIELDS
from llm.client import LLMCallError, LLMClient, StructuredLLMClient
from models.schemas import TRAIT_NAMES, AgentTraits

EXPERIMENT_ID = "experiment_001f_trait_attribute_separability"
OUTPUT_DIR = PROJECT_ROOT / "data" / "experiments" / EXPERIMENT_ID
BASE_AGENT_COUNT = 10
TRAIT_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
ATTRIBUTE_LEVELS = (0.1, 0.3, 0.5, 0.7, 0.9)
FIXED_ATTRIBUTE = 0.5
FIT_COMPONENTS = ("crowding_fit", "visual_fit", "distance_fit", "novelty_fit")
LEAKAGE_RATIO_SEPARABLE = 0.25
LEAKAGE_RATIO_HOLISTIC = 1.0
EXPECTED_CALLS = BASE_AGENT_COUNT * len(TRAIT_LEVELS) * len(ATTRIBUTE_LEVELS) * 3
CLASSIFICATION_RULE = (
    "Descriptive classification of component separability, not a validity pass threshold. "
    "Change is the absolute difference from the trait 0.00 score for the same agent and the same attribute level. "
    "A if the target mean absolute change is positive and the largest non-target mean absolute change "
    "is at most one quarter of that target change. "
    "B if the target changes and the largest non-target change is between one quarter and the full target change. "
    "C if the target mean absolute change is zero or any non-target mean absolute change is at least as large as the target change."
)
OVERALL_NOTE = (
    "Within-trait attribute contrasts are the overall_utility summaries. "
    "A shift in overall_utility across trait levels is not interpreted as a shared cardinal scale."
)
CONSTRUCTS = (
    {
        "construct_id": "crowd",
        "trait": "crowd_aversion",
        "attribute": "crowding",
        "target_component": "crowding_fit",
        "expected_span_contrast": "negative",
        "expectation": "As crowding rises, crowding_fit should fall more at high crowd_aversion than at low crowd_aversion.",
    },
    {
        "construct_id": "novelty",
        "trait": "novelty_seeking",
        "attribute": "novelty",
        "target_component": "novelty_fit",
        "expected_span_contrast": "positive",
        "expectation": "As novelty rises, novelty_fit should rise more at high novelty_seeking than at low novelty_seeking.",
    },
    {
        "construct_id": "photography",
        "trait": "photography_interest",
        "attribute": "visual_attractiveness",
        "target_component": "visual_fit",
        "expected_span_contrast": "positive",
        "expectation": "As visual_attractiveness rises, visual_fit should rise more at high photography_interest than at low photography_interest.",
    },
)


def build_tasks(bases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tasks = []
    for spec in CONSTRUCTS:
        for base in bases:
            for trait_level in TRAIT_LEVELS:
                traits = _with_trait(base["traits"], spec["trait"], trait_level)
                for attribute_level in ATTRIBUTE_LEVELS:
                    place = _place(spec["attribute"], attribute_level)
                    messages = build_utility_messages(traits, place, prompt_version=PROMPT_VERSION)
                    tasks.append(
                        {
                            "construct_id": spec["construct_id"],
                            "trait_name": spec["trait"],
                            "attribute_name": spec["attribute"],
                            "target_component": spec["target_component"],
                            "base_agent_id": base["base_agent_id"],
                            "trait_level": trait_level,
                            "attribute_level": attribute_level,
                            "place": place,
                            "messages": messages,
                            "prompt_sha256": prompt_sha256(messages),
                        }
                    )
    return tasks


def classify_component(target_mean_absolute_change: float, non_target_mean_absolute_changes: list[float]) -> dict[str, Any]:
    peak = max(non_target_mean_absolute_changes) if non_target_mean_absolute_changes else 0.0
    ratio = None if target_mean_absolute_change == 0 else peak / target_mean_absolute_change
    if target_mean_absolute_change == 0 or ratio is None or ratio >= LEAKAGE_RATIO_HOLISTIC:
        code = "C"
        label = "holistic sub-scores"
    elif ratio <= LEAKAGE_RATIO_SEPARABLE:
        code = "A"
        label = "separable component measures"
    else:
        code = "B"
        label = "partially separable heuristic scores"
    return {
        "class_code": code,
        "label": label,
        "leakage_ratio": ratio,
        "target_mean_absolute_change": target_mean_absolute_change,
        "largest_non_target_mean_absolute_change": peak,
        "rule": CLASSIFICATION_RULE,
    }


def summarize_evaluations(rows: list[dict[str, Any]], *, expected_call_count: int) -> dict[str, Any]:
    valid = [row for row in rows if row.get("schema_valid") and all(row.get(field) is not None for field in SCORE_FIELDS)]
    constructs = []
    for spec in CONSTRUCTS:
        selected = [row for row in valid if row.get("construct_id") == spec["construct_id"]]
        if selected:
            constructs.append(_summarize_construct(spec, selected))
    return {
        "experiment_id": EXPERIMENT_ID,
        "analysis_level": "pilot",
        "measurement_process": PROMPT_VERSION,
        "call_count": len(rows),
        "valid_call_count": len(valid),
        "expected_call_count": expected_call_count,
        "complete": len(rows) == expected_call_count and len(valid) == expected_call_count,
        "schema_failure_count": sum(int(row.get("schema_failure_count", 0)) for row in rows),
        "invalid_response_count": sum(1 for row in rows if not row.get("schema_valid")),
        "retry_count": sum(int(row.get("retry_count", 0)) for row in rows),
        "classification_rule": CLASSIFICATION_RULE,
        "p_values_are_not_the_classification_criterion": True,
        "constructs": constructs,
    }


def run_experiment(
    settings: Settings | None = None,
    client: LLMClient | None = None,
    *,
    persona_dir: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    active = get_settings() if settings is None else settings
    _require_gate_settings(active)
    destination = output_dir or OUTPUT_DIR
    forbidden = {
        EXPERIMENT_001_DIR.resolve(),
        EXPERIMENT_001B_DIR.resolve(),
        EXPERIMENT_001C_DIR.resolve(),
        EXPERIMENT_001D_DIR.resolve(),
        EXPERIMENT_001E_DIR.resolve(),
    }
    if destination.resolve() in forbidden:
        raise RuntimeError("experiment 001F must not overwrite earlier experiments")
    destination.mkdir(parents=True, exist_ok=True)
    source = persona_dir or (active.personas_dir / "persona_set_v1")
    bases = load_base_traits(source, BASE_AGENT_COUNT)
    design = _design_document(bases)
    write_json(destination / "design.json", design)
    configuration = _model_configuration(active)
    config = _config_snapshot(active, configuration, design["design_sha256"])
    write_json(destination / "config.json", config)
    llm = client if client is not None else StructuredLLMClient(active)
    audit_path = destination / "llm_audit.jsonl"
    evaluations_path = destination / "evaluations.jsonl"
    completed = _completed_keys(evaluations_path)
    tasks = build_tasks(bases)
    for index, task in enumerate(tasks, start=1):
        if _task_key(task) in completed:
            continue
        print(
            f"{index}/{len(tasks)} {task['construct_id']} {task['base_agent_id']} "
            f"{_level_label(task['trait_level'])} {_level_label(task['attribute_level'])}",
            flush=True,
        )
        record = _execute_task(active, llm, task, configuration, audit_path)
        append_jsonl(evaluations_path, record)
    evaluations = _read_jsonl(evaluations_path)
    summary = summarize_evaluations(evaluations, expected_call_count=len(tasks))
    summary["model_configuration"] = configuration
    summary["config_sha256"] = config["config_sha256"]
    summary["design_sha256"] = design["design_sha256"]
    write_json(destination / "summary.json", summary)
    print(
        "classification "
        + " ".join(f"{item['construct_id']}={item['classification']['class_code']}" for item in summary["constructs"]),
        flush=True,
    )
    return destination


def _summarize_construct(spec: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    target = spec["target_component"]
    non_targets = [name for name in FIT_COMPONENTS if name != target]
    leakage = _leakage(rows, target, non_targets)
    classification = classify_component(
        float(leakage["target"]["mean_absolute_change"]),
        [float(leakage["non_targets"][name]["mean_absolute_change"]) for name in non_targets],
    )
    surface = _surface(rows, target)
    spans = _spans(surface)
    model = _fit_surface(rows, target)
    predicted = _predicted_surface(model)
    return {
        "construct_id": spec["construct_id"],
        "trait": spec["trait"],
        "attribute": spec["attribute"],
        "target_component": target,
        "stated_expectation": spec["expectation"],
        "n": len(rows),
        "model": _public_model(model),
        "observed_surface": surface,
        "predicted_surface": predicted,
        "monotonicity": _monotonicity(surface, spec["expected_span_contrast"]),
        "effect_sizes": _effect_sizes(model, spans),
        "observed_direction_matches_stated_expectation": _direction_matches(spans["span_contrast"], spec["expected_span_contrast"]),
        "ceiling_floor": _ceiling_floor(rows, target),
        "leakage": leakage,
        "overall_utility": _overall_utility(rows),
        "classification": classification,
    }


def _leakage(rows: list[dict[str, Any]], target: str, non_targets: list[str]) -> dict[str, Any]:
    return {
        "definition": "absolute change from trait 0.00, holding the agent and the attribute level fixed",
        "target": _change_stats(rows, target),
        "non_targets": {name: _change_stats(rows, name) for name in non_targets},
    }


def _change_stats(rows: list[dict[str, Any]], component: str) -> dict[str, Any]:
    baseline: dict[tuple[str, str], int] = {}
    for row in rows:
        if _level_label(float(row["trait_level"])) != _level_label(TRAIT_LEVELS[0]):
            continue
        baseline[(str(row["base_agent_id"]), _level_label(float(row["attribute_level"])))] = int(row[component])
    deltas = []
    for row in rows:
        if _level_label(float(row["trait_level"])) == _level_label(TRAIT_LEVELS[0]):
            continue
        key = (str(row["base_agent_id"]), _level_label(float(row["attribute_level"])))
        if key not in baseline:
            continue
        deltas.append(int(row[component]) - baseline[key])
    absolute = [abs(value) for value in deltas]
    values = [int(row[component]) for row in rows]
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    mac = 0.0 if not absolute else statistics.fmean(absolute)
    if sd == 0:
        standardized = 0.0 if mac == 0 else None
    else:
        standardized = mac / sd
    return {
        "n_changes": len(deltas),
        "mean_absolute_change": mac,
        "median_absolute_change": 0.0 if not absolute else float(statistics.median(absolute)),
        "proportion_exact_unchanged": 0.0 if not deltas else sum(1 for value in deltas if value == 0) / len(deltas),
        "standardized_change": standardized,
    }


def _surface(rows: list[dict[str, Any]], component: str) -> dict[str, dict[str, dict[str, float | int]]]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in rows:
        grouped[(_level_label(float(row["trait_level"])), _level_label(float(row["attribute_level"])))].append(int(row[component]))
    surface: dict[str, dict[str, dict[str, float | int]]] = {}
    for trait_level in TRAIT_LEVELS:
        trait_label = _level_label(trait_level)
        surface[trait_label] = {}
        for attribute_level in ATTRIBUTE_LEVELS:
            attribute_label = _level_label(attribute_level)
            scores = grouped.get((trait_label, attribute_label), [])
            surface[trait_label][attribute_label] = {
                "mean": 0.0 if not scores else statistics.fmean(scores),
                "n": len(scores),
            }
    return surface


def _spans(surface: dict[str, dict[str, dict[str, float | int]]]) -> dict[str, float | None]:
    low_trait = _level_label(TRAIT_LEVELS[0])
    high_trait = _level_label(TRAIT_LEVELS[-1])
    low_span = _attribute_span(surface, low_trait)
    high_span = _attribute_span(surface, high_trait)
    contrast = None if low_span is None or high_span is None else high_span - low_span
    return {"span_at_trait_low": low_span, "span_at_trait_high": high_span, "span_contrast": contrast}


def _attribute_span(surface: dict[str, dict[str, dict[str, float | int]]], trait_label: str) -> float | None:
    low = surface[trait_label][_level_label(ATTRIBUTE_LEVELS[0])]
    high = surface[trait_label][_level_label(ATTRIBUTE_LEVELS[-1])]
    if int(low["n"]) == 0 or int(high["n"]) == 0:
        return None
    return float(high["mean"]) - float(low["mean"])


def _monotonicity(surface: dict[str, dict[str, dict[str, float | int]]], expected_contrast: str) -> dict[str, Any]:
    decreasing = expected_contrast == "negative"
    by_trait = {}
    for trait_level in TRAIT_LEVELS:
        label = _level_label(trait_level)
        means = [float(surface[label][_level_label(attribute)]["mean"]) for attribute in ATTRIBUTE_LEVELS]
        counts = [int(surface[label][_level_label(attribute)]["n"]) for attribute in ATTRIBUTE_LEVELS]
        if any(count == 0 for count in counts):
            non_strict = None
            strict = None
        elif decreasing:
            non_strict = all(means[index] >= means[index + 1] for index in range(len(means) - 1))
            strict = all(means[index] > means[index + 1] for index in range(len(means) - 1))
        else:
            non_strict = all(means[index] <= means[index + 1] for index in range(len(means) - 1))
            strict = all(means[index] < means[index + 1] for index in range(len(means) - 1))
        by_trait[label] = {"means": means, "non_strict": non_strict, "strict": strict}
    flags = [item["non_strict"] for item in by_trait.values()]
    strict_flags = [item["strict"] for item in by_trait.values()]
    return {
        "expected_attribute_direction": "decreasing" if decreasing else "increasing",
        "by_trait_level": by_trait,
        "non_strict_trait_levels": sum(1 for flag in flags if flag is True),
        "strict_trait_levels": sum(1 for flag in strict_flags if flag is True),
    }


def _effect_sizes(model: dict[str, Any], spans: dict[str, float | None]) -> dict[str, Any]:
    coefficients = model.get("coefficients", {})
    interaction = coefficients.get("trait_value:attribute_value", {}).get("coefficient")
    attribute = coefficients.get("attribute_value", {}).get("coefficient")
    width = ATTRIBUTE_LEVELS[-1] - ATTRIBUTE_LEVELS[0]
    predicted_low = _predicted_span(attribute, interaction, TRAIT_LEVELS[0], width)
    predicted_high = _predicted_span(attribute, interaction, TRAIT_LEVELS[-1], width)
    predicted_contrast = None if predicted_low is None or predicted_high is None else predicted_high - predicted_low
    return {
        "interaction_coefficient": interaction,
        "observed_attribute_span_at_trait_0": spans["span_at_trait_low"],
        "observed_attribute_span_at_trait_1": spans["span_at_trait_high"],
        "observed_span_contrast": spans["span_contrast"],
        "predicted_attribute_span_at_trait_0": predicted_low,
        "predicted_attribute_span_at_trait_1": predicted_high,
        "predicted_span_contrast": predicted_contrast,
        "span_definition": "mean at attribute 0.90 minus mean at attribute 0.10",
    }


def _predicted_span(attribute: float | None, interaction: float | None, trait: float, width: float) -> float | None:
    if attribute is None or interaction is None:
        return None
    return (attribute + interaction * trait) * width


def _direction_matches(contrast: float | None, expected: str) -> bool | None:
    if contrast is None:
        return None
    if expected == "negative":
        return contrast < 0
    return contrast > 0


def _ceiling_floor(rows: list[dict[str, Any]], component: str) -> dict[str, Any]:
    scores = [int(row[component]) for row in rows]
    return {
        "n": len(scores),
        "floor_0_count": sum(1 for score in scores if score == 0),
        "floor_0_rate": _rate(sum(1 for score in scores if score == 0), len(scores)),
        "ceiling_1000_count": sum(1 for score in scores if score == 1000),
        "ceiling_1000_rate": _rate(sum(1 for score in scores if score == 1000), len(scores)),
        "by_trait_level": {
            _level_label(level): _extreme_rates(rows, component, "trait_level", level) for level in TRAIT_LEVELS
        },
        "by_attribute_level": {
            _level_label(level): _extreme_rates(rows, component, "attribute_level", level) for level in ATTRIBUTE_LEVELS
        },
    }


def _extreme_rates(rows: list[dict[str, Any]], component: str, field: str, level: float) -> dict[str, float | int]:
    selected = [int(row[component]) for row in rows if _level_label(float(row[field])) == _level_label(level)]
    floor = sum(1 for score in selected if score == 0)
    ceiling = sum(1 for score in selected if score == 1000)
    return {
        "n": len(selected),
        "floor_0_rate": _rate(floor, len(selected)),
        "ceiling_1000_rate": _rate(ceiling, len(selected)),
    }


def _overall_utility(rows: list[dict[str, Any]]) -> dict[str, Any]:
    surface = _surface(rows, "overall_utility")
    contrasts = {}
    for trait_level in TRAIT_LEVELS:
        label = _level_label(trait_level)
        contrasts[label] = _attribute_span(surface, label)
    model = _fit_surface(rows, "overall_utility")
    return {
        "interpretation": OVERALL_NOTE,
        "within_trait_attribute_contrast": contrasts,
        "within_trait_surface": surface,
        "model": _public_model(model),
    }


def _fit_surface(rows: list[dict[str, Any]], outcome: str) -> dict[str, Any]:
    formula = f"{outcome} ~ trait_value + attribute_value + trait_value:attribute_value"
    report: dict[str, Any] = {"formula": formula, "outcome": outcome, "status": "not_fit", "coefficients": {}}
    frame = pd.DataFrame(
        {
            outcome: [int(row[outcome]) for row in rows],
            "trait_value": [float(row["trait_level"]) for row in rows],
            "attribute_value": [float(row["attribute_level"]) for row in rows],
            "base_agent_id": [str(row["base_agent_id"]) for row in rows],
        }
    )
    if frame.empty or len(frame) < 30 or frame["base_agent_id"].nunique() < 2:
        report["status"] = "insufficient_data"
        return report
    if frame[outcome].nunique() < 2:
        report["status"] = "no_outcome_variation"
        return report
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = smf.ols(formula, data=frame).fit(
                cov_type="cluster",
                cov_kwds={"groups": frame["base_agent_id"]},
            )
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        return report
    report["status"] = "ok"
    report["_fitted"] = fitted
    report["_frame"] = frame
    confidence = fitted.conf_int()
    for name in fitted.params.index:
        report["coefficients"][str(name)] = _coefficient_row(fitted, confidence, str(name))
    return report


def _public_model(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "formula": model["formula"],
        "outcome": model["outcome"],
        "status": model["status"],
        "coefficients": model.get("coefficients", {}),
        "error": model.get("error"),
    }


def _predicted_surface(model: dict[str, Any]) -> dict[str, dict[str, float | None]]:
    fitted = model.get("_fitted")
    if fitted is None:
        return {}
    surface: dict[str, dict[str, float | None]] = {}
    for trait_level in TRAIT_LEVELS:
        trait_label = _level_label(trait_level)
        surface[trait_label] = {}
        for attribute_level in ATTRIBUTE_LEVELS:
            grid = pd.DataFrame({"trait_value": [trait_level], "attribute_value": [attribute_level]})
            predicted = float(fitted.predict(grid).iloc[0])
            surface[trait_label][_level_label(attribute_level)] = _finite_or_none(predicted)
    return surface


def _coefficient_row(fitted: Any, confidence: Any, name: str) -> dict[str, Any]:
    low, high = (float(value) for value in confidence.loc[name])
    se = float(fitted.bse[name])
    row = {
        "coefficient": _finite_or_none(float(fitted.params[name])),
        "se": _finite_or_none(se),
        "p_value": _finite_or_none(float(fitted.pvalues[name])),
        "ci95": [_finite_or_none(low), _finite_or_none(high)],
    }
    if row["se"] is None:
        row["se_note"] = "Cluster-robust SE is undefined for this coefficient."
    return row


def _with_trait(traits: AgentTraits, name: str, value: float) -> AgentTraits:
    payload = {field: getattr(traits, field) for field in TRAIT_NAMES}
    payload[name] = value
    return AgentTraits.model_validate(payload)


def _place(attribute: str, value: float) -> dict[str, float]:
    place = {field: FIXED_ATTRIBUTE for field in PROMPT_FIELDS}
    place[attribute] = value
    return place


def _design_document(bases: list[dict[str, Any]]) -> dict[str, Any]:
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "prompt_version": PROMPT_VERSION,
        "base_agent_ids": [base["base_agent_id"] for base in bases],
        "trait_levels": [_level_label(level) for level in TRAIT_LEVELS],
        "attribute_levels": [_level_label(level) for level in ATTRIBUTE_LEVELS],
        "fixed_attribute": _level_label(FIXED_ATTRIBUTE),
        "constructs": [
            {
                "construct_id": spec["construct_id"],
                "trait": spec["trait"],
                "attribute": spec["attribute"],
                "target_component": spec["target_component"],
                "expected_span_contrast": spec["expected_span_contrast"],
            }
            for spec in CONSTRUCTS
        ],
        "expected_call_count": BASE_AGENT_COUNT * len(TRAIT_LEVELS) * len(ATTRIBUTE_LEVELS) * len(CONSTRUCTS),
    }
    payload["design_sha256"] = canonical_sha256(payload)
    return payload


def _execute_task(
    settings: Settings,
    client: LLMClient,
    task: dict[str, Any],
    configuration: dict[str, Any],
    audit_path: Path,
) -> dict[str, Any]:
    messages = task["messages"]
    raw_response = ""
    parsed: PlaceUtility | None = None
    schema_failures = 0
    error = ""
    for attempt in range(1, settings.decision_max_attempts + 1):
        try:
            result = client.complete(messages=messages, response_model=PlaceUtility)
        except LLMCallError as exc:
            raw_response = exc.raw_text or ""
            schema_failures += 1
            error = str(exc)
            _audit(audit_path, task, configuration, messages, raw_response, False, error, attempt)
            continue
        except ValidationError as exc:
            schema_failures += 1
            error = str(exc)
            _audit(audit_path, task, configuration, messages, raw_response, False, error, attempt)
            continue
        raw_response = result.raw_text
        if isinstance(result.parsed, PlaceUtility):
            parsed = result.parsed
            _audit(audit_path, task, configuration, messages, raw_response, True, "", attempt)
            break
        schema_failures += 1
        error = "parsed object was not a PlaceUtility"
        _audit(audit_path, task, configuration, messages, raw_response, False, error, attempt)
    return _evaluation_record(task, configuration, parsed, raw_response, schema_failures, error)


def _evaluation_record(
    task: dict[str, Any],
    configuration: dict[str, Any],
    parsed: PlaceUtility | None,
    raw_response: str,
    schema_failures: int,
    error: str,
) -> dict[str, Any]:
    scores = {field: getattr(parsed, field) if parsed is not None else None for field in SCORE_FIELDS}
    attempts = schema_failures if parsed is None else schema_failures + 1
    return {
        "construct_id": task["construct_id"],
        "trait_name": task["trait_name"],
        "attribute_name": task["attribute_name"],
        "target_component": task["target_component"],
        "base_agent_id": task["base_agent_id"],
        "trait_level": task["trait_level"],
        "attribute_level": task["attribute_level"],
        "place": task["place"],
        "prompt_sha256": task["prompt_sha256"],
        **scores,
        "model_configuration": configuration,
        "response_hash": hashlib.sha256(raw_response.encode("utf-8")).hexdigest(),
        "schema_valid": parsed is not None,
        "schema_failure_count": schema_failures,
        "retry_count": max(attempts - 1, 0),
        "error": error or None,
    }


def _audit(
    path: Path,
    task: dict[str, Any],
    configuration: dict[str, Any],
    messages: list[dict[str, str]],
    raw_response: str,
    parsed_ok: bool,
    error: str,
    attempt: int,
) -> None:
    append_jsonl(
        path,
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "construct_id": task["construct_id"],
            "base_agent_id": task["base_agent_id"],
            "trait_level": task["trait_level"],
            "attribute_level": task["attribute_level"],
            "attempt": attempt,
            "model_configuration": configuration,
            "request_messages": messages,
            "raw_response": raw_response,
            "parsed_ok": parsed_ok,
            "error": error or None,
        },
    )


def _config_snapshot(settings: Settings, configuration: dict[str, Any], design_sha256: str) -> dict[str, Any]:
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "persona_set": "persona_set_v1",
        "model_configuration": configuration,
        "design_sha256": design_sha256,
        "expected_call_count": EXPECTED_CALLS,
    }
    payload["config_sha256"] = canonical_sha256(payload)
    del settings
    return payload


def _model_configuration(settings: Settings) -> dict[str, Any]:
    return {
        "model": settings.model,
        "temperature": settings.temperature,
        "seed": settings.llm_seed,
        "reasoning_effort": settings.reasoning_effort,
        "api_endpoint": settings.api_endpoint,
        "decision_prompt_version": PROMPT_VERSION,
    }


def _require_gate_settings(settings: Settings) -> None:
    if settings.temperature != 0.0:
        raise RuntimeError("experiment 001F requires temperature 0")
    if settings.reasoning_effort != "none":
        raise RuntimeError("experiment 001F requires reasoning_effort none")
    if settings.llm_seed != 42:
        raise RuntimeError("experiment 001F requires llm seed 42")


def _completed_keys(path: Path) -> set[tuple[str, str, str, str]]:
    return {_task_key(row) for row in _read_jsonl(path)}


def _task_key(task: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(task["construct_id"]),
        str(task["base_agent_id"]),
        _level_label(float(task["trait_level"])),
        _level_label(float(task["attribute_level"])),
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _finite_or_none(value: float) -> float | None:
    number = float(value)
    if not number == number or number in (float("inf"), float("-inf")):
        return None
    return number


def _rate(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return count / total


def _level_label(value: float) -> str:
    return f"{value:.2f}"


def main() -> None:
    run_experiment()


if __name__ == "__main__":
    if __package__ is None:
        raise SystemExit("請在 worldlab 目錄執行：python -m experiments.experiment_001f")
    main()
