"""Experiment 001D: separate position, label, and schema-enum order.

The place-choice prompt keeps the place_choice_v2 instructions and trait
definitions. The only changes are which physical option is written first,
which letter it carries, and whether the response schema lists A or B first.
This module does not rewrite Experiment 001, 001B, or 001C, and it does not
try to remove the bias.
"""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Literal

import pandas as pd
import statsmodels.formula.api as smf
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from agents.persona_generator import append_jsonl, write_json
from config.settings import PROJECT_ROOT, Settings, get_settings
from experiments.decision_engine import PLACE_GUIDE, traits_for_condition
from experiments.experiment_001 import OUTPUT_DIR as EXPERIMENT_001_DIR
from experiments.experiment_001 import load_base_traits
from experiments.experiment_001b import OUTPUT_DIR as EXPERIMENT_001B_DIR
from experiments.experiment_001b import calibration_trait_definitions
from experiments.experiment_001c import OUTPUT_DIR as EXPERIMENT_001C_DIR
from experiments.experiment_001c import load_frozen_scenarios
from experiments.reproducibility_check import canonical_sha256
from experiments.tradeoff_scenarios import prompt_places
from llm.client import LLMCallError, LLMClient, StructuredLLMClient
from models.schemas import TRAIT_NAMES, PlaceChoiceDecision, export_response_schema

EXPERIMENT_ID = "experiment_001d_label_position_schema"
OUTPUT_DIR = PROJECT_ROOT / "data" / "experiments" / EXPERIMENT_ID
OPTION_1 = "physical_option_1"
OPTION_2 = "physical_option_2"
BASE_AGENT_COUNT = 10
CROWD_AVERSION_LEVELS = (0.0, 0.5, 1.0)
COST_LEVELS = (1, 4, 6)
BLOCKS = ("A", "B", "C", "D")
POSITIONS = ("first", "second")
LABELS = ("A", "B")
ENUM_ORDERS = (("A", "B"), ("B", "A"))
EXPECTED_DECISIONS = BASE_AGENT_COUNT * len(CROWD_AVERSION_LEVELS) * 12 * 8
FORMULA_MAIN = (
    "chosen_physical_option_1 ~ physical_option_1_is_first + physical_option_1_is_A "
    "+ schema_A_first + crowd_aversion + avoidance_cost_level "
    "+ C(tradeoff_block, Treatment(reference='A'))"
)
FORMULA_INTERACTION = (
    "chosen_physical_option_1 ~ physical_option_1_is_first + physical_option_1_is_A "
    "+ schema_A_first + physical_option_1_is_first:physical_option_1_is_A "
    "+ physical_option_1_is_first:schema_A_first + physical_option_1_is_A:schema_A_first "
    "+ crowd_aversion + avoidance_cost_level "
    "+ C(tradeoff_block, Treatment(reference='A'))"
)
PILOT_NOTE = (
    "Pilot description only. Cluster-robust standard errors are grouped by base_agent. "
    "p-values are not the success criterion. The prompt is not rewritten to remove bias."
)
TECHNICAL_COLUMNS = (
    "physical_option_1_is_first",
    "physical_option_1_is_A",
    "schema_A_first",
)
INTERACTION_COLUMNS = (
    "physical_option_1_is_first:physical_option_1_is_A",
    "physical_option_1_is_first:schema_A_first",
    "physical_option_1_is_A:schema_A_first",
)
_BANNED_PROMPT_TEXT = (
    "ignore ordering",
    "avoid bias",
    "carefully compare",
    "do not prefer",
    "don't prefer",
    "first option",
    "position bias",
)


class PlaceChoiceDecisionEnumBA(BaseModel):
    """Same decision schema with the choice enum listed as B then A."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    response_schema_name: ClassVar[str] = "place_choice"

    choice: Literal["B", "A"]
    reason_codes: list[
        Literal["crowding", "visual_attractiveness", "distance", "novelty", "other"]
    ] = Field(min_length=1, max_length=5)

    @field_validator("reason_codes")
    @classmethod
    def _unique_codes(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("reason_codes must not contain duplicates")
        return values


def schema_for_enum(enum_order: tuple[str, str]) -> type[BaseModel]:
    if enum_order == ("A", "B"):
        return PlaceChoiceDecision
    if enum_order == ("B", "A"):
        return PlaceChoiceDecisionEnumBA
    raise ValueError(f"unknown enum order {enum_order}")


def select_scenarios(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    chosen = [
        scenario
        for scenario in matrix["scenarios"]
        if scenario["tradeoff_block"] in BLOCKS and int(scenario["avoidance_cost_level"]) in COST_LEVELS
    ]
    chosen.sort(key=lambda scenario: scenario["scenario_id"])
    if len(chosen) != 12:
        raise RuntimeError("expected 12 frozen scenarios")
    covered = {(scenario["tradeoff_block"], int(scenario["avoidance_cost_level"])) for scenario in chosen}
    expected = {(block, cost) for block in BLOCKS for cost in COST_LEVELS}
    if covered != expected:
        raise RuntimeError("scenario sample does not cover every block and cost")
    return chosen


def build_design(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = [_substantive_pair(scenario) for scenario in scenarios]
    cells = [
        {
            "physical_option_1_position": position,
            "physical_option_1_label": label,
            "schema_enum_order": list(enum_order),
        }
        for position in POSITIONS
        for label in LABELS
        for enum_order in ENUM_ORDERS
    ]
    payload = {
        "source": "experiment_001b_tradeoff_calibration",
        "cost_levels": {"low": COST_LEVELS[0], "medium": COST_LEVELS[1], "high": COST_LEVELS[2]},
        "pairs": pairs,
        "factorial_cells": cells,
        "schema_choice_enum": {
            "AB": export_response_schema(PlaceChoiceDecision)["properties"]["choice"]["enum"],
            "BA": export_response_schema(PlaceChoiceDecisionEnumBA)["properties"]["choice"]["enum"],
        },
    }
    payload["design_sha256"] = canonical_sha256(payload)
    return payload


def build_factorial_messages(
    traits: Any,
    pair: dict[str, Any],
    *,
    position: str,
    label: str,
    decision_prompt_version: str,
) -> list[dict[str, str]]:
    definitions = calibration_trait_definitions()
    trait_payload = {name: getattr(traits, name) for name in TRAIT_NAMES}
    rendered = "\n".join(f"- {name}: {definitions[name]}" for name in TRAIT_NAMES)
    slots = _text_slots(pair, position, label)
    option_text = "\n".join(_option_block(slot_label, attributes) for slot_label, attributes in slots)
    system = (
        "You choose one place for a simulated agent.\n"
        "Use only the numerical traits and the two place attribute lists.\n"
        "reason_codes may contain only crowding, visual_attractiveness, distance, novelty, or other.\n"
        "Return JSON only, matching the schema. No markdown.\n"
    )
    user = (
        f"prompt_version: {decision_prompt_version}\n\n"
        "Trait definitions:\n"
        f"{rendered}\n\n"
        "TRAITS_JSON\n"
        f"{_dump(trait_payload)}\n"
        "END_TRAITS_JSON\n\n"
        "Place attribute definitions:\n"
        f"{PLACE_GUIDE}\n\n"
        f"{option_text}\n"
        "Choose A or B.\n"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    _require_prompt_is_unrepaired(messages)
    return messages


def substantive_choice(raw_choice: str, option_1_label: str) -> str:
    if raw_choice == option_1_label:
        return OPTION_1
    if raw_choice in {"A", "B"}:
        return OPTION_2
    raise ValueError(f"unknown choice {raw_choice}")


def run_experiment(
    settings: Settings | None = None,
    client: LLMClient | None = None,
    *,
    persona_dir: Path | None = None,
    output_dir: Path | None = None,
    scenario_path: Path | None = None,
) -> Path:
    active = get_settings() if settings is None else settings
    _require_gate_settings(active)
    destination = output_dir or OUTPUT_DIR
    forbidden = {
        EXPERIMENT_001_DIR.resolve(),
        EXPERIMENT_001B_DIR.resolve(),
        EXPERIMENT_001C_DIR.resolve(),
    }
    if destination.resolve() in forbidden:
        raise RuntimeError("experiment 001D must not overwrite earlier experiments")
    destination.mkdir(parents=True, exist_ok=True)
    source = persona_dir or (active.personas_dir / "persona_set_v1")
    bases = load_base_traits(source, BASE_AGENT_COUNT)
    matrix = load_frozen_scenarios(scenario_path or (EXPERIMENT_001B_DIR / "scenarios.json"))
    design = build_design(select_scenarios(matrix))
    write_json(destination / "design.json", design)
    configuration = _model_configuration(active)
    config = _config_snapshot(active, configuration, matrix["matrix_sha256"], design["design_sha256"])
    write_json(destination / "config.json", config)
    llm = client if client is not None else StructuredLLMClient(active)
    audit_path = destination / "llm_audit.jsonl"
    decisions_path = destination / "decisions.jsonl"
    completed = _completed_keys(decisions_path)
    tasks = _tasks(bases, design["pairs"])
    for index, task in enumerate(tasks, start=1):
        key = _task_key(task)
        if key in completed:
            continue
        print(
            f"{index}/{len(tasks)} {task['base_agent_id']} {_level_label(task['crowd_aversion'])} "
            f"{task['scenario_id']} {task['physical_option_1_position']} "
            f"{task['physical_option_1_label']} {''.join(task['schema_enum_order'])}",
            flush=True,
        )
        record = _execute_task(active, llm, task, configuration, audit_path)
        append_jsonl(decisions_path, record)
    decisions = _read_jsonl(decisions_path)
    summary = summarize_decomposition(
        decisions,
        matrix_sha256=matrix["matrix_sha256"],
        design_sha256=design["design_sha256"],
        expected_decision_count=len(tasks),
    )
    summary["model_configuration"] = configuration
    summary["config_sha256"] = config["config_sha256"]
    write_json(destination / "summary.json", summary)
    regression = fit_decomposition_models(decisions)
    write_json(destination / "regression.json", regression)
    print(
        "effects "
        f"first={summary['first_position_advantage']['difference']:.3f} "
        f"label={summary['a_label_advantage']['difference']:.3f} "
        f"enum={summary['enum_order_advantage']['difference']:.3f}",
        flush=True,
    )
    return destination


def summarize_decomposition(
    decisions: list[dict[str, Any]],
    *,
    matrix_sha256: str,
    design_sha256: str,
    expected_decision_count: int | None = None,
) -> dict[str, Any]:
    valid = [row for row in decisions if row.get("schema_valid")]
    invariance = _invariance(valid)
    dose = _dose_response(valid)
    return {
        "experiment_id": EXPERIMENT_ID,
        "analysis_level": "pilot",
        "pass_threshold": None,
        "decision_count": len(decisions),
        "valid_decision_count": len(valid),
        "expected_decision_count": expected_decision_count,
        "complete": expected_decision_count is not None and len(decisions) == expected_decision_count,
        "schema_failure_count": sum(int(row.get("schema_failure_count", 0)) for row in decisions),
        "invalid_response_count": sum(1 for row in decisions if not row.get("schema_valid")),
        "retry_count": sum(int(row.get("retry_count", 0)) for row in decisions),
        "first_position_advantage": _position_advantage(valid),
        "a_label_advantage": _label_advantage(valid),
        "enum_order_advantage": _enum_advantage(valid),
        "substantive_invariance": invariance,
        "dose_response": dose,
        "scenario_matrix_sha256": matrix_sha256,
        "design_sha256": design_sha256,
    }


def fit_decomposition_models(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in decisions if row.get("schema_valid")]
    frame = _model_frame(valid)
    main = _fit(frame, FORMULA_MAIN)
    interaction = _fit(frame, FORMULA_INTERACTION)
    if main["status"] == "ok":
        main["technical_effects"] = {
            column: _predicted_binary_effect(main["_fitted"], frame, column) for column in TECHNICAL_COLUMNS
        }
        main["crowd_aversion_predictions"] = _predicted_at_aversion(main["_fitted"], frame)
        for column in TECHNICAL_COLUMNS:
            if column in main["coefficients"]:
                main["technical_effects"][column]["coefficient"] = main["coefficients"][column]
        main["technical_effects"]["crowd_aversion"] = main["coefficients"].get("crowd_aversion")
    if interaction["status"] == "ok":
        interaction["interaction_effects"] = {
            column: interaction["coefficients"][column]
            for column in INTERACTION_COLUMNS
            if column in interaction["coefficients"]
        }
        interaction["interaction_probability_contrasts"] = _interaction_contrasts(interaction["_fitted"], frame)
    main.pop("_fitted", None)
    interaction.pop("_fitted", None)
    return {
        "analysis_level": "pilot",
        "note": PILOT_NOTE,
        "success_criterion": "effect size and predicted probability, not p-values",
        "n_observations": len(valid),
        "n_clusters": len({row["base_agent_id"] for row in valid}),
        "main_model": main,
        "interaction_model": interaction,
    }


def _substantive_pair(scenario: dict[str, Any]) -> dict[str, Any]:
    place_a, place_b = prompt_places(scenario)
    low_id = OPTION_1 if scenario["lower_crowding_side"] == "A" else OPTION_2
    return {
        "scenario_id": scenario["scenario_id"],
        "tradeoff_block": scenario["tradeoff_block"],
        "avoidance_cost_level": int(scenario["avoidance_cost_level"]),
        "physical_option_1": place_a,
        "physical_option_2": place_b,
        "lower_crowding_substantive_id": low_id,
    }


def _text_slots(pair: dict[str, Any], position: str, label: str) -> list[tuple[str, dict[str, float]]]:
    other = "B" if label == "A" else "A"
    option_1 = (label, dict(pair["physical_option_1"]))
    option_2 = (other, dict(pair["physical_option_2"]))
    if position == "first":
        return [option_1, option_2]
    if position == "second":
        return [option_2, option_1]
    raise ValueError(f"unknown position {position}")


def _option_block(label: str, attributes: dict[str, float]) -> str:
    return f"OPTION {label}\n{_dump(attributes)}\nEND_OPTION_{label}\n"


def _tasks(bases: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tasks = []
    for base in bases:
        for level in CROWD_AVERSION_LEVELS:
            traits = traits_for_condition(base["traits"], level)
            for pair in pairs:
                for position in POSITIONS:
                    for label in LABELS:
                        for enum_order in ENUM_ORDERS:
                            tasks.append(
                                {
                                    "base_agent_id": base["base_agent_id"],
                                    "crowd_aversion": level,
                                    "traits": traits,
                                    "scenario_id": pair["scenario_id"],
                                    "tradeoff_block": pair["tradeoff_block"],
                                    "avoidance_cost_level": pair["avoidance_cost_level"],
                                    "physical_option_1": pair["physical_option_1"],
                                    "physical_option_2": pair["physical_option_2"],
                                    "lower_crowding_substantive_id": pair["lower_crowding_substantive_id"],
                                    "physical_option_1_position": position,
                                    "physical_option_1_label": label,
                                    "schema_enum_order": list(enum_order),
                                }
                            )
    return tasks


def _execute_task(
    settings: Settings,
    client: LLMClient,
    task: dict[str, Any],
    configuration: dict[str, Any],
    audit_path: Path,
) -> dict[str, Any]:
    messages = build_factorial_messages(
        task["traits"],
        task,
        position=task["physical_option_1_position"],
        label=task["physical_option_1_label"],
        decision_prompt_version=settings.calibration_prompt_version,
    )
    response_model = schema_for_enum(tuple(task["schema_enum_order"]))
    raw_response = ""
    parsed: BaseModel | None = None
    schema_failures = 0
    error = ""
    for attempt in range(1, settings.decision_max_attempts + 1):
        try:
            result = client.complete(messages=messages, response_model=response_model)
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
        if isinstance(result.parsed, response_model):
            parsed = result.parsed
            _audit(audit_path, task, configuration, messages, raw_response, True, "", attempt)
            break
        schema_failures += 1
        error = "parsed object was not the requested choice schema"
        _audit(audit_path, task, configuration, messages, raw_response, False, error, attempt)
    return _decision_record(task, configuration, parsed, raw_response, schema_failures, error)


def _decision_record(
    task: dict[str, Any],
    configuration: dict[str, Any],
    parsed: BaseModel | None,
    raw_response: str,
    schema_failures: int,
    error: str,
) -> dict[str, Any]:
    raw_choice = getattr(parsed, "choice", None) if parsed is not None else None
    selected = None
    chosen_option_1 = None
    chosen_low = None
    if raw_choice is not None:
        selected = substantive_choice(raw_choice, task["physical_option_1_label"])
        chosen_option_1 = int(selected == OPTION_1)
        chosen_attrs = task["physical_option_1"] if selected == OPTION_1 else task["physical_option_2"]
        other_attrs = task["physical_option_2"] if selected == OPTION_1 else task["physical_option_1"]
        if chosen_attrs["crowding"] == other_attrs["crowding"]:
            raise RuntimeError("substantive options have equal crowding")
        chosen_low = int(chosen_attrs["crowding"] < other_attrs["crowding"])
        if chosen_low != int(selected == task["lower_crowding_substantive_id"]):
            raise RuntimeError("low-crowd id disagrees with attributes")
    attempts = schema_failures if parsed is None else schema_failures + 1
    return {
        "base_agent_id": task["base_agent_id"],
        "crowd_aversion": task["crowd_aversion"],
        "scenario_id": task["scenario_id"],
        "tradeoff_block": task["tradeoff_block"],
        "avoidance_cost_level": task["avoidance_cost_level"],
        "physical_option_1": task["physical_option_1"],
        "physical_option_2": task["physical_option_2"],
        "lower_crowding_substantive_id": task["lower_crowding_substantive_id"],
        "physical_option_1_position": task["physical_option_1_position"],
        "physical_option_1_label": task["physical_option_1_label"],
        "schema_enum_order": list(task["schema_enum_order"]),
        "physical_option_1_is_first": int(task["physical_option_1_position"] == "first"),
        "physical_option_1_is_A": int(task["physical_option_1_label"] == "A"),
        "schema_A_first": int(list(task["schema_enum_order"]) == ["A", "B"]),
        "raw_choice": raw_choice,
        "substantive_choice": selected,
        "chosen_physical_option_1": chosen_option_1,
        "chosen_lower_crowding": chosen_low,
        "reason_codes": list(parsed.reason_codes) if parsed is not None else [],
        "model_configuration": configuration,
        "response_hash": hashlib.sha256(raw_response.encode("utf-8")).hexdigest(),
        "schema_valid": parsed is not None,
        "schema_failure_count": schema_failures,
        "retry_count": max(attempts - 1, 0),
        "error": error or None,
    }


def _position_advantage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return _factor_advantage(rows, _first_option_id, "as_first", "as_second")


def _label_advantage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return _factor_advantage(rows, _label_a_option_id, "as_A", "as_B")


def _factor_advantage(rows: list[dict[str, Any]], locate, high_name: str, low_name: str) -> dict[str, Any]:
    per_option = {}
    for option in (OPTION_1, OPTION_2):
        high = [row for row in rows if locate(row) == option]
        low = [row for row in rows if locate(row) != option]
        rate_high = _mean_selected(high, option)
        rate_low = _mean_selected(low, option)
        per_option[option] = {high_name: rate_high, low_name: rate_low, "difference": rate_high - rate_low}
    pooled_high = _safe_rate(sum(1 for row in rows if row["substantive_choice"] == locate(row)), len(rows))
    pooled_low = _safe_rate(sum(1 for row in rows if row["substantive_choice"] != locate(row)), len(rows))
    per_option["pooled"] = {
        high_name: pooled_high,
        low_name: pooled_low,
        "difference": pooled_high - pooled_low,
        "n": len(rows),
    }
    per_option["difference"] = pooled_high - pooled_low
    return per_option


def _enum_advantage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped = {"AB": [], "BA": []}
    for row in rows:
        key = "AB" if row["schema_A_first"] else "BA"
        grouped[key].append(row)
    rate_ab = _safe_rate(sum(1 for row in grouped["AB"] if row["raw_choice"] == "A"), len(grouped["AB"]))
    rate_ba = _safe_rate(sum(1 for row in grouped["BA"] if row["raw_choice"] == "A"), len(grouped["BA"]))
    return {
        "enum_AB": {"rate": rate_ab, "n": len(grouped["AB"])},
        "enum_BA": {"rate": rate_ba, "n": len(grouped["BA"])},
        "difference": rate_ab - rate_ba,
    }


def _invariance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (
            str(row["base_agent_id"]),
            _level_label(float(row["crowd_aversion"])),
            str(row["scenario_id"]),
            str(row["physical_option_1_label"]),
            "AB" if row["schema_A_first"] else "BA",
        )
        grouped[key][str(row["physical_option_1_position"])] = row
    complete = []
    for key, shows in grouped.items():
        if "first" in shows and "second" in shows:
            same = shows["first"]["substantive_choice"] == shows["second"]["substantive_choice"]
            complete.append({"label": key[3], "schema": key[4], "order_invariant": int(same)})
    hits = sum(item["order_invariant"] for item in complete)
    return {
        "definition": "same substantive option when position flips and label plus schema enum stay fixed",
        "n": len(complete),
        "invariant": hits,
        "mismatch_count": len(complete) - hits,
        "rate": _safe_rate(hits, len(complete)),
        "by_label": _rate_by(complete, "label"),
        "by_schema": _rate_by(complete, "schema"),
    }


def _dose_response(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_position: dict[str, list[dict[str, Any]]] = {"first": [], "second": []}
    by_label: dict[str, list[dict[str, Any]]] = {"A": [], "B": []}
    for row in rows:
        low_id = row["lower_crowding_substantive_id"]
        by_position[_option_position(row, low_id)].append(row)
        by_label[_option_label(row, low_id)].append(row)
    return {
        "by_low_crowd_position": {name: _aversion_curve(items) for name, items in by_position.items()},
        "by_low_crowd_label": {name: _aversion_curve(items) for name, items in by_label.items()},
    }


def _aversion_curve(rows: list[dict[str, Any]]) -> dict[str, Any]:
    levels = {}
    rates = []
    for level in CROWD_AVERSION_LEVELS:
        selected = [row for row in rows if _level_label(float(row["crowd_aversion"])) == _level_label(level)]
        stats = _mean_flag(selected, "chosen_lower_crowding")
        levels[_level_label(level)] = stats
        rates.append(float(stats["rate"]))
    return {
        "rates": levels,
        "non_strict_increasing": all(rates[index] <= rates[index + 1] for index in range(len(rates) - 1)),
        "strict_increasing": all(rates[index] < rates[index + 1] for index in range(len(rates) - 1)),
    }


def _model_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "chosen_physical_option_1": [int(row["chosen_physical_option_1"]) for row in rows],
            "physical_option_1_is_first": [int(row["physical_option_1_is_first"]) for row in rows],
            "physical_option_1_is_A": [int(row["physical_option_1_is_A"]) for row in rows],
            "schema_A_first": [int(row["schema_A_first"]) for row in rows],
            "crowd_aversion": [float(row["crowd_aversion"]) for row in rows],
            "avoidance_cost_level": [int(row["avoidance_cost_level"]) for row in rows],
            "tradeoff_block": [str(row["tradeoff_block"]) for row in rows],
            "base_agent_id": [str(row["base_agent_id"]) for row in rows],
        }
    )


def _fit(frame: pd.DataFrame, formula: str) -> dict[str, Any]:
    report: dict[str, Any] = {"formula": formula, "status": "not_fit", "coefficients": {}}
    if frame.empty or len(frame) < 30 or frame["base_agent_id"].nunique() < 2:
        report["status"] = "insufficient_data"
        return report
    if frame["chosen_physical_option_1"].nunique() < 2:
        report["status"] = "no_outcome_variation"
        return report
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = smf.logit(formula, data=frame).fit(
                disp=False,
                cov_type="cluster",
                cov_kwds={"groups": frame["base_agent_id"]},
            )
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        return report
    report["status"] = "ok"
    report["_fitted"] = fitted
    confidence = fitted.conf_int()
    for name in fitted.params.index:
        report["coefficients"][str(name)] = _coefficient_row(fitted, confidence, str(name))
    return report


def _predicted_binary_effect(fitted: Any, frame: pd.DataFrame, column: str) -> dict[str, float]:
    high = frame.copy()
    low = frame.copy()
    high[column] = 1
    low[column] = 0
    high_probability = float(fitted.predict(high).mean())
    low_probability = float(fitted.predict(low).mean())
    return {
        "predicted_probability_when_1": high_probability,
        "predicted_probability_when_0": low_probability,
        "predicted_difference": high_probability - low_probability,
    }


def _predicted_at_aversion(fitted: Any, frame: pd.DataFrame) -> dict[str, float]:
    predictions = {}
    for level in CROWD_AVERSION_LEVELS:
        grid = frame.copy()
        grid["crowd_aversion"] = level
        predictions[_level_label(level)] = float(fitted.predict(grid).mean())
    return predictions


def _interaction_contrasts(fitted: Any, frame: pd.DataFrame) -> dict[str, float]:
    contrasts = {}
    for left, right in (
        ("physical_option_1_is_first", "physical_option_1_is_A"),
        ("physical_option_1_is_first", "schema_A_first"),
        ("physical_option_1_is_A", "schema_A_first"),
    ):
        cells = {}
        for left_value in (0, 1):
            for right_value in (0, 1):
                grid = frame.copy()
                grid[left] = left_value
                grid[right] = right_value
                cells[(left_value, right_value)] = float(fitted.predict(grid).mean())
        contrasts[f"{left}:{right}"] = cells[(1, 1)] - cells[(1, 0)] - cells[(0, 1)] + cells[(0, 0)]
    return contrasts


def _coefficient_row(fitted: Any, confidence: Any, name: str) -> dict[str, Any]:
    coefficient = float(fitted.params[name])
    se = float(fitted.bse[name])
    p_value = float(fitted.pvalues[name])
    low, high = (float(value) for value in confidence.loc[name])
    row = {
        "coefficient": _finite_or_none(coefficient),
        "se": _finite_or_none(se),
        "p_value": _finite_or_none(p_value),
        "ci95": [_finite_or_none(low), _finite_or_none(high)],
        "odds_ratio": _finite_or_none(math.exp(coefficient)) if math.isfinite(coefficient) else None,
    }
    if row["se"] is None:
        row["se_note"] = "Cluster-robust SE is undefined because this factor did not change the fitted likelihood."
    return row


def _finite_or_none(value: float) -> float | None:
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _require_prompt_is_unrepaired(messages: list[dict[str, str]]) -> None:
    text = "\n".join(message["content"] for message in messages).lower()
    for banned in _BANNED_PROMPT_TEXT:
        if banned in text:
            raise RuntimeError(f"prompt contains a repair instruction: {banned}")


def _config_snapshot(
    settings: Settings,
    configuration: dict[str, Any],
    matrix_sha256: str,
    design_sha256: str,
) -> dict[str, Any]:
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "persona_set": "persona_set_v1",
        "source_experiment": "experiment_001b_tradeoff_calibration",
        "model_configuration": configuration,
        "crowd_aversion_levels": list(CROWD_AVERSION_LEVELS),
        "base_agent_count": BASE_AGENT_COUNT,
        "expected_decision_count": EXPECTED_DECISIONS,
        "scenario_matrix_sha256": matrix_sha256,
        "design_sha256": design_sha256,
    }
    payload["config_sha256"] = canonical_sha256(payload)
    return payload


def _model_configuration(settings: Settings) -> dict[str, Any]:
    return {
        "model": settings.model,
        "temperature": settings.temperature,
        "seed": settings.llm_seed,
        "reasoning_effort": settings.reasoning_effort,
        "api_endpoint": settings.api_endpoint,
        "decision_prompt_version": settings.calibration_prompt_version,
    }


def _require_gate_settings(settings: Settings) -> None:
    if settings.temperature != 0.0:
        raise RuntimeError("experiment 001D requires temperature 0")
    if settings.reasoning_effort != "none":
        raise RuntimeError("experiment 001D requires reasoning_effort none")
    if settings.llm_seed != 42:
        raise RuntimeError("experiment 001D requires llm seed 42")
    if settings.calibration_prompt_version != "place_choice_v2":
        raise RuntimeError("experiment 001D requires place_choice_v2 trait prompt")


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
            "base_agent_id": task["base_agent_id"],
            "crowd_aversion": task["crowd_aversion"],
            "scenario_id": task["scenario_id"],
            "physical_option_1_position": task["physical_option_1_position"],
            "physical_option_1_label": task["physical_option_1_label"],
            "schema_enum_order": list(task["schema_enum_order"]),
            "attempt": attempt,
            "model_configuration": configuration,
            "request_messages": messages,
            "raw_response": raw_response,
            "parsed_ok": parsed_ok,
            "error": error or None,
        },
    )


def _completed_keys(path: Path) -> set[tuple[str, str, str, str, str, str]]:
    return {_task_key(row) for row in _read_jsonl(path)}


def _task_key(task: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        str(task["base_agent_id"]),
        _level_label(float(task["crowd_aversion"])),
        str(task["scenario_id"]),
        str(task["physical_option_1_position"]),
        str(task["physical_option_1_label"]),
        "".join(task["schema_enum_order"]),
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _first_option_id(row: dict[str, Any]) -> str:
    return OPTION_1 if row["physical_option_1_position"] == "first" else OPTION_2


def _label_a_option_id(row: dict[str, Any]) -> str:
    return OPTION_1 if row["physical_option_1_label"] == "A" else OPTION_2


def _option_position(row: dict[str, Any], option_id: str) -> str:
    if option_id == OPTION_1:
        return str(row["physical_option_1_position"])
    return "second" if row["physical_option_1_position"] == "first" else "first"


def _option_label(row: dict[str, Any], option_id: str) -> str:
    if option_id == OPTION_1:
        return str(row["physical_option_1_label"])
    return "B" if row["physical_option_1_label"] == "A" else "A"


def _mean_selected(rows: list[dict[str, Any]], option: str) -> float:
    return _safe_rate(sum(1 for row in rows if row["substantive_choice"] == option), len(rows))


def _mean_flag(rows: list[dict[str, Any]], key: str) -> dict[str, float | int]:
    if not rows:
        return {"rate": 0.0, "n": 0}
    hits = sum(int(row[key]) for row in rows)
    return {"rate": hits / len(rows), "n": len(rows)}


def _rate_by(items: list[dict[str, Any]], key: str) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[str(item[key])].append(item)
    report = {}
    for name, rows in sorted(grouped.items()):
        hits = sum(int(row["order_invariant"]) for row in rows)
        report[name] = {"rate": _safe_rate(hits, len(rows)), "n": len(rows), "mismatch_count": len(rows) - hits}
    return report


def _level_label(value: float) -> str:
    return f"{value:.2f}"


def _safe_rate(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return count / total


def _dump(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)


def main() -> None:
    run_experiment()


if __name__ == "__main__":
    if __package__ is None:
        raise SystemExit("請在 worldlab 目錄執行：python -m experiments.experiment_001d")
    main()
