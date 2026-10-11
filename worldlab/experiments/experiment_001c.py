"""Experiment 001C: whether place-choice order changes the substantive selection.

Each Experiment 001B scenario is shown twice. The mirrored show swaps Place A
and Place B and leaves every attribute value unchanged. Invariance means both
shows select the same physical option, not the same letter. This module does
not write into Experiment 001 or 001B and does not create a world.
"""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import statsmodels.formula.api as smf
from pydantic import ValidationError
from statsmodels.stats.contingency_tables import mcnemar

from agents.persona_generator import append_jsonl, write_json
from config.settings import PROJECT_ROOT, Settings, get_settings
from experiments.decision_engine import (
    build_decision_messages,
    chosen_lower_crowding,
    traits_for_condition,
)
from experiments.experiment_001 import OUTPUT_DIR as EXPERIMENT_001_DIR
from experiments.experiment_001 import load_base_traits
from experiments.experiment_001b import (
    CROWD_AVERSION_LEVELS,
    OUTPUT_DIR as EXPERIMENT_001B_DIR,
    calibration_trait_definitions,
)
from experiments.reproducibility_check import canonical_sha256
from experiments.tradeoff_scenarios import generate_scenario_matrix, prompt_places
from llm.client import LLMCallError, LLMClient, StructuredLLMClient
from models.schemas import TRAIT_NAMES, PlaceChoiceDecision

EXPERIMENT_ID = "experiment_001c_order_invariance"
OUTPUT_DIR = PROJECT_ROOT / "data" / "experiments" / EXPERIMENT_ID
OPTION_1 = "physical_option_1"
OPTION_2 = "physical_option_2"
PRESENTATIONS = ("original", "mirrored")
EXPECTED_DECISIONS = 20 * len(CROWD_AVERSION_LEVELS) * 24 * len(PRESENTATIONS)
FORMULA_SIDE = (
    "chosen_lower_crowding ~ crowd_aversion * avoidance_cost_level "
    "+ C(presentation_side, Treatment(reference='low_crowd_is_A')) "
    "+ C(tradeoff_block, Treatment(reference='A'))"
)
FORMULA_INTERACTION = (
    "chosen_lower_crowding ~ crowd_aversion * avoidance_cost_level "
    "+ crowd_aversion * C(presentation_side, Treatment(reference='low_crowd_is_A')) "
    "+ C(tradeoff_block, Treatment(reference='A'))"
)
PILOT_NOTE = (
    "Pilot description only. Cluster-robust standard errors are grouped by base_agent. "
    "McNemar and logit p-values are not the quality criterion. "
    "No pass threshold is applied to the invariance rate."
)
SIDE_CONTRAST = "low_crowd_is_B relative to low_crowd_is_A"


def build_pair_map(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = [_pair(scenario) for scenario in scenarios]
    payload = {"source_scenario_count": len(pairs), "pairs": pairs}
    payload["pair_map_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "pair_map_sha256"}
    )
    return payload


def substantive_id(choice: str, presentation: str) -> str:
    if presentation == "original":
        return OPTION_1 if choice == "A" else OPTION_2
    if presentation == "mirrored":
        return OPTION_2 if choice == "A" else OPTION_1
    raise ValueError(f"unknown presentation {presentation}")


def order_invariant(original_choice: str, mirrored_choice: str) -> int:
    original_option = substantive_id(original_choice, "original")
    mirrored_option = substantive_id(mirrored_choice, "mirrored")
    return int(original_option == mirrored_option)


def load_frozen_scenarios(path: Path) -> dict[str, Any]:
    saved = json.loads(path.read_text(encoding="utf-8"))
    fresh = generate_scenario_matrix()
    if saved.get("matrix_sha256") != fresh["matrix_sha256"]:
        raise RuntimeError("frozen scenario matrix does not match the 001B generator")
    return saved


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
    forbidden = {EXPERIMENT_001_DIR.resolve(), EXPERIMENT_001B_DIR.resolve()}
    if destination.resolve() in forbidden:
        raise RuntimeError("experiment 001C must not overwrite experiment 001 or 001B")
    destination.mkdir(parents=True, exist_ok=True)
    source = persona_dir or (active.personas_dir / "persona_set_v1")
    bases = load_base_traits(source, active.calibration_base_agent_count)
    matrix = load_frozen_scenarios(scenario_path or (EXPERIMENT_001B_DIR / "scenarios.json"))
    pair_map = build_pair_map(matrix["scenarios"])
    write_json(destination / "pair_map.json", pair_map)
    configuration = _model_configuration(active)
    config = _config_snapshot(active, configuration, matrix["matrix_sha256"], pair_map["pair_map_sha256"])
    write_json(destination / "config.json", config)
    llm = client if client is not None else StructuredLLMClient(active)
    audit_path = destination / "llm_audit.jsonl"
    decisions_path = destination / "decisions.jsonl"
    completed = _completed_keys(decisions_path)
    tasks = _tasks(bases, pair_map["pairs"])
    definitions = calibration_trait_definitions()
    for index, task in enumerate(tasks, start=1):
        key = (
            task["base_agent_id"],
            _level_label(task["crowd_aversion"]),
            task["scenario_id"],
            task["presentation"],
        )
        if key in completed:
            continue
        print(
            f"{index}/{len(tasks)} {task['base_agent_id']} {_level_label(task['crowd_aversion'])} "
            f"{task['scenario_id']} {task['presentation']}",
            flush=True,
        )
        record = _execute_task(active, llm, task, configuration, definitions, audit_path)
        append_jsonl(decisions_path, record)
    decisions = _read_jsonl(decisions_path)
    summary = summarize_invariance(
        decisions,
        matrix_sha256=matrix["matrix_sha256"],
        pair_map_sha256=pair_map["pair_map_sha256"],
        expected_decision_count=len(tasks),
    )
    summary["model_configuration"] = configuration
    summary["config_sha256"] = config["config_sha256"]
    write_json(destination / "pair_outcomes.json", summary.pop("pairs"))
    write_json(destination / "summary.json", summary)
    regression = fit_presentation_models(decisions)
    write_json(destination / "regression.json", regression)
    print(
        f"invariance={summary['order_invariance_rate']:.3f} mismatches={summary['mismatch_count']}",
        flush=True,
    )
    return destination


def summarize_invariance(
    decisions: list[dict[str, Any]],
    *,
    matrix_sha256: str,
    pair_map_sha256: str,
    expected_decision_count: int | None = None,
) -> dict[str, Any]:
    valid = [row for row in decisions if row.get("schema_valid")]
    pairs = _pair_rows(valid)
    complete_pairs = [pair for pair in pairs if pair["complete"]]
    invariant = sum(int(pair["order_invariant"]) for pair in complete_pairs)
    mismatches = len(complete_pairs) - invariant
    by_side = _dose_response_by_side(valid)
    a_rates = {
        presentation: _choice_a_rate(valid, presentation) for presentation in PRESENTATIONS
    }
    discordance = _discordance(complete_pairs)
    quality = _quality(complete_pairs, invariant, by_side, a_rates)
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
        "pair_count": len(pairs),
        "complete_pair_count": len(complete_pairs),
        "order_invariance_rate": _safe_rate(invariant, len(complete_pairs)),
        "mismatch_count": mismatches,
        "mismatch_rate": _safe_rate(mismatches, len(complete_pairs)),
        "invariance_by_crowd_aversion": _invariance_by(complete_pairs, "crowd_aversion_label"),
        "invariance_by_tradeoff_block": _invariance_by(complete_pairs, "tradeoff_block"),
        "invariance_by_avoidance_cost_level": _invariance_by(complete_pairs, "avoidance_cost_level"),
        "invariance_by_base_agent": _invariance_by(complete_pairs, "base_agent_id"),
        "a_choice_rate": a_rates,
        "low_crowd_choice_by_presentation_side": {
            side: _overall_rate(rows) for side, rows in _rows_by_side(valid).items()
        },
        "dose_response_by_presentation_side": by_side,
        "presentation_effect": _presentation_effect(valid),
        "discordance": discordance,
        "quality": quality,
        "scenario_matrix_sha256": matrix_sha256,
        "pair_map_sha256": pair_map_sha256,
        "pairs": complete_pairs,
    }


def fit_presentation_models(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in decisions if row.get("schema_valid")]
    report: dict[str, Any] = {
        "analysis_level": "pilot",
        "note": PILOT_NOTE,
        "success_criterion": "invariance rate and presentation effect, not p-values",
        "n_observations": len(valid),
        "n_clusters": len({row["base_agent_id"] for row in valid}),
        "presentation_model": _fit_one(valid, FORMULA_SIDE, _presentation_main_name),
        "interaction_model": _fit_one(valid, FORMULA_INTERACTION, _presentation_interaction_name),
    }
    return report


def _pair(scenario: dict[str, Any]) -> dict[str, Any]:
    place_a, place_b = prompt_places(scenario)
    low_side = scenario["lower_crowding_side"]
    low_id = OPTION_1 if low_side == "A" else OPTION_2
    return {
        "scenario_id": scenario["scenario_id"],
        "tradeoff_block": scenario["tradeoff_block"],
        "avoidance_cost_level": scenario["avoidance_cost_level"],
        "physical_option_1": place_a,
        "physical_option_2": place_b,
        "lower_crowding_substantive_id": low_id,
        "original": _view("original", dict(place_a), dict(place_b), OPTION_1, OPTION_2, low_id),
        "mirrored": _view("mirrored", dict(place_b), dict(place_a), OPTION_2, OPTION_1, low_id),
    }


def _view(
    presentation: str,
    place_a: dict[str, float],
    place_b: dict[str, float],
    place_a_id: str,
    place_b_id: str,
    low_id: str,
) -> dict[str, Any]:
    low_on_a = place_a_id == low_id
    return {
        "presentation": presentation,
        "place_a": place_a,
        "place_b": place_b,
        "place_a_substantive_id": place_a_id,
        "place_b_substantive_id": place_b_id,
        "presentation_side": "low_crowd_is_A" if low_on_a else "low_crowd_is_B",
    }


def _tasks(bases: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tasks = []
    for base in bases:
        for level in CROWD_AVERSION_LEVELS:
            traits = traits_for_condition(base["traits"], level)
            for pair in pairs:
                for presentation in PRESENTATIONS:
                    view = pair[presentation]
                    tasks.append(
                        {
                            "base_agent_id": base["base_agent_id"],
                            "crowd_aversion": level,
                            "traits": traits,
                            "scenario_id": pair["scenario_id"],
                            "tradeoff_block": pair["tradeoff_block"],
                            "avoidance_cost_level": pair["avoidance_cost_level"],
                            "lower_crowding_substantive_id": pair["lower_crowding_substantive_id"],
                            "presentation": presentation,
                            "presentation_side": view["presentation_side"],
                            "place_a": view["place_a"],
                            "place_b": view["place_b"],
                            "place_a_substantive_id": view["place_a_substantive_id"],
                            "place_b_substantive_id": view["place_b_substantive_id"],
                        }
                    )
    return tasks


def _execute_task(
    settings: Settings,
    client: LLMClient,
    task: dict[str, Any],
    configuration: dict[str, Any],
    definitions: dict[str, str],
    audit_path: Path,
) -> dict[str, Any]:
    place_a, place_b = prompt_places(task)
    messages = build_decision_messages(
        task["traits"],
        place_a,
        place_b,
        decision_prompt_version=settings.calibration_prompt_version,
        trait_definitions=definitions,
    )
    raw_response = ""
    parsed: PlaceChoiceDecision | None = None
    schema_failures = 0
    error = ""
    for attempt in range(1, settings.decision_max_attempts + 1):
        try:
            result = client.complete(messages=messages, response_model=PlaceChoiceDecision)
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
        if isinstance(result.parsed, PlaceChoiceDecision):
            parsed = result.parsed
            _audit(audit_path, task, configuration, messages, raw_response, True, "", attempt)
            break
        schema_failures += 1
        error = "parsed object was not a PlaceChoiceDecision"
        _audit(audit_path, task, configuration, messages, raw_response, False, error, attempt)
    return _decision_record(task, configuration, parsed, raw_response, schema_failures, error)


def _decision_record(
    task: dict[str, Any],
    configuration: dict[str, Any],
    parsed: PlaceChoiceDecision | None,
    raw_response: str,
    schema_failures: int,
    error: str,
) -> dict[str, Any]:
    place_a, place_b = prompt_places(task)
    choice = parsed.choice if parsed is not None else None
    lower = None
    selected = None
    if choice is not None:
        lower = chosen_lower_crowding(choice, place_a, place_b)
        selected = substantive_id(choice, task["presentation"])
        by_id = int(selected == task["lower_crowding_substantive_id"])
        if lower != by_id:
            raise RuntimeError("substantive option disagrees with crowding")
    attempts = schema_failures if parsed is None else schema_failures + 1
    return {
        "base_agent_id": task["base_agent_id"],
        "crowd_aversion": task["crowd_aversion"],
        "scenario_id": task["scenario_id"],
        "presentation": task["presentation"],
        "presentation_side": task["presentation_side"],
        "tradeoff_block": task["tradeoff_block"],
        "avoidance_cost_level": task["avoidance_cost_level"],
        "place_a": place_a,
        "place_b": place_b,
        "place_a_substantive_id": task["place_a_substantive_id"],
        "place_b_substantive_id": task["place_b_substantive_id"],
        "lower_crowding_substantive_id": task["lower_crowding_substantive_id"],
        "traits": {name: getattr(task["traits"], name) for name in TRAIT_NAMES},
        "choice": choice,
        "substantive_choice": selected,
        "chosen_lower_crowding": lower,
        "reason_codes": list(parsed.reason_codes) if parsed is not None else [],
        "model_configuration": configuration,
        "response_hash": hashlib.sha256(raw_response.encode("utf-8")).hexdigest(),
        "schema_valid": parsed is not None,
        "schema_failure_count": schema_failures,
        "retry_count": max(attempts - 1, 0),
        "error": error or None,
    }


def _pair_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (str(row["base_agent_id"]), _level_label(float(row["crowd_aversion"])), str(row["scenario_id"]))
        grouped[key][str(row["presentation"])] = row
    pairs = []
    for (agent_id, label, scenario_id), shows in sorted(grouped.items()):
        original = shows.get("original")
        mirrored = shows.get("mirrored")
        complete = original is not None and mirrored is not None
        record: dict[str, Any] = {
            "base_agent_id": agent_id,
            "crowd_aversion_label": label,
            "scenario_id": scenario_id,
            "complete": complete,
        }
        if complete:
            original_choice = str(original["choice"])
            mirrored_choice = str(mirrored["choice"])
            original_option = substantive_id(original_choice, "original")
            mirrored_option = substantive_id(mirrored_choice, "mirrored")
            record.update(
                {
                    "tradeoff_block": original["tradeoff_block"],
                    "avoidance_cost_level": int(original["avoidance_cost_level"]),
                    "crowd_aversion": float(original["crowd_aversion"]),
                    "original_choice": original_choice,
                    "mirrored_choice": mirrored_choice,
                    "original_substantive_choice": original_option,
                    "mirrored_substantive_choice": mirrored_option,
                    "order_invariant": int(original_option == mirrored_option),
                }
            )
        pairs.append(record)
    return pairs


def _invariance_by(pairs: list[dict[str, Any]], key: str) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        grouped[str(pair[key])].append(pair)
    report = {}
    for name, items in sorted(grouped.items(), key=lambda item: _sort_key(item[0])):
        hits = sum(int(item["order_invariant"]) for item in items)
        report[name] = {
            "rate": _safe_rate(hits, len(items)),
            "n": len(items),
            "mismatch_count": len(items) - hits,
        }
    return report


def _choice_a_rate(rows: list[dict[str, Any]], presentation: str) -> dict[str, float | int]:
    selected = [row for row in rows if row["presentation"] == presentation]
    if not selected:
        return {"rate": 0.0, "n": 0}
    hits = sum(1 for row in selected if row["choice"] == "A")
    return {"rate": hits / len(selected), "n": len(selected)}


def _rows_by_side(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped = {"low_crowd_is_A": [], "low_crowd_is_B": []}
    for row in rows:
        grouped[str(row["presentation_side"])].append(row)
    return grouped


def _overall_rate(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    if not rows:
        return {"rate": 0.0, "n": 0}
    hits = sum(int(row["chosen_lower_crowding"]) for row in rows)
    return {"rate": hits / len(rows), "n": len(rows)}


def _dose_response_by_side(rows: list[dict[str, Any]]) -> dict[str, Any]:
    report = {}
    for side, selected in _rows_by_side(rows).items():
        levels = {}
        rates = []
        for label in _level_labels():
            level_rows = [row for row in selected if _level_label(float(row["crowd_aversion"])) == label]
            stats = _overall_rate(level_rows)
            levels[label] = stats
            rates.append(float(stats["rate"]))
        report[side] = {
            "rates": levels,
            "non_strict_increasing": _non_strict(rates),
            "strict_increasing": _strict(rates),
        }
    return report


def _presentation_effect(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_option = {}
    for option in (OPTION_1, OPTION_2):
        as_a = [row for row in rows if row["place_a_substantive_id"] == option]
        as_b = [row for row in rows if row["place_b_substantive_id"] == option]
        rate_a = _safe_rate(sum(1 for row in as_a if _selected(row) == option), len(as_a))
        rate_b = _safe_rate(sum(1 for row in as_b if _selected(row) == option), len(as_b))
        per_option[option] = {
            "as_A": rate_a,
            "as_B": rate_b,
            "n": len(as_a),
            "difference_A_minus_B": rate_a - rate_b,
        }
    pooled_a = _safe_rate(sum(1 for row in rows if row["choice"] == "A"), len(rows))
    pooled_b = _safe_rate(sum(1 for row in rows if row["choice"] == "B"), len(rows))
    per_option["pooled"] = {
        "as_A": pooled_a,
        "as_B": pooled_b,
        "n": len(rows),
        "difference_A_minus_B": pooled_a - pooled_b,
    }
    return per_option


def _selected(row: dict[str, Any]) -> str:
    return substantive_id(str(row["choice"]), str(row["presentation"]))


def _discordance(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    both_1 = 0
    orig1_mir2 = 0
    orig2_mir1 = 0
    both_2 = 0
    for pair in pairs:
        original = pair["original_substantive_choice"]
        mirrored = pair["mirrored_substantive_choice"]
        if original == OPTION_1 and mirrored == OPTION_1:
            both_1 += 1
        elif original == OPTION_1 and mirrored == OPTION_2:
            orig1_mir2 += 1
        elif original == OPTION_2 and mirrored == OPTION_1:
            orig2_mir1 += 1
        elif original == OPTION_2 and mirrored == OPTION_2:
            both_2 += 1
    table = [[both_1, orig1_mir2], [orig2_mir1, both_2]]
    auxiliary: dict[str, Any] = {"note": "Descriptive auxiliary. Not the quality criterion."}
    if orig1_mir2 + orig2_mir1 == 0:
        auxiliary["status"] = "no_discordance"
    else:
        try:
            result = mcnemar(table, exact=False, correction=True)
            auxiliary["status"] = "ok"
            auxiliary["statistic"] = float(result.statistic)
            auxiliary["p_value"] = float(result.pvalue)
        except Exception as exc:
            auxiliary["status"] = "failed"
            auxiliary["error"] = f"{type(exc).__name__}: {exc}"
    return {
        "both_option_1": both_1,
        "original_option_1_mirror_option_2": orig1_mir2,
        "original_option_2_mirror_option_1": orig2_mir1,
        "both_option_2": both_2,
        "mcnemar": auxiliary,
    }


def _quality(
    pairs: list[dict[str, Any]],
    invariant: int,
    by_side: dict[str, Any],
    a_rates: dict[str, dict[str, float | int]],
) -> dict[str, Any]:
    total = len(pairs)
    return {
        "pass_threshold": None,
        "invariance_rate": _safe_rate(invariant, total),
        "mismatch_rate": _safe_rate(total - invariant, total),
        "a_choice_rate_original": a_rates["original"]["rate"],
        "a_choice_rate_mirrored": a_rates["mirrored"]["rate"],
        "dose_response_strict_on_both_sides": all(side["strict_increasing"] for side in by_side.values()),
        "dose_response_non_strict_on_both_sides": all(
            side["non_strict_increasing"] for side in by_side.values()
        ),
        "note": "Descriptive only. No pass threshold is applied.",
    }


def _fit_one(rows: list[dict[str, Any]], formula: str, focus) -> dict[str, Any]:
    report: dict[str, Any] = {
        "formula": formula,
        "status": "not_fit",
        "coefficients": {},
        "focus": None,
        "contrast": SIDE_CONTRAST,
    }
    if len(rows) < 30 or len({row["base_agent_id"] for row in rows}) < 2:
        report["status"] = "insufficient_data"
        return report
    if len({int(row["chosen_lower_crowding"]) for row in rows}) < 2:
        report["status"] = "no_outcome_variation"
        return report
    frame = pd.DataFrame(
        {
            "chosen_lower_crowding": [int(row["chosen_lower_crowding"]) for row in rows],
            "crowd_aversion": [float(row["crowd_aversion"]) for row in rows],
            "avoidance_cost_level": [int(row["avoidance_cost_level"]) for row in rows],
            "presentation_side": [str(row["presentation_side"]) for row in rows],
            "tradeoff_block": [str(row["tradeoff_block"]) for row in rows],
            "base_agent_id": [str(row["base_agent_id"]) for row in rows],
        }
    )
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
    confidence = fitted.conf_int()
    for name in fitted.params.index:
        report["coefficients"][str(name)] = _coefficient_row(fitted, confidence, str(name))
    focus_name = focus(fitted.params.index)
    if focus_name is not None:
        report["focus"] = dict(report["coefficients"][focus_name])
        report["focus"]["name"] = focus_name
        report["focus"]["contrast"] = SIDE_CONTRAST
    return report


def _presentation_main_name(names: Any) -> str | None:
    for name in names:
        text = str(name)
        if "presentation_side" in text and ":" not in text:
            return text
    return None


def _presentation_interaction_name(names: Any) -> str | None:
    for name in names:
        text = str(name)
        if "presentation_side" in text and "crowd_aversion" in text and ":" in text and "avoidance_cost" not in text:
            return text
    return None


def _coefficient_row(fitted: Any, confidence: Any, name: str) -> dict[str, Any]:
    coefficient = float(fitted.params[name])
    low, high = (float(value) for value in confidence.loc[name])
    return {
        "coefficient": coefficient,
        "se": float(fitted.bse[name]),
        "p_value": float(fitted.pvalues[name]),
        "ci95": [low, high],
        "odds_ratio": float(math.exp(coefficient)),
    }


def _config_snapshot(
    settings: Settings,
    configuration: dict[str, Any],
    matrix_sha256: str,
    pair_map_sha256: str,
) -> dict[str, Any]:
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "persona_set": "persona_set_v1",
        "source_experiment": "experiment_001b_tradeoff_calibration",
        "model_configuration": configuration,
        "crowd_aversion_levels": list(CROWD_AVERSION_LEVELS),
        "base_agent_count": settings.calibration_base_agent_count,
        "expected_decision_count": EXPECTED_DECISIONS,
        "scenario_matrix_sha256": matrix_sha256,
        "pair_map_sha256": pair_map_sha256,
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
        raise RuntimeError("experiment 001C requires temperature 0")
    if settings.reasoning_effort != "none":
        raise RuntimeError("experiment 001C requires reasoning_effort none")
    if settings.llm_seed != 42:
        raise RuntimeError("experiment 001C requires llm seed 42")
    if settings.calibration_base_agent_count != 20:
        raise RuntimeError("experiment 001C requires 20 base agents")
    if settings.calibration_prompt_version != "place_choice_v2":
        raise RuntimeError("experiment 001C requires place_choice_v2")


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
            "presentation": task["presentation"],
            "attempt": attempt,
            "model_configuration": configuration,
            "request_messages": messages,
            "raw_response": raw_response,
            "parsed_ok": parsed_ok,
            "error": error or None,
        },
    )


def _completed_keys(path: Path) -> set[tuple[str, str, str, str]]:
    keys = set()
    for row in _read_jsonl(path):
        keys.add(
            (
                row["base_agent_id"],
                _level_label(float(row["crowd_aversion"])),
                row["scenario_id"],
                row["presentation"],
            )
        )
    return keys


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _level_labels() -> list[str]:
    return [_level_label(level) for level in CROWD_AVERSION_LEVELS]


def _level_label(value: float) -> str:
    return f"{value:.2f}"


def _safe_rate(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return count / total


def _non_strict(rates: list[float]) -> bool:
    return all(rates[index] <= rates[index + 1] for index in range(len(rates) - 1))


def _strict(rates: list[float]) -> bool:
    return all(rates[index] < rates[index + 1] for index in range(len(rates) - 1))


def _sort_key(value: str) -> tuple[int, float | str]:
    try:
        return (0, float(value))
    except ValueError:
        return (1, value)


def main() -> None:
    run_experiment()


if __name__ == "__main__":
    if __package__ is None:
        raise SystemExit("請在 worldlab 目錄執行：python -m experiments.experiment_001c")
    main()
