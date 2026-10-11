"""Experiment 001B: crowd-avoidance trade-off calibration.

Matched agents differ only in crowd_aversion. Scenarios are a fixed cost
ladder. The decision engine receives numerical traits only. This module does
not create a world, memory, social interaction, or UI, and it does not write
into the Experiment 001 directory.
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

from agents.persona_generator import TRAIT_DEFINITIONS, append_jsonl, write_json
from config.settings import PROJECT_ROOT, Settings, get_settings
from experiments.decision_engine import (
    build_decision_messages,
    chosen_lower_crowding,
    traits_for_condition,
)
from experiments.experiment_001 import OUTPUT_DIR as EXPERIMENT_001_DIR
from experiments.experiment_001 import load_base_traits
from experiments.reproducibility_check import canonical_sha256
from experiments.tradeoff_scenarios import generate_scenario_matrix, prompt_places
from llm.client import LLMCallError, LLMClient, StructuredLLMClient
from models.schemas import TRAIT_NAMES, PlaceChoiceDecision

EXPERIMENT_ID = "experiment_001b_tradeoff_calibration"
OUTPUT_DIR = PROJECT_ROOT / "data" / "experiments" / EXPERIMENT_ID
CROWD_AVERSION_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
EXPECTED_DECISIONS = 20 * len(CROWD_AVERSION_LEVELS) * 24
FORMULA = (
    "chosen_lower_crowding ~ crowd_aversion * avoidance_cost_level "
    "+ C(tradeoff_block, Treatment(reference='A'))"
)
PILOT_NOTE = (
    "Pilot only. Cluster-robust standard errors are grouped by base_agent. "
    "p-values are not the success criterion. "
    "Read dose-response, switch thresholds, and predicted probabilities."
)
CROWD_AVERSION_DEFINITION = (
    "0 means crowding has almost no negative effect on the choice; "
    "1 means an extremely strong tendency to avoid crowded places"
)
CEILING_RATE = 0.95
FLOOR_RATE = 0.05


def calibration_trait_definitions() -> dict[str, str]:
    definitions = dict(TRAIT_DEFINITIONS)
    definitions["crowd_aversion"] = CROWD_AVERSION_DEFINITION
    return definitions


def run_experiment(
    settings: Settings | None = None,
    client: LLMClient | None = None,
    *,
    persona_dir: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    active = get_settings() if settings is None else settings
    _require_calibration_settings(active)
    destination = output_dir or OUTPUT_DIR
    if destination.resolve() == EXPERIMENT_001_DIR.resolve():
        raise RuntimeError("experiment 001B must not overwrite experiment 001")
    destination.mkdir(parents=True, exist_ok=True)
    source = persona_dir or (active.personas_dir / "persona_set_v1")
    bases = load_base_traits(source, active.calibration_base_agent_count)
    matrix = generate_scenario_matrix()
    write_json(destination / "scenarios.json", matrix)
    configuration = _model_configuration(active)
    config = _config_snapshot(active, configuration, matrix["matrix_sha256"])
    write_json(destination / "config.json", config)
    llm = client if client is not None else StructuredLLMClient(active)
    audit_path = destination / "llm_audit.jsonl"
    decisions_path = destination / "decisions.jsonl"
    completed = _completed_keys(decisions_path)
    tasks = _tasks(bases, matrix["scenarios"])
    definitions = calibration_trait_definitions()
    for index, task in enumerate(tasks, start=1):
        key = (task["base_agent_id"], _level_label(task["crowd_aversion"]), task["scenario_id"])
        if key in completed:
            continue
        print(
            f"{index}/{len(tasks)} {task['base_agent_id']} {_level_label(task['crowd_aversion'])} {task['scenario_id']}",
            flush=True,
        )
        record = _execute_task(active, llm, task, configuration, definitions, audit_path)
        append_jsonl(decisions_path, record)
    decisions = _read_jsonl(decisions_path)
    summary = summarize_calibration(
        decisions,
        matrix_sha256=matrix["matrix_sha256"],
        expected_decision_count=len(tasks),
    )
    summary["model_configuration"] = configuration
    summary["config_sha256"] = config["config_sha256"]
    write_json(destination / "summary.json", summary)
    regression = fit_pilot_logit(decisions)
    write_json(destination / "regression.json", regression)
    rates = " ".join(
        f"{label}={summary['choice_rates_by_level'][label]['rate']:.3f}"
        for label in summary["choice_rates_by_level"]
    )
    print(f"choice_rates {rates}", flush=True)
    return destination


def summarize_calibration(
    decisions: list[dict[str, Any]],
    *,
    matrix_sha256: str,
    expected_decision_count: int | None = None,
) -> dict[str, Any]:
    valid = [row for row in decisions if row.get("schema_valid")]
    by_level = _rates_by_level(valid)
    matched = _matched_units(valid)
    thresholds = _switch_thresholds(valid)
    rates = [_rate_value(by_level[label]) for label in _level_labels()]
    return {
        "experiment_id": EXPERIMENT_ID,
        "analysis_level": "pilot",
        "decision_count": len(decisions),
        "valid_decision_count": len(valid),
        "expected_decision_count": expected_decision_count,
        "complete": expected_decision_count is not None and len(decisions) == expected_decision_count,
        "schema_failure_count": sum(int(row.get("schema_failure_count", 0)) for row in decisions),
        "invalid_response_count": sum(1 for row in decisions if not row.get("schema_valid")),
        "retry_count": sum(int(row.get("retry_count", 0)) for row in decisions),
        "choice_rates_by_level": by_level,
        "difficulty_by_aversion": _rate_matrix(valid, "avoidance_cost_level"),
        "block_by_aversion": _rate_matrix(valid, "tradeoff_block"),
        "difficulty_by_block_and_aversion": _difficulty_by_block(valid),
        "switch_threshold": thresholds,
        "matched_units": matched,
        "monotonicity": _monotonicity(rates, matched),
        "ceiling_floor": _ceiling_floor(valid, by_level),
        "criteria": _criteria(rates, thresholds, matched),
        "scenario_matrix_sha256": matrix_sha256,
    }


def fit_pilot_logit(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in decisions if row.get("schema_valid")]
    report: dict[str, Any] = {
        "analysis_level": "pilot",
        "note": PILOT_NOTE,
        "formula": FORMULA,
        "success_criterion": "predicted probabilities, not p-values",
        "n_observations": len(valid),
        "n_clusters": len({row["base_agent_id"] for row in valid}),
        "status": "not_fit",
        "coefficients": {},
        "interaction": None,
        "predicted_probabilities": [],
        "curves_mean_across_blocks": [],
    }
    if len(valid) < 30 or report["n_clusters"] < 2:
        report["status"] = "insufficient_data"
        return report
    outcomes = {int(row["chosen_lower_crowding"]) for row in valid}
    if len(outcomes) < 2:
        report["status"] = "no_outcome_variation"
        return report
    frame = pd.DataFrame(
        {
            "chosen_lower_crowding": [int(row["chosen_lower_crowding"]) for row in valid],
            "crowd_aversion": [float(row["crowd_aversion"]) for row in valid],
            "avoidance_cost_level": [int(row["avoidance_cost_level"]) for row in valid],
            "tradeoff_block": [str(row["tradeoff_block"]) for row in valid],
            "base_agent_id": [str(row["base_agent_id"]) for row in valid],
        }
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = smf.logit(FORMULA, data=frame).fit(
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
    interaction_name = _interaction_name(fitted.params.index)
    if interaction_name is not None:
        report["interaction"] = report["coefficients"][interaction_name]
        report["interaction"]["name"] = interaction_name
    grid = _prediction_grid()
    try:
        predicted = fitted.get_prediction(grid).summary_frame()
        probability_column = "predicted" if "predicted" in predicted.columns else "mean"
        probabilities = [float(value) for value in predicted[probability_column]]
        lower_name = "ci_lower" if "ci_lower" in predicted.columns else "mean_ci_lower"
        upper_name = "ci_upper" if "ci_upper" in predicted.columns else "mean_ci_upper"
        lower = [float(value) for value in predicted[lower_name]] if lower_name in predicted.columns else None
        upper = [float(value) for value in predicted[upper_name]] if upper_name in predicted.columns else None
    except Exception:
        probabilities = [float(value) for value in fitted.predict(grid)]
        lower = None
        upper = None
    detailed = []
    for index, row in grid.iterrows():
        item = {
            "crowd_aversion": float(row["crowd_aversion"]),
            "avoidance_cost_level": int(row["avoidance_cost_level"]),
            "tradeoff_block": str(row["tradeoff_block"]),
            "probability": probabilities[index],
        }
        if lower is not None and upper is not None:
            item["ci95"] = [lower[index], upper[index]]
        detailed.append(item)
    report["predicted_probabilities"] = detailed
    report["curves_mean_across_blocks"] = _mean_curves(detailed)
    return report


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
    if choice is not None:
        lower = chosen_lower_crowding(choice, place_a, place_b)
    attempts = schema_failures if parsed is None else schema_failures + 1
    return {
        "base_agent_id": task["base_agent_id"],
        "crowd_aversion": task["crowd_aversion"],
        "scenario_id": task["scenario_id"],
        "tradeoff_block": task["tradeoff_block"],
        "avoidance_cost_level": task["avoidance_cost_level"],
        "place_a": place_a,
        "place_b": place_b,
        "traits": {name: getattr(task["traits"], name) for name in TRAIT_NAMES},
        "choice": choice,
        "chosen_lower_crowding": lower,
        "reason_codes": list(parsed.reason_codes) if parsed is not None else [],
        "model_configuration": configuration,
        "response_hash": hashlib.sha256(raw_response.encode("utf-8")).hexdigest(),
        "schema_valid": parsed is not None,
        "schema_failure_count": schema_failures,
        "retry_count": max(attempts - 1, 0),
        "error": error or None,
    }


def _tasks(bases: list[dict[str, Any]], scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tasks = []
    for base in bases:
        for level in CROWD_AVERSION_LEVELS:
            traits = traits_for_condition(base["traits"], level)
            for scenario in scenarios:
                tasks.append(
                    {
                        "base_agent_id": base["base_agent_id"],
                        "crowd_aversion": level,
                        "traits": traits,
                        "scenario_id": scenario["scenario_id"],
                        "tradeoff_block": scenario["tradeoff_block"],
                        "avoidance_cost_level": scenario["avoidance_cost_level"],
                        "place_a": scenario["place_a"],
                        "place_b": scenario["place_b"],
                    }
                )
    return tasks


def _rates_by_level(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    return {label: _proportion(rows, label) for label in _level_labels()}


def _rate_matrix(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, dict[str, float | int]]]:
    values = sorted({row[key] for row in rows}, key=lambda value: (isinstance(value, str), value))
    return {str(value): _rates_for_slice(rows, key, value) for value in values}


def _difficulty_by_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    blocks = sorted({str(row["tradeoff_block"]) for row in rows})
    return {
        block: _rate_matrix([row for row in rows if row["tradeoff_block"] == block], "avoidance_cost_level")
        for block in blocks
    }


def _rates_for_slice(rows: list[dict[str, Any]], key: str, value: Any) -> dict[str, dict[str, float | int]]:
    selected = [row for row in rows if row[key] == value]
    return {label: _proportion(selected, label) for label in _level_labels()}


def _proportion(rows: list[dict[str, Any]], label: str) -> dict[str, float | int]:
    selected = [row for row in rows if _level_label(float(row["crowd_aversion"])) == label]
    if not selected:
        return {"rate": 0.0, "n": 0}
    hits = sum(int(row["chosen_lower_crowding"]) for row in selected)
    return {"rate": hits / len(selected), "n": len(selected)}


def _matched_units(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["base_agent_id"]), str(row["scenario_id"]))].append(row)
    counts = {"unchanged": 0, "expected_direction": 0, "reverse": 0, "nonmonotonic": 0, "incomplete": 0}
    for unit_rows in grouped.values():
        counts[_classify_unit(unit_rows)] += 1
    complete = sum(counts[name] for name in ("unchanged", "expected_direction", "reverse", "nonmonotonic"))
    changed = counts["expected_direction"] + counts["reverse"] + counts["nonmonotonic"]
    return {
        "n": len(grouped),
        "complete_n": complete,
        "denominator": "complete matched base-agent by scenario units",
        "unchanged": counts["unchanged"],
        "unchanged_rate": _safe_rate(counts["unchanged"], complete),
        "changed": changed,
        "switch_rate": _safe_rate(changed, complete),
        "expected_direction": counts["expected_direction"],
        "expected_direction_rate": _safe_rate(counts["expected_direction"], complete),
        "reverse": counts["reverse"],
        "reverse_rate": _safe_rate(counts["reverse"], complete),
        "nonmonotonic": counts["nonmonotonic"],
        "nonmonotonic_rate": _safe_rate(counts["nonmonotonic"], complete),
        "incomplete": counts["incomplete"],
    }


def _classify_unit(rows: list[dict[str, Any]]) -> str:
    ordered = sorted(rows, key=lambda row: float(row["crowd_aversion"]))
    labels = [_level_label(float(row["crowd_aversion"])) for row in ordered]
    if labels != _level_labels() or len(rows) != len(CROWD_AVERSION_LEVELS):
        return "incomplete"
    choices = [int(row["chosen_lower_crowding"]) for row in ordered]
    if len(set(choices)) == 1:
        return "unchanged"
    steps = [choices[index + 1] - choices[index] for index in range(len(choices) - 1)]
    if all(step >= 0 for step in steps):
        return "expected_direction"
    if all(step <= 0 for step in steps):
        return "reverse"
    return "nonmonotonic"


def _switch_thresholds(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row["base_agent_id"]), str(row["tradeoff_block"]), _level_label(float(row["crowd_aversion"])))
        grouped[key].append(row)
    by_level: dict[str, list[int]] = {label: [] for label in _level_labels()}
    by_level_block: dict[str, dict[str, list[int]]] = {label: defaultdict(list) for label in _level_labels()}
    for (agent_id, block, label), unit_rows in grouped.items():
        del agent_id
        levels = {int(row["avoidance_cost_level"]) for row in unit_rows}
        if levels != set(range(1, 7)):
            continue
        accepted = [int(row["avoidance_cost_level"]) for row in unit_rows if int(row["chosen_lower_crowding"]) == 1]
        threshold = max(accepted) if accepted else 0
        by_level[label].append(threshold)
        by_level_block[label][block].append(threshold)
    return {
        "definition": "highest avoidance_cost_level at which the low-crowd place is still chosen; 0 if it is never chosen",
        "mean_by_level": {label: _mean(values) for label, values in by_level.items()},
        "mean_by_level_and_block": {
            label: {block: _mean(values) for block, values in sorted(blocks.items())}
            for label, blocks in by_level_block.items()
        },
    }


def _monotonicity(rates: list[float], matched: dict[str, Any]) -> dict[str, Any]:
    usable = all(math.isfinite(rate) for rate in rates) and len(rates) == len(CROWD_AVERSION_LEVELS)
    return {
        "expected_direction": " <= ".join(_level_labels()),
        "holds": usable and all(rates[index] <= rates[index + 1] for index in range(len(rates) - 1)),
        "strict": usable and all(rates[index] < rates[index + 1] for index in range(len(rates) - 1)),
        "rates": {label: rates[index] for index, label in enumerate(_level_labels())},
        "matched_units_non_strictly_increasing": matched["unchanged"] + matched["expected_direction"],
        "matched_units_strictly_increasing": matched["expected_direction"],
    }


def _ceiling_floor(
    rows: list[dict[str, Any]],
    by_level: dict[str, dict[str, float | int]],
) -> dict[str, Any]:
    by_difficulty = _rate_matrix(rows, "avoidance_cost_level")
    return {
        "ceiling_rule": f"choice rate >= {CEILING_RATE}",
        "floor_rule": f"choice rate <= {FLOOR_RATE}",
        "by_level": {label: _bound_flags(_rate_value(stats)) for label, stats in by_level.items()},
        "by_difficulty": {
            level: {label: _bound_flags(_rate_value(stats)) for label, stats in levels.items()}
            for level, levels in by_difficulty.items()
        },
        "level_rate_min": min(_rate_value(stats) for stats in by_level.values()),
        "level_rate_max": max(_rate_value(stats) for stats in by_level.values()),
        "level_rate_range": max(_rate_value(stats) for stats in by_level.values())
        - min(_rate_value(stats) for stats in by_level.values()),
    }


def _criteria(rates: list[float], thresholds: dict[str, Any], matched: dict[str, Any]) -> dict[str, Any]:
    threshold_means = [float(thresholds["mean_by_level"][label]) for label in _level_labels()]
    rate_range = max(rates) - min(rates) if rates else 0.0
    return {
        "A_dose_response": {
            "non_strict_increasing": all(rates[index] <= rates[index + 1] for index in range(len(rates) - 1)),
            "strict_increasing": all(rates[index] < rates[index + 1] for index in range(len(rates) - 1)),
            "rates": {label: rates[index] for index, label in enumerate(_level_labels())},
        },
        "B_tradeoff_sensitivity": {
            "non_strict_increasing_mean_threshold": all(
                threshold_means[index] <= threshold_means[index + 1] for index in range(len(threshold_means) - 1)
            ),
            "mean_switch_threshold": thresholds["mean_by_level"],
        },
        "C_discrimination": {
            "rate_range": rate_range,
            "switch_rate": matched["switch_rate"],
            "note": "Descriptive separation only. This does not establish construct validity.",
        },
    }


def _prediction_grid() -> pd.DataFrame:
    rows = []
    for block in ("A", "B", "C", "D"):
        for level in CROWD_AVERSION_LEVELS:
            for cost in range(1, 7):
                rows.append(
                    {
                        "tradeoff_block": block,
                        "crowd_aversion": level,
                        "avoidance_cost_level": cost,
                    }
                )
    return pd.DataFrame(rows)


def _mean_curves(detailed: list[dict[str, Any]]) -> list[dict[str, float | int]]:
    buckets: dict[tuple[float, int], list[float]] = defaultdict(list)
    for item in detailed:
        buckets[(float(item["crowd_aversion"]), int(item["avoidance_cost_level"]))].append(float(item["probability"]))
    return [
        {
            "crowd_aversion": level,
            "avoidance_cost_level": cost,
            "probability": _mean(values),
        }
        for (level, cost), values in sorted(buckets.items())
    ]


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


def _interaction_name(names: Any) -> str | None:
    for name in names:
        text = str(name)
        if "crowd_aversion" in text and "avoidance_cost_level" in text and ":" in text:
            return text
    return None


def _bound_flags(rate: float) -> dict[str, float | bool]:
    return {"rate": rate, "ceiling": rate >= CEILING_RATE, "floor": rate <= FLOOR_RATE}


def _config_snapshot(settings: Settings, configuration: dict[str, Any], matrix_sha256: str) -> dict[str, Any]:
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "persona_set": "persona_set_v1",
        "model_configuration": configuration,
        "crowd_aversion_levels": list(CROWD_AVERSION_LEVELS),
        "crowd_aversion_definition": CROWD_AVERSION_DEFINITION,
        "base_agent_count": settings.calibration_base_agent_count,
        "expected_decision_count": EXPECTED_DECISIONS,
        "scenario_matrix_sha256": matrix_sha256,
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


def _require_calibration_settings(settings: Settings) -> None:
    if settings.temperature != 0.0:
        raise RuntimeError("experiment 001B requires temperature 0")
    if settings.reasoning_effort != "none":
        raise RuntimeError("experiment 001B requires reasoning_effort none")
    if settings.calibration_base_agent_count != 20:
        raise RuntimeError("experiment 001B requires 20 base agents")
    if settings.calibration_prompt_version != "place_choice_v2":
        raise RuntimeError("experiment 001B requires place_choice_v2")


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
            "attempt": attempt,
            "model_configuration": configuration,
            "request_messages": messages,
            "raw_response": raw_response,
            "parsed_ok": parsed_ok,
            "error": error or None,
        },
    )


def _completed_keys(path: Path) -> set[tuple[str, str, str]]:
    keys = set()
    for row in _read_jsonl(path):
        keys.add((row["base_agent_id"], _level_label(float(row["crowd_aversion"])), row["scenario_id"]))
    return keys


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _level_labels() -> list[str]:
    return [_level_label(level) for level in CROWD_AVERSION_LEVELS]


def _level_label(value: float) -> str:
    return f"{value:.2f}"


def _rate_value(stats: dict[str, float | int]) -> float:
    return float(stats["rate"])


def _safe_rate(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return count / total


def _mean(values: list[int] | list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def main() -> None:
    run_experiment()


if __name__ == "__main__":
    if __package__ is None:
        raise SystemExit("請在 worldlab 目錄執行：python -m experiments.experiment_001b")
    main()
