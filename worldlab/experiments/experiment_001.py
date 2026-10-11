"""Experiment 001: whether crowd_aversion changes lower-crowding place choice.

The decision engine receives matched numerical traits only. Behavioral profile
text is not sent. This module does not create a world, memory, or UI.
Results are a pilot-level analysis.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import statsmodels.api as sm
from pydantic import ValidationError

from agents.persona_generator import append_jsonl, write_json
from config.settings import PROJECT_ROOT, Settings, get_settings
from experiments.decision_engine import (
    build_decision_messages,
    chosen_lower_crowding,
    traits_for_condition,
)
from experiments.place_scenarios import generate_scenario_matrix_from_settings
from experiments.reproducibility_check import canonical_sha256
from llm.client import LLMCallError, LLMClient, StructuredLLMClient
from models.schemas import TRAIT_NAMES, AgentTraits, PlaceChoiceDecision

EXPERIMENT_ID = "experiment_001_crowd_aversion"
OUTPUT_DIR = PROJECT_ROOT / "data" / "experiments" / EXPERIMENT_ID
CONDITIONS = ("LOW", "MID", "HIGH")
FORMULA = (
    "chosen_lower_crowding ~ crowd_aversion + attractiveness_diff "
    "+ distance_diff + novelty_diff"
)
PILOT_NOTE = (
    "Cluster-robust standard errors are grouped by base_agent. "
    "With ten clusters this is a pilot-level analysis, not a confirmatory estimate."
)


def run_experiment(
    settings: Settings | None = None,
    client: LLMClient | None = None,
    *,
    persona_dir: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    active = get_settings() if settings is None else settings
    _require_decision_settings(active)
    destination = output_dir or OUTPUT_DIR
    destination.mkdir(parents=True, exist_ok=True)
    source = persona_dir or (active.personas_dir / "persona_set_v1")
    bases = load_base_traits(source, active.base_agent_count)
    matrix = generate_scenario_matrix_from_settings(active)
    write_json(destination / "scenarios.json", matrix)
    llm = client if client is not None else StructuredLLMClient(active)
    audit_path = destination / "llm_audit.jsonl"
    decisions_path = destination / "decisions.jsonl"
    completed = _completed_keys(decisions_path)
    configuration = _model_configuration(active)
    tasks = list(_tasks(bases, matrix["scenarios"], active))
    for index, task in enumerate(tasks, start=1):
        key = (task["base_agent_id"], task["condition"], task["scenario_id"])
        if key in completed:
            continue
        print(f"{index}/{len(tasks)} {task['base_agent_id']} {task['condition']} {task['scenario_id']}", flush=True)
        record = _execute_task(active, llm, task, configuration, audit_path)
        append_jsonl(decisions_path, record)
    decisions = _read_jsonl(decisions_path)
    summary = summarize_decisions(
        decisions,
        matrix_sha256=matrix["matrix_sha256"],
        expected_decision_count=len(tasks),
    )
    summary["model_configuration"] = configuration
    summary["scenario_matrix_sha256"] = matrix["matrix_sha256"]
    summary["analysis_level"] = "pilot"
    write_json(destination / "summary.json", summary)
    regression = fit_clustered_logit(decisions)
    write_json(destination / "regression.json", regression)
    print(
        "proportions "
        f"LOW={summary['proportions']['LOW']} "
        f"MID={summary['proportions']['MID']} "
        f"HIGH={summary['proportions']['HIGH']}",
        flush=True,
    )
    return destination


def load_base_traits(persona_dir: Path, count: int) -> list[dict[str, Any]]:
    paths = sorted(persona_dir.glob("agent_*.json"))[:count]
    if len(paths) != count:
        raise RuntimeError(f"expected {count} base agents in {persona_dir}")
    bases = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        traits = AgentTraits.model_validate(payload["traits"])
        bases.append({"base_agent_id": payload["agent_id"], "traits": traits})
    return bases


def summarize_decisions(
    decisions: list[dict[str, Any]],
    *,
    matrix_sha256: str,
    expected_decision_count: int | None = None,
) -> dict[str, Any]:
    valid = [row for row in decisions if row.get("schema_valid")]
    proportions = {condition: _proportion(valid, condition) for condition in CONDITIONS}
    monotonic = _monotonic(proportions)
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
        "proportions": proportions,
        "monotonicity": monotonic,
        "by_base_agent": _grouped_proportions(valid, "base_agent_id"),
        "by_scenario": _grouped_proportions(valid, "scenario_id"),
        "scenario_matrix_sha256": matrix_sha256,
    }


def fit_clustered_logit(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in decisions if row.get("schema_valid")]
    report: dict[str, Any] = {
        "analysis_level": "pilot",
        "note": PILOT_NOTE,
        "formula": FORMULA,
        "n_observations": len(valid),
        "n_clusters": len({row["base_agent_id"] for row in valid}),
        "status": "not_fit",
        "coefficients": {},
    }
    if len(valid) < 10 or report["n_clusters"] < 2:
        report["status"] = "insufficient_data"
        return report
    outcome = [int(row["chosen_lower_crowding"]) for row in valid]
    if len(set(outcome)) < 2:
        report["status"] = "no_outcome_variation"
        return report
    import pandas as pd

    exog = pd.DataFrame(
        {
            "const": [1.0] * len(valid),
            "crowd_aversion": [float(row["crowd_aversion"]) for row in valid],
            "attractiveness_diff": [_attractiveness_diff(row) for row in valid],
            "distance_diff": [_distance_diff(row) for row in valid],
            "novelty_diff": [_novelty_diff(row) for row in valid],
        }
    )
    groups = [row["base_agent_id"] for row in valid]
    try:
        fitted = sm.Logit(outcome, exog).fit(
            disp=False,
            cov_type="cluster",
            cov_kwds={"groups": groups},
        )
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        return report
    report["status"] = "ok"
    confidence = fitted.conf_int()
    for name in fitted.params.index:
        coefficient = float(fitted.params[name])
        low, high = (float(value) for value in confidence.loc[name])
        report["coefficients"][str(name)] = {
            "coefficient": coefficient,
            "se": float(fitted.bse[name]),
            "p_value": float(fitted.pvalues[name]),
            "ci95": [low, high],
            "odds_ratio": float(math.exp(coefficient)),
        }
    return report


def _execute_task(
    settings: Settings,
    client: LLMClient,
    task: dict[str, Any],
    configuration: dict[str, Any],
    audit_path: Path,
) -> dict[str, Any]:
    messages = build_decision_messages(
        task["traits"],
        task["place_a"],
        task["place_b"],
        decision_prompt_version=settings.decision_prompt_version,
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
    record = _decision_record(task, configuration, parsed, raw_response, schema_failures, error)
    return record


def _decision_record(
    task: dict[str, Any],
    configuration: dict[str, Any],
    parsed: PlaceChoiceDecision | None,
    raw_response: str,
    schema_failures: int,
    error: str,
) -> dict[str, Any]:
    choice = parsed.choice if parsed is not None else None
    lower = None
    if choice is not None:
        lower = chosen_lower_crowding(choice, task["place_a"], task["place_b"])
    attempts = schema_failures if parsed is None else schema_failures + 1
    return {
        "base_agent_id": task["base_agent_id"],
        "condition": task["condition"],
        "crowd_aversion": task["crowd_aversion"],
        "scenario_id": task["scenario_id"],
        "place_a": task["place_a"],
        "place_b": task["place_b"],
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


def _tasks(bases: list[dict[str, Any]], scenarios: list[dict[str, Any]], settings: Settings) -> list[dict[str, Any]]:
    levels = {
        "LOW": settings.crowd_aversion_low,
        "MID": settings.crowd_aversion_mid,
        "HIGH": settings.crowd_aversion_high,
    }
    tasks = []
    for base in bases:
        for condition in CONDITIONS:
            traits = traits_for_condition(base["traits"], levels[condition])
            for scenario in scenarios:
                tasks.append(
                    {
                        "base_agent_id": base["base_agent_id"],
                        "condition": condition,
                        "crowd_aversion": levels[condition],
                        "traits": traits,
                        "scenario_id": scenario["scenario_id"],
                        "place_a": scenario["place_a"],
                        "place_b": scenario["place_b"],
                    }
                )
    return tasks


def _attractiveness_diff(row: dict[str, Any]) -> float:
    low, high = _low_high(row)
    return low["visual_attractiveness"] - high["visual_attractiveness"]


def _distance_diff(row: dict[str, Any]) -> float:
    low, high = _low_high(row)
    return low["distance"] - high["distance"]


def _novelty_diff(row: dict[str, Any]) -> float:
    low, high = _low_high(row)
    return low["novelty"] - high["novelty"]


def _low_high(row: dict[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    place_a = row["place_a"]
    place_b = row["place_b"]
    if place_a["crowding"] < place_b["crowding"]:
        return place_a, place_b
    return place_b, place_a


def _proportion(rows: list[dict[str, Any]], condition: str) -> dict[str, float | int]:
    selected = [row for row in rows if row["condition"] == condition]
    if not selected:
        return {"rate": 0.0, "n": 0}
    hits = sum(int(row["chosen_lower_crowding"]) for row in selected)
    return {"rate": hits / len(selected), "n": len(selected)}


def _grouped_proportions(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return {
        name: {condition: _proportion(items, condition) for condition in CONDITIONS}
        for name, items in sorted(grouped.items())
    }


def _monotonic(proportions: dict[str, dict[str, float | int]]) -> dict[str, Any]:
    rates = [float(proportions[condition]["rate"]) for condition in CONDITIONS]
    counts = [int(proportions[condition]["n"]) for condition in CONDITIONS]
    usable = all(count > 0 for count in counts)
    return {
        "expected_direction": "LOW <= MID <= HIGH",
        "holds": usable and rates[0] <= rates[1] <= rates[2],
        "strict": usable and rates[0] < rates[1] < rates[2],
        "rates": {condition: rates[index] for index, condition in enumerate(CONDITIONS)},
    }


def _model_configuration(settings: Settings) -> dict[str, Any]:
    return {
        "model": settings.model,
        "temperature": settings.temperature,
        "seed": settings.llm_seed,
        "reasoning_effort": settings.reasoning_effort,
        "api_endpoint": settings.api_endpoint,
        "decision_prompt_version": settings.decision_prompt_version,
    }


def _require_decision_settings(settings: Settings) -> None:
    if settings.temperature != 0.0:
        raise RuntimeError("experiment 001 requires temperature 0")
    if settings.reasoning_effort != "none":
        raise RuntimeError("experiment 001 requires reasoning_effort none")
    if settings.base_agent_count != 10 or settings.scenario_count != 30:
        raise RuntimeError("experiment 001 requires 10 base agents and 30 scenarios")


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
            "condition": task["condition"],
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
    if not path.exists():
        return set()
    keys = set()
    for row in _read_jsonl(path):
        keys.add((row["base_agent_id"], row["condition"], row["scenario_id"]))
    return keys


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def response_hash(raw_response: str) -> str:
    return hashlib.sha256(raw_response.encode("utf-8")).hexdigest()


def matrix_fingerprint(matrix: dict[str, Any]) -> str:
    return canonical_sha256({key: value for key, value in matrix.items() if key != "matrix_sha256"})


def main() -> None:
    run_experiment()


if __name__ == "__main__":
    if __package__ is None:
        raise SystemExit("請在 worldlab 目錄執行：python -m experiments.experiment_001")
    main()
