"""Experiment 001E: label-free independent utility estimates.

Each place is scored in its own stateless call. The model sees numerical
traits and one place. It does not see the other place, a letter, a position,
or a scenario id. Python compares the two places afterward. Ties stay ties
in the primary analysis. This module does not modify earlier experiments and
does not create a world.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from scipy.stats import pearsonr, spearmanr

from agents.persona_generator import append_jsonl, write_json
from config.settings import PROJECT_ROOT, Settings, get_settings
from experiments.decision_engine import PLACE_GUIDE, traits_for_condition
from experiments.experiment_001 import OUTPUT_DIR as EXPERIMENT_001_DIR
from experiments.experiment_001 import load_base_traits
from experiments.experiment_001b import OUTPUT_DIR as EXPERIMENT_001B_DIR
from experiments.experiment_001b import calibration_trait_definitions
from experiments.experiment_001c import OUTPUT_DIR as EXPERIMENT_001C_DIR
from experiments.experiment_001c import load_frozen_scenarios
from experiments.experiment_001d import (
    BASE_AGENT_COUNT,
    CROWD_AVERSION_LEVELS,
    OPTION_1,
    OPTION_2,
    OUTPUT_DIR as EXPERIMENT_001D_DIR,
    build_design,
    select_scenarios,
)
from experiments.reproducibility_check import canonical_sha256
from experiments.tradeoff_scenarios import PROMPT_FIELDS
from llm.client import LLMCallError, LLMClient, StructuredLLMClient
from models.schemas import TRAIT_NAMES

EXPERIMENT_ID = "experiment_001e_independent_utility"
OUTPUT_DIR = PROJECT_ROOT / "data" / "experiments" / EXPERIMENT_ID
PROMPT_VERSION = "independent_utility_v3"
REPLICATES = (1, 2)
SCORE_FIELDS = ("overall_utility", "crowding_fit", "visual_fit", "distance_fit", "novelty_fit")
EXPECTED_CALLS = BASE_AGENT_COUNT * len(CROWD_AVERSION_LEVELS) * 12 * 2 * len(REPLICATES)
TIED_BREAK_ALGORITHM = "worldlab.utility_tiebreak.v1"
ARCHITECTURE_NOTE = (
    "label_invariance and position_invariance are 1.0 because those variables "
    "were never sent to the model and never caused another call. "
    "This is architecture-level invariance, not evidence about the model's psychology."
)
COMPARISON_NOTE = (
    "Direct-choice v2 and independent-utility v3 are different measurement processes. "
    "Their choice rates are not interchangeable."
)
_BANNED_PROMPT_TEXT = (
    "option a",
    "option b",
    "place_a",
    "place_b",
    "physical_option",
    "scenario_id",
    "replicate",
    "ranking",
    "preferred option",
    "choose a",
    "choose b",
    "choose between",
    "first option",
    "second option",
    "enum order",
    "pairwise",
)


class PlaceUtility(BaseModel):
    """Fit scores for one place. The model does not select between places."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    response_schema_name: ClassVar[str] = "place_utility"

    overall_utility: int = Field(ge=0, le=1000)
    crowding_fit: int = Field(ge=0, le=1000)
    visual_fit: int = Field(ge=0, le=1000)
    distance_fit: int = Field(ge=0, le=1000)
    novelty_fit: int = Field(ge=0, le=1000)


def build_utility_messages(traits: Any, place: dict[str, float], *, prompt_version: str) -> list[dict[str, str]]:
    attributes = {field: place[field] for field in PROMPT_FIELDS}
    definitions = calibration_trait_definitions()
    trait_payload = {name: getattr(traits, name) for name in TRAIT_NAMES}
    rendered = "\n".join(f"- {name}: {definitions[name]}" for name in TRAIT_NAMES)
    system = (
        "You evaluate how well one place fits one simulated agent.\n"
        "Use only the numerical traits and the single place attribute list.\n"
        "Scores are integers from 0 to 1000.\n"
        "0 means an extremely poor fit.\n"
        "1000 means an extremely strong fit.\n"
        "overall_utility is the overall fit of this place.\n"
        "crowding_fit, visual_fit, distance_fit, and novelty_fit score those attributes.\n"
        "Return JSON only, matching the schema. No markdown.\n"
    )
    user = (
        f"prompt_version: {prompt_version}\n\n"
        "Trait definitions:\n"
        f"{rendered}\n\n"
        "TRAITS_JSON\n"
        f"{_dump(trait_payload)}\n"
        "END_TRAITS_JSON\n\n"
        "Place attribute definitions:\n"
        f"{PLACE_GUIDE}\n\n"
        "PLACE\n"
        f"{_dump(attributes)}\n"
        "END_PLACE\n\n"
        "Rate this place.\n"
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    _require_single_place_prompt(messages)
    return messages


def prompt_sha256(messages: list[dict[str, str]]) -> str:
    payload = "\n\n".join(message["content"] for message in messages)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def seeded_tie_break(agent_id: str, crowd_aversion: float, scenario_id: str, seed: int) -> str:
    material = f"{TIED_BREAK_ALGORITHM}|{seed}|{agent_id}|{_level_label(crowd_aversion)}|{scenario_id}"
    bit = hashlib.sha256(material.encode("utf-8")).digest()[0] & 1
    return OPTION_1 if bit == 0 else OPTION_2


def run_experiment(
    settings: Settings | None = None,
    client: LLMClient | None = None,
    *,
    persona_dir: Path | None = None,
    output_dir: Path | None = None,
    scenario_path: Path | None = None,
    direct_choice_dir: Path | None = None,
) -> Path:
    active = get_settings() if settings is None else settings
    _require_gate_settings(active)
    destination = output_dir or OUTPUT_DIR
    forbidden = {
        EXPERIMENT_001_DIR.resolve(),
        EXPERIMENT_001B_DIR.resolve(),
        EXPERIMENT_001C_DIR.resolve(),
        EXPERIMENT_001D_DIR.resolve(),
    }
    if destination.resolve() in forbidden:
        raise RuntimeError("experiment 001E must not overwrite earlier experiments")
    destination.mkdir(parents=True, exist_ok=True)
    source = persona_dir or (active.personas_dir / "persona_set_v1")
    bases = load_base_traits(source, BASE_AGENT_COUNT)
    matrix = load_frozen_scenarios(scenario_path or (EXPERIMENT_001B_DIR / "scenarios.json"))
    design = build_design(select_scenarios(matrix))
    write_json(destination / "source_design.json", design)
    configuration = _model_configuration(active)
    config = _config_snapshot(active, configuration, matrix["matrix_sha256"], design["design_sha256"])
    write_json(destination / "config.json", config)
    llm = client if client is not None else StructuredLLMClient(active)
    audit_path = destination / "llm_audit.jsonl"
    evaluations_path = destination / "evaluations.jsonl"
    completed = _completed_keys(evaluations_path)
    tasks = _tasks(bases, design["pairs"])
    for index, task in enumerate(tasks, start=1):
        if _task_key(task) in completed:
            continue
        print(
            f"{index}/{len(tasks)} {task['base_agent_id']} {_level_label(task['crowd_aversion'])} "
            f"{task['scenario_id']} {task['substantive_place_id']} r{task['replicate_id']}",
            flush=True,
        )
        record = _execute_task(active, llm, task, configuration, audit_path)
        append_jsonl(evaluations_path, record)
    evaluations = _read_jsonl(evaluations_path)
    units = aggregate_units(evaluations, design["pairs"], active.llm_seed)
    summary = summarize_utilities(
        evaluations,
        units,
        matrix_sha256=matrix["matrix_sha256"],
        design_sha256=design["design_sha256"],
        expected_call_count=len(tasks),
    )
    reference = direct_choice_dir or EXPERIMENT_001D_DIR
    summary["comparison"] = compare_with_direct_choice(summary, _read_json(reference / "summary.json"), _read_jsonl(reference / "decisions.jsonl"))
    summary["model_configuration"] = configuration
    summary["config_sha256"] = config["config_sha256"]
    write_json(destination / "choices.json", units)
    write_json(destination / "summary.json", summary)
    print(
        "utility_advantage "
        + " ".join(
            f"{label}={summary['utility_advantage_by_aversion'][label]['mean']}"
            for label in summary["utility_advantage_by_aversion"]
        ),
        flush=True,
    )
    return destination


def aggregate_units(
    evaluations: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    tie_seed: int,
) -> list[dict[str, Any]]:
    pair_by_id = {pair["scenario_id"]: pair for pair in pairs}
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in evaluations:
        if not row.get("schema_valid"):
            continue
        key = (
            str(row["base_agent_id"]),
            _level_label(float(row["crowd_aversion"])),
            str(row["scenario_id"]),
            str(row["substantive_place_id"]),
        )
        grouped[key].append(row)
    units = []
    scenario_ids = sorted({key[2] for key in grouped} | set(pair_by_id))
    agents = sorted({key[0] for key in grouped})
    for agent_id in agents:
        for level in CROWD_AVERSION_LEVELS:
            for scenario_id in scenario_ids:
                pair = pair_by_id.get(scenario_id)
                if pair is None:
                    continue
                place_means = {}
                complete = True
                for place_id in (OPTION_1, OPTION_2):
                    rows = grouped.get((agent_id, _level_label(level), scenario_id, place_id), [])
                    scores = [int(row["overall_utility"]) for row in rows]
                    if len(scores) != len(REPLICATES):
                        complete = False
                        place_means[place_id] = None
                    else:
                        place_means[place_id] = sum(scores) / len(scores)
                if not complete:
                    units.append(
                        {
                            "base_agent_id": agent_id,
                            "crowd_aversion": level,
                            "scenario_id": scenario_id,
                            "complete": False,
                            "tie": None,
                            "primary_choice": None,
                            "low_crowd_chosen": None,
                        }
                    )
                    continue
                mean_1 = float(place_means[OPTION_1])
                mean_2 = float(place_means[OPTION_2])
                low_id = pair["lower_crowding_substantive_id"]
                if mean_1 == mean_2:
                    tie = True
                    primary = None
                    low_chosen = None
                elif mean_1 > mean_2:
                    tie = False
                    primary = OPTION_1
                    low_chosen = primary == low_id
                else:
                    tie = False
                    primary = OPTION_2
                    low_chosen = primary == low_id
                secondary = primary if primary is not None else seeded_tie_break(agent_id, level, scenario_id, tie_seed)
                units.append(
                    {
                        "base_agent_id": agent_id,
                        "crowd_aversion": level,
                        "scenario_id": scenario_id,
                        "tradeoff_block": pair["tradeoff_block"],
                        "avoidance_cost_level": pair["avoidance_cost_level"],
                        "lower_crowding_substantive_id": low_id,
                        "complete": True,
                        "mean_utility_place_1": mean_1,
                        "mean_utility_place_2": mean_2,
                        "mean_utility_low_crowd": mean_1 if low_id == OPTION_1 else mean_2,
                        "mean_utility_high_crowd": mean_2 if low_id == OPTION_1 else mean_1,
                        "utility_advantage": (mean_1 - mean_2) if low_id == OPTION_1 else (mean_2 - mean_1),
                        "tie": tie,
                        "primary_choice": primary,
                        "low_crowd_chosen": low_chosen,
                        "secondary_choice": secondary,
                        "secondary_tie_break_used": tie,
                        "secondary_low_crowd_chosen": secondary == low_id,
                    }
                )
    return units


def summarize_utilities(
    evaluations: list[dict[str, Any]],
    units: list[dict[str, Any]],
    *,
    matrix_sha256: str,
    design_sha256: str,
    expected_call_count: int | None = None,
) -> dict[str, Any]:
    valid = [row for row in evaluations if row.get("schema_valid")]
    complete_units = [unit for unit in units if unit.get("complete")]
    decisive = [unit for unit in complete_units if unit["tie"] is False]
    ties = [unit for unit in complete_units if unit["tie"] is True]
    return {
        "experiment_id": EXPERIMENT_ID,
        "analysis_level": "pilot",
        "measurement_process": "independent_utility_v3",
        "call_count": len(evaluations),
        "valid_call_count": len(valid),
        "expected_call_count": expected_call_count,
        "complete": expected_call_count is not None and len(evaluations) == expected_call_count,
        "schema_failure_count": sum(int(row.get("schema_failure_count", 0)) for row in evaluations),
        "invalid_response_count": sum(1 for row in evaluations if not row.get("schema_valid")),
        "retry_count": sum(int(row.get("retry_count", 0)) for row in evaluations),
        "repeatability": {field: _repeatability(valid, field) for field in SCORE_FIELDS},
        "low_crowd_utility_by_aversion": _mean_by_aversion(complete_units, "mean_utility_low_crowd"),
        "utility_advantage_by_aversion": _mean_by_aversion(complete_units, "utility_advantage"),
        "low_crowd_choice_rate_by_aversion": _choice_rate_by_aversion(decisive),
        "choice_rate_denominator": "complete non-tie units; ties are excluded from the primary choice rate",
        "tradeoff_by_block": _tradeoff_table(complete_units, decisive, "tradeoff_block"),
        "tradeoff_by_avoidance_cost": _tradeoff_table(complete_units, decisive, "avoidance_cost_level"),
        "incomplete_units": sum(1 for unit in units if not unit.get("complete")),
        "ties": _tie_report(complete_units, ties),
        "replicate_choice_agreement": _replicate_agreement(valid),
        "secondary_seeded_tie_break": {
            "algorithm": TIED_BREAK_ALGORITHM,
            "label": "secondary",
            "choice_rate_by_aversion": _secondary_rate_by_aversion(complete_units),
        },
        "architecture_invariance": _architecture_invariance(valid),
        "scenario_matrix_sha256": matrix_sha256,
        "source_design_sha256": design_sha256,
    }


def compare_with_direct_choice(
    utility_summary: dict[str, Any],
    direct_summary: dict[str, Any] | None,
    direct_decisions: list[dict[str, Any]],
) -> dict[str, Any]:
    direct_valid = [row for row in direct_decisions if row.get("schema_valid")]
    report: dict[str, Any] = {
        "note": COMPARISON_NOTE,
        "independent_utility_v3": {
            "behavioral_separation": _separation(utility_summary.get("low_crowd_choice_rate_by_aversion", {})),
            "utility_advantage_separation": _separation(utility_summary.get("utility_advantage_by_aversion", {})),
            "technical_artifact_magnitude": {
                "label": 0.0,
                "position": 0.0,
                "schema_enum_order": 0.0,
                "basis": "architecture; these variables were not in the prompt",
            },
            "usable_matched_unit_rate": _usable_utility_rate(utility_summary),
            "decision_reproducibility": utility_summary.get("replicate_choice_agreement"),
        },
        "direct_choice_v2": {
            "low_crowd_choice_rate_by_aversion": _direct_choice_rates(direct_valid),
            "technical_artifact_magnitude": _direct_artifacts(direct_summary),
            "usable_matched_unit_rate": _direct_complete_rate(direct_valid),
            "decision_reproducibility": None if direct_summary is None else direct_summary.get("substantive_invariance", {}).get("rate"),
            "reproducibility_definition": "substantive invariance when position flips and label plus schema stay fixed",
        },
    }
    report["direct_choice_v2"]["behavioral_separation"] = _separation(
        report["direct_choice_v2"]["low_crowd_choice_rate_by_aversion"]
    )
    return report


def _tasks(bases: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tasks = []
    for base in bases:
        for level in CROWD_AVERSION_LEVELS:
            traits = traits_for_condition(base["traits"], level)
            for pair in pairs:
                for place_id, attributes in (
                    (OPTION_1, pair["physical_option_1"]),
                    (OPTION_2, pair["physical_option_2"]),
                ):
                    messages = build_utility_messages(traits, attributes, prompt_version=PROMPT_VERSION)
                    digest = prompt_sha256(messages)
                    for replicate_id in REPLICATES:
                        tasks.append(
                            {
                                "base_agent_id": base["base_agent_id"],
                                "crowd_aversion": level,
                                "traits": traits,
                                "scenario_id": pair["scenario_id"],
                                "tradeoff_block": pair["tradeoff_block"],
                                "avoidance_cost_level": pair["avoidance_cost_level"],
                                "substantive_place_id": place_id,
                                "place": {field: attributes[field] for field in PROMPT_FIELDS},
                                "replicate_id": replicate_id,
                                "prompt_sha256": digest,
                                "messages": messages,
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
        "base_agent_id": task["base_agent_id"],
        "crowd_aversion": task["crowd_aversion"],
        "scenario_id": task["scenario_id"],
        "tradeoff_block": task["tradeoff_block"],
        "avoidance_cost_level": task["avoidance_cost_level"],
        "substantive_place_id": task["substantive_place_id"],
        "replicate_id": task["replicate_id"],
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


def _repeatability(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str, str], dict[int, int]] = defaultdict(dict)
    for row in rows:
        if row.get(field) is None:
            continue
        key = (
            str(row["base_agent_id"]),
            _level_label(float(row["crowd_aversion"])),
            str(row["scenario_id"]),
            str(row["substantive_place_id"]),
        )
        grouped[key][int(row["replicate_id"])] = int(row[field])
    pairs = [(values[1], values[2]) for values in grouped.values() if 1 in values and 2 in values]
    if not pairs:
        return {"n_pairs": 0}
    diffs = [abs(left - right) for left, right in pairs]
    left = [pair[0] for pair in pairs]
    right = [pair[1] for pair in pairs]
    return {
        "n_pairs": len(pairs),
        "exact_match_rate": sum(1 for diff in diffs if diff == 0) / len(diffs),
        "mean_absolute_difference": sum(diffs) / len(diffs),
        "median_absolute_difference": float(statistics.median(diffs)),
        "pearson_r": _correlation(left, right, pearsonr),
        "spearman_r": _correlation(left, right, spearmanr),
        "icc_1_1": _icc_1_1(pairs),
        "icc_definition": "one-way random, single measurement, absolute agreement",
    }


def _replicate_agreement(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], dict[int, dict[str, int]]] = defaultdict(dict)
    for row in rows:
        if row.get("overall_utility") is None:
            continue
        key = (str(row["base_agent_id"]), _level_label(float(row["crowd_aversion"])), str(row["scenario_id"]))
        replicate = int(row["replicate_id"])
        grouped[key].setdefault(replicate, {})[str(row["substantive_place_id"])] = int(row["overall_utility"])
    comparable = 0
    agreed = 0
    for replicates in grouped.values():
        winners = []
        complete = True
        for replicate_id in REPLICATES:
            scores = replicates.get(replicate_id, {})
            if OPTION_1 not in scores or OPTION_2 not in scores:
                complete = False
                break
            if scores[OPTION_1] == scores[OPTION_2]:
                winners.append(None)
            elif scores[OPTION_1] > scores[OPTION_2]:
                winners.append(OPTION_1)
            else:
                winners.append(OPTION_2)
        if not complete or any(winner is None for winner in winners):
            continue
        comparable += 1
        agreed += int(winners[0] == winners[1])
    return {
        "definition": "same strict winner from replicate 1 and replicate 2; units with a within-replicate tie are excluded",
        "n": comparable,
        "agreement_rate": None if comparable == 0 else agreed / comparable,
    }


def _choice_rate_by_aversion(units: list[dict[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    report = {}
    for level in CROWD_AVERSION_LEVELS:
        selected = [unit for unit in units if _level_label(float(unit["crowd_aversion"])) == _level_label(level)]
        if not selected:
            report[_level_label(level)] = {"rate": None, "n": 0}
            continue
        hits = sum(1 for unit in selected if unit["low_crowd_chosen"] is True)
        report[_level_label(level)] = {"rate": hits / len(selected), "n": len(selected)}
    return report


def _secondary_rate_by_aversion(units: list[dict[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    report = {}
    for level in CROWD_AVERSION_LEVELS:
        selected = [unit for unit in units if _level_label(float(unit["crowd_aversion"])) == _level_label(level)]
        if not selected:
            report[_level_label(level)] = {"rate": None, "n": 0}
            continue
        hits = sum(1 for unit in selected if unit["secondary_low_crowd_chosen"] is True)
        report[_level_label(level)] = {"rate": hits / len(selected), "n": len(selected)}
    return report


def _mean_by_aversion(units: list[dict[str, Any]], key: str) -> dict[str, dict[str, float | int | None]]:
    report = {}
    for level in CROWD_AVERSION_LEVELS:
        values = [
            float(unit[key])
            for unit in units
            if _level_label(float(unit["crowd_aversion"])) == _level_label(level) and unit.get(key) is not None
        ]
        report[_level_label(level)] = {
            "mean": None if not values else sum(values) / len(values),
            "n": len(values),
        }
    return report


def _tradeoff_table(
    complete_units: list[dict[str, Any]],
    decisive: list[dict[str, Any]],
    key: str,
) -> dict[str, Any]:
    names = sorted({unit[key] for unit in complete_units}, key=lambda value: (isinstance(value, str), value))
    report = {}
    for name in names:
        label = str(name)
        scoped = [unit for unit in complete_units if unit[key] == name]
        decided = [unit for unit in decisive if unit[key] == name]
        report[label] = {
            "utility_advantage_by_aversion": _mean_by_aversion(scoped, "utility_advantage"),
            "low_crowd_choice_rate_by_aversion": _choice_rate_by_aversion(decided),
            "tie_count": sum(1 for unit in scoped if unit["tie"]),
        }
    return report


def _tie_report(complete_units: list[dict[str, Any]], ties: list[dict[str, Any]]) -> dict[str, Any]:
    by_level: dict[str, int] = defaultdict(int)
    by_block: dict[str, int] = defaultdict(int)
    for row in ties:
        by_level[_level_label(float(row["crowd_aversion"]))] += 1
        by_block[str(row["tradeoff_block"])] += 1
    return {
        "tie_count": len(ties),
        "tie_rate": 0.0 if not complete_units else len(ties) / len(complete_units),
        "complete_units": len(complete_units),
        "by_crowd_aversion": dict(sorted(by_level.items())),
        "by_block": dict(sorted(by_block.items())),
        "primary_analysis": "ties are not assigned to either place",
    }


def _architecture_invariance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "label_invariance": 1.0,
        "position_invariance": 1.0,
        "schema_enum_invariance": 1.0,
        "basis": "architecture",
        "interpretation": ARCHITECTURE_NOTE,
        "unique_prompt_count": len({row["prompt_sha256"] for row in rows}),
        "metadata_not_sent": ["label", "position", "schema_enum_order", "scenario_id", "substantive_place_id", "replicate_id"],
    }


def _icc_1_1(pairs: list[tuple[int, int]]) -> float | None:
    count = len(pairs)
    if count < 2:
        return None
    ratings = 2
    grand = sum(value for pair in pairs for value in pair) / (count * ratings)
    means = [sum(pair) / ratings for pair in pairs]
    between = ratings * sum((mean - grand) ** 2 for mean in means) / (count - 1)
    within = sum((value - mean) ** 2 for pair, mean in zip(pairs, means, strict=True) for value in pair) / (count * (ratings - 1))
    denominator = between + (ratings - 1) * within
    if denominator == 0:
        return None
    return (between - within) / denominator


def _correlation(left: list[int], right: list[int], function: Any) -> float | None:
    if len(set(left)) < 2 or len(set(right)) < 2:
        return None
    result = function(left, right)
    value = float(result.statistic)
    if value != value:
        return None
    return value


def _separation(rates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    values = [float(item["mean"] if "mean" in item else item["rate"]) for item in rates.values() if _rate_number(item) is not None]
    if len(values) < 2:
        return {"range": None, "values": values}
    return {"range": max(values) - min(values), "min": min(values), "max": max(values)}


def _rate_number(item: dict[str, Any]) -> float | None:
    raw = item.get("mean", item.get("rate"))
    if raw is None:
        return None
    return float(raw)


def _usable_utility_rate(summary: dict[str, Any]) -> dict[str, Any]:
    rates = summary.get("low_crowd_choice_rate_by_aversion", {})
    decisive = sum(int(item.get("n", 0)) for item in rates.values())
    ties = int(summary.get("ties", {}).get("tie_count", 0))
    total = decisive + ties
    return {
        "decisive_units": decisive,
        "tie_units": ties,
        "decisive_rate": None if total == 0 else decisive / total,
    }


def _direct_choice_rates(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    report = {}
    for level in CROWD_AVERSION_LEVELS:
        selected = [row for row in rows if _level_label(float(row["crowd_aversion"])) == _level_label(level)]
        if not selected:
            report[_level_label(level)] = {"rate": None, "n": 0}
            continue
        hits = sum(int(row["chosen_lower_crowding"]) for row in selected)
        report[_level_label(level)] = {"rate": hits / len(selected), "n": len(selected)}
    return report


def _direct_artifacts(summary: dict[str, Any] | None) -> dict[str, Any]:
    if summary is None:
        return {"label": None, "position": None, "schema_enum_order": None}
    return {
        "label": summary.get("a_label_advantage", {}).get("difference"),
        "position": summary.get("first_position_advantage", {}).get("difference"),
        "schema_enum_order": summary.get("enum_order_advantage", {}).get("difference"),
        "basis": "estimated from direct-choice presentations",
    }


def _direct_complete_rate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"valid_decisions": len(rows), "expected_decisions": 2880, "rate": len(rows) / 2880 if rows else 0.0}


def _config_snapshot(settings: Settings, configuration: dict[str, Any], matrix_sha256: str, design_sha256: str) -> dict[str, Any]:
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "persona_set": "persona_set_v1",
        "source_experiment": "experiment_001d_label_position_schema",
        "prompt_version": PROMPT_VERSION,
        "model_configuration": configuration,
        "crowd_aversion_levels": list(CROWD_AVERSION_LEVELS),
        "base_agent_count": BASE_AGENT_COUNT,
        "replicates": list(REPLICATES),
        "expected_call_count": EXPECTED_CALLS,
        "scenario_matrix_sha256": matrix_sha256,
        "source_design_sha256": design_sha256,
        "tie_break_algorithm": TIED_BREAK_ALGORITHM,
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
        "decision_prompt_version": PROMPT_VERSION,
    }


def _require_gate_settings(settings: Settings) -> None:
    if settings.temperature != 0.0:
        raise RuntimeError("experiment 001E requires temperature 0")
    if settings.reasoning_effort != "none":
        raise RuntimeError("experiment 001E requires reasoning_effort none")
    if settings.llm_seed != 42:
        raise RuntimeError("experiment 001E requires llm seed 42")


def _require_single_place_prompt(messages: list[dict[str, str]]) -> None:
    text = "\n".join(message["content"] for message in messages).lower()
    for banned in _BANNED_PROMPT_TEXT:
        if banned in text:
            raise RuntimeError(f"utility prompt contains forbidden material: {banned}")


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
            "substantive_place_id": task["substantive_place_id"],
            "replicate_id": task["replicate_id"],
            "attempt": attempt,
            "model_configuration": configuration,
            "request_messages": messages,
            "raw_response": raw_response,
            "parsed_ok": parsed_ok,
            "error": error or None,
        },
    )


def _completed_keys(path: Path) -> set[tuple[str, str, str, str, int]]:
    return {_task_key(row) for row in _read_jsonl(path)}


def _task_key(task: dict[str, Any]) -> tuple[str, str, str, str, int]:
    return (
        str(task["base_agent_id"]),
        _level_label(float(task["crowd_aversion"])),
        str(task["scenario_id"]),
        str(task["substantive_place_id"]),
        int(task["replicate_id"]),
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _level_label(value: float) -> str:
    return f"{value:.2f}"


def _dump(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)


def main() -> None:
    run_experiment()


if __name__ == "__main__":
    if __package__ is None:
        raise SystemExit("請在 worldlab 目錄執行：python -m experiments.experiment_001e")
    main()
