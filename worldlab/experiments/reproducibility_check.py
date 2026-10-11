"""Reproducibility Gate 0.

Run persona generation three times with the settings in config/settings.py,
then compare traits and behavioral profiles. This module does not create a
world, a UI, or agent memory.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from agents.persona_generator import generate_traits, write_json
from config.settings import PROJECT_ROOT, Settings, ensure_python, get_settings
from experiments.persona_pilot import run_pilot
from llm.client import LLMClient, StructuredLLMClient
from models.schemas import AgentPersona, BehavioralProfile, RunMetadata

PROFILE_FIELDS: tuple[str, ...] = tuple(BehavioralProfile.model_fields)
FORBIDDEN_DEMOGRAPHIC_KEYS = frozenset(
    {
        "gender",
        "sex",
        "nationality",
        "ethnicity",
        "race",
        "religion",
        "occupation",
        "job",
        "age",
        "name",
    }
)
DEMOGRAPHIC_TERMS = (
    "gender",
    "nationality",
    "ethnicity",
    "religion",
    "occupation",
    "male",
    "female",
)
GATE_REPEATS = 3
REPRODUCIBILITY_ROOT = PROJECT_ROOT / "data" / "reproducibility"


def assert_gate_settings(settings: Settings) -> None:
    problems: list[str] = []
    if settings.temperature != 0.0:
        problems.append("temperature")
    if settings.master_seed != 42 or settings.llm_seed != 42:
        problems.append("seed")
    if settings.reasoning_effort != "none":
        problems.append("reasoning_effort")
    if settings.prompt_version != "persona_behavioral_profile_v1":
        problems.append("prompt_version")
    if settings.pilot_num_agents != 20:
        problems.append("pilot_num_agents")
    if problems:
        raise RuntimeError(f"Reproducibility Gate 0 settings drifted: {problems}")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_run_manifest(run_dir: Path) -> dict[str, Any]:
    metadata = RunMetadata.model_validate_json((run_dir / "metadata.json").read_text(encoding="utf-8"))
    agents = []
    for path in sorted(run_dir.glob("agent_*.json")):
        agents.append(
            {
                "agent_id": path.stem,
                "file_sha256": file_sha256(path),
                "path": path.name,
            }
        )
    manifest = {
        "run_id": metadata.run_id,
        "number_of_agents": len(agents),
        "agents": agents,
    }
    write_json(run_dir / "manifest.json", manifest)
    return manifest


def evaluate_runs(run_dirs: list[Path], *, experiment_id: str) -> dict[str, Any]:
    if len(run_dirs) < 2:
        raise ValueError("reproducibility comparison needs at least two runs")
    loaded = [_load_run(run_dir) for run_dir in run_dirs]
    agent_ids = _shared_agent_ids(loaded)
    violations: list[dict[str, str]] = []
    schema_failure_count = 0
    for run in loaded:
        schema_failure_count += run["schema_failure_count"]
        violations.extend(run["violations"])

    comparison_rows: list[dict[str, Any]] = []
    hash_rows: list[dict[str, str]] = []
    for agent_id in agent_ids:
        records = [run["agents"][agent_id] for run in loaded]
        row = _compare_agent(agent_id, records)
        comparison_rows.append(row)
        for record in records:
            hash_rows.append(
                {
                    "run_id": record["run_id"],
                    "agent_id": agent_id,
                    "file_sha256": record["file_sha256"],
                    "traits_sha256": record["traits_sha256"],
                    "profile_sha256": record["profile_sha256"],
                }
            )

    agent_count = len(comparison_rows)
    trait_matches = sum(1 for row in comparison_rows if row["traits_exact_match"])
    profile_matches = sum(1 for row in comparison_rows if row["profile_exact_match"])
    field_match_rates = {
        field: _rate(sum(1 for row in comparison_rows if row[f"{field}_match"]), agent_count)
        for field in PROFILE_FIELDS
    }
    differing_fields = [field for field, rate in field_match_rates.items() if rate < 1.0]
    summary = {
        "experiment_id": experiment_id,
        "run_ids": [run["run_id"] for run in loaded],
        "settings": _settings_snapshot(loaded),
        "number_of_agents": agent_count,
        "trait_exact_match_rate": _rate(trait_matches, agent_count),
        "full_profile_exact_match_rate": _rate(profile_matches, agent_count),
        "field_match_rates": field_match_rates,
        "differing_fields": differing_fields,
        "agents_with_profile_differences": [
            row["agent_id"] for row in comparison_rows if not row["profile_exact_match"]
        ],
        "schema_failure_count": schema_failure_count,
        "invariant_violation_count": len(violations),
        "violations": violations,
    }
    run_manifest = {
        "experiment_id": experiment_id,
        "runs": [
            {
                "run_id": run["run_id"],
                "run_dir": str(run["run_dir"]),
                "agents": [
                    {"agent_id": item["agent_id"], "file_sha256": item["file_sha256"]}
                    for item in run["manifest"]["agents"]
                ],
            }
            for run in loaded
        ],
    }
    return {
        "summary": summary,
        "comparison_rows": comparison_rows,
        "hash_rows": hash_rows,
        "run_manifest": run_manifest,
    }


def write_report(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", report["summary"])
    write_json(output_dir / "run_manifest.json", report["run_manifest"])
    _write_csv(output_dir / "comparison.csv", report["comparison_rows"])
    _write_csv(output_dir / "hashes.csv", report["hash_rows"])


def run_gate(*, repeats: int = GATE_REPEATS, settings: Settings | None = None, client: LLMClient | None = None) -> Path:
    ensure_python()
    active = get_settings() if settings is None else settings
    assert_gate_settings(active)
    experiment_id = "repro_gate0_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dirs: list[Path] = []
    for index in range(repeats):
        label = f"run {index + 1}/{repeats}"
        print(label, flush=True)
        wrapped = client if client is not None else _ProgressClient(StructuredLLMClient(active), label)
        result = run_pilot(active, wrapped)
        if result.metadata.model_parameters.temperature != 0.0:
            raise RuntimeError("persona run did not record temperature 0")
        manifest = build_run_manifest(result.run_dir)
        print(f"{label} run_id={result.metadata.run_id} agents={len(manifest['agents'])}", flush=True)
        run_dirs.append(result.run_dir)
    report = evaluate_runs(run_dirs, experiment_id=experiment_id)
    output_dir = REPRODUCIBILITY_ROOT / experiment_id
    write_report(report, output_dir)
    summary = report["summary"]
    print(f"experiment_id={experiment_id}", flush=True)
    print(f"trait_exact_match_rate={summary['trait_exact_match_rate']}", flush=True)
    print(f"full_profile_exact_match_rate={summary['full_profile_exact_match_rate']}", flush=True)
    print(f"output={output_dir}", flush=True)
    return output_dir


def main() -> None:
    run_gate()


class _ProgressClient:
    def __init__(self, inner: StructuredLLMClient, label: str) -> None:
        self._inner = inner
        self._label = label
        self._count = 0

    def complete(self, *, messages: list[dict[str, str]], response_model: type[Any]) -> Any:
        self._count += 1
        print(f"{self._label} calling agent {self._count}", flush=True)
        return self._inner.complete(messages=messages, response_model=response_model)


def _load_run(run_dir: Path) -> dict[str, Any]:
    metadata = RunMetadata.model_validate_json((run_dir / "metadata.json").read_text(encoding="utf-8"))
    manifest_path = run_dir / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else build_run_manifest(run_dir)
    )
    agents: dict[str, dict[str, Any]] = {}
    violations: list[dict[str, str]] = []
    schema_failure_count = 0
    for path in sorted(run_dir.glob("agent_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        agent_id = str(payload.get("agent_id", path.stem))
        persona, file_violations, schema_failed = _inspect_persona(payload, metadata.master_seed, metadata.run_id)
        if schema_failed:
            schema_failure_count += 1
        violations.extend(file_violations)
        profile = payload.get("behavioral_profile")
        traits = payload.get("traits")
        agents[agent_id] = {
            "run_id": metadata.run_id,
            "agent_id": agent_id,
            "persona": persona,
            "traits": traits if isinstance(traits, dict) else None,
            "profile": profile if isinstance(profile, dict) else None,
            "file_sha256": file_sha256(path),
            "traits_sha256": canonical_sha256(traits) if isinstance(traits, dict) else "",
            "profile_sha256": canonical_sha256(profile) if isinstance(profile, dict) else "",
        }
    return {
        "run_dir": run_dir,
        "run_id": metadata.run_id,
        "metadata": metadata,
        "manifest": manifest,
        "agents": agents,
        "violations": violations,
        "schema_failure_count": schema_failure_count,
    }


def _inspect_persona(
    payload: dict[str, Any],
    master_seed: int,
    run_id: str,
) -> tuple[AgentPersona | None, list[dict[str, str]], bool]:
    agent_id = str(payload.get("agent_id", ""))
    violations: list[dict[str, str]] = []
    for key_path in _forbidden_key_paths(payload):
        violations.append(
            _violation(run_id, agent_id, "demographic_field", key_path)
        )
    profile = payload.get("behavioral_profile")
    if isinstance(profile, dict):
        for field, text in _string_values(profile):
            for term in _demographic_terms(text):
                violations.append(
                    _violation(run_id, agent_id, "demographic_inference", f"{field}: {term}")
                )
    schema_failed = False
    persona: AgentPersona | None = None
    try:
        persona = AgentPersona.model_validate(payload)
    except ValidationError as exc:
        schema_failed = True
        for error in exc.errors():
            if error.get("type") == "missing":
                location = ".".join(str(part) for part in error.get("loc", ()))
                violations.append(_violation(run_id, agent_id, "missing_field", location))
    if persona is not None:
        expected, trait_seed = generate_traits(master_seed, persona.agent_index)
        if persona.trait_seed != trait_seed or not _same_json(persona.traits.model_dump(), expected.model_dump()):
            violations.append(_violation(run_id, agent_id, "traits_mutated", "traits differ from python rng"))
        for name, value in persona.traits.model_dump().items():
            if not 0.0 <= float(value) <= 1.0:
                violations.append(_violation(run_id, agent_id, "trait_out_of_range", name))
    return persona, violations, schema_failed


def _compare_agent(agent_id: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    traits = [record["traits"] for record in records]
    profiles = [record["profile"] for record in records]
    traits_exact = all(item is not None for item in traits) and all(_same_json(item, traits[0]) for item in traits[1:])
    field_matches = {field: _field_matches(profiles, field) for field in PROFILE_FIELDS}
    profile_exact = all(item is not None for item in profiles) and all(field_matches.values())
    matched = sum(1 for field in PROFILE_FIELDS if field_matches[field])
    row: dict[str, Any] = {
        "agent_id": agent_id,
        "traits_exact_match": traits_exact,
        "profile_exact_match": profile_exact,
        "field_level_match_rate": _rate(matched, len(PROFILE_FIELDS)),
    }
    for field, matched_field in field_matches.items():
        row[f"{field}_match"] = matched_field
    return row


def _field_matches(profiles: list[dict[str, Any] | None], field: str) -> bool:
    if any(not isinstance(profile, dict) or field not in profile for profile in profiles):
        return False
    first = profiles[0]
    assert isinstance(first, dict)
    return all(_same_json(profile[field], first[field]) for profile in profiles[1:] if isinstance(profile, dict))


def _shared_agent_ids(loaded: list[dict[str, Any]]) -> list[str]:
    sets = [set(run["agents"]) for run in loaded]
    if not sets or any(item != sets[0] for item in sets[1:]):
        raise RuntimeError("persona runs do not contain identical agent IDs")
    return sorted(sets[0])


def _settings_snapshot(loaded: list[dict[str, Any]]) -> dict[str, Any]:
    metadata = loaded[0]["metadata"]
    parameters = metadata.model_parameters
    return {
        "model": metadata.model,
        "temperature": parameters.temperature,
        "llm_seed": parameters.seed,
        "reasoning_effort": parameters.reasoning_effort,
        "master_seed": metadata.master_seed,
        "prompt_version": metadata.prompt_version,
    }


def _forbidden_key_paths(node: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key) in FORBIDDEN_DEMOGRAPHIC_KEYS:
                found.append(path)
            found.extend(_forbidden_key_paths(value, path))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(_forbidden_key_paths(value, f"{prefix}[{index}]"))
    return found


def _string_values(node: Any, prefix: str = "") -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if isinstance(node, str):
        found.append((prefix or "<root>", node))
    elif isinstance(node, dict):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            found.extend(_string_values(value, path))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(_string_values(value, f"{prefix}[{index}]"))
    return found


def _demographic_terms(text: str) -> list[str]:
    lowered = text.casefold()
    return [term for term in DEMOGRAPHIC_TERMS if re.search(rf"\b{re.escape(term)}\b", lowered)]


def _violation(run_id: str, agent_id: str, code: str, detail: str) -> dict[str, str]:
    return {"run_id": run_id, "agent_id": agent_id, "code": code, "detail": detail}


def _same_json(left: Any, right: Any) -> bool:
    return canonical_sha256(left) == canonical_sha256(right)


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    if __package__ is None:
        raise SystemExit("請在 worldlab 目錄執行：python -m experiments.reproducibility_check")
    main()
