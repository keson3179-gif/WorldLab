from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.persona_generator import generate_traits
from experiments.reproducibility_check import (
    FORBIDDEN_DEMOGRAPHIC_KEYS,
    PROFILE_FIELDS,
    build_run_manifest,
    evaluate_runs,
    file_sha256,
)
from models.schemas import TRAIT_NAMES, AgentPersona, BehavioralProfile
from tests.support import VALID_PROFILE


def test_fixed_seed_traits_are_reproducible() -> None:
    first, first_seed = generate_traits(42, 0)
    second, second_seed = generate_traits(42, 0)
    assert first_seed == second_seed
    assert first.model_dump() == second.model_dump()


def test_trait_values_are_inside_the_unit_interval() -> None:
    for index in range(20):
        traits, _ = generate_traits(42, index)
        for value in traits.model_dump().values():
            assert 0.0 <= value <= 1.0


def test_identical_agent_ids_and_exact_profile_match(tmp_path: Path) -> None:
    first = _write_run(tmp_path / "run-a", "run-a")
    second = _write_run(tmp_path / "run-b", "run-b")
    third = _write_run(tmp_path / "run-c", "run-c")
    report = evaluate_runs([first, second, third], experiment_id="gate-test")
    summary = report["summary"]
    assert summary["trait_exact_match_rate"] == 1.0
    assert summary["full_profile_exact_match_rate"] == 1.0
    assert summary["schema_failure_count"] == 0
    assert summary["invariant_violation_count"] == 0
    assert [row["agent_id"] for row in report["comparison_rows"]] == ["agent_0000", "agent_0001"]
    assert {row["agent_id"] for row in report["hash_rows"]} == {"agent_0000", "agent_0001"}


def test_schema_validation_counts_a_missing_field(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path / "broken", "broken")
    path = run_dir / "agent_0000.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["behavioral_profile"]["summary"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    other = _write_run(tmp_path / "other", "other")
    report = evaluate_runs([run_dir, other], experiment_id="schema-test")
    assert report["summary"]["schema_failure_count"] == 1
    assert any(item["code"] == "missing_field" for item in report["summary"]["violations"])
    agent = next(row for row in report["comparison_rows"] if row["agent_id"] == "agent_0000")
    assert agent["profile_exact_match"] is False


def test_demographic_fields_are_violations(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path / "demo", "demo")
    path = run_dir / "agent_0001.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["gender"] = "unspecified"
    path.write_text(json.dumps(payload), encoding="utf-8")
    other = _write_run(tmp_path / "clean", "clean")
    report = evaluate_runs([run_dir, other], experiment_id="demographic-test")
    assert any(item["code"] == "demographic_field" for item in report["summary"]["violations"])
    assert FORBIDDEN_DEMOGRAPHIC_KEYS.isdisjoint(BehavioralProfile.model_fields)
    assert FORBIDDEN_DEMOGRAPHIC_KEYS.isdisjoint(TRAIT_NAMES)


def test_llm_cannot_mutate_numerical_traits(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path / "mutated", "mutated")
    path = run_dir / "agent_0000.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["traits"]["openness"] = 0.1
    path.write_text(json.dumps(payload), encoding="utf-8")
    other = _write_run(tmp_path / "intact", "intact")
    report = evaluate_runs([run_dir, other], experiment_id="mutation-test")
    assert any(item["code"] == "traits_mutated" for item in report["summary"]["violations"])
    stored = AgentPersona.model_validate_json((other / "agent_0000.json").read_text(encoding="utf-8"))
    expected, _ = generate_traits(42, 0)
    assert stored.traits.model_dump() == expected.model_dump()


def test_field_level_match_rate_counts_only_changed_fields(tmp_path: Path) -> None:
    first = _write_run(tmp_path / "same-a", "same-a")
    second = _write_run(tmp_path / "same-b", "same-b")
    changed = _write_run(tmp_path / "changed", "changed", summary="A different summary.")
    report = evaluate_runs([first, second, changed], experiment_id="field-test")
    agent = report["comparison_rows"][0]
    assert agent["summary_match"] is False
    assert agent["planning_style_match"] is True
    assert agent["field_level_match_rate"] == pytest.approx((len(PROFILE_FIELDS) - 1) / len(PROFILE_FIELDS))
    assert report["summary"]["differing_fields"] == ["summary"]
    assert report["summary"]["field_match_rates"]["summary"] == 0.5


def test_manifest_records_persona_file_hash(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path / "hashed", "hashed")
    manifest = build_run_manifest(run_dir)
    recorded = next(item for item in manifest["agents"] if item["agent_id"] == "agent_0000")
    assert recorded["file_sha256"] == file_sha256(run_dir / "agent_0000.json")


def _write_run(run_dir: Path, run_id: str, *, summary: str | None = None) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "run_id": run_id,
        "timestamp": "2026-09-26T00:00:00+00:00",
        "status": "completed",
        "model": "qwen3.5:9b",
        "model_parameters": {"temperature": 0.0, "seed": 42, "reasoning_effort": "none"},
        "api_endpoint": "http://127.0.0.1:9/v1",
        "master_seed": 42,
        "prompt_version": "persona_behavioral_profile_v1",
        "prompt_sha256": "abc",
        "response_schema_sha256": "def",
        "number_of_agents": 2,
        "completed_agents": 2,
        "trait_algorithm": "worldlab.traits.v1",
        "trait_rng": "random.Random",
        "python_version": "3.14.3",
        "error": None,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    for index in range(2):
        traits, trait_seed = generate_traits(42, index)
        profile = dict(VALID_PROFILE)
        if summary is not None and index == 0:
            profile["summary"] = summary
        persona = {
            "agent_id": f"agent_{index:04d}",
            "agent_index": index,
            "run_id": run_id,
            "trait_source": "python_rng",
            "trait_seed": trait_seed,
            "traits": traits.model_dump(),
            "profile_source": "llm",
            "behavioral_profile": profile,
            "prompt_version": "persona_behavioral_profile_v1",
        }
        (run_dir / f"agent_{index:04d}.json").write_text(json.dumps(persona), encoding="utf-8")
    return run_dir
