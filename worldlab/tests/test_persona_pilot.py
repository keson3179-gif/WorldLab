from __future__ import annotations

import inspect
import json
from datetime import datetime, timezone

import pytest

from agents.persona_generator import (
    PersonaGenerator,
    generate_traits,
    make_run_id,
)
from experiments.persona_pilot import run_pilot
from models.schemas import TRAIT_NAMES, BehavioralProfile
from tests.support import VALID_PROFILE, VALID_PROFILE_JSON, ScriptedLLM


WHEN = datetime(2026, 9, 26, 1, 2, 3, tzinfo=timezone.utc)


def test_pilot_source_reads_the_configured_count() -> None:
    source = inspect.getsource(run_pilot)
    assert "pilot_num_agents" in source
    assert "num_agents=20" not in source
    assert "range(20)" not in source


def test_pilot_writes_twenty_agents(tmp_path) -> None:
    settings = _settings(tmp_path)
    client = ScriptedLLM(settings, VALID_PROFILE_JSON)
    result = run_pilot(settings, client, now=WHEN)

    assert settings.pilot_num_agents == 20
    assert result.metadata.status == "completed"
    assert result.metadata.number_of_agents == 20
    assert result.metadata.completed_agents == 20
    assert result.metadata.timestamp == "2026-09-26T01:02:03+00:00"
    assert result.metadata.run_id == "persona_20260926T010203000000Z_seed99"
    assert result.metadata.run_id == make_run_id(WHEN, 99)
    assert result.metadata.model == "unit-test-model"
    assert result.metadata.model_parameters.temperature == 0.3
    assert result.metadata.model_parameters.seed == 7
    assert result.metadata.master_seed == 99
    assert result.metadata.api_endpoint == "http://127.0.0.1:9/v1"
    assert result.metadata.prompt_version == "persona_behavioral_profile_v1"
    assert result.metadata.error is None
    assert len(result.metadata.prompt_sha256) == 64
    assert len(result.metadata.response_schema_sha256) == 64

    names = sorted(path.name for path in result.run_dir.iterdir())
    assert names == [f"agent_{index:04d}.json" for index in range(20)] + ["metadata.json"]
    metadata_on_disk = json.loads((result.run_dir / "metadata.json").read_text(encoding="utf-8"))
    for key in (
        "timestamp",
        "model",
        "model_parameters",
        "master_seed",
        "prompt_version",
        "number_of_agents",
    ):
        assert key in metadata_on_disk

    lines = result.audit_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 20
    assert client.models == [BehavioralProfile] * 20
    for index, line in enumerate(lines):
        expected, trait_seed = generate_traits(99, index)
        persona = json.loads((result.run_dir / f"agent_{index:04d}.json").read_text(encoding="utf-8"))
        audit = json.loads(line)
        assert persona["traits"] == expected.model_dump()
        assert persona["trait_seed"] == trait_seed
        assert persona["trait_source"] == "python_rng"
        assert persona["profile_source"] == "llm"
        assert persona["behavioral_profile"] == VALID_PROFILE
        assert audit["traits"] == expected.model_dump()
        assert audit["raw_response"] == VALID_PROFILE_JSON
        assert audit["parsed_ok"] is True
        assert audit["error"] is None
        assert audit["model_parameters"] == {
            "temperature": 0.3,
            "seed": 7,
            "reasoning_effort": "none",
        }
        assert audit["master_seed"] == 99
        assert audit["timestamp"] == "2026-09-26T01:02:03+00:00"
        rendered = json.dumps(
            {name: getattr(expected, name) for name in TRAIT_NAMES},
            ensure_ascii=False,
            indent=2,
        )
        assert rendered in client.messages[index][1]["content"]


def test_failed_call_keeps_python_traits_in_the_audit(tmp_path) -> None:
    settings = _settings(tmp_path)
    bad_raw = '{"gender": "unspecified"}'
    client = ScriptedLLM(settings, VALID_PROFILE_JSON, fail_at=1, failure_text=bad_raw)
    generator = PersonaGenerator(settings, client)
    when = datetime(2026, 9, 26, 2, 0, tzinfo=timezone.utc)
    with pytest.raises(Exception, match="scripted schema failure"):
        generator.generate_run(2, now=when)

    run_dir = settings.personas_dir / make_run_id(when, settings.master_seed)
    assert (run_dir / "agent_0000.json").exists()
    assert not (run_dir / "agent_0001.json").exists()
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "failed"
    assert metadata["completed_agents"] == 1
    assert metadata["number_of_agents"] == 2
    assert metadata["error"]

    lines = (settings.logs_dir / f"{metadata['run_id']}.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    failed = json.loads(lines[1])
    expected, trait_seed = generate_traits(settings.master_seed, 1)
    assert failed["parsed_ok"] is False
    assert failed["raw_response"] == bad_raw
    assert failed["traits"] == expected.model_dump()
    assert failed["trait_seed"] == trait_seed


def test_parameter_drift_does_not_save_a_persona(tmp_path) -> None:
    settings = _settings(tmp_path)
    client = ScriptedLLM(settings, VALID_PROFILE_JSON, drift_temperature=True)
    generator = PersonaGenerator(settings, client)
    when = datetime(2026, 9, 26, 3, 0, tzinfo=timezone.utc)
    with pytest.raises(RuntimeError, match="do not match settings"):
        generator.generate_run(1, now=when)

    run_dir = settings.personas_dir / make_run_id(when, settings.master_seed)
    assert list(run_dir.glob("agent_*.json")) == []
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "failed"
    assert metadata["completed_agents"] == 0
    audit = json.loads((settings.logs_dir / f"{metadata['run_id']}.jsonl").read_text(encoding="utf-8"))
    assert audit["parsed_ok"] is True
    assert audit["error"]
    assert audit["raw_response"] == VALID_PROFILE_JSON
    assert audit["model_parameters"]["temperature"] == settings.temperature + 1.0


def test_naive_clock_writes_nothing(tmp_path) -> None:
    settings = _settings(tmp_path)
    generator = PersonaGenerator(settings, ScriptedLLM(settings, VALID_PROFILE_JSON))
    with pytest.raises(ValueError, match="timezone-aware"):
        generator.generate_run(1, now=datetime(2026, 9, 26))
    assert list(tmp_path.rglob("*")) == []


def _settings(tmp_path):
    from config.settings import get_settings

    return get_settings().model_copy(
        update={
            "personas_dir": tmp_path / "personas",
            "logs_dir": tmp_path / "logs",
            "model": "unit-test-model",
            "temperature": 0.3,
            "llm_seed": 7,
            "master_seed": 99,
            "api_endpoint": "http://127.0.0.1:9/v1",
        }
    )
