from __future__ import annotations

import pytest
from pydantic import ValidationError

from config.settings import Settings, get_settings


def test_phase1_defaults() -> None:
    settings = get_settings()
    assert settings.api_endpoint == "http://localhost:11434/v1"
    assert settings.api_key == "ollama"
    assert settings.model == "qwen3.5:9b"
    assert settings.temperature == 0.0
    assert settings.reasoning_effort == "none"
    assert settings.llm_seed == 42
    assert settings.master_seed == 42
    assert settings.pilot_num_agents == 20
    assert settings.sdk_max_retries == 0
    assert settings.prompt_version == "persona_behavioral_profile_v1"
    assert settings.profile_language == "English"


def test_settings_ignore_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://example.invalid/v1")
    monkeypatch.setenv("WORLD_MODEL", "other-model")
    settings = get_settings()
    assert settings.api_key == "ollama"
    assert settings.api_endpoint == "http://localhost:11434/v1"
    assert settings.model == "qwen3.5:9b"


def test_endpoint_is_normalized_and_must_be_openai_compatible() -> None:
    settings = Settings(api_endpoint="http://localhost:11434/v1/")
    assert settings.api_endpoint == "http://localhost:11434/v1"
    with pytest.raises(ValidationError):
        Settings(api_endpoint="http://localhost:11434")


def test_model_name_strips_surrounding_space() -> None:
    assert Settings(model="qwen3.5:9b ").model == "qwen3.5:9b"
    with pytest.raises(ValidationError):
        Settings(model="  ")
    with pytest.raises(ValidationError):
        get_settings().model_copy(update={"pilot_num_agents": 0})
    with pytest.raises(ValidationError):
        Settings(temperature=2.5)
    with pytest.raises(ValidationError):
        Settings(extra_switch=True)  # type: ignore[call-arg]
