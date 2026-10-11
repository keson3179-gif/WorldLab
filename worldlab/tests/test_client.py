from __future__ import annotations

import pytest

from config.settings import get_settings
from llm.client import LLMCallError, StructuredLLMClient
from models.schemas import BEHAVIORAL_PROFILE_SCHEMA_NAME, TRAIT_NAMES, BehavioralProfile
from tests.support import VALID_PROFILE_JSON


class _Message:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str | None) -> None:
        self.message = _Message(content)


class _Response:
    def __init__(self, content: str | None) -> None:
        self.choices = [_Choice(content)]


def test_client_sends_only_settings_and_keeps_raw_text(monkeypatch: pytest.MonkeyPatch) -> None:
    sink: dict[str, object] = {"content": VALID_PROFILE_JSON}

    class _Completions:
        def create(self, **kwargs: object) -> _Response:
            sink["kwargs"] = kwargs
            content = sink["content"]
            assert isinstance(content, str) or content is None
            return _Response(content)

    class _Chat:
        def __init__(self) -> None:
            self.completions = _Completions()

    class _FakeOpenAI:
        def __init__(self, *, base_url: str, api_key: str, timeout: float, max_retries: int) -> None:
            sink["init"] = {
                "base_url": base_url,
                "api_key": api_key,
                "timeout": timeout,
                "max_retries": max_retries,
            }
            self.chat = _Chat()

    monkeypatch.setattr("llm.client.OpenAI", _FakeOpenAI)
    settings = get_settings().model_copy(
        update={
            "api_endpoint": "http://127.0.0.1:9/v1",
            "api_key": "test-key",
            "model": "unit-test-model",
            "temperature": 0.31,
            "llm_seed": 123,
            "master_seed": 5,
            "reasoning_effort": "low",
            "request_timeout_seconds": 9.0,
            "sdk_max_retries": 0,
        }
    )
    result = StructuredLLMClient(settings).complete(
        messages=[{"role": "user", "content": "profile"}],
        response_model=BehavioralProfile,
    )

    assert sink["init"] == {
        "base_url": "http://127.0.0.1:9/v1",
        "api_key": "test-key",
        "timeout": 9.0,
        "max_retries": 0,
    }
    kwargs = sink["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["model"] == "unit-test-model"
    assert kwargs["temperature"] == 0.31
    assert kwargs["seed"] == 123
    assert kwargs["reasoning_effort"] == "low"
    assert kwargs["stream"] is False
    assert "tools" not in kwargs
    assert "functions" not in kwargs
    response_format = kwargs["response_format"]
    assert isinstance(response_format, dict)
    assert response_format["type"] == "json_schema"
    json_schema = response_format["json_schema"]
    assert isinstance(json_schema, dict)
    assert json_schema["name"] == BEHAVIORAL_PROFILE_SCHEMA_NAME
    assert json_schema["strict"] is True
    schema = json_schema["schema"]
    assert isinstance(schema, dict)
    properties = schema["properties"]
    assert isinstance(properties, dict)
    assert set(TRAIT_NAMES).isdisjoint(properties)
    assert isinstance(result.parsed, BehavioralProfile)
    assert result.raw_text == VALID_PROFILE_JSON
    assert result.llm_seed == 123
    assert result.model == "unit-test-model"


def test_client_preserves_invalid_raw_text(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = '{"openness": 0.9, "summary": "no"}'

    class _Completions:
        def create(self, **kwargs: object) -> _Response:
            return _Response(raw)

    class _Chat:
        def __init__(self) -> None:
            self.completions = _Completions()

    class _FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            self.chat = _Chat()

    monkeypatch.setattr("llm.client.OpenAI", _FakeOpenAI)
    with pytest.raises(LLMCallError) as caught:
        StructuredLLMClient(get_settings()).complete(
            messages=[{"role": "user", "content": "profile"}],
            response_model=BehavioralProfile,
        )
    assert caught.value.raw_text == raw


def test_client_wraps_transport_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Completions:
        def create(self, **kwargs: object) -> _Response:
            raise RuntimeError("down")

    class _Chat:
        def __init__(self) -> None:
            self.completions = _Completions()

    class _FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            self.chat = _Chat()

    monkeypatch.setattr("llm.client.OpenAI", _FakeOpenAI)
    with pytest.raises(LLMCallError) as caught:
        StructuredLLMClient(get_settings()).complete(
            messages=[{"role": "user", "content": "profile"}],
            response_model=BehavioralProfile,
        )
    assert caught.value.raw_text is None
    assert isinstance(caught.value.__cause__, RuntimeError)
