"""Chat client for a local OpenAI-compatible server.

Sampling parameters and the endpoint come from Settings. The client sends a
Pydantic JSON Schema and validates the raw text with that same model. It does
not pass tools, and it does not execute model output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from config.settings import Settings, ensure_python
from models.schemas import BEHAVIORAL_PROFILE_SCHEMA_NAME, export_response_schema


class LLMCallError(RuntimeError):
    def __init__(self, message: str, *, raw_text: str | None) -> None:
        super().__init__(message)
        self.raw_text = raw_text


@dataclass(frozen=True)
class LLMCallResult:
    raw_text: str
    parsed: BaseModel
    model: str
    temperature: float
    llm_seed: int
    reasoning_effort: str
    api_endpoint: str


class LLMClient(Protocol):
    def complete(
        self,
        *,
        messages: list[dict[str, str]],
        response_model: type[BaseModel],
    ) -> LLMCallResult:
        """Return a schema-valid result, or raise LLMCallError with raw text."""


class StructuredLLMClient:
    def __init__(self, settings: Settings) -> None:
        ensure_python()
        self._settings = settings
        sdk = OpenAI(
            base_url=settings.api_endpoint,
            api_key=settings.api_key,
            timeout=settings.request_timeout_seconds,
            max_retries=settings.sdk_max_retries,
        )
        self._create = sdk.chat.completions.create

    def complete(
        self,
        *,
        messages: list[dict[str, str]],
        response_model: type[BaseModel],
    ) -> LLMCallResult:
        model = self._settings.model
        temperature = self._settings.temperature
        llm_seed = self._settings.llm_seed
        reasoning_effort = self._settings.reasoning_effort
        api_endpoint = self._settings.api_endpoint
        kwargs: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "seed": llm_seed,
            "reasoning_effort": reasoning_effort,
            "stream": False,
            "messages": messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": getattr(
                        response_model,
                        "response_schema_name",
                        BEHAVIORAL_PROFILE_SCHEMA_NAME,
                    ),
                    "schema": export_response_schema(response_model),
                    "strict": True,
                },
            },
        }
        try:
            response = self._create(**kwargs)
            content = response.choices[0].message.content
        except Exception as exc:
            raise LLMCallError(f"{type(exc).__name__}: {exc}", raw_text=None) from exc

        if not isinstance(content, str) or content.strip() == "":
            raw_text = content if isinstance(content, str) else None
            raise LLMCallError(
                "model returned empty content. Thinking models leave content empty "
                "unless reasoning_effort is none in config/settings.py.",
                raw_text=raw_text,
            )

        try:
            parsed = response_model.model_validate_json(content)
        except ValidationError as exc:
            raise LLMCallError(str(exc), raw_text=content) from exc

        return LLMCallResult(
            raw_text=content,
            parsed=parsed,
            model=model,
            temperature=temperature,
            llm_seed=llm_seed,
            reasoning_effort=reasoning_effort,
            api_endpoint=api_endpoint,
        )
