"""Shared fixtures for persona tests. The scripted client never calls a server."""

from __future__ import annotations

import json

from pydantic import BaseModel

from config.settings import Settings
from llm.client import LLMCallError, LLMCallResult
from models.schemas import TRAIT_NAMES

VALID_PROFILE: dict[str, object] = {
    "summary": "Keeps a short plan, samples a few new options, and steps away from dense groups.",
    "planning_style": "Makes a short ordered plan and changes it only when a stop is blocked.",
    "social_behavior": "Responds when addressed and does not try to gather other people.",
    "exploration_behavior": "Tries one unfamiliar option, then returns to the planned route.",
    "crowd_response": "Slows or leaves when a place becomes densely packed.",
    "photography_behavior": "Stops only when a scene is clear, then moves on.",
    "risk_behavior": "Declines uncertain shortcuts and stays with visible paths.",
    "priorities": ["keep the plan readable", "limit time in dense places"],
    "behavioral_constraints": ["do not enter a packed space", "do not drop the plan for an impulse"],
}

VALID_PROFILE_JSON = json.dumps(VALID_PROFILE)


def trait_values(**overrides: float) -> dict[str, float]:
    values = {name: 0.5 for name in TRAIT_NAMES}
    values.update(overrides)
    return values


class ScriptedLLM:
    def __init__(
        self,
        settings: Settings,
        raw_text: str,
        *,
        fail_at: int | None = None,
        failure_text: str | None = None,
        drift_temperature: bool = False,
    ) -> None:
        self._settings = settings
        self._raw_text = raw_text
        self._failure_text = failure_text
        self._fail_at = fail_at
        self._drift_temperature = drift_temperature
        self.calls = 0
        self.messages: list[list[dict[str, str]]] = []
        self.models: list[type[BaseModel]] = []

    def complete(
        self,
        *,
        messages: list[dict[str, str]],
        response_model: type[BaseModel],
    ) -> LLMCallResult:
        index = self.calls
        self.calls += 1
        self.messages.append(messages)
        self.models.append(response_model)
        if self._fail_at is not None and index == self._fail_at:
            raw_text = self._failure_text if self._failure_text is not None else self._raw_text
            raise LLMCallError("scripted schema failure", raw_text=raw_text)
        parsed = response_model.model_validate_json(self._raw_text)
        temperature = self._settings.temperature + (1.0 if self._drift_temperature else 0.0)
        return LLMCallResult(
            raw_text=self._raw_text,
            parsed=parsed,
            model=self._settings.model,
            temperature=temperature,
            llm_seed=self._settings.llm_seed,
            reasoning_effort=self._settings.reasoning_effort,
            api_endpoint=self._settings.api_endpoint,
        )
