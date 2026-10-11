"""Pydantic schemas for traits, profiles, run metadata, and audit records.

BehavioralProfile is the persona schema. It has no trait scores and no
demographic identity fields. Changing it requires a new prompt_version.
PlaceChoiceDecision is the decision schema and contains no profile prose.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AgentTraits(BaseModel):
    """Host-assigned unit-interval traits. Field order is part of the RNG contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    openness: float = Field(ge=0.0, le=1.0)
    conscientiousness: float = Field(ge=0.0, le=1.0)
    extraversion: float = Field(ge=0.0, le=1.0)
    agreeableness: float = Field(ge=0.0, le=1.0)
    neuroticism: float = Field(ge=0.0, le=1.0)
    novelty_seeking: float = Field(ge=0.0, le=1.0)
    crowd_aversion: float = Field(ge=0.0, le=1.0)
    social_orientation: float = Field(ge=0.0, le=1.0)
    photography_interest: float = Field(ge=0.0, le=1.0)
    risk_tolerance: float = Field(ge=0.0, le=1.0)

    @field_validator("*")
    @classmethod
    def _finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("trait must be finite")
        return value


TRAIT_NAMES: tuple[str, ...] = tuple(AgentTraits.model_fields)


class BehavioralProfile(BaseModel):
    """Behavioral tendencies conditioned on fixed traits.

    No trait scores and no demographic identity fields.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    summary: str = Field(min_length=1)
    planning_style: str = Field(min_length=1)
    social_behavior: str = Field(min_length=1)
    exploration_behavior: str = Field(min_length=1)
    crowd_response: str = Field(min_length=1)
    photography_behavior: str = Field(min_length=1)
    risk_behavior: str = Field(min_length=1)
    priorities: list[str] = Field(min_length=2, max_length=4)
    behavioral_constraints: list[str] = Field(min_length=2, max_length=4)

    @field_validator("priorities", "behavioral_constraints")
    @classmethod
    def _nonblank_items(cls, values: list[str]) -> list[str]:
        cleaned = [item.strip() for item in values]
        if any(not item for item in cleaned):
            raise ValueError("list items must not be blank")
        return cleaned


BEHAVIORAL_PROFILE_SCHEMA_NAME = "behavioral_profile"


class AgentPersona(BaseModel):
    """One saved agent. Traits and the profile come from different sources."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_id: str
    agent_index: int = Field(ge=0)
    run_id: str
    trait_source: Literal["python_rng"]
    trait_seed: int = Field(ge=0)
    traits: AgentTraits
    profile_source: Literal["llm"]
    behavioral_profile: BehavioralProfile
    prompt_version: str


class ModelParameters(BaseModel):
    """Sampling parameters actually sent on the OpenAI-compatible request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    temperature: float = Field(ge=0.0)
    seed: int = Field(ge=0)
    reasoning_effort: Literal["none", "low", "medium", "high"]


class RunMetadata(BaseModel):
    """One record per persona run. Required research fields live here."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    timestamp: str
    status: Literal["completed", "failed"]
    model: str
    model_parameters: ModelParameters
    api_endpoint: str
    master_seed: int = Field(ge=0)
    prompt_version: str
    prompt_sha256: str
    response_schema_sha256: str
    number_of_agents: int = Field(ge=1)
    completed_agents: int = Field(ge=0)
    trait_algorithm: str
    trait_rng: Literal["random.Random"]
    python_version: str
    error: str | None = None

    @model_validator(mode="after")
    def _status_matches_progress(self) -> Self:
        if self.completed_agents > self.number_of_agents:
            raise ValueError("completed_agents exceeds number_of_agents")
        if self.status == "completed" and self.completed_agents != self.number_of_agents:
            raise ValueError("completed run must finish every agent")
        if self.status == "completed" and self.error is not None:
            raise ValueError("completed run cannot carry an error")
        if self.status == "failed" and not self.error:
            raise ValueError("failed run must record an error")
        return self


class PlaceAttributes(BaseModel):
    """Numeric attributes of one place. Higher crowding means a denser place."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    crowding: float = Field(ge=0.0, le=1.0)
    visual_attractiveness: float = Field(ge=0.0, le=1.0)
    distance: float = Field(ge=0.0, le=1.0)
    novelty: float = Field(ge=0.0, le=1.0)


class PlaceChoiceDecision(BaseModel):
    """LLM choice. reason_codes are an audit trail, not the study outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    response_schema_name: ClassVar[str] = "place_choice"

    choice: Literal["A", "B"]
    reason_codes: list[
        Literal["crowding", "visual_attractiveness", "distance", "novelty", "other"]
    ] = Field(min_length=1, max_length=5)

    @field_validator("reason_codes")
    @classmethod
    def _unique_codes(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("reason_codes must not contain duplicates")
        return values


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["system", "user"]
    content: str = Field(min_length=1)


class LLMAuditRecord(BaseModel):
    """One JSONL line. This is the raw model call, not the persona file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    timestamp: str
    run_id: str
    agent_id: str
    agent_index: int = Field(ge=0)
    trait_seed: int = Field(ge=0)
    traits: AgentTraits
    model: str
    model_parameters: ModelParameters
    api_endpoint: str
    master_seed: int = Field(ge=0)
    prompt_version: str
    request_messages: list[ChatMessage]
    raw_response: str | None
    parsed_ok: bool
    error: str | None = None


def export_response_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Flat JSON Schema for the OpenAI-compatible response_format."""

    raw = model.model_json_schema()
    if _contains_key(raw, "$ref") or _contains_key(raw, "$defs"):
        raise ValueError("phase-1 response schemas must be flat JSON Schema without $ref")
    trimmed = _strip_key(_strip_key(raw, "title"), "$schema")
    if not isinstance(trimmed, dict):
        raise ValueError("response schema must be an object")
    return trimmed


def response_schema_sha256(model: type[BaseModel]) -> str:
    payload = json.dumps(
        export_response_schema(model),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _contains_key(node: Any, key: str) -> bool:
    if isinstance(node, dict):
        if key in node:
            return True
        return any(_contains_key(value, key) for value in node.values())
    if isinstance(node, list):
        return any(_contains_key(value, key) for value in node)
    return False


def _strip_key(node: Any, key: str) -> Any:
    if isinstance(node, dict):
        return {name: _strip_key(value, key) for name, value in node.items() if name != key}
    if isinstance(node, list):
        return [_strip_key(value, key) for value in node]
    return node
