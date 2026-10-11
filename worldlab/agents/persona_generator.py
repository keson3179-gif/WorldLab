"""Batch persona writer.

Traits are drawn here with random.Random. The LLM only fills a behavioral
profile and cannot change those traits. This module does not run an agent loop,
does not call tools, and does not execute model output.
"""

from __future__ import annotations

import hashlib
import json
import platform
import random
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from config.settings import Settings, ensure_python
from llm.client import LLMCallResult, LLMClient
from models.schemas import (
    TRAIT_NAMES,
    AgentPersona,
    AgentTraits,
    BehavioralProfile,
    ChatMessage,
    LLMAuditRecord,
    ModelParameters,
    RunMetadata,
    response_schema_sha256,
)

TRAIT_ALGORITHM = "worldlab.traits.v1"
TRAIT_RNG = "random.Random"

TRAIT_DEFINITIONS: dict[str, str] = {
    "openness": "preference for unfamiliar ideas and varied experience",
    "conscientiousness": "preference for planning and follow-through",
    "extraversion": "preference for stimulation from social contact",
    "agreeableness": "preference for cooperation rather than confrontation",
    "neuroticism": "tendency toward stress reactivity",
    "novelty_seeking": "tendency to choose unfamiliar options",
    "crowd_aversion": "tendency to avoid dense crowds",
    "social_orientation": "tendency to organize activity around other people",
    "photography_interest": "tendency to stop and photograph a scene",
    "risk_tolerance": "willingness to accept uncertain outcomes",
}

USER_PREFIX = (
    "The host program assigned these traits. "
    "The scale runs from 0 (low) to 1 (high). "
    "They are immutable."
)
USER_SUFFIX = "Write the behavioral profile now."


def _check_trait_definitions() -> None:
    if tuple(TRAIT_DEFINITIONS) != TRAIT_NAMES:
        raise RuntimeError("trait definition order drifted from AgentTraits")


_check_trait_definitions()


def derive_trait_seed(master_seed: int, agent_index: int) -> int:
    if master_seed < 0:
        raise ValueError("master_seed must be >= 0")
    if agent_index < 0:
        raise ValueError("agent_index must be >= 0")
    material = f"{TRAIT_ALGORITHM}|{master_seed}|{agent_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def generate_traits(master_seed: int, agent_index: int) -> tuple[AgentTraits, int]:
    """Draw traits from Python only. The signature has no model and no profile."""

    trait_seed = derive_trait_seed(master_seed, agent_index)
    rng = random.Random(trait_seed)
    values = {name: rng.random() for name in TRAIT_NAMES}
    return AgentTraits.model_validate(values), trait_seed


def render_system_prompt(profile_language: str) -> str:
    return (
        "You fill in a behavioral profile form for a research simulation.\n"
        "You are not a participant and you do not act inside the simulation.\n"
        "You cannot run code, call tools, or change the host program.\n"
        "Do not emit Python or any other executable code.\n"
        "\n"
        "Numerical traits are inputs that the host program has already assigned.\n"
        "They are immutable. Do not edit, replace, rescale, or add scores.\n"
        "Do not put trait names or trait numbers into output fields as data.\n"
        "Describe tendencies in words.\n"
        "\n"
        "Do not assign or imply a personal name, gender, nationality, ethnicity, "
        "religion, occupation, or age.\n"
        "Do not invent a life history that depends on those categories.\n"
        "Write only observable behavioral tendencies that follow from the given traits.\n"
        "\n"
        f"Write in {profile_language}.\n"
        "Return JSON only, matching the schema. No markdown.\n"
    )


def trait_definition_block() -> str:
    return "\n".join(f"- {name}: {TRAIT_DEFINITIONS[name]}" for name in TRAIT_NAMES)


def compute_prompt_sha256(prompt_version: str, profile_language: str) -> str:
    material = "\n\n".join(
        [
            render_system_prompt(profile_language),
            prompt_version,
            USER_PREFIX,
            trait_definition_block(),
            USER_SUFFIX,
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def build_messages(
    traits: AgentTraits,
    *,
    prompt_version: str,
    profile_language: str,
) -> list[dict[str, str]]:
    traits_json = json.dumps(
        {name: getattr(traits, name) for name in TRAIT_NAMES},
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )
    user = (
        f"prompt_version: {prompt_version}\n\n"
        f"{USER_PREFIX}\n\n"
        "Trait definitions:\n"
        f"{trait_definition_block()}\n\n"
        "TRAITS_JSON\n"
        f"{traits_json}\n"
        "END_TRAITS_JSON\n\n"
        f"{USER_SUFFIX}\n"
    )
    return [
        {"role": "system", "content": render_system_prompt(profile_language)},
        {"role": "user", "content": user},
    ]


def make_run_id(when: datetime, master_seed: int) -> str:
    stamp = ensure_utc(when).strftime("%Y%m%dT%H%M%S%fZ")
    return f"persona_{stamp}_seed{master_seed}"


def ensure_utc(when: datetime) -> datetime:
    if when.tzinfo is None or when.tzinfo.utcoffset(when) is None:
        raise ValueError("timestamp must be timezone-aware")
    return when.astimezone(timezone.utc)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")


class PersonaRunResult:
    def __init__(self, metadata: RunMetadata, run_dir: Path, audit_path: Path) -> None:
        self.metadata = metadata
        self.run_dir = run_dir
        self.audit_path = audit_path


class PersonaGenerator:
    """Write a fixed number of personas. The host loop is not an acting agent."""

    def __init__(self, settings: Settings, client: LLMClient) -> None:
        ensure_python()
        self._settings = settings
        self._client = client

    def generate_run(self, num_agents: int, *, now: datetime | None = None) -> PersonaRunResult:
        if num_agents < 1:
            raise ValueError("num_agents must be >= 1")

        when = ensure_utc(now or datetime.now(timezone.utc))
        run_id = make_run_id(when, self._settings.master_seed)
        run_dir = self._settings.personas_dir / run_id
        audit_path = self._settings.logs_dir / f"{run_id}.jsonl"
        run_dir.mkdir(parents=True, exist_ok=True)
        audit_path.parent.mkdir(parents=True, exist_ok=True)

        completed = 0
        try:
            for agent_index in range(num_agents):
                self._generate_one(
                    agent_index=agent_index,
                    run_id=run_id,
                    run_dir=run_dir,
                    audit_path=audit_path,
                    timestamp=when.isoformat(),
                )
                completed += 1
        except Exception as exc:
            metadata = self._metadata(
                run_id=run_id,
                timestamp=when.isoformat(),
                status="failed",
                completed_agents=completed,
                number_of_agents=num_agents,
                error=f"{type(exc).__name__}: {exc}",
            )
            self._write_metadata(run_dir, metadata)
            raise

        metadata = self._metadata(
            run_id=run_id,
            timestamp=when.isoformat(),
            status="completed",
            completed_agents=completed,
            number_of_agents=num_agents,
            error=None,
        )
        self._write_metadata(run_dir, metadata)
        return PersonaRunResult(metadata=metadata, run_dir=run_dir, audit_path=audit_path)

    def _generate_one(
        self,
        *,
        agent_index: int,
        run_id: str,
        run_dir: Path,
        audit_path: Path,
        timestamp: str,
    ) -> None:
        traits, trait_seed = generate_traits(self._settings.master_seed, agent_index)
        trait_snapshot = traits.model_dump()
        messages = build_messages(
            traits,
            prompt_version=self._settings.prompt_version,
            profile_language=self._settings.profile_language,
        )
        validated_messages = [ChatMessage.model_validate(message) for message in messages]
        request_messages = [message.model_dump() for message in validated_messages]

        result: LLMCallResult | None = None
        try:
            result = self._client.complete(
                messages=request_messages,
                response_model=BehavioralProfile,
            )
            self._assert_parameters(result)
            persona = self._assemble(
                agent_index=agent_index,
                run_id=run_id,
                traits=traits,
                trait_seed=trait_seed,
                trait_snapshot=trait_snapshot,
                parsed=result.parsed,
            )
        except Exception as exc:
            raw_text = result.raw_text if result is not None else getattr(exc, "raw_text", None)
            self._audit(
                audit_path=audit_path,
                timestamp=timestamp,
                run_id=run_id,
                agent_index=agent_index,
                traits=traits,
                trait_seed=trait_seed,
                result=result,
                raw_response=raw_text if isinstance(raw_text, str) or raw_text is None else None,
                parsed_ok=result is not None,
                error=f"{type(exc).__name__}: {exc}",
                request_messages=validated_messages,
            )
            raise

        self._audit(
            audit_path=audit_path,
            timestamp=timestamp,
            run_id=run_id,
            agent_index=agent_index,
            traits=traits,
            trait_seed=trait_seed,
            result=result,
            raw_response=result.raw_text,
            parsed_ok=True,
            error=None,
            request_messages=validated_messages,
        )
        write_json(run_dir / f"{persona.agent_id}.json", persona.model_dump(mode="json"))

    def _assert_parameters(self, result: LLMCallResult) -> None:
        expected = {
            "model": self._settings.model,
            "temperature": self._settings.temperature,
            "llm_seed": self._settings.llm_seed,
            "reasoning_effort": self._settings.reasoning_effort,
            "api_endpoint": self._settings.api_endpoint,
        }
        actual = {
            "model": result.model,
            "temperature": result.temperature,
            "llm_seed": result.llm_seed,
            "reasoning_effort": result.reasoning_effort,
            "api_endpoint": result.api_endpoint,
        }
        if actual != expected:
            raise RuntimeError(f"LLM parameters do not match settings: {actual}")

    def _assemble(
        self,
        *,
        agent_index: int,
        run_id: str,
        traits: AgentTraits,
        trait_seed: int,
        trait_snapshot: dict[str, Any],
        parsed: BaseModel,
    ) -> AgentPersona:
        if not isinstance(parsed, BehavioralProfile):
            raise TypeError("LLM output was not a BehavioralProfile")
        persona = AgentPersona(
            agent_id=f"agent_{agent_index:04d}",
            agent_index=agent_index,
            run_id=run_id,
            trait_source="python_rng",
            trait_seed=trait_seed,
            traits=traits,
            profile_source="llm",
            behavioral_profile=parsed,
            prompt_version=self._settings.prompt_version,
        )
        if persona.traits.model_dump() != trait_snapshot:
            raise RuntimeError("traits changed after LLM profile generation")
        return persona

    def _audit(
        self,
        *,
        audit_path: Path,
        timestamp: str,
        run_id: str,
        agent_index: int,
        traits: AgentTraits,
        trait_seed: int,
        result: LLMCallResult | None,
        raw_response: str | None,
        parsed_ok: bool,
        error: str | None,
        request_messages: list[ChatMessage],
    ) -> None:
        if result is None:
            model = self._settings.model
            temperature = self._settings.temperature
            llm_seed = self._settings.llm_seed
            reasoning_effort = self._settings.reasoning_effort
            api_endpoint = self._settings.api_endpoint
        else:
            model = result.model
            temperature = result.temperature
            llm_seed = result.llm_seed
            reasoning_effort = result.reasoning_effort
            api_endpoint = result.api_endpoint
        record = LLMAuditRecord(
            timestamp=timestamp,
            run_id=run_id,
            agent_id=f"agent_{agent_index:04d}",
            agent_index=agent_index,
            trait_seed=trait_seed,
            traits=traits,
            model=model,
            model_parameters=ModelParameters(
                temperature=temperature,
                seed=llm_seed,
                reasoning_effort=reasoning_effort,
            ),
            api_endpoint=api_endpoint,
            master_seed=self._settings.master_seed,
            prompt_version=self._settings.prompt_version,
            request_messages=request_messages,
            raw_response=raw_response,
            parsed_ok=parsed_ok,
            error=error,
        )
        append_jsonl(audit_path, record.model_dump(mode="json"))

    def _metadata(
        self,
        *,
        run_id: str,
        timestamp: str,
        status: Literal["completed", "failed"],
        completed_agents: int,
        number_of_agents: int,
        error: str | None,
    ) -> RunMetadata:
        return RunMetadata(
            run_id=run_id,
            timestamp=timestamp,
            status=status,
            model=self._settings.model,
            model_parameters=ModelParameters(
                temperature=self._settings.temperature,
                seed=self._settings.llm_seed,
                reasoning_effort=self._settings.reasoning_effort,
            ),
            api_endpoint=self._settings.api_endpoint,
            master_seed=self._settings.master_seed,
            prompt_version=self._settings.prompt_version,
            prompt_sha256=compute_prompt_sha256(
                self._settings.prompt_version,
                self._settings.profile_language,
            ),
            response_schema_sha256=response_schema_sha256(BehavioralProfile),
            number_of_agents=number_of_agents,
            completed_agents=completed_agents,
            trait_algorithm=TRAIT_ALGORITHM,
            trait_rng=TRAIT_RNG,
            python_version=platform.python_version(),
            error=error,
        )

    def _write_metadata(self, run_dir: Path, metadata: RunMetadata) -> None:
        write_json(run_dir / "metadata.json", metadata.model_dump(mode="json"))
