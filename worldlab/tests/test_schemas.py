from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from models.schemas import (
    TRAIT_NAMES,
    AgentPersona,
    AgentTraits,
    BehavioralProfile,
    RunMetadata,
    export_response_schema,
)
from tests.support import VALID_PROFILE, trait_values

EXPECTED_TRAITS = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
    "novelty_seeking",
    "crowd_aversion",
    "social_orientation",
    "photography_interest",
    "risk_tolerance",
]

EXPECTED_PROFILE_FIELDS = [
    "summary",
    "planning_style",
    "social_behavior",
    "exploration_behavior",
    "crowd_response",
    "photography_behavior",
    "risk_behavior",
    "priorities",
    "behavioral_constraints",
]

FORBIDDEN_FIELDS = {
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


def test_trait_field_order_is_the_rng_contract() -> None:
    assert list(TRAIT_NAMES) == EXPECTED_TRAITS
    assert list(AgentTraits.model_fields) == EXPECTED_TRAITS


def test_profile_fields_and_forbidden_identity_fields() -> None:
    assert list(BehavioralProfile.model_fields) == EXPECTED_PROFILE_FIELDS
    for model in (AgentTraits, BehavioralProfile, AgentPersona):
        assert FORBIDDEN_FIELDS.isdisjoint(model.model_fields)


def test_traits_accept_endpoints_and_reject_out_of_range() -> None:
    AgentTraits.model_validate(trait_values(openness=0.0, risk_tolerance=1.0))
    with pytest.raises(ValidationError):
        AgentTraits.model_validate(trait_values(openness=-0.01))
    with pytest.raises(ValidationError):
        AgentTraits.model_validate(trait_values(openness=1.01))
    with pytest.raises(ValidationError):
        AgentTraits.model_validate(trait_values(openness=float("nan")))
    with pytest.raises(ValidationError):
        AgentTraits.model_validate({**trait_values(), "gender": "unspecified"})


def test_traits_are_frozen() -> None:
    traits = AgentTraits.model_validate(trait_values())
    with pytest.raises(ValidationError):
        traits.openness = 0.1


def test_profile_rejects_trait_and_demographic_fields() -> None:
    with pytest.raises(ValidationError):
        BehavioralProfile.model_validate({**VALID_PROFILE, "openness": 0.2})
    with pytest.raises(ValidationError):
        BehavioralProfile.model_validate({**VALID_PROFILE, "gender": "unspecified"})
    with pytest.raises(ValidationError):
        BehavioralProfile.model_validate({**VALID_PROFILE, "priorities": ["only one"]})
    with pytest.raises(ValidationError):
        BehavioralProfile.model_validate(
            {
                **VALID_PROFILE,
                "priorities": ["one", "two", "three", "four", "five"],
            }
        )
    with pytest.raises(ValidationError):
        BehavioralProfile.model_validate(
            {**VALID_PROFILE, "behavioral_constraints": ["ok", "  "]}
        )


def test_persona_cannot_relabel_trait_source() -> None:
    traits = AgentTraits.model_validate(trait_values())
    profile = BehavioralProfile.model_validate(VALID_PROFILE)
    with pytest.raises(ValidationError):
        AgentPersona(
            agent_id="agent_0000",
            agent_index=0,
            run_id="run",
            trait_source="llm",  # type: ignore[arg-type]
            trait_seed=1,
            traits=traits,
            profile_source="llm",
            behavioral_profile=profile,
            prompt_version="v",
        )


def test_response_schema_is_flat_and_has_no_trait_fields() -> None:
    schema = export_response_schema(BehavioralProfile)
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == set(EXPECTED_PROFILE_FIELDS)
    assert set(schema["required"]) == set(EXPECTED_PROFILE_FIELDS)
    assert not _has_key(schema, "title")
    assert not _has_key(schema, "$schema")
    assert FORBIDDEN_FIELDS.isdisjoint(schema["properties"])
    assert set(TRAIT_NAMES).isdisjoint(schema["properties"])


def test_nested_response_schema_is_rejected() -> None:
    class Inner(BaseModel):
        value: str

    class Outer(BaseModel):
        inner: Inner

    with pytest.raises(ValueError, match="flat"):
        export_response_schema(Outer)


def test_completed_metadata_must_finish_every_agent() -> None:
    with pytest.raises(ValidationError):
        RunMetadata(
            run_id="run",
            timestamp="2026-09-26T01:02:03+00:00",
            status="completed",
            model="unit-test-model",
            model_parameters={"temperature": 0.2, "seed": 1, "reasoning_effort": "none"},
            api_endpoint="http://127.0.0.1:9/v1",
            master_seed=1,
            prompt_version="v",
            prompt_sha256="abc",
            response_schema_sha256="def",
            number_of_agents=2,
            completed_agents=1,
            trait_algorithm="worldlab.traits.v1",
            trait_rng="random.Random",
            python_version="3.11.0",
            error=None,
        )


def _has_key(node: object, key: str) -> bool:
    if isinstance(node, dict):
        if key in node:
            return True
        return any(_has_key(value, key) for value in node.values())
    if isinstance(node, list):
        return any(_has_key(value, key) for value in node)
    return False
