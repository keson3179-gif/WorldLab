from __future__ import annotations

import hashlib
import inspect
import json
import random

import pytest

from agents.persona_generator import (
    USER_PREFIX,
    USER_SUFFIX,
    build_messages,
    compute_prompt_sha256,
    derive_trait_seed,
    generate_traits,
    render_system_prompt,
)
from models.schemas import TRAIT_NAMES, AgentTraits

# Pinned output of worldlab.traits.v1 at master_seed=42, agent_index=0.
# random.Random.random() is in [0.0, 1.0); stored values still satisfy 0 <= x <= 1.
PINNED_TRAIT_SEED = 14271812464600022187
PINNED_TRAITS = {
    "openness": 0.9045126467541891,
    "conscientiousness": 0.3769595232255477,
    "extraversion": 0.16067561483053994,
    "agreeableness": 0.439952010929893,
    "neuroticism": 0.9150665839661599,
    "novelty_seeking": 0.7993128678014918,
    "crowd_aversion": 0.06932668865605385,
    "social_orientation": 0.9026477964726438,
    "photography_interest": 0.5343285323376186,
    "risk_tolerance": 0.5214409591076492,
}


def test_trait_function_does_not_accept_model_inputs() -> None:
    assert list(inspect.signature(generate_traits).parameters) == ["master_seed", "agent_index"]


def test_traits_match_the_documented_rng_and_a_pinned_vector() -> None:
    traits, trait_seed = generate_traits(42, 0)
    assert trait_seed == _oracle_seed(42, 0)
    assert traits.model_dump() == _oracle_traits(42, 0)
    assert trait_seed == PINNED_TRAIT_SEED
    assert traits.model_dump() == PINNED_TRAITS
    again, again_seed = generate_traits(42, 0)
    assert again.model_dump() == traits.model_dump()
    assert again_seed == trait_seed


def test_trait_values_stay_inside_the_unit_interval() -> None:
    for index in range(20):
        traits, trait_seed = generate_traits(42, index)
        assert 0 <= trait_seed < 2**64
        for value in traits.model_dump().values():
            assert 0.0 <= value <= 1.0


def test_agent_indexes_do_not_share_a_trait_vector() -> None:
    vectors = [generate_traits(42, index)[0].model_dump() for index in range(20)]
    assert len({tuple(vector.values()) for vector in vectors}) == 20
    assert generate_traits(1, 0)[0].model_dump() != generate_traits(2, 0)[0].model_dump()


def test_negative_indexes_are_rejected() -> None:
    with pytest.raises(ValueError):
        derive_trait_seed(42, -1)
    with pytest.raises(ValueError):
        generate_traits(-1, 0)


def test_prompt_conditions_on_traits_and_forbids_identity_and_code() -> None:
    traits, _ = generate_traits(42, 0)
    messages = build_messages(
        traits,
        prompt_version="persona_behavioral_profile_v1",
        profile_language="English",
    )
    assert [message["role"] for message in messages] == ["system", "user"]
    system = messages[0]["content"]
    user = messages[1]["content"]
    for word in ("name", "gender", "nationality", "ethnicity", "religion", "occupation", "age", "Python"):
        assert word in system
    assert "immutable" in system
    assert "You are not a participant" in system
    assert "Write in English." in system
    assert USER_PREFIX in user
    assert USER_SUFFIX in user
    rendered = json.dumps(traits.model_dump(), ensure_ascii=False, indent=2)
    assert rendered in user
    positions = [user.index(name) for name in TRAIT_NAMES]
    assert positions == sorted(positions)


def test_prompt_hash_tracks_language_and_version() -> None:
    first = compute_prompt_sha256("persona_behavioral_profile_v1", "English")
    assert first == compute_prompt_sha256("persona_behavioral_profile_v1", "English")
    assert first != compute_prompt_sha256("persona_behavioral_profile_v1", "Traditional Chinese")
    assert first != compute_prompt_sha256("persona_behavioral_profile_v2", "English")
    assert "Traditional Chinese" in render_system_prompt("Traditional Chinese")


def test_pinned_vector_is_valid() -> None:
    AgentTraits.model_validate(PINNED_TRAITS)


def _oracle_seed(master_seed: int, agent_index: int) -> int:
    material = f"worldlab.traits.v1|{master_seed}|{agent_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _oracle_traits(master_seed: int, agent_index: int) -> dict[str, float]:
    rng = random.Random(_oracle_seed(master_seed, agent_index))
    return {name: rng.random() for name in TRAIT_NAMES}
