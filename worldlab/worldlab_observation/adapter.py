"""Map WorldLab choice probabilities onto a monthly reported-visit share.

Synthetic place ids stay synthetic. A reported share is not a social-exposure measure.
"""

from __future__ import annotations

import math
from random import Random

from world.choice import softmax_choice
from world.experiment_002 import REFERENCE_CHOICE_TEMPERATURE, REFERENCE_RECONSIDERATION_RATE
from world.experiment_005 import SCORE_CLIP
from worldlab_data.phase12 import valid_share_vector
from worldlab_data.phase13 import MODELS, information_cutoff

ANALYSIS_ROLE = "exploratory_development"
TARGET_NAME = "reported_visit_distribution"
AGENT_COUNT = 20
SHARE_COMPONENT_SE = 0.01
REQUIRED_REPLICATES_FOR_SHARE_SE = 125
STEP_TO_MONTH = "one_decision_step_per_target_month_is_a_mapping_assumption"
SYNTHETIC_PLACE_IDS = ("synth_a", "synth_b", "synth_c")
FORBIDDEN_SYNTHETIC_IDS = ("site_01", "site_02", "site_03", "site_04", "site_05")
ATLAS_ATTRIBUTES = (
    "visual_attractiveness",
    "novelty",
    "capacity",
    "pairwise_distance",
    "respondent_traits",
)
SOCIAL_SOURCES = (
    "reported_visit_share_conditional",
    "model_endogenous_presence",
    "synthetic_smoke",
)
PHASE14_PARENTS = ("玉山國家公園", "太魯閣國家公園", "雪霸國家公園")
REUSABLE_INTERFACES = (
    "world.choice.softmax_choice",
    "world.choice.choice_seed",
    "world.choice.agent_reconsiders",
    "world.experiment_002.REFERENCE_CHOICE_TEMPERATURE",
    "world.experiment_002.REFERENCE_RECONSIDERATION_RATE",
    "world.experiment_004.popularity_scores",
    "world.experiment_004.SocialState",
    "world.experiment_004.commit_social",
    "world.experiment_005.centered_scores",
    "world.experiment_005.RollingMemory",
    "world.experiment_005.DecayMemory",
    "world.experiment_017.score_memory",
    "world.experiment_018.q_components",
    "world.experiment_020.source_at",
    "world.geography.PLACE_IDS",
    "worldlab_data.phase13.predict_shares",
    "worldlab_data.phase13.information_cutoff",
)


def softmax_probabilities(utilities: list[float], temperature: float = REFERENCE_CHOICE_TEMPERATURE) -> list[float]:
    """Peak-shifted softmax. The draw inside softmax_choice is not part of this vector."""
    if temperature <= 0.0:
        raise ValueError("choice temperature must be positive")
    if not utilities:
        raise ValueError("utilities must align with places")
    peak = max(utilities)
    weights = [math.exp((utility - peak) / temperature) for utility in utilities]
    total = math.fsum(weights)
    probabilities = [weight / total for weight in weights]
    if not valid_share_vector(probabilities):
        raise RuntimeError("softmax did not return a share vector")
    return probabilities


def probabilities_match_engine(utilities: list[float], temperature: float = REFERENCE_CHOICE_TEMPERATURE) -> bool:
    places = [f"p{index}" for index in range(len(utilities))]
    _selected, engine = softmax_choice(places, [int(round(value)) for value in utilities], temperature, seed=1)
    direct = softmax_probabilities([float(int(round(value))) for value in utilities], temperature)
    return all(abs(engine[place] - direct[index]) <= 1e-12 for index, place in enumerate(places))


def decision_q(base_utilities: list[float], social_scores: list[float], weight: float, temperature: float = REFERENCE_CHOICE_TEMPERATURE) -> float:
    """In-model counterfactual probability distance. This is not an observed share TV."""
    if len(base_utilities) != len(social_scores):
        raise ValueError("social scores must align with base utilities")
    base = softmax_probabilities(base_utilities, temperature)
    social = softmax_probabilities(
        [base_value + float(weight) * score for base_value, score in zip(base_utilities, social_scores, strict=True)],
        temperature,
    )
    return 0.5 * sum(abs(left - right) for left, right in zip(base, social, strict=True))


def stay_adjusted_probability(choice_probability: list[float], stay_index: int, reconsideration_rate: float) -> list[float]:
    if not 0.0 <= reconsideration_rate <= 1.0:
        raise ValueError("reconsideration rate must be inside 0 and 1")
    if stay_index < 0 or stay_index >= len(choice_probability):
        raise ValueError("stay index is outside the place set")
    adjusted = [reconsideration_rate * value for value in choice_probability]
    adjusted[stay_index] += 1.0 - reconsideration_rate
    if not valid_share_vector(adjusted):
        raise RuntimeError("stay-adjusted probability left the simplex")
    return adjusted


def expected_share(agent_probabilities: list[list[float]]) -> list[float]:
    """Mean of agent choice probabilities. Agent count is not a reported visit total."""
    if not agent_probabilities:
        raise ValueError("at least one agent probability is required")
    width = len(agent_probabilities[0])
    if any(len(row) != width or not valid_share_vector(row) for row in agent_probabilities):
        raise ValueError("agent probabilities must share one simplex")
    averaged = [sum(row[index] for row in agent_probabilities) / len(agent_probabilities) for index in range(width)]
    if not valid_share_vector(averaged):
        raise RuntimeError("expected share left the simplex")
    return averaged


def undefined_share(total_visits: float) -> str | None:
    if total_visits < 0:
        raise ValueError("visits cannot be negative")
    if total_visits <= 0:
        return "all_zero_group_total"
    return None


def replicates_for_share_se(agent_count: int = AGENT_COUNT, standard_error: float = SHARE_COMPONENT_SE, worst_probability: float = 0.5) -> int:
    """Replicates so the worst-case multinomial mean has the requested component standard error."""
    if agent_count < 1 or standard_error <= 0.0:
        raise ValueError("agent count and standard error must be positive")
    variance = worst_probability * (1.0 - worst_probability)
    return math.ceil(variance / (agent_count * standard_error * standard_error))


def multinomial_share(probabilities: list[float], agent_count: int, rng: Random) -> list[float]:
    if agent_count < 1 or not valid_share_vector(probabilities):
        raise ValueError("multinomial share needs a simplex and a positive agent count")
    counts = [0 for _ in probabilities]
    for _ in range(agent_count):
        draw = rng.random()
        cursor = 0.0
        chosen = len(probabilities) - 1
        for index, probability in enumerate(probabilities):
            cursor += probability
            if draw <= cursor:
                chosen = index
                break
        counts[chosen] += 1
    return [count / agent_count for count in counts]


def monte_carlo_mean(probabilities: list[float], replicates: int, seed: int, agent_count: int = AGENT_COUNT) -> dict[str, object]:
    rng = Random(seed)
    draws = [multinomial_share(probabilities, agent_count, rng) for _ in range(replicates)]
    mean = [sum(draw[index] for draw in draws) / replicates for index in range(len(probabilities))]
    max_abs = max(abs(mean[index] - probabilities[index]) for index in range(len(probabilities)))
    one = draws[0]
    one_tv = 0.5 * sum(abs(one[index] - probabilities[index]) for index in range(len(probabilities)))
    return {
        "mean_share": mean,
        "max_abs_error": max_abs,
        "one_world_tv": one_tv,
        "replicates": replicates,
        "agent_count": agent_count,
        "seed": seed,
    }


def logmax_scores(leave_one_out_counts: list[float]) -> list[float]:
    """J-general form of Experiment 004 popularity scores. Own presence is already removed."""
    if any(count < 0 for count in leave_one_out_counts):
        raise ValueError("leave-one-out presence cannot be negative")
    raw = [math.log1p(count) for count in leave_one_out_counts]
    peak = max(raw) if raw else 0.0
    if peak == 0.0:
        return [0.0 for _ in raw]
    return [value / peak for value in raw]


def zraw_scores(leave_one_out_counts: list[float], clip: float = SCORE_CLIP) -> list[float]:
    """J-general form of Experiment 005 centered scores, clipped to the engine clip."""
    if len(leave_one_out_counts) < 2:
        raise ValueError("centered scores need at least two places")
    values = [float(count) for count in leave_one_out_counts]
    if max(values) == min(values):
        return [0.0 for _ in values]
    center = sum(values) / len(values)
    scale = math.sqrt(sum((value - center) ** 2 for value in values) / len(values))
    unclipped = [(value - center) / scale for value in values]
    return [min(clip, max(-clip, value)) for value in unclipped]


def refuse_synthetic_alias(place_id: str, site_uid: str) -> None:
    if place_id in FORBIDDEN_SYNTHETIC_IDS and site_uid.startswith("stat_"):
        raise ValueError("a synthetic atlas place cannot be renamed as a real site")


class SocialSignal:
    def __init__(self, place_ids: tuple[str, ...], values: list[float], source: str) -> None:
        if source not in SOCIAL_SOURCES:
            raise ValueError("social signal source is not labeled")
        if len(place_ids) != len(values):
            raise ValueError("social signal must align with places")
        self.place_ids = place_ids
        self.values = [float(value) for value in values]
        self.source = source
        self.exposure_claim = False

    def as_record(self) -> dict[str, object]:
        return {
            "place_ids": list(self.place_ids),
            "values": self.values,
            "source": self.source,
            "exposure_claim": self.exposure_claim,
        }


class ReportedHistory:
    """Observed months. A model prediction is not written back into this history."""

    def __init__(self, place_ids: tuple[str, ...]) -> None:
        self.place_ids = place_ids
        self.months: dict[str, list[float]] = {}

    def add_observed(self, year_month: str, shares: list[float]) -> None:
        if len(shares) != len(self.place_ids) or not valid_share_vector(shares):
            raise ValueError("observed share must match the fixed place set")
        self.months[year_month] = [float(value) for value in shares]


class EndogenousHistory:
    """Model-produced presence. Reported months cannot be inserted here."""

    def __init__(self, place_ids: tuple[str, ...]) -> None:
        self.place_ids = place_ids
        self.months: dict[str, list[float]] = {}

    def add_model_share(self, year_month: str, shares: list[float], source: str) -> None:
        if source != "model_endogenous_presence":
            raise ValueError("free-running history only accepts model endogenous presence")
        if len(shares) != len(self.place_ids) or not valid_share_vector(shares):
            raise ValueError("model share must match the fixed place set")
        self.months[year_month] = [float(value) for value in shares]


def conditional_one_step(
    reported: ReportedHistory,
    target: str,
    scenario: str,
    agent_probabilities: list[list[float]] | None,
) -> dict[str, object]:
    cutoff = information_cutoff(target, scenario)
    used = sorted(month for month in reported.months if month <= cutoff and month < target)
    leaked = [month for month in used if month > cutoff]
    if leaked:
        raise RuntimeError("conditional features passed the cutoff")
    if agent_probabilities is None:
        status = "prediction_unavailable"
        shares = None
        reason = "real_site_attributes_missing"
    else:
        status = "predicted"
        shares = expected_share(agent_probabilities)
        reason = ""
    return {
        "mode": "conditional_one_step",
        "target": target,
        "scenario": scenario,
        "information_cutoff": cutoff,
        "available_at_assumption": scenario,
        "source_months": used,
        "feature_lineage": f"reported_history:<={cutoff}",
        "prediction_status": status,
        "skip_reason": reason,
        "predicted_share": shares,
        "history_kind": "reported_conditional",
        "mapping": STEP_TO_MONTH,
        "reported_months_after": sorted(reported.months),
    }


def free_running_step(endogenous: EndogenousHistory, target: str, agent_probabilities: list[list[float]], source: str) -> dict[str, object]:
    if source != "model_endogenous_presence":
        raise ValueError("free-running cannot be corrected with a reported month")
    shares = expected_share(agent_probabilities)
    return {
        "mode": "free_running",
        "target": target,
        "source_months": sorted(endogenous.months),
        "social_source": source,
        "predicted_share": shares,
        "history_kind": "model_endogenous",
        "mapping": STEP_TO_MONTH,
        "uses_mid_run_reported_correction": False,
    }


def attribute_gaps(available_columns: set[str]) -> dict[str, object]:
    missing = [name for name in ATLAS_ATTRIBUTES if name not in available_columns]
    return {
        "utility_status": "unavailable" if missing else "attribute_only_ready",
        "missing_attributes": missing,
        "invented_attributes": False,
        "llm_generation_status": "not_run",
        "neutral_place_codes_required": True,
        "validation_popularity_in_prompt": False,
    }


def scale_tradeoff_same_probability(base: list[float], scores: list[float], weight: float) -> bool:
    """Doubling the score and halving the weight leaves the softmax unchanged."""
    left = softmax_probabilities([base_value + weight * score for base_value, score in zip(base, scores, strict=True)])
    right = softmax_probabilities([base_value + (weight / 2.0) * (score * 2.0) for base_value, score in zip(base, scores, strict=True)])
    return all(abs(a - b) <= 1e-12 for a, b in zip(left, right, strict=True))


def fair_one_step_status(member_safe_to_link: list[bool]) -> str:
    if not member_safe_to_link or not all(member_safe_to_link):
        return "unavailable_incomplete_2024_identity"
    return "requirements_recorded_not_executed"


def mae_equals_scaled_tv(left: list[float], right: list[float]) -> bool:
    width = len(left)
    if width == 0 or width != len(right):
        return False
    tv = 0.5 * sum(abs(a - b) for a, b in zip(left, right, strict=True))
    mae = sum(abs(a - b) for a, b in zip(left, right, strict=True)) / width
    return abs(mae - (2.0 * tv / width)) <= 1e-12


def llm_utility_schema() -> dict[str, object]:
    return {
        "place_code": "neutral_code",
        "card_version": None,
        "attributes_sent": [],
        "excluded_fields": [
            "visit_counts",
            "shares",
            "hhi",
            "validation_period_popularity",
            "descriptions_published_after_the_observation_month",
        ],
        "model_version": None,
        "prompt_sha256": None,
        "parse_rule": None,
        "response_retained": False,
        "generation_status": "not_run",
        "knowledge_contamination_check": "not_run",
    }


def engine_constants() -> dict[str, object]:
    return {
        "choice_temperature": REFERENCE_CHOICE_TEMPERATURE,
        "reconsideration_rate": REFERENCE_RECONSIDERATION_RATE,
        "score_clip": SCORE_CLIP,
        "agent_count_in_engine": AGENT_COUNT,
        "baseline_models": list(MODELS),
        "required_replicates_for_component_se_0_01": replicates_for_share_se(),
    }
