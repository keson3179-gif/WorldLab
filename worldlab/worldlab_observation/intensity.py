"""Aggregate intensity bridge for reported-visit shares.

This is a statistical reparameterization of pre-cutoff shares. It is not an agent
choice probability and not a causal social-influence coefficient.
"""

from __future__ import annotations

import math

from worldlab_data.phase11 import share_total_variation
from worldlab_data.phase12 import valid_share_vector
from worldlab_data.phase13 import information_cutoff, is_defined_share_month, shift_month
from worldlab_observation.adapter import PHASE14_PARENTS, softmax_probabilities

ANALYSIS_ROLE = "exploratory_development"
MODEL_FAMILY = "aggregate_intensity_bridge"
TARGET_NAME = "reported_visit_distribution"
PREDICTION_QUANTITY = "aggregate_intensity_share"
TEMPERATURE = 1.0
PI_FLOOR = 1e-6
ROLLING_MONTHS = 3
HALF_LIFE_MONTHS = 6.0
BETA_GRID = (-1.0, -0.5, 0.0, 0.5, 1.0)
INTERNAL_VALIDATION_MONTHS = 12
MIN_ALPHA_MONTHS = 12
SIGNALS = ("OBS_CUM", "OBS_ROLLING", "OBS_EXP")
PHASE16_PARENTS = PHASE14_PARENTS
FORBIDDEN_QUANTITY_NAMES = (
    "agent_choice_probability",
    "causal_social_influence",
    "tourist_preference",
)
TIE_TOLERANCE = 1e-12


def refuse_quantity_alias(name: str) -> None:
    if name in FORBIDDEN_QUANTITY_NAMES or name == "reported_visit_share":
        raise ValueError("the intensity bridge is an aggregate share, not an agent choice probability")


def refuse_independent_social_claim(claim_independent: bool) -> None:
    if claim_independent:
        raise ValueError("a reported-share transform is not an identified social-influence coefficient")


def refuse_aliased_recent_alpha(alpha_months: list[str], signal_months: list[str], signal_name: str) -> None:
    if signal_name == "OBS_ROLLING" and list(alpha_months) == list(signal_months):
        raise ValueError("alpha was refit on the same recent window used as the signal")


def frozen_parents(parents: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    if tuple(parents) != PHASE16_PARENTS:
        raise RuntimeError("group membership is frozen and is not chosen by forecast error")
    return PHASE16_PARENTS


def accepts_attribute_for_year(temporal_class: str, source_year: int | None, target_year: int) -> bool:
    if temporal_class != "historically_documented":
        return False
    if source_year is None or source_year > target_year:
        return False
    return True


def m2_generation_status(historically_valid_description_count: int, site_count: int = 9) -> str:
    if historically_valid_description_count < site_count:
        return "not_run"
    return "not_run"


def smooth_and_normalize(shares: list[float], floor: float = PI_FLOOR) -> list[float]:
    if floor <= 0.0:
        raise ValueError("the share floor must be positive")
    lifted = [float(share) + floor for share in shares]
    total = math.fsum(lifted)
    normalized = [value / total for value in lifted]
    if not valid_share_vector(normalized):
        raise RuntimeError("smoothed shares left the simplex")
    return normalized


def center(values: list[float]) -> list[float]:
    if not values:
        raise ValueError("centering needs at least one value")
    location = math.fsum(values) / len(values)
    return [value - location for value in values]


def alpha_from_shares(shares: list[float]) -> list[float]:
    smoothed = smooth_and_normalize(shares)
    return center([math.log(value) for value in smoothed])


def center_and_scale(values: list[float]) -> list[float]:
    """Center across sites. A constant vector becomes zeros. Otherwise divide by the max absolute value."""
    centered = center(values)
    scale = max(abs(value) for value in centered)
    if scale == 0.0:
        return [0.0 for _value in centered]
    return [value / scale for value in centered]


def intensity_softmax(utilities: list[float]) -> list[float]:
    return softmax_probabilities(utilities, temperature=TEMPERATURE)


def training_months(history: dict[str, dict[str, float]], membership: tuple[str, ...], cutoff: str, target: str) -> list[str]:
    return sorted(
        month
        for month, shares in history.items()
        if month <= cutoff and month < target and is_defined_share_month(shares, membership)
    )


def _mean_share(history: dict[str, dict[str, float]], membership: tuple[str, ...], months: list[str]) -> list[float]:
    return [
        math.fsum(float(history[month][uid]) for month in months) / len(months)
        for uid in membership
    ]


def calendar_window(end: str, width: int) -> list[str]:
    if width <= 0:
        raise ValueError("a calendar window needs a positive width")
    return [shift_month(end, offset) for offset in range(-(width - 1), 1)]


def _month_index(year_month: str) -> int:
    year, month = (int(part) for part in str(year_month).split("-"))
    return year * 12 + (month - 1)


def signal_vector(
    history: dict[str, dict[str, float]],
    membership: tuple[str, ...],
    cutoff: str,
    target: str,
    signal_name: str,
) -> dict[str, object]:
    if signal_name not in SIGNALS:
        raise ValueError(signal_name)
    available = training_months(history, membership, cutoff, target)
    if signal_name == "OBS_ROLLING":
        window = calendar_window(cutoff, ROLLING_MONTHS)
        if any(month not in history or not is_defined_share_month(history[month], membership) for month in window):
            return _signal_unavailable("rolling_window_incomplete", window)
        if any(month >= target for month in window):
            return _signal_unavailable("rolling_window_after_target", window)
        raw = _mean_share(history, membership, window)
        months = window
        same_as_training_mean = False
    elif signal_name == "OBS_CUM":
        if not available:
            return _signal_unavailable("no_training_month", [])
        raw = _mean_share(history, membership, available)
        months = available
        same_as_training_mean = True
    else:
        if not available:
            return _signal_unavailable("no_training_month", [])
        weights = [0.5 ** ((_month_index(cutoff) - _month_index(month)) / HALF_LIFE_MONTHS) for month in available]
        weight_total = math.fsum(weights)
        raw = [
            math.fsum(weight * float(history[month][uid]) for month, weight in zip(available, weights, strict=True)) / weight_total
            for uid in membership
        ]
        months = available
        same_as_training_mean = False
    scaled = center_and_scale(raw)
    return {
        "status": "ready",
        "skip_reason": "",
        "values": scaled,
        "source_months": months,
        "same_information_as_training_mean": same_as_training_mean,
        "memory_unit": "calendar_month",
        "rolling_months": ROLLING_MONTHS if signal_name == "OBS_ROLLING" else "",
        "half_life_months": HALF_LIFE_MONTHS if signal_name == "OBS_EXP" else "",
    }


def _signal_unavailable(reason: str, months: list[str]) -> dict[str, object]:
    return {
        "status": "prediction_unavailable",
        "skip_reason": reason,
        "values": None,
        "source_months": months,
        "same_information_as_training_mean": False,
        "memory_unit": "calendar_month",
        "rolling_months": "",
        "half_life_months": "",
    }


def m0_components(
    history: dict[str, dict[str, float]],
    membership: tuple[str, ...],
    target: str,
    scenario: str,
) -> dict[str, object]:
    cutoff = information_cutoff(target, scenario)
    months = training_months(history, membership, cutoff, target)
    if not months:
        return {
            "status": "prediction_unavailable",
            "skip_reason": "no_training_month",
            "shares": None,
            "alpha": None,
            "training_mean": None,
            "source_months": [],
            "information_cutoff": cutoff,
            "reparameterization_gap": None,
        }
    training_mean = _mean_share(history, membership, months)
    smoothed = smooth_and_normalize(training_mean)
    alpha = center([math.log(value) for value in smoothed])
    shares = intensity_softmax(alpha)
    gap = max(abs(left - right) for left, right in zip(shares, smoothed, strict=True))
    return {
        "status": "predicted",
        "skip_reason": "",
        "shares": shares,
        "alpha": alpha,
        "training_mean": training_mean,
        "source_months": months,
        "information_cutoff": cutoff,
        "reparameterization_gap": gap,
        "quantity": PREDICTION_QUANTITY,
        "temperature": TEMPERATURE,
    }


def m1_shares(components: dict[str, object], signal: dict[str, object], beta: float) -> dict[str, object]:
    if components["status"] != "predicted":
        return {
            "status": "prediction_unavailable",
            "skip_reason": components["skip_reason"],
            "shares": None,
            "beta": float(beta),
        }
    if abs(float(beta)) <= TIE_TOLERANCE:
        return {"status": "predicted", "skip_reason": "", "shares": list(components["shares"]), "beta": 0.0}
    if signal["status"] != "ready" or signal["values"] is None:
        return {
            "status": "prediction_unavailable",
            "skip_reason": signal["skip_reason"] or "signal_unavailable",
            "shares": None,
            "beta": float(beta),
        }
    alpha = list(components["alpha"])
    scores = list(signal["values"])
    refuse_aliased_recent_alpha(list(components["source_months"]), list(signal["source_months"]), "OBS_ROLLING" if signal["rolling_months"] else "other")
    utilities = [base + float(beta) * score for base, score in zip(alpha, scores, strict=True)]
    return {"status": "predicted", "skip_reason": "", "shares": intensity_softmax(utilities), "beta": float(beta)}


def shares_close(left: list[float] | None, right: list[float] | None, tolerance: float = 1e-12) -> bool:
    if left is None or right is None or len(left) != len(right):
        return False
    return all(abs(a - b) <= tolerance for a, b in zip(left, right, strict=True))


def scale_equivalent(alpha: list[float], scores: list[float], beta: float, scale: float) -> bool:
    if scale == 0.0:
        raise ValueError("scale must be nonzero")
    left = intensity_softmax([base + beta * score for base, score in zip(alpha, scores, strict=True)])
    right = intensity_softmax([base + (beta / scale) * (scale * score) for base, score in zip(alpha, scores, strict=True)])
    return shares_close(left, right)


def eligible_internal_months(
    history: dict[str, dict[str, float]],
    membership: tuple[str, ...],
    target: str,
    scenario: str,
) -> list[str]:
    eligible: list[str] = []
    defined = sorted(month for month, shares in history.items() if is_defined_share_month(shares, membership))
    for month in defined:
        if month >= target:
            continue
        cutoff = information_cutoff(month, scenario)
        prior = [item for item in defined if item <= cutoff and item < month]
        if len(prior) >= MIN_ALPHA_MONTHS:
            eligible.append(month)
    if len(eligible) < INTERNAL_VALIDATION_MONTHS:
        return []
    return eligible[-INTERNAL_VALIDATION_MONTHS:]


def choose_beta(means: dict[float, float]) -> float:
    """Lowest internal TV wins. A tie keeps beta 0, then the smaller absolute value."""
    if not means:
        raise ValueError("beta selection needs at least one internal score")
    best = min(means.values())
    tied = [beta for beta, value in means.items() if abs(value - best) <= TIE_TOLERANCE]
    return sorted(tied, key=lambda beta: (abs(beta) > TIE_TOLERANCE, abs(beta), BETA_GRID.index(beta)))[0]


def select_beta(
    history: dict[str, dict[str, float]],
    membership: tuple[str, ...],
    target: str,
    scenario: str,
    signal_name: str,
) -> dict[str, object]:
    """Choose beta from months before the target. The target outcome is not a selection feature."""
    internal = eligible_internal_months(history, membership, target, scenario)
    if not internal:
        return {
            "status": "insufficient_history",
            "skip_reason": "insufficient_history",
            "beta": None,
            "internal_months": [],
            "internal_tv": {},
        }
    internal_tv: dict[float, list[float]] = {beta: [] for beta in BETA_GRID}
    for month in internal:
        components = m0_components(history, membership, month, scenario)
        signal = signal_vector(history, membership, str(components["information_cutoff"]), month, signal_name)
        actual = [float(history[month][uid]) for uid in membership]
        for beta in BETA_GRID:
            predicted = m1_shares(components, signal, beta)
            if predicted["status"] != "predicted":
                continue
            internal_tv[beta].append(
                share_total_variation(
                    {uid: actual[index] for index, uid in enumerate(membership)},
                    {uid: predicted["shares"][index] for index, uid in enumerate(membership)},
                )
            )
    means = {
        beta: sum(values) / len(values)
        for beta, values in internal_tv.items()
        if len(values) == len(internal)
    }
    if not means:
        return {
            "status": "insufficient_history",
            "skip_reason": "insufficient_history",
            "beta": None,
            "internal_months": internal,
            "internal_tv": {},
        }
    chosen = choose_beta(means)
    return {
        "status": "selected",
        "skip_reason": "",
        "beta": chosen,
        "internal_months": internal,
        "internal_tv": {str(beta): means[beta] for beta in BETA_GRID if beta in means},
    }


def prediction_ignores_target(
    history: dict[str, dict[str, float]],
    membership: tuple[str, ...],
    target: str,
    scenario: str,
) -> bool:
    components = m0_components(history, membership, target, scenario)
    altered = {month: dict(shares) for month, shares in history.items()}
    if target in altered:
        width = len(membership)
        altered[target] = {uid: 1.0 if index == 0 else 0.0 for index, uid in enumerate(membership)}
        if width > 1 and not valid_share_vector([altered[target][uid] for uid in membership]):
            raise RuntimeError("altered target left the simplex")
    later = [month for month in list(altered) if month > target]
    for month in later:
        del altered[month]
    shadow = m0_components(altered, membership, target, scenario)
    return shares_close(components["shares"], shadow["shares"]) and components["source_months"] == shadow["source_months"]
