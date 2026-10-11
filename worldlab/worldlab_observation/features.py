"""Site-month feature eligibility. A common group-month value cancels in the share softmax."""

from __future__ import annotations

from worldlab_observation.adapter import PHASE14_PARENTS, softmax_probabilities

PHASE17_PARENTS = PHASE14_PARENTS
HISTORICAL_ASOF = "historical_asof_supported"
RETROSPECTIVE_PROXY = "retrospective_proxy_only"
CURRENT_SNAPSHOT = "current_snapshot_only"
UNAVAILABLE = "unavailable_or_unreliable"
EX_POST_WEATHER = "ex_post_conditional_analysis"
SOCIAL_UNOBSERVED = "external_social_information_unobserved"
DECISION_NO_GO = "NO-GO"
DECISION_LIMITED = "LIMITED_GO"
DECISION_GO = "GO"


def constant_shift_cancels(utilities: list[float], shift: float) -> bool:
    base = softmax_probabilities(utilities, temperature=1.0)
    moved = softmax_probabilities([value + shift for value in utilities], temperature=1.0)
    return all(abs(left - right) <= 1e-12 for left, right in zip(base, moved, strict=True))


def common_feature_cancels(site_values: list[float]) -> bool:
    if len(site_values) < 2:
        return False
    return max(site_values) - min(site_values) <= 1e-12


def has_site_variation(site_values: list[float]) -> bool:
    observed = [value for value in site_values if value is not None]
    if len(observed) < 2:
        return False
    return not common_feature_cancels(observed)


def has_temporal_variation(month_values: list[float]) -> bool:
    observed = [value for value in month_values if value is not None]
    return len(set(observed)) > 1


def reject_imputed_missing(was_missing: bool, was_filled: bool) -> None:
    if was_missing and was_filled:
        raise ValueError("a missing feature was imputed")


def allows_strict_cutoff(availability: str, leakage_risk: str) -> bool:
    if leakage_risk == EX_POST_WEATHER:
        return False
    return availability == HISTORICAL_ASOF


def refuse_visit_share_as_social(source_name: str) -> None:
    blocked = {"reported_visit_share", "historical_visits", "reported_visit_share_conditional"}
    if source_name in blocked:
        raise ValueError("past visits are not observed social exposure")


def identity_blocks_link(confirmed_manual: bool, human_reviewed: bool) -> bool:
    return not (confirmed_manual and human_reviewed)


def decision_from_eligibility(rows: list[dict[str, str]]) -> str:
    """Within-group share gain requires an as-of feature that differs across the full group."""
    for row in rows:
        if row.get("share_increment") == "true" and row.get("historical_availability_status") == HISTORICAL_ASOF:
            return DECISION_GO
        if row.get("share_increment") == "true" and row.get("analysis_role") == EX_POST_WEATHER:
            return DECISION_LIMITED
    return DECISION_NO_GO
