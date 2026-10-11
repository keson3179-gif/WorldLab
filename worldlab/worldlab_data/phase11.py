from __future__ import annotations

import math
from collections import defaultdict

HOLDOUT_YEAR = 2025
BREAK_THRESHOLD = 0.10
SHOCK_START = "2020-01"
SHOCK_END = "2022-12"
MIN_RETRO_TRAIN_MONTHS = 24
RETRO_HORIZON_MONTHS = 12
HYPOTHETICAL_SCALE_FACTORS = (0.8, 1.2)


def previous_month(year_month: str) -> str:
    year, month = (int(part) for part in str(year_month).split("-"))
    month -= 1
    if month == 0:
        month = 12
        year -= 1
    return f"{year:04d}-{month:02d}"


def month_span(start: str, end: str) -> list[str]:
    year, month = (int(part) for part in str(start).split("-"))
    end_year, end_month = (int(part) for part in str(end).split("-"))
    months: list[str] = []
    while (year, month) <= (end_year, end_month):
        months.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            month = 1
            year += 1
    return months


def exclude_holdout(year_months: list[str], holdout_year: int = HOLDOUT_YEAR) -> tuple[list[str], list[str]]:
    """Return analysis months and the holdout months that were kept out of the series."""
    kept: list[str] = []
    held: list[str] = []
    for year_month in year_months:
        year_text = str(year_month)[:4]
        if year_text.isdigit() and int(year_text) >= holdout_year:
            held.append(str(year_month))
        else:
            kept.append(str(year_month))
    return kept, held


def concentration_metrics(shares: dict[str, float]) -> dict[str, object]:
    """HHI is undefined when the share vector is empty or the visits sum to zero."""
    empty = {
        "raw_hhi": None,
        "normalized_hhi": None,
        "top_site_share": None,
        "effective_number_of_sites": None,
        "distribution_entropy": None,
        "leading_site_uid": "",
        "leader_tie": False,
    }
    if not shares:
        return empty
    total = float(sum(shares.values()))
    if total <= 0:
        return empty
    weights = [float(value) / total for value in shares.values()]
    raw = float(sum(value * value for value in weights))
    site_count = len(weights)
    normalized = None
    if site_count > 1:
        normalized = (raw - (1.0 / site_count)) / (1.0 - (1.0 / site_count))
    top = max(weights)
    leaders = sorted(uid for uid, value in shares.items() if abs((float(value) / total) - top) <= 1e-12)
    tie = len(leaders) != 1
    entropy = 0.0
    for value in weights:
        if value > 0:
            entropy -= value * math.log(value)
    return {
        "raw_hhi": raw,
        "normalized_hhi": normalized,
        "top_site_share": top,
        "effective_number_of_sites": None if raw <= 0 else 1.0 / raw,
        "distribution_entropy": entropy,
        "leading_site_uid": "" if tie else leaders[0],
        "leader_tie": tie,
    }


def share_total_variation(left: dict[str, float] | None, right: dict[str, float] | None) -> float | None:
    """Total-variation distance between two share vectors. This is not a counterfactual Q."""
    if not left or not right or set(left) != set(right):
        return None
    return 0.5 * float(sum(abs(float(left[key]) - float(right[key])) for key in left))


def leader_turnover(previous_leader: str, previous_tie: bool, leader: str, tie: bool) -> bool | None:
    """A tie does not invent a unique leader and does not count as a resolved turnover."""
    if previous_tie or tie or not previous_leader or not leader:
        return None
    return previous_leader != leader


def assess_group_month(
    values: dict[str, float | None],
    source_conflict: bool,
    within_uid_conflict: bool = False,
) -> dict[str, object]:
    """Classify one group-month. A zero total leaves HHI undefined rather than zero."""
    present = {uid: float(value) for uid, value in values.items() if value is not None}
    expected = len(values)
    reported = len(present)
    all_reported = expected > 0 and reported == expected
    total = float(sum(present.values())) if present else None
    reason = ""
    if within_uid_conflict and source_conflict:
        reason = "conflicting_visits_within_uid_and_source_conflict"
    elif within_uid_conflict:
        reason = "conflicting_visits_within_uid"
    elif source_conflict:
        reason = "source_conflict"
    elif not all_reported:
        reason = "incomplete_site_reports"
    elif total is None or total <= 0:
        reason = "all_zero_group_total"
    reported_flag = int(all_reported and not source_conflict and not within_uid_conflict)
    positive_flag = int(reported_flag == 1 and total is not None and total > 0)
    share_defined = positive_flag == 1
    shares: dict[str, float] = {}
    metrics = concentration_metrics({})
    if share_defined and total is not None and total > 0:
        shares = {uid: present[uid] / total for uid in sorted(present)}
        metrics = concentration_metrics(shares)
    return {
        "expected_sites": expected,
        "reported_sites": reported,
        "group_total_visits": 0.0 if reason == "all_zero_group_total" else total,
        "reported_months": reported_flag,
        "positive_total_months": positive_flag,
        "share_defined": share_defined,
        "source_conflict": bool(source_conflict),
        "exclusion_reason": "" if share_defined else reason,
        "shares": shares,
        **metrics,
    }


def adjacent_share_tv(
    previous_month_defined: bool,
    current_month_defined: bool,
    previous_shares: dict[str, float] | None,
    current_shares: dict[str, float] | None,
) -> float | None:
    if not previous_month_defined or not current_month_defined:
        return None
    return share_total_variation(previous_shares, current_shares)


def candidate_common_shock(year_month: str) -> bool:
    return SHOCK_START <= str(year_month) <= SHOCK_END


def candidate_structural_break(previous_hhi: float | None, current_hhi: float | None) -> bool:
    """A large month-to-month move is only a candidate breakpoint, not a sourced event."""
    if previous_hhi is None or current_hhi is None:
        return False
    return abs(float(current_hhi) - float(previous_hhi)) >= BREAK_THRESHOLD


def identity_semantics(statuses: list[str]) -> dict[str, object]:
    """Rule coverage is not an externally verified identity probability."""
    if not statuses:
        coverage = None
    else:
        coverage = sum(status in {"provisional_rule_match", "singleton"} for status in statuses) / len(statuses)
    return {
        "rule_match_coverage": coverage,
        "externally_verified_identity_status": "unknown",
    }


def hypothetical_scaled_hhi(visits: dict[str, float], site_uid: str, factor: float) -> float | None:
    """Rescale one site by a pre-specified factor. The factor is not a calibration coefficient."""
    if site_uid not in visits or factor <= 0:
        return None
    scaled = {uid: (float(value) * factor if uid == site_uid else float(value)) for uid, value in visits.items()}
    metrics = concentration_metrics(scaled)
    value = metrics["normalized_hhi"]
    return None if value is None else float(value)


def retrospective_origin_count(
    defined_months: list[str],
    train_months: int = MIN_RETRO_TRAIN_MONTHS,
    horizon: int = RETRO_HORIZON_MONTHS,
) -> int:
    """Count calendar blocks that could host a retrospective window. This does not fit a model."""
    defined = set(defined_months)
    count = 0
    for start in sorted(defined):
        evaluation = _forward(start, horizon)
        if any(month not in defined for month in evaluation):
            continue
        train_end = previous_month(start)
        training = _backward(train_end, train_months)
        if any(month not in defined for month in training):
            continue
        count += 1
    return count


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 3:
        return None
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    numerator = sum((x_value - mean_left) * (y_value - mean_right) for x_value, y_value in zip(left, right))
    left_scale = math.sqrt(sum((value - mean_left) ** 2 for value in left))
    right_scale = math.sqrt(sum((value - mean_right) ** 2 for value in right))
    if left_scale == 0 or right_scale == 0:
        return None
    return numerator / (left_scale * right_scale)


def series_diagnostics(rows: list[dict[str, object]]) -> dict[str, object]:
    defined = [row for row in rows if row.get("share_defined")]
    values = [float(row["normalized_hhi"]) for row in defined if row.get("normalized_hhi") is not None]
    pairs = []
    by_month = {str(row["year_month"]): row for row in defined}
    for year_month, row in sorted(by_month.items()):
        previous = by_month.get(previous_month(year_month))
        if previous is not None:
            pairs.append((previous, row))
    hhi_lag = pearson(
        [float(left["normalized_hhi"]) for left, _right in pairs if left.get("normalized_hhi") is not None and _right.get("normalized_hhi") is not None],
        [float(right["normalized_hhi"]) for left, right in pairs if left.get("normalized_hhi") is not None and right.get("normalized_hhi") is not None],
    )
    share_correlations = _share_lag_correlations(pairs)
    deltas_hhi = []
    deltas_total = []
    for left, right in pairs:
        if left.get("normalized_hhi") is None or right.get("normalized_hhi") is None:
            continue
        left_total = float(left.get("group_total_visits") or 0)
        right_total = float(right.get("group_total_visits") or 0)
        if left_total <= 0 or right_total <= 0:
            continue
        deltas_hhi.append(float(right["normalized_hhi"]) - float(left["normalized_hhi"]))
        deltas_total.append(math.log(right_total) - math.log(left_total))
    turnovers = [row.get("leading_site_turnover") for _left, row in pairs]
    resolved = [value for value in turnovers if value is not None]
    seasonal: dict[int, list[float]] = defaultdict(list)
    for row in defined:
        if row.get("normalized_hhi") is None:
            continue
        seasonal[int(str(row["year_month"]).split("-")[1])].append(float(row["normalized_hhi"]))
    seasonal_means = {month: sum(bucket) / len(bucket) for month, bucket in seasonal.items() if bucket}
    amplitude = None
    if seasonal_means:
        amplitude = max(seasonal_means.values()) - min(seasonal_means.values())
    first = values[:12]
    last = values[-12:]
    return {
        "defined_months": len(defined),
        "normalized_hhi_mean": _mean(values),
        "normalized_hhi_min": min(values) if values else None,
        "normalized_hhi_max": max(values) if values else None,
        "normalized_hhi_sd": _sd(values),
        "first_12_mean": _mean(first),
        "last_12_mean": _mean(last),
        "mean_tv": _mean([float(row["month_to_month_share_tv"]) for row in defined if row.get("month_to_month_share_tv") is not None]),
        "hhi_lag1_correlation": hhi_lag,
        "median_share_lag1_correlation": _median(share_correlations),
        "total_hhi_change_correlation": pearson(deltas_total, deltas_hhi),
        "turnover_resolved_pairs": len(resolved),
        "turnover_count": sum(1 for value in resolved if value),
        "seasonal_amplitude": amplitude,
        "candidate_break_count": sum(1 for row in defined if row.get("candidate_structural_break")),
        "retrospective_origins": retrospective_origin_count([str(row["year_month"]) for row in defined]),
    }


def _share_lag_correlations(pairs: list[tuple[dict[str, object], dict[str, object]]]) -> list[float]:
    by_site: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for left, right in pairs:
        left_shares = dict(left.get("shares") or {})
        right_shares = dict(right.get("shares") or {})
        for uid in left_shares.keys() & right_shares.keys():
            by_site[uid].append((float(left_shares[uid]), float(right_shares[uid])))
    correlations = []
    for samples in by_site.values():
        value = pearson([item[0] for item in samples], [item[1] for item in samples])
        if value is not None:
            correlations.append(value)
    return correlations


def _forward(start: str, count: int) -> list[str]:
    months = [start]
    current = start
    for _ in range(count - 1):
        year, month = (int(part) for part in current.split("-"))
        month += 1
        if month == 13:
            month = 1
            year += 1
        current = f"{year:04d}-{month:02d}"
        months.append(current)
    return months


def _backward(end: str, count: int) -> list[str]:
    months = [end]
    current = end
    for _ in range(count - 1):
        current = previous_month(current)
        months.append(current)
    return months


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _sd(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    center = sum(values) / len(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2)
