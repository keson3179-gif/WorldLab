from __future__ import annotations

from worldlab_data.phase11 import concentration_metrics, month_span, previous_month, share_total_variation
from worldlab_data.phase12 import valid_share_vector

ANALYSIS_ROLE = "exploratory_retrospective"
TARGET_NAME = "reported_visit_distribution"
COMPARABILITY_LABEL = "reported_visits_conditional"
PREDICTION_HORIZON_MONTHS = 1
TRAIN_MONTHS = 24
EVALUATION_MONTHS = 12
HOLDOUT_YEAR = 2025
MIN_BLOCKS_FOR_INTERVAL = 30
FIXED_RECENT_WEIGHT = 0.5
FIXED_SEASONAL_WEIGHT = 0.5
WEIGHTS_TUNED_ON_EVALUATION = False
SHOCK_START = "2020-01"
SHOCK_END = "2022-12"

EVALUATION_PARENTS = (
    "玉山國家公園",
    "太魯閣國家公園",
    "雪霸國家公園",
    "金門國家公園",
)

SCENARIO_LAG = {
    "ex_post_max_information": 1,
    "assumed_publication_delay": 3,
}

MODELS = (
    "persistence",
    "same_month_last_year",
    "seasonal_month_mean",
    "trailing_3_mean",
    "trailing_12_mean",
    "fixed_equal_weight",
    "uniform_reference",
)

RECENT_MODELS = (
    "persistence",
    "trailing_3_mean",
    "trailing_12_mean",
    "fixed_equal_weight",
)


def shift_month(year_month: str, delta: int) -> str:
    year, month = (int(part) for part in str(year_month).split("-"))
    index = year * 12 + (month - 1) + int(delta)
    if index < 0:
        raise ValueError(year_month)
    return f"{index // 12:04d}-{(index % 12) + 1:02d}"


def information_cutoff(target: str, scenario: str) -> str:
    return shift_month(target, -SCENARIO_LAG[scenario])


def reject_holdout_months(year_months: list[str], holdout_year: int = HOLDOUT_YEAR) -> tuple[list[str], list[str]]:
    kept: list[str] = []
    held: list[str] = []
    for year_month in year_months:
        if int(str(year_month)[:4]) >= holdout_year:
            held.append(str(year_month))
        else:
            kept.append(str(year_month))
    return kept, held


def is_defined_share_month(shares: dict[str, float] | None, membership: tuple[str, ...]) -> bool:
    """A zero total, a missing site, or a broken simplex is not a usable month."""
    if not shares or not membership:
        return False
    if set(shares) != set(membership):
        return False
    values = [float(shares[uid]) for uid in membership]
    if sum(values) <= 0:
        return False
    return valid_share_vector(values)


def rolling_origins(
    defined_months: list[str] | set[str],
    train_months: int = TRAIN_MONTHS,
    horizon: int = EVALUATION_MONTHS,
) -> list[str]:
    """Starts of calendar blocks with train_months before them and horizon defined months inside."""
    defined = set(defined_months)
    origins: list[str] = []
    for start in sorted(defined):
        window = month_span(start, shift_month(start, horizon - 1))
        if len(window) != horizon or any(month not in defined for month in window):
            continue
        train_end = previous_month(start)
        train_start = shift_month(train_end, -(train_months - 1))
        training = month_span(train_start, train_end)
        if len(training) != train_months or any(month not in defined for month in training):
            continue
        origins.append(start)
    return origins


def evaluable_months(
    defined_months: list[str] | set[str],
    train_months: int = TRAIN_MONTHS,
    horizon: int = EVALUATION_MONTHS,
) -> list[str]:
    """Unique one-step target months. Overlapping origins do not repeat a month."""
    selected: set[str] = set()
    for start in rolling_origins(defined_months, train_months, horizon):
        selected.update(month_span(start, shift_month(start, horizon - 1)))
    return sorted(selected)


def nonoverlapping_blocks(months: list[str], size: int = EVALUATION_MONTHS) -> list[list[str]]:
    remaining = set(months)
    blocks: list[list[str]] = []
    for start in sorted(remaining):
        if start not in remaining:
            continue
        window = month_span(start, shift_month(start, size - 1))
        if len(window) == size and all(month in remaining for month in window):
            blocks.append(window)
            remaining.difference_update(window)
    return blocks


def independent_sample_size(unique_months: list[str]) -> int:
    """Calendar months are the descriptive count. Rolling origins are not extra samples."""
    return len(set(unique_months))


def common_scoring_months(available_by_model: dict[str, set[str]]) -> list[str]:
    if not available_by_model:
        return []
    shared = set.intersection(*(set(months) for months in available_by_model.values()))
    return sorted(shared)


def pandemic_stratum(year_month: str) -> str:
    if str(year_month) < SHOCK_START:
        return "before_2020"
    if str(year_month) <= SHOCK_END:
        return "candidate_common_shock_2020_2022"
    return "after_2022"


def _usable(history: dict[str, dict[str, float]], month: str, membership: tuple[str, ...]) -> dict[str, float] | None:
    if not is_defined_share_month(history.get(month), membership):
        return None
    return {uid: float(history[month][uid]) for uid in membership}


def _mean_vectors(vectors: list[dict[str, float]], membership: tuple[str, ...]) -> dict[str, float] | None:
    if not vectors:
        return None
    averaged = {
        uid: sum(vector[uid] for vector in vectors) / len(vectors)
        for uid in membership
    }
    if not valid_share_vector([averaged[uid] for uid in membership]):
        return None
    return averaged


def _unavailable(reason: str, cutoff: str, lineage: str) -> dict[str, object]:
    return {
        "status": "prediction_unavailable",
        "skip_reason": reason,
        "shares": None,
        "source_months": [],
        "feature_lineage": lineage,
        "information_cutoff": cutoff,
        "prediction_horizon_months": PREDICTION_HORIZON_MONTHS,
    }


def _ready(
    shares: dict[str, float],
    source_months: list[str],
    lineage: str,
    cutoff: str,
) -> dict[str, object]:
    copied = {uid: float(shares[uid]) for uid in shares}
    return {
        "status": "predicted",
        "skip_reason": "",
        "shares": copied,
        "source_months": list(source_months),
        "feature_lineage": lineage,
        "information_cutoff": cutoff,
        "prediction_horizon_months": PREDICTION_HORIZON_MONTHS,
    }


def _trailing(history: dict[str, dict[str, float]], membership: tuple[str, ...], cutoff: str, width: int) -> dict[str, object]:
    window = [shift_month(cutoff, offset) for offset in range(-(width - 1), 1)]
    vectors: list[dict[str, float]] = []
    for month in window:
        vector = _usable(history, month, membership)
        if vector is None:
            return _unavailable("trailing_window_incomplete", cutoff, f"trailing_{width}_mean:incomplete")
        vectors.append(vector)
    averaged = _mean_vectors(vectors, membership)
    if averaged is None:
        return _unavailable("invalid_simplex", cutoff, f"trailing_{width}_mean:invalid")
    return _ready(averaged, window, f"trailing_{width}_mean:" + ",".join(window), cutoff)


def predict_shares(
    history: dict[str, dict[str, float]],
    membership: tuple[str, ...],
    target: str,
    scenario: str,
    model: str,
) -> dict[str, object]:
    """One-step share forecast. History at or after the cutoff is not a feature."""
    if model not in MODELS:
        raise ValueError(model)
    if scenario not in SCENARIO_LAG:
        raise ValueError(scenario)
    cutoff = information_cutoff(target, scenario)
    if not membership:
        return _unavailable("empty_membership", cutoff, f"{model}:empty_membership")
    if model == "uniform_reference":
        share = 1.0 / len(membership)
        return _ready({uid: share for uid in membership}, [], "uniform_reference:no_fitted_parameter", cutoff)
    if model == "persistence":
        vector = _usable(history, cutoff, membership)
        if vector is None:
            return _unavailable("cutoff_month_undefined", cutoff, f"persistence:{cutoff}:missing")
        return _ready(vector, [cutoff], f"persistence:{cutoff}", cutoff)
    if model == "same_month_last_year":
        source = shift_month(target, -12)
        if source > cutoff:
            return _unavailable("source_after_cutoff", cutoff, f"same_month_last_year:{source}:after_cutoff")
        vector = _usable(history, source, membership)
        if vector is None:
            return _unavailable("same_month_last_year_missing", cutoff, f"same_month_last_year:{source}:missing")
        return _ready(vector, [source], f"same_month_last_year:{source}", cutoff)
    if model == "seasonal_month_mean":
        calendar_month = str(target)[-2:]
        sources = sorted(
            month
            for month in history
            if month <= cutoff and month < target and str(month)[-2:] == calendar_month
        )
        vectors = []
        used = []
        for month in sources:
            vector = _usable(history, month, membership)
            if vector is None:
                continue
            vectors.append(vector)
            used.append(month)
        averaged = _mean_vectors(vectors, membership)
        if averaged is None:
            return _unavailable("no_prior_same_calendar_month", cutoff, "seasonal_month_mean:none")
        return _ready(averaged, used, "seasonal_month_mean:" + ",".join(used), cutoff)
    if model == "trailing_3_mean":
        return _trailing(history, membership, cutoff, 3)
    if model == "trailing_12_mean":
        return _trailing(history, membership, cutoff, 12)
    recent = predict_shares(history, membership, target, scenario, "persistence")
    seasonal = predict_shares(history, membership, target, scenario, "seasonal_month_mean")
    if recent["status"] != "predicted" or seasonal["status"] != "predicted":
        return _unavailable(
            "component_unavailable",
            cutoff,
            "fixed_equal_weight:0.5*persistence+0.5*seasonal_month_mean:unavailable",
        )
    recent_shares = dict(recent["shares"])
    seasonal_shares = dict(seasonal["shares"])
    combined = {
        uid: FIXED_RECENT_WEIGHT * float(recent_shares[uid]) + FIXED_SEASONAL_WEIGHT * float(seasonal_shares[uid])
        for uid in membership
    }
    if not valid_share_vector([combined[uid] for uid in membership]):
        return _unavailable("invalid_simplex", cutoff, "fixed_equal_weight:invalid")
    source_months = sorted(set(recent["source_months"]) | set(seasonal["source_months"]))
    lineage = (
        f"fixed_equal_weight:{FIXED_RECENT_WEIGHT}*persistence+{FIXED_SEASONAL_WEIGHT}*seasonal_month_mean|"
        + str(recent["feature_lineage"])
        + "|"
        + str(seasonal["feature_lineage"])
    )
    return _ready(combined, source_months, lineage, cutoff)


def leakage_failures(prediction: dict[str, object], target: str) -> list[str]:
    failures = []
    cutoff = str(prediction["information_cutoff"])
    if cutoff != information_cutoff(target, _scenario_from_cutoff(target, cutoff)):
        failures.append("cutoff_mismatch")
    if int(prediction["prediction_horizon_months"]) != PREDICTION_HORIZON_MONTHS:
        failures.append("horizon_mismatch")
    for month in prediction["source_months"]:
        if str(month) > cutoff or str(month) >= target:
            failures.append(f"source_after_cutoff:{month}")
    shares = prediction["shares"]
    if prediction["status"] == "predicted":
        if not isinstance(shares, dict) or not valid_share_vector([float(shares[key]) for key in shares]):
            failures.append("invalid_simplex")
    elif shares is not None:
        failures.append("unavailable_with_shares")
    return failures


def _scenario_from_cutoff(target: str, cutoff: str) -> str:
    for scenario, lag in SCENARIO_LAG.items():
        if information_cutoff(target, scenario) == cutoff:
            return scenario
    return "ex_post_max_information"


def score_share_forecast(actual: dict[str, float], predicted: dict[str, float]) -> dict[str, object]:
    if set(actual) != set(predicted):
        return {"status": "membership_mismatch"}
    if not valid_share_vector([float(value) for value in predicted.values()]):
        return {"status": "invalid_simplex"}
    actual_metrics = concentration_metrics(actual)
    predicted_metrics = concentration_metrics(predicted)
    actual_hhi = actual_metrics["normalized_hhi"]
    predicted_hhi = predicted_metrics["normalized_hhi"]
    hhi_error = None
    if actual_hhi is not None and predicted_hhi is not None:
        hhi_error = abs(float(predicted_hhi) - float(actual_hhi))
    actual_leader = str(actual_metrics["leading_site_uid"])
    predicted_leader = str(predicted_metrics["leading_site_uid"])
    if actual_metrics["leader_tie"] or predicted_metrics["leader_tie"] or not actual_leader or not predicted_leader:
        leader_hit = None
    else:
        leader_hit = actual_leader == predicted_leader
    return {
        "status": "scored",
        "share_vector_TV": share_total_variation(actual, predicted),
        "share_MAE": sum(abs(float(predicted[uid]) - float(actual[uid])) for uid in actual) / len(actual),
        "normalized_HHI_absolute_error": hhi_error,
        "leading_site_hit": leader_hit,
        "share_bias": {uid: float(predicted[uid]) - float(actual[uid]) for uid in sorted(actual)},
    }


def summarize_scored(rows: list[dict[str, object]]) -> dict[str, object]:
    scored = [row for row in rows if row.get("status") == "scored"]
    hits = [row["leading_site_hit"] for row in scored if row.get("leading_site_hit") is not None]
    hhi_values = [
        float(row["normalized_HHI_absolute_error"])
        for row in scored
        if row.get("normalized_HHI_absolute_error") is not None
    ]
    decisive = len(hits)
    return {
        "n_months": len(scored),
        "share_vector_TV": _mean([float(row["share_vector_TV"]) for row in scored]),
        "share_MAE": _mean([float(row["share_MAE"]) for row in scored]),
        "normalized_HHI_absolute_error": _mean(hhi_values),
        "leading_site_hits": sum(1 for hit in hits if hit),
        "leading_site_decisive": decisive,
        "leading_site_accuracy": None if decisive == 0 else sum(1 for hit in hits if hit) / decisive,
    }


def delay_increase(score_a: float | None, score_b: float | None) -> float | None:
    if score_a is None or score_b is None:
        return None
    return float(score_b) - float(score_a)


def lower_error(left: float | None, right: float | None, tolerance: float = 1e-8) -> str:
    if left is None or right is None:
        return "unavailable"
    if abs(left - right) <= tolerance:
        return "tie"
    return "left" if left < right else "right"


def adapter_research_role(evaluable_month_count: int, nonoverlap_block_count: int) -> str:
    """Length and block count only. Prediction error does not choose the group."""
    if evaluable_month_count >= 96 and nonoverlap_block_count >= 4:
        return "candidate_for_adapter_research"
    if evaluable_month_count >= 12:
        return "short_series_contrast_only"
    return "not_enough_evaluable_months"


def interval_note(nonoverlap_block_count: int) -> str:
    if nonoverlap_block_count < MIN_BLOCKS_FOR_INTERVAL:
        return "有效非重疊區塊數不足，本輪不估計信賴區間。"
    return "區塊數達到門檻時也只報告區塊平均的範圍，不把 p 值當成模型等價。"


def is_priority_identity(category_change_reason: str, months_match_excel: bool) -> bool:
    return category_change_reason == "unresolved" and bool(months_match_excel)


def classify_priority_identity(
    months_match_excel: bool,
    human_reviewed: bool,
    method_2023: str,
    method_2024: str,
    duplicate_excel_name_rows: int,
    attraction_name_found: bool,
) -> dict[str, object]:
    """Twelve matching months are a same-year cross-source candidate, not a manual confirmation."""
    method_note = _method_note(method_2023, method_2024)
    if human_reviewed:
        review_status = "human_reviewed_pending_record"
    else:
        review_status = "cross_source_evidence_candidate" if months_match_excel else "not_priority"
    return {
        "same_year_statistical_correspondence": bool(months_match_excel),
        "physical_or_statistical_identity_continuity": review_status,
        "measurement_continuity": "not_established",
        "method_label_note": method_note,
        "duplicate_excel_name_rows_2024": int(duplicate_excel_name_rows),
        "local_attraction_name_found": bool(attraction_name_found),
        "split_merge_evidence": "no_documented_split_or_merge_in_sources_checked",
        "review_status": review_status,
        "confirmed_manual": False,
        "human_reviewed": bool(human_reviewed),
        "baseline_sample_changed": False,
        "evidence_note": _evidence_note(months_match_excel, method_note, duplicate_excel_name_rows, attraction_name_found),
    }


def _method_note(method_2023: str, method_2024: str) -> str:
    left = (method_2023 or "").strip()
    right = (method_2024 or "").strip()
    if left and right and left != right:
        return "method_label_differs"
    if left and right and left == right:
        return "same_method_label_not_documented_comparable"
    return "method_label_missing"


def _evidence_note(months_match: bool, method_note: str, duplicates: int, attraction_found: bool) -> str:
    parts = []
    if months_match:
        parts.append("2024 年 CSV 與 Excel 十二個月人次一致，只支持同年兩份來源對到同一統計列。")
    parts.append("這不能單獨證明與 2023 年的統計口徑連續，也沒有升成 confirmed_manual。")
    parts.append(f"計數標籤狀態是 {method_note}。")
    if duplicates > 1:
        parts.append("同年 Excel 有同名重複列。")
    else:
        parts.append("同年 Excel 沒有同名重複列。")
    if attraction_found:
        parts.append("地方景點清單有同名，沒有拆分或合併文件。")
    else:
        parts.append("地方景點清單沒有同名，這不構成範圍變動證據。")
    return "".join(parts)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))
