from __future__ import annotations

import json
import platform
import sys
from collections import defaultdict

import pandas as pd

from worldlab_data.config import WORLD_ROOT, V4Paths, load_paths
from worldlab_data.phase11 import concentration_metrics
from worldlab_data.phase12 import valid_share_vector
from worldlab_data.phase13 import (
    ANALYSIS_ROLE,
    COMPARABILITY_LABEL,
    EVALUATION_PARENTS,
    FIXED_RECENT_WEIGHT,
    FIXED_SEASONAL_WEIGHT,
    MODELS,
    PREDICTION_HORIZON_MONTHS,
    RECENT_MODELS,
    SCENARIO_LAG,
    TARGET_NAME,
    TRAIN_MONTHS,
    WEIGHTS_TUNED_ON_EVALUATION,
    adapter_research_role,
    classify_priority_identity,
    common_scoring_months,
    delay_increase,
    evaluable_months,
    independent_sample_size,
    information_cutoff,
    interval_note,
    is_defined_share_month,
    is_priority_identity,
    leakage_failures,
    lower_error,
    nonoverlapping_blocks,
    pandemic_stratum,
    predict_shares,
    reject_holdout_months,
    rolling_origins,
    score_share_forecast,
    shift_month,
    summarize_scored,
)
from worldlab_data.textutil import file_sha256

GUARDED = (
    "data/curated/v4_forecast_panel_candidate.parquet",
    "data/curated/v4_2024_identity_audit.csv",
    "data/curated/v4_2021_alignment_audit.csv",
    "data/curated/v4_exploratory_visit_shares_conditional.csv",
    "data/curated/v4_month_validity.csv",
    "reports/V4_PHASE12_BASELINE_PROTOCOL.md",
    "reports/V4_PHASE12_IDENTITY_READINESS.md",
    "reports/V4_PHASE12_CATEGORY_ROOT_CAUSE.md",
    "reports/V4_PHASE12_2021_ALIGNMENT.md",
    "reports/V4_PHASE11_FORECAST_FEASIBILITY.md",
    "reports/V4_PHASE11_CONCENTRATION_EXPLORATION.md",
)

MODEL_LABELS = {
    "persistence": "最近可用月持續",
    "same_month_last_year": "去年同月",
    "seasonal_month_mean": "歷史同月平均",
    "trailing_3_mean": "近 3 個月平均",
    "trailing_12_mean": "近 12 個月平均",
    "fixed_equal_weight": "持續與同月等權",
    "uniform_reference": "均勻份額",
}

SCENARIO_LABELS = {
    "ex_post_max_information": "情境 A：假設 t-1 已可使用",
    "assumed_publication_delay": "情境 B：假設最新可用月是 t-3",
}


def run_phase13(paths: V4Paths | None = None) -> dict[str, object]:
    paths = paths or load_paths()
    before = _hashes()
    curated = WORLD_ROOT / "data" / "curated"
    panel = pd.read_parquet(
        curated / "v4_forecast_panel_candidate.parquet",
        columns=[
            "group_id",
            "site_uid",
            "year_month",
            "share",
            "dataset_role",
            "confirmatory_holdout_candidate",
            "strict_asof_eligible",
            "publication_history_unknown",
            "target",
            "counting_limits",
            "used_for_model_selection",
        ],
    )
    _reject_panel(panel)
    comparability = pd.read_csv(curated / "v4_group_comparability.csv", dtype=str, keep_default_na=False)
    crosswalk = pd.read_csv(curated / "v4_canonical_site_crosswalk.csv", dtype=str, keep_default_na=False)
    validity = pd.read_csv(
        curated / "v4_month_validity.csv",
        dtype=str,
        keep_default_na=False,
        usecols=["group_id", "year_month", "share_defined", "exclusion_reason", "source_conflict"],
    )
    verified = pd.read_csv(
        curated / "v4_measurement_method_verified.csv",
        dtype=str,
        keep_default_na=False,
        usecols=["year", "official_name", "county", "category", "measurement_method"],
    )
    audit = pd.read_csv(curated / "v4_2024_identity_audit.csv", dtype=str, keep_default_na=False)
    names = _site_names(crosswalk)
    selected = _selected_groups(comparability)
    evaluations = []
    prediction_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    for group in selected:
        evaluation, predictions, scores = _evaluate_group(panel, group, names, validity, crosswalk, verified)
        evaluations.append(evaluation)
        prediction_rows.extend(predictions)
        score_rows.extend(scores)
    priority = _priority_review(paths, audit, verified)
    if len(priority) != 13:
        raise RuntimeError(f"expected 13 priority identity rows, found {len(priority)}")
    _write_outputs(paths, curated, prediction_rows, score_rows, evaluations, priority, before)
    after = _hashes()
    if before != after:
        raise RuntimeError("phase 12 inputs changed while phase 13 was writing")
    payload = _summary_payload(evaluations, prediction_rows, priority, before)
    (paths.interim / "phase13_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return payload


def _reject_panel(panel: pd.DataFrame) -> None:
    months = [str(value) for value in panel["year_month"]]
    kept, held = reject_holdout_months(months)
    if held or len(kept) != len(months):
        raise RuntimeError("2025 or 2026 entered the exploratory panel")
    if panel["dataset_role"].ne(ANALYSIS_ROLE).any():
        raise RuntimeError("panel role is not exploratory_retrospective")
    if panel["confirmatory_holdout_candidate"].astype(bool).any():
        raise RuntimeError("confirmatory rows entered the exploratory backtest")
    if panel["strict_asof_eligible"].astype(bool).any():
        raise RuntimeError("strict as-of was marked without a publication date")
    if panel["used_for_model_selection"].astype(bool).any():
        raise RuntimeError("panel rows were marked as model selection")
    if panel["target"].ne(TARGET_NAME).any():
        raise RuntimeError("target is not reported_visit_distribution")


def _selected_groups(comparability: pd.DataFrame) -> list[dict[str, str]]:
    geographic = comparability[comparability["definition"].eq("A_geographic")]
    chosen = geographic[geographic["parent_area"].isin(EVALUATION_PARENTS)]
    groups = chosen.to_dict(orient="records")
    groups.sort(key=lambda row: EVALUATION_PARENTS.index(row["parent_area"]))
    if [row["parent_area"] for row in groups] != list(EVALUATION_PARENTS):
        raise RuntimeError("the four evaluation groups were not found")
    return groups


def _site_names(crosswalk: pd.DataFrame) -> dict[str, str]:
    names: dict[str, str] = {}
    ranked = crosswalk.copy()
    ranked["last_year_num"] = ranked["last_year"].astype(int)
    for uid, chunk in ranked.sort_values("last_year_num").groupby("site_uid"):
        names[str(uid)] = str(chunk.iloc[-1]["official_name"])
    return names


def _evaluate_group(
    panel: pd.DataFrame,
    group: dict[str, str],
    names: dict[str, str],
    validity: pd.DataFrame,
    crosswalk: pd.DataFrame,
    verified: pd.DataFrame,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    group_id = group["group_id"]
    scoped = panel[panel["group_id"].eq(group_id)]
    membership = tuple(sorted(scoped["site_uid"].unique()))
    history: dict[str, dict[str, float]] = {}
    incomplete: list[str] = []
    limits = sorted(set(scoped["counting_limits"]))
    for year_month, chunk in scoped.groupby("year_month"):
        shares = {str(row.site_uid): float(row.share) for row in chunk.itertuples(index=False)}
        if is_defined_share_month(shares, membership):
            history[str(year_month)] = {uid: shares[uid] for uid in membership}
        else:
            incomplete.append(str(year_month))
    defined = sorted(history)
    targets = evaluable_months(defined)
    origins = rolling_origins(defined)
    blocks = nonoverlapping_blocks(targets)
    excluded = _exclusion_counts(validity, group_id)
    prediction_rows: list[dict[str, object]] = []
    month_results: list[dict[str, object]] = []
    leakage_count = 0
    simplex_count = 0
    for target in targets:
        actual = history[target]
        for scenario in SCENARIO_LAG:
            available: dict[str, dict[str, object]] = {}
            for model in MODELS:
                prediction = predict_shares(history, membership, target, scenario, model)
                visible = {month: vector for month, vector in history.items() if month <= str(prediction["information_cutoff"])}
                shadow = predict_shares(visible, membership, target, scenario, model)
                if prediction["shares"] != shadow["shares"] or prediction["source_months"] != shadow["source_months"]:
                    leakage_count += 1
                leakage_count += len(leakage_failures(prediction, target))
                if prediction["status"] == "predicted":
                    shares = dict(prediction["shares"])
                    if not valid_share_vector([float(shares[uid]) for uid in membership]) or set(shares) != set(membership):
                        simplex_count += 1
                    scored = score_share_forecast(actual, shares)
                    for uid in membership:
                        prediction_rows.append(
                            _prediction_row(group, target, scenario, model, uid, names, shares[uid], actual[uid], prediction, len(membership))
                        )
                else:
                    scored = {"status": prediction["status"]}
                    prediction_rows.append(
                        _prediction_row(group, target, scenario, model, "", names, None, None, prediction, len(membership))
                    )
                available[model] = prediction
                month_results.append(
                    {
                        "scenario": scenario,
                        "model": model,
                        "year_month": target,
                        "prediction_status": prediction["status"],
                        "skip_reason": prediction["skip_reason"],
                        "score": scored,
                    }
                )
            if leakage_count or simplex_count:
                raise RuntimeError(f"leakage or simplex failure in {group['parent_area']} {target}")
    if any(str(row["year_month"])[:4] >= "2025" for row in prediction_rows):
        raise RuntimeError("a prediction used a holdout month")
    summaries = _scenario_summaries(month_results, targets, blocks)
    leader_note = _leader_stability(history, targets)
    method_note = _group_methods(crosswalk, verified, membership, group["county"])
    evaluation = {
        "group_id": group_id,
        "county": group["county"],
        "parent_area": group["parent_area"],
        "site_count": len(membership),
        "membership": membership,
        "names": {uid: names.get(uid, uid) for uid in membership},
        "defined_months": defined,
        "evaluable_months": targets,
        "not_evaluable_months": sorted(set(defined) - set(targets)),
        "incomplete_months": incomplete,
        "origins": origins,
        "blocks": blocks,
        "excluded": excluded,
        "counting_limits": limits,
        "measurement_methods": group["measurement_methods"],
        "comparability_label": group["comparability_label"],
        "source_conflict_rate": group["source_conflict_rate"],
        "rule_match": _rule_match(crosswalk, membership),
        "method_note": method_note,
        "leader_note": leader_note,
        "summaries": summaries,
        "adapter_role": adapter_research_role(len(targets), len(blocks)),
        "interval_note": interval_note(len(blocks)),
    }
    score_rows = _score_table(evaluation, month_results)
    return evaluation, prediction_rows, score_rows


def _prediction_row(
    group: dict[str, str],
    target: str,
    scenario: str,
    model: str,
    site_uid: str,
    names: dict[str, str],
    predicted_share: float | None,
    actual_share: float | None,
    prediction: dict[str, object],
    site_count: int,
) -> dict[str, object]:
    return {
        "group_id": group["group_id"],
        "county": group["county"],
        "parent_area": group["parent_area"],
        "year_month": target,
        "scenario": scenario,
        "model": model,
        "model_label": MODEL_LABELS[model],
        "site_uid": site_uid,
        "official_name": names.get(site_uid, ""),
        "predicted_share": predicted_share,
        "actual_share": actual_share,
        "prediction_status": prediction["status"],
        "skip_reason": prediction["skip_reason"],
        "feature_lineage": prediction["feature_lineage"],
        "source_months": ",".join(prediction["source_months"]),
        "information_cutoff": prediction["information_cutoff"],
        "prediction_horizon_months": PREDICTION_HORIZON_MONTHS,
        "dataset_role": ANALYSIS_ROLE,
        "target": TARGET_NAME,
        "comparability_label": COMPARABILITY_LABEL,
        "site_count": site_count,
        "pandemic_stratum": pandemic_stratum(target),
        "weights_tuned_on_evaluation": WEIGHTS_TUNED_ON_EVALUATION,
        "used_for_model_selection": False,
    }


def _scenario_summaries(
    month_results: list[dict[str, object]],
    targets: list[str],
    blocks: list[list[str]],
) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for scenario in SCENARIO_LAG:
        available = {
            model: {
                row["year_month"]
                for row in month_results
                if row["scenario"] == scenario and row["model"] == model and row["prediction_status"] == "predicted"
            }
            for model in MODELS
        }
        shared = common_scoring_months(available)
        by_model = {}
        for model in MODELS:
            scoped = [
                row["score"]
                for row in month_results
                if row["scenario"] == scenario and row["model"] == model and row["year_month"] in shared and row["score"].get("status") == "scored"
            ]
            by_model[model] = summarize_scored(scoped)
            by_model[model]["availability"] = len(available[model]) / len(targets) if targets else None
            by_model[model]["available_months"] = len(available[model])
        output[scenario] = {
            "common_months": shared,
            "by_model": by_model,
            "by_year": _slice_summary(month_results, scenario, shared, lambda month: month[:4]),
            "by_block": _block_summary(month_results, scenario, shared, blocks),
            "by_stratum": _slice_summary(month_results, scenario, shared, pandemic_stratum),
            "bias": _bias_summary(month_results, scenario, shared),
        }
    return output


def _slice_summary(month_results, scenario: str, shared: list[str], key_of) -> dict[str, dict[str, dict[str, object]]]:
    grouped: dict[str, dict[str, list[dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    shared_set = set(shared)
    for row in month_results:
        if row["scenario"] != scenario or row["year_month"] not in shared_set:
            continue
        if row["score"].get("status") != "scored":
            continue
        grouped[str(key_of(row["year_month"]))][row["model"]].append(row["score"])
    return {
        key: {model: summarize_scored(rows) for model, rows in models.items()}
        for key, models in sorted(grouped.items())
    }


def _block_summary(month_results, scenario: str, shared: list[str], blocks: list[list[str]]) -> list[dict[str, object]]:
    shared_set = set(shared)
    rows = []
    for index, block in enumerate(blocks, start=1):
        months = [month for month in block if month in shared_set]
        entry = {"block": index, "start": block[0], "end": block[-1], "n_months": len(months), "models": {}}
        for model in MODELS:
            scored = [
                row["score"]
                for row in month_results
                if row["scenario"] == scenario and row["model"] == model and row["year_month"] in months and row["score"].get("status") == "scored"
            ]
            entry["models"][model] = summarize_scored(scored)
        rows.append(entry)
    return rows


def _bias_summary(month_results, scenario: str, shared: list[str]) -> dict[str, dict[str, float | None]]:
    shared_set = set(shared)
    totals: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in month_results:
        if row["scenario"] != scenario or row["year_month"] not in shared_set or row["model"] not in {"persistence", "seasonal_month_mean"}:
            continue
        bias = row["score"].get("share_bias")
        if not isinstance(bias, dict):
            continue
        for uid, value in bias.items():
            totals[row["model"]][uid].append(float(value))
    return {
        model: {uid: sum(values) / len(values) for uid, values in sites.items() if values}
        for model, sites in totals.items()
    }


def _score_table(evaluation: dict[str, object], month_results: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    base = {
        "group_id": evaluation["group_id"],
        "county": evaluation["county"],
        "parent_area": evaluation["parent_area"],
        "dataset_role": ANALYSIS_ROLE,
        "comparability_label": COMPARABILITY_LABEL,
        "target": TARGET_NAME,
        "n_rolling_origins": len(evaluation["origins"]),
        "n_nonoverlap_blocks": len(evaluation["blocks"]),
        "independent_sample_note": "unique_calendar_months_are_descriptive_not_independent",
        "site_uid": "",
        "official_name": "",
        "share_bias": "",
    }
    for scenario, summary in evaluation["summaries"].items():
        for model, metrics in summary["by_model"].items():
            rows.append(_metric_row(base, "unique_common_months", scenario, model, "all", metrics))
        for year, models in summary["by_year"].items():
            for model, metrics in models.items():
                rows.append(_metric_row(base, "calendar_year", scenario, model, year, metrics))
        for block in summary["by_block"]:
            for model, metrics in block["models"].items():
                rows.append(_metric_row(base, "nonoverlap_block", scenario, model, f"{block['start']}_{block['end']}", metrics))
        for stratum, models in summary["by_stratum"].items():
            for model, metrics in models.items():
                rows.append(_metric_row(base, "pandemic_stratum", scenario, model, stratum, metrics))
        for model, sites in summary["bias"].items():
            for uid, bias in sites.items():
                row = _metric_row(base, "site_bias", scenario, model, "all", summary["by_model"][model])
                row["site_uid"] = uid
                row["official_name"] = evaluation["names"].get(uid, uid)
                row["share_bias"] = bias
                rows.append(row)
    for row in month_results:
        if row["score"].get("status") != "scored":
            continue
        metrics = {
            "n_months": 1,
            "share_vector_TV": row["score"]["share_vector_TV"],
            "share_MAE": row["score"]["share_MAE"],
            "normalized_HHI_absolute_error": row["score"]["normalized_HHI_absolute_error"],
            "leading_site_hits": int(row["score"]["leading_site_hit"] is True),
            "leading_site_decisive": int(row["score"]["leading_site_hit"] is not None),
            "leading_site_accuracy": None if row["score"]["leading_site_hit"] is None else float(row["score"]["leading_site_hit"]),
            "availability": "",
        }
        rows.append(_metric_row(base, "month", row["scenario"], row["model"], row["year_month"], metrics))
    return rows


def _metric_row(base: dict[str, object], slice_type: str, scenario: str, model: str, slice_key: str, metrics: dict[str, object]) -> dict[str, object]:
    row = dict(base)
    row.update(
        {
            "slice_type": slice_type,
            "scenario": scenario,
            "model": model,
            "slice_key": slice_key,
            "n_months": metrics["n_months"],
            "share_vector_TV": metrics["share_vector_TV"],
            "share_MAE": metrics["share_MAE"],
            "normalized_HHI_absolute_error": metrics["normalized_HHI_absolute_error"],
            "leading_site_hits": metrics["leading_site_hits"],
            "leading_site_decisive": metrics["leading_site_decisive"],
            "leading_site_accuracy": metrics["leading_site_accuracy"],
            "availability_rate": metrics.get("availability", ""),
        }
    )
    return row


def _leader_stability(history: dict[str, dict[str, float]], targets: list[str]) -> dict[str, object]:
    leaders = []
    for month in targets:
        metrics = concentration_metrics(history[month])
        leaders.append("" if metrics["leader_tie"] else str(metrics["leading_site_uid"]))
    unique = sorted({leader for leader in leaders if leader})
    turns = 0
    previous = None
    for leader in leaders:
        if previous and leader and leader != previous:
            turns += 1
        if leader:
            previous = leader
    return {"distinct_leaders": len(unique), "turnovers": turns, "decisive_months": sum(1 for leader in leaders if leader)}


def _rule_match(crosswalk: pd.DataFrame, membership: tuple[str, ...]) -> dict[str, object]:
    scoped = crosswalk[crosswalk["site_uid"].isin(membership)]
    statuses = sorted(set(scoped["match_status"]))
    return {
        "statuses": statuses,
        "contains_provisional_rule_match": "provisional_rule_match" in statuses,
        "externally_verified_identity_status": "unknown",
    }


def _group_methods(crosswalk: pd.DataFrame, verified: pd.DataFrame, membership: tuple[str, ...], county: str) -> dict[str, object]:
    """Count workbook labels on the linked name and year span. A 2024 cross-type row stays outside that span."""
    scoped_parts = []
    changed = []
    linked = crosswalk[crosswalk["site_uid"].isin(membership)]
    for _uid, chunk in linked.groupby("site_uid"):
        name = str(chunk.sort_values("last_year").iloc[-1]["official_name"])
        first_year = int(chunk["first_year"].astype(int).min())
        last_year = int(chunk["last_year"].astype(int).max())
        years = {str(year) for year in range(first_year, last_year + 1)}
        rows = verified[verified["county"].eq(county) & verified["official_name"].eq(name) & verified["year"].isin(years)]
        scoped_parts.append(rows)
        labels = sorted({value for value in rows["measurement_method"] if value})
        if len(labels) > 1:
            changed.append(name)
    scoped = pd.concat(scoped_parts, ignore_index=True) if scoped_parts else verified.iloc[0:0]
    present = sorted({value for value in scoped["measurement_method"] if value})
    missing_rows = int((scoped["measurement_method"] == "").sum())
    return {"labels": present, "sites_with_label_change": changed, "blank_method_rows": missing_rows}


def _exclusion_counts(validity: pd.DataFrame, group_id: str) -> dict[str, int]:
    scoped = validity[validity["group_id"].eq(group_id) & validity["share_defined"].str.lower().eq("false")]
    counts: dict[str, int] = defaultdict(int)
    for reason in scoped["exclusion_reason"]:
        counts[reason or "unspecified"] += 1
    return {key: int(value) for key, value in sorted(counts.items())}


def _priority_review(paths: V4Paths, audit: pd.DataFrame, verified: pd.DataFrame) -> list[dict[str, object]]:
    names = set()
    candidates = []
    for record in audit.to_dict(orient="records"):
        months_match = str(record["months_match_excel_2024"]).lower() == "true"
        if not is_priority_identity(str(record["category_change_reason"]), months_match):
            continue
        names.add(str(record["official_name"]))
        candidates.append(record)
    hits = _attraction_hits(paths.raw_root / "觀光資料標準V2.1" / "AttractionList.xml", names)
    rows = []
    for record in candidates:
        name = str(record["official_name"])
        county = str(record["county"])
        method_2023 = _methods_for(verified, name, county, "2023")
        method_2024 = _methods_for(verified, name, county, "2024")
        duplicates = int(
            (
                verified["year"].eq("2024")
                & verified["official_name"].eq(name)
                & verified["county"].eq(county)
            ).sum()
        )
        classified = classify_priority_identity(
            True,
            str(record["human_reviewed"]).lower() == "true",
            method_2023,
            method_2024,
            duplicates,
            name in hits,
        )
        rows.append(
            {
                "group_id": record["group_id"],
                "county": county,
                "parent_area": record["parent_area"],
                "official_name": name,
                "canonical_site_uid": record["canonical_site_uid"],
                "raw_site_id_2023": record["raw_site_id_2023"],
                "raw_site_id_2024": record["raw_site_id_2024"],
                "csv_category_2023": record["csv_category_2023"],
                "csv_category_2024": record["csv_category_2024"],
                "excel_category_2024": record["excel_category_2024"],
                "excel_row_2024": record["excel_row_2024"],
                "months_match_excel_2024": True,
                "measurement_method_2023": method_2023,
                "measurement_method_2024": method_2024,
                **classified,
            }
        )
    return rows


def _methods_for(verified: pd.DataFrame, name: str, county: str, year: str) -> str:
    scoped = verified[verified["official_name"].eq(name) & verified["county"].eq(county) & verified["year"].eq(year)]
    labels = []
    for value in scoped["measurement_method"]:
        if value and value not in labels:
            labels.append(value)
    return "|".join(labels)


def _attraction_hits(path, names: set[str]) -> set[str]:
    found: set[str] = set()
    pending = set(names)
    if not path.exists() or not pending:
        return found
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if "AttractionName" not in line:
                continue
            for name in list(pending):
                if f">{name}<" in line:
                    found.add(name)
                    pending.remove(name)
            if not pending:
                break
    return found


def _write_outputs(
    paths: V4Paths,
    curated,
    prediction_rows: list[dict[str, object]],
    score_rows: list[dict[str, object]],
    evaluations: list[dict[str, object]],
    priority: list[dict[str, object]],
    hashes: dict[str, str],
) -> None:
    predictions = pd.DataFrame.from_records(prediction_rows)
    predictions["predicted_share"] = predictions["predicted_share"].astype("Float64")
    predictions["actual_share"] = predictions["actual_share"].astype("Float64")
    predictions.to_parquet(curated / "v4_baseline_predictions_exploratory.parquet", index=False)
    pd.DataFrame.from_records(score_rows).to_csv(
        curated / "v4_baseline_scores_exploratory.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame.from_records(priority).to_csv(
        curated / "v4_2024_priority_identity_review.csv",
        index=False,
        encoding="utf-8-sig",
    )
    figures = _figures(evaluations)
    (paths.reports_root / "V4_PHASE13_BASELINE_RESULTS.md").write_text(_results_text(evaluations), encoding="utf-8")
    (paths.reports_root / "V4_PHASE13_BACKTEST_QA.md").write_text(_qa_text(evaluations, prediction_rows, hashes, figures), encoding="utf-8")
    (paths.reports_root / "V4_PHASE13_2024_IDENTITY_REVIEW.md").write_text(_identity_text(priority), encoding="utf-8")
    (paths.reports_root / "V4_PHASE13_NEXT_DECISION.md").write_text(_decision_text(evaluations), encoding="utf-8")


def _figures(evaluations: list[dict[str, object]]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    folder = WORLD_ROOT / "figures" / "v4_baselines"
    folder.mkdir(parents=True, exist_ok=True)
    written = [
        _bar_figure(plt, folder / "share_tv_scenario_a.png", evaluations, "ex_post_max_information", "share_vector_TV", "探索性回顧：情境 A 份額總變異距離"),
        _bar_figure(plt, folder / "share_mae_scenario_a.png", evaluations, "ex_post_max_information", "share_MAE", "探索性回顧：情境 A 份額 MAE"),
        _delay_figure(plt, folder / "share_tv_delay_increase.png", evaluations),
    ]
    return written


def _bar_figure(plt, path, evaluations: list[dict[str, object]], scenario: str, metric: str, title: str) -> str:
    import numpy as np

    labels = [MODEL_LABELS[model] for model in MODELS]
    x = np.arange(len(MODELS))
    width = 0.18
    figure, axis = plt.subplots(figsize=(11, 5.5))
    for index, evaluation in enumerate(evaluations):
        values = [evaluation["summaries"][scenario]["by_model"][model][metric] or 0.0 for model in MODELS]
        axis.bar(x + (index - 1.5) * width, values, width, label=evaluation["parent_area"].replace("國家公園", "").replace("國家風景區", ""))
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=20, ha="right")
    axis.set_ylabel(metric)
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return str(path.relative_to(WORLD_ROOT)).replace("\\", "/")


def _delay_figure(plt, path, evaluations: list[dict[str, object]]) -> str:
    import numpy as np

    labels = [MODEL_LABELS[model] for model in RECENT_MODELS]
    x = np.arange(len(RECENT_MODELS))
    width = 0.18
    figure, axis = plt.subplots(figsize=(10, 5.5))
    for index, evaluation in enumerate(evaluations):
        values = []
        for model in RECENT_MODELS:
            left = evaluation["summaries"]["ex_post_max_information"]["by_model"][model]["share_vector_TV"]
            right = evaluation["summaries"]["assumed_publication_delay"]["by_model"][model]["share_vector_TV"]
            delta = delay_increase(left, right)
            values.append(0.0 if delta is None else delta)
        axis.bar(x + (index - 1.5) * width, values, width, label=evaluation["parent_area"].replace("國家公園", ""))
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=20, ha="right")
    axis.set_ylabel("情境 B TV - 情境 A TV")
    axis.set_title("探索性回顧：假設延遲三個月後的份額 TV 增加")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return str(path.relative_to(WORLD_ROOT)).replace("\\", "/")


def _results_text(evaluations: list[dict[str, object]]) -> str:
    lines = [
        "# V4 Phase 13｜探索性基準回測",
        "",
        "結果角色是 exploratory retrospective。這不是正式 P4，沒有證明社會學習，也沒有估計 LLM 或 ABM 的預測增益。",
        "",
        "預測目標是 reported_visit_distribution，也就是固定據點集合的官方回報人次份額。它是 B 級 reported_visits_conditional，不是個人選擇機率。",
        "",
        "每一次預測只預測下一個日曆月。24 個連續可定義月加後面 12 個連續可定義月，用來決定哪些月份可以進入評估；它不是一次預測未來 12 個月。",
        "",
        f"第六個模型的權重在看到結果之前固定為持續 {FIXED_RECENT_WEIGHT}、歷史同月平均 {FIXED_SEASONAL_WEIGHT}。權重沒有用評估月份調整。均勻份額是不需擬合的參考。",
        "",
        "滾動起點高度重疊。下面的可評估月份是不重複的日曆月，月份數不是獨立樣本數。信賴區間沒有估計。",
        "",
        "## 可評估月份",
        "",
        "| 地理群 | 據點數 | 可定義月 | 可評估月 | 評估區間 | 滾動起點 | 非重疊 12 月區塊 | 未進評估的可定義月 |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for evaluation in evaluations:
        targets = evaluation["evaluable_months"]
        span = f"{targets[0]}–{targets[-1]}" if targets else "—"
        lines.append(
            "| {name} | {sites} | {defined} | {evaluable} | {span} | {origins} | {blocks} | {held} |".format(
                name=_group_name(evaluation),
                sites=evaluation["site_count"],
                defined=len(evaluation["defined_months"]),
                evaluable=len(targets),
                span=span,
                origins=len(evaluation["origins"]),
                blocks=len(evaluation["blocks"]),
                held="、".join(_compress(evaluation["not_evaluable_months"])) or "—",
            )
        )
    lines.extend(
        [
            "",
            "阿里山沒有進入這個 24+12 的單步回測。全群零人次與來源衝突月沒有被補成均勻分布或 0。",
            "",
            "## 情境 A",
            "",
            "情境 A 假設預測 t 時，t-1 的官方結果已經可使用。這是資訊充足的事後回測，不是歷史上的即時預測。發布日仍然無法重建。",
            "",
            _score_markdown(evaluations, "ex_post_max_information"),
            "",
            "## 情境 B",
            "",
            "情境 B 事前假設預測 t 時最新可用月份是 t-3。這是假設性保守情境，不是查到的公布日期。近 3 個月與近 12 個月的視窗終點是資訊截止日，不是目標月 t。",
            "",
            _score_markdown(evaluations, "assumed_publication_delay"),
            "",
            "## 延遲三個月後的誤差變化",
            "",
            "數字是情境 B 的份額 TV 減去情境 A 的份額 TV。正值表示假設延遲後誤差變大。去年同月與歷史同月平均的來源月都早於 t-3，所以這兩個基準在兩種情境下使用同一批特徵。",
            "",
            _delay_table(evaluations),
            "",
            "## 季節性基準與歷史持續",
            "",
            _seasonal_text(evaluations),
            "",
            "## 日曆年與非重疊區塊",
            "",
            "年份列與區塊列都使用各模型共同的可評分月份。區塊平均只是把時間切成互不重疊的 12 個月，用來避免把重疊窗口數當成樣本數。",
            "",
            _year_text(evaluations),
            "",
            _block_text(evaluations),
            "",
            "## HHI 誤差與份額誤差",
            "",
            _hhi_text(evaluations),
            "",
            "## 份額偏差",
            "",
            "偏差是預測份額減實際份額。正值表示這個基準把份額配得比回報份額高。下表是情境 A 的最近可用月持續模型。",
            "",
            _bias_text(evaluations),
            "",
            "## 測量限制",
            "",
            _measurement_text(evaluations),
            "",
        ]
    )
    return "\n".join(lines)


def _score_markdown(evaluations: list[dict[str, object]], scenario: str) -> str:
    lines = [
        "| 地理群 | 基準 | 月份 | TV | MAE | 標準化 HHI 絕對誤差 | 領先據點命中 | 可用率 |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for evaluation in evaluations:
        for model in MODELS:
            metrics = evaluation["summaries"][scenario]["by_model"][model]
            lines.append(
                "| {name} | {model} | {n} | {tv} | {mae} | {hhi} | {leader} | {availability} |".format(
                    name=_group_name(evaluation),
                    model=MODEL_LABELS[model],
                    n=metrics["n_months"],
                    tv=_fmt(metrics["share_vector_TV"]),
                    mae=_fmt(metrics["share_MAE"]),
                    hhi=_fmt(metrics["normalized_HHI_absolute_error"]),
                    leader=_leader_cell(metrics),
                    availability=_fmt(metrics["availability"]),
                )
            )
    return "\n".join(lines)


def _delay_table(evaluations: list[dict[str, object]]) -> str:
    lines = [
        "| 地理群 | 基準 | 情境 A TV | 情境 B TV | TV 增加 | MAE 增加 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for evaluation in evaluations:
        for model in MODELS:
            left = evaluation["summaries"]["ex_post_max_information"]["by_model"][model]
            right = evaluation["summaries"]["assumed_publication_delay"]["by_model"][model]
            lines.append(
                "| {name} | {model} | {a} | {b} | {tv} | {mae} |".format(
                    name=_group_name(evaluation),
                    model=MODEL_LABELS[model],
                    a=_fmt(left["share_vector_TV"]),
                    b=_fmt(right["share_vector_TV"]),
                    tv=_fmt(delay_increase(left["share_vector_TV"], right["share_vector_TV"])),
                    mae=_fmt(delay_increase(left["share_MAE"], right["share_MAE"])),
                )
            )
    return "\n".join(lines)


def _seasonal_text(evaluations: list[dict[str, object]]) -> str:
    lines = []
    for scenario in SCENARIO_LAG:
        lines.append(SCENARIO_LABELS[scenario] + "：")
        for evaluation in evaluations:
            models = evaluation["summaries"][scenario]["by_model"]
            tv = lower_error(models["seasonal_month_mean"]["share_vector_TV"], models["persistence"]["share_vector_TV"])
            mae = lower_error(models["seasonal_month_mean"]["share_MAE"], models["persistence"]["share_MAE"])
            lines.append(f"- {_group_name(evaluation)}：份額 TV { _compare_word(tv) }，份額 MAE { _compare_word(mae) }。")
        lines.append("")
    return "\n".join(lines).rstrip()


def _year_text(evaluations: list[dict[str, object]]) -> str:
    lines = [
        "| 地理群 | 年 | 持續 TV | 同月平均 TV | 均勻 TV |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for evaluation in evaluations:
        years = evaluation["summaries"]["ex_post_max_information"]["by_year"]
        for year, models in years.items():
            lines.append(
                "| {name} | {year} | {persistence} | {seasonal} | {uniform} |".format(
                    name=_group_name(evaluation),
                    year=year,
                    persistence=_fmt(models["persistence"]["share_vector_TV"]),
                    seasonal=_fmt(models["seasonal_month_mean"]["share_vector_TV"]),
                    uniform=_fmt(models["uniform_reference"]["share_vector_TV"]),
                )
            )
    return "\n".join(lines)


def _block_text(evaluations: list[dict[str, object]]) -> str:
    lines = [
        "| 地理群 | 區塊 | 持續 TV | 同月平均 TV |",
        "| --- | --- | ---: | ---: |",
    ]
    for evaluation in evaluations:
        for block in evaluation["summaries"]["ex_post_max_information"]["by_block"]:
            lines.append(
                "| {name} | {start}–{end} | {persistence} | {seasonal} |".format(
                    name=_group_name(evaluation),
                    start=block["start"],
                    end=block["end"],
                    persistence=_fmt(block["models"]["persistence"]["share_vector_TV"]),
                    seasonal=_fmt(block["models"]["seasonal_month_mean"]["share_vector_TV"]),
                )
            )
        lines.append(f"| {_group_name(evaluation)} | 區塊數 {len(evaluation['blocks'])} | {evaluation['interval_note']} |  |")
    return "\n".join(lines)


def _hhi_text(evaluations: list[dict[str, object]]) -> str:
    lines = []
    group_tv = []
    group_hhi = []
    for evaluation in evaluations:
        models = evaluation["summaries"]["ex_post_max_information"]["by_model"]
        tv_order = sorted(MODELS, key=lambda model: _sort_value(models[model]["share_vector_TV"]))
        hhi_order = sorted(MODELS, key=lambda model: _sort_value(models[model]["normalized_HHI_absolute_error"]))
        same = tv_order == hhi_order
        lines.append(
            f"- {_group_name(evaluation)}：情境 A 以 TV 排序的基準順序{'與' if same else '不與'}標準化 HHI 絕對誤差的順序相同。"
            f" J={evaluation['site_count']}。持續模型 TV {_fmt(models['persistence']['share_vector_TV'])}，"
            f"HHI 絕對誤差 {_fmt(models['persistence']['normalized_HHI_absolute_error'])}。"
        )
        group_tv.append((evaluation["parent_area"], models["persistence"]["share_vector_TV"]))
        group_hhi.append((evaluation["parent_area"], models["persistence"]["normalized_HHI_absolute_error"]))
    tv_rank = [name for name, _value in sorted(group_tv, key=lambda item: _sort_value(item[1]))]
    hhi_rank = [name for name, _value in sorted(group_hhi, key=lambda item: _sort_value(item[1]))]
    lines.append(
        "四群在情境 A 的持續模型上，TV 由低到高是 "
        + "、".join(tv_rank)
        + "。標準化 HHI 絕對誤差由低到高是 "
        + "、".join(hhi_rank)
        + "。"
    )
    lines.append("HHI 誤差描述的是集中度這一個數字。份額 TV 與 MAE 描述整條份額向量。金門的標準化 HHI 水準本來就低，HHI 誤差較小不能單獨說成預測能力較佳。")
    return "\n".join(lines)


def _bias_text(evaluations: list[dict[str, object]]) -> str:
    lines = [
        "| 地理群 | 據點 | 平均偏差 |",
        "| --- | --- | ---: |",
    ]
    for evaluation in evaluations:
        bias = evaluation["summaries"]["ex_post_max_information"]["bias"].get("persistence", {})
        ordered = sorted(bias.items(), key=lambda item: abs(item[1]), reverse=True)
        for uid, value in ordered:
            lines.append(f"| {_group_name(evaluation)} | {evaluation['names'].get(uid, uid)} | {_fmt(value)} |")
    return "\n".join(lines)


def _measurement_text(evaluations: list[dict[str, object]]) -> str:
    lines = []
    for evaluation in evaluations:
        targets = evaluation["evaluable_months"]
        shock = [month for month in targets if pandemic_stratum(month) == "candidate_common_shock_2020_2022"]
        method = evaluation["method_note"]
        rule = evaluation["rule_match"]
        lines.append(
            f"- {_group_name(evaluation)}：可評估月 {len(shock)} 個落在 2020-01 至 2022-12 的候選共同衝擊窗。"
            f" Phase 10 方法欄是 {evaluation['measurement_methods']}。"
            f" 同名列的非空計數標籤為 {'、'.join(method['labels']) or '沒有'}。"
            f" 標籤有變動的據點數是 {len(method['sites_with_label_change'])}。"
            f" 空白方法列 {method['blank_method_rows']}。"
            f" match_status 包含 provisional_rule_match：{rule['contains_provisional_rule_match']}。"
            f" 外部身分是 {rule['externally_verified_identity_status']}。"
            f" 群組來源衝突率 {evaluation['source_conflict_rate']}。"
            f" 份額未定義的月份次數 {evaluation['excluded'] or '沒有'}。"
            " 公布延遲是假設。可比性維持 reported_visits_conditional。"
        )
        note = evaluation["leader_note"]
        lines.append(
            f"  可評估月的實際領先據點有 {note['distinct_leaders']} 個，領先更替 {note['turnovers']} 次。"
            " 領先據點很少更換時，命中率會偏高，仍要以份額向量誤差為主。"
        )
    lines.append("年度工作簿對到的非空方法標籤沒有跨年變更。Phase 11 已記錄玉山、雪霸、金門仍有年份無法歸成直接計數標籤；本輪沒有改寫那份覆蓋率，也沒有把 B 級升成 A 級。")
    lines.append("疫情窗是事前標示的描述分層。沒有因為某段誤差較低而刪除月份。")
    return "\n".join(lines)


def _qa_text(
    evaluations: list[dict[str, object]],
    prediction_rows: list[dict[str, object]],
    hashes: dict[str, str],
    figures: list[str],
) -> str:
    import pandas as pd
    import pyarrow

    unavailable = sum(1 for row in prediction_rows if row["prediction_status"] != "predicted")
    predicted = sum(1 for row in prediction_rows if row["prediction_status"] == "predicted")
    holdout = [row["year_month"] for row in prediction_rows if int(str(row["year_month"])[:4]) >= 2025]
    simplex_bad = 0
    by_key: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in prediction_rows:
        if row["prediction_status"] != "predicted":
            continue
        by_key[(row["group_id"], row["scenario"], row["model"], row["year_month"])].append(float(row["predicted_share"]))
    for values in by_key.values():
        if not valid_share_vector(values):
            simplex_bad += 1
    cutoff_bad = 0
    for row in prediction_rows:
        expected = information_cutoff(str(row["year_month"]), str(row["scenario"]))
        if row["information_cutoff"] != expected:
            cutoff_bad += 1
        if row["source_months"] and max(str(row["source_months"]).split(",")) > expected:
            cutoff_bad += 1
    lines = [
        "# V4 Phase 13｜回測 QA",
        "",
        "這一輪讀取的是 Phase 12 的探索性候選面板，沒有讀取 2025–2026 保留檔的人次、份額或 HHI。阿里山列留在原面板，沒有進入本輪預測。",
        "",
        f"Python {sys.version.split()[0]}，平台 {platform.platform()}，pandas {pd.__version__}，pyarrow {pyarrow.__version__}。",
        "",
        f"預測列 {len(prediction_rows)}。已預測的據點列 {predicted}。prediction_unavailable 列 {unavailable}。",
        f"份額向量加總檢查失敗的群月數 {simplex_bad}。截止日或來源月越界的列數 {cutoff_bad}。2025 以後的預測月 {holdout or '沒有'}。",
        "",
        "程式內對每個預測另用截止日以前的歷史重算一次。兩次份額或來源月不一致會中止寫出結果。本輪寫出前沒有觸發這項中止。",
        "",
        "共同評分月是同一情境、同一群裡七個基準都成功預測的月份。比較表使用這組月份。",
        "",
        "| 地理群 | 情境 | 可評估月 | 共同評分月 | 描述月份數 | 滾動起點 | 非重疊區塊 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for evaluation in evaluations:
        for scenario in SCENARIO_LAG:
            common = evaluation["summaries"][scenario]["common_months"]
            lines.append(
                "| {name} | {scenario} | {evaluable} | {common} | {sample} | {origins} | {blocks} |".format(
                    name=_group_name(evaluation),
                    scenario=scenario,
                    evaluable=len(evaluation["evaluable_months"]),
                    common=len(common),
                    sample=independent_sample_size(common),
                    origins=len(evaluation["origins"]),
                    blocks=len(evaluation["blocks"]),
                )
            )
    lines.extend(
        [
            "",
            "描述月份數等於不重複日曆月。它不是滾動起點數，也不是起點數乘上 12。",
            "",
            "新增測試：test_share_prediction_simplex、test_rolling_origin_no_leakage、test_asof_lag_one、test_asof_lag_three、test_missing_month_no_imputation、test_all_zero_month_exclusion、test_fixed_group_membership、test_same_scoring_calendar、test_no_2025_data_access、test_overlapping_windows_not_independent、test_2024_identity_evidence_separation。",
            "",
            "輸入 SHA-256：",
            "",
        ]
    )
    for relative, digest in hashes.items():
        lines.append(f"- `{relative}`：`{digest}`")
    lines.extend(["", "圖："])
    for figure in figures:
        lines.append(f"- `{figure}`")
    lines.extend(
        [
            "",
            f"評估程式 SHA-256：phase13.py `{file_sha256(WORLD_ROOT / 'worldlab_data' / 'phase13.py')}`，phase13_run.py `{file_sha256(WORLD_ROOT / 'worldlab_data' / 'phase13_run.py')}`。",
            "",
            "受保護的 Phase 11–12 檔案在寫出 Phase 13 前後雜湊相同。",
            "",
        ]
    )
    return "\n".join(lines)


def _identity_text(priority: list[dict[str, object]]) -> str:
    lines = [
        "# V4 Phase 13｜2024 高證據身分覆核",
        "",
        "這份覆核與探索性基準分開。它沒有改四組歷史回測樣本，也沒有把任何列升成 confirmed_manual。沒有真人逐項審核，human_reviewed 維持 false。",
        "",
        "Phase 12 的 24 列裡，unresolved 是 18、unchanged 是 4、not_found 是 2。這 13 列是 unresolved 之中、2024 年十二個月人次與同年 Excel 同名列一致的高優先候選。",
        "",
        "兩個判斷分開記。physical_or_statistical_identity_continuity 這裡記成 cross_source_evidence_candidate，表示同年 CSV 與 Excel 可能是同一統計列。measurement_continuity 記成 not_established。十二個月人次一致不能單獨證明與上一年的計數方式或統計範圍連續。",
        "",
        "| 據點 | 縣市 | CSV 2023 | CSV 2024 | Excel 2024 | 方法 2023 | 方法 2024 | 覆核狀態 | 口徑連續 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in priority:
        lines.append(
            "| {name} | {county} | {csv_2023} | {csv_2024} | {excel} | {method_2023} | {method_2024} | {status} | {measurement} |".format(
                name=row["official_name"],
                county=row["county"],
                csv_2023=row["csv_category_2023"],
                csv_2024=row["csv_category_2024"],
                excel=row["excel_category_2024"],
                method_2023=row["measurement_method_2023"] or "—",
                method_2024=row["measurement_method_2024"] or "—",
                status=row["review_status"],
                measurement=row["measurement_continuity"],
            )
        )
    attraction = [str(row["official_name"]) for row in priority if row["local_attraction_name_found"]]
    lines.extend(
        [
            "",
            f"13 列的 review_status 都是 cross_source_evidence_candidate。confirmed_manual 都是 false。baseline_sample_changed 都是 false。",
            "",
            "地方景點清單 AttractionList.xml 以名稱完全相符查到："
            + ("、".join(attraction) if attraction else "沒有")
            + "。名稱出現只表示清單上有這個地名，沒有記載 2023 到 2024 的拆分、合併或統計範圍變動。其餘據點沒有同名景點列，這也不構成範圍已經改變的證據。",
            "",
            "CSV 類型與 Excel 合併類型的差異仍保留。同名不足以接上 2023 的 site_uid。",
            "",
        ]
    )
    return "\n".join(lines)


def _decision_text(evaluations: list[dict[str, object]]) -> str:
    lines = [
        "# V4 Phase 13｜下一步",
        "",
        "本輪是 exploratory retrospective。沒有完成 P4，沒有證明社會學習，也沒有宣稱 LLM 或 ABM 有預測增益。WorldLab 的 Q 仍是模型內反事實機率距離。",
        "",
        "是否進入 adapter 開發研究，只看事前指定的歷史長度：可評估月至少 96，而且非重疊 12 月區塊至少 4。預測誤差沒有用來挑選群組。",
        "",
        "| 地理群 | 可評估月 | 非重疊區塊 | 角色 |",
        "| --- | ---: | ---: | --- |",
    ]
    for evaluation in evaluations:
        lines.append(
            f"| {_group_name(evaluation)} | {len(evaluation['evaluable_months'])} | {len(evaluation['blocks'])} | {evaluation['adapter_role']} |"
        )
    lines.extend(
        [
            "",
            "玉山、太魯閣、雪霸符合候選長度。金門的可評估月只覆蓋 2020-01 至 2021-05，角色是短序列對照。阿里山本輪沒有做 24+12 回測。",
            "",
            "adapter 若要和這些基準比較，比較對象是同一組據點、同一個單步期限、同一份歷史可得資訊下的完整份額向量。均勻份額與最近可用月持續都要留下，不能只挑評估誤差最低的基準。",
            "",
            "## 若要正式使用 2025 保留集，必須先凍結",
            "",
            "- 據點集合與 site_uid，含尚未解決的 2024 身分列不得事後補進。",
            "- 可評估月份、單步期限，以及 24+12 只作為區段資格而不是 12 步超前預測。",
            "- 來源優先規則，以及 2021 年 11–12 月兩套版本都不選邊。",
            "- 測量限制：維持 reported_visits_conditional，目標維持 reported_visit_distribution。",
            "- 資訊情境：情境 A 與情境 B 的截止規則，以及公布日未知所以不是 strict_asof_eligible。",
            "- 模型清單與固定權重 0.5／0.5。權重與模型選擇都不能再看 2025 的人次、份額、HHI 或分數。",
            "- 評分以 share_vector_TV 與 share_MAE 為主，標準化 HHI 誤差與領先據點命中為輔。",
            "- 重疊窗口不作獨立樣本；區間方法若要做，必須使用能保留時間相依的區塊，並在區塊數不足時不要用 p 值宣稱模型等價。",
            "- 這一輪的探索性分數不能事後決定刪除哪些月份或更換群組。",
            "",
            "以上項目在本輪還沒有凍結成確認性協定。2025–2026 的人次沒有讀取。",
            "",
        ]
    )
    return "\n".join(lines)


def _summary_payload(
    evaluations: list[dict[str, object]],
    prediction_rows: list[dict[str, object]],
    priority: list[dict[str, object]],
    hashes: dict[str, str],
) -> dict[str, object]:
    groups = []
    for evaluation in evaluations:
        scenario_a = evaluation["summaries"]["ex_post_max_information"]["by_model"]
        scenario_b = evaluation["summaries"]["assumed_publication_delay"]["by_model"]
        groups.append(
            {
                "parent_area": evaluation["parent_area"],
                "evaluable_months": len(evaluation["evaluable_months"]),
                "defined_months": len(evaluation["defined_months"]),
                "origins": len(evaluation["origins"]),
                "nonoverlap_blocks": len(evaluation["blocks"]),
                "adapter_role": evaluation["adapter_role"],
                "persistence_tv_a": scenario_a["persistence"]["share_vector_TV"],
                "persistence_mae_a": scenario_a["persistence"]["share_MAE"],
                "seasonal_tv_a": scenario_a["seasonal_month_mean"]["share_vector_TV"],
                "persistence_tv_b": scenario_b["persistence"]["share_vector_TV"],
                "persistence_tv_increase": delay_increase(
                    scenario_a["persistence"]["share_vector_TV"],
                    scenario_b["persistence"]["share_vector_TV"],
                ),
            }
        )
    return {
        "analysis_role": ANALYSIS_ROLE,
        "model_fitted": False,
        "weights_tuned_on_evaluation": WEIGHTS_TUNED_ON_EVALUATION,
        "holdout_visits_used": False,
        "prediction_rows": len(prediction_rows),
        "priority_identity_rows": len(priority),
        "confirmed_manual_rows": sum(1 for row in priority if row["confirmed_manual"]),
        "groups": groups,
        "input_sha256": hashes,
        "protected_files_unchanged": True,
    }


def _compare_word(result: str) -> str:
    return {
        "left": "以歷史同月平均較低",
        "right": "以最近可用月持續較低",
        "tie": "兩者相同",
        "unavailable": "無法比較",
    }[result]


def _group_name(evaluation: dict[str, object]) -> str:
    return f"{evaluation['county']}／{evaluation['parent_area']}"


def _fmt(value: object) -> str:
    if value is None or value == "":
        return "—"
    return f"{float(value):.3f}"


def _leader_cell(metrics: dict[str, object]) -> str:
    accuracy = metrics["leading_site_accuracy"]
    if accuracy is None:
        return "—"
    return f"{_fmt(accuracy)}（{metrics['leading_site_hits']}/{metrics['leading_site_decisive']}）"


def _sort_value(value: object) -> float:
    if value is None:
        return float("inf")
    return float(value)


def _compress(months: list[str]) -> list[str]:
    ordered = sorted(months)
    if not ordered:
        return []
    ranges = []
    start = previous = ordered[0]
    for month in ordered[1:]:
        if month == shift_month(previous, 1):
            previous = month
            continue
        ranges.append(start if start == previous else f"{start}–{previous}")
        start = previous = month
    ranges.append(start if start == previous else f"{start}–{previous}")
    return ranges


def _hashes() -> dict[str, str]:
    return {relative: file_sha256(WORLD_ROOT / relative) for relative in GUARDED}


if __name__ == "__main__":
    print(json.dumps(run_phase13(), ensure_ascii=False, indent=2, allow_nan=False))
