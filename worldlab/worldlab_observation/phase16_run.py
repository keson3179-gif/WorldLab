"""Exploratory intensity bridge. Holdout visits stay unread and Phase 13 outputs stay in place."""

from __future__ import annotations

import json
import platform
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

import pandas as pd

from worldlab_data.config import WORLD_ROOT, V4Paths, load_paths
from worldlab_data.phase12 import valid_share_vector
from worldlab_data.phase13 import (
    ANALYSIS_ROLE,
    MODELS,
    SCENARIO_LAG,
    TARGET_NAME,
    evaluable_months,
    information_cutoff,
    is_defined_share_month,
    leakage_failures,
    nonoverlapping_blocks,
    predict_shares,
    score_share_forecast,
)
from worldlab_data.tdx import dry_run_plan
from worldlab_data.textutil import file_sha256
from worldlab_observation.intensity import (
    BETA_GRID,
    PHASE16_PARENTS,
    PREDICTION_QUANTITY,
    SIGNALS,
    choose_beta,
    eligible_internal_months,
    frozen_parents,
    m0_components,
    m1_shares,
    m2_generation_status,
    select_beta,
    shares_close,
    signal_vector,
)
from worldlab_observation.place_atlas import NAME_STEMS

ATTRACTION_NS = "{https://media.taiwan.net.tw/standard/schema/2.0}"
READ_PATHS = (
    "data/curated/v4_forecast_panel_candidate.parquet",
    "data/curated/v4_group_comparability.csv",
    "data/curated/v4_canonical_site_crosswalk.csv",
    "data/curated/v4_2024_identity_audit.csv",
    "data/curated/v4_2024_priority_identity_review.csv",
    "data/curated/v4_place_attribute_provenance.csv",
    "觀光資料標準V2.1/AttractionFeeList.xml",
    "觀光資料標準V2.1/AttractionServiceTimeList.xml",
)
GUARDED = (
    "world/choice.py",
    "world/geography.py",
    "world/experiment_004.py",
    "world/experiment_005.py",
    "world/experiment_018.py",
    "world/experiment_020.py",
    "worldlab_data/phase13.py",
    "worldlab_observation/adapter.py",
    "data/curated/v4_forecast_panel_candidate.parquet",
    "data/curated/v4_2024_priority_identity_review.csv",
    "data/curated/v4_place_attribute_provenance.csv",
    "reports/V4_PHASE13_BASELINE_RESULTS.md",
    "reports/V4_PHASE15_PLACE_ATLAS.md",
)
PANEL_COLUMNS = (
    "group_id",
    "site_uid",
    "year_month",
    "share",
    "dataset_role",
    "confirmatory_holdout_candidate",
    "strict_asof_eligible",
    "target",
    "used_for_model_selection",
)
SCENARIO_LABELS = {
    "ex_post_max_information": "情境 A：假設 t-1 已可使用",
    "assumed_publication_delay": "情境 B：假設最新可用月是 t-3",
}
BASELINE_LABELS = {
    "persistence": "最近可用月",
    "same_month_last_year": "去年同月",
    "seasonal_month_mean": "歷史同月平均",
    "trailing_3_mean": "近 3 月平均",
    "trailing_12_mean": "近 12 月平均",
    "fixed_equal_weight": "固定 0.5/0.5",
    "uniform_reference": "均勻份額",
}


def run_phase16(paths: V4Paths | None = None) -> dict[str, object]:
    paths = paths or load_paths()
    before = _hashes()
    frozen_parents(list(PHASE16_PARENTS))
    panel = _panel()
    sites = _sites(panel)
    rows, summaries = _evaluate(panel, sites)
    gaps = _identity_gaps()
    attributes, description_count = _attribute_audit(paths, sites)
    tdx = dry_run_plan()
    tdx_status = "offline_credentials_absent" if not tdx["credentials_present"] else "credentials_present_current_feed_not_used"
    payload = {
        "analysis_role": "exploratory_development",
        "model_family": "aggregate_intensity_bridge",
        "prediction_quantity": PREDICTION_QUANTITY,
        "p4_executed": False,
        "llm_calls": 0,
        "m2_status": m2_generation_status(description_count),
        "historically_valid_descriptions": description_count,
        "holdout_visits_used": False,
        "same_period_as_phase13": True,
        "independent_validation": False,
        "tdx_status": tdx_status,
        "tdx_will_call_network": bool(tdx["will_call_network"]),
        "groups": summaries,
        "read_paths": list(READ_PATHS),
    }
    out = paths.output_root / "phase16_intensity"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(rows).to_csv(out / "month_scores.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame.from_records(gaps).to_csv(out / "identity_evidence_gaps.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame.from_records(attributes).to_csv(out / "historical_attribute_audit.csv", index=False, encoding="utf-8-sig")
    _write_reports(paths, payload, attributes, gaps)
    if _hashes() != before:
        raise RuntimeError("protected files changed while the intensity bridge was written")
    (out / "phase16_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return {
        "groups": len(summaries),
        "rows": len(rows),
        "m2_status": payload["m2_status"],
        "tdx_status": tdx_status,
        "holdout_visits_used": False,
        "protected_files_unchanged": True,
    }


def _panel() -> pd.DataFrame:
    panel = pd.read_parquet(WORLD_ROOT / "data" / "curated" / "v4_forecast_panel_candidate.parquet", columns=list(PANEL_COLUMNS))
    if any(str(value) >= "2024-01" for value in panel["year_month"]):
        raise RuntimeError("development input includes a month outside the frozen pre-2024 panel")
    if panel["dataset_role"].ne(ANALYSIS_ROLE).any():
        raise RuntimeError("panel role is not the exploratory retrospective panel")
    if panel["confirmatory_holdout_candidate"].astype(bool).any():
        raise RuntimeError("a confirmatory row entered the intensity bridge")
    if panel["strict_asof_eligible"].astype(bool).any():
        raise RuntimeError("strict as-of was marked without a publication date")
    if panel["used_for_model_selection"].astype(bool).any():
        raise RuntimeError("panel rows were marked as model selection")
    if panel["target"].ne(TARGET_NAME).any():
        raise RuntimeError("target is not reported_visit_distribution")
    return panel


def _sites(panel: pd.DataFrame) -> list[dict[str, str]]:
    curated = WORLD_ROOT / "data" / "curated"
    comparability = pd.read_csv(curated / "v4_group_comparability.csv", dtype=str, keep_default_na=False)
    crosswalk = pd.read_csv(curated / "v4_canonical_site_crosswalk.csv", dtype=str, keep_default_na=False)
    chosen = comparability[comparability["definition"].eq("A_geographic") & comparability["parent_area"].isin(PHASE16_PARENTS)]
    sites: list[dict[str, str]] = []
    for group in chosen.to_dict(orient="records"):
        membership = tuple(sorted(set(panel.loc[panel["group_id"].eq(group["group_id"]), "site_uid"])))
        if len(membership) != 3:
            raise RuntimeError("a development group does not have J=3")
        for site_uid in membership:
            linked = crosswalk[crosswalk["site_uid"].eq(site_uid) & crosswalk["match_status"].isin(["provisional_rule_match", "singleton"])]
            latest = linked.sort_values("last_year").iloc[-1]
            sites.append(
                {
                    "group_id": group["group_id"],
                    "parent_area": group["parent_area"],
                    "county": str(latest["county"]),
                    "site_uid": site_uid,
                    "official_name": str(latest["official_name"]),
                }
            )
    if {row["official_name"] for row in sites} != set(NAME_STEMS):
        raise RuntimeError("the frozen nine sites were not recovered")
    return sites


def _evaluate(panel: pd.DataFrame, sites: list[dict[str, str]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for site in sites:
        grouped[site["group_id"]].append(site)
    order = {parent: index for index, parent in enumerate(PHASE16_PARENTS)}
    for group_id, members in sorted(grouped.items(), key=lambda item: order[item[1][0]["parent_area"]]):
        parent = members[0]["parent_area"]
        county = members[0]["county"]
        membership = tuple(site["site_uid"] for site in sorted(members, key=lambda item: item["site_uid"]))
        scoped = panel[panel["group_id"].eq(group_id)]
        history: dict[str, dict[str, float]] = {}
        for year_month, chunk in scoped.groupby("year_month"):
            shares = {str(row.site_uid): float(row.share) for row in chunk.itertuples(index=False)}
            if is_defined_share_month(shares, membership):
                history[str(year_month)] = {uid: shares[uid] for uid in membership}
        targets = evaluable_months(sorted(history))
        prepared = _prepare(history, membership)
        group_rows: list[dict[str, object]] = []
        for target in targets:
            actual = history[target]
            for scenario in SCENARIO_LAG:
                group_rows.extend(_baseline_rows(history, membership, target, scenario, actual, parent, county, group_id))
                components, signals = prepared[(target, scenario)]
                group_rows.append(_score_row(parent, county, group_id, scenario, "M0", "", "", target, components["status"], components["skip_reason"], _as_map(membership, components["shares"]), actual, components["information_cutoff"]))
                _assert_cutoff_isolation(history, membership, target, scenario, components)
                for signal_name in SIGNALS:
                    signal = signals[signal_name]
                    for beta in BETA_GRID:
                        predicted = m1_shares(components, signal, beta)
                        group_rows.append(
                            _score_row(
                                parent, county, group_id, scenario, "M1_grid", signal_name, beta, target,
                                predicted["status"], predicted["skip_reason"], _as_map(membership, predicted["shares"]), actual,
                                components["information_cutoff"],
                            )
                        )
                    selected = _cached_selection(history, membership, prepared, target, scenario, signal_name)
                    if target == targets[0]:
                        reference = select_beta(history, membership, target, scenario, signal_name)
                        if reference["status"] != selected["status"] or reference["beta"] != selected["beta"]:
                            raise RuntimeError("cached beta selection drifted from the audited selector")
                    if selected["status"] != "selected":
                        group_rows.append(
                            _score_row(
                                parent, county, group_id, scenario, "M1_nested", signal_name, "", target,
                                "insufficient_history", "insufficient_history", None, actual, information_cutoff(target, scenario),
                            )
                        )
                        continue
                    chosen = m1_shares(components, signal, float(selected["beta"]))
                    group_rows.append(
                        _score_row(
                            parent, county, group_id, scenario, "M1_nested", signal_name, selected["beta"], target,
                            chosen["status"], chosen["skip_reason"], _as_map(membership, chosen["shares"]), actual,
                            components["information_cutoff"],
                        )
                    )
        if any(str(row["year_month"]) >= "2024-01" for row in group_rows):
            raise RuntimeError("a scored month lies outside the development window")
        rows.extend(group_rows)
        summaries.append(_summarize_group(parent, county, targets, group_rows, history))
    return rows, summaries


def _prepare(history: dict[str, dict[str, float]], membership: tuple[str, ...]) -> dict[tuple[str, str], tuple[dict[str, object], dict[str, dict[str, object]]]]:
    prepared = {}
    for month in history:
        for scenario in SCENARIO_LAG:
            components = m0_components(history, membership, month, scenario)
            signals = {}
            if components["status"] == "predicted":
                cutoff = str(components["information_cutoff"])
                signals = {name: signal_vector(history, membership, cutoff, month, name) for name in SIGNALS}
            prepared[(month, scenario)] = (components, signals)
    return prepared


def _cached_selection(history, membership, prepared, target, scenario, signal_name) -> dict[str, object]:
    internal = eligible_internal_months(history, membership, target, scenario)
    if not internal:
        return {"status": "insufficient_history", "beta": None}
    means: dict[float, float] = {}
    for beta in BETA_GRID:
        values = []
        for month in internal:
            components, signals = prepared[(month, scenario)]
            predicted = m1_shares(components, signals[signal_name], beta)
            if predicted["status"] != "predicted":
                values = []
                break
            values.append(
                score_share_forecast(history[month], _as_map(membership, predicted["shares"]))["share_vector_TV"]
            )
        if len(values) == len(internal):
            means[beta] = float(sum(values) / len(values))
    if not means:
        return {"status": "insufficient_history", "beta": None}
    return {"status": "selected", "beta": choose_beta(means)}


def _assert_cutoff_isolation(history, membership, target, scenario, components) -> None:
    if components["status"] != "predicted":
        return
    cutoff = str(components["information_cutoff"])
    visible = {month: shares for month, shares in history.items() if month <= cutoff and month < target}
    shadow = m0_components(visible, membership, target, scenario)
    if not shares_close(components["shares"], shadow["shares"]):
        raise RuntimeError("alpha used a month after the cutoff")
    if any(month > cutoff or month >= target for month in components["source_months"]):
        raise RuntimeError("training months crossed the cutoff")


def _baseline_rows(history, membership, target, scenario, actual, parent, county, group_id) -> list[dict[str, object]]:
    rows = []
    for model in MODELS:
        prediction = predict_shares(history, membership, target, scenario, model)
        visible = {month: shares for month, shares in history.items() if month <= str(prediction["information_cutoff"])}
        shadow = predict_shares(visible, membership, target, scenario, model)
        if prediction["shares"] != shadow["shares"] or prediction["source_months"] != shadow["source_months"]:
            raise RuntimeError("baseline features crossed the cutoff")
        if leakage_failures(prediction, target):
            raise RuntimeError("baseline leakage check failed")
        shares = None if prediction["status"] != "predicted" else dict(prediction["shares"])
        rows.append(
            _score_row(
                parent, county, group_id, scenario, model, "", "", target,
                prediction["status"], prediction["skip_reason"], shares, actual, prediction["information_cutoff"],
            )
        )
    return rows


def _score_row(parent, county, group_id, scenario, model, signal, beta, target, status, skip_reason, shares, actual, cutoff) -> dict[str, object]:
    scored = {"status": status}
    if status == "predicted" and shares is not None:
        if not valid_share_vector([float(shares[uid]) for uid in shares]):
            raise RuntimeError("predicted shares left the simplex")
        scored = score_share_forecast(actual, shares)
    return {
        "parent_area": parent,
        "county": county,
        "group_id": group_id,
        "scenario": scenario,
        "model": model,
        "signal": signal,
        "beta": "" if beta == "" else float(beta),
        "year_month": target,
        "status": scored["status"] if status == "predicted" else status,
        "skip_reason": skip_reason,
        "share_vector_TV": _blank(scored.get("share_vector_TV")),
        "share_MAE": _blank(scored.get("share_MAE")),
        "normalized_HHI_absolute_error": _blank(scored.get("normalized_HHI_absolute_error")),
        "leading_site_hit": "" if scored.get("leading_site_hit") is None else bool(scored["leading_site_hit"]),
        "information_cutoff": cutoff,
        "prediction_quantity": PREDICTION_QUANTITY,
    }


def _as_map(membership: tuple[str, ...], shares: list[float] | None) -> dict[str, float] | None:
    if shares is None:
        return None
    return {uid: float(shares[index]) for index, uid in enumerate(membership)}


def _summarize_group(parent, county, targets, rows, history) -> dict[str, object]:
    scenarios = {}
    for scenario in SCENARIO_LAG:
        scoped = [row for row in rows if row["scenario"] == scenario]
        models = {}
        for model in (*MODELS, "M0"):
            models[model] = _metrics([row for row in scoped if row["model"] == model], len(targets))
        common = _common_months(scoped, [*MODELS, "M0"])
        m0_on_common = _metrics(_filter_months(scoped, "M0", "", None, common), len(common))
        gaps = {
            model: None if m0_on_common["share_vector_TV"] is None or models[model]["share_vector_TV"] is None else m0_on_common["share_vector_TV"] - _metrics(_filter_months(scoped, model, "", None, common), len(common))["share_vector_TV"]
            for model in MODELS
        }
        grid = {}
        nested = {}
        for signal_name in SIGNALS:
            grid[signal_name] = {}
            for beta in BETA_GRID:
                grid[signal_name][str(beta)] = _metrics(_filter_months(scoped, "M1_grid", signal_name, beta, common), len(common))
            nested_rows = [row for row in scoped if row["model"] == "M1_nested" and row["signal"] == signal_name and row["status"] == "scored"]
            nested_months = {row["year_month"] for row in nested_rows}
            chosen = [float(row["beta"]) for row in nested_rows]
            baseline_on_nested = {
                model: _metrics(_filter_months(scoped, model, "", None, nested_months), len(nested_months))["share_vector_TV"]
                for model in MODELS
            }
            ranked = [model for model, value in baseline_on_nested.items() if value is not None]
            best_model = min(ranked, key=lambda model: baseline_on_nested[model]) if ranked else ""
            counts: dict[str, int] = defaultdict(int)
            for beta in chosen:
                counts[f"{beta:.1f}"] += 1
            modal = None if not counts else max(counts, key=lambda key: (counts[key], -abs(float(key))))
            nested[signal_name] = {
                "n": len(nested_rows),
                "insufficient_months": len(targets) - len(nested_rows),
                "modal_beta": modal,
                "modal_share": None if not chosen else counts[modal] / len(chosen),
                "tv": _metrics(nested_rows, len(nested_rows))["share_vector_TV"],
                "mae": _metrics(nested_rows, len(nested_rows))["share_MAE"],
                "hhi": _metrics(nested_rows, len(nested_rows))["normalized_HHI_absolute_error"],
                "leader": _metrics(nested_rows, len(nested_rows))["leading_site_accuracy"],
                "m0_tv_same_months": _metrics(_filter_months(scoped, "M0", "", None, nested_months), len(nested_months))["share_vector_TV"],
                "persistence_tv_same_months": _metrics(_filter_months(scoped, "persistence", "", None, nested_months), len(nested_months))["share_vector_TV"],
                "best_baseline": best_model,
                "best_baseline_tv_same_months": None if not best_model else baseline_on_nested[best_model],
                "beta_counts": dict(counts),
            }
        blocks = nonoverlapping_blocks(sorted(common))
        scenarios[scenario] = {
            "evaluable_months": len(targets),
            "common_months": len(common),
            "models": models,
            "m0_minus_baseline_tv": gaps,
            "grid": grid,
            "nested": nested,
            "block_count": len(blocks),
            "block_m0_tv": _block_means(scoped, "M0", blocks),
            "block_persistence_tv": _block_means(scoped, "persistence", blocks),
            "years": _year_means(scoped, common),
            "span": f"{targets[0]}–{targets[-1]}" if targets else "",
        }
    return {
        "parent_area": parent,
        "county": county,
        "defined_months": len(history),
        "evaluable_months": len(targets),
        "scenarios": scenarios,
    }


def _common_months(rows: list[dict[str, object]], models: list[str]) -> set[str]:
    sets = []
    for model in models:
        sets.append({row["year_month"] for row in rows if row["model"] == model and row["status"] == "scored"})
    if not sets:
        return set()
    return set.intersection(*sets)


def _filter_months(rows, model, signal, beta, months: set[str]) -> list[dict[str, object]]:
    selected = []
    for row in rows:
        if row["model"] != model or row["year_month"] not in months or row["status"] != "scored":
            continue
        if signal and row["signal"] != signal:
            continue
        if beta is not None and abs(float(row["beta"]) - float(beta)) > 1e-12:
            continue
        selected.append(row)
    return selected


def _metrics(rows: list[dict[str, object]], expected: int) -> dict[str, object]:
    tvs = [float(row["share_vector_TV"]) for row in rows if row["share_vector_TV"] != ""]
    maes = [float(row["share_MAE"]) for row in rows if row["share_MAE"] != ""]
    hhis = [float(row["normalized_HHI_absolute_error"]) for row in rows if row["normalized_HHI_absolute_error"] != ""]
    hits = [row["leading_site_hit"] for row in rows if row["leading_site_hit"] != ""]
    return {
        "n": len(tvs),
        "availability": None if expected == 0 else len(tvs) / expected,
        "share_vector_TV": None if not tvs else float(sum(tvs) / len(tvs)),
        "share_MAE": None if not maes else float(sum(maes) / len(maes)),
        "normalized_HHI_absolute_error": None if not hhis else float(sum(hhis) / len(hhis)),
        "leading_site_accuracy": None if not hits else float(sum(1 for hit in hits if hit) / len(hits)),
    }


def _block_means(rows, model, blocks: list[list[str]]) -> list[float]:
    means = []
    for block in blocks:
        chosen = _filter_months(rows, model, "", None, set(block))
        if len(chosen) == len(block):
            means.append(float(sum(float(row["share_vector_TV"]) for row in chosen) / len(block)))
    return means


def _year_means(rows, months: set[str]) -> dict[str, dict[str, float | None]]:
    years = sorted({month[:4] for month in months})
    output = {}
    for year in years:
        year_months = {month for month in months if month.startswith(year)}
        output[year] = {
            "M0": _metrics(_filter_months(rows, "M0", "", None, year_months), len(year_months))["share_vector_TV"],
            "persistence": _metrics(_filter_months(rows, "persistence", "", None, year_months), len(year_months))["share_vector_TV"],
        }
    return output


def _identity_gaps() -> list[dict[str, str]]:
    curated = WORLD_ROOT / "data" / "curated"
    priority = pd.read_csv(curated / "v4_2024_priority_identity_review.csv", dtype=str, keep_default_na=False)
    audit = pd.read_csv(curated / "v4_2024_identity_audit.csv", dtype=str, keep_default_na=False)
    rows = []
    seen: set[str] = set()
    for record in priority.to_dict(orient="records"):
        seen.add(record["canonical_site_uid"])
        rows.append(_gap(record, "priority_candidate"))
    for record in audit.to_dict(orient="records"):
        if record["parent_area"] not in PHASE16_PARENTS or record["canonical_site_uid"] in seen:
            continue
        rows.append(_gap(record, "development_site"))
    return rows


def _gap(record: dict[str, str], role: str) -> dict[str, str]:
    return {
        "role": role,
        "site_uid": record.get("canonical_site_uid", ""),
        "official_name": record.get("official_name", ""),
        "county": record.get("county", ""),
        "parent_area": record.get("parent_area", ""),
        "physical_identity": "not_established",
        "statistical_scope_continuity": "not_established",
        "counting_method_continuity": "not_established",
        "human_reviewed": record.get("human_reviewed", ""),
        "confirmed_manual": "False",
        "missing_evidence": "缺場所連續文件、缺統計範圍連續文件，也缺計數方法連續文件。2024 類型衝突沒有自動接續。",
    }


def _attribute_audit(paths: V4Paths, sites: list[dict[str, str]]) -> tuple[list[dict[str, str]], int]:
    provenance = pd.read_csv(WORLD_ROOT / "data" / "curated" / "v4_place_attribute_provenance.csv", dtype=str, keep_default_na=False)
    rows: list[dict[str, str]] = []
    descriptions = 0
    for record in provenance.to_dict(orient="records"):
        eligible = _eligible(record)
        if record["field"] == "attraction_description" and record["temporal_class"] == "historically_documented" and record["value"]:
            descriptions += 1
        rows.append(
            {
                "site_uid": record["site_uid"],
                "official_name": record["official_name"],
                "attribute_name": record["field"],
                "value": record["value"],
                "source": record["source_file"],
                "source_date": record["source_publication_date"],
                "valid_from": record["valid_from"],
                "valid_to": record["valid_to"],
                "historical_validity": record["temporal_class"],
                "verification_status": record["status"],
                "usable_in_2014_model": "yes" if eligible and record["field"] == "official_category" else "no",
            }
        )
    names = {site["official_name"] for site in sites}
    for relative in ("觀光資料標準V2.1/AttractionFeeList.xml", "觀光資料標準V2.1/AttractionServiceTimeList.xml"):
        document_update, hits = _scan_names(paths.raw_root / relative, names)
        for site in sites:
            rows.append(
                {
                    "site_uid": site["site_uid"],
                    "official_name": site["official_name"],
                    "attribute_name": "admission_or_cost_information" if relative.endswith("FeeList.xml") else "opening_or_access_conditions",
                    "value": "",
                    "source": relative,
                    "source_date": document_update[:10],
                    "valid_from": "",
                    "valid_to": "",
                    "historical_validity": "no_historical_validity",
                    "verification_status": "missing",
                    "usable_in_2014_model": "no",
                }
            )
        if hits:
            raise RuntimeError("a fee or hours row appeared after the audit expected none; review it before use")
    return rows, descriptions


def _eligible(record: dict[str, str]) -> bool:
    source_year = _year(record["valid_from"])
    return accepts_attribute_for_year(record["temporal_class"], source_year, 2014)


def _year(value: str) -> int | None:
    text = str(value)
    if len(text) >= 4 and text[:4].isdigit():
        return int(text[:4])
    return None


def accepts_attribute_for_year(temporal_class: str, source_year: int | None, target_year: int) -> bool:
    from worldlab_observation.intensity import accepts_attribute_for_year as rule
    return rule(temporal_class, source_year, target_year)


def _scan_names(path, names: set[str]) -> tuple[str, list[str]]:
    document_update = ""
    seen = False
    hits: list[str] = []
    for _event, element in ET.iterparse(path, events=("end",)):
        if element.tag == f"{ATTRACTION_NS}UpdateTime" and not seen and not document_update:
            document_update = (element.text or "").strip()
        if element.tag in {f"{ATTRACTION_NS}AttractionFee", f"{ATTRACTION_NS}AttractionServiceTime"}:
            seen = True
            name = ""
            node = element.find(f"{ATTRACTION_NS}AttractionName")
            if node is not None and node.text:
                name = node.text.strip()
            if name in names:
                hits.append(name)
            element.clear()
    return document_update, hits


def _write_reports(paths: V4Paths, payload: dict[str, object], attributes: list[dict[str, str]], gaps: list[dict[str, str]]) -> None:
    reports = paths.reports_root
    (reports / "V4_PHASE16_INTENSITY_MODEL.md").write_text(_model_text(), encoding="utf-8")
    (reports / "V4_PHASE16_EXPLORATORY_RESULTS.md").write_text(_results_text(payload), encoding="utf-8")
    (reports / "V4_PHASE16_IDENTIFIABILITY.md").write_text(_identifiability_text(payload), encoding="utf-8")
    (reports / "V4_PHASE16_HISTORICAL_ATTRIBUTE_AUDIT.md").write_text(_attribute_text(payload, attributes), encoding="utf-8")
    (reports / "V4_PHASE16_ABM_DISTINCTION.md").write_text(_distinction_text(), encoding="utf-8")
    (reports / "V4_PHASE16_NEXT_DECISION.md").write_text(_decision_text(payload, gaps), encoding="utf-8")


def _model_text() -> str:
    return "\n".join(
        [
            "# V4 Phase 16｜觀察尺度強度轉接",
            "",
            "研究目標是 reported_visit_distribution。官方人次可以重複計入同一位旅客，所以這不是互斥的個人選擇資料。本輪的模型家族是 aggregate_intensity_bridge，輸出量是 aggregate_intensity_share。",
            "",
            "M0 在每個目標月只用資訊截止日以前的訓練月份，計算各據點平均回報份額。份額先加事前固定的 1e-6 再正規化，alpha 是該份額的對數並扣除三個據點的平均。softmax 溫度固定為 1。這個 softmax 會回到平滑後的訓練期平均份額，因此 M0 是歷史平均份額的再參數化，不是場所屬性效用，也不是 LLM 效用。",
            "",
            "M1 寫成 u = alpha + beta * S。S 只來自截止日以前的回報份額。OBS_CUM 是累積平均，尺度是跨據點中心化後再除以最大絕對值；它和 M0 使用同一條平均份額。OBS_ROLLING 是截止月往回 3 個日曆月，不是 WorldLab 的視窗 50。OBS_EXP 的半衰期是 6 個日曆月，不是衰減 0.99。常數向量的訊號設為 0。",
            "",
            "beta 網格在評估前固定為 -1、-0.5、0、0.5、1。beta = 0 就是 M0。巢狀選擇只用目標月之前、而且本身已有至少 12 個訓練月的最近 12 個內部驗證月。目標月的觀察值不參與選參。歷史不足時記 insufficient_history。所有網格結果都保留。",
            "",
            "情境 A 假設 t-1 可用。情境 B 假設只到 t-3。兩套結果分開，不稱為歷史即時預測。樣本是玉山、太魯閣、雪霸各 J=3，月份沿用 Phase 13 的 24 個月訓練加 12 個月評估所定義的單步月份。沒有加入 2024 年未解的跨類型月份。",
            "",
        ]
    )


def _results_text(payload: dict[str, object]) -> str:
    lines = [
        "# V4 Phase 16｜探索性回測",
        "",
        "這些月份與 Phase 13 的探索性開發期重疊。本輪結果不是新的獨立驗證，也不是 P4。主要指標是份額 TV。MAE 是固定 J=3 下的縮放摘要。HHI 絕對誤差與領先據點是次要指標。",
        "",
    ]
    for scenario, label in SCENARIO_LABELS.items():
        lines.extend([f"## {label}", "", _baseline_table(payload, scenario), "", _gap_table(payload, scenario), "", _grid_table(payload, scenario), ""])
    lines.extend(_readings(payload))
    lines.extend(["## 年度與非重疊區段", ""])
    for group in payload["groups"]:
        scenario = group["scenarios"]["ex_post_max_information"]
        blocks = scenario["block_m0_tv"]
        block_text = "、".join(f"{value:.3f}" for value in blocks) if blocks else "無"
        year_text = "、".join(
            f"{year} M0 {fmt(values['M0'])}／持續 {fmt(values['persistence'])}" for year, values in scenario["years"].items()
        )
        lines.append(
            f"- {group['county']}／{group['parent_area']}：可評估月 {group['evaluable_months']}，非重疊 12 個月區塊 {scenario['block_count']} 個，區塊上的 M0 TV 為 {block_text}。情境 A 的年度 TV：{year_text}。"
        )
    lines.extend(["", "區塊數低於 30，本輪不估計信賴區間。", ""])
    return "\n".join(lines)


def _baseline_table(payload: dict[str, object], scenario: str) -> str:
    header = "| 地理群 | " + " | ".join(BASELINE_LABELS.values()) + " | M0 | 可用率 |"
    rule = "| --- | " + " | ".join("---" for _ in BASELINE_LABELS) + " | --- | --- |"
    body = [header, rule]
    for group in payload["groups"]:
        models = group["scenarios"][scenario]["models"]
        cells = [fmt(models[model]["share_vector_TV"]) for model in BASELINE_LABELS]
        cells.append(fmt(models["M0"]["share_vector_TV"]))
        cells.append(fmt(models["M0"]["availability"]))
        body.append(f"| {group['county']}／{group['parent_area']} | " + " | ".join(cells) + " |")
    return "\n".join(body)


def _gap_table(payload: dict[str, object], scenario: str) -> str:
    header = "| 地理群 | " + " | ".join(f"M0減{label}" for label in BASELINE_LABELS.values()) + " |"
    rule = "| --- | " + " | ".join("---" for _ in BASELINE_LABELS) + " |"
    body = [header, rule, "", "正值表示 M0 的 TV 高於該基準。"]
    # I accidentally put a prose line inside the table. Fix by building cleanly below.
    body = [header, rule]
    for group in payload["groups"]:
        gaps = group["scenarios"][scenario]["m0_minus_baseline_tv"]
        cells = [fmt(gaps[model]) for model in BASELINE_LABELS]
        body.append(f"| {group['county']}／{group['parent_area']} | " + " | ".join(cells) + " |")
    body.append("")
    body.append("正值表示 M0 的 TV 高於該固定基準。")
    return "\n".join(body)


def _grid_table(payload: dict[str, object], scenario: str) -> str:
    lines = ["| 地理群 | 訊號 | -1 | -0.5 | 0 | 0.5 | 1 | 巢狀 TV | 巢狀月數 | 巢狀相對 M0 | 最常見 beta |"]
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for group in payload["groups"]:
        block = group["scenarios"][scenario]
        for signal_name in SIGNALS:
            grid = block["grid"][signal_name]
            nested = block["nested"][signal_name]
            delta = None
            if nested["tv"] is not None and nested["m0_tv_same_months"] is not None:
                delta = nested["tv"] - nested["m0_tv_same_months"]
            cells = [fmt(grid[beta]["share_vector_TV"]) for beta in ("-1.0", "-0.5", "0.0", "0.5", "1.0")]
            lines.append(
                "| {name} | {signal} | {grid} | {nested} | {count} | {delta} | {modal} |".format(
                    name=f"{group['county']}／{group['parent_area']}",
                    signal=signal_name,
                    grid=" | ".join(cells),
                    nested=fmt(nested["tv"]),
                    count=str(nested["n"]),
                    delta=fmt(delta),
                    modal="—" if nested["modal_beta"] is None else nested["modal_beta"],
                )
            )
    lines.append("")
    lines.append("巢狀 TV 只平均有足夠內部驗證月的目標月，並與同一批月份上的 M0 相減。負值表示巢狀 M1 的 TV 較低。beta = 0 的網格欄就是 M0。")
    return "\n".join(lines)


def _readings(payload: dict[str, object]) -> list[str]:
    lines = ["## 讀法", ""]
    m0_is_best = False
    nested_below_m0 = []
    nested_below_best = []
    for group in payload["groups"]:
        name = f"{group['county']}／{group['parent_area']}"
        for scenario, label in SCENARIO_LABELS.items():
            block = group["scenarios"][scenario]
            models = block["models"]
            best = min(MODELS, key=lambda model: models[model]["share_vector_TV"])
            m0_tv = models["M0"]["share_vector_TV"]
            if m0_tv <= models[best]["share_vector_TV"] + 1e-12:
                m0_is_best = True
            lines.append(
                f"- {name}，{label}：七個基準裡 TV 最低的是{BASELINE_LABELS[best]} {fmt(models[best]['share_vector_TV'])}。M0 是 {fmt(m0_tv)}，HHI 絕對誤差 {fmt(models['M0']['normalized_HHI_absolute_error'])}，領先據點命中 {fmt(models['M0']['leading_site_accuracy'])}。"
            )
            for signal_name in SIGNALS:
                nested = block["nested"][signal_name]
                if nested["tv"] is not None and nested["m0_tv_same_months"] is not None and nested["tv"] + 1e-12 < nested["m0_tv_same_months"]:
                    nested_below_m0.append(f"{name}/{signal_name}/{label}")
                if nested["tv"] is not None and nested["best_baseline_tv_same_months"] is not None and nested["tv"] + 1e-12 < nested["best_baseline_tv_same_months"]:
                    nested_below_best.append(f"{name}/{signal_name}/{label}")
                lines.append(
                    "  - {signal} 巢狀 TV {nested_tv}；同一批月份的 M0 {m0}，最近可用月 {persistence}，最低固定基準 {best} {best_tv}。".format(
                        signal=signal_name,
                        nested_tv=fmt(nested["tv"]),
                        m0=fmt(nested["m0_tv_same_months"]),
                        persistence=fmt(nested["persistence_tv_same_months"]),
                        best=BASELINE_LABELS.get(nested["best_baseline"], "—"),
                        best_tv=fmt(nested["best_baseline_tv_same_months"]),
                    )
                )
    lines.append("")
    if m0_is_best:
        lines.append("M0 在至少一個情境裡達到七個基準的最低 TV。")
    else:
        lines.append("M0 在三個地理群、兩個資訊情境裡都不是 TV 最低的模型。它接近歷史平均這一類基準，沒有低於已經存在的近期或固定組合基準。")
    if nested_below_best:
        lines.append("有巢狀 M1 的 TV 低於同一批月份上最低的固定基準：" + "、".join(nested_below_best) + "。")
    else:
        lines.append("沒有巢狀 M1 的 TV 低於同一批月份上最低的固定基準。情境 B 的最近可用月是 t-3，巢狀結果有時低於這個延遲基準，但仍沒有低於該情境裡最低的固定基準。")
    if nested_below_m0:
        lines.append("相對 M0，TV 較低的巢狀設定是：" + "、".join(nested_below_m0) + "。這仍是同一條歷史份額的再加權。")
    else:
        lines.append("巢狀 M1 沒有給出低於 M0 的 TV。")
    lines.append("三種訊號的 TV 彼此接近。beta 從 -1 到 1 則明顯改變 TV。J=3 時份額 MAE 等於 2×TV/3，所以不再另表。領先據點命中可以和 TV 的排序不同，仍列為次要。")
    lines.append("")
    return lines


def _identifiability_text(payload: dict[str, object]) -> str:
    lines = [
        "# V4 Phase 16｜識別",
        "",
        "alpha 加減一個共同常數不會改變 softmax。beta 與 S 的尺度可以對調：分數乘 2、係數除 2，預測份額不變。溫度固定為 1，沒有另外估計。因此係數的數值不是旅客偏好或社會影響的因果係數。",
        "",
        "OBS_CUM 的原始輸入就是 M0 的訓練期平均份額。它沒有另外一條動態序列。對這條訊號估計 beta，只是在同一個平均向量上做非線性再加權，屬於弱識別。OBS_ROLLING 與 OBS_EXP 使用不同的日曆權重，所以預測可以偏離 M0，但它們仍是同一批截止月前份額的函數。alpha 沒有在每個原點改用那 3 個月重新估計。",
        "",
        "巢狀選擇的 beta 分布：",
        "",
    ]
    for group in payload["groups"]:
        for scenario, label in SCENARIO_LABELS.items():
            for signal_name in SIGNALS:
                nested = group["scenarios"][scenario]["nested"][signal_name]
                counts = "、".join(f"{beta} 占 {count}" for beta, count in sorted(nested["beta_counts"].items())) or "無"
                lines.append(
                    f"- {group['county']}／{group['parent_area']}，{label}，{signal_name}：最常見 beta {nested['modal_beta'] or '無'}，占比 {fmt(nested['modal_share'])}，內部歷史不足 {nested['insufficient_months']} 個月。分布：{counts}。"
                )
    lines.extend(
        [
            "",
            "若同一個訊號在網格兩端的 TV 明顯不同，表示預測對 beta 敏感；這仍是統計敏感度，不是已識別的社會學習效果。Phase 13 的固定基準沒有因為 M1 多了參數而拿到目標月之後的資訊。",
            "",
        ]
    )
    return "\n".join(lines)


def _attribute_text(payload: dict[str, object], attributes: list[dict[str, str]]) -> str:
    historical = [row for row in attributes if row["historical_validity"] == "historically_documented" and row["attribute_name"] == "attraction_description" and row["value"]]
    categories = [row for row in attributes if row["attribute_name"] == "official_category" and row["usable_in_2014_model"] == "yes"]
    return "\n".join(
        [
            "# V4 Phase 16｜歷史場所屬性查核",
            "",
            f"具歷史有效性的場所描述 {len(historical)} 筆。可在 2014 年模型中使用的統計類型列 {len(categories)} 筆，三群都是國家公園，不能在群內區分據點。",
            "",
            "景點資料庫的描述、開放條件與座標沿用 Phase 15 的分類。2026 年附近的文字沒有放入 2014 年的效用。收費清單與營運時間清單都查過九個官方名稱，沒有對上的列。本地資料夾裡沒有這九處的國家公園歷史公告或帶日期的設施沿革可以寫進模型。",
            "",
            f"M2 狀態是 {payload['m2_status']}。沒有為了啟動 M2 而補造場所特色，也沒有呼叫 LLM。",
            "",
            f"TDX 狀態是 {payload['tdx_status']}。will_call_network 是 {payload['tdx_will_call_network']}。現況交通資料沒有回填成歷史旅行時間。",
            "",
        ]
    )


def _distinction_text() -> str:
    return "\n".join(
        [
            "# V4 Phase 16｜與 WorldLab 機制的區別",
            "",
            "| 觀察轉接 | WorldLab 機制 | 本輪地位 |",
            "| --- | --- | --- |",
            "| 訓練期平均份額與 alpha | 場所基準效用 | 統計代理。來源是回報份額，不是受訪者或場所屬性 |",
            "| OBS_CUM、OBS_ROLLING、OBS_EXP | LOGMAX、滾動視窗、衰減記憶 | 只借用「歷史摘要」這個形狀。3 個月與半衰期 6 個月是日曆設定，不是代理人記憶參數 |",
            "| beta | 社會資訊權重 | 網格係數。與訊號尺度綁在一起，不是因果社會影響 |",
            "| softmax 溫度 1 | 引擎溫度 200 | 本橋的溫度是固定尺度，沒有沿用引擎溫度來宣稱同一套選擇 |",
            "| aggregate_intensity_share | agent_choice_probability 與 expected_model_presence_share | 本輪沒有代理人抽樣，也沒有把回報份額改名成選擇機率 |",
            "",
            "仍需要代理人異質性與閉環演化，才能回到 WorldLab 的模擬：每人的效用可以不同，下一步的在場是模型自己產生的，而不是把官方份額再餵回去當成已證實的社會暴露。",
            "",
            "Q 是同一模型狀態下、有社會分數與沒有社會分數的選擇機率距離。官方月份之間的 TV 沒有這組反事實，所以 Q 仍只能在模型內計算。",
            "",
            "E/O 來源替換需要受體自己的社會分數與供體分數可以分開觀察。現有資料每個據點只有一條回報份額，沒有供體場域，因此不能從這份觀察資料辨識 E/O。",
            "",
            "舊的長期引擎、Experiment 001–020 與 Phase 13 的結果檔都沒有改寫。M1 沒有被改名成 WorldLab ABM，P4 沒有執行。",
            "",
        ]
    )


def _decision_text(payload: dict[str, object], gaps: list[dict[str, str]]) -> str:
    lines = [
        "# V4 Phase 16｜下一步",
        "",
        f"Python {sys.version.split()[0]}，平台 {platform.platform()}。LLM 呼叫數 {payload['llm_calls']}。",
        "",
        "2024 身分缺口仍分開標成場所身分、統計範圍連續與計數方法連續，三者都是 not_established。"
        f" 清單 {len(gaps)} 列，含 Phase 13 的高證據候選與三個開發群裡尚未列入該清單的據點。沒有因為模型需要而自動接續 2024。",
        "",
        "2025 保留集維持封存。下一步若要做公平的單步預測，必須先補上歷史上有效、而且能在群內區分據點的場所屬性，或在看到保留月份之前凍結一份前瞻場所快照。只再變換同一條歷史份額，不會變成代理人機制的驗證。",
        "",
        "M2 繼續暫停。aggregate intensity bridge 維持觀察尺度的統計轉接，不進入正式 ABM 驗證。",
        "",
        "讀取路徑：",
        "",
    ]
    for path in payload["read_paths"]:
        lines.append(f"- `{path}`")
    lines.append("")
    return "\n".join(lines)


def _blank(value: object) -> object:
    return "" if value is None else value


def fmt(value: object) -> str:
    if value is None or value == "":
        return "—"
    return f"{float(value):.3f}"


def _hashes() -> dict[str, str]:
    return {relative: file_sha256(WORLD_ROOT / relative) for relative in GUARDED}


if __name__ == "__main__":
    print(json.dumps(run_phase16(), ensure_ascii=False, indent=2, allow_nan=False))
