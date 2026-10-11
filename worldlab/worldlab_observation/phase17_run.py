"""Inventory external features for the three development groups. Visit shares are not features."""

from __future__ import annotations

import calendar
import json
import platform
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import date
from pathlib import Path

import pandas as pd

from worldlab_data.config import WORLD_ROOT, V4Paths, load_paths
from worldlab_data.phase13 import ANALYSIS_ROLE, TARGET_NAME, shift_month
from worldlab_data.tdx import dry_run_plan
from worldlab_data.textutil import file_sha256
from worldlab_observation.features import (
    CURRENT_SNAPSHOT,
    DECISION_NO_GO,
    EX_POST_WEATHER,
    HISTORICAL_ASOF,
    PHASE17_PARENTS,
    RETROSPECTIVE_PROXY,
    SOCIAL_UNOBSERVED,
    UNAVAILABLE,
    allows_strict_cutoff,
    common_feature_cancels,
    decision_from_eligibility,
    has_temporal_variation,
    identity_blocks_link,
)
from worldlab_observation.place_atlas import NAME_STEMS

READ_PATHS = (
    "data/curated/v4_forecast_panel_candidate.parquet",
    "data/curated/v4_group_comparability.csv",
    "data/curated/v4_canonical_site_crosswalk.csv",
    "data/curated/v4_2024_identity_audit.csv",
    "data/curated/v4_2024_priority_identity_review.csv",
    "data/curated/v4_real_place_atlas.csv",
    "政府行事曆",
    "氣象資料/02-pipeline/data_processed/station_month_heat_metrics_v0.2.csv",
    "氣象資料/02-pipeline/metadata/attraction_station_primary_high_v0.2.csv",
    "氣象資料/02-pipeline/metadata/station_registry.csv",
    "活動 - 觀光資訊資料庫/EventList.xml",
)
GUARDED = (
    "world/choice.py",
    "world/geography.py",
    "world/experiment_020.py",
    "worldlab_data/phase13.py",
    "worldlab_observation/adapter.py",
    "worldlab_observation/intensity.py",
    "data/curated/v4_forecast_panel_candidate.parquet",
    "reports/V4_PHASE16_EXPLORATORY_RESULTS.md",
    "reports/V4_PHASE15_PLACE_ATLAS.md",
)
PANEL_COLUMNS = ("group_id", "site_uid", "year_month", "dataset_role", "target", "confirmatory_holdout_candidate")
WEEKDAY = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6}
EVENT_NS = "{https://media.taiwan.net.tw/standard/schema/2.0}"
NAME_STEMS_EVENT = ("雪見", "布洛灣", "塔塔加", "排雲", "觀霧", "汶水", "玉山管理處", "臺八線", "太魯閣")


def run_phase17(paths: V4Paths | None = None) -> dict[str, object]:
    paths = paths or load_paths()
    before = _hashes()
    sites, months_by_site = _sites()
    weekend_rows, holiday_rows, calendar_note = _calendars(paths, sites, months_by_site)
    weather_rows, weather_note = _weather(paths, sites, months_by_site)
    event_note = _events(paths)
    tdx = dry_run_plan()
    candidates = weekend_rows + holiday_rows + weather_rows
    if any(str(row["year_month"]) >= "2024-01" for row in candidates):
        raise RuntimeError("a feature month lies outside the development window")
    if any(row.get("imputed") for row in candidates):
        raise RuntimeError("a feature value was imputed")
    eligibility = _eligibility(sites, months_by_site, candidates, event_note, tdx)
    provenance = _provenance(calendar_note, weather_note, event_note, tdx)
    gaps = _identity_gaps()
    decision = decision_from_eligibility(eligibility)
    curated = WORLD_ROOT / "data" / "curated"
    pd.DataFrame.from_records(candidates).to_parquet(curated / "v4_site_month_feature_candidates.parquet", index=False)
    pd.DataFrame.from_records(provenance).to_csv(curated / "v4_feature_provenance.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame.from_records(eligibility).to_csv(curated / "v4_feature_eligibility.csv", index=False, encoding="utf-8-sig")
    payload = {
        "decision": decision,
        "llm_calls": 0,
        "p4_executed": False,
        "holdout_visits_used": False,
        "candidate_rows": len(candidates),
        "calendar": calendar_note,
        "weather": weather_note,
        "events": event_note,
        "tdx_status": "offline_credentials_absent" if not tdx["credentials_present"] else "credentials_present_not_used",
        "tdx_will_call_network": bool(tdx["will_call_network"]),
        "social_status": SOCIAL_UNOBSERVED,
        "eligibility": eligibility,
        "identity_rows": len(gaps),
        "read_paths": list(READ_PATHS),
    }
    _write_reports(paths, payload, gaps)
    if _hashes() != before:
        raise RuntimeError("protected files changed while the feature audit was written")
    out = paths.output_root / "phase17_features"
    out.mkdir(parents=True, exist_ok=True)
    (out / "phase17_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame.from_records(gaps).to_csv(out / "identity_gaps.csv", index=False, encoding="utf-8-sig")
    return {
        "decision": decision,
        "candidate_rows": len(candidates),
        "social_status": SOCIAL_UNOBSERVED,
        "holdout_visits_used": False,
        "protected_files_unchanged": True,
    }


def _sites() -> tuple[list[dict[str, str]], dict[str, list[str]]]:
    curated = WORLD_ROOT / "data" / "curated"
    panel = pd.read_parquet(curated / "v4_forecast_panel_candidate.parquet", columns=list(PANEL_COLUMNS))
    if any(str(value) >= "2024-01" for value in panel["year_month"]):
        raise RuntimeError("the development panel contains a month outside the frozen window")
    if panel["dataset_role"].ne(ANALYSIS_ROLE).any() or panel["target"].ne(TARGET_NAME).any():
        raise RuntimeError("the panel is not the exploratory reported-visit panel")
    if panel["confirmatory_holdout_candidate"].astype(bool).any():
        raise RuntimeError("a confirmatory row entered the feature audit")
    comparability = pd.read_csv(curated / "v4_group_comparability.csv", dtype=str, keep_default_na=False)
    crosswalk = pd.read_csv(curated / "v4_canonical_site_crosswalk.csv", dtype=str, keep_default_na=False)
    chosen = comparability[comparability["definition"].eq("A_geographic") & comparability["parent_area"].isin(PHASE17_PARENTS)]
    sites: list[dict[str, str]] = []
    months_by_site: dict[str, list[str]] = {}
    for group in chosen.to_dict(orient="records"):
        scoped = panel[panel["group_id"].eq(group["group_id"])]
        membership = tuple(sorted(set(scoped["site_uid"])))
        if len(membership) != 3:
            raise RuntimeError("a development group does not have J=3")
        for site_uid in membership:
            linked = crosswalk[crosswalk["site_uid"].eq(site_uid) & crosswalk["match_status"].isin(["provisional_rule_match", "singleton"])]
            latest = linked.sort_values("last_year").iloc[-1]
            present = sorted({_month_key(value) for value in scoped.loc[scoped["site_uid"].eq(site_uid), "year_month"]})
            group_months = []
            for year_month, chunk in scoped.groupby("year_month"):
                if {str(row.site_uid) for row in chunk.itertuples(index=False)} == set(membership):
                    group_months.append(_month_key(year_month))
            months_by_site[site_uid] = sorted(set(group_months))
            sites.append(
                {
                    "group_id": group["group_id"],
                    "parent_area": group["parent_area"],
                    "county": str(latest["county"]),
                    "site_uid": site_uid,
                    "official_name": str(latest["official_name"]),
                    "panel_months": str(len(present)),
                }
            )
    if {row["official_name"] for row in sites} != set(NAME_STEMS):
        raise RuntimeError("the frozen nine sites were not recovered")
    return sites, months_by_site


def _calendars(paths: V4Paths, sites: list[dict[str, str]], months_by_site: dict[str, list[str]]) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    files = _calendar_files(paths.raw_root / "政府行事曆")
    named: dict[str, int] = defaultdict(int)
    mismatches = 0
    checked = 0
    for year, path in files.items():
        if year >= 2024:
            continue
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        for record in frame.to_dict(orient="records"):
            parsed = _parse_day(str(record["西元日期"]))
            if parsed is None or parsed.year != year or parsed.year >= 2024:
                continue
            checked += 1
            expected = WEEKDAY.get(str(record["星期"]))
            if expected is None or expected != parsed.weekday():
                mismatches += 1
            if str(record["備註"]).strip():
                named[f"{parsed.year:04d}-{parsed.month:02d}"] += 1
    weekend_rows: list[dict[str, object]] = []
    holiday_rows: list[dict[str, object]] = []
    for site in sites:
        for year_month in months_by_site[site["site_uid"]]:
            year, month = (int(part) for part in year_month.split("-"))
            weekends = sum(1 for day in range(1, calendar.monthrange(year, month)[1] + 1) if date(year, month, day).weekday() >= 5)
            weekend_rows.append(_row(site, "weekend_days", year_month, float(weekends), "gregorian_calendar", "", f"{year}-01", f"{year}-12", shift_month(year_month, -1), HISTORICAL_ASOF, "false", "true", "civil_calendar", "none"))
            if year in files and year < 2024:
                holiday_rows.append(
                    _row(
                        site, "named_holiday_days", year_month, float(named[year_month]),
                        files[year].name, "", f"{year}-01", f"{year}-12", "",
                        RETROSPECTIVE_PROXY, "false", "true", "office_calendar_file", "publication_date_unknown",
                    )
                )
    note = {
        "files_used_before_2024": sorted(path.name for year, path in files.items() if year < 2024),
        "years_without_office_calendar": [year for year in range(2012, 2024) if year not in files],
        "weekday_mismatches": mismatches,
        "days_checked": checked,
        "named_holiday_months": len(named),
    }
    if mismatches:
        raise RuntimeError("an office-calendar weekday does not match the civil date")
    return weekend_rows, holiday_rows, note


def _calendar_files(directory: Path) -> dict[int, Path]:
    chosen: dict[int, Path] = {}
    for path in sorted(directory.glob("*.csv")):
        if "Google" in path.name:
            continue
        match = re.match(r"(\d{3})", path.name)
        if not match:
            continue
        year = int(match.group(1)) + 1911
        current = chosen.get(year)
        if current is None or ("修正" in current.name and "修正" not in path.name):
            chosen[year] = path
    return chosen


def _weather(paths: V4Paths, sites: list[dict[str, str]], months_by_site: dict[str, list[str]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    link_path = paths.raw_root / "氣象資料" / "02-pipeline" / "metadata" / "attraction_station_primary_high_v0.2.csv"
    metric_path = paths.raw_root / "氣象資料" / "02-pipeline" / "data_processed" / "station_month_heat_metrics_v0.2.csv"
    registry_path = paths.raw_root / "氣象資料" / "02-pipeline" / "metadata" / "station_registry.csv"
    links = pd.read_csv(link_path, dtype=str, keep_default_na=False)
    linked = links[links["attraction_name_original"].isin(NAME_STEMS)]
    by_name = {str(row.attraction_name_original): row for row in linked.itertuples(index=False)}
    registry = pd.read_csv(registry_path, dtype=str, keep_default_na=False)
    elevation = {str(row.stno): str(row.alt_m) for row in registry.itertuples(index=False)}
    metrics = pd.read_csv(metric_path, dtype=str, keep_default_na=False, usecols=["station_id", "year_month", "weather_valid", "mean_tmax"])
    metrics = metrics[metrics["year_month"].lt("2024-01") & metrics["weather_valid"].eq("1") & metrics["mean_tmax"].ne("")]
    values = {(str(row.station_id), str(row.year_month)): float(row.mean_tmax) for row in metrics.itertuples(index=False)}
    rows: list[dict[str, object]] = []
    link_notes = []
    for site in sites:
        link = by_name.get(site["official_name"])
        if link is None:
            link_notes.append({"official_name": site["official_name"], "station_id": "", "station_km": "", "station_elevation_m": ""})
            continue
        station_id = str(link.station_primary)
        link_notes.append(
            {
                "official_name": site["official_name"],
                "station_id": station_id,
                "station_name": str(link.station_primary_name),
                "station_km": str(link.station_primary_km),
                "station_elevation_m": elevation.get(station_id, ""),
                "coordinate_confidence": str(link.coordinate_confidence),
                "elevation_difference_checked": False,
            }
        )
        for year_month in months_by_site[site["site_uid"]]:
            if year_month >= "2024-01":
                continue
            if (station_id, year_month) in values:
                rows.append(
                    _row(
                        site, "same_month_mean_daily_max_c", year_month, values[(station_id, year_month)],
                        metric_path.name, "", year_month, year_month, "",
                        RETROSPECTIVE_PROXY, "partial_one_site", "true", "prior_station_link_not_revalidated", EX_POST_WEATHER,
                    )
                )
            previous = shift_month(year_month, -1)
            if (station_id, previous) in values:
                rows.append(
                    _row(
                        site, "lagged_mean_daily_max_c", year_month, values[(station_id, previous)],
                        metric_path.name, "", previous, previous, "",
                        RETROSPECTIVE_PROXY, "partial_one_site", "true", "prior_station_link_not_revalidated", "publication_date_unknown",
                    )
                )
    note = {
        "linked_sites": [item["official_name"] for item in link_notes if item["station_id"]],
        "unlinked_sites": [item["official_name"] for item in link_notes if not item["station_id"]],
        "links": link_notes,
        "metric_months_used_before_2024": sorted({row["year_month"] for row in rows}),
        "climatology_columns_excluded": True,
        "elevation_difference_checked": False,
    }
    return rows, note


def _events(paths: V4Paths) -> dict[str, object]:
    path = paths.raw_root / "活動 - 觀光資訊資料庫" / "EventList.xml"
    starts: list[str] = []
    hits: list[dict[str, str]] = []
    document_update = ""
    seen = False
    count = 0
    for _event, element in ET.iterparse(path, events=("end",)):
        if element.tag == f"{EVENT_NS}UpdateTime" and not seen and not document_update:
            document_update = (element.text or "").strip()
        if element.tag != f"{EVENT_NS}Event":
            continue
        seen = True
        count += 1
        name = _xml_text(element, "EventName")
        start = _xml_text(element, "StartDateTime")[:7]
        if start:
            starts.append(start)
        if any(stem in name for stem in NAME_STEMS_EVENT):
            hits.append({"event_name": name, "start_month": start})
        element.clear()
    return {
        "document_update": document_update,
        "event_count": count,
        "start_min": min(starts) if starts else "",
        "start_max": max(starts) if starts else "",
        "development_window_hits": sum(1 for item in hits if item["start_month"] and item["start_month"] < "2024-01"),
        "name_hits": hits,
    }


def _eligibility(sites, months_by_site, candidates, event_note, tdx) -> list[dict[str, str]]:
    by_feature: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in candidates:
        by_feature[str(row["feature_name"])].append(row)
    output = []
    for feature_name, rows in by_feature.items():
        for parent in PHASE17_PARENTS:
            members = [site for site in sites if site["parent_area"] == parent]
            member_ids = {site["site_uid"] for site in members}
            group_rows = [row for row in rows if row["site_uid"] in member_ids]
            expected = sum(len(months_by_site[site["site_uid"]]) for site in members)
            observed_sites = {row["site_uid"] for row in group_rows}
            months = sorted({row["year_month"] for row in group_rows})
            contrasts = []
            for year_month in months:
                values = [float(row["value"]) for row in group_rows if row["year_month"] == year_month]
                contrasts.append(len(observed_sites) == 3 and len(values) == 3 and not common_feature_cancels(values))
            series = [float(row["value"]) for row in group_rows]
            availability = str(rows[0]["historical_availability_status"])
            leakage = str(rows[0]["leakage_risk"])
            share_increment = "true" if any(contrasts) else "false"
            output.append(
                {
                    "feature_name": feature_name,
                    "parent_area": parent,
                    "sites_with_values": str(len(observed_sites)),
                    "historical_months": str(len(months)),
                    "coverage": f"{len(group_rows) / expected:.3f}" if expected else "",
                    "within_group_variation": "true" if any(contrasts) else "false",
                    "temporal_variation": "true" if has_temporal_variation(series) else "false",
                    "publication_date_share": "1.000" if rows[0]["source_date"] else "0.000",
                    "strict_cutoff": "true" if allows_strict_cutoff(availability, leakage) else "false",
                    "ex_post_only": "true" if leakage == EX_POST_WEATHER else "false",
                    "independent_of_visit_share": "true",
                    "share_increment": share_increment,
                    "historical_availability_status": availability,
                    "analysis_role": EX_POST_WEATHER if leakage == EX_POST_WEATHER else "not_a_share_model",
                    "imputed_values": "0",
                }
            )
    output.append(
        {
            "feature_name": "public_attention",
            "parent_area": "all",
            "sites_with_values": "0",
            "historical_months": "0",
            "coverage": "0.000",
            "within_group_variation": "false",
            "temporal_variation": "false",
            "publication_date_share": "0.000",
            "strict_cutoff": "false",
            "ex_post_only": "false",
            "independent_of_visit_share": "true",
            "share_increment": "false",
            "historical_availability_status": UNAVAILABLE,
            "analysis_role": SOCIAL_UNOBSERVED,
            "imputed_values": "0",
        }
    )
    output.append(
        {
            "feature_name": "event_count",
            "parent_area": "all",
            "sites_with_values": "0",
            "historical_months": "0",
            "coverage": "0.000",
            "within_group_variation": "false",
            "temporal_variation": "false",
            "publication_date_share": "0.000",
            "strict_cutoff": "false",
            "ex_post_only": "false",
            "independent_of_visit_share": "true",
            "share_increment": "false",
            "historical_availability_status": CURRENT_SNAPSHOT,
            "analysis_role": "current_event_feed_not_historical",
            "imputed_values": "0",
        }
    )
    return output


def _provenance(calendar_note, weather_note, event_note, tdx) -> list[dict[str, str]]:
    rows = [
        {
            "feature_name": "weekend_days",
            "source_file": "gregorian_calendar",
            "source_date": "",
            "historical_availability_status": HISTORICAL_ASOF,
            "note": "週末天數由曆法決定，三個據點同月相同，softmax 會抵消。",
        },
        {
            "feature_name": "named_holiday_days",
            "source_file": "政府行事曆",
            "source_date": "",
            "historical_availability_status": RETROSPECTIVE_PROXY,
            "note": "辦公日曆檔沒有公布日。2012–2016 沒有本地檔，沒有填補。" + f" 缺年：{calendar_note['years_without_office_calendar']}。",
        },
        {
            "feature_name": "lagged_mean_daily_max_c",
            "source_file": "station_month_heat_metrics_v0.2.csv",
            "source_date": "",
            "historical_availability_status": RETROSPECTIVE_PROXY,
            "note": "沿用 v0.2 主站，沒有重配，也沒有景點高程。氣候距平欄未使用。發布日不可重建。",
        },
        {
            "feature_name": "same_month_mean_daily_max_c",
            "source_file": "station_month_heat_metrics_v0.2.csv",
            "source_date": "",
            "historical_availability_status": RETROSPECTIVE_PROXY,
            "note": "目標月的實際最高溫平均是事後觀測，分析角色是 ex_post_conditional_analysis。",
        },
        {
            "feature_name": "event_count",
            "source_file": "EventList.xml",
            "source_date": event_note["document_update"][:10],
            "historical_availability_status": CURRENT_SNAPSHOT,
            "note": f"活動 {event_note['event_count']} 筆，開始月 {event_note['start_min']} 至 {event_note['start_max']}。開發窗內同名活動 {event_note['development_window_hits']} 筆。",
        },
        {
            "feature_name": "public_attention",
            "source_file": "",
            "source_date": "",
            "historical_availability_status": UNAVAILABLE,
            "note": SOCIAL_UNOBSERVED,
        },
        {
            "feature_name": "tdx_access",
            "source_file": "",
            "source_date": "",
            "historical_availability_status": UNAVAILABLE,
            "note": "offline" if not tdx["credentials_present"] else "credentials_present_current_feed_not_historical",
        },
    ]
    for link in weather_note["links"]:
        rows.append(
            {
                "feature_name": "prior_station_link",
                "source_file": "attraction_station_primary_high_v0.2.csv",
                "source_date": "",
                "historical_availability_status": RETROSPECTIVE_PROXY if link["station_id"] else UNAVAILABLE,
                "note": f"{link['official_name']} station {link['station_id'] or 'none'} km {link.get('station_km', '')} elevation_checked false",
            }
        )
    return rows


def _identity_gaps() -> list[dict[str, str]]:
    curated = WORLD_ROOT / "data" / "curated"
    priority = pd.read_csv(curated / "v4_2024_priority_identity_review.csv", dtype=str, keep_default_na=False)
    audit = pd.read_csv(curated / "v4_2024_identity_audit.csv", dtype=str, keep_default_na=False)
    rows = []
    seen: set[str] = set()
    for record in priority.to_dict(orient="records"):
        if record["parent_area"] not in PHASE17_PARENTS:
            continue
        seen.add(record["canonical_site_uid"])
        rows.append(_gap(record))
    for record in audit.to_dict(orient="records"):
        if record["parent_area"] not in PHASE17_PARENTS or record["canonical_site_uid"] in seen:
            continue
        rows.append(_gap(record))
    return rows


def _gap(record: dict[str, str]) -> dict[str, str]:
    confirmed = str(record.get("confirmed_manual", "False")).lower() == "true"
    reviewed = str(record.get("human_reviewed", "False")).lower() == "true"
    if not identity_blocks_link(confirmed, reviewed):
        raise RuntimeError("an unresolved identity row was treated as linked")
    return {
        "site_uid": record.get("canonical_site_uid", ""),
        "official_name": record.get("official_name", ""),
        "parent_area": record.get("parent_area", ""),
        "physical_identity_continuity": "not_established",
        "statistical_scope_continuity": "not_established",
        "measurement_continuity": "not_established",
        "confirmed_manual": "False",
        "human_reviewed": record.get("human_reviewed", "False"),
        "missing_evidence": "沒有場所連續文件、統計範圍連續文件或計數方法連續文件。同名與同年人次一致都沒有升成口徑連續。",
    }


def _row(site, feature, year_month, value, source_file, source_date, valid_from, valid_to, available_at, availability, site_variation, temporal, measurement, leakage) -> dict[str, object]:
    return {
        "feature_name": feature,
        "group_id": site["group_id"],
        "parent_area": site["parent_area"],
        "site_uid": site["site_uid"],
        "official_name": site["official_name"],
        "year_month": year_month,
        "value": float(value),
        "source_file": source_file,
        "source_date": source_date,
        "valid_from": valid_from,
        "valid_to": valid_to,
        "available_at": available_at,
        "historical_availability_status": availability,
        "site_specific_variation": site_variation,
        "temporal_variation": temporal,
        "measurement_status": measurement,
        "leakage_risk": leakage,
        "imputed": False,
    }


def _month_key(value: object) -> str:
    return str(value)[:7]


def _parse_day(value: str) -> date | None:
    digits = re.sub(r"\D", "", value)
    if len(digits) != 8:
        return None
    return date(int(digits[:4]), int(digits[4:6]), int(digits[6:8]))


def _xml_text(parent: ET.Element, tag: str) -> str:
    node = parent.find(f"{EVENT_NS}{tag}")
    if node is None or node.text is None:
        return ""
    return node.text.strip()


def _write_reports(paths: V4Paths, payload: dict[str, object], gaps: list[dict[str, str]]) -> None:
    reports = paths.reports_root
    (reports / "V4_PHASE17_FEATURE_INVENTORY.md").write_text(_inventory_text(payload), encoding="utf-8")
    (reports / "V4_PHASE17_IDENTIFICATION_AUDIT.md").write_text(_identification_text(payload), encoding="utf-8")
    (reports / "V4_PHASE17_SOCIAL_INFORMATION_FEASIBILITY.md").write_text(_social_text(payload), encoding="utf-8")
    (reports / "V4_PHASE17_2024_IDENTITY_GAPS.md").write_text(_identity_text(gaps), encoding="utf-8")
    (reports / "V4_PHASE17_GO_NO_GO.md").write_text(_decision_text(payload), encoding="utf-8")


def _inventory_text(payload: dict[str, object]) -> str:
    calendar_note = payload["calendar"]
    weather_note = payload["weather"]
    event_note = payload["events"]
    lines = [
        "# V4 Phase 17｜外部特徵盤點",
        "",
        "固定群組是玉山、太魯閣、雪霸，各 J=3。沒有依 Phase 16 的誤差改群。特徵矩陣沒有讀取回報份額，也沒有填補缺值。",
        "",
        f"週末天數來自曆法，辦公日曆在 2024 年以前用到的檔案有 {len(calendar_note['files_used_before_2024'])} 個。沒有本地檔的年份是 {calendar_note['years_without_office_calendar']}。星期比對不一致 {calendar_note['weekday_mismatches']} 天。",
        "",
        f"v0.2 氣象主站對上的據點是：{'、'.join(weather_note['linked_sites']) or '無'}。沒有對上的是：{'、'.join(weather_note['unlinked_sites'])}。景點高程沒有檢查。氣候距平欄沒有使用。候選矩陣只保留 2024 年以前的測站月。",
        "",
        f"活動清單更新時間 {event_note['document_update']}，共 {event_note['event_count']} 筆，開始月 {event_note['start_min']} 至 {event_note['start_max']}。開發月份裡沒有可回接的同名活動。",
        "",
        f"TDX 狀態 {payload['tdx_status']}。網路關注資料狀態 {payload['social_status']}。",
        "",
        "| 特徵 | 地理群 | 有值據點 | 月份 | 覆蓋率 | 群內變異 | 時間變異 | 嚴格截止 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["eligibility"]:
        if row["parent_area"] == "all":
            continue
        lines.append(
            f"| {row['feature_name']} | {row['parent_area']} | {row['sites_with_values']} | {row['historical_months']} | {row['coverage']} | {row['within_group_variation']} | {row['temporal_variation']} | {row['strict_cutoff']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _identification_text(payload: dict[str, object]) -> str:
    return "\n".join(
        [
            "# V4 Phase 17｜識別稽核",
            "",
            "同一個群、同一個月，若三個據點的特徵值相同，共同係數會在 softmax 正規化時抵消。週末天數與有名稱的放假日都屬於這種群月共同衝擊。它們可以進入總需求的討論，不能記成群內份額的獨立增益。",
            "",
            "落後一個月的日最高溫平均，以及目標月本身的日最高溫平均，只出現在已有 v0.2 主站的據點。雪霸只有雪見，太魯閣只有遊客中心，玉山沒有。雪見的主站距離是 8.925 公里；太魯閣遊客中心的 4.819 公里來自舊研究座標，Phase 15 沒有把該座標記成同名同縣市的已核對位置。缺測的據點沒有被填上鄰近站的值，所以群內沒有完整的三點對照。景點與測站的高程差沒有檢查。",
            "",
            "目標月天氣的 leakage_risk 是 ex_post_conditional_analysis。落後天氣有歷史數值，但氣象月檔沒有發布日，因此是 retrospective_proxy_only，不能進入嚴格資訊截止。",
            "",
            "官方類型在三個據點都是國家公園，和場所固定效果裡的共同標籤共線。歷史回報份額沒有被放進這份特徵矩陣，也沒有被標成社會曝光。",
            "",
            "沒有事前指定、而且已有資料支持的據點交互作用。本輪沒有為了尋找顯著結果去配對特徵。",
            "",
        ]
    )


def _social_text(payload: dict[str, object]) -> str:
    event_note = payload["events"]
    return "\n".join(
        [
            "# V4 Phase 17｜歷史社會資訊",
            "",
            f"狀態是 {SOCIAL_UNOBSERVED}。本地公開資料沒有景點網路關注量、帶日期的評論或貼文數、歷史搜尋指標，也沒有可重建到月份的官方熱門排行。",
            "",
            "活動清單是當期快照。名稱提到太魯閣的活動，開始月落在開發窗之外，不能回填 2012–2023。公開活動則數也不等於旅客實際看到的資訊。",
            "",
            f"清單共 {event_note['event_count']} 筆，開始月範圍 {event_note['start_min']} 至 {event_note['start_max']}。",
            "",
            "過去人次仍是回報份額，不是已觀察的社會曝光。本輪沒有用它代替關注量。",
            "",
        ]
    )


def _identity_text(gaps: list[dict[str, str]]) -> str:
    lines = [
        "# V4 Phase 17｜2024 身分缺口",
        "",
        "玉山、太魯閣、雪霸的據點沿用 Phase 13 的跨來源候選與 Phase 12 的身分稽核。同名或年度人次一致沒有被升成口徑連續。",
        "",
        "| 據點 | 場所身分 | 統計範圍 | 計數方法 |",
        "| --- | --- | --- | --- |",
    ]
    for row in gaps:
        lines.append(
            f"| {row['official_name']} | {row['physical_identity_continuity']} | {row['statistical_scope_continuity']} | {row['measurement_continuity']} |"
        )
    lines.extend(
        [
            "",
            "三項都是 not_established。觀霧與雪見的 CSV 類型未變只是規則推定，human_reviewed 仍是 false。2025 保留集沒有讀取。在這三項仍未確認時，不能把 2023-12 當成 2025-01 的 t-1，也不能把 2024 的跨類型列併進開發月份。",
            "",
        ]
    )
    return "\n".join(lines)


def _decision_text(payload: dict[str, object]) -> str:
    return "\n".join(
        [
            "# V4 Phase 17｜是否進入 M0 + X",
            "",
            f"Python {sys.version.split()[0]}，平台 {platform.platform()}。決策是 {payload['decision']}。LLM 呼叫數 {payload['llm_calls']}。",
            "",
            "沒有一項特徵同時滿足：預測原點前可取得、三個據點都有值、而且同月的值在群內不同。週末與放假日在嚴格意義上週末可事前知道，但群內沒有差異。氣象只有部分據點的事後或發布日不明的代理。社會資訊沒有獨立來源。",
            "",
            "因此下一輪不開發 M0 + X 的份額回測，也不把缺值填起來、擴大 beta 搜尋或呼叫 LLM。若要改變這個決定，需要新的、帶發布日、而且能在同一群三個據點之間分開的歷史資料。",
            "",
            "仍不能由現有觀察資料識別的命題：旅客個人偏好、社會曝光的因果效果、Q、E/O 來源替換，以及 2024 以後的公平單步預測。",
            "",
            "讀取路徑：",
            "",
            *[f"- `{path}`" for path in payload["read_paths"]],
            "",
        ]
    )


def _hashes() -> dict[str, str]:
    return {relative: file_sha256(WORLD_ROOT / relative) for relative in GUARDED}


if __name__ == "__main__":
    print(json.dumps(run_phase17(), ensure_ascii=False, indent=2, allow_nan=False))
