from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from worldlab_data.calendar_data import is_google_calendar, monthly_from_daily, read_calendar, version_conflicts
from worldlab_data.catalog import events_monthly, load_class_labels, parse_attractions, parse_events
from worldlab_data.config import V4Paths
from worldlab_data.inventory import scan_directory, write_inventory
from worldlab_data.matching import apply_manual_crosswalk, attraction_master, candidate_matches
from worldlab_data.panel import build_panel, candidate_groups
from worldlab_data.visitors import load_wide_visitor_csv, profile_xlsx, write_visitor_outputs
from worldlab_data.weather_data import prior_station_links, station_month_from_frozen


PLAN_NOTE = "V4 計畫書位於工作區「論文本文/WorldLab_NSTC_V4_Observational_Validation_Refined.docx」。資料定義以該文件的觀察式設計與實際檔案為準，不改寫論文。"


def run_inventory(paths: V4Paths, *, dry_run: bool) -> dict[str, object]:
    if not paths.raw_root.is_dir():
        raise FileNotFoundError(f"找不到原始資料目錄：{paths.raw_root}")
    frame = scan_directory(paths.raw_root)
    summary = {
        "files": int(len(frame)),
        "raw_root": str(paths.raw_root),
        "dry_run": dry_run,
        "plan_note": PLAN_NOTE,
    }
    if not dry_run:
        paths.ensure_output_dirs()
        write_inventory(frame, paths, PLAN_NOTE)
        _write_pointer(paths)
    return summary


def run_clean(paths: V4Paths, *, dry_run: bool) -> dict[str, object]:
    visitor_paths = [path for path in paths.raw_root.glob("*.csv") if "月別" in path.name]
    if not visitor_paths:
        raise FileNotFoundError("原始目錄直下沒有月別遊客人次 CSV")
    frames = []
    encodings = {}
    for path in visitor_paths:
        frame, encoding, layout = load_wide_visitor_csv(path)
        frames.append(frame)
        encodings[path.name] = {"encoding": encoding, "layout": layout, "rows": int(len(frame))}
    visits = pd.concat(frames, ignore_index=True)
    xlsx_dir = _find_dir(paths.raw_root, "累計表")
    profiles = [profile_xlsx(path) for path in sorted(xlsx_dir.glob("*.xlsx"))] if xlsx_dir else []
    calendars = []
    calendar_notes = []
    calendar_dir = _find_dir(paths.raw_root, "行事曆")
    if calendar_dir:
        for path in sorted(calendar_dir.glob("*.csv")):
            if is_google_calendar(path):
                calendar_notes.append(f"略過 Google 複本 {path.name}")
                continue
            try:
                daily, encoding = read_calendar(path)
            except (UnicodeError, ValueError) as exc:
                calendar_notes.append(f"{path.name} 未解析：{exc}")
                continue
            calendars.append(monthly_from_daily(daily))
            calendar_notes.append(f"{path.name} encoding={encoding}")
    calendar = pd.concat(calendars, ignore_index=True) if calendars else pd.DataFrame()
    conflicts = version_conflicts(calendar) if not calendar.empty else pd.DataFrame()
    attraction_path = _find_file(paths.raw_root, "AttractionList.xml")
    event_path = _find_file(paths.raw_root, "EventList.xml")
    attractions, attraction_updated = parse_attractions(attraction_path) if attraction_path else (pd.DataFrame(), "")
    events, event_updated = parse_events(event_path) if event_path else (pd.DataFrame(), "")
    labels = load_class_labels(paths.raw_root / "氣象資料" / "02-pipeline" / "metadata" / "official_class_crosswalk_v0.2.csv")
    if not attractions.empty and labels:
        attractions["category_labels"] = attractions["category_codes"].map(lambda codes: _label_codes(str(codes), labels))
    weather_metrics = paths.raw_root / "氣象資料" / "02-pipeline" / "data_processed" / "station_month_heat_metrics_v0.2.csv"
    weather_registry = paths.raw_root / "氣象資料" / "02-pipeline" / "metadata" / "station_registry.csv"
    station_links = paths.raw_root / "氣象資料" / "02-pipeline" / "metadata" / "attraction_station_primary_high_v0.2.csv"
    weather = station_month_from_frozen(weather_metrics, weather_registry) if weather_metrics.exists() else pd.DataFrame()
    links = prior_station_links(station_links) if station_links.exists() else pd.DataFrame()
    summary = {
        "dry_run": dry_run,
        "visitor_sources": encodings,
        "xlsx_unparsed": len(profiles),
        "calendar_files": calendar_notes,
        "calendar_conflict_months": int(len(conflicts)),
        "attractions": int(len(attractions)),
        "attraction_document_update": attraction_updated,
        "events": int(len(events)),
        "event_document_update": event_updated,
        "implausible_event_ends": int(events["implausible_end_year"].sum()) if not events.empty else 0,
        "weather_rows": int(len(weather)),
        "prior_station_links": int(len(links)),
        "wide_visit_rows_before_collapse": int(len(visits)),
    }
    if dry_run:
        return summary
    paths.ensure_output_dirs()
    _write_pointer(paths)
    visit_summary = write_visitor_outputs(visits, paths.processed, paths.interim, paths.reports_root, profiles)
    summary.update(visit_summary)
    if not calendar.empty:
        calendar.to_parquet(paths.processed / "calendar_monthly.parquet", index=False)
    if not conflicts.empty:
        conflicts.to_csv(paths.reports_root / "calendar_version_conflicts.csv", index=False, encoding="utf-8-sig")
    if not attractions.empty:
        attractions.to_parquet(paths.interim / "poi_attractions.parquet", index=False)
    if not events.empty:
        events.to_parquet(paths.interim / "events.parquet", index=False)
        events_monthly(events).to_parquet(paths.processed / "events_monthly.parquet", index=False)
    if not weather.empty:
        weather.to_parquet(paths.processed / "weather_monthly.parquet", index=False)
    if not links.empty:
        links.to_csv(paths.interim / "prior_station_links_not_revalidated.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        columns=["site_id", "year_month", "attention_value", "attention_proxy", "source", "notes"]
    ).to_parquet(paths.processed / "attention_monthly.parquet", index=False)
    (paths.interim / "clean_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_visitor_qa(paths, summary)
    return summary


def run_match(paths: V4Paths, *, dry_run: bool) -> dict[str, object]:
    visits = pd.read_parquet(paths.processed / "site_month_visits.parquet")
    pois = pd.read_parquet(paths.interim / "poi_attractions.parquet")
    sites = visits.drop_duplicates("site_id")[
        ["site_id", "official_name", "county", "parent_area", "category"]
    ]
    candidates = candidate_matches(sites, pois)
    crosswalk = apply_manual_crosswalk(candidates, paths.manual_crosswalk)
    master = attraction_master(sites, crosswalk)
    summary = {
        "dry_run": dry_run,
        "sites": int(len(master)),
        "candidate_rows": int(len(candidates)),
        "confirmed_manual": int((crosswalk["match_status"] == "confirmed_manual").sum()) if not crosswalk.empty else 0,
        "needs_manual_review": int((master["match_status"] == "needs_manual_review").sum()) if not master.empty else 0,
    }
    if dry_run:
        return summary
    paths.ensure_output_dirs()
    master.to_parquet(paths.processed / "attraction_master.parquet", index=False)
    crosswalk.to_csv(paths.curated / "site_crosswalk.csv", index=False, encoding="utf-8-sig")
    candidates.to_csv(paths.reports_root / "site_matching_review.csv", index=False, encoding="utf-8-sig")
    return summary


def run_panel(paths: V4Paths, *, dry_run: bool) -> dict[str, object]:
    visits = pd.read_parquet(paths.processed / "site_month_visits.parquet")
    history = pd.read_parquet(paths.processed / "site_status_history.parquet")
    membership = candidate_groups(history, visits)
    exploratory = build_panel(visits, membership, require_backtest=False)
    official = build_panel(visits, membership, require_backtest=True)
    included = int(membership["included"].sum()) if not membership.empty else 0
    groups = int(membership.loc[membership["included"] == True, "competition_group_id"].nunique()) if not membership.empty else 0  # noqa: E712
    summary = {
        "dry_run": dry_run,
        "candidate_groups": groups,
        "included_site_rows": included,
        "exploratory_rows": int(len(exploratory)),
        "official_backtest_rows": int(len(official)),
        "official_reason": "計數方式未知，且同一遊憩區的細分據點尚未確認可以互相替代，所以正式回測面板沒有群組。",
    }
    if dry_run:
        return summary
    paths.ensure_output_dirs()
    membership.to_csv(paths.curated / "competition_groups.csv", index=False, encoding="utf-8-sig")
    exploratory.to_parquet(paths.processed / "exploratory_visit_share_panel.parquet", index=False)
    official.to_parquet(paths.processed / "site_month_panel.parquet", index=False)
    return summary


def _write_pointer(paths: V4Paths) -> None:
    payload = {
        "raw_root": str(paths.raw_root),
        "copied": False,
        "notes": "原始檔維持在來源目錄，唯讀。這個資料夾只記錄路徑，不複製檔案。",
    }
    (paths.raw_pointer / "SOURCE_POINTER.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_visitor_qa(paths: V4Paths, summary: dict[str, object]) -> None:
    lines = [
        "# 遊客人次清洗紀錄",
        "",
        f"- 寬表展開後、版本折疊前：{summary['wide_visit_rows_before_collapse']} 列",
        f"- 折疊後月列：{summary.get('visit_rows')}",
        f"- 據點數：{summary.get('sites')}",
        f"- 合計與月份不一致的據點年：{summary.get('annual_mismatch_site_years')}。月份原值保留，沒有改寫。",
        f"- 隔離列：{summary.get('quarantine_rows')}",
        f"- 可能改名、未合併：{summary.get('rename_candidates')}",
        f"- 年度 Excel 只做版面盤點、未換算人次：{summary['xlsx_unparsed']} 個檔",
        f"- 日曆版本衝突月份：{summary['calendar_conflict_months']}",
        "",
        "缺值維持缺值。文字中的停業、未統計與無法解析的內容沒有填成 0。",
        "發布時間無法從檔案重建，全部標記 publication_history_unknown。",
        "",
    ]
    (paths.reports_root / "visitor_data_qa.md").write_text("\n".join(lines), encoding="utf-8")


def _find_dir(root: Path, token: str) -> Path | None:
    for path in root.iterdir():
        if path.is_dir() and token in path.name:
            return path
    return None


def _find_file(root: Path, name: str) -> Path | None:
    matches = [path for path in root.rglob(name) if path.is_file()]
    return matches[0] if matches else None


def _label_codes(codes: str, labels: dict[str, str]) -> str:
    names = [labels.get(code, code) for code in codes.split(",") if code]
    return ",".join(names)
