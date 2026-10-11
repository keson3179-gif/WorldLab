from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd

from worldlab_data.textutil import (
    comparison_key,
    county_parts,
    file_sha256,
    nfkc,
    parse_calendar_year,
    parse_count,
    read_text_with_encoding,
    stable_site_id,
)

MONTH_COLUMNS = [f"{month}月" for month in range(1, 13)]
WIDE_REQUIRED = ["年別", "類型", "觀光遊憩區", "細分", "縣市", *MONTH_COLUMNS]
VISIT_STATUSES_KEPT = {"reported", "reported_zero"}


def read_monthly_csv(path: Path) -> tuple[pd.DataFrame, str]:
    text, encoding = read_text_with_encoding(path)
    frame = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    frame.columns = [nfkc(column) for column in frame.columns]
    return frame, encoding


def _wide_records(frame: pd.DataFrame, source_file: str, source_version: str) -> pd.DataFrame:
    missing = [column for column in WIDE_REQUIRED if column not in frame.columns]
    if missing:
        raise ValueError(f"{source_file} 缺少欄位 {missing}")
    records: list[dict[str, object]] = []
    for row in frame.to_dict(orient="records"):
        year = parse_calendar_year(row.get("年別"))
        county, county_en = county_parts(row.get("縣市"))
        area = nfkc(row.get("觀光遊憩區"))
        subdivision = nfkc(row.get("細分"))
        category = nfkc(row.get("類型"))
        site_id = stable_site_id([category, area, subdivision, county])
        annual_raw = nfkc(row.get("合計", ""))
        annual_value, annual_status = parse_count(annual_raw)
        month_values: list[tuple[int, float | None, str]] = []
        for month_number, column in enumerate(MONTH_COLUMNS, start=1):
            visits, status = parse_count(row.get(column))
            month_values.append((month_number, visits, status))
        numeric_months = [value for _, value, status in month_values if status in VISIT_STATUSES_KEPT and value is not None]
        complete = all(status in VISIT_STATUSES_KEPT for _, _, status in month_values)
        annual_flag = "not_checked"
        if annual_status in {"missing", "missing_token"}:
            annual_flag = "annual_total_missing"
        elif annual_status == "invalid_negative":
            annual_flag = "annual_total_invalid"
        elif annual_status == "unparsed_token":
            annual_flag = "annual_total_unparsed"
        elif complete and annual_status in VISIT_STATUSES_KEPT:
            annual_flag = "annual_total_match" if abs(float(sum(numeric_months)) - float(annual_value)) < 1e-6 else "annual_total_mismatch"
        else:
            annual_flag = "annual_total_incomplete_year"
        for month_number, visits, status in month_values:
            records.append(
                {
                    "site_id": site_id,
                    "year_month": f"{year:04d}-{month_number:02d}" if year else "",
                    "year": year,
                    "month": month_number,
                    "visits": visits,
                    "observation_status": status,
                    "measurement_method": "unknown",
                    "source_file": source_file,
                    "source_version": source_version,
                    "available_at": "",
                    "published_at": "",
                    "downloaded_at": "",
                    "revised_at": "",
                    "publication_history_unknown": True,
                    "revision_flag": False,
                    "annual_reconciliation": annual_flag,
                    "category": category,
                    "parent_area": area,
                    "official_name": subdivision,
                    "county": county,
                    "county_en": county_en,
                    "county_code": nfkc(row.get("縣市別")),
                    "source_row_key": comparison_key("|".join([str(year), category, area, subdivision, county])),
                }
            )
    return pd.DataFrame.from_records(records)


def load_wide_visitor_csv(path: Path) -> tuple[pd.DataFrame, str, str]:
    frame, encoding = read_monthly_csv(path)
    long_frame = _wide_records(frame, path.name, file_sha256(path))
    return long_frame, encoding, "wide_month_columns"


def profile_xlsx(path: Path) -> dict[str, object]:
    import openpyxl

    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheets: list[dict[str, object]] = []
    try:
        for sheet in workbook.worksheets:
            preview: list[list[str]] = []
            for index, row in enumerate(sheet.iter_rows(values_only=True)):
                if index >= 12:
                    break
                preview.append(["" if cell is None else nfkc(cell) for cell in row[:12]])
            header_hit = any(any(cell in MONTH_COLUMNS for cell in row) for row in preview)
            sheets.append(
                {
                    "sheet_name": sheet.title,
                    "header_has_month_columns": header_hit,
                    "preview_rows": preview,
                }
            )
    finally:
        workbook.close()
    return {"file_name": path.name, "sheets": sheets, "parsed_as_visits": False, "notes": "版面尚未確認，不把工作表換算成人次"}


def collapse_versions(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Identical multi-source values collapse to one row. Disagreements are quarantined."""
    if frame.empty:
        return frame.copy(), frame.copy()
    retained_rows: list[dict[str, object]] = []
    quarantine_rows: list[dict[str, object]] = []
    for _, group in frame.groupby(["site_id", "year_month"], dropna=False):
        versions = sorted(set(group["source_file"].astype(str)))
        comparable = []
        for record in group.to_dict(orient="records"):
            if record["observation_status"] in VISIT_STATUSES_KEPT:
                comparable.append((float(record["visits"]), record["observation_status"]))
        unique_values = set(comparable)
        if len(versions) > 1 and len(unique_values) > 1:
            for record in group.to_dict(orient="records"):
                quarantine_rows.append(
                    {
                        **record,
                        "visits": None,
                        "observation_status": "conflicting_source_version",
                        "revision_flag": True,
                    }
                )
            continue
        chosen = group.iloc[0].to_dict()
        if len(versions) > 1:
            chosen["source_file"] = "|".join(versions)
            chosen["revision_flag"] = False
        if chosen["annual_reconciliation"] == "annual_total_mismatch":
            chosen["strict_month_eligible"] = False
        else:
            chosen["strict_month_eligible"] = chosen["observation_status"] in VISIT_STATUSES_KEPT
        retained_rows.append(chosen)
    retained = pd.DataFrame.from_records(retained_rows)
    quarantine = pd.DataFrame.from_records(quarantine_rows) if quarantine_rows else pd.DataFrame(columns=list(frame.columns) + ["revision_flag"])
    return retained, quarantine


def analysis_months(frame: pd.DataFrame) -> pd.DataFrame:
    """Months that can enter a descriptive panel. Mismatched annual totals stay flagged, not rewritten."""
    if frame.empty:
        return frame.copy()
    mask = frame["observation_status"].isin(VISIT_STATUSES_KEPT)
    out = frame.loc[mask].copy()
    out["strict_asof_eligible"] = False
    out.loc[out["annual_reconciliation"] == "annual_total_mismatch", "panel_use"] = "descriptive_unreconciled_total"
    out.loc[out["annual_reconciliation"] != "annual_total_mismatch", "panel_use"] = "descriptive_reconciled_or_unchecked"
    return out


def status_history(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for site_id, group in frame.groupby("site_id"):
        present = group[group["observation_status"].isin(VISIT_STATUSES_KEPT)]["year_month"].dropna().astype(str)
        present_set = set(present)
        if not present_set:
            rows.append(
                {
                    "site_id": site_id,
                    "official_name": group["official_name"].iloc[0],
                    "county": group["county"].iloc[0],
                    "parent_area": group["parent_area"].iloc[0],
                    "first_year_month": "",
                    "last_year_month": "",
                    "reported_months": 0,
                    "missing_months_inside_span": 0,
                    "reported_zero_months": int((group["observation_status"] == "reported_zero").sum()),
                    "status_label": "no_reported_month",
                    "closure_confirmed": False,
                }
            )
            continue
        first = min(present_set)
        last = max(present_set)
        span = pd.period_range(first, last, freq="M")
        span_keys = [str(period) for period in span]
        missing = [key for key in span_keys if key not in present_set]
        rows.append(
            {
                "site_id": site_id,
                "official_name": group["official_name"].iloc[0],
                "county": group["county"].iloc[0],
                "parent_area": group["parent_area"].iloc[0],
                "category": group["category"].iloc[0],
                "first_year_month": first,
                "last_year_month": last,
                "reported_months": len(present_set),
                "missing_months_inside_span": len(missing),
                "reported_zero_months": int((group["observation_status"] == "reported_zero").sum()),
                "status_label": "gap_inside_span" if missing else "continuous_reported_span",
                "closure_confirmed": False,
                "notes": "區間兩端只表示這份檔案有回報，不表示開幕或結束營業",
            }
        )
    return pd.DataFrame.from_records(rows)


def anomaly_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if frame.empty:
        return pd.DataFrame()
    for record in frame.to_dict(orient="records"):
        status = record["observation_status"]
        if status in {"invalid_negative", "unparsed_token", "conflicting_source_version"}:
            rows.append({**_anomaly_identity(record), "anomaly_type": status, "detail": record.get("visits")})
        if record.get("annual_reconciliation") == "annual_total_mismatch" and record.get("month") == 1:
            rows.append({**_anomaly_identity(record), "anomaly_type": "annual_total_mismatch", "detail": "合計不等於十二個月相加，月份原值保留"})
    yearly = (
        frame[frame["observation_status"].isin(VISIT_STATUSES_KEPT)]
        .groupby(["site_id", "year"], dropna=False)["visits"]
        .sum()
        .reset_index()
    )
    for site_id, group in yearly.groupby("site_id"):
        ordered = group.sort_values("year")
        previous = None
        for record in ordered.to_dict(orient="records"):
            current = float(record["visits"])
            if previous not in (None, 0) and current >= 0:
                ratio = current / float(previous)
                if ratio >= 5 or ratio <= 0.2:
                    rows.append(
                        {
                            "site_id": site_id,
                            "year_month": f"{int(record['year']):04d}",
                            "anomaly_type": "level_shift_review",
                            "detail": f"相鄰年合計比 {ratio:.3f}；這是覆核門檻，不是錯誤判定",
                        }
                    )
            previous = current
    return pd.DataFrame.from_records(rows)


def _anomaly_identity(record: dict[str, object]) -> dict[str, object]:
    return {
        "site_id": record.get("site_id"),
        "year_month": record.get("year_month"),
        "official_name": record.get("official_name"),
        "county": record.get("county"),
        "source_file": record.get("source_file"),
    }


def possible_renames(history: pd.DataFrame) -> pd.DataFrame:
    """Flag adjacent spans with similar names. Do not merge them."""
    from rapidfuzz import fuzz

    if history.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    grouped = list(history.groupby(["county", "parent_area"]))
    for (_, _), group in grouped:
        records = group.to_dict(orient="records")
        for left_index, left in enumerate(records):
            for right in records[left_index + 1 :]:
                score = fuzz.ratio(comparison_key(left["official_name"]), comparison_key(right["official_name"]))
                if score < 90 or left["site_id"] == right["site_id"]:
                    continue
                rows.append(
                    {
                        "left_site_id": left["site_id"],
                        "right_site_id": right["site_id"],
                        "left_name": left["official_name"],
                        "right_name": right["official_name"],
                        "county": left["county"],
                        "parent_area": left["parent_area"],
                        "name_score": score,
                        "decision": "needs_manual_review",
                        "notes": "名稱接近但未合併",
                    }
                )
    return pd.DataFrame.from_records(rows)


def write_visitor_outputs(
    frame: pd.DataFrame,
    processed: Path,
    interim: Path,
    reports: Path,
    xlsx_profiles: list[dict[str, object]],
) -> dict[str, int]:
    processed.mkdir(parents=True, exist_ok=True)
    interim.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    retained, quarantine = collapse_versions(frame)
    history = status_history(retained)
    combined = pd.concat([retained, quarantine], ignore_index=True) if not quarantine.empty else retained
    anomalies = anomaly_table(combined)
    renames = possible_renames(history)
    retained.to_parquet(processed / "site_month_visits.parquet", index=False)
    history.to_parquet(processed / "site_status_history.parquet", index=False)
    anomalies.to_csv(reports / "visitor_anomalies.csv", index=False, encoding="utf-8-sig")
    if not quarantine.empty:
        quarantine.to_parquet(processed / "site_month_visits_quarantine.parquet", index=False)
    renames.to_csv(interim / "possible_renames.csv", index=False, encoding="utf-8-sig")
    (interim / "xlsx_profiles.json").write_text(json.dumps(xlsx_profiles, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "visit_rows": int(len(retained)),
        "quarantine_rows": int(len(quarantine)),
        "sites": int(retained["site_id"].nunique()) if not retained.empty else 0,
        "annual_mismatch_site_years": int(
            retained.loc[retained["annual_reconciliation"] == "annual_total_mismatch", ["site_id", "year"]].drop_duplicates().shape[0]
        )
        if not retained.empty
        else 0,
        "rename_candidates": int(len(renames)),
    }
