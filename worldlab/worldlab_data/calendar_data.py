from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from worldlab_data.textutil import nfkc, read_text_with_encoding


def is_google_calendar(path: Path) -> bool:
    return "google" in path.name.casefold()


def read_calendar(path: Path) -> tuple[pd.DataFrame, str]:
    text, encoding = read_text_with_encoding(path)
    frame = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    frame.columns = [nfkc(column) for column in frame.columns]
    renamed = frame.rename(
        columns={
            "西元日期": "date_text",
            "星期": "weekday",
            "是否放假": "holiday_code",
            "備註": "note",
        }
    )
    required = {"date_text", "weekday", "holiday_code"}
    if not required.issubset(renamed.columns):
        raise ValueError(f"{path.name} 欄位不是西元日期/星期/是否放假，實際為 {list(frame.columns)}")
    renamed["date"] = pd.to_datetime(renamed["date_text"], format="%Y%m%d", errors="coerce")
    renamed["source_file"] = path.name
    renamed["encoding"] = encoding
    return renamed, encoding


def monthly_from_daily(frame: pd.DataFrame) -> pd.DataFrame:
    usable = frame.dropna(subset=["date"]).copy()
    usable["year"] = usable["date"].dt.year.astype(int)
    usable["month"] = usable["date"].dt.month.astype(int)
    usable["holiday_code"] = usable["holiday_code"].map(nfkc)
    usable["is_weekend"] = usable["weekday"].isin(["六", "日"])
    usable["is_official_holiday"] = usable["holiday_code"] == "2"
    usable["note"] = usable.get("note", "")
    rows: list[dict[str, object]] = []
    for (source_file, year, month), group in usable.groupby(["source_file", "year", "month"]):
        ordered = group.sort_values("date")
        flags = ordered["is_official_holiday"].tolist()
        long_days = _days_in_long_runs(flags)
        notes = " ".join(str(value) for value in ordered["note"].tolist())
        unknown_codes = sorted(set(ordered["holiday_code"]) - {"0", "2", ""})
        rows.append(
            {
                "year": int(year),
                "month": int(month),
                "year_month": f"{int(year):04d}-{int(month):02d}",
                "days_in_month": int(len(ordered)),
                "weekend_days": int(ordered["is_weekend"].sum()),
                "official_holiday_days": int(ordered["is_official_holiday"].sum()),
                "long_holiday_days": int(long_days),
                "lunar_new_year_indicator": int("春節" in notes),
                "source_file": source_file,
                "unknown_holiday_codes": ",".join(unknown_codes),
                "announcement_date": "",
                "publication_history_unknown": True,
            }
        )
    return pd.DataFrame.from_records(rows)


def _days_in_long_runs(flags: list[bool], minimum: int = 3) -> int:
    counted = 0
    run = 0
    runs: list[int] = []
    for flag in [*flags, False]:
        if flag:
            run += 1
        elif run:
            runs.append(run)
            run = 0
    for length in runs:
        if length >= minimum:
            counted += length
    return counted


def version_conflicts(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return monthly.copy()
    rows: list[dict[str, object]] = []
    value_columns = ["days_in_month", "weekend_days", "official_holiday_days", "long_holiday_days", "lunar_new_year_indicator"]
    for (year, month), group in monthly.groupby(["year", "month"]):
        if len(group) < 2:
            continue
        signatures = group[value_columns].drop_duplicates()
        if len(signatures) > 1:
            rows.append(
                {
                    "year_month": f"{int(year):04d}-{int(month):02d}",
                    "source_files": "|".join(group["source_file"].astype(str)),
                    "decision": "needs_manual_review",
                    "notes": "同一月份有多份日曆且數值不同，兩份都保留，不挑選",
                }
            )
    return pd.DataFrame.from_records(rows)
