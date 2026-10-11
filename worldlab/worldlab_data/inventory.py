from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from worldlab_data.config import V4Paths
from worldlab_data.textutil import detect_encoding, file_sha256, nfkc

PROFILE_EXTENSIONS = {".csv", ".tsv", ".json", ".xml", ".xlsx", ".xls", ".pdf", ".parquet"}
PROFILE_BYTE_LIMIT = 40_000_000


def scan_directory(raw_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(raw_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in PROFILE_EXTENSIONS:
            continue
        rows.append(_profile(raw_root, path))
    return pd.DataFrame.from_records(rows)


def _profile(raw_root: Path, path: Path) -> dict[str, object]:
    relative = path.relative_to(raw_root)
    category = relative.parts[0] if len(relative.parts) > 1 else "root"
    suffix = path.suffix.lower()
    try:
        return _profile_inner(path, relative, category, suffix)
    except Exception as exc:
        return _empty_profile(path, relative, category, suffix, f"盤點此檔時發生錯誤，已跳過內容：{type(exc).__name__}")


def _profile_inner(path: Path, relative: Path, category: str, suffix: str) -> dict[str, object]:
    encoding = "not_applicable"
    sheet_name = ""
    row_count = None
    column_count = None
    null_count = None
    duplicate_count = None
    min_date = ""
    max_date = ""
    date_granularity = ""
    geographic = ""
    primary_key = ""
    notes = ""
    publication = ""
    digest = file_sha256(path)
    large = path.stat().st_size > PROFILE_BYTE_LIMIT
    if large and suffix in {".csv", ".json"}:
        notes = "檔案較大，只記錄大小與雜湊，不做逐列缺值統計"
    elif suffix == ".csv":
        encoding, row_count, column_count, null_count, duplicate_count, headers, min_date, max_date = _csv_profile(path)
        date_granularity, geographic, primary_key, publication = _guess_csv(path.name, headers, min_date, max_date)
    elif suffix == ".xml":
        encoding = "utf-8"
        row_count, publication, primary_key, date_granularity = _xml_profile(path)
        geographic = "景點或活動點位（座標）" if row_count else ""
    elif suffix == ".xlsx":
        sheet_name, row_count, column_count, notes = _xlsx_profile(path, notes)
        encoding = "not_applicable"
    elif suffix == ".parquet":
        row_count, column_count = _parquet_profile(path)
        encoding = "not_applicable"
    elif suffix == ".json":
        encoding = "utf-8"
        row_count, column_count, notes = _json_profile(path, notes)
    elif suffix == ".pdf":
        encoding = "not_applicable"
        notes = "本次未擷取 PDF 內文，不把作業要點改寫成計數規則。"
    elif suffix == ".xls":
        notes = "沒有讀取舊版 xls 的套件，未解析內容。"
    return _filled_profile(
        path,
        relative,
        category,
        suffix,
        encoding,
        sheet_name,
        row_count,
        column_count,
        min_date,
        max_date,
        date_granularity,
        geographic,
        primary_key,
        null_count,
        duplicate_count,
        digest,
        publication,
        notes,
    )


def _filled_profile(
    path: Path,
    relative: Path,
    category: str,
    suffix: str,
    encoding: str,
    sheet_name: str,
    row_count: int | None,
    column_count: int | None,
    min_date: str,
    max_date: str,
    date_granularity: str,
    geographic: str,
    primary_key: str,
    null_count: int | None,
    duplicate_count: int | None,
    digest: str,
    publication: str,
    notes: str,
) -> dict[str, object]:
    return {
        "file_name": path.name,
        "relative_path": relative.as_posix(),
        "source_category": category,
        "source_organization": _organization(category, path.name),
        "source_url": "",
        "file_format": suffix.lstrip("."),
        "encoding": encoding,
        "sheet_name": sheet_name,
        "row_count": row_count,
        "column_count": column_count,
        "min_date": min_date,
        "max_date": max_date,
        "date_granularity": date_granularity,
        "geographic_granularity": geographic,
        "primary_key_candidate": primary_key,
        "null_count": null_count,
        "duplicate_count": duplicate_count,
        "sha256": digest,
        "download_time": "",
        "source_publication_time": publication,
        "notes": notes.strip(),
        "inventory_run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _empty_profile(path: Path, relative: Path, category: str, suffix: str, notes: str) -> dict[str, object]:
    return _filled_profile(path, relative, category, suffix, "not_applicable", "", None, None, "", "", "", "", "", None, None, "", "", notes)


def _csv_profile(path: Path) -> tuple[str, int, int, int, int, list[str], str, str]:
    encoding = detect_encoding(path)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding=encoding)
    headers = [nfkc(column) for column in frame.columns]
    frame.columns = headers
    null_count = int(frame.apply(lambda column: column.map(lambda value: nfkc(value) == "")).sum().sum())
    duplicate_count = int(frame.duplicated().sum())
    min_date, max_date = _date_bounds(frame)
    return encoding, int(len(frame)), int(len(headers)), null_count, duplicate_count, headers, min_date, max_date


def _date_bounds(frame: pd.DataFrame) -> tuple[str, str]:
    if "年別" in frame.columns:
        years = [int(value) for value in frame["年別"] if str(value).isdigit()]
        if years:
            return str(min(years)), str(max(years))
    if "西元日期" in frame.columns:
        values = sorted(value for value in frame["西元日期"].astype(str) if value.isdigit())
        if values:
            return values[0], values[-1]
    if "year_month" in frame.columns:
        values = sorted(value for value in frame["year_month"].astype(str) if value)
        if values:
            return values[0], values[-1]
    return "", ""


def _guess_csv(file_name: str, headers: list[str], min_date: str, max_date: str) -> tuple[str, str, str, str]:
    header_set = set(headers)
    if {"年別", "1月", "合計"} <= header_set:
        return "year_month_wide", "統計據點（縣市，無座標）", "年別+類型+觀光遊憩區+細分+縣市", ""
    if {"西元日期", "是否放假"} <= header_set:
        return "day", "全國辦公日曆（無行政區）", "西元日期+來源檔", ""
    if "year_month" in header_set and "station_id" in header_set:
        return "month", "氣象站", "station_id+year_month", ""
    return "", "", "", ""


def _xml_profile(path: Path) -> tuple[int, str, str, str]:
    import xml.etree.ElementTree as ET

    count = 0
    publication = ""
    if path.name == "AttractionList.xml":
        record_tag = "Attraction"
        key = "AttractionID"
    elif path.name == "EventList.xml":
        record_tag = "Event"
        key = "EventID"
    else:
        record_tag = ""
        key = ""
    granularity = "snapshot" if record_tag else ""
    if path.name.startswith("schema-"):
        return 0, "", key, granularity
    for _, element in ET.iterparse(path, events=("end",)):
        local = element.tag.rsplit("}", 1)[-1]
        if local == "UpdateTime" and not publication and element.text and count == 0:
            publication = nfkc(element.text)
        if record_tag and local == record_tag:
            count += 1
        element.clear()
    return count, publication, key, granularity


def _xlsx_profile(path: Path, notes: str) -> tuple[str, int | None, int | None, str]:
    import openpyxl

    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        names = [sheet.title for sheet in workbook.worksheets]
        first = workbook.worksheets[0]
        rows = 0
        columns = 0
        for index, row in enumerate(first.iter_rows(values_only=True)):
            rows += 1
            columns = max(columns, len(row))
            if index >= 5000:
                notes = (notes + " 工作表列數超過抽樣上限，row_count 是下限。").strip()
                break
    finally:
        workbook.close()
    return "|".join(names), rows, columns, (notes + " 年度簿只做表頭盤點，未直接換算成人次。").strip()


def _parquet_profile(path: Path) -> tuple[int | None, int | None]:
    import pyarrow.parquet as pq

    metadata = pq.ParquetFile(path).metadata
    return int(metadata.num_rows), int(metadata.num_columns)


def _json_profile(path: Path, notes: str) -> tuple[int | None, int | None, str]:
    if path.stat().st_size > 5_000_000:
        return None, None, (notes + " JSON 較大，未展開。").strip()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        columns = len(payload[0]) if payload and isinstance(payload[0], dict) else None
        return len(payload), columns, notes
    if isinstance(payload, dict):
        return 1, len(payload), notes
    return None, None, notes


def _organization(category: str, file_name: str) -> str:
    if "氣象" in category:
        return "中央氣象署／既有高溫研究整理"
    if "行事曆" in category or "辦公日曆" in file_name:
        return "行政院人事行政總處"
    if "觀光" in category or "活動" in category or "遊客" in file_name or "景點" in file_name:
        return "交通部觀光署"
    return ""


def write_inventory(frame: pd.DataFrame, paths: V4Paths, plan_note: str) -> None:
    paths.reports_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(paths.reports_root / "data_inventory.csv", index=False, encoding="utf-8-sig")
    mapping_rows: list[dict[str, str]] = []
    for record in frame.to_dict(orient="records"):
        if record["file_format"] != "csv" or not record["relative_path"]:
            continue
        source = paths.raw_root / str(record["relative_path"])
        if not source.exists() or source.stat().st_size > PROFILE_BYTE_LIMIT:
            continue
        try:
            header = pd.read_csv(source, nrows=0, encoding=str(record["encoding"]) if record["encoding"] not in {"", "not_applicable"} else "utf-8-sig")
        except (OSError, UnicodeError, pd.errors.ParserError):
            continue
        for column in header.columns:
            mapping_rows.append(
                {
                    "relative_path": str(record["relative_path"]),
                    "column_name": nfkc(column),
                    "source_category": str(record["source_category"]),
                    "notes": "欄名照錄，不推定跨檔相同名稱就是同一個據點",
                }
            )
    pd.DataFrame(mapping_rows).to_csv(paths.reports_root / "source_field_mapping.csv", index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    lines = [
        "# 資料盤點",
        "",
        plan_note,
        "",
        f"原始目錄：`{paths.raw_root}`",
        f"檔案數：{len(frame)}",
        "",
        "完整欄位見 `data_inventory.csv`。這裡只列各類檔案數。同名景點沒有被當成同一個據點。",
        "",
    ]
    if not frame.empty:
        counts = frame.groupby("source_category").size()
        lines.append("| 類別 | 檔案數 |")
        lines.append("| --- | ---: |")
        for name, count in counts.items():
            lines.append(f"| {name} | {count} |")
    (paths.reports_root / "data_inventory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    gaps = [
        "# 缺少或不能直接使用的項目",
        "",
        plan_note,
        "",
        "- V4 計畫書已位於「論文本文/WorldLab_NSTC_V4_Observational_Validation_Refined.docx」。論文要求的計數口徑，須對照觀光署統計作業要點；該 PDF 本次仍未擷取內文。",
        "- TDX 交通資料尚未取得，金鑰不在環境中。現況 API 不能回填 2018 或 2020 的交通條件。",
        "- 沒有 Google Trends、Wikimedia 或公開評論檔，無法建立 attention_proxy 的實值。",
        "- 遊客人次表沒有可重建的歷史發布日期，不能放進嚴格的當時可得預測。",
        "- 各據點的計數方式（售票、人數感應或推估）不在已解析的表內。《主要觀光遊憩據點統計作業要點》PDF 本次未擷取內文。",
        "- 景點主檔與統計據點沒有人工確認的 crosswalk。模糊比對只進待審清單。",
        "- 年度 Excel 若版面與月別 CSV 不同，不換算成人次。",
        "- 活動檔是 2025–2026 的現況清單，沒有 2012–2024 的歷史活動，也沒有公告日期。",
        "- 辦公日曆不涵蓋 2012–2016。民國 114 年有兩份非 Google 版本，數值不同時兩份都保留。",
        "- 氣象月檔沿用既有 v0.2 凍結表。沒有景點高程，不能檢查海拔差。日均溫不在 TX02 裡。",
        "- geopandas 未納入這次環境，空間距離不重新計算。",
        "",
    ]
    (paths.reports_root / "missing_data_requirements.md").write_text("\n".join(gaps), encoding="utf-8")
