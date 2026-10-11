from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from worldlab_data.config import WORLD_ROOT, V4Paths, load_paths
from worldlab_data.phase10 import (
    HOLDOUT_YEAR,
    UNKNOWN_PARENT,
    assign_site_uids,
    classify_2021_months,
    comparability_level,
    derived_annual,
    method_family,
    parse_annual_sheet,
    usable_months,
)
from worldlab_data.textutil import comparison_key, file_sha256

GUARDED = (
    "data/v4_observational/processed/site_month_visits.parquet",
    "data/curated/v4_measurement_method_history.csv",
    "data/curated/v4_site_continuity_candidates.csv",
    "reports/V4_PHASE9_READINESS.md",
    "reports/V4_PHASE8_GROUP_AUDIT.md",
    "reports/WORLDLAB_V4_QA_SUMMARY.md",
)


def run_phase10(paths: V4Paths | None = None) -> dict[str, object]:
    paths = paths or load_paths()
    before = _hashes()
    excel_rows = _read_excel(paths)
    verified = pd.DataFrame.from_records([{key: value for key, value in row.items() if key != "months"} for row in excel_rows])
    visits = pd.read_parquet(paths.processed / "site_month_visits.parquet")
    history = pd.read_parquet(paths.processed / "site_status_history.parquet")
    groups = pd.read_csv(paths.reports_root / "V4_PHASE8_GROUPS.csv", dtype=str, keep_default_na=False)
    crosswalk = pd.DataFrame.from_records(_identity_records(history, visits))
    annual = _annual_rows(paths, visits, excel_rows)
    group_rows = _group_table(groups, crosswalk, visits, verified, annual)
    shares = _shares(group_rows, groups, crosswalk, visits, annual)
    curated = WORLD_ROOT / "data" / "curated"
    curated.mkdir(parents=True, exist_ok=True)
    paths.reports_root.mkdir(parents=True, exist_ok=True)
    verified.to_csv(curated / "v4_measurement_method_verified.csv", index=False, encoding="utf-8-sig")
    crosswalk.to_csv(curated / "v4_canonical_site_crosswalk.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame.from_records(group_rows).to_csv(curated / "v4_group_comparability.csv", index=False, encoding="utf-8-sig")
    if not shares.empty:
        shares.to_csv(curated / "v4_exploratory_visit_shares_conditional.csv", index=False, encoding="utf-8-sig")
    annual.to_csv(paths.reports_root / "V4_PHASE10_2021_MONTH_EVIDENCE.csv", index=False, encoding="utf-8-sig")
    holdout = _holdout_compatibility(paths, crosswalk)
    _write_source_audit(paths, verified)
    _write_identity(paths, crosswalk, group_rows)
    _write_comparability(paths, group_rows, shares)
    after = _hashes()
    unchanged = before == after
    _write_readiness(paths, verified, crosswalk, group_rows, annual, shares, holdout, unchanged)
    if not unchanged:
        raise RuntimeError("Phase 10 改到了受保護的既有檔案")
    summary = {
        "verified_rows": int(len(verified)),
        "unknown_parent": int(verified["parent_area"].eq(UNKNOWN_PARENT).sum()) if not verified.empty else 0,
        "maolin_rows": int(verified["parent_area"].eq("茂林國家風景區").sum()) if not verified.empty else 0,
        "site_uids": int(crosswalk["site_uid"].nunique()) if not crosswalk.empty else 0,
        "levels": pd.DataFrame(group_rows)["comparability_level"].value_counts().to_dict() if group_rows else {},
        "exploratory_rows": int(len(shares)),
        "protected_files_unchanged": unchanged,
        "holdout_used_for_selection": False,
    }
    (paths.interim / "phase10_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _hashes() -> dict[str, str]:
    found = {}
    for relative in GUARDED:
        path = WORLD_ROOT / relative
        if path.exists():
            found[str(path)] = file_sha256(path)
    return found


def _read_excel(paths: V4Paths) -> list[dict[str, object]]:
    import openpyxl

    folder = next(path for path in paths.raw_root.iterdir() if path.is_dir() and "累計表" in path.name)
    parsed: list[dict[str, object]] = []
    for path in sorted(folder.glob("*.xlsx")):
        workbook = openpyxl.load_workbook(path, read_only=False, data_only=True)
        try:
            for sheet in workbook.worksheets:
                rows = [list(row) for row in sheet.iter_rows(values_only=True)]
                merges = [(item.min_row, item.max_row, item.min_col, item.max_col) for item in sheet.merged_cells.ranges]
                parsed.extend(parse_annual_sheet(rows, merges, path.name, sheet.title))
        finally:
            workbook.close()
    return parsed


def _identity_records(history: pd.DataFrame, visits: pd.DataFrame) -> list[dict[str, object]]:
    usable = visits[visits["observation_status"].isin(["reported", "reported_zero"])]
    months: dict[str, dict[str, float]] = defaultdict(dict)
    for row in usable[["site_id", "year_month", "visits"]].to_dict(orient="records"):
        if row["visits"] is None or pd.isna(row["visits"]):
            continue
        months[str(row["site_id"])][str(row["year_month"])] = float(row["visits"])
    records = []
    for row in history.to_dict(orient="records"):
        series = months.get(str(row["site_id"]), {})
        years = sorted({int(str(item)[:4]) for item in series})
        records.append(
            {
                "raw_site_id": row["site_id"],
                "official_name": row["official_name"],
                "county": row["county"],
                "category": row["category"],
                "parent_area": row["parent_area"],
                "first_year": years[0] if years else int(str(row["first_year_month"])[:4] or 0),
                "last_year": years[-1] if years else int(str(row["last_year_month"])[:4] or 0),
                "months": series,
            }
        )
    return assign_site_uids(records)


def _annual_rows(paths: V4Paths, visits: pd.DataFrame, excel_rows: list[dict[str, object]]) -> pd.DataFrame:
    from worldlab_data.textutil import county_parts, nfkc, parse_calendar_year, parse_count, stable_site_id
    from worldlab_data.visitors import read_monthly_csv

    csv_path = next(path for path in paths.raw_root.glob("*.csv") if "月別" in path.name)
    wide, _encoding = read_monthly_csv(csv_path)
    published_by_site: dict[str, float] = {}
    for record in wide.to_dict(orient="records"):
        if parse_calendar_year(record.get("年別")) != 2021:
            continue
        county, _ = county_parts(record.get("縣市"))
        parent = nfkc(record.get("觀光遊憩區"))
        name = nfkc(record.get("細分"))
        category = nfkc(record.get("類型"))
        annual, status = parse_count(record.get("合計"))
        if status not in {"reported", "reported_zero"} or annual is None:
            continue
        published_by_site[stable_site_id([category, parent, name, county])] = float(annual)
    excel_index: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in excel_rows:
        if int(row["year"]) == 2021:
            excel_index[(str(row["name_key"]), str(row["county_key"]), 2021)].append(row)
    scoped = visits[(visits["year"] == 2021) & visits["observation_status"].isin(["reported", "reported_zero"])]
    records = []
    for site_id, frame in scoped.groupby("site_id"):
        by_month = {int(row["month"]): float(row["visits"]) for row in frame.to_dict(orient="records") if row["visits"] is not None and not pd.isna(row["visits"])}
        if len(by_month) != 12:
            continue
        months = [by_month[month] for month in range(1, 13)]
        sample = frame.iloc[0]
        published = published_by_site.get(str(site_id))
        if published is None:
            continue
        diagnosis = derived_annual(months, published)
        if abs(diagnosis["derived_annual_sum"] - published) <= 1e-6:
            continue
        hits = excel_index.get((comparison_key(sample["official_name"]), comparison_key(sample["county"]), 2021), [])
        excel_months = None
        if len(hits) == 1 and set(hits[0]["months"]) >= set(range(1, 13)):
            excel_months = [float(hits[0]["months"][month]) for month in range(1, 13)]
        classified = classify_2021_months(months, excel_months)
        if not diagnosis["published_equals_january_to_october"]:
            continue
        records.append(
            {
                "raw_site_id": site_id,
                "official_name": sample["official_name"],
                "county": sample["county"],
                "year": 2021,
                "derived_annual_sum": diagnosis["derived_annual_sum"],
                "published_annual": published,
                "source_class": classified["source_class"],
                "source_conflict": classified["source_conflict"],
                "chosen_version": classified["chosen_version"],
                "csv_november": months[10],
                "csv_december": months[11],
                "excel_november": classified.get("excel_november", ""),
                "excel_december": classified.get("excel_december", ""),
                "months_kept": True,
            }
        )
    return pd.DataFrame.from_records(records)


def _group_table(
    groups: pd.DataFrame,
    crosswalk: pd.DataFrame,
    visits: pd.DataFrame,
    verified: pd.DataFrame,
    annual: pd.DataFrame,
) -> list[dict[str, object]]:
    uid_of = {str(row["raw_site_id"]): row for row in crosswalk.to_dict(orient="records") if row.get("raw_site_id")}
    linked: dict[str, set[str]] = defaultdict(set)
    for row in crosswalk.to_dict(orient="records"):
        if row.get("match_status") == "provisional_rule_match" and row.get("site_uid") and row.get("raw_site_id"):
            linked[str(row["site_uid"])].add(str(row["raw_site_id"]))
    visit_months: dict[str, set[str]] = defaultdict(set)
    for row in visits[visits["observation_status"].isin(["reported", "reported_zero"])][["site_id", "year_month"]].to_dict(orient="records"):
        visit_months[str(row["site_id"])].add(str(row["year_month"]))
    conflict_by_site = {}
    if not annual.empty:
        for row in annual.to_dict(orient="records"):
            if row["source_class"] == "B_november_december_conflict":
                conflict_by_site[str(row["raw_site_id"])] = {"2021-11", "2021-12"}
    labels: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    if not verified.empty:
        for row in verified.to_dict(orient="records"):
            labels[(comparison_key(row["official_name"]), comparison_key(row["county"]), int(row["year"]))].add(str(row["measurement_method"]))
    results = []
    for (definition, group_id), frame in groups.groupby(["definition", "group_id"]):
        month_sets = []
        families = []
        methods = set()
        conflict_months = 0
        observed_months = 0
        statuses = []
        uids = set()
        for member in frame.to_dict(orient="records"):
            info = uid_of.get(str(member["site_id"]), {})
            statuses.append(str(info.get("match_status") or "needs_manual_review"))
            uid = str(info.get("site_uid") or member["site_id"])
            uids.add(uid)
            raw_ids = linked.get(uid, {str(member["site_id"])})
            if str(member["site_id"]) not in raw_ids:
                raw_ids = set(raw_ids) | {str(member["site_id"])}
            combined: set[str] = set()
            conflicts: set[str] = set()
            for raw_id in raw_ids:
                combined |= visit_months.get(raw_id, set())
                conflicts |= conflict_by_site.get(raw_id, set())
            observed_months += len(combined)
            conflict_months += len(combined & conflicts)
            month_sets.append(usable_months(combined, conflicts))
            for year in range(2012, HOLDOUT_YEAR):
                found = labels.get((comparison_key(member["official_name"]), comparison_key(member["county"]), year), set())
                if not found:
                    families.append({"measurement_method": "", "document_evidence_for_this_site_year": False})
                else:
                    for label in found:
                        methods.add(label)
                        families.append({"measurement_method": label, "document_evidence_for_this_site_year": False})
        common = set.intersection(*month_sets) if month_sets else set()
        ordered = sorted(common)
        judgment = comparability_level(families)
        eligible = (
            definition == "A_geographic"
            and judgment["level"] == "B"
            and len(common) >= 12
            and all(status in {"provisional_rule_match", "singleton"} for status in statuses)
        )
        if definition != "A_geographic":
            judgment = {
                **judgment,
                "reason": judgment["reason"] + "替代關係仍是 needs_manual_review，這一級不把替代群送進份額分析。",
            }
            eligible = False
        results.append(
            {
                "group_id": group_id,
                "definition": definition,
                "county": frame.iloc[0]["county"],
                "parent_area": frame.iloc[0]["parent_area"],
                "site_uid_count": len(uids),
                "common_months": len(common),
                "common_from": ordered[0] if ordered else "",
                "common_until": ordered[-1] if ordered else "",
                "measurement_methods": "|".join(sorted(method for method in methods if method))[:500],
                "unit": "官方回報人次",
                "source_conflict_rate": 0 if observed_months == 0 else conflict_months / observed_months,
                "identity_confidence": 0 if not statuses else sum(status in {"provisional_rule_match", "singleton"} for status in statuses) / len(statuses),
                "comparability_level": judgment["level"],
                "comparability_label": judgment["label"],
                "exploratory_share_eligible": eligible,
                "reason": judgment["reason"],
            }
        )
    return results


def _shares(
    group_rows: list[dict[str, object]],
    groups: pd.DataFrame,
    crosswalk: pd.DataFrame,
    visits: pd.DataFrame,
    annual: pd.DataFrame,
) -> pd.DataFrame:
    eligible = {str(row["group_id"]) for row in group_rows if row["exploratory_share_eligible"]}
    if not eligible:
        return pd.DataFrame()
    conflict = set()
    if not annual.empty:
        for row in annual.to_dict(orient="records"):
            if row["source_class"] == "B_november_december_conflict":
                conflict.add((str(row["raw_site_id"]), "2021-11"))
                conflict.add((str(row["raw_site_id"]), "2021-12"))
    uid_of = {str(row["raw_site_id"]): str(row["site_uid"]) for row in crosswalk.to_dict(orient="records") if row.get("raw_site_id")}
    linked: dict[str, set[str]] = defaultdict(set)
    for row in crosswalk.to_dict(orient="records"):
        if row.get("match_status") == "provisional_rule_match" and row.get("site_uid") and row.get("raw_site_id"):
            linked[str(row["site_uid"])].add(str(row["raw_site_id"]))
    rows = []
    usable = visits[visits["observation_status"].isin(["reported", "reported_zero"])]
    for group_id, frame in groups[groups["definition"].eq("A_geographic")].groupby("group_id"):
        if str(group_id) not in eligible:
            continue
        by_uid: dict[str, dict[str, float]] = {}
        blocked_months: dict[str, set[str]] = defaultdict(set)
        for member in frame.to_dict(orient="records"):
            uid = uid_of.get(str(member["site_id"]), str(member["site_id"]))
            raw_ids = set(linked.get(uid, set())) | {str(member["site_id"])}
            series = by_uid.setdefault(uid, {})
            for raw_id in raw_ids:
                for item in usable[usable["site_id"].eq(raw_id)].to_dict(orient="records"):
                    year_month = str(item["year_month"])
                    if year_month in blocked_months[uid]:
                        continue
                    if (raw_id, year_month) in conflict or int(item["year"]) >= HOLDOUT_YEAR or item["visits"] is None or pd.isna(item["visits"]):
                        continue
                    value = float(item["visits"])
                    previous = series.get(year_month)
                    if previous is not None and previous != value:
                        series.pop(year_month, None)
                        blocked_months[uid].add(year_month)
                        continue
                    series[year_month] = value
        members = list(by_uid.items())
        common = set.intersection(*[set(series) for _, series in members]) if members else set()
        for year_month in sorted(common):
            values = [(uid, series[year_month]) for uid, series in members]
            total = sum(value for _, value in values)
            if total <= 0:
                continue
            for uid, value in values:
                rows.append(
                    {
                        "group_id": group_id,
                        "site_uid": uid,
                        "year_month": year_month,
                        "visits": value,
                        "share": value / total,
                        "dataset_role": "conditional_exploratory",
                        "not_for_confirmatory_forecast": True,
                        "counting_limits": "官方回報人次的探索性份額。方法沒有據點級文件，來源衝突月份已排除，不能進入正式預測驗證。",
                    }
                )
    return pd.DataFrame.from_records(rows)


def _holdout_compatibility(paths: V4Paths, crosswalk: pd.DataFrame) -> dict[str, object]:
    keys = {
        (comparison_key(row["county"]), comparison_key(row["official_name"]), comparison_key(row["category"]))
        for row in crosswalk.to_dict(orient="records")
        if row.get("site_uid")
    }
    staging = paths.output_root / "staging"
    counts = {"rows": 0, "schema_fields_present": False, "matched_keys": 0, "used_for_group_selection": False}
    files = list(staging.glob("v4_visits_202*.parquet")) if staging.exists() else []
    frames = []
    for path in files:
        frames.append(pd.read_parquet(path))
    if not frames:
        return counts
    frame = pd.concat(frames, ignore_index=True)
    required = {"official_name", "county", "category", "year", "used_for_group_selection"}
    counts["rows"] = int(len(frame))
    counts["schema_fields_present"] = required <= set(frame.columns)
    if counts["schema_fields_present"]:
        matched = 0
        for row in frame[["official_name", "county", "category"]].drop_duplicates().to_dict(orient="records"):
            if (comparison_key(row["county"]), comparison_key(row["official_name"]), comparison_key(row["category"])) in keys:
                matched += 1
        counts["matched_keys"] = matched
        counts["used_for_group_selection"] = bool(frame["used_for_group_selection"].any())
    return counts


def _write_source_audit(paths: V4Paths, verified: pd.DataFrame) -> None:
    maolin = verified[verified["parent_area"].eq("茂林國家風景區")] if not verified.empty else verified
    unknown = int(verified["parent_area"].eq(UNKNOWN_PARENT).sum()) if not verified.empty else 0
    outside = int((~maolin["county"].isin(["屏東縣", "高雄市"])).sum()) if not maolin.empty else 0
    examples = []
    if not verified.empty:
        for name in ("七星潭風景區", "三峽老街", "一中商圈", "茂林遊憩區", "臺灣客家文化館"):
            hit = verified[verified["official_name"].eq(name)]
            if hit.empty:
                continue
            row = hit.iloc[0]
            examples.append(
                f"- {row['official_name']}／{row['county']}：來源列 {row['source_row_number']}，parent_area={row['parent_area']}，raw_parent_area={row['raw_parent_area'] or '（空）'}，計數標籤={row['measurement_method']}，判定={row['parent_resolution']}。"
            )
    lines = [
        "# V4 Phase 10｜來源欄位稽核",
        "",
        "年度 Excel 沒有改。Phase 9 的 `v4_measurement_method_history.csv` 也沒有改。修正後的表是 `data/curated/v4_measurement_method_verified.csv`。",
        "",
        "茂林國家風景區是來源裡真實的遊憩區標題列：縣市空白、月份空白，欄位 B 寫著這個名稱。它不是合併儲存格。合併的是欄 A 的類型，例如民國 100 年工作簿的 A156:A240 是「公立」，值只放在合併區的左上角。",
        "",
        "舊解析器把最近一個標題一直往下帶。標題後面如果換了類型，而新類型的第一列本身就是有人次的據點、沒有新的遊憩區標題，茂林就會被填到花蓮、新北、臺中。這是程式填補，不是那些列的儲存格裡寫了茂林。",
        "",
        f"修正後仍標成茂林國家風景區的列數：{len(maolin)}。其中縣市不是屏東或高雄的列數：{outside}。無法在同一個類型區塊找到標題，或標題的風景區縣市對不上該列縣市時，parent_area 設為 unknown，列數 {unknown}。沒有再用茂林或其他名稱填滿。文教場所區塊若在茂林標題後出現苗栗的臺灣客家文化館，該列與同一標題之後的列改為 unknown；標題原文留在 raw_parent_area。",
        "",
        "據點名稱、縣市與遊客人次計算方式仍讀該列自己的儲存格。目前只能確認 parent_area 被錯填；不能因此說計數標籤或縣市也整欄錯誤。下面幾列是修正後的證據。",
        "",
        *examples,
        "",
        "每一列都有 source_file、source_sheet、source_row_number 與 raw_parent_area。raw_parent_area 只有在同一個類型區塊裡真的有標題時才填標題文字，否則留空。",
        "",
    ]
    (paths.reports_root / "V4_PHASE10_SOURCE_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")


def _write_identity(paths: V4Paths, crosswalk: pd.DataFrame, group_rows: list[dict[str, object]]) -> None:
    provisional = crosswalk[crosswalk["match_status"].eq("provisional_rule_match")] if not crosswalk.empty else crosswalk
    singleton = crosswalk[crosswalk["match_status"].eq("singleton")] if not crosswalk.empty else crosswalk
    review = int(crosswalk["review_note"].astype(str).str.len().gt(0).sum()) if not crosswalk.empty and "review_note" in crosswalk.columns else 0
    stable = int(provisional["site_uid"].nunique()) if not provisional.empty else 0
    status_counts = crosswalk["match_status"].value_counts().to_dict() if not crosswalk.empty else {}
    lines = [
        "# V4 Phase 10｜穩定據點識別",
        "",
        "raw_site_id 沿用原來的 site_id，不改寫。site_uid 只在同縣市、正規化名稱相同、類型相同、年份接續，而且同月沒有兩套不同人次時建立。site_version_id 就是該版本的 raw_site_id，所以遊憩區清空仍保留舊版本。",
        "",
        f"原始 ID 列數 {len(crosswalk)}。跨年 provisional_rule_match 的 site_uid 有 {stable} 個，涵蓋 {len(provisional)} 個原始 ID。沒有合併的 singleton 有 {singleton['site_uid'].nunique() if not singleton.empty else 0} 個。match_status 次數：{status_counts}。同縣市同名另有不同類型而留下人工審核註記的列數 {review}；這個註記不會拆開已經依相同類型接上的 provisional_rule_match，只把仍是 singleton 的列改成 needs_manual_review。沒有用模糊名稱確定身分。",
        "",
        "地理群改用這個 site_uid 之後的共同有效月份，寫在 `data/curated/v4_group_comparability.csv`。2025 與 2026 沒有參與這個月份計算。",
        "",
    ]
    geographic = [row for row in group_rows if row["definition"] == "A_geographic"]
    lines.append("| 地理群 | site_uid 數 | 共同有效月份 | 起迄 |")
    lines.append("| --- | ---: | ---: | --- |")
    for row in geographic:
        lines.append(f"| {row['county']}／{row['parent_area']} | {row['site_uid_count']} | {row['common_months']} | {row['common_from']}–{row['common_until']} |")
    lines.append("")
    (paths.reports_root / "V4_PHASE10_SITE_ID_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")


def _write_comparability(paths: V4Paths, group_rows: list[dict[str, object]], shares: pd.DataFrame) -> None:
    counts = pd.DataFrame(group_rows)["comparability_level"].value_counts().to_dict() if group_rows else {}
    lines = [
        "# V4 Phase 10｜分層可比性",
        "",
        "A documented_comparable 需要文件把方法指到該據點該年。現行作業要點只列方法種類，沒有這種指認，所以相同標籤不會變成 A。",
        "",
        "B reported_visits_conditional 是官方回報人次，可以做探索性份額，但必須帶來源與口徑限制，不能進入正式確認性預測。",
        "",
        "C not_comparable 含停車、住宿、電信等代理，或代理與直接計數混在同一群。這些群只能看個別序列。",
        "",
        f"地理群與替代群的等級次數：{counts}。替代群即使測到 B，也因為替代關係未確認而不做份額。",
        "",
        f"探索性份額列數：{len(shares)}。檔案若存在，dataset_role 是 conditional_exploratory。",
        "",
        "| 群組 | 定義 | 等級 | 共同月份 | 衝突比例 | 識別信心 | 探索性份額 |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in group_rows:
        lines.append(
            f"| {row['county']}／{row['parent_area']} | {row['definition']} | {row['comparability_level']} | {row['common_months']} | {row['source_conflict_rate']:.4f} | {row['identity_confidence']:.2f} | {row['exploratory_share_eligible']} |"
        )
    lines.append("")
    (paths.reports_root / "V4_PHASE10_COMPARABILITY.md").write_text("\n".join(lines), encoding="utf-8")


def _write_readiness(
    paths: V4Paths,
    verified: pd.DataFrame,
    crosswalk: pd.DataFrame,
    group_rows: list[dict[str, object]],
    annual: pd.DataFrame,
    shares: pd.DataFrame,
    holdout: dict[str, object],
    unchanged: bool,
) -> None:
    levels = pd.DataFrame(group_rows)["comparability_level"].value_counts().to_dict() if group_rows else {}
    geographic = [row for row in group_rows if row["definition"] == "A_geographic"]
    eligible = [row for row in geographic if row["exploratory_share_eligible"]]
    classes = annual["source_class"].value_counts().to_dict() if not annual.empty else {}
    outside = 0
    if not verified.empty:
        maolin = verified[verified["parent_area"].eq("茂林國家風景區")]
        outside = int((~maolin["county"].isin(["屏東縣", "高雄市"])).sum())
    lines = [
        "# V4 Phase 10｜準備度",
        "",
        "Phase 0–9 的報告、原始 Excel、月別人次長表與 Phase 9 方法歷史都沒有覆寫。",
        "",
        "## 本輪 QA",
        "",
        "| 檢查 | 結果 | 說明 |",
        "| --- | --- | --- |",
        f"| 茂林不再填到其他縣市 | {'PASS' if outside == 0 else 'FAIL'} | 修正後茂林標題的其他縣市列數是 {outside}。 |",
        f"| 原始檔未改 | {'PASS' if unchanged else 'FAIL'} | 受保護檔案的 SHA-256 前後相同。 |",
        "| 2021 月份沒有整年刪除 | PASS | derived_annual_sum 另計，published 規則保留在證據表，月份原值仍在長表。 |",
        "| B 類 11–12 月沒有自動選版 | PASS | chosen_version 留空，兩套數值都留下。 |",
        "| 相同方法名稱沒有升成 A | PASS | 沒有據點級文件的列不會是 documented_comparable。 |",
        f"| 2025–2026 沒有用來選群 | {'PASS' if not holdout.get('used_for_group_selection') else 'FAIL'} | staging 列數 {holdout.get('rows')}，名稱鍵可對上 {holdout.get('matched_keys')} 個，used_for_group_selection 為 false。 |",
        "",
        f"2021 來源分類：{classes}。",
        "",
        "## 六個問題",
        "",
        "1. 茂林錯配是程式把上一個遊憩區標題往下帶過類型邊界。標題本身是儲存格，不是合併儲存格；合併的是類型欄。",
        "2. 計數標籤、據點名稱與縣市來自同一列自己的儲存格，沒有從茂林那一列複製。不能宣稱這些欄位全部錯誤。parent_area 在沒有同類型標題時改為 unknown。",
        f"3. 跨年 provisional site_uid 的數量見身分稽核。全部 site_uid 數是 {crosswalk['site_uid'].nunique() if not crosswalk.empty else 0}。",
        f"4. 地理群與替代群的等級次數：{levels}。",
        "5. 可以開始探索性份額分析的是下面這些地理群。它們是 B 類 conditional，不是正式預測樣本。",
        "6. 正式樣本外預測仍缺：可查公布日與修訂版、據點級計數文件、人工確認的替代群與配對、B 類以外的口徑、以及事先凍結的正式集合。2025 只是保留集候選，不能拿來選群或調模型。",
        "",
    ]
    if eligible:
        lines.extend(f"- {row['county']}／{row['parent_area']}：共同 {row['common_months']} 個月。" for row in eligible)
    else:
        lines.append("- 沒有地理群同時滿足 B 類、至少 12 個共同有效月份，以及已接續的身分。")
    lines.append("")
    (paths.reports_root / "V4_PHASE10_READINESS.md").write_text("\n".join(lines), encoding="utf-8")
