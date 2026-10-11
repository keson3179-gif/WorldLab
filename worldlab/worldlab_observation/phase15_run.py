"""Build the real-place atlas from local sources. Holdout visits are not read."""

from __future__ import annotations

import json
import platform
import sys

import pandas as pd

from worldlab_data.config import WORLD_ROOT, V4Paths, load_paths
from worldlab_data.tdx import dry_run_plan
from worldlab_data.textutil import file_sha256
from worldlab_observation.adapter import PHASE14_PARENTS, fair_one_step_status
from worldlab_observation.place_atlas import (
    NAME_STEMS,
    STATUS_MISSING,
    STATUS_SOURCE,
    TEMPORAL_CURRENT,
    TEMPORAL_HISTORICAL,
    TEMPORAL_STABLE_UNCONFIRMED,
    accessibility_row,
    identifiability_demo,
    iter_attractions,
    m0_prediction_status,
    m1_prediction_status,
    match_grade,
    snapshot_rows,
    statistical_category_row,
)

GUARDED = (
    "world/choice.py",
    "world/geography.py",
    "world/experiment_004.py",
    "world/experiment_005.py",
    "worldlab_observation/adapter.py",
    "data/curated/v4_forecast_panel_candidate.parquet",
    "data/curated/v4_2024_identity_audit.csv",
    "data/curated/v4_canonical_site_crosswalk.csv",
    "reports/V4_PHASE14_ADAPTER_ARCHITECTURE.md",
)
READ_PATHS = (
    "data/curated/v4_forecast_panel_candidate.parquet",
    "data/curated/v4_2024_identity_audit.csv",
    "data/curated/v4_group_comparability.csv",
    "data/curated/v4_canonical_site_crosswalk.csv",
    "觀光資料標準V2.1/AttractionList.xml",
)
PANEL_COLUMNS = ("group_id", "site_uid", "year_month")


def run_phase15(paths: V4Paths | None = None) -> dict[str, object]:
    paths = paths or load_paths()
    before = _hashes()
    sites = _sites()
    attraction_path = paths.raw_root / "觀光資料標準V2.1" / "AttractionList.xml"
    document_update, attractions = iter_attractions(attraction_path)
    tdx = dry_run_plan()
    if tdx["credentials_present"]:
        tdx_status = "credentials_present_but_current_feed_not_used_as_history"
    else:
        tdx_status = "offline_credentials_absent"
    provenance, atlas_rows, access_rows, summaries = _assemble(sites, attractions, document_update, tdx_status)
    curated = WORLD_ROOT / "data" / "curated"
    pd.DataFrame.from_records(atlas_rows).to_csv(curated / "v4_real_place_atlas.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame.from_records(provenance).to_csv(curated / "v4_place_attribute_provenance.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame.from_records(access_rows).to_csv(curated / "v4_accessibility_candidates.csv", index=False, encoding="utf-8-sig")
    payload = {
        "analysis_role": "exploratory_development",
        "p4_executed": False,
        "llm_calls": 0,
        "holdout_visits_used": False,
        "document_update": document_update,
        "tdx_status": tdx_status,
        "tdx_will_call_network": bool(tdx["will_call_network"]),
        "site_count": len(atlas_rows),
        "exact_matches": sum(1 for row in atlas_rows if row["coordinate_match_grade"] == "exact_name_county"),
        "groups": summaries,
        "identifiability": identifiability_demo(),
        "read_paths": list(READ_PATHS),
        "attraction_sha256": file_sha256(attraction_path),
    }
    _write_reports(paths, payload, atlas_rows, access_rows)
    if _hashes() != before:
        raise RuntimeError("protected files changed while the place atlas was written")
    (paths.output_root / "adapter_smoke").mkdir(parents=True, exist_ok=True)
    (paths.output_root / "adapter_smoke" / "phase15_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return {
        "site_count": payload["site_count"],
        "exact_matches": payload["exact_matches"],
        "tdx_status": tdx_status,
        "m0": [group["m0_status"] for group in summaries],
        "holdout_visits_used": False,
        "protected_files_unchanged": True,
    }


def _sites() -> list[dict[str, str]]:
    curated = WORLD_ROOT / "data" / "curated"
    comparability = pd.read_csv(curated / "v4_group_comparability.csv", dtype=str, keep_default_na=False)
    panel = pd.read_parquet(curated / "v4_forecast_panel_candidate.parquet", columns=list(PANEL_COLUMNS))
    crosswalk = pd.read_csv(curated / "v4_canonical_site_crosswalk.csv", dtype=str, keep_default_na=False)
    audit = pd.read_csv(curated / "v4_2024_identity_audit.csv", dtype=str, keep_default_na=False)
    if panel["year_month"].astype(str).str.slice(0, 4).astype(int).ge(2025).any():
        raise RuntimeError("a holdout year entered the place atlas")
    chosen = comparability[comparability["definition"].eq("A_geographic") & comparability["parent_area"].isin(PHASE14_PARENTS)]
    sites: list[dict[str, str]] = []
    for group in chosen.to_dict(orient="records"):
        membership = sorted(set(panel.loc[panel["group_id"].eq(group["group_id"]), "site_uid"]))
        for site_uid in membership:
            linked = crosswalk[crosswalk["site_uid"].eq(site_uid) & crosswalk["match_status"].isin(["provisional_rule_match", "singleton"])]
            if linked.empty:
                raise RuntimeError(f"missing statistical chain for {site_uid}")
            latest = linked.sort_values("last_year").iloc[-1]
            categories = sorted(set(linked["category"]))
            category = categories[0] if len(categories) == 1 else ""
            identity = audit[audit["canonical_site_uid"].eq(site_uid)]
            safe = [str(value).lower() == "true" for value in identity["safe_to_link"]]
            sites.append(
                {
                    "group_id": group["group_id"],
                    "parent_area": group["parent_area"],
                    "county": str(latest["county"]),
                    "site_uid": site_uid,
                    "official_name": str(latest["official_name"]),
                    "category": category,
                    "first_year": str(int(linked["first_year"].astype(int).min())),
                    "last_year": str(int(linked["last_year"].astype(int).max())),
                    "safe_to_link": str(bool(safe) and all(safe)),
                    "identity_status": "|".join(sorted(set(identity["identity_status"]))) if not identity.empty else "",
                }
            )
    sites.sort(key=lambda row: (PHASE14_PARENTS.index(row["parent_area"]), row["official_name"]))
    if len(sites) != 9 or {row["official_name"] for row in sites} != set(NAME_STEMS):
        raise RuntimeError("the nine development sites were not recovered from the frozen groups")
    return sites


def _assemble(
    sites: list[dict[str, str]],
    attractions: list[dict[str, str]],
    document_update: str,
    tdx_status: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], list[dict[str, object]]]:
    provenance: list[dict[str, str]] = []
    atlas_rows: list[dict[str, str]] = []
    access_rows: list[dict[str, str]] = []
    by_group: dict[str, list[dict[str, str]]] = {}
    for site in sites:
        exact = [
            row
            for row in attractions
            if match_grade(site["official_name"], site["county"], row["name"], row["county"]) == "exact_name_county"
        ]
        fuzzy = [
            row
            for row in attractions
            if match_grade(site["official_name"], site["county"], row["name"], row["county"]) == "fuzzy_candidate"
        ]
        if len(exact) > 1:
            raise RuntimeError(f"more than one exact attraction row for {site['official_name']}")
        attraction = exact[0] if exact else None
        grade = "exact_name_county" if attraction else ""
        category = statistical_category_row(site, site["category"], site["first_year"], site["last_year"])
        fields = [category, *snapshot_rows(site, attraction, grade)]
        provenance.extend(fields)
        by_group.setdefault(site["group_id"], []).extend(fields)
        atlas_rows.append(_atlas_row(site, fields, attraction, document_update, grade))
        if attraction:
            access_rows.append(
                accessibility_row(
                    site,
                    attraction["name"],
                    attraction["county"],
                    "exact_name_county",
                    attraction["latitude"],
                    attraction["longitude"],
                    "",
                    "straight_line_unavailable_until_two_sites_in_the_group_match",
                    tdx_status,
                    "直線距離尚未計算。同群沒有第二個同名同縣市座標。這不是旅行時間。",
                )
            )
        else:
            access_rows.append(
                accessibility_row(
                    site,
                    "",
                    site["county"],
                    "",
                    "",
                    "",
                    "",
                    "missing",
                    tdx_status,
                    "沒有同名同縣市的景點資料庫列，不估算距離。",
                )
            )
        for candidate in fuzzy:
            access_rows.append(
                accessibility_row(
                    site,
                    candidate["name"],
                    candidate["county"],
                    "fuzzy_candidate",
                    candidate["latitude"],
                    candidate["longitude"],
                    "",
                    "not_computed_for_fuzzy_candidate",
                    tdx_status,
                    "名稱不同，座標只留在候選欄，沒有寫入場所卡。",
                )
            )
    summaries = []
    for parent in PHASE14_PARENTS:
        group_sites = [row for row in sites if row["parent_area"] == parent]
        group_id = group_sites[0]["group_id"]
        records = by_group[group_id]
        m0 = m0_prediction_status(records)
        m1 = m1_prediction_status(m0)
        safe = [site["safe_to_link"] == "True" for site in group_sites]
        exact_names = [
            row["official_name"]
            for row in atlas_rows
            if row["group_id"] == group_id and row["coordinate_match_grade"] == "exact_name_county"
        ]
        summaries.append(
            {
                "parent_area": parent,
                "county": group_sites[0]["county"],
                "site_count": len(group_sites),
                "exact_names": exact_names,
                "m0_status": m0["prediction_status"],
                "m0_reason": m0["skip_reason"],
                "m1_status": m1["prediction_status"],
                "m1_reason": m1["skip_reason"],
                "fair_2025_one_step": fair_one_step_status(safe),
                "historical_fields": sum(1 for row in records if row["temporal_class"] == TEMPORAL_HISTORICAL and row["status"] == STATUS_SOURCE),
                "stable_unconfirmed_fields": sum(1 for row in records if row["temporal_class"] == TEMPORAL_STABLE_UNCONFIRMED),
                "current_fields": sum(1 for row in records if row["temporal_class"] == TEMPORAL_CURRENT),
            }
        )
    return provenance, atlas_rows, access_rows, summaries


def _atlas_row(site: dict[str, str], fields: list[dict[str, str]], attraction: dict[str, str] | None, document_update: str, grade: str) -> dict[str, str]:
    by_field = {row["field"]: row for row in fields}
    def value(field: str) -> str:
        row = by_field[field]
        return row["value"] if row["status"] != STATUS_MISSING else ""
    def status(field: str) -> str:
        return by_field[field]["status"]
    return {
        "group_id": site["group_id"],
        "parent_area": site["parent_area"],
        "site_uid": site["site_uid"],
        "official_name": site["official_name"],
        "county": site["county"],
        "latitude": value("latitude"),
        "latitude_status": status("latitude"),
        "longitude": value("longitude"),
        "longitude_status": status("longitude"),
        "official_category": value("official_category"),
        "official_category_status": status("official_category"),
        "official_category_valid_from": by_field["official_category"]["valid_from"],
        "official_category_valid_to": by_field["official_category"]["valid_to"],
        "activity_tags": value("activity_tags"),
        "activity_tags_status": status("activity_tags"),
        "attraction_description": value("attraction_description"),
        "attraction_description_status": status("attraction_description"),
        "opening_or_access_conditions": value("opening_or_access_conditions"),
        "opening_or_access_conditions_status": status("opening_or_access_conditions"),
        "admission_or_cost_information": value("admission_or_cost_information"),
        "admission_or_cost_information_status": status("admission_or_cost_information"),
        "available_facilities": value("available_facilities"),
        "available_facilities_status": status("available_facilities"),
        "source_file": by_field["latitude"]["source_file"] or by_field["official_category"]["source_file"],
        "source_record_id": "" if attraction is None else attraction["attraction_id"],
        "source_url": "" if attraction is None else attraction["website"],
        "source_publication_date": "" if attraction is None else attraction["record_update"][:10],
        "attraction_file_update": document_update[:10],
        "observation_valid_from": by_field["official_category"]["valid_from"],
        "observation_valid_to": by_field["official_category"]["valid_to"],
        "snapshot_valid_on": "" if attraction is None else attraction["record_update"][:10],
        "verification_status": "source_reported_snapshot" if grade == "exact_name_county" else "statistical_category_only",
        "coordinate_match_grade": grade,
        "identity_status": site["identity_status"],
        "safe_to_link_2024": site["safe_to_link"],
    }


def _write_reports(paths: V4Paths, payload: dict[str, object], atlas_rows: list[dict[str, str]], access_rows: list[dict[str, str]]) -> None:
    reports = paths.reports_root
    (reports / "V4_PHASE15_PLACE_ATLAS.md").write_text(_atlas_text(payload, atlas_rows, access_rows), encoding="utf-8")
    (reports / "V4_PHASE15_ATTRIBUTE_COVERAGE.md").write_text(_coverage_text(payload, atlas_rows), encoding="utf-8")
    (reports / "V4_PHASE15_UTILITY_MODEL_SPEC.md").write_text(_utility_text(payload), encoding="utf-8")
    (reports / "V4_PHASE15_IDENTIFIABILITY.md").write_text(_identifiability_text(payload), encoding="utf-8")
    (reports / "V4_PHASE15_READINESS.md").write_text(_readiness_text(payload, access_rows), encoding="utf-8")


def _atlas_text(payload: dict[str, object], atlas_rows: list[dict[str, str]], access_rows: list[dict[str, str]]) -> str:
    lines = [
        "# V4 Phase 15｜真實場所卡",
        "",
        "這一輪是 exploratory development。沒有執行 P4，沒有呼叫 LLM，也沒有讀取 2025–2026 保留人次。九處據點來自 Phase 14 已固定的三個地理群，不是依預測誤差挑選。",
        "",
        f"景點資料庫檔案更新時間是 {payload['document_update']}。同名同縣市的欄位記為 source_reported，沒有升成 verified。模糊名稱只出現在可達性候選表。",
        "",
        "| 據點 | 縣市 | 座標配對 | 統計類別 | 2024 身分狀態 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in atlas_rows:
        lines.append(
            f"| {row['official_name']} | {row['county']} | {row['coordinate_match_grade'] or '無同名列'} | {row['official_category']} | {row['identity_status']} |"
        )
    lines.append("")
    lines.append("觀霧與雪見的 safe_to_link 只是 CSV 類型未變的規則推定，human_reviewed 仍是 false。雪見的 Excel 合併類型用字和 CSV 不同。人次相同沒有被當成口徑連續。")
    fuzzy = [row for row in access_rows if row["match_grade"] == "fuzzy_candidate"]
    lines.extend(["", "模糊候選沒有寫入場所卡座標："])
    for row in fuzzy:
        lines.append(f"- {row['official_name']} 對上「{row['candidate_name']}」（{row['candidate_county']}）。")
    lines.append("")
    return "\n".join(lines)


def _coverage_text(payload: dict[str, object], atlas_rows: list[dict[str, str]]) -> str:
    lines = [
        "# V4 Phase 15｜屬性覆蓋",
        "",
        "歷史上有統計發布佐證的，只有各 site_uid 在自己年份區間裡的官方類型。三群在這段鏈上的類型都是國家公園，所以它不能在群內區分據點。",
        "",
        "座標、描述、開放條件、票價與設施若來自景點資料庫，日期是該筆 2026 年附近的更新日。描述與開放條件只代表當期。座標標成合理穩定但尚未證實，不能進入嚴格歷史預測。",
        "",
        "| 據點 | 緯度狀態 | 描述狀態 | 開放條件狀態 | 票價狀態 | 設施狀態 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in atlas_rows:
        lines.append(
            "| {name} | {lat} | {desc} | {open_} | {fee} | {facilities} |".format(
                name=row["official_name"],
                lat=row["latitude_status"],
                desc=row["attraction_description_status"],
                open_=row["opening_or_access_conditions_status"],
                fee=row["admission_or_cost_information_status"],
                facilities=row["available_facilities_status"],
            )
        )
    lines.extend(
        [
            "",
            f"同名同縣市配對 {payload['exact_matches']} 處。其餘座標、描述、開放條件、票價與設施維持 missing。",
            "",
            "空的設施或票價標籤沒有被解讀成「沒有設施」或「免費」。免費代碼沒有對照表，因此沒有寫成票價。活動代碼 1、2、7、8、12 也沒有翻譯成設施名稱。",
            "",
        ]
    )
    return "\n".join(lines)


def _utility_text(payload: dict[str, object]) -> str:
    lines = [
        "# V4 Phase 15｜最小效用模型規格",
        "",
        "三種層次都接在 Phase 14 的 observation adapter 上。agent_choice_probability、expected_model_presence_share、reported_visit_share 保持三個不同的量。一位旅客可以在同一個月出現在多個據點，所以模型份額對到官方回報份額只是對應假設，不是個人選擇機率。",
        "",
        "M0 使用可觀察、而且屬於 historically_documented 的場所屬性。三群目前只有相同的統計類型符合這一類。沒有用常數填上視覺吸引或容量。三群的 M0 都是 prediction_unavailable。",
        "",
        "M1 要加在 M0 之上，並使用截止月以前的回報人氣摘要。記憶規則沿用可適配的 LOGMAX、RW50、EXP99 介面，來源標成 reported_visit_share_conditional，exposure_claim 為 false。M0 沒有基準效用時，M1 也是 prediction_unavailable，沒有把歷史份額改名成 ABM 預測。",
        "",
        "M2 只定義契約：中性場所代碼、凍結的場所卡版本、來源可追溯的描述、排除驗證期人氣與觀察期之後的描述、模型版本、提示雜湊、解析規則與回應保存。本輪 generation_status 是 not_run。九處裡只有同名配對的兩處有當期描述，而且那兩段文字的日期不是評估年。",
        "",
        "沒有受訪者資料。合成偏好只在合成場所上檢查：屬性完全相同時，不同偏好權重得到相同份額。這不代表臺灣旅客人口。沒有獨立容量，所以不估計擁擠成本。同一條人氣序列不能同時當成吸引與擁擠。",
        "",
        "| 地理群 | M0 | M0 原因 | M1 | M1 原因 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for group in payload["groups"]:
        lines.append(
            f"| {group['county']}／{group['parent_area']} | {group['m0_status']} | {group['m0_reason']} | {group['m1_status']} | {group['m1_reason']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _identifiability_text(payload: dict[str, object]) -> str:
    demo = payload["identifiability"]
    return "\n".join(
        [
            "# V4 Phase 15｜識別限制",
            "",
            "場所固定效用與社會權重會一起進入 softmax。合成檢查把社會分數乘 2、權重除 2，選擇機率不變。"
            f" 這次檢查成立：{demo['weight_scale_tradeoff']}。溫度與此係數也共享尺度，不能從一條份額分開識別。",
            "",
            "屬性在三個場所都相同時，兩組不同的合成偏好得到同一份額。"
            f" 這次檢查成立：{demo['identical_attributes_drop_preference']}。總體人次配出來的權重因此不是個人心理參數。",
            "",
            "歷史人氣若直接當成預測份額，它已經是 Phase 13 的持續或季節基準所使用的資訊。在基準效用仍缺的情況下，再把這條人氣叫成 M1，不會多出一份獨立訊息。Q 仍是模型內有社會分數與沒有社會分數的機率距離，不是回報份額 TV。",
            "",
            "M2 尚未生成，所以不能檢查語義效用是否帶來額外預測訊息。校準沒有使用 2025 資料。",
            "",
        ]
    )


def _readiness_text(payload: dict[str, object], access_rows: list[dict[str, str]]) -> str:
    fuzzy_count = sum(1 for row in access_rows if row["match_grade"] == "fuzzy_candidate")
    lines = [
        "# V4 Phase 15｜就緒狀態",
        "",
        f"Python {sys.version.split()[0]}，平台 {platform.platform()}。LLM 呼叫數 {payload['llm_calls']}。",
        "",
        f"TDX 狀態是 {payload['tdx_status']}。dry_run_plan 的 will_call_network 是 {payload['tdx_will_call_network']}。沒有呼叫交通 API。直線距離也沒有改稱為旅行時間。現況交通供給不能回填 2014 年。",
        "",
        f"模糊候選列數 {fuzzy_count}。同群之內沒有兩處同名同縣市座標，所以沒有場所間直線距離矩陣。",
        "",
        "2024 身分沒有因為 CSV 與 Excel 人次相同而升成口徑連續。三群的 2025 單步資格："
        ,
    ]
    for group in payload["groups"]:
        lines.append(f"- {group['county']}／{group['parent_area']}：{group['fair_2025_one_step']}。")
    lines.extend(
        [
            "",
            "備援方案是前瞻快照，不是改截止日。若要讓未來某個月能做公平單步預測，必須在該月分數被計算之前，凍結據點集合、場所卡版本、資訊情境與主指標 share_vector_TV。2024 身分未完整時，不能拿 2023-12 當作 t-1。本輪沒有凍結確認性協定，也沒有產生真實場所的探索性份額預測。",
            "",
            "下一輪還不能正式比較 ABM 與七個統計基準的真實景點分數。比較框架可以保留；M0 與 M1 在真實場所上仍是 prediction_unavailable。",
            "",
            "讀取路徑：",
            "",
        ]
    )
    for path in payload["read_paths"]:
        lines.append(f"- `{path}`")
    lines.extend(["", f"景點資料庫 SHA-256：`{payload['attraction_sha256']}`。", ""])
    return "\n".join(lines)


def _hashes() -> dict[str, str]:
    return {relative: file_sha256(WORLD_ROOT / relative) for relative in GUARDED}


if __name__ == "__main__":
    print(json.dumps(run_phase15(), ensure_ascii=False, indent=2, allow_nan=False))
