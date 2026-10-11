"""Small Wikipedia pageview audit and the prospective-registry specification.

The calls stop at 2023-12. Holdout visit files are not read.
"""

from __future__ import annotations

import json
import platform
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import pandas as pd

from worldlab_data.config import WORLD_ROOT, V4Paths, load_paths
from worldlab_data.textutil import file_sha256
from worldlab_observation.external_evidence import (
    ARTICLE_LICENSE,
    PAGEVIEW_API,
    PAGEVIEW_END,
    PAGEVIEW_LICENSE,
    PAGEVIEW_START,
    SOURCE_API,
    USER_AGENT,
    classify_article,
    coverage_status,
    month_span,
    place_stem,
    select_alternate_candidates,
)
from worldlab_observation.place_atlas import NAME_STEMS

READ_PATHS = (
    "data/curated/v4_real_place_atlas.csv",
    "data/curated/v4_canonical_site_crosswalk.csv",
)
GUARDED = (
    "world/choice.py",
    "world/geography.py",
    "worldlab_observation/adapter.py",
    "worldlab_observation/intensity.py",
    "reports/V4_PHASE17_GO_NO_GO.md",
)
ATLAS_COLUMNS = (
    "site_uid",
    "official_name",
    "county",
    "parent_area",
    "latitude",
    "longitude",
    "latitude_status",
    "longitude_status",
)


def run_phase18(paths: V4Paths | None = None, pause_seconds: float = 1.0) -> dict[str, object]:
    paths = paths or load_paths()
    before = _hashes()
    sites = _sites()
    retrieved_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    matches, limited = _audit_queue(sites, retrieved_at, pause_seconds)
    usable = [row for row in matches if row["usable_series"] == "true"]
    groups_with_pair = _groups_with_pair(usable)
    alternates: list[dict[str, object]] = []
    alternate_pool = select_alternate_candidates(_crosswalk_rows(), {site["official_name"] for site in sites})
    if not limited and (len(usable) < 3 or not groups_with_pair):
        time.sleep(max(pause_seconds, 45))
        alternates, limited = _audit_queue(alternate_pool, retrieved_at, pause_seconds)
    payload = {
        "retrieved_at": retrieved_at,
        "pageview_window": f"{PAGEVIEW_START}:{PAGEVIEW_END}",
        "llm_calls": 0,
        "p4_executed": False,
        "holdout_visits_used": False,
        "phase17_decision_unchanged": "NO-GO",
        "matches": matches,
        "usable_count": len(usable),
        "groups_with_two_usable_articles": groups_with_pair,
        "alternate_pool_count": len(alternate_pool),
        "alternates": alternates,
        "alternate_usable_count": sum(1 for row in alternates if row["usable_series"] == "true"),
        "rate_limited": limited,
        "commons_osm": "NO-GO",
        "pageviews_decision": "INCOMPLETE" if limited else ("NO-GO" if len(usable) < 3 or not groups_with_pair else "LIMITED_GO"),
        "read_paths": list(READ_PATHS),
    }
    out = paths.output_root / "phase18_external"
    out.mkdir(parents=True, exist_ok=True)
    (out / "pageviews_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame.from_records([_flat(row) for row in matches + alternates]).to_csv(
        out / "page_matches.csv", index=False, encoding="utf-8-sig"
    )
    _write_reports(paths, payload, alternate_pool)
    if _hashes() != before:
        raise RuntimeError("protected files changed while the external audit was written")
    return {
        "pageviews_decision": payload["pageviews_decision"],
        "usable_count": payload["usable_count"],
        "alternate_usable_count": payload["alternate_usable_count"],
        "commons_osm": "NO-GO",
        "holdout_visits_used": False,
        "protected_files_unchanged": True,
    }


def _sites() -> list[dict[str, str]]:
    frame = pd.read_csv(WORLD_ROOT / "data" / "curated" / "v4_real_place_atlas.csv", dtype=str, keep_default_na=False, usecols=list(ATLAS_COLUMNS))
    sites = frame.to_dict(orient="records")
    if {row["official_name"] for row in sites} != set(NAME_STEMS) or len(sites) != 9:
        raise RuntimeError("the nine development sites were not recovered")
    return sites


def _crosswalk_rows() -> list[dict[str, str]]:
    frame = pd.read_csv(
        WORLD_ROOT / "data" / "curated" / "v4_canonical_site_crosswalk.csv",
        dtype=str,
        keep_default_na=False,
        usecols=["site_uid", "official_name", "county", "parent_area", "match_status", "first_year", "last_year"],
    )
    return frame.to_dict(orient="records")


def _audit_queue(sites: list[dict[str, str]], retrieved_at: str, pause_seconds: float) -> tuple[list[dict[str, object]], bool]:
    rows: list[dict[str, object]] = []
    limited = False
    for index, site in enumerate(sites):
        row = _audit_site(site, retrieved_at, pause_seconds)
        if str(row.get("pageview_error", "")).startswith("HTTP429"):
            time.sleep(_retry_seconds(str(row.get("pageview_error", ""))))
            row = _audit_site(site, retrieved_at, pause_seconds)
        rows.append(row)
        if str(row.get("pageview_error", "")).startswith("HTTP429"):
            limited = True
            rows.extend(_not_attempted(later, retrieved_at) for later in sites[index + 1 :])
            break
    return rows, limited


def _audit_site(site: dict[str, str], retrieved_at: str, pause_seconds: float) -> dict[str, object]:
    official_name = site["official_name"]
    exact = _title_query(official_name)
    if exact and str(exact.get("error", "")).startswith("HTTP429"):
        return _empty(site, retrieved_at, "lookup_failed", str(exact["error"]))
    time.sleep(pause_seconds)
    used = exact
    redirected = bool(exact and exact.get("redirected_from") == official_name)
    stem = place_stem(official_name)
    if (exact is None or exact.get("missing")) and stem != official_name:
        used = _title_query(stem)
        time.sleep(pause_seconds)
        redirected = False
    if used is None or used.get("missing"):
        error = str((used or {}).get("error", ""))
        evidence = "lookup_failed" if error else "no_unique_article"
        return _empty(site, retrieved_at, evidence, error)
    classification = classify_article(
        official_name,
        str(used.get("title", "")),
        site.get("county", ""),
        site.get("parent_area", ""),
        str(used.get("extract", "")),
        bool(used.get("disambiguation")),
        _float_or_none(site.get("latitude", "")),
        _float_or_none(site.get("longitude", "")),
        _float_or_none(used.get("latitude", "")),
        _float_or_none(used.get("longitude", "")),
        redirected,
    )
    pageviews = {"months": [], "error": ""}
    if classification["accepted"] == "true":
        pageviews = _pageviews(str(used["title"]))
        time.sleep(pause_seconds)
    coverage = coverage_status(pageviews["months"])
    return {
        "site_uid": site["site_uid"],
        "official_name": official_name,
        "county": site.get("county", ""),
        "parent_area": site.get("parent_area", ""),
        "page_id": str(used.get("pageid", "")),
        "article_title": str(used.get("title", "")),
        "language": "zh",
        "wikibase_item": str(used.get("wikibase_item", "")),
        "match_evidence": classification["match_evidence"],
        "accepted": classification["accepted"],
        "earliest_available_month": coverage["earliest_available_month"] if classification["accepted"] == "true" else "",
        "monthly_coverage": coverage["monthly_coverage"] if classification["accepted"] == "true" else 0.0,
        "observed_months": coverage["observed_months"] if classification["accepted"] == "true" else 0,
        "missing_months": coverage["missing_months"] if classification["accepted"] == "true" else month_span(),
        "missing_month_count": len(coverage["missing_months"]) if classification["accepted"] == "true" else coverage["expected_months"],
        "usable_series": "true" if classification["accepted"] == "true" and coverage["usable_series"] else "false",
        "retrieval_timestamp": retrieved_at,
        "source_api": SOURCE_API,
        "pageview_api": PAGEVIEW_API,
        "license": f"{ARTICLE_LICENSE}; {PAGEVIEW_LICENSE}" if classification["accepted"] == "true" else ARTICLE_LICENSE,
        "pageview_error": pageviews["error"],
        "role": site.get("selection_rule", "development_nine"),
    }


def _title_query(title: str) -> dict[str, object] | None:
    params = {
        "action": "query",
        "format": "json",
        "utf8": "1",
        "redirects": "1",
        "prop": "info|pageprops|coordinates|extracts",
        "exintro": "1",
        "explaintext": "1",
        "exchars": "240",
        "titles": title,
    }
    payload, error = _get_json(f"{SOURCE_API}?{urllib.parse.urlencode(params)}")
    if error or not payload:
        return {"missing": True, "error": error}
    pages = payload.get("query", {}).get("pages", {})
    if not pages:
        return {"missing": True, "error": "no_pages"}
    page = next(iter(pages.values()))
    redirects = payload.get("query", {}).get("redirects", [])
    coords = page.get("coordinates") or [{}]
    return {
        "missing": "missing" in page,
        "title": page.get("title", ""),
        "pageid": page.get("pageid", ""),
        "extract": page.get("extract", ""),
        "wikibase_item": (page.get("pageprops") or {}).get("wikibase_item", ""),
        "disambiguation": "disambiguation" in (page.get("pageprops") or {}),
        "latitude": coords[0].get("lat", "") if coords else "",
        "longitude": coords[0].get("lon", "") if coords else "",
        "redirected_from": redirects[0].get("from", "") if redirects else "",
        "error": error,
    }


def _pageviews(title: str) -> dict[str, object]:
    encoded = urllib.parse.quote(title.replace(" ", "_"), safe="")
    url = f"{PAGEVIEW_API}/zh.wikipedia/all-access/user/{encoded}/monthly/2015070100/2023123100"
    payload, error = _get_json(url)
    if error or not payload:
        return {"months": [], "error": error or "empty_response"}
    months = []
    for item in payload.get("items", []):
        stamp = str(item.get("timestamp", ""))
        if len(stamp) >= 6:
            months.append(f"{stamp[:4]}-{stamp[4:6]}")
    return {"months": months, "error": ""}


def _get_json(url: str) -> tuple[dict | None, str]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8")), ""
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            retry_after = exc.headers.get("Retry-After", "") if exc.headers else ""
            return None, f"HTTP429{':' + retry_after if retry_after else ''}"
        return None, f"HTTP{exc.code}"
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return None, exc.__class__.__name__


def _empty(site: dict[str, str], retrieved_at: str, evidence: str, error: str) -> dict[str, object]:
    return {
        "site_uid": site["site_uid"],
        "official_name": site["official_name"],
        "county": site.get("county", ""),
        "parent_area": site.get("parent_area", ""),
        "page_id": "",
        "article_title": "",
        "language": "zh",
        "wikibase_item": "",
        "match_evidence": evidence,
        "accepted": "false",
        "earliest_available_month": "",
        "monthly_coverage": 0.0,
        "observed_months": 0,
        "missing_months": month_span(),
        "missing_month_count": len(month_span()),
        "usable_series": "false",
        "retrieval_timestamp": retrieved_at,
        "source_api": SOURCE_API,
        "pageview_api": PAGEVIEW_API,
        "license": ARTICLE_LICENSE,
        "pageview_error": error,
        "role": site.get("selection_rule", "development_nine"),
    }


def _retry_seconds(error: str) -> int:
    suffix = error.split(":", 1)[1] if ":" in error else ""
    try:
        return min(int(suffix), 60)
    except ValueError:
        return 30


def _not_attempted(site: dict[str, str], retrieved_at: str) -> dict[str, object]:
    row = _empty(site, retrieved_at, "not_attempted_after_rate_limit", "HTTP429")
    return row


def _groups_with_pair(usable: list[dict[str, object]]) -> list[str]:
    counts: dict[str, int] = {}
    for row in usable:
        parent = str(row.get("parent_area", ""))
        counts[parent] = counts.get(parent, 0) + 1
    return sorted(parent for parent, count in counts.items() if count >= 2 and parent)


def _flat(row: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in row.items() if key != "missing_months"}


def _float_or_none(value: object) -> float | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _write_reports(paths: V4Paths, payload: dict[str, object], alternate_pool: list[dict[str, str]]) -> None:
    reports = paths.reports_root
    (reports / "V4_PHASE18_EXTERNAL_DATA_FEASIBILITY.md").write_text(_feasibility_text(payload, alternate_pool), encoding="utf-8")
    (reports / "V4_PHASE18_PROSPECTIVE_PROTOCOL.md").write_text(_protocol_text(), encoding="utf-8")
    (reports / "V4_PHASE18_P4_REDESIGN_OPTIONS.md").write_text(_redesign_text(payload), encoding="utf-8")
    (reports / "V4_PHASE18_GO_NO_GO.md").write_text(_decision_text(payload), encoding="utf-8")


def _feasibility_text(payload: dict[str, object], alternate_pool: list[dict[str, str]]) -> str:
    lines = [
        "# V4 Phase 18｜外部資料可行性",
        "",
        f"查核時間 {payload['retrieved_at']}。瀏覽量窗口固定為 {payload['pageview_window']}，沒有讀取 2025–2026 的人次、份額或 HHI。Phase 17 的 NO-GO 沒有改寫。",
        "",
        "中文維基條目必須指向該據點。國家公園總頁不當作園內遊客中心或沿線景觀。瀏覽量是網路關注代理，不是旅客實際曝光，也不是遊客人次。",
        "",
        "| 據點 | 條目 | 證據 | 最早月份 | 覆蓋率 | 可用 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["matches"]:
        lines.append(
            f"| {row['official_name']} | {row['article_title'] or '—'} | {row['match_evidence']} | {row['earliest_available_month'] or '—'} | {float(row['monthly_coverage']):.3f} | {row['usable_series']} |"
        )
    lines.extend(
        [
            "",
            f"原九處裡，覆蓋率達到 0.80 且至少 36 個月的條目有 {payload['usable_count']}。名稱對上、但覆蓋不足的不計入。同一地理群有兩條可用序列的是：{'、'.join(payload['groups_with_two_usable_articles']) or '無'}。玉山仍缺第三處條目，所以原本的 J=3 沒有補齊。同名重新導向只表示百科使用該標題，條目範圍可能廣於統計據點。",
            "",
            "Wikimedia Commons 地理影像與 OSM 歷史版本沒有本地、帶月份的授權底本。照片上傳量不是遊客人次，OSM 編輯日期也不是設施開放日。這兩項是 NO-GO，本輪不建立擷取工程。",
            "",
            "若原九處不能形成對照，候選清單只按 2015 年以前已有統計、且連續到 2023 年、名稱不是遊客中心或管理處、每縣一處來選。沒有使用預測誤差。",
            "",
        ]
    )
    if payload["alternates"]:
        lines.append("| 候選 | 縣市 | 條目 | 證據 | 覆蓋率 | 可用 |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for row in payload["alternates"]:
            lines.append(
                f"| {row['official_name']} | {row['county']} | {row['article_title'] or '—'} | {row['match_evidence']} | {float(row['monthly_coverage']):.3f} | {row['usable_series']} |"
            )
    else:
        lines.append("原九處已足夠，沒有啟動候選查核。預先篩出的候選數是 " + str(len(alternate_pool)) + "。")
    lines.append("")
    lines.append(
        "配對欄位寫在 `data/v4_observational/phase18_external/page_matches.csv` 與 `pageviews_audit.json`：site_uid、page_id、article_title、language、match_evidence、earliest_available_month、monthly_coverage、missing_months、retrieval_timestamp、source_api、license。缺月清單只放在 JSON。"
    )
    lines.append("")
    lines.append(
        f"跨縣且覆蓋達標的條目有 {payload['alternate_usable_count']}。它們分屬不同縣市，不是同一個選擇集合，也不能把原三個國家公園群補成 J=3。"
    )
    lines.append("")
    return "\n".join(lines)


def _protocol_text() -> str:
    return "\n".join(
        [
            "# V4 Phase 18｜前瞻快照規格",
            "",
            "這份規格用合成內容測試，沒有對現有 2025–2026 保留檔取用。原始快照寫入後不可覆寫。資料日後修訂時另開 snapshot_id，第一版的 content_sha256 保持不變。",
            "",
            "每次官方資料更新要留下 snapshot_id、retrieved_at、source_url、http_metadata、content_sha256、data_version、latest_observation_month、available_for_forecast_at、forecast_origin、model_version、forecast_manifest_hash、prediction_file_hash。",
            "",
            "預測的 frozen_at 必須早於目標資料的 available_for_forecast_at。觀察結果寫入之後，同一版預測不能再改 prediction_file_hash。多個事前凍結的模型可以並存。API 失敗、資料延遲與缺報各寫一筆不可覆寫的失敗紀錄。",
            "",
            "實作在 `worldlab_observation/external_evidence.py` 的 ProspectiveRegistry。它現在只證明這些不變量，還沒有接上真實的觀光署更新。",
            "",
        ]
    )


def _redesign_text(payload: dict[str, object]) -> str:
    return "\n".join(
        [
            "# V4 Phase 18｜P4 兩種設計",
            "",
            f"目前頁面瀏覽的判定是 {payload['pageviews_decision']}。這不是模型有效的證據，Phase 17 的 NO-GO 仍然有效。",
            "",
            "## 方案 A：獨立歷史關注的觀察式預測比較",
            "",
            "能回答的問題是：在官方人次之外、而且於截止月前可取得的關注序列，是否降低群內份額 TV。不能識別旅客是否真的看到該頁，也不能把係數解釋成社會學習的因果效果。",
            "",
            "需要至少兩個據點各自有唯一條目、2015-07 至 2023-12 的覆蓋率達到事前門檻，以及可重建的頁面發布時間。可以不發問卷。模型仍是既有統計基準，外加一個事前固定的關注特徵；本輪沒有擬合。最小驗收是配對、覆蓋與截止日三項同時成立。失敗時仍留下配對稽核與 NO-GO 紀錄。",
            "",
            "## 方案 B：外部動態一致性與未來快照",
            "",
            "能回答的問題是：模擬份額的形狀與官方回報分布在哪些限制下可以並排描述，以及未來一份凍結預測是否早於下一次官方發布。不能識別偏好、曝光或 E/O 來源替換。",
            "",
            "需要前瞻 registry 的不可覆寫快照，不需要問卷，也不需要把瀏覽量補進舊月份。主要比較仍是份額 TV 對上已凍結的統計基準。最小驗收是快照、預測與事後觀察三個時間戳的順序正確。失敗時仍留下協定與合成測試。",
            "",
            "原三群沒有三個達標條目。跨縣的完整瀏覽序列只說明某些統計名稱能對上百科，不是園內選擇集合。P4 因此不維持「用現有九處做 M0+X」的預測命題，改以方案 B 的前瞻測試為可執行設計。缺資料不是模型有效。",
            "",
        ]
    )


def _decision_text(payload: dict[str, object]) -> str:
    return "\n".join(
        [
            "# V4 Phase 18｜決策",
            "",
            f"Python {sys.version.split()[0]}，平台 {platform.platform()}。原九處的頁面瀏覽判定是 {payload['pageviews_decision']}，可用條目 {payload['usable_count']}。跨縣候選另有 {payload['alternate_usable_count']} 條達標序列，但不是同一選擇集合。速率限制中止：{payload['rate_limited']}。Commons 與 OSM {payload['commons_osm']}。前瞻協定已寫成規格，尚未用於保留集。",
            "",
            "Phase 17 對玉山、太魯閣、雪霸的 M0+X 仍是 NO-GO。本輪沒有新增預測模型，沒有呼叫 LLM，也沒有改寫國科會 V4 文件。",
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
    print(json.dumps(run_phase18(), ensure_ascii=False, indent=2))
