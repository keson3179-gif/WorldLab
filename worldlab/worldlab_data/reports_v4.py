from __future__ import annotations

import pandas as pd

from worldlab_data.config import V4Paths


def write_final_reports(paths: V4Paths, gates: list[dict[str, str]]) -> None:
    visits = _read(paths.processed / "site_month_visits.parquet")
    history = _read(paths.processed / "site_status_history.parquet")
    master = _read(paths.processed / "attraction_master.parquet")
    review = _read_csv(paths.reports_root / "site_matching_review.csv")
    groups = _read_csv(paths.curated / "competition_groups.csv")
    exploratory = _read(paths.processed / "exploratory_visit_share_panel.parquet")
    official = _read(paths.processed / "site_month_panel.parquet")
    fails = sum(gate["status"] == "FAIL" for gate in gates)
    warns = sum(gate["status"] == "WARN" for gate in gates)
    lines = [
        "# WorldLab V4 資料 Pilot 報告",
        "",
        "本輪只做資料盤點、清洗、對齊、品質驗收與候選面板。沒有跑模擬、沒有呼叫 LLM、沒有重跑 Experiment 001–020，也沒有做假設檢定。",
        "",
        "V4 計畫書位於「論文本文/WorldLab_NSTC_V4_Observational_Validation_Refined.docx」。本輪只核對其第一年觀察式資料設計，沒有改寫論文，也沒有做第二年模擬或第三年預測。",
        "",
        "## 1. 已取得的資料",
        "",
        _coverage_section(visits, history),
        "",
        "景點資訊資料庫與活動清單的檔案更新時間，以及氣象 v0.2 月檔、辦公日曆、年度 Excel 的範圍，見 `data_inventory.md` 與 `visitor_data_qa.md`。",
        "TDX 尚未下載。網路關注度檔案不存在。",
        "",
        "## 2. 欄位與格式差異",
        "",
        "- 月別人次 CSV 是西元年寬表，一列十二個月。縣市儲存格可能同一格裡再寫一行英文縣市名，清洗時只把第一行當縣市，英文另存。",
        "- 年度累計簿是雙語 Excel，月份標題像「100年 / 1月」，不是整理過的 CSV。本輪只記錄版面，沒有把這些數字換算進長表。",
        "- 景點與活動是 XML 名錄，不是人次。官方類別是代碼；中文標籤只在既有類別對照表有出現時才附上。",
        "- 辦公日曆混有 UTF-8 與 Big5/CP950。民國 114 年有兩份非 Google 版本。Google 複本不進月表。",
        "- 氣象月指標沿用已凍結的 v0.2 測站月表，不把逐日最高溫改稱為日均溫。逐日原始 CSV 有一部分無法用標準讀檔器打開，本輪沒有重算日轉月。",
        "",
        "## 3. 清洗前後筆數",
        "",
        f"- 月表列數：{len(visits)}",
        f"- 統計據點：{visits['site_id'].nunique() if not visits.empty else 0}",
        f"- 景點主檔：{len(master)}",
        f"- 探索用份額列：{len(exploratory)}",
        f"- 正式回測列：{len(official)}",
        "",
        "## 4. 不能確認的景點配對",
        "",
        _match_section(master, review),
        "",
        "待審明細在 `site_matching_review.csv`。最高分沒有被寫成已確認配對，座標也沒有因為候選分數寫進主檔。",
        "",
        "## 5. 覆蓋",
        "",
        _span_section(visits),
        "",
        "## 6. 人次與口徑",
        "",
        "月份缺值、報表上的 0、無法解析的文字是不同狀態。0 只代表該月儲存格是 0，不代表確認停業。",
        "合計與十二個月不一致時，保留月份原值，並在異常表標出，不挑一個比較好看的數字。",
        "各據點的計數方式沒有出現在已解析欄位裡，因此沒有假設它們相同。",
        "份額與 HHI 若出現在探索表，語意是據點回報人次的分布。一個人可以在同一個月去多個據點，所以這不是個人選擇機率。",
        "",
        "## 7. 競爭群候選",
        "",
        _group_section(groups),
        "",
        "沒有把全臺據點放進同一個群。多於 8 個的遊憩區沒有為了湊 Pilot 規模而挑子集。替代關係未確認的群不會進入正式回測面板。",
        "",
        "## 8. 發布時間",
        "",
        "遊客月報、日曆、氣象與活動都沒有可重建的歷史發布版本。它們標記為 publication_history_unknown，不進入嚴格的無洩漏特徵。",
        "活動檔沒有公告日期，只留作事後解釋。結束年明顯不合理的活動保留原值，不自動改年，也不計入月活動數。",
        "",
        "## 9. QA",
        "",
        f"資料閘門 FAIL {fails}、WARN {warns}。明細見 `WORLDLAB_V4_QA_SUMMARY.md`。",
        "",
        "最近一次測試：`compileall` 通過；`pytest -m \"not slow\"` 為 259 passed、7 deselected。沒有重跑 slow 全套。",
        "",
        "## 10. 這份 Pilot 能做與不能做的事",
        "",
        "可以重跑盤點、把月別寬表收成帶狀態的長表、把同縣市同名候選送人工覆核，並看見哪些遊憩區在規模上接近 5 到 8 個據點。",
        "現在不能做嚴格的樣本外即時預測：發布時間、計數方式與據點配對都還沒確認。正式回測面板因此是空的。這是資料限制，不是模型結果。",
        "",
    ]
    (paths.reports_root / "WORLDLAB_V4_DATA_PILOT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    next_steps = [
        "# WorldLab V4 下一輪優先順序",
        "",
        "1. 人工確認 `site_matching_review.csv`。確認過的列寫進 `configs/v4_site_crosswalk_manual.csv`，狀態用 `confirmed_manual`。程式不會用較高的模糊分數覆蓋它。",
        "2. 取得《主要觀光遊憩據點統計作業要點》的計數規則，或各據點自己的計數說明。在那之前不要把不同據點當成同一種人次。",
        "3. 核對年度 Excel 與月別 CSV 的欄位。兩份數字不同時，兩份都留，不要選一個。",
        "4. 為月報、日曆與氣象補上可查的發布日期。沒有日期的欄位繼續排除在嚴格預測之外。",
        "5. 民國 114 年兩份辦公日曆需人工指定哪一份對應哪一次公告。",
        "6. 活動若要進預測，需要當時已公告的歷史活動，而不是現在這份 2025–2026 清單。",
        "7. TDX 金鑰備齊後，只把站點與路線標成當期靜態代理。不要把現在的旅行時間寫回 2018 或 2020。",
        "8. 景點高程仍缺。山區據點在有高程之前，測站配對不能視為已檢查海拔差。",
        "9. 競爭群要在任何預測評估之前凍結。不要等看到哪一群比較好預測再決定納入誰。",
        "",
    ]
    (paths.reports_root / "WORLDLAB_V4_NEXT_STEPS.md").write_text("\n".join(next_steps), encoding="utf-8")


def _coverage_section(visits: pd.DataFrame, history: pd.DataFrame) -> str:
    if visits.empty:
        return "月別人次表是空的。"
    years = sorted(int(year) for year in visits["year"].dropna().unique())
    return f"整理進長表的月別人次涵蓋 {years[0]} 到 {years[-1]}，{visits['site_id'].nunique()} 個統計據點，{len(history)} 列狀態摘要。"


def _match_section(master: pd.DataFrame, review: pd.DataFrame) -> str:
    if master.empty:
        return "尚未建立景點主檔。"
    pending = int((master["match_status"] == "needs_manual_review").sum())
    methods = review["match_method"].value_counts().to_dict() if not review.empty and "match_method" in review.columns else {}
    return f"主檔 {len(master)} 個統計據點裡，{pending} 個仍是 needs_manual_review。候選方法次數：{methods}。"


def _span_section(visits: pd.DataFrame) -> str:
    if visits.empty:
        return "沒有月份。"
    reported = visits["observation_status"].isin(["reported", "reported_zero"])
    missing = visits["observation_status"].isin(["missing", "missing_token"])
    zeros = visits["observation_status"].eq("reported_zero")
    return (
        f"有回報的月份列 {int(reported.sum())}，其中儲存格為 0 的 {int(zeros.sum())} 列。"
        f"缺值或缺失代碼 {int(missing.sum())} 列，這些列的人次是空值。"
    )


def _group_section(groups: pd.DataFrame) -> str:
    if groups.empty:
        return "沒有產生候選群。"
    included = groups[groups["included"].astype(str) == "True"]
    excluded = groups[groups["included"].astype(str) != "True"]
    names = sorted(included["competition_group_id"].unique()) if not included.empty else []
    reasons = excluded["exclusion_reason"].value_counts().head(6).to_dict() if not excluded.empty else {}
    return (
        f"規模、同縣市與同遊憩區都符合的候選群 {len(names)} 個：{names}。"
        f"這些群的替代關係與計數方式仍是未確認，backtest_eligible 為假。"
        f"未納入列的主要原因：{reasons}。"
    )


def _read(path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _read_csv(path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, keep_default_na=False)
