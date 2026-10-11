"""Write Phase 14 architecture reports and a small synthetic smoke record.

The runner reads pre-2024 share columns for a wiring check. It does not read holdout visits.
"""

from __future__ import annotations

import json
import platform
import sys

import pandas as pd

from worldlab_data.config import WORLD_ROOT, V4Paths, load_paths
from worldlab_data.phase13 import evaluable_months, predict_shares
from worldlab_data.textutil import file_sha256
from worldlab_observation.adapter import (
    AGENT_COUNT,
    ANALYSIS_ROLE,
    PHASE14_PARENTS,
    REUSABLE_INTERFACES,
    STEP_TO_MONTH,
    attribute_gaps,
    conditional_one_step,
    engine_constants,
    expected_share,
    fair_one_step_status,
    free_running_step,
    llm_utility_schema,
    monte_carlo_mean,
    refuse_synthetic_alias,
    replicates_for_share_se,
    scale_tradeoff_same_probability,
    softmax_probabilities,
    stay_adjusted_probability,
    undefined_share,
    EndogenousHistory,
    ReportedHistory,
)

GUARDED = (
    "world/choice.py",
    "world/geography.py",
    "world/experiment_002.py",
    "world/experiment_004.py",
    "world/experiment_005.py",
    "world/experiment_018.py",
    "world/experiment_020.py",
    "reports/V4_PHASE13_BASELINE_RESULTS.md",
    "data/curated/v4_forecast_panel_candidate.parquet",
    "data/curated/v4_2024_identity_audit.csv",
)
READ_PATHS = (
    "data/curated/v4_forecast_panel_candidate.parquet",
    "data/curated/v4_2024_identity_audit.csv",
    "data/curated/v4_group_comparability.csv",
)
PANEL_COLUMNS = ("group_id", "site_uid", "year_month", "share", "dataset_role")
SMOKE_SEED = 14014
SMOKE_PROBABILITY = (0.7, 0.2, 0.1)


def run_phase14(paths: V4Paths | None = None) -> dict[str, object]:
    paths = paths or load_paths()
    before = _hashes()
    _refuse_forbidden_reads()
    groups = _groups()
    smoke = _synthetic_smoke()
    payload = {
        "analysis_role": ANALYSIS_ROLE,
        "p4_executed": False,
        "llm_calls": 0,
        "holdout_visits_used": False,
        "model_selected_on_scores": False,
        "mapping": STEP_TO_MONTH,
        "read_paths": list(READ_PATHS),
        "panel_columns": list(PANEL_COLUMNS),
        "reusable_interfaces": list(REUSABLE_INTERFACES),
        "engine": engine_constants(),
        "required_replicates": replicates_for_share_se(),
        "smoke": smoke,
        "groups": groups,
        "llm_schema": llm_utility_schema(),
        "protected_files_unchanged": True,
    }
    _write(paths, payload)
    if _hashes() != before:
        raise RuntimeError("protected WorldLab or Phase 13 files changed")
    destination = paths.output_root / "adapter_smoke"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "phase14_smoke.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return {
        "analysis_role": ANALYSIS_ROLE,
        "required_replicates": payload["required_replicates"],
        "one_world_tv": smoke["one_world_tv"],
        "mean_max_abs_error": smoke["mean_max_abs_error"],
        "groups": [
            {
                "parent_area": row["parent_area"],
                "evaluable_months": row["evaluable_months"],
                "utility_status": row["utility_status"],
                "fair_2025_one_step": row["fair_2025_one_step"],
            }
            for row in groups
        ],
        "holdout_visits_used": False,
        "protected_files_unchanged": True,
    }


def _groups() -> list[dict[str, object]]:
    curated = WORLD_ROOT / "data" / "curated"
    comparability = pd.read_csv(curated / "v4_group_comparability.csv", dtype=str, keep_default_na=False)
    panel = pd.read_parquet(curated / "v4_forecast_panel_candidate.parquet", columns=list(PANEL_COLUMNS))
    audit = pd.read_csv(curated / "v4_2024_identity_audit.csv", dtype=str, keep_default_na=False)
    if panel["year_month"].astype(str).str.slice(0, 4).astype(int).ge(2025).any():
        raise RuntimeError("a holdout year entered the development panel")
    chosen = comparability[
        comparability["definition"].eq("A_geographic") & comparability["parent_area"].isin(PHASE14_PARENTS)
    ].to_dict(orient="records")
    chosen.sort(key=lambda row: PHASE14_PARENTS.index(row["parent_area"]))
    if [row["parent_area"] for row in chosen] != list(PHASE14_PARENTS):
        raise RuntimeError("the three development groups were not found")
    gaps = attribute_gaps(set(panel.columns))
    rows = []
    for group in chosen:
        scoped = panel[panel["group_id"].eq(group["group_id"])]
        membership = tuple(sorted(scoped["site_uid"].unique()))
        history: dict[str, dict[str, float]] = {}
        for year_month, chunk in scoped.groupby("year_month"):
            shares = {str(item.site_uid): float(item.share) for item in chunk.itertuples(index=False)}
            if set(shares) == set(membership):
                history[str(year_month)] = shares
        targets = evaluable_months(list(history))
        target = targets[0]
        reported = ReportedHistory(membership)
        for month in history:
            if month < target:
                reported.add_observed(month, [history[month][uid] for uid in membership])
        step_a = conditional_one_step(reported, target, "ex_post_max_information", None)
        step_b = conditional_one_step(reported, target, "assumed_publication_delay", None)
        baselines = {}
        for model in engine_constants()["baseline_models"]:
            predicted = predict_shares(history, membership, target, "ex_post_max_information", str(model))
            baselines[str(model)] = predicted["status"]
        safe = [
            str(record["safe_to_link"]).lower() == "true"
            for record in audit.to_dict(orient="records")
            if record["group_id"] == group["group_id"]
        ]
        rows.append(
            {
                "parent_area": group["parent_area"],
                "county": group["county"],
                "site_count": len(membership),
                "evaluable_months": len(targets),
                "development_month": target,
                "development_role": ANALYSIS_ROLE,
                "scenario_a_cutoff": step_a["information_cutoff"],
                "scenario_b_cutoff": step_b["information_cutoff"],
                "scenario_a_last_source": step_a["source_months"][-1] if step_a["source_months"] else "",
                "scenario_b_last_source": step_b["source_months"][-1] if step_b["source_months"] else "",
                "abm_status": step_a["prediction_status"],
                "abm_skip_reason": step_a["skip_reason"],
                "baseline_status": baselines,
                "utility_status": gaps["utility_status"],
                "missing_attributes": gaps["missing_attributes"],
                "safe_to_link": safe,
                "fair_2025_one_step": fair_one_step_status(safe),
                "reported_history_unchanged": sorted(reported.months) == step_a["reported_months_after"],
            }
        )
    return rows


def _synthetic_smoke() -> dict[str, object]:
    probabilities = softmax_probabilities([1.2, 0.2, -0.4])
    homogeneous = expected_share([probabilities, probabilities])
    stay = stay_adjusted_probability(probabilities, 0, 0.5)
    one = monte_carlo_mean(list(SMOKE_PROBABILITY), 1, SMOKE_SEED, AGENT_COUNT)
    averaged = monte_carlo_mean(list(SMOKE_PROBABILITY), replicates_for_share_se(), SMOKE_SEED, AGENT_COUNT)
    replay = monte_carlo_mean(list(SMOKE_PROBABILITY), 4, SMOKE_SEED, AGENT_COUNT)
    replay_again = monte_carlo_mean(list(SMOKE_PROBABILITY), 4, SMOKE_SEED, AGENT_COUNT)
    if replay["mean_share"] != replay_again["mean_share"]:
        raise RuntimeError("the same seed did not replay")
    endogenous = EndogenousHistory(("synth_a", "synth_b", "synth_c"))
    first = free_running_step(endogenous, "t1", [probabilities], "model_endogenous_presence")
    endogenous.add_model_share("t1", list(first["predicted_share"]), "model_endogenous_presence")
    second = free_running_step(endogenous, "t2", [stay], "model_endogenous_presence")
    zero = undefined_share(0.0)
    try:
        refuse_synthetic_alias("site_01", "stat_example")
        alias_blocked = False
    except ValueError:
        alias_blocked = True
    return {
        "probability": probabilities,
        "homogeneous_expected_share": homogeneous,
        "stay_adjusted_share": stay,
        "one_world_tv": one["one_world_tv"],
        "mean_max_abs_error": averaged["max_abs_error"],
        "replicates": averaged["replicates"],
        "seed_replayed": True,
        "scale_tradeoff": scale_tradeoff_same_probability([1.0, 2.0, 3.0], [0.0, 0.2, 1.0], 4.0),
        "free_running_steps": 2,
        "free_running_second_source_months": second["source_months"],
        "zero_demand_reason": zero,
        "synthetic_alias_blocked": alias_blocked,
        "input": "synthetic_probabilities",
    }


def _write(paths: V4Paths, payload: dict[str, object]) -> None:
    reports = paths.reports_root
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "V4_PHASE14_ADAPTER_ARCHITECTURE.md").write_text(_architecture(payload), encoding="utf-8")
    (reports / "V4_PHASE14_UTILITY_IDENTIFIABILITY.md").write_text(_identifiability(payload), encoding="utf-8")
    (reports / "V4_PHASE14_OBSERVATION_MAPPING.md").write_text(_mapping(payload), encoding="utf-8")
    (reports / "V4_PHASE14_SMOKE_QA.md").write_text(_qa(payload), encoding="utf-8")
    (reports / "V4_PHASE14_NEXT_DECISION.md").write_text(_decision(payload), encoding="utf-8")


def _architecture(payload: dict[str, object]) -> str:
    interfaces = "\n".join(f"- `{name}`" for name in payload["reusable_interfaces"])
    return "\n".join(
        [
            "# V4 Phase 14｜觀察轉接架構",
            "",
            "這一輪是 exploratory development。沒有執行 P4，沒有呼叫 LLM，也沒有讀取 2025–2026 保留人次。",
            "",
            "原始核心與 Experiment 001–020 沒有改寫。舊 Atlas 的 site_01 到 site_05 是五個合成場所，不能改名成玉山、太魯閣或雪霸的效用。",
            "",
            "## 可直接重用的介面",
            "",
            interfaces,
            "",
            "這些函式仍服務 J=5 的長期模型。觀察層只讀它們的契約，不把呼叫結果寫回 data/worlds。",
            "",
            "## 資料流",
            "",
            "WorldLab simulation state → agent probability or presence → observation aggregation → monthly predicted share → comparison with reported_visit_distribution。",
            "",
            "模式 A conditional_one_step 用截止月以前的真實回報份額建立狀態，預測下一個月，預測值不寫回觀測歷史。模式 B free_running 只接受 model_endogenous_presence。本輪模式 B 只在合成資料上做 smoke test。",
            "",
            "社會訊號必須帶來源標籤。回報份額的標籤是 reported_visit_share_conditional，exposure_claim 固定為 false。",
            "",
            f"選擇溫度、再考慮率與分數截斷沿用引擎常數：{payload['engine']}。",
            "",
        ]
    )


def _identifiability(payload: dict[str, object]) -> str:
    smoke = payload["smoke"]
    return "\n".join(
        [
            "# V4 Phase 14｜效用與可識別性",
            "",
            "第一版是非 LLM 的 attribute-only 介面。三個真實地理群的視覺吸引、新奇、容量、場所間距離與受訪者特質都不在觀測面板欄位裡。缺欄時效用狀態是 unavailable，沒有填入虛構旅遊特徵。",
            "",
            "LLM 效用只保留 schema：中性場所代碼、卡片版本、送出的屬性、排除的人氣欄位、模型版本、提示雜湊、解析規則與回應保存位。generation_status 是 not_run。",
            "",
            "社會權重與分數尺度不能從一條總體份額分開識別。合成例子把分數乘 2、權重除 2，softmax 機率保持相同。"
            f" 這次合成檢查的結果是 {smoke['scale_tradeoff']}。",
            "",
            "因此 social weight、LOGMAX 與 ZRAW 的尺度、記憶窗、衰減、選擇溫度與再考慮率，都只是校準用潛在量。它們不是由回報份額識別出來的個體心理參數。Q 仍是同一模型狀態下、有社會分數與沒有社會分數的選擇機率距離。",
            "",
            "事前固定的可行性檢查只使用合成機率與引擎既有常數。沒有在真實月份上搜尋權重，也沒有使用 2025 資料。",
            "",
        ]
    )


def _mapping(payload: dict[str, object]) -> str:
    smoke = payload["smoke"]
    lines = [
        "# V4 Phase 14｜觀察尺度對應",
        "",
        "比較目標是 reported_visit_distribution，也就是 B 級 conditional 的官方回報人次份額。它不是互斥的個人選擇機率。",
        "",
        "代理人數 20 是引擎人口，不是當月回報人次。一次決策步對到一個目標月，只是本輪寫明的對應假設，不是把 500 或 2,000 步換成某段日曆。單一世界的佔有率是 20 次多項抽樣；期望份額是各代理人機率的平均。多人多次的平均用來檢查 Monte Carlo 誤差，不取代分析式期望。",
        "",
        "J=3 與 J=5 共用 softmax、單純形與 Q 的定義。核心的累積、50 期滾動與 0.99 衰減記憶仍綁在五個合成場所。觀察層的 LOGMAX 與 ZRAW 以場所數為參數，並在測試裡對回 J=5 的原始函式。",
        "",
        "群組總需求為 0 時，份額是 undefined，原因是 all_zero_group_total，不改成均勻分布。",
        "",
        f"合成機率的單次 20-agent 世界對期望份額的 TV 是 {smoke['one_world_tv']:.3f}。"
        f" 把 {smoke['replicates']} 次重複平均之後，分量最大絕對誤差是 {smoke['mean_max_abs_error']:.3f}。",
        "",
        f"控制最壞情況 p=0.5 的分量標準誤到 0.01，需要的重複次數是 {payload['required_replicates']}。這控制的是平均份額的標準誤，不是單一個 20 人世界。",
        "",
        "固定方法是：可評分的預測使用分析式期望份額；Monte Carlo 只用來估計離散誤差。再考慮率 0.5 會把停留機率混進選擇機率，smoke 另記一條 stay-adjusted 向量，沒有把它當成真實月份。",
        "",
        "Phase 13 的七個基準，包含 0.5／0.5 組合，都保留。固定 J 時 share_MAE = 2 × share_vector_TV / J，所以 MAE 不是第二項獨立證據。主指標仍是 share_vector_TV。",
        "",
        "情境 A 的最新資訊是 t-1，情境 B 的最新資訊是 t-3。情境 B 是公布延遲假設，不是三步超前預測。",
        "",
    ]
    return "\n".join(lines)


def _qa(payload: dict[str, object]) -> str:
    lines = [
        "# V4 Phase 14｜工程 smoke QA",
        "",
        f"Python {sys.version.split()[0]}，平台 {platform.platform()}。角色是 {payload['analysis_role']}。LLM 呼叫數 {payload['llm_calls']}。",
        "",
        "讀取路徑只包含：",
        "",
    ]
    for path in payload["read_paths"]:
        lines.append(f"- `{path}`")
    lines.extend(["", "面板只讀這些欄："])
    for column in payload["panel_columns"]:
        lines.append(f"- `{column}`")
    lines.extend(
        [
            "",
            "沒有讀取保留檔的人次、份額或 HHI。合成別名 site_01 對到 stat_ 識別碼時會被拒絕。"
            f" 這次檢查是 {payload['smoke']['synthetic_alias_blocked']}。相同 seed 重跑是 {payload['smoke']['seed_replayed']}。",
            "",
            "開發月是各群第一個事前可評估月，用來檢查截止日與基準介面。它不是 P4，也不用來改測試月或參數。",
            "",
            "| 地理群 | 可評估月 | 開發月 | 情境 A 最後來源 | 情境 B 最後來源 | ABM | 2025 單步 |",
            "| --- | ---: | --- | --- | --- | --- | --- |",
        ]
    )
    for group in payload["groups"]:
        lines.append(
            "| {name} | {months} | {development} | {last_a} | {last_b} | {abm} | {holdout} |".format(
                name=f"{group['county']}／{group['parent_area']}",
                months=group["evaluable_months"],
                development=group["development_month"],
                last_a=group["scenario_a_last_source"],
                last_b=group["scenario_b_last_source"],
                abm=group["abm_status"],
                holdout=group["fair_2025_one_step"],
            )
        )
    lines.extend(
        [
            "",
            "七個基準在這個開發月都回傳 predicted。ABM 通道因為真實屬性不足而 prediction_unavailable。預測沒有寫回觀測歷史。",
            "",
            "受保護檔案在寫出報告前後的雜湊相同。",
            "",
        ]
    )
    return "\n".join(lines)


def _decision(payload: dict[str, object]) -> str:
    return "\n".join(
        [
            "# V4 Phase 14｜下一步",
            "",
            "比較框架已經能把同一組據點、同一個單步期限與同一個資料截止日接到 Phase 13 的七個基準。0.5／0.5 組合基準保留。Phase 13 裡玉山與雪霸的固定組合、太魯閣的近期份額基準較接近，那是探索性描述，不是事後選出的獨立驗證。",
            "",
            "真實景點的屬性效用 ABM 目前沒有可評分的份額。缺的是視覺吸引、新奇、容量、已核對的場所間距離，以及對應回報人次的受訪者特質。下一輪可以繼續做探索性比較的工程與合成閉環，不應把一個尚未有屬性的 ABM 分數寫成勝過統計基準。",
            "",
            "若要做公平的 2025 單步預測，三群目前都是 unavailable_incomplete_2024_identity。不能改用 2023 年 12 月，然後宣稱已經取得 t-1。玉山的共同窗口停在 2021，太魯閣沒有安全接上 2024 的成員，雪霸只有兩個成員安全接上，三人向量不完整。",
            "",
            "2025 單步還需要事前凍結：據點集合、截止情境、來源優先、測量限制、模型與權重、主指標 share_vector_TV，以及重疊窗口不作獨立樣本。這些項目本輪沒有凍結成確認性協定。",
            "",
            "模式 B 的介面已在合成資料上走通兩步。中途塞入回報份額會被拒絕。它還沒有接到三個地理群。",
            "",
        ]
    )


def _refuse_forbidden_reads() -> None:
    for path in READ_PATHS:
        if "2025" in path or "2026" in path:
            raise RuntimeError("a read path names a holdout year")


def _hashes() -> dict[str, str]:
    return {relative: file_sha256(WORLD_ROOT / relative) for relative in GUARDED}


if __name__ == "__main__":
    print(json.dumps(run_phase14(), ensure_ascii=False, indent=2, allow_nan=False))
