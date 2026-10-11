"""Build the v0.16 research-synthesis tables, figures, and source manifest.

This script reads frozen analysis files and writes only under
docs/research_synthesis_v016/. It does not run a world experiment and does
not call a model. The source manifest is a document-provenance record.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
FIGURE_DIR = OUT_DIR / "figures"
DATA_DIR = OUT_DIR / "derived"
LAMBDAS = ("0.40", "0.60", "0.80", "1.00")
HORIZONS = ("H500", "H1000", "H1500", "H2000")
ARCHITECTURES = ("RW50", "EXP99")
COHORTS = {
    "v0.15": {
        "world": "world_v0_15",
        "summary": "experiment_015_summary.json",
        "seeds": "50-99",
        "n": 50,
    },
    "v0.16": {
        "world": "world_v0_16",
        "summary": "experiment_016_summary.json",
        "seeds": "100-199",
        "n": 100,
    },
}
RELATION_ORDER = ("AA", "AP", "PA", "PP")
RELATION_LABELS = {
    "AA": "AA\n兩者皆未持續",
    "AP": "AP\n僅較高 λ 持續",
    "PA": "PA\n僅較低 λ 持續",
    "PP": "PP\n兩者皆持續",
}


PLOTTING_REQUIREMENTS = "docs/research_synthesis_v016/plotting-requirements.txt"
INSTALL_COMMAND = f".\\.venv\\Scripts\\python.exe -m pip install -r {PLOTTING_REQUIREMENTS}"


def load_plotting():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "SVG 與 PNG 需要繪圖環境。請在 worldlab/ 先安裝 requirements-lock.txt，再執行："
            f"{INSTALL_COMMAND}。這個程序不會自動安裝套件。"
        ) from exc
    return matplotlib, plt, font_manager


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def dig(document, pointer: str):
    current = document
    for part in pointer.strip("/").split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        current = current[int(part)] if isinstance(current, list) else current[part]
    return current


def cell_record(cohort: str, architecture: str, summary: dict, uncertainty: dict, lambda_label: str, horizon: str) -> dict:
    level = summary["regimes"][architecture]["levels"][lambda_label]["horizons"][horizon]
    interval = uncertainty["regimes"][architecture]["cells"][f"{horizon}:{lambda_label}"]
    meta = COHORTS[cohort]
    if interval["n_seeds"] != meta["n"] or level["distribution"]["n"] != meta["n"]:
        raise RuntimeError(f"{cohort} {architecture} {horizon} {lambda_label} does not keep its own sample size")
    return {
        "cohort": cohort,
        "seeds": meta["seeds"],
        "n": meta["n"],
        "architecture": architecture,
        "lambda": lambda_label,
        "horizon": horizon,
        "mean_d": level["D_horizon"],
        "ci95_low": interval["ci95_low"],
        "ci95_high": interval["ci95_high"],
        "ci_degenerate_zero": interval["degenerate_zero"],
        "median": level["distribution"]["median"],
        "persistence_class": level["persistence_class"],
        "persistent_count": level["seed_persistent_count"],
        "wilson95_low": level["wilson95"][0],
        "wilson95_high": level["wilson95"][1],
        "long_horizon_class": summary["regimes"][architecture]["levels"][lambda_label]["long_horizon_class"],
    }


def extract_tables() -> dict:
    rows = []
    identity = []
    for cohort, meta in COHORTS.items():
        directory = f"data/worlds/{meta['world']}"
        summary = load_json(f"{directory}/{meta['summary']}")
        uncertainty = load_json(f"{directory}/uncertainty_analysis.json")
        seed_identity = load_json(f"{directory}/seed_identity_analysis.json")
        if summary["llm_calls"] != 0 or uncertainty["llm_calls"] != 0:
            raise RuntimeError(f"{cohort} analysis recorded a model call")
        for architecture in ARCHITECTURES:
            for lambda_label in LAMBDAS:
                for horizon in HORIZONS:
                    rows.append(cell_record(cohort, architecture, summary, uncertainty, lambda_label, horizon))
        transition = seed_identity["regimes"]["EXP99"]["adjacent_lambda"]["H2000"]["0.40->0.60"]
        counts = transition["counts"]
        identity.append(
            {
                "cohort": cohort,
                "seeds": meta["seeds"],
                "n": meta["n"],
                "architecture": "EXP99",
                "horizon": "H2000",
                "contrast": "0.40->0.60",
                "AA": counts["AA"],
                "AP": counts["AP"],
                "PA": counts["PA"],
                "PP": counts["PP"],
                "jaccard": transition["jaccard"],
                "relation": transition["relation"],
            }
        )
    if {row["n"] for row in rows} != {50, 100}:
        raise RuntimeError("synthesis tables pooled the two cohorts")
    replication = load_json("data/worlds/world_v0_16/replication_analysis.json")
    reference = replication["primary_identity"]["relation"]
    return {"cells": rows, "identity": identity, "reference": reference, "replication": replication}


CLAIMS = [
    {
        "claim_id": "C01",
        "claim_zh": "直接二選一介面中，同一物理選項標成 A 與標成 B 的選擇率相差 0.6025。",
        "experiment": "experiment_001c",
        "source_path": "data/experiments/experiment_001c_order_invariance/summary.json",
        "json_path": "/presentation_effect/pooled/difference_A_minus_B",
        "value": 0.6025,
        "conditions": "鏡像呈現的直接選擇；n=4800 個 pooled 呈現。",
        "limitations": "這是測量介面的 artifact，不是場所效用本身的估計。",
        "evidence_role": "測量診斷",
    },
    {
        "claim_id": "C02",
        "claim_zh": "在位置與 schema 分開操作後，A 標籤優勢仍約 0.561，第一位置優勢約 -0.086，schema enum 順序差為 0。",
        "experiment": "experiment_001d",
        "source_path": "data/experiments/experiment_001d_label_position_schema/summary.json",
        "json_path": "/a_label_advantage/difference",
        "value": 0.5611111111111111,
        "conditions": "標籤、位置與 enum 順序的因子設計。",
        "limitations": "標籤效果大於位置效果；不能把直接選擇率當成與標籤無關的偏好。",
        "evidence_role": "測量診斷",
    },
    {
        "claim_id": "C03",
        "claim_zh": "001d 的 schema enum 順序優勢為 0。",
        "experiment": "experiment_001d",
        "source_path": "data/experiments/experiment_001d_label_position_schema/summary.json",
        "json_path": "/enum_order_advantage/difference",
        "value": 0.0,
        "conditions": "AB 與 BA enum，各 n=1440。",
        "limitations": "只排除這個設計裡的 enum 順序差，不表示所有介面都無順序效果。",
        "evidence_role": "測量診斷",
    },
    {
        "claim_id": "C04",
        "claim_zh": "獨立場所效用下，高 crowd aversion 的低擁擠選擇率高於低 aversion。",
        "experiment": "experiment_001e",
        "source_path": "data/experiments/experiment_001e_independent_utility/summary.json",
        "json_path": "/low_crowd_choice_rate_by_aversion/1.00/rate",
        "value": 0.975,
        "conditions": "每個地點獨立評分後再由 Python 比較；aversion 1.00 的 n=120。",
        "limitations": "這是凍結前的語意測量，不是下游世界的行為驗證。",
        "evidence_role": "測量診斷",
    },
    {
        "claim_id": "C05",
        "claim_zh": "photography 的效用成分被分類為 C，holistic sub-scores，不能當獨立心理量尺。",
        "experiment": "experiment_001f",
        "source_path": "data/experiments/experiment_001f_trait_attribute_separability/summary.json",
        "json_path": "/constructs/2/classification/class_code",
        "value": "C",
        "conditions": "固定其他欄位，比較目標成分與非目標成分的平均絕對變化。",
        "limitations": "分類是描述性規則，不是效度通過門檻。crowd 與 novelty 為部分可分離的 B。",
        "evidence_role": "測量診斷",
    },
    {
        "claim_id": "C06",
        "claim_zh": "Dynamics Calibration 的 regime 標籤沒有真人校準。",
        "experiment": "dynamics_calibration_001",
        "source_path": "data/worlds/dynamics_calibration_001/summary.json",
        "json_path": "/regime_note",
        "value": "Regime labels locate cells relative to this 16-cell grid. They are not a ranking of human realism. This design has no human calibration.",
        "conditions": "16 格 temperature × reconsideration grid，seeds 0–19，100 steps，0 LLM calls。",
        "limitations": "後續使用的 temperature 200 與 reconsideration 0.5 是技術參考點。",
        "evidence_role": "測量診斷",
    },
    {
        "claim_id": "C07",
        "claim_zh": "內生 familiarity reinforcement 沒有滿足預先規定的路徑放大規則。",
        "experiment": "experiment_003",
        "source_path": "data/worlds/world_v0_3/experiment_003_summary.json",
        "json_path": "/amplification/history_reinforcement_amplified_path_dependence",
        "value": False,
        "conditions": "R0、R50、R150 的內生熟悉度比較。",
        "limitations": "部分檢查為真，但 between-seed 長期分歧沒有上升，因此整條規則不成立。",
        "evidence_role": "受控對比",
    },
    {
        "claim_id": "C08",
        "claim_zh": "累積人氣項的結果落在預先分類之外，內生與外生 signal 的差距小於 margin。",
        "experiment": "experiment_004",
        "source_path": "data/worlds/world_v0_4/experiment_004_summary.json",
        "json_path": "/classification/label",
        "value": "mixed pattern outside the pre-specified classes",
        "conditions": "familiarity weight 0，lagged leave-one-out popularity，margin 0.01。",
        "limitations": "不能把這個混合結果寫成內生社會回饋已放大集體路徑依賴。",
        "evidence_role": "受控對比",
    },
    {
        "claim_id": "C09",
        "claim_zh": "rolling 與 exponential memory 使用 centered scores，social weight 150 不能跨架構當成同等強度。",
        "experiment": "experiment_005",
        "source_path": "data/worlds/world_v0_5/experiment_005_summary.json",
        "json_path": "/weights_not_equivalent_across_architectures",
        "value": True,
        "conditions": "CUM 沿用 004 的 log1p/max；RW 與 EXP 使用 centered normalization。",
        "limitations": "004 到 005 同時改變記憶表示與 normalization，不能把差異全歸因於 memory timescale。",
        "evidence_role": "測量診斷",
    },
    {
        "claim_id": "C10",
        "claim_zh": "v0.6 的技術參考格是 RW50、social weight 100，合格格只有 2 個。",
        "experiment": "experiment_006",
        "source_path": "data/worlds/world_v0_6/experiment_006_summary.json",
        "json_path": "/reference_regime/selection/cell_id",
        "value": "RW50_w100",
        "conditions": "四種 centered architectures，weights 25、50、100、150；CUM 排除在這個尺度之外。",
        "limitations": "這是模擬操作區間的選擇，不是真人行為校準。",
        "evidence_role": "測量診斷",
    },
    {
        "claim_id": "C10b",
        "claim_zh": "v0.6 參考規則下的合格格數是 2。",
        "experiment": "experiment_006",
        "source_path": "data/worlds/world_v0_6/experiment_006_summary.json",
        "json_path": "/reference_regime/eligible_count",
        "value": 2,
        "conditions": "centered architectures 與 weights 25、50、100、150。",
        "limitations": "合格是操作區間規則，不是真人校準通過。",
        "evidence_role": "測量診斷",
    },
    {
        "claim_id": "C11",
        "claim_zh": "在當時的 500-step 設計裡，早期擾動留下集體殘差。",
        "experiment": "experiment_007",
        "source_path": "data/worlds/world_v0_7/experiment_007_summary.json",
        "json_path": "/conclusion",
        "value": "early perturbations leave persistent collective residuals",
        "conditions": "t=21 兩個 agent override，其後自由演化；S0、RW50、EXP99。",
        "limitations": "後續 014 顯示這個 500-step 殘差對 horizon 敏感。結論不建立永久路徑依賴。",
        "evidence_role": "受控對比",
    },
    {
        "claim_id": "C12",
        "claim_zh": "對稱替代歷史在社會回饋下留下分歧，但沒有方向保留。",
        "experiment": "experiment_008",
        "source_path": "data/worlds/world_v0_8/experiment_008_summary.json",
        "json_path": "/conclusion",
        "value": "symmetric alternative early histories produce persistent but nondirectional collective divergence under social feedback",
        "conditions": "兩條同樣大小、指向不同地點的早期 override。",
        "limitations": "nondirectional 表示不能把殘差解釋成被指定目的地鎖定。",
        "evidence_role": "受控對比",
    },
    {
        "claim_id": "C13",
        "claim_zh": "在受測架構中，集體殘差可由多個內部狀態通道承載。",
        "experiment": "experiment_009",
        "source_path": "data/worlds/world_v0_9/experiment_009_summary.json",
        "json_path": "/conclusion",
        "value": "persistent collective divergence can be carried by multiple sufficient internal state channels under the tested architecture",
        "conditions": "t=21 之後交換 non-social state 或 social memory。",
        "limitations": "sufficiency 以後續既有動態仍運作為前提。Hybrid branches 不是自然歷史。",
        "evidence_role": "受控對比",
    },
    {
        "claim_id": "C14",
        "claim_zh": "不對稱比較裡，關掉社會影響後殘差仍在，重播 control social field 則壓低殘差。",
        "experiment": "experiment_010",
        "source_path": "data/worlds/world_v0_10/experiment_010_summary.json",
        "json_path": "/conclusion",
        "value": "persistent divergence survives removal of social influence, while replay of the control social field suppresses that residual under the tested architecture",
        "conditions": "t=22 起的 control-field replay 與 social-off，兩條歷史的後續環境並不對稱。",
        "limitations": "這個對比不能單獨識別 social-off 下的歷史持續。011 把環境拉齊後，social-off 的早期擾動衰減。",
        "evidence_role": "受控對比",
    },
    {
        "claim_id": "C15",
        "claim_zh": "對稱 social-off 下早期擾動衰減；共享非零社會場會同步兩條歷史。010 的放大 social-off 對比混合了早期擾動與不對稱後處理。",
        "experiment": "experiment_011",
        "source_path": "data/worlds/world_v0_11/experiment_011_summary.json",
        "json_path": "/conclusion",
        "value": "the early perturbation attenuates when both histories evolve under the same post-t22 social-off environment when post-t22 social influence is removed symmetrically from both histories, the early perturbation attenuates under the tested architecture a shared nonzero social field synchronizes alternative histories regardless of whether that field is generated by the control or perturbed donor trajectory shared social fields synchronize alternative histories independently of donor history under the tested architecture Experiment 010 established the preregistered asymmetric closure contrast. Experiment 011 separately identifies early-history persistence under symmetric post-t22 environments. the amplified SOCIAL_OFF contrast in Experiment 010 reflected the combined effect of early-history perturbation and asymmetric post-t22 social treatment, rather than isolated history persistence under social-off dynamics",
        "conditions": "control 與 perturbed 進入相同的 endogenous、shared-donor 或 social-off 環境。",
        "limitations": "010 的放大 social-off 對比混合了早期擾動與不對稱後處理。完整敘述見該 summary 的 conclusion。",
        "evidence_role": "受控對比",
    },
    {
        "claim_id": "C16",
        "claim_zh": "在已測 λ 上，歷史持續的平均反應呈分級 onset，粗區間是 (0.20, 0.40]。",
        "experiment": "experiment_012",
        "source_path": "data/worlds/world_v0_12/experiment_012_summary.json",
        "json_path": "/conclusion",
        "value": "history persistence shows a graded onset within the tested autonomy interval (0.20, 0.40]",
        "conditions": "seeds 0–49，500 steps。λ 混合共享中點場與分支內生場。",
        "limitations": "λ 是模擬控制參數。區間內部未測點不是精確臨界值。",
        "evidence_role": "受控對比",
    },
    {
        "claim_id": "C17",
        "claim_zh": "seeds 50–99 重現 012 的粗略持續型態，但兩個架構的細 onset band 不同。",
        "experiment": "experiment_013",
        "source_path": "data/worlds/world_v0_13/experiment_013_summary.json",
        "json_path": "/conclusion",
        "value": "the coarse feedback-autonomy persistence pattern from Experiment 012 replicated on held-out stochastic seeds in both social-memory regimes. RW50 refined onset band is (0.35, 0.40]. EXP99 refined onset band is (0.30, 0.35]. the held-out feedback-autonomy onset interval differs between the two tested social-memory architectures",
        "conditions": "同一架構、Atlas 與介入規則；500 steps；seeds 50–99。",
        "limitations": "這是 stochastic replication，不是新人口或外部驗證。",
        "evidence_role": "stochastic replication",
    },
    {
        "claim_id": "C18",
        "claim_zh": "EXP99、λ=.40 的 H2000 平均 D 為 0.01547，屬 attenuated，但 seed 持續比例仍是 0.22。",
        "experiment": "experiment_014",
        "source_path": "data/worlds/world_v0_14/horizon_persistence_analysis.json",
        "json_path": "/regimes/EXP99/levels/0.40/D_horizon/H2000",
        "value": 0.01547,
        "conditions": "seeds 50–99，t=0–2000。當時社會 λ 網格含 0.30、0.35、0.40、1.00。",
        "limitations": "平均分類與個別持續比例是不同結果。H2000 onset 落在 (0.40, 1.00]，內部未測。",
        "evidence_role": "受控對比",
    },
    {
        "claim_id": "C19",
        "claim_zh": "014 的兩個 social architectures 都是 ONSET_SHIFTS_UP。",
        "experiment": "experiment_014",
        "source_path": "data/worlds/world_v0_14/horizon_persistence_analysis.json",
        "json_path": "/regimes/EXP99/onset_stability",
        "value": "ONSET_SHIFTS_UP",
        "conditions": "H500、H1000、H1500、H2000 分開分類。",
        "limitations": "horizon sensitivity 不是 phase transition。RW50 同樣為 ONSET_SHIFTS_UP。",
        "evidence_role": "受控對比",
    },
    {
        "claim_id": "C20",
        "claim_zh": "v0.15 的 H2000 已測 onset band 在兩個架構都是 (0.40, 0.60]。",
        "experiment": "experiment_015",
        "source_path": "data/worlds/world_v0_15/horizon_onset_analysis.json",
        "json_path": "/regimes/RW50/horizon_onsets/H2000/horizon_onset_band/label",
        "value": "(0.40, 0.60]",
        "conditions": "seeds 50–99，λ=.40、.60、.80、1.00，n=50。",
        "limitations": "(0.40, 0.60) 內部沒有測試。這不是 critical λ。",
        "evidence_role": "受控對比",
    },
    {
        "claim_id": "C21",
        "claim_zh": "v0.15、EXP99、H2000、λ=.40→.60 的持續集合同時有退出與進入。",
        "experiment": "experiment_015",
        "source_path": "data/worlds/world_v0_15/seed_identity_analysis.json",
        "json_path": "/regimes/EXP99/adjacent_lambda/H2000/0.40->0.60/counts/PA",
        "value": 5,
        "conditions": "同一 cohort 內的 seed 集合比較，n=50。AP=21，PP=6。",
        "limitations": "Jaccard 差異不能直接比較到另一組 seed IDs。",
        "evidence_role": "探索性結果",
    },
    {
        "claim_id": "C22",
        "claim_zh": "v0.16 的 H2000 主要 onset 分類是 FULL_ONSET_REPLICATION。",
        "experiment": "experiment_016",
        "source_path": "data/worlds/world_v0_16/replication_analysis.json",
        "json_path": "/onset/overall",
        "value": "FULL_ONSET_REPLICATION",
        "conditions": "seeds 100–199 單獨分類。兩個架構都要同時滿足 .40 衰減且 .60、.80、1.00 達標。",
        "limitations": "這個標籤只描述主要 onset 判準，不涵蓋所有次要格子。",
        "evidence_role": "stochastic replication",
    },
    {
        "claim_id": "C23",
        "claim_zh": "v0.16、RW50、H2000、λ=.60 的平均 D 為 0.025925，分類為 persistent residual，中位數為 0。",
        "experiment": "experiment_016",
        "source_path": "data/worlds/world_v0_16/experiment_016_summary.json",
        "json_path": "/regimes/RW50/levels/0.60/horizons/H2000/D_horizon",
        "value": 0.025924999999999997,
        "conditions": "n=100。分類使用平均，不使用中位數或 CI。",
        "limitations": "點估計只比 0.025 高 0.000925。95% CI 跨過門檻。這不是高精度邊界定位。",
        "evidence_role": "stochastic replication",
    },
    {
        "claim_id": "C24",
        "claim_zh": "v0.16 主要 identity 格的觀察重疊是 12，期望重疊是 12.2，參考區間是 [8, 16]。",
        "experiment": "experiment_016",
        "source_path": "data/worlds/world_v0_16/replication_analysis.json",
        "json_path": "/primary_identity/relation/observed_overlap",
        "value": 12,
        "conditions": "EXP99、H2000、λ=.40→.60，n=100，kA=20，kB=61。",
        "limitations": "參考區間假設 seed labels 可交換且集合大小固定。它不是效應信賴區間，也不識別機制。",
        "evidence_role": "探索性結果",
    },
    {
        "claim_id": "C25",
        "claim_zh": "v0.16、EXP99、H1000、λ=.40 的平均分類是 attenuated。",
        "experiment": "experiment_016",
        "source_path": "data/worlds/world_v0_16/experiment_016_summary.json",
        "json_path": "/regimes/EXP99/levels/0.40/horizons/H1000/persistence_class",
        "value": "attenuated",
        "conditions": "新 cohort n=100。舊 cohort 在同一格的平均分類不同。",
        "limitations": "這個次要 horizon 的差異不改變 H2000 的主要 onset 判準。",
        "evidence_role": "stochastic replication",
    },
    {
        "claim_id": "C26",
        "claim_zh": "v0.16 正式模擬的 LLM calls 為 0，正式 worlds 為 2000。",
        "experiment": "experiment_016",
        "source_path": "data/worlds/world_v0_16/run_manifest.json",
        "json_path": "/llm_calls",
        "value": 0,
        "conditions": "manifest 同時記錄 world_count=2000、seed_count=100、complete=true。",
        "limitations": "這是工程驗證，不增加科學主張的外部效度。",
        "evidence_role": "工程驗證",
    },
]


def verify_claims() -> None:
    cache = {}
    roles = {"工程驗證", "測量診斷", "受控對比", "探索性結果", "stochastic replication"}
    for claim in CLAIMS:
        if claim["evidence_role"] not in roles:
            raise RuntimeError(f"{claim['claim_id']} has an unknown evidence role")
        if claim["source_path"] not in cache:
            cache[claim["source_path"]] = load_json(claim["source_path"])
        actual = dig(cache[claim["source_path"]], claim["json_path"])
        if actual != claim["value"]:
            raise RuntimeError(f"{claim['claim_id']} expected {claim['value']!r} but found {actual!r}")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_evidence_index() -> None:
    verify_claims()
    write_csv(
        OUT_DIR / "evidence_index.csv",
        [
            {
                "claim_id": claim["claim_id"],
                "claim_zh": claim["claim_zh"],
                "experiment": claim["experiment"],
                "source_path": claim["source_path"],
                "json_path": claim["json_path"],
                "value": json.dumps(claim["value"], ensure_ascii=False),
                "conditions": claim["conditions"],
                "limitations": claim["limitations"],
                "evidence_role": claim["evidence_role"],
            }
            for claim in CLAIMS
        ],
    )


def configure_font() -> str:
    _matplotlib, plt, font_manager = load_plotting()
    for name in ("Microsoft JhengHei", "Microsoft JhengHei UI", "PMingLiU", "SimHei"):
        if any(font.name == name for font in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return name
    raise RuntimeError("no Traditional Chinese font is available for the synthesis figures")


def finish_figure(figure, footnote: str, stem: str) -> None:
    _matplotlib, plt, _font_manager = load_plotting()
    figure.text(0.01, 0.01, footnote, ha="left", va="bottom", fontsize=8, wrap=True)
    figure.savefig(FIGURE_DIR / f"{stem}.svg", bbox_inches="tight")
    figure.savefig(FIGURE_DIR / f"{stem}.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_figures(tables: dict, font_name: str) -> None:
    _matplotlib, plt, _font_manager = load_plotting()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    colors = {"v0.15": "#0072B2", "v0.16": "#E69F00"}
    cells = tables["cells"]
    h2000 = [row for row in cells if row["horizon"] == "H2000"]
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 8.2), sharex="col")
    for row_index, architecture in enumerate(ARCHITECTURES):
        mean_axis = axes[row_index, 0]
        rate_axis = axes[row_index, 1]
        for cohort in COHORTS:
            subset = [item for item in h2000 if item["architecture"] == architecture and item["cohort"] == cohort]
            positions = [index + (-0.12 if cohort == "v0.15" else 0.12) for index in range(len(LAMBDAS))]
            means = [item["mean_d"] for item in subset]
            mean_axis.errorbar(
                positions,
                means,
                yerr=[
                    [max(0.0, item["mean_d"] - item["ci95_low"]) for item in subset],
                    [max(0.0, item["ci95_high"] - item["mean_d"]) for item in subset],
                ],
                fmt="o",
                color=colors[cohort],
                label=f"{cohort}（seeds {COHORTS[cohort]['seeds']}，n={COHORTS[cohort]['n']}）",
                capsize=3,
            )
            rates = [100 * item["persistent_count"] / item["n"] for item in subset]
            rate_axis.errorbar(
                positions,
                rates,
                yerr=[
                    [max(0.0, 100 * (item["persistent_count"] / item["n"] - item["wilson95_low"])) for item in subset],
                    [max(0.0, 100 * (item["wilson95_high"] - item["persistent_count"] / item["n"])) for item in subset],
                ],
                fmt="o",
                color=colors[cohort],
                capsize=3,
            )
        mean_axis.axhline(0.025, color="#444444", linestyle="--", linewidth=1, label="平均 D 分類門檻 .025")
        mean_axis.set_ylim(0, 0.13)
        mean_axis.set_ylabel("平均 occupancy TV")
        mean_axis.set_title(f"{architecture}：H2000 平均 D")
        rate_axis.set_ylim(0, 108)
        rate_axis.set_ylabel("個別持續比例（%）")
        rate_axis.set_title(f"{architecture}：H2000 持續比例")
        for axis in (mean_axis, rate_axis):
            axis.set_xticks(range(len(LAMBDAS)), LAMBDAS)
            axis.set_xlabel("λ")
            axis.grid(axis="y", color="#dddddd")
    axes[0, 0].legend(fontsize=8, loc="upper left")
    figure.suptitle("圖 1. H2000 的平均殘差與個別持續比例", fontsize=14)
    finish_figure(
        figure,
        "左側誤差線是以 seed 為單位的 percentile bootstrap 95% CI；右側是 Wilson 95% CI。兩者都是 pointwise，不是同時覆蓋。\n"
        "虛線只用於平均 D 的分類，不是持續比例的判準。v0.15 與 v0.16 分開繪製，沒有合併。t=2000 是有限時域。來源：各 cohort 的 summary 與 uncertainty_analysis.json。",
        "fig1_h2000_mean_and_persistence",
    )

    figure, axes = plt.subplots(2, 2, figsize=(11.2, 8.2), sharex=True, sharey=True)
    lambda_colors = {"0.40": "#009E73", "0.60": "#56B4E9", "0.80": "#E69F00", "1.00": "#D55E00"}
    for row_index, architecture in enumerate(ARCHITECTURES):
        for column_index, cohort in enumerate(COHORTS):
            axis = axes[row_index, column_index]
            for lambda_label in LAMBDAS:
                subset = [
                    item
                    for item in cells
                    if item["architecture"] == architecture and item["cohort"] == cohort and item["lambda"] == lambda_label
                ]
                axis.plot(
                    range(len(HORIZONS)),
                    [item["mean_d"] for item in subset],
                    marker="o",
                    color=lambda_colors[lambda_label],
                    label=f"λ={lambda_label}",
                )
            axis.axhline(0.025, color="#444444", linestyle="--", linewidth=1)
            axis.set_title(f"{architecture}，{cohort}（n={COHORTS[cohort]['n']}）")
            axis.grid(axis="y", color="#dddddd")
            axis.set_xticks(range(len(HORIZONS)), ["H500\n401–500", "H1000\n901–1000", "H1500\n1401–1500", "H2000\n1901–2000"])
            if column_index == 0:
                axis.set_ylabel("平均 occupancy TV")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4, fontsize=9, frameon=False)
    axes[0, 0].set_ylim(0, 0.13)
    figure.suptitle("圖 2. 四個指定視窗的平均殘差", fontsize=14, y=1.12)
    finish_figure(
        figure,
        "連線只連接已測視窗，不是未測時間的估計。虛線是平均 D=.025 的分類門檻。四個面板使用同一尺度。\n"
        "分類由各視窗的 regime mean 決定。來源：world_v0_15 與 world_v0_16 的 experiment summary。有限時域止於 t=2000。",
        "fig2_horizon_mean_d",
    )

    figure, axis = plt.subplots(figsize=(10.4, 6.6))
    width = 0.36
    for index, cohort in enumerate(COHORTS):
        row = next(item for item in tables["identity"] if item["cohort"] == cohort)
        proportions = [row[name] / row["n"] for name in RELATION_ORDER]
        positions = [item + (-width / 2 if cohort == "v0.15" else width / 2) for item in range(len(RELATION_ORDER))]
        bars = axis.bar(positions, proportions, width=width, color=colors[cohort], label=f"{cohort}（n={row['n']}，Jaccard={row['jaccard']:.4f}）")
        for bar, name in zip(bars, RELATION_ORDER, strict=True):
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{row[name]}/{row['n']}", ha="center", va="bottom", fontsize=8)
    axis.set_xticks(range(len(RELATION_ORDER)), [RELATION_LABELS[name] for name in RELATION_ORDER])
    axis.set_ylim(0, 0.72)
    axis.set_ylabel("占該 cohort seeds 的比例")
    axis.set_title("圖 3. EXP99、H2000、λ=.40→.60 的 seed 集合轉換")
    axis.legend(fontsize=8)
    axis.grid(axis="y", color="#dddddd")
    reference = tables["reference"]
    finish_figure(
        figure,
        "兩個 cohort 的 seed IDs 不互相比對。Jaccard 不同不直接證明 identity stability 改變。\n"
        f"v0.16 觀察重疊 PP={reference['observed_overlap']}，固定集合大小的期望重疊 {reference['expected_overlap']}，"
        f"中央 95% 參考區間 {reference['hypergeometric_central_95']}。這不是效應的信賴區間。來源：兩份 seed_identity_analysis.json 與 v0.16 replication_analysis.json。",
        "fig3_exp99_h2000_identity",
    )
    del font_name


def write_manifest(font_name: str) -> None:
    sources = sorted({claim["source_path"] for claim in CLAIMS})
    sources.extend(
        [
            "data/worlds/world_v0_15/experiment_015_summary.json",
            "data/worlds/world_v0_15/uncertainty_analysis.json",
            "data/worlds/world_v0_15/seed_identity_analysis.json",
            "data/worlds/world_v0_16/experiment_016_summary.json",
            "data/worlds/world_v0_16/uncertainty_analysis.json",
            "data/worlds/world_v0_16/seed_identity_analysis.json",
        ]
    )
    unique_sources = sorted(set(sources))
    recorded_outputs = [
        ROOT / "docs" / "RESEARCH_SYNTHESIS_V016.md",
        OUT_DIR / "plotting-requirements.txt",
        OUT_DIR / "REBUILD.md",
        OUT_DIR / "evidence_index.csv",
        *DATA_DIR.glob("*"),
        *FIGURE_DIR.glob("*"),
    ]
    outputs = sorted(
        path.relative_to(ROOT).as_posix()
        for path in recorded_outputs
        if path.is_file() and path.name != "source_manifest.json"
    )
    matplotlib, _plt, _font_manager = load_plotting()

    payload = {
        "record_type": "research_synthesis_source_manifest",
        "not_an_experiment_run_manifest": True,
        "synthesis_id": "research_synthesis_v016",
        "formal_worlds_added": 0,
        "llm_calls": 0,
        "derived_method": "Read frozen summary, uncertainty, identity, onset, and replication JSON. Keep v0.15 and v0.16 as separate cohorts. Plot mean D with percentile bootstrap intervals and persistent rates with Wilson intervals. Identity bars use frozen AA/AP/PA/PP counts.",
        "script": "docs/research_synthesis_v016/build_synthesis.py",
        "script_sha256": sha256(Path(__file__)),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "matplotlib": matplotlib.__version__,
        "font": font_name,
        "sources": [{"path": relative, "sha256": sha256(ROOT / relative)} for relative in unique_sources],
        "outputs": [{"path": relative, "sha256": sha256(ROOT / relative)} for relative in outputs],
    }
    (OUT_DIR / "source_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    tables = extract_tables()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(DATA_DIR / "horizon_cells.csv", tables["cells"])
    write_csv(DATA_DIR / "identity_transitions.csv", tables["identity"])
    (DATA_DIR / "v016_identity_reference.json").write_text(
        json.dumps(tables["reference"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_evidence_index()
    font_name = configure_font()
    plot_figures(tables, font_name)
    write_manifest(font_name)


if __name__ == "__main__":
    main()
