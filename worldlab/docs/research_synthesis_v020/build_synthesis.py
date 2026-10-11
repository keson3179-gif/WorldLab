"""Build Research Synthesis v0.20 tables and figures from frozen sources.

The script does not run a world experiment, refit weights, or recompute a
formal bootstrap. It writes only under docs/research_synthesis_v020/ and the
synthesis markdown. Matplotlib is imported only while drawing figures.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
FIGURE_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "derived_tables"
DATA = ROOT / "data"
PLOTTING_REQUIREMENTS = "docs/research_synthesis_v020/plotting-requirements.txt"
HASH_CHUNK = 1024 * 1024
LAMBDAS = ("0.40", "0.60", "0.80", "1.00")
Q_LABELS = ("q05", "q15", "q25")
MEMORIES = ("CUM", "RW50", "EXP99")
TRANSFORMS = ("LOGMAX", "ZRAW")
PATHS = ("EE", "EO", "OE", "OO")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(HASH_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_pointer(document: Any, pointer: str) -> Any:
    node = document
    for part in pointer.split("/"):
        if isinstance(node, list):
            node = node[int(part)]
        else:
            node = node[part]
    return node


def refuse_data_write(path: Path) -> None:
    resolved = path.resolve()
    data_root = DATA.resolve()
    if resolved == data_root or data_root in resolved.parents:
        raise RuntimeError("synthesis refuses to write into data/")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    refuse_data_write(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def condition_id(memory: str, transform: str, label: str) -> str:
    return f"{memory}_{transform}__{label}"


def extract_tables() -> dict[str, Any]:
    summary_016 = read_json(DATA / "worlds/world_v0_16/experiment_016_summary.json")
    uncertainty_017 = read_json(DATA / "worlds/world_v0_17/uncertainty_analysis.json")
    uncertainty_018 = read_json(DATA / "worlds/world_v0_18/uncertainty_analysis.json")
    uncertainty_019 = read_json(DATA / "worlds/world_v0_19/uncertainty_analysis.json")
    uncertainty_020 = read_json(DATA / "worlds/world_v0_20/uncertainty_analysis.json")
    audit_020 = read_json(ROOT / "docs/experiment_020_audit/audit_results.json")
    places_019 = read_json(DATA / "worlds/world_v0_19/source_place_count.json")
    summary_001c = read_json(DATA / "experiments/experiment_001c_order_invariance/summary.json")
    summary_001d = read_json(DATA / "experiments/experiment_001d_label_position_schema/summary.json")
    h2000 = []
    for architecture in ("RW50", "EXP99"):
        for lam in LAMBDAS:
            row = summary_016["regimes"][architecture]["levels"][lam]["horizons"]["H2000"]
            h2000.append(
                {
                    "cohort": "v0.16",
                    "seeds": "100-199",
                    "n": row["n"],
                    "architecture": architecture,
                    "lambda": lam,
                    "horizon": "H2000",
                    "outcome": "paired occupancy total variation",
                    "mean_d": row["D_horizon"],
                    "ci_low": row["mean_ci95"][0],
                    "ci_high": row["mean_ci95"][1],
                    "persistence_class": row["persistence_class"],
                    "seed_persistent_rate": row["seed_persistent_rate"],
                    "threshold": 0.025,
                    "ci_crosses_threshold": row["mean_ci95"][0] < 0.025 < row["mean_ci95"][1],
                }
            )
    weight_rows = []
    for layer_name, outcome in (("layer_a", "full-horizon mean Q at social weight 150"), ("layer_b", "late visit-share HHI, committed t=101-500")):
        layer = uncertainty_017["cohort_b"][layer_name]
        for cell in ("CUM_LOGMAX", "CUM_ZRAW", "RW50_LOGMAX", "RW50_ZRAW", "EXP99_LOGMAX", "EXP99_ZRAW", "S0"):
            if cell not in layer["summaries"]:
                continue
            item = layer["summaries"][cell]
            weight_rows.append(
                {
                    "cohort": "017 cohort B",
                    "seeds": layer["index_meaning"],
                    "n": item["n"],
                    "cell": cell,
                    "outcome": outcome,
                    "mean": item["mean"],
                    "ci_low": item["ci_low"],
                    "ci_high": item["ci_high"],
                }
            )
    matched_q = []
    for cell in uncertainty_018["matching"]["cells"]:
        hhi = uncertainty_018["summaries"][f"hhi::{cell['condition_id']}"]
        matched_q.append(
            {
                "cohort": "018 held-out",
                "seeds": "700-1699",
                "n": hhi["n"],
                "condition_id": cell["condition_id"],
                "q_window": "full horizon, S0 reference history",
                "hhi_window": "committed t=101-500 cumulative presence",
                "mean_q": cell["mean_q"],
                "q_ci_low": cell["ci_low"],
                "q_ci_high": cell["ci_high"],
                "q_match_band": cell["band"],
                "q_ci_inside_match_band": cell["ci_inside_match_band"],
                "late_hhi": hhi["mean"],
                "hhi_ci_low": hhi["ci_low"],
                "hhi_ci_high": hhi["ci_high"],
            }
        )
    late_q = []
    for memory in MEMORIES:
        for label in Q_LABELS:
            for transform, native_history in (("LOGMAX", "HL"), ("ZRAW", "HZ")):
                native = uncertainty_019["summaries"][f"q_{memory}_{label}_{native_history}_{transform}"]
                reference = uncertainty_019["summaries"][f"q_{memory}_{label}_H0_{transform}"]
                late_q.append(
                    {
                        "cohort": "019 replay of 018 seeds",
                        "seeds": "700-1699",
                        "n": native["n"],
                        "memory": memory,
                        "transform": transform,
                        "label": label,
                        "window": "decision t=101-500",
                        "native_q": native["mean"],
                        "native_ci_low": native["ci_low"],
                        "native_ci_high": native["ci_high"],
                        "s0_history_q": reference["mean"],
                        "s0_ci_low": reference["ci_low"],
                        "s0_ci_high": reference["ci_high"],
                    }
                )
    paths = []
    maintenance = []
    for memory in MEMORIES:
        for transform in TRANSFORMS:
            for label in Q_LABELS:
                spec = condition_id(memory, transform, label)
                for schedule in PATHS:
                    hhi = uncertainty_020["summaries"][f"hhi_{spec}_{schedule}"]
                    level = uncertainty_020["levels"][spec][schedule]
                    paths.append(
                        {
                            "cohort": "020",
                            "seeds": "1700-2699",
                            "n": hhi["n"],
                            "condition_id": spec,
                            "schedule": schedule,
                            "hhi_window": "committed t=101-500 cumulative presence",
                            "late_hhi": hhi["mean"],
                            "hhi_ci_low": hhi["ci_low"],
                            "hhi_ci_high": hhi["ci_high"],
                            "early_rate": level["early_rate"],
                            "late_new": level["late_new"],
                            "event_absent": level["event_absent"],
                            "eligible_n": level["eligible_n"],
                            "maintained_rate": level["maintained_rate"],
                            "released_rate": level["released_rate"],
                            "transfer_rate": level["transfer_rate"],
                            "plot_maintenance": level["eligible_n"] > 0 and level["maintained_rate"] is not None,
                            "terminals": level["terminals"],
                        }
                    )
                for schedule in ("EE", "EO", "OE", "OO"):
                    level = uncertainty_020["levels"][spec][schedule]
                    maintenance.append(
                        {
                            "condition_id": spec,
                            "schedule": schedule,
                            "eligible_n": level["eligible_n"],
                            "maintained_rate": level["maintained_rate"],
                            "released_rate": level["released_rate"],
                            "transfer_rate": level["transfer_rate"],
                            "plot_as_rate": level["eligible_n"] > 0,
                        }
                    )
    exits = []
    for spec, schedules in audit_020["maintenance"]["conditions"].items():
        row = schedules["EO"]
        if row["eligible_n"] == 0:
            continue
        exits.append(
            {
                "condition_id": spec,
                "schedule": "EO",
                "eligible_n": row["eligible_n"],
                "original_exit": row["original_exit"],
                "release_time": row["release_time"],
                "maintained_n": row["maintained_n"],
                "released_n": row["released_n"],
                "transfer_n": row["transfer_n"],
                "exit_matches_release": row["original_exit"] == row["release_time"],
            }
        )
    intervals = audit_020["early_given_o"]
    return {
        "cohorts_are_pooled": False,
        "h2000": h2000,
        "weight150": weight_rows,
        "matched_q": matched_q,
        "late_q_019": late_q,
        "paths_020": paths,
        "maintenance_020": maintenance,
        "eo_exits": exits,
        "early_given_o": intervals,
        "place_relation_019_q25_logmax": places_019,
        "measurement": {
            "presentation_difference": summary_001c["presentation_effect"]["pooled"]["difference_A_minus_B"],
            "a_label_advantage": summary_001d["a_label_advantage"]["difference"],
            "first_position_advantage": summary_001d["first_position_advantage"]["difference"],
            "enum_order_advantage": summary_001d["enum_order_advantage"]["difference"],
        },
        "onset_016": summary_016["replication"]["onset"]["overall"],
        "field_gap": audit_020["field_gap"],
        "manifest_020": {
            "checked_count": audit_020["manifest"]["checked_count"],
            "match_count": audit_020["manifest"]["match_count"],
            "manifest_all_matched": audit_020["manifest"]["manifest_all_matched"],
            "progress_explained_as_final_finished_append": audit_020["manifest"]["progress_explained_as_final_finished_append"],
            "other_listed_files_match": audit_020["manifest"]["other_listed_files_match"],
        },
        "finished_elapsed_020": audit_020["elapsed"]["finished_progress_seconds"],
    }


def evidence_claims(tables: dict[str, Any]) -> list[dict[str, Any]]:
    rw = next(row for row in tables["h2000"] if row["architecture"] == "RW50" and row["lambda"] == "0.60")
    exception = next(row for row in tables["early_given_o"] if row["contrast"] == "early_given_O_EXP99_ZRAW__q25")
    exception_index = next(index for index, row in enumerate(tables["early_given_o"]) if row["contrast"] == exception["contrast"])
    cum_q25 = next(row for row in tables["eo_exits"] if row["condition_id"] == "CUM_LOGMAX__q25")
    return [
        {
            "claim_id": "M01",
            "claim_zh": "在 001c 的直接二選一介面中，同一物理選項標成 A 與標成 B 的 pooled 選擇率差是 0.6025。",
            "experiment": "experiment_001c",
            "conditions": "鏡像呈現的直接選擇",
            "unit": "pooled presentations",
            "n": 4800,
            "outcome": "choice-rate difference A minus B",
            "window": "the 001c presentation task",
            "contrast": "same physical option labeled A versus B",
            "source_path": "data/experiments/experiment_001c_order_invariance/summary.json",
            "json_pointer": "presentation_effect/pooled/difference_A_minus_B",
            "value": tables["measurement"]["presentation_difference"],
            "evidence_role": "measurement",
            "scope": "這個直接選擇介面的測量差。",
            "not_implied": "不是 A-label effect，也不是真人場所偏好。",
            "verified_this_round": True,
        },
        {
            "claim_id": "M02",
            "claim_zh": "001d 把標籤、位置與 enum 順序拆開後，A 標籤 pooled 優勢是 0.5611111111111111（n＝2880），第一位置差是 -0.08611111111111108，enum 順序差是 0。",
            "experiment": "experiment_001d",
            "conditions": "label, position, and schema-order factorial",
            "unit": "pooled decisions for the A-label contrast",
            "n": 2880,
            "outcome": "A-label, first-position, and enum-order advantages",
            "window": "the 001d rating task",
            "contrast": "A label versus B label, holding the other factors in that design",
            "source_path": "data/experiments/experiment_001d_label_position_schema/summary.json",
            "json_pointer": "a_label_advantage/difference",
            "value": tables["measurement"]["a_label_advantage"],
            "evidence_role": "measurement",
            "scope": "這個因子設計裡的標籤項。",
            "not_implied": "不能與 001c 的 0.6025 互換名稱。",
            "verified_this_round": True,
        },
        {
            "claim_id": "H01",
            "claim_zh": "016 的 H2000 已測 onset band 是 (0.40, 0.60]，整體分類是 FULL_ONSET_REPLICATION。",
            "experiment": "experiment_016",
            "conditions": "RW50 and EXP99, lambda 0.40, 0.60, 0.80, 1.00",
            "unit": "paired seeds",
            "n": 100,
            "outcome": "occupancy total variation between matched histories",
            "window": "H2000",
            "contrast": "perturbed history minus its matched-RNG control",
            "source_path": "data/worlds/world_v0_16/experiment_016_summary.json",
            "json_pointer": "replication/onset/overall",
            "value": tables["onset_016"],
            "evidence_role": "replication",
            "scope": "seeds 100-199 與事先固定的點估計規則。",
            "not_implied": "不是區間內部已測量，不是臨界點，也不是永久路徑依賴。",
            "verified_this_round": True,
        },
        {
            "claim_id": "H02",
            "claim_zh": "RW50、λ=.60、H2000 的平均 TV 約 0.025925，95% 區間跨過 0.025。",
            "experiment": "experiment_016",
            "conditions": "RW50, lambda 0.60",
            "unit": "paired seeds",
            "n": rw["n"],
            "outcome": "mean paired occupancy TV",
            "window": "H2000",
            "contrast": "matched-RNG perturbed versus control",
            "source_path": "data/worlds/world_v0_16/experiment_016_summary.json",
            "json_pointer": "regimes/RW50/levels/0.60/horizons/H2000/D_horizon",
            "value": rw["mean_d"],
            "evidence_role": "replication",
            "scope": "這個格子的點估計分類仍是 persistent residual。",
            "not_implied": "區間跨過門檻，所以邊界沒有被精確定位。",
            "verified_this_round": True,
        },
        {
            "claim_id": "C01",
            "claim_zh": "017 cohort B 在同一個 social weight 150 下，LOGMAX 的全時段平均 Q 遠低於 ZRAW。",
            "experiment": "experiment_017",
            "conditions": "six memory by transformation cells at weight 150",
            "unit": "seeds",
            "n": 500,
            "outcome": "Layer A mean Q",
            "window": "full horizon on the shared reference history",
            "contrast": "LOGMAX minus the same memory's ZRAW is not the stored claim; the two means are reported separately",
            "source_path": "data/worlds/world_v0_17/uncertainty_analysis.json",
            "json_pointer": "cohort_b/layer_a/summaries/CUM_LOGMAX/mean",
            "value": next(row["mean"] for row in tables["weight150"] if row["cell"] == "CUM_LOGMAX" and row["outcome"].startswith("full-horizon")),
            "evidence_role": "description",
            "scope": "weight 150 與 cohort B。",
            "not_implied": "相同 weight 不是相同決策影響，這個 Q 也不是 late HHI。",
            "verified_this_round": True,
        },
        {
            "claim_id": "C02",
            "claim_zh": "018 的 18 個 matched 條件在 held-out seeds 上，全時段參考 Q 的區間落在目標帶內，但 late HHI 仍然分開。",
            "experiment": "experiment_018",
            "conditions": "18 matched memory by transformation by q label",
            "unit": "seeds",
            "n": 1000,
            "outcome": "full-horizon mean Q on the S0 reference history, and late visit-share HHI",
            "window": "Q is the full horizon; HHI is committed t=101-500 cumulative presence",
            "contrast": "held-out mean Q versus the calibration target band",
            "source_path": "data/worlds/world_v0_18/uncertainty_analysis.json",
            "json_pointer": "matching/cells/0/ci_inside_match_band",
            "value": tables["matched_q"][0]["q_ci_inside_match_band"],
            "evidence_role": "description",
            "scope": "seeds 700-1699 與已固定 weights。",
            "not_implied": "平均 Q 匹配不保證閉環 HHI、時間分布或目的地相同。",
            "verified_this_round": True,
        },
        {
            "claim_id": "C03",
            "claim_zh": "019 在 decision t=101-500，把同一規則放在自身閉環與 S0 歷史上，高權重 LOGMAX 的 Q 明顯不同。",
            "experiment": "experiment_019",
            "conditions": "CUM LOGMAX q25, native history versus S0 history",
            "unit": "seeds",
            "n": 1000,
            "outcome": "mean Q on the reconsideration-conditional five-option softmax",
            "window": "decision t=101-500",
            "contrast": "native LOGMAX history versus the same rule scored on S0",
            "source_path": "data/worlds/world_v0_19/uncertainty_analysis.json",
            "json_pointer": "summaries/q_CUM_q25_HL_LOGMAX/mean",
            "value": next(row["native_q"] for row in tables["late_q_019"] if row["memory"] == "CUM" and row["transform"] == "LOGMAX" and row["label"] == "q25"),
            "evidence_role": "offline score",
            "scope": "既有 018 軌跡上的交叉評分，不是新的來源干預。",
            "not_implied": "不能把 018 的全時段參考 Q 減去這個 late window，也不能把事件對齊當成因果方向。",
            "verified_this_round": True,
        },
        {
            "claim_id": "I01",
            "claim_zh": "020 在指定 weights 與 seeds 1700-2699 上，替換 social-field 來源與其 t=51 時程，改變高權重 LOGMAX 的集中形成與維持。",
            "experiment": "experiment_020",
            "conditions": "18 matched conditions and schedules EE, EO, OE, OO",
            "unit": "seeds",
            "n": 1000,
            "outcome": "late visit-share HHI and the 18/20 concentration event",
            "window": "HHI uses committed t=101-500; the switch is decision t=51",
            "contrast": "paired source-and-timing schedules",
            "source_path": "data/worlds/world_v0_20/uncertainty_analysis.json",
            "json_pointer": "summaries/hhi_CUM_LOGMAX__q25_EE/mean",
            "value": next(row["late_hhi"] for row in tables["paths_020"] if row["condition_id"] == "CUM_LOGMAX__q25" and row["schedule"] == "EE"),
            "evidence_role": "intervention",
            "scope": "這個來源替換政策。O 只替換 social field。",
            "not_implied": "不是等 Q 的純 feedback 占比，也不是永久吸收態。",
            "verified_this_round": True,
        },
        {
            "claim_id": "I02",
            "claim_zh": "CUM LOGMAX q25 的合格 EO 世界在 t=51 退出 999 次、在 t=52 退出 1 次。",
            "experiment": "experiment_020",
            "conditions": "EO, pre-switch eligible seeds",
            "unit": "eligible seeds",
            "n": cum_q25["eligible_n"],
            "outcome": "original_exit timestep",
            "window": "after decision t=51",
            "contrast": "counts at t=51 and t=52",
            "source_path": "docs/experiment_020_audit/audit_results.json",
            "json_pointer": "maintenance/conditions/CUM_LOGMAX__q25/EO/original_exit/51",
            "value": cum_q25["original_exit"]["51"],
            "evidence_role": "intervention",
            "scope": "已保存 seed metrics 的重新彙整。",
            "not_implied": "中位數是 51，不能寫成每一個合格世界都在 t=51 退出。",
            "verified_this_round": True,
        },
        {
            "claim_id": "I03",
            "claim_zh": "18 個 EO−OO 區間有 17 個包含 0；EXP99 ZRAW q25 的區間高於 0。",
            "experiment": "experiment_020",
            "conditions": "early_given_O contrasts",
            "unit": "seeds",
            "n": 1000,
            "outcome": "late HHI difference EO minus OO",
            "window": "committed t=101-500",
            "contrast": "early E versus early O, with late source fixed at O",
            "source_path": "docs/experiment_020_audit/audit_results.json",
            "json_pointer": f"early_given_o/{exception_index}/mean",
            "value": exception["mean"],
            "evidence_role": "intervention",
            "scope": "既有 percentile 區間，沒有重算。",
            "not_implied": "微小幅度不是已證明沒有差異，本輪也沒有等價判準。",
            "verified_this_round": True,
        },
    ]


def verify_claims(tables: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    tables = tables or extract_tables()
    claims = evidence_claims(tables)
    documents: dict[str, Any] = {}
    for claim in claims:
        path = ROOT / claim["source_path"]
        if claim["source_path"] not in documents:
            documents[claim["source_path"]] = read_json(path)
        resolved = resolve_pointer(documents[claim["source_path"]], claim["json_pointer"])
        if resolved != claim["value"]:
            raise RuntimeError(f"{claim['claim_id']} does not match {claim['json_pointer']}")
    if tables["measurement"]["first_position_advantage"] != resolve_pointer(
        read_json(DATA / "experiments/experiment_001d_label_position_schema/summary.json"),
        "first_position_advantage/difference",
    ):
        raise RuntimeError("001d position effect does not match the stored table")
    if tables["measurement"]["enum_order_advantage"] != 0.0:
        raise RuntimeError("001d enum-order effect is no longer zero")
    summary_001c = read_json(DATA / "experiments/experiment_001c_order_invariance/summary.json")
    summary_001d = read_json(DATA / "experiments/experiment_001d_label_position_schema/summary.json")
    if next(item["n"] for item in claims if item["claim_id"] == "M01") != summary_001c["presentation_effect"]["pooled"]["n"]:
        raise RuntimeError("001c pooled n does not match the claim")
    if next(item["n"] for item in claims if item["claim_id"] == "M02") != summary_001d["a_label_advantage"]["pooled"]["n"]:
        raise RuntimeError("001d pooled n does not match the claim")
    if tables["onset_016"] != "FULL_ONSET_REPLICATION":
        raise RuntimeError("016 onset label changed")
    if not all(row["q_ci_inside_match_band"] for row in tables["matched_q"]):
        raise RuntimeError("an 018 held-out Q interval left its match band")
    if sum(1 for row in tables["early_given_o"] if row["contains_zero"]) != 17:
        raise RuntimeError("EO-OO zero coverage is not 17 of 18")
    if any(row["maintained_rate"] == 0 for row in tables["maintenance_020"] if row["eligible_n"] == 0):
        raise RuntimeError("an empty maintenance denominator was stored as zero")
    if tables["manifest_020"]["manifest_all_matched"] is not False:
        raise RuntimeError("the 020 manifest was marked fully matched")
    if tables["manifest_020"]["match_count"] != 2019 or tables["manifest_020"]["checked_count"] != 2020:
        raise RuntimeError("the 020 manifest counts changed")
    if {row["n"] for row in tables["h2000"]} != {100}:
        raise RuntimeError("016 sample size was mixed with another cohort")
    if {row["n"] for row in tables["paths_020"]} != {1000}:
        raise RuntimeError("020 sample size changed")
    return claims


def write_tables(tables: dict[str, Any]) -> None:
    refuse_data_write(TABLE_DIR)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(TABLE_DIR / "h2000_016.csv", tables["h2000"])
    _write_csv(TABLE_DIR / "weight150_017.csv", tables["weight150"])
    _write_csv(TABLE_DIR / "matched_q_hhi_018.csv", tables["matched_q"])
    _write_csv(TABLE_DIR / "late_q_019.csv", tables["late_q_019"])
    path_rows = [{key: value for key, value in row.items() if key != "terminals"} for row in tables["paths_020"]]
    _write_csv(TABLE_DIR / "paths_020.csv", path_rows)
    _write_csv(TABLE_DIR / "maintenance_020.csv", tables["maintenance_020"])
    _write_csv(TABLE_DIR / "eo_exits_020.csv", tables["eo_exits"])
    terminals = []
    for row in tables["paths_020"]:
        if row["schedule"] != "EE":
            continue
        for place, count in sorted(row["terminals"].items()):
            terminals.append({"condition_id": row["condition_id"], "schedule": "EE", "place": place, "count": count, "n": row["n"]})
    _write_csv(TABLE_DIR / "ee_terminals_020.csv", terminals)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    refuse_data_write(path)
    if not rows:
        raise RuntimeError(f"no rows for {path.name}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if value is None:
        return ""
    return value


def configure_font() -> str:
    try:
        import matplotlib
        from matplotlib import font_manager
    except ImportError as exc:
        raise RuntimeError(
            "繪圖需要先安裝核心環境，再安裝繪圖依賴："
            " .\\.venv\\Scripts\\python.exe -m pip install -r requirements-lock.txt"
            f" 與 .\\.venv\\Scripts\\python.exe -m pip install -r {PLOTTING_REQUIREMENTS}"
        ) from exc
    matplotlib.use("Agg")
    names = {font.name for font in font_manager.fontManager.ttflist}
    if "Microsoft JhengHei" not in names:
        raise RuntimeError("中文圖需要 Microsoft JhengHei，重建已停止")
    return "Microsoft JhengHei"


def draw_figures(tables: dict[str, Any]) -> str:
    font = configure_font()
    from matplotlib import pyplot as plt

    plt.rcParams["font.sans-serif"] = [font]
    plt.rcParams["axes.unicode_minus"] = False
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    _figure_lines(plt)
    _figure_h2000(plt, tables)
    _figure_calibration(plt, tables)
    _figure_intervention(plt, tables)
    return font


def _save(figure: Any, stem: str) -> None:
    for suffix in (".svg", ".png"):
        path = FIGURE_DIR / f"{stem}{suffix}"
        refuse_data_write(path)
        figure.savefig(path, bbox_inches="tight")


def _figure_lines(plt: Any) -> None:
    figure, axis = plt.subplots(figsize=(11, 4.8))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 6)
    axis.axis("off")
    boxes = [
        (0.4, 2.2, "A  選擇介面與量測\n001c 呈現差\n001d 標籤／位置／順序\n001e 獨立場所效用"),
        (4.2, 2.2, "B  有限時域歷史差異\n007–016\n配對 occupancy TV\nH2000 與 λ 網格"),
        (8.0, 2.2, "C  閉環集中\n017–020\nQ、late HHI\n與來源時程干預"),
    ]
    for x, y, text in boxes:
        axis.add_patch(plt.Rectangle((x, y), 3.4, 2.6, fill=False, linewidth=1.2))
        axis.text(x + 1.7, y + 1.3, text, ha="center", va="center", fontsize=9)
    axis.annotate("", xy=(4.15, 3.5), xytext=(3.85, 3.5), arrowprops={"arrowstyle": "->", "lw": 1.1})
    axis.annotate("", xy=(7.95, 3.5), xytext=(7.65, 3.5), arrowprops={"arrowstyle": "->", "lw": 1.1})
    axis.text(6.0, 5.35, "箭頭只表示研究推進與證據依賴", ha="center", fontsize=10)
    axis.text(6.0, 0.7, "三個結果使用不同 cohort，沒有併成同一批樣本", ha="center", fontsize=9)
    _save(figure, "fig1_research_lines")
    plt.close(figure)


def _figure_h2000(plt: Any, tables: dict[str, Any]) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.8))
    positions = {lam: index for index, lam in enumerate(LAMBDAS)}
    for architecture, offset in (("RW50", -0.08), ("EXP99", 0.08)):
        rows = [row for row in tables["h2000"] if row["architecture"] == architecture]
        xs = [positions[row["lambda"]] + offset for row in rows]
        ys = [row["mean_d"] for row in rows]
        lows = [row["mean_d"] - row["ci_low"] for row in rows]
        highs = [row["ci_high"] - row["mean_d"] for row in rows]
        axis.errorbar(xs, ys, yerr=[lows, highs], fmt="o", capsize=3, label=architecture)
    axis.axhline(0.025, color="black", linewidth=0.8, label="0.025 門檻")
    axis.set_xticks(range(len(LAMBDAS)), [f"λ={lam}" for lam in LAMBDAS])
    axis.set_ylabel("H2000 平均 occupancy TV")
    axis.set_xlabel("已測 λ。RW50、λ=.60 的區間跨過 0.025。(0.40, 0.60) 內部沒有測量。")
    axis.set_title("016 seeds 100–199，n=100；只畫已測 λ，點與點之間沒有連線")
    axis.legend(fontsize=8)
    _save(figure, "fig2_h2000_tv")
    plt.close(figure)


def _figure_calibration(plt: Any, tables: dict[str, Any]) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(12, 9))
    labels = [row["condition_id"].replace("__", "\n") for row in tables["matched_q"]]
    xs = list(range(len(labels)))
    q_means = [row["mean_q"] for row in tables["matched_q"]]
    q_err = [[row["mean_q"] - row["q_ci_low"] for row in tables["matched_q"]], [row["q_ci_high"] - row["mean_q"] for row in tables["matched_q"]]]
    axes[0].errorbar(xs, q_means, yerr=q_err, fmt="o", capsize=2)
    for target in (0.05, 0.15, 0.25):
        axes[0].axhline(target, color="gray", linewidth=0.6)
    axes[0].set_ylabel("全時段參考 Q")
    axes[0].set_title("018 held-out，seeds 700–1699，n=1000；S0 參考歷史")
    hhi = [row["late_hhi"] for row in tables["matched_q"]]
    hhi_err = [[row["late_hhi"] - row["hhi_ci_low"] for row in tables["matched_q"]], [row["hhi_ci_high"] - row["late_hhi"] for row in tables["matched_q"]]]
    axes[1].errorbar(xs, hhi, yerr=hhi_err, fmt="o", capsize=2)
    axes[1].set_ylabel("late HHI，t=101–500")
    axes[1].set_title("018 同一 cohort 的閉環 late HHI。HHI=1 表示該世界集中於一處，不同 seed 的場所可以不同")
    native = [row["native_q"] for row in tables["late_q_019"]]
    reference = [row["s0_history_q"] for row in tables["late_q_019"]]
    late_labels = [f"{row['memory']}\n{row['transform']}\n{row['label']}" for row in tables["late_q_019"]]
    late_x = list(range(len(late_labels)))
    axes[2].bar([item - 0.15 for item in late_x], reference, width=0.3, label="同一規則，S0 歷史")
    axes[2].bar([item + 0.15 for item in late_x], native, width=0.3, label="同一規則，自身閉環")
    axes[2].set_ylabel("decision t=101–500 的 Q")
    axes[2].set_title("019，seeds 700–1699，decision t=101–500；與 018 的全時段參考 Q 分開")
    axes[2].legend(fontsize=8)
    for axis, names in ((axes[0], labels), (axes[1], labels), (axes[2], late_labels)):
        axis.set_xticks(range(len(names)), names, fontsize=6)
    figure.tight_layout(h_pad=1.4)
    _save(figure, "fig3_calibration_and_closed_loop")
    plt.close(figure)


def _figure_intervention(plt: Any, tables: dict[str, Any]) -> None:
    figure = plt.figure(figsize=(12, 12))
    grid = figure.add_gridspec(4, 3, height_ratios=[1, 1, 1.3, 1.3])
    panel_specs = []
    for memory in MEMORIES:
        for transform in TRANSFORMS:
            panel_specs.append([condition_id(memory, transform, label) for label in Q_LABELS])
    for index, group in enumerate(panel_specs):
        axis = figure.add_subplot(grid[index // 3, index % 3])
        xs = list(range(3))
        for offset, schedule in enumerate(PATHS):
            means = []
            for spec in group:
                row = next(item for item in tables["paths_020"] if item["condition_id"] == spec and item["schedule"] == schedule)
                means.append(row["late_hhi"])
            axis.bar([item + (offset - 1.5) * 0.18 for item in xs], means, width=0.18, label=schedule)
        axis.set_xticks(xs, [spec.split("__")[1] for spec in group], fontsize=8)
        axis.set_ylim(0, 1.05)
        axis.set_title(group[0].split("__")[0], fontsize=8)
        if index == 0:
            axis.legend(fontsize=6, ncol=4)
    formation_axis = figure.add_subplot(grid[2, :])
    labels = []
    early = []
    other = []
    ordered = [condition_id(memory, transform, label) for memory in MEMORIES for transform in TRANSFORMS for label in Q_LABELS]
    for spec in ordered:
        ee = next(item for item in tables["paths_020"] if item["condition_id"] == spec and item["schedule"] == "EE")
        oe = next(item for item in tables["paths_020"] if item["condition_id"] == spec and item["schedule"] == "OE")
        labels.append(spec.replace("__", "\n"))
        early.append(ee["early_rate"])
        other.append(oe["early_rate"])
    xs = list(range(len(labels)))
    formation_axis.bar([item - 0.15 for item in xs], early, width=0.3, label="E-early")
    formation_axis.bar([item + 0.15 for item in xs], other, width=0.3, label="O-early")
    formation_axis.set_xticks(xs, labels, fontsize=5)
    formation_axis.set_ylabel("早期形成比例")
    formation_axis.set_xlabel("O-early 在這 18 個條件的早期形成比例都是 0，柱高貼在底線。")
    formation_axis.legend(fontsize=7)
    maintain_axis = figure.add_subplot(grid[3, :])
    shown = [row for row in tables["maintenance_020"] if row["schedule"] == "EE" and row["plot_as_rate"]]
    labels = [f"{row['condition_id']}\nn={row['eligible_n']}" for row in shown]
    ee_rates = [row["maintained_rate"] for row in shown]
    eo_rates = []
    for row in shown:
        pair = next(item for item in tables["maintenance_020"] if item["condition_id"] == row["condition_id"] and item["schedule"] == "EO")
        eo_rates.append(pair["maintained_rate"])
    xs = list(range(len(labels)))
    if labels:
        maintain_axis.bar([item - 0.15 for item in xs], ee_rates, width=0.3, label="EE")
        maintain_axis.bar([item + 0.15 for item in xs], eo_rates, width=0.3, label="EO")
        maintain_axis.set_xticks(xs, labels, fontsize=6)
    maintain_axis.set_ylabel("原場所維持到 t=500")
    maintain_axis.set_xlabel("合格 EO 的維持率是 0，橙色柱貼齊底線。O-early 合格數為 0，維持率是 NA，沒有畫成 0。")
    maintain_axis.legend(fontsize=7)
    maintain_axis.set_title("只畫切換前合格的條件；刻度上的 n 是合格數", fontsize=9)
    figure.tight_layout()
    _save(figure, "fig4_source_schedules")
    plt.close(figure)
    if len(ordered) != 18 or len(panel_specs) != 6:
        raise RuntimeError("020 figure did not cover every matched condition")


def direct_sources() -> list[str]:
    return [
        "data/experiments/experiment_001c_order_invariance/summary.json",
        "data/experiments/experiment_001d_label_position_schema/summary.json",
        "data/worlds/world_v0_16/experiment_016_summary.json",
        "data/worlds/world_v0_17/uncertainty_analysis.json",
        "data/worlds/world_v0_18/uncertainty_analysis.json",
        "data/worlds/world_v0_19/uncertainty_analysis.json",
        "data/worlds/world_v0_19/source_place_count.json",
        "data/worlds/world_v0_20/uncertainty_analysis.json",
        "docs/experiment_020_audit/audit_results.json",
        "docs/EXPERIMENT_020_ERRATA.md",
        "docs/EXPERIMENT_020_REPORT.md",
    ]


def indirect_sources() -> list[dict[str, Any]]:
    return [
        {
            "path": "docs/research_synthesis_v016/evidence_index.csv",
            "role": "index for experiments 001-016; this round does not reverify those trajectories",
            "verified_this_round": False,
        },
        {
            "path": "docs/RESEARCH_SYNTHESIS_V016.md",
            "role": "earlier synthesis kept unchanged",
            "verified_this_round": False,
        },
    ]


def output_paths() -> list[str]:
    names = [
        "docs/RESEARCH_SYNTHESIS_V020.md",
        "docs/research_synthesis_v020/build_synthesis.py",
        "docs/research_synthesis_v020/REBUILD.md",
        "docs/research_synthesis_v020/plotting-requirements.txt",
        "docs/research_synthesis_v020/evidence_index.json",
        "docs/research_synthesis_v020/derived_tables/h2000_016.csv",
        "docs/research_synthesis_v020/derived_tables/weight150_017.csv",
        "docs/research_synthesis_v020/derived_tables/matched_q_hhi_018.csv",
        "docs/research_synthesis_v020/derived_tables/late_q_019.csv",
        "docs/research_synthesis_v020/derived_tables/paths_020.csv",
        "docs/research_synthesis_v020/derived_tables/maintenance_020.csv",
        "docs/research_synthesis_v020/derived_tables/eo_exits_020.csv",
        "docs/research_synthesis_v020/derived_tables/ee_terminals_020.csv",
        "docs/research_synthesis_v020/figures/fig1_research_lines.svg",
        "docs/research_synthesis_v020/figures/fig1_research_lines.png",
        "docs/research_synthesis_v020/figures/fig2_h2000_tv.svg",
        "docs/research_synthesis_v020/figures/fig2_h2000_tv.png",
        "docs/research_synthesis_v020/figures/fig3_calibration_and_closed_loop.svg",
        "docs/research_synthesis_v020/figures/fig3_calibration_and_closed_loop.png",
        "docs/research_synthesis_v020/figures/fig4_source_schedules.svg",
        "docs/research_synthesis_v020/figures/fig4_source_schedules.png",
    ]
    return names


def write_manifest(font: str) -> None:
    import matplotlib

    payload = {
        "record_type": "research_synthesis_v020_source_manifest",
        "python": platform.python_version(),
        "matplotlib": matplotlib.__version__,
        "font": font,
        "worlds_resimulated": False,
        "bootstrap_recomputed": False,
        "llm_calls": 0,
        "historical_trajectories_reverified": False,
        "direct_sources": [{"path": path, "sha256": file_sha256(ROOT / path), "verified_this_round": True} for path in direct_sources()],
        "indirect_sources": indirect_sources(),
        "outputs": [{"path": path, "sha256": file_sha256(ROOT / path)} for path in output_paths()],
        "excluded_from_this_manifest": ["docs/research_synthesis_v020/source_manifest.json"],
    }
    write_json(OUT_DIR / "source_manifest.json", payload)


def main() -> None:
    before = {path: file_sha256(ROOT / path) for path in direct_sources()}
    tables = extract_tables()
    claims = verify_claims(tables)
    write_json(OUT_DIR / "evidence_index.json", {"claims": claims, "llm_calls": 0})
    write_tables(tables)
    font = draw_figures(tables)
    after = {path: file_sha256(ROOT / path) for path in direct_sources()}
    if before != after:
        raise RuntimeError("a frozen source changed while building the synthesis")
    write_manifest(font)


if __name__ == "__main__":
    main()
