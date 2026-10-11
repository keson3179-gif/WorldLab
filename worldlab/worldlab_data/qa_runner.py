from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandera.errors import SchemaError
import pandera.pandas as pa

from worldlab_data.config import V4Paths

MISSING_STATUSES = {"missing", "missing_token", "closed_reported", "not_in_statistics"}


def visit_schema() -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        {
            "site_id": pa.Column(str),
            "year_month": pa.Column(str),
            "visits": pa.Column(float, nullable=True, coerce=True),
            "observation_status": pa.Column(str),
        },
        strict=False,
    )


def evaluate_gates(paths: V4Paths) -> list[dict[str, str]]:
    visits_path = paths.processed / "site_month_visits.parquet"
    master_path = paths.processed / "attraction_master.parquet"
    panel_path = paths.processed / "site_month_panel.parquet"
    membership_path = paths.curated / "competition_groups.csv"
    crosswalk_path = paths.curated / "site_crosswalk.csv"
    gates: list[dict[str, str]] = []
    if not visits_path.exists():
        return [_gate("files", "visit table exists", "FAIL", "尚未產生 site_month_visits.parquet")]
    visits = pd.read_parquet(visits_path)
    gates.append(_schema_gate(visits))
    gates.append(_duplicate_gate(visits))
    gates.append(_missing_zero_gate(visits))
    gates.append(_negative_gate(visits))
    gates.append(_annual_gate(visits))
    if master_path.exists():
        master = pd.read_parquet(master_path)
        duplicate_ids = int(master["site_id"].duplicated().sum())
        gates.append(_gate("site_id", "attraction master site_id unique", "PASS" if duplicate_ids == 0 else "FAIL", str(duplicate_ids)))
    else:
        gates.append(_gate("site_id", "attraction master exists", "FAIL", "missing"))
    if crosswalk_path.exists():
        crosswalk = pd.read_csv(crosswalk_path, dtype=str, keep_default_na=False)
        fuzzy_confirmed = 0
        if not crosswalk.empty and "match_status" in crosswalk.columns:
            fuzzy_confirmed = int(
                ((crosswalk["match_status"] == "confirmed_manual") & (crosswalk.get("match_origin", "") != "manual")).sum()
            )
        gates.append(
            _gate(
                "crosswalk",
                "fuzzy score did not confirm a site",
                "PASS" if fuzzy_confirmed == 0 else "FAIL",
                str(fuzzy_confirmed),
            )
        )
    else:
        gates.append(_gate("crosswalk", "crosswalk exists", "WARN", "尚無候選檔"))
    if membership_path.exists():
        membership = pd.read_csv(membership_path, dtype=str, keep_default_na=False)
        included = membership[membership["included"] == "True"] if not membership.empty else membership
        duplicate_membership = int(included["site_id"].duplicated().sum()) if not included.empty else 0
        gates.append(_gate("groups", "a site is in at most one included group", "PASS" if duplicate_membership == 0 else "FAIL", str(duplicate_membership)))
        official = int((membership.get("backtest_eligible", "False") == "True").sum()) if not membership.empty else 0
        gates.append(_gate("groups", "official backtest groups", "WARN" if official == 0 else "PASS", f"eligible_rows={official}"))
    if panel_path.exists():
        panel = pd.read_parquet(panel_path)
        leaked = 0
        if not panel.empty:
            leaked = int(((panel["strict_feature_eligible"] == True) & (panel["publication_history_unknown"] == True)).sum())  # noqa: E712
        gates.append(_gate("leakage", "strict features require known availability", "PASS" if leaked == 0 else "FAIL", str(leaked)))
        conflicted = 0
        if not visits.empty and "observation_status" in visits.columns:
            bad_ids = set(visits.loc[visits["observation_status"] == "conflicting_source_version", "site_id"].astype(str))
            if bad_ids and not panel.empty:
                conflicted = int(panel["site_id"].astype(str).isin(bad_ids).sum())
        gates.append(_gate("versions", "conflicting visit versions stay out of the panel", "PASS" if conflicted == 0 else "FAIL", str(conflicted)))
    else:
        gates.append(_gate("panel", "panel file exists", "WARN", "尚未建立"))
    gates.append(_gate("publication", "visitor publication history", "WARN", "publication_history_unknown"))
    gates.append(_gate("measurement", "counting method", "WARN", "measurement_method=unknown，未推定各據點相同"))
    return gates


def _schema_gate(visits: pd.DataFrame) -> dict[str, str]:
    try:
        visit_schema().validate(visits[["site_id", "year_month", "visits", "observation_status"]], lazy=True)
    except SchemaError as exc:
        return _gate("schema", "pandera visit schema", "FAIL", str(exc).splitlines()[0][:300])
    return _gate("schema", "pandera visit schema", "PASS", f"rows={len(visits)}")


def _duplicate_gate(visits: pd.DataFrame) -> dict[str, str]:
    duplicates = int(visits.duplicated(["site_id", "year_month"]).sum()) if not visits.empty else 0
    return _gate("duplicates", "one row per site and month", "PASS" if duplicates == 0 else "FAIL", str(duplicates))


def _missing_zero_gate(visits: pd.DataFrame) -> dict[str, str]:
    if visits.empty:
        return _gate("missing", "missing is not stored as zero", "PASS", "0")
    mask = visits["observation_status"].isin(MISSING_STATUSES) & (visits["visits"] == 0)
    count = int(mask.sum())
    return _gate("missing", "missing is not stored as zero", "PASS" if count == 0 else "FAIL", str(count))


def _negative_gate(visits: pd.DataFrame) -> dict[str, str]:
    if visits.empty:
        return _gate("range", "no negative reported visits", "PASS", "0")
    count = int(((visits["observation_status"].isin(["reported", "reported_zero"])) & (visits["visits"] < 0)).sum())
    return _gate("range", "no negative reported visits", "PASS" if count == 0 else "FAIL", str(count))


def _annual_gate(visits: pd.DataFrame) -> dict[str, str]:
    if visits.empty or "annual_reconciliation" not in visits.columns:
        return _gate("annual", "annual total reconciliation recorded", "WARN", "no flag column")
    mismatches = int(visits.loc[visits["annual_reconciliation"] == "annual_total_mismatch", ["site_id", "year"]].drop_duplicates().shape[0])
    status = "WARN" if mismatches else "PASS"
    return _gate("annual", "month sum versus published total", status, f"unreconciled_site_years={mismatches}; month values were not rewritten")


def _gate(stage: str, check: str, status: str, detail: str) -> dict[str, str]:
    return {"stage": stage, "check": check, "status": status, "detail": detail}


def write_gate_report(gates: list[dict[str, str]], reports: Path) -> int:
    reports.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(gates)
    frame.to_csv(reports / "qa_gates.csv", index=False, encoding="utf-8-sig")
    fails = int((frame["status"] == "FAIL").sum()) if not frame.empty else 0
    warns = int((frame["status"] == "WARN").sum()) if not frame.empty else 0
    passes = int((frame["status"] == "PASS").sum()) if not frame.empty else 0
    lines = [
        "# WorldLab V4 QA 摘要",
        "",
        f"PASS {passes}、WARN {warns}、FAIL {fails}。FAIL 會讓 `qa` 指令以非零狀態結束。",
        "",
        "| 階段 | 檢查 | 結果 | 說明 |",
        "| --- | --- | --- | --- |",
    ]
    for gate in gates:
        lines.append(f"| {gate['stage']} | {gate['check']} | {gate['status']} | {gate['detail']} |")
    lines.append("")
    (reports / "WORLDLAB_V4_QA_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    return fails
