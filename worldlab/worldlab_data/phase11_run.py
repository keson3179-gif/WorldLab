from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd

from worldlab_data.config import WORLD_ROOT, V4Paths, load_paths
from worldlab_data.phase10 import method_family
from worldlab_data.phase11 import (
    HOLDOUT_YEAR,
    HYPOTHETICAL_SCALE_FACTORS,
    assess_group_month,
    candidate_common_shock,
    candidate_structural_break,
    exclude_holdout,
    hypothetical_scaled_hhi,
    identity_semantics,
    leader_turnover,
    month_span,
    previous_month,
    series_diagnostics,
    share_total_variation,
)
from worldlab_data.textutil import comparison_key, file_sha256

GUARDED = (
    "data/v4_observational/processed/site_month_visits.parquet",
    "data/curated/v4_exploratory_visit_shares_conditional.csv",
    "data/curated/v4_group_comparability.csv",
    "data/curated/v4_canonical_site_crosswalk.csv",
    "data/curated/v4_measurement_method_verified.csv",
    "reports/V4_PHASE10_READINESS.md",
)

COUNTING_LIMIT = "探索性、conditional。官方回報人次份額，沒有據點級計數文件。全群零人次的 HHI 無定義。2021 年來源衝突月份沒有選定版本。"


def run_phase11(paths: V4Paths | None = None) -> dict[str, object]:
    paths = paths or load_paths()
    before = _hashes()
    curated = WORLD_ROOT / "data" / "curated"
    groups = pd.read_csv(paths.reports_root / "V4_PHASE8_GROUPS.csv", dtype=str, keep_default_na=False)
    comparability = pd.read_csv(curated / "v4_group_comparability.csv", dtype=str, keep_default_na=False)
    crosswalk = pd.read_csv(curated / "v4_canonical_site_crosswalk.csv", dtype=str, keep_default_na=False)
    verified = pd.read_csv(curated / "v4_measurement_method_verified.csv", dtype=str, keep_default_na=False)
    shares = pd.read_csv(curated / "v4_exploratory_visit_shares_conditional.csv", dtype=str, keep_default_na=False)
    evidence = pd.read_csv(paths.reports_root / "V4_PHASE10_2021_MONTH_EVIDENCE.csv", dtype=str, keep_default_na=False)
    visits = pd.read_parquet(paths.processed / "site_month_visits.parquet")
    panel, summaries = _build(groups, comparability, crosswalk, verified, shares, evidence, visits)
    _check_known_months(panel)
    leaked = [row["year_month"] for row in panel if int(str(row["year_month"])[:4]) >= HOLDOUT_YEAR]
    if leaked:
        raise RuntimeError("holdout months entered the exploratory panel")
    holdout = _holdout_flag(paths)
    name_map = {site["site_uid"]: site["official_name"] for item in summaries for site in item["continuity"]}
    figures = _figures(panel, name_map)
    frame = pd.DataFrame.from_records([{key: value for key, value in row.items() if key != "shares"} for row in panel])
    validity = frame[
        [
            "group_id",
            "county",
            "parent_area",
            "year_month",
            "expected_sites",
            "reported_sites",
            "group_total_visits",
            "reported_months",
            "positive_total_months",
            "share_defined",
            "source_conflict",
            "exclusion_reason",
        ]
    ].copy()
    curated.mkdir(parents=True, exist_ok=True)
    validity.to_csv(curated / "v4_month_validity.csv", index=False, encoding="utf-8-sig")
    frame.to_parquet(curated / "v4_monthly_concentration_conditional.parquet", index=False)
    _write_reports(paths, summaries, figures, holdout, panel)
    payload = {
        "analysis_role": "exploratory",
        "rows": len(panel),
        "share_defined_rows": int(sum(row["share_defined"] for row in panel)),
        "excluded_rows": int(sum(not row["share_defined"] for row in panel)),
        "groups": [
            {
                "parent_area": item["parent_area"],
                "defined_months": item["diagnostics"]["defined_months"],
                "normalized_hhi_sd": item["diagnostics"]["normalized_hhi_sd"],
                "hhi_lag1_correlation": item["diagnostics"]["hhi_lag1_correlation"],
                "total_hhi_change_correlation": item["diagnostics"]["total_hhi_change_correlation"],
                "retrospective_origins": item["diagnostics"]["retrospective_origins"],
                "freeze_at_2024": item["freeze_at_2024"],
            }
            for item in summaries
        ],
        "holdout_visits_used": False,
        "holdout_used_for_selection": holdout["used_for_group_selection"],
        "protected_files_unchanged": _hashes() == before,
        "model_fitted": False,
    }
    (paths.interim / "phase11_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if not payload["protected_files_unchanged"]:
        raise RuntimeError("phase 11 changed a protected file")
    return payload


def _build(groups, comparability, crosswalk, verified, shares, evidence, visits):
    eligible = comparability[
        comparability["definition"].eq("A_geographic") & comparability["exploratory_share_eligible"].str.lower().eq("true")
    ]
    membership = groups[groups["definition"].eq("A_geographic")]
    lookup = _visit_lookup(visits)
    conflicts, gaps = _conflicts(evidence)
    labels = _labels(verified)
    names = _names(crosswalk)
    panel: list[dict[str, object]] = []
    summaries = []
    for group in eligible.to_dict(orient="records"):
        members = membership[membership["group_id"].eq(group["group_id"])].to_dict(orient="records")
        uids, raw_ids, statuses = _uids(members, crosswalk)
        identity = identity_semantics(statuses)
        months, held = exclude_holdout(month_span(group["common_from"], group["common_until"]))
        if held:
            raise RuntimeError("common window contains holdout months")
        rows = []
        previous = None
        for year_month in months:
            values: dict[str, float | None] = {}
            source_conflict = False
            within = False
            for uid in uids:
                value, conflict_hit, disagree = _uid_value(raw_ids[uid], year_month, lookup, conflicts)
                values[uid] = value
                source_conflict = source_conflict or conflict_hit
                within = within or disagree
            assessed = assess_group_month(values, source_conflict, within)
            tv = None
            turnover = None
            structural = False
            if previous is not None and previous["year_month"] == previous_month(year_month) and assessed["share_defined"] and previous["share_defined"]:
                tv = share_total_variation(previous["shares"], assessed["shares"])
                turnover = leader_turnover(
                    str(previous["leading_site_uid"]),
                    bool(previous["leader_tie"]),
                    str(assessed["leading_site_uid"]),
                    bool(assessed["leader_tie"]),
                )
                structural = candidate_structural_break(previous.get("normalized_hhi"), assessed.get("normalized_hhi"))
            row = {
                "analysis_role": "exploratory",
                "comparability_label": "reported_visits_conditional",
                "group_id": group["group_id"],
                "county": group["county"],
                "parent_area": group["parent_area"],
                "year_month": year_month,
                "year": int(year_month[:4]),
                "month": int(year_month[5:7]),
                "group_total_reported_visits": assessed["group_total_visits"],
                "leading_site_name": names.get(str(assessed["leading_site_uid"]), ""),
                "month_to_month_share_tv": tv,
                "leading_site_turnover": turnover,
                "share_vector": json.dumps(assessed["shares"], ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                "candidate_common_shock": candidate_common_shock(year_month),
                "candidate_structural_break": structural,
                "rule_match_coverage": identity["rule_match_coverage"],
                "externally_verified_identity_status": identity["externally_verified_identity_status"],
                "counting_limits": COUNTING_LIMIT,
                "conflict_detail": _gap_text(raw_ids, year_month, gaps) if source_conflict else "",
                **assessed,
                **{key: value for key, value in identity.items() if key not in assessed},
            }
            rows.append(row)
            if assessed["share_defined"]:
                previous = row
        _reconcile(group, rows, shares)
        measurement = _measurement(uids, raw_ids, crosswalk, labels, months)
        sensitivity = _sensitivity(rows)
        continuity = _continuity(uids, raw_ids, crosswalk, lookup, names)
        diagnostics = series_diagnostics(rows)
        panel.extend(rows)
        summaries.append(
            {
                "group_id": group["group_id"],
                "county": group["county"],
                "parent_area": group["parent_area"],
                "common_from": group["common_from"],
                "common_until": group["common_until"],
                "common_months_phase10": int(group["common_months"]),
                "reason": group["reason"],
                "measurement_methods": group["measurement_methods"],
                "rows": rows,
                "diagnostics": diagnostics,
                "measurement": measurement,
                "sensitivity": sensitivity,
                "continuity": continuity,
                "identity": identity,
                "freeze_at_2024": all(item["months_2024"] == 12 for item in continuity),
                "conflict_examples": [row for row in rows if row["source_conflict"]],
                "zero_examples": [row for row in rows if row["exclusion_reason"] == "all_zero_group_total"],
            }
        )
    return panel, summaries


def _visit_lookup(visits: pd.DataFrame) -> dict[tuple[str, str], float]:
    scoped = visits[visits["observation_status"].isin(["reported", "reported_zero"])]
    lookup = {}
    for row in scoped[["site_id", "year_month", "year", "visits"]].to_dict(orient="records"):
        if int(row["year"]) >= HOLDOUT_YEAR or row["visits"] is None or pd.isna(row["visits"]):
            continue
        lookup[(str(row["site_id"]), str(row["year_month"]))] = float(row["visits"])
    return lookup


def _conflicts(evidence: pd.DataFrame) -> tuple[set[tuple[str, str]], dict[tuple[str, str], dict[str, object]]]:
    conflicts: set[tuple[str, str]] = set()
    gaps: dict[tuple[str, str], dict[str, object]] = {}
    for row in evidence.to_dict(orient="records"):
        if row.get("source_class") != "B_november_december_conflict":
            continue
        site_id = str(row["raw_site_id"])
        for month_number, csv_key, excel_key in (
            ("11", "csv_november", "excel_november"),
            ("12", "csv_december", "excel_december"),
        ):
            year_month = f"2021-{month_number}"
            conflicts.add((site_id, year_month))
            gaps[(site_id, year_month)] = {
                "official_name": row.get("official_name", ""),
                "csv": row.get(csv_key, ""),
                "excel": row.get(excel_key, ""),
            }
    return conflicts, gaps


def _labels(verified: pd.DataFrame) -> dict[tuple[str, str, int], set[str]]:
    found: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    for row in verified.to_dict(orient="records"):
        found[(comparison_key(row["official_name"]), comparison_key(row["county"]), int(row["year"]))].add(str(row["measurement_method"]))
    return found


def _names(crosswalk: pd.DataFrame) -> dict[str, str]:
    chosen: dict[str, tuple[int, str]] = {}
    for row in crosswalk.to_dict(orient="records"):
        uid = str(row["site_uid"])
        last_year = int(row["last_year"] or 0)
        if uid not in chosen or last_year >= chosen[uid][0]:
            chosen[uid] = (last_year, str(row["official_name"]))
    return {uid: name for uid, (_year, name) in chosen.items()}


def _uids(members: list[dict[str, str]], crosswalk: pd.DataFrame) -> tuple[list[str], dict[str, set[str]], list[str]]:
    by_raw = {str(row["raw_site_id"]): row for row in crosswalk.to_dict(orient="records")}
    by_uid: dict[str, set[str]] = defaultdict(set)
    for row in crosswalk.to_dict(orient="records"):
        if row.get("match_status") == "provisional_rule_match":
            by_uid[str(row["site_uid"])].add(str(row["raw_site_id"]))
    ordered = []
    raw_ids: dict[str, set[str]] = {}
    statuses = []
    for member in members:
        info = by_raw.get(str(member["site_id"]), {})
        uid = str(info.get("site_uid") or member["site_id"])
        if uid not in raw_ids:
            ordered.append(uid)
        raw_ids.setdefault(uid, set()).update(by_uid.get(uid, set()))
        raw_ids[uid].add(str(member["site_id"]))
        statuses.append(str(info.get("match_status") or "needs_manual_review"))
    return ordered, raw_ids, statuses


def _uid_value(raw_ids: set[str], year_month: str, lookup, conflicts) -> tuple[float | None, bool, bool]:
    found = []
    conflict_hit = False
    for raw_id in raw_ids:
        if (raw_id, year_month) in conflicts:
            conflict_hit = True
        if (raw_id, year_month) in lookup:
            found.append(lookup[(raw_id, year_month)])
    if not found:
        return None, conflict_hit, False
    disagree = max(found) - min(found) > 1e-6
    return found[0], conflict_hit, disagree


def _gap_text(raw_ids: dict[str, set[str]], year_month: str, gaps) -> str:
    parts = []
    for ids in raw_ids.values():
        for raw_id in sorted(ids):
            item = gaps.get((raw_id, year_month))
            if not item:
                continue
            parts.append(f"{item['official_name']} CSV {item['csv']}／Excel {item['excel']}")
    return "；".join(parts)


def _reconcile(group: dict[str, str], rows: list[dict[str, object]], shares: pd.DataFrame) -> None:
    defined = {row["year_month"] for row in rows if row["share_defined"]}
    observed = set(shares.loc[shares["group_id"].eq(group["group_id"]), "year_month"])
    reported = sum(int(row["reported_months"]) for row in rows)
    if defined != observed:
        raise RuntimeError(f"{group['parent_area']} share months do not match the phase 10 share file")
    if reported != int(group["common_months"]):
        raise RuntimeError(f"{group['parent_area']} reported-month count does not match phase 10")


def _check_known_months(panel: list[dict[str, object]]) -> None:
    by_parent: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in panel:
        by_parent[str(row["parent_area"])].append(row)
    alishan = by_parent["阿里山國家風景區"]
    kinmen = by_parent["金門國家公園"]
    if sum(row["share_defined"] for row in alishan) != 45 or sum(row["reported_months"] for row in alishan) != 46:
        raise RuntimeError("Alishan month counts drifted")
    if sum(row["share_defined"] for row in kinmen) != 71 or sum(row["reported_months"] for row in kinmen) != 72:
        raise RuntimeError("Kinmen month counts drifted")
    zeros = {row["parent_area"] for row in panel if row["year_month"] == "2021-06" and row["exclusion_reason"] == "all_zero_group_total"}
    if "阿里山國家風景區" not in zeros or "金門國家公園" not in zeros:
        raise RuntimeError("2021-06 all-zero months were not retained")
    conflicts = {row["year_month"] for row in alishan if row["source_conflict"]}
    if conflicts != {"2021-11", "2021-12"}:
        raise RuntimeError("Alishan source-conflict months were not retained")


def _measurement(uids, raw_ids, crosswalk, labels, months) -> dict[str, object]:
    by_raw = {str(row["raw_site_id"]): row for row in crosswalk.to_dict(orient="records")}
    years = sorted({int(month[:4]) for month in months})
    covered = 0
    active = 0
    families: dict[str, int] = defaultdict(int)
    per_uid: dict[str, set[str]] = defaultdict(set)
    for uid in uids:
        for year in years:
            raw = _active_raw(raw_ids[uid], by_raw, year)
            if raw is None:
                continue
            active += 1
            found = labels.get((comparison_key(raw["official_name"]), comparison_key(raw["county"]), year), set())
            nonempty = {label for label in found if label.strip()}
            if nonempty:
                covered += 1
            for label in nonempty:
                per_uid[uid].add(label)
                families[method_family(label)] += 1
            if not nonempty:
                families["missing"] += 1
    changed = sum(1 for labels_for_uid in per_uid.values() if len(labels_for_uid) > 1)
    return {
        "active_uid_years": active,
        "labeled_uid_years": covered,
        "label_coverage": None if active == 0 else covered / active,
        "family_counts": dict(families),
        "uids_with_label_change": changed,
        "uid_labels": {uid: sorted(values) for uid, values in per_uid.items()},
    }


def _active_raw(raw_ids: set[str], by_raw: dict[str, dict[str, str]], year: int) -> dict[str, str] | None:
    chosen = None
    for raw_id in raw_ids:
        row = by_raw.get(raw_id)
        if row is None:
            continue
        if int(row["first_year"] or 0) <= year <= int(row["last_year"] or 0):
            chosen = row
    return chosen


def _sensitivity(rows: list[dict[str, object]]) -> dict[str, object]:
    defined = [row for row in rows if row["share_defined"] and row.get("normalized_hhi") is not None]
    baseline = _mean([float(row["normalized_hhi"]) for row in defined])
    shifts = {}
    for factor in HYPOTHETICAL_SCALE_FACTORS:
        deltas = []
        for row in defined:
            shares = dict(row["shares"])
            leader = str(row["leading_site_uid"])
            if not leader:
                continue
            visits = {uid: float(share) for uid, share in shares.items()}
            scaled = hypothetical_scaled_hhi(visits, leader, factor)
            if scaled is None:
                continue
            deltas.append(scaled - float(row["normalized_hhi"]))
        shifts[str(factor)] = _mean(deltas)
    shock = [row for row in defined if row["candidate_common_shock"]]
    outside = [row for row in defined if not row["candidate_common_shock"]]
    labeled_years = _years_with_complete_direct_labels(rows)
    labeled = [row for row in defined if int(row["year"]) in labeled_years]
    return {
        "baseline_mean_normalized_hhi": baseline,
        "hypothetical_leader_scale_mean_delta": shifts,
        "shock_window_mean": _mean([float(row["normalized_hhi"]) for row in shock]),
        "outside_shock_window_mean": _mean([float(row["normalized_hhi"]) for row in outside]),
        "shock_window_months": len(shock),
        "direct_label_years": sorted(labeled_years),
        "direct_label_year_mean": _mean([float(row["normalized_hhi"]) for row in labeled]),
        "direct_label_year_months": len(labeled),
    }


def _years_with_complete_direct_labels(rows: list[dict[str, object]]) -> set[int]:
    return set()


def _continuity(uids, raw_ids, crosswalk, lookup, names) -> list[dict[str, object]]:
    by_raw = {str(row["raw_site_id"]): row for row in crosswalk.to_dict(orient="records")}
    rows = []
    for uid in uids:
        months_2024 = {year_month for raw_id in raw_ids[uid] for (site_id, year_month) in lookup if site_id == raw_id and year_month.startswith("2024-")}
        versions = []
        seed = next(iter(raw_ids[uid]))
        seed_row = by_raw.get(seed, {})
        for row in crosswalk.to_dict(orient="records"):
            if row.get("official_name") == seed_row.get("official_name") and row.get("county") == seed_row.get("county"):
                versions.append(
                    {
                        "site_uid": row["site_uid"],
                        "category": row["category"],
                        "match_status": row["match_status"],
                        "first_year": row["first_year"],
                        "last_year": row["last_year"],
                        "review_note": row["review_note"],
                    }
                )
        own = [row for row in by_raw.values() if row.get("site_uid") == uid]
        last_year = max((int(row["last_year"] or 0) for row in own), default=0)
        rows.append(
            {
                "site_uid": uid,
                "official_name": names.get(uid, ""),
                "last_year": last_year,
                "months_2024": len(months_2024),
                "versions": versions,
                "needs_formal_identity_review": any(item["review_note"] for item in versions) or len({item["site_uid"] for item in versions}) > 1 or last_year < 2024,
            }
        )
    return rows


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _holdout_flag(paths: V4Paths) -> dict[str, object]:
    staging = paths.output_root / "staging"
    files = sorted(staging.glob("v4_visits_202*.parquet")) if staging.exists() else []
    years = []
    flags = []
    rows = 0
    for path in files:
        frame = pd.read_parquet(path, columns=["year", "used_for_group_selection"])
        rows += int(len(frame))
        years.extend(int(value) for value in frame["year"].dropna().unique())
        flags.extend(bool(value) for value in frame["used_for_group_selection"].dropna().unique())
    return {
        "rows": rows,
        "years": sorted(set(years)),
        "used_for_group_selection": any(flags),
        "visit_values_read": False,
    }


def _figures(panel: list[dict[str, object]], names: dict[str, str]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    folder = WORLD_ROOT / "figures" / "v4_exploratory"
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in panel:
        grouped[str(row["group_id"])].append(row)
    for group_id, rows in grouped.items():
        slug = _slug(str(rows[0]["parent_area"]))
        title = f"{rows[0]['county']}／{rows[0]['parent_area']}"
        written.append(_plot_hhi(plt, folder, slug, title, rows))
        written.append(_plot_shares(plt, folder, slug, title, rows, names))
        written.append(_plot_ranks(plt, folder, slug, title, rows, names))
    return written


def _slug(parent: str) -> str:
    if "玉山" in parent:
        return "yushan"
    if "阿里山" in parent:
        return "alishan"
    if "太魯閣" in parent:
        return "taroko"
    if "雪霸" in parent:
        return "sheipa"
    if "金門" in parent:
        return "kinmen"
    return "group"


def _plot_hhi(plt, folder: Path, slug: str, title: str, rows: list[dict[str, object]]) -> str:
    months = [str(row["year_month"]) for row in rows]
    hhi = [float(row["normalized_hhi"]) if row["share_defined"] and row["normalized_hhi"] is not None else math.nan for row in rows]
    totals = [float(row["group_total_visits"]) if row["share_defined"] and row["group_total_visits"] is not None else math.nan for row in rows]
    figure, (axis, strip) = plt.subplots(2, 1, sharex=True, figsize=(10.5, 4.8), gridspec_kw={"height_ratios": [4, 0.7]})
    axis.plot(months, hhi, color="#1f4e79", linewidth=1.2, label="標準化 HHI")
    twin = axis.twinx()
    twin.plot(months, totals, color="#b85c38", linewidth=1.0, alpha=0.8, label="群總人次")
    _shade_shock(axis, months)
    breaks = [index for index, row in enumerate(rows) if row["candidate_structural_break"]]
    if breaks:
        axis.scatter([months[index] for index in breaks], [hhi[index] for index in breaks], color="#6b3fa0", s=18, zorder=3, label="候選斷點")
    axis.set_ylabel("標準化 HHI")
    twin.set_ylabel("回報人次")
    axis.set_title(f"{title}｜探索性集中程度")
    axis.set_ylim(-0.05, 1.05)
    colors = [_status_color(row) for row in rows]
    strip.bar(months, [1] * len(months), color=colors, width=1.0)
    strip.set_yticks([])
    strip.set_ylabel("月份")
    _sparse_ticks(axis, months)
    figure.tight_layout()
    path = folder / f"{slug}_hhi_total.png"
    figure.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return str(path.relative_to(WORLD_ROOT)).replace("\\", "/")


def _plot_shares(plt, folder: Path, slug: str, title: str, rows: list[dict[str, object]], names: dict[str, str]) -> str:
    defined = [row for row in rows if row["share_defined"]]
    uids = _site_order(defined)
    names = {**names, **_name_lookup(rows)}
    months = [str(row["year_month"]) for row in defined]
    stacks = [[float(dict(row["shares"]).get(uid, 0.0)) for row in defined] for uid in uids]
    figure, axis = plt.subplots(figsize=(10.5, 4.8))
    axis.stackplot(range(len(months)), stacks, labels=[names.get(uid, uid) for uid in uids])
    axis.set_ylim(0, 1)
    axis.set_ylabel("條件式回報份額")
    axis.set_title(f"{title}｜探索性份額")
    axis.set_xticks(_tick_positions(months))
    axis.set_xticklabels([months[index] for index in _tick_positions(months)], rotation=45, ha="right")
    axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    figure.tight_layout()
    path = folder / f"{slug}_shares.png"
    figure.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return str(path.relative_to(WORLD_ROOT)).replace("\\", "/")


def _plot_ranks(plt, folder: Path, slug: str, title: str, rows: list[dict[str, object]], names: dict[str, str]) -> str:
    defined = [row for row in rows if row["share_defined"]]
    uids = _site_order(defined)
    names = {**names, **_name_lookup(rows)}
    months = [str(row["year_month"]) for row in defined]
    figure, axis = plt.subplots(figsize=(10.5, 4.8))
    for uid in uids:
        ranks = []
        for row in defined:
            shares = dict(row["shares"])
            value = float(shares[uid])
            ranks.append(1 + sum(1 for share in shares.values() if float(share) > value + 1e-12))
        axis.plot(range(len(months)), ranks, linewidth=1.1, label=names.get(uid, uid))
    axis.invert_yaxis()
    axis.set_ylabel("名次（1 為最高份額）")
    axis.set_title(f"{title}｜探索性名次")
    axis.set_xticks(_tick_positions(months))
    axis.set_xticklabels([months[index] for index in _tick_positions(months)], rotation=45, ha="right")
    axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    figure.tight_layout()
    path = folder / f"{slug}_ranks.png"
    figure.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return str(path.relative_to(WORLD_ROOT)).replace("\\", "/")


def _site_order(rows: list[dict[str, object]]) -> list[str]:
    totals: dict[str, float] = defaultdict(float)
    for row in rows:
        for uid, share in dict(row["shares"]).items():
            totals[uid] += float(share)
    return sorted(totals, key=lambda uid: (-totals[uid], uid))


def _name_lookup(rows: list[dict[str, object]]) -> dict[str, str]:
    names = {}
    for row in rows:
        if row.get("leading_site_uid") and row.get("leading_site_name"):
            names[str(row["leading_site_uid"])] = str(row["leading_site_name"])
    return names


def _status_color(row: dict[str, object]) -> str:
    if row["share_defined"]:
        return "#1f4e79"
    if row["exclusion_reason"] == "all_zero_group_total":
        return "#c4a35a"
    if row["source_conflict"]:
        return "#9c2f2f"
    return "#b0b0b0"


def _shade_shock(axis, months: list[str]) -> None:
    indexes = [index for index, month in enumerate(months) if candidate_common_shock(month)]
    if not indexes:
        return
    axis.axvspan(min(indexes) - 0.5, max(indexes) + 0.5, color="#f2d6a2", alpha=0.35, label="潛在共同衝擊", zorder=0)


def _sparse_ticks(axis, months: list[str]) -> None:
    positions = _tick_positions(months)
    axis.set_xticks(positions)
    axis.set_xticklabels([months[index] for index in positions], rotation=45, ha="right")


def _tick_positions(months: list[str]) -> list[int]:
    if not months:
        return []
    step = 12 if len(months) > 36 else 6
    positions = list(range(0, len(months), step))
    if positions[-1] != len(months) - 1:
        positions.append(len(months) - 1)
    return positions


def _write_reports(paths: V4Paths, summaries: list[dict[str, object]], figures: list[str], holdout: dict[str, object], panel: list[dict[str, object]]) -> None:
    (paths.reports_root / "V4_PHASE11_CONCENTRATION_EXPLORATION.md").write_text(_concentration_text(summaries, figures), encoding="utf-8")
    (paths.reports_root / "V4_PHASE11_MEASUREMENT_SENSITIVITY.md").write_text(_sensitivity_text(summaries), encoding="utf-8")
    (paths.reports_root / "V4_PHASE11_FORECAST_FEASIBILITY.md").write_text(_forecast_text(summaries, holdout), encoding="utf-8")
    (paths.reports_root / "V4_PHASE11_ABM_OBSERVATION_BRIDGE.md").write_text(_bridge_text(summaries), encoding="utf-8")
    excluded = sum(1 for row in panel if not row["share_defined"])
    if excluded < 1:
        raise RuntimeError("exclusion list is empty")


def _concentration_text(summaries: list[dict[str, object]], figures: list[str]) -> str:
    lines = [
        "# V4 Phase 11｜探索性集中動態",
        "",
        "本輪是 exploratory。它描述五個 B 級地理群的官方回報人次份額，不是正式 P4 驗證，沒有擬合預測模型，也沒有估計社會資訊的因果效果。",
        "",
        "月份沒有靜默刪除。全群零人次與 2021 年 11–12 月的來源衝突都留在 `data/curated/v4_month_validity.csv`。份額可定義時，HHI 與份額向量寫入 `data/curated/v4_monthly_concentration_conditional.parquet`。全群零人次的 HHI 是空值。",
        "",
        "相鄰月份份額向量的總變異距離是 0.5 乘上份額差的絕對值和。它描述觀察到的回報份額移動了多少。WorldLab 的 Q 是同一模型狀態下有社會資訊與沒有社會資訊的反事實選擇機率差，不能由這些人次算出。",
        "",
        "2020-01 至 2022-12 標成潛在共同衝擊。這個區間沒有對到某一條可追溯的閉館、災害或政策日期，月份仍留在序列裡。標準化 HHI 的相鄰月絕對變化達到 0.10 時，記為候選結構斷點，同樣沒有對到外部事件日期。",
        "",
        "## 月份資格",
        "",
        "| 地理群 | J | 窗口 | 共同回報月 | 正人次月 | 份額可定義 | 排除 |",
        "| --- | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for item in summaries:
        rows = item["rows"]
        lines.append(
            f"| {item['county']}／{item['parent_area']} | {rows[0]['expected_sites']} | {item['common_from']}–{item['common_until']} | {sum(int(row['reported_months']) for row in rows)} | {sum(int(row['positive_total_months']) for row in rows)} | {sum(int(row['share_defined']) for row in rows)} | {_exclusion_brief(rows)} |"
        )
    lines.extend(["", "阿里山的共同回報月是 46，其中 2021-06 全群回報 0，份額可定義的是其餘 45 個月。2021-11 與 2021-12 兩個來源的人次不同，列在排除清單，沒有選邊。", ""])
    for row in next(item["conflict_examples"] for item in summaries if item["parent_area"] == "阿里山國家風景區"):
        lines.append(f"- {row['year_month']}：{row['conflict_detail']}")
    lines.extend(
        [
            "",
            "上面的 11 月數字裡，觸口遊客中心的 Excel 值等於梅山太平雲梯的 CSV 值，達娜伊谷的 Excel 值等於觸口遊客中心的 CSV 值。12 月也有同樣的相鄰對齊。這只是兩個來源並列之後看得到的數字關係，沒有被拿來改寫人次，也沒有選定哪一個版本。",
            "",
            "金門的共同回報月是 72。2021-06 全群回報 0，份額可定義的是 71 個月。",
            "",
            "## 集中動態",
            "",
        ]
    )
    lines.append("| 地理群 | J | 標準化 HHI 平均 | 最小 | 最大 | 標準差 | 前 12 月 | 後 12 月 | 平均 TV | 領先更替 | HHI 落後相關 | 人次變動與 HHI 變動相關 | 季節振幅 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for item in summaries:
        stats = item["diagnostics"]
        rows = item["rows"]
        lines.append(
            "| {name} | {j} | {mean} | {low} | {high} | {sd} | {first} | {last} | {tv} | {turn} | {lag} | {move} | {season} |".format(
                name=f"{item['county']}／{item['parent_area']}",
                j=rows[0]["expected_sites"],
                mean=_fmt(stats["normalized_hhi_mean"]),
                low=_fmt(stats["normalized_hhi_min"]),
                high=_fmt(stats["normalized_hhi_max"]),
                sd=_fmt(stats["normalized_hhi_sd"]),
                first=_fmt(stats["first_12_mean"]),
                last=_fmt(stats["last_12_mean"]),
                tv=_fmt(stats["mean_tv"]),
                turn=f"{stats['turnover_count']}/{stats['turnover_resolved_pairs']}",
                lag=_fmt(stats["hhi_lag1_correlation"]),
                move=_fmt(stats["total_hhi_change_correlation"]),
                season=_fmt(stats["seasonal_amplitude"]),
            )
        )
    lines.extend(
        [
            "",
            "跨群比較要同時看 J、標準化 HHI、觀察窗口和計數限制。原始 HHI 會隨 J 變小，不能直接把金門的原始 HHI 和玉山的原始 HHI 當成同一種集中程度。",
            "",
            "標準化 HHI 的月度標準差大於 0 的群，月份之間看得到集中程度起伏。金門的標準差是 0.014，水準幾乎不動。玉山、阿里山、太魯閣、雪霸的標準差在 0.110 到 0.183。前 12 個月與後 12 個月的平均如果分開，代表窗口兩端的水準不同。這些數字是描述。落後相關與領先景點的連續出現，是份額序列的持續性，不是社會學習或累積優勢的因果證據。",
            "",
            "相鄰月標準化 HHI 絕對變化達到 0.10 的次數：",
            "",
        ]
    )
    for item in summaries:
        stats = item["diagnostics"]
        rows = item["rows"]
        leaders = {}
        for row in rows:
            if row["share_defined"] and row["leading_site_name"]:
                leaders[str(row["leading_site_name"])] = leaders.get(str(row["leading_site_name"]), 0) + 1
        leader = max(leaders, key=leaders.get) if leaders else ""
        lines.append(
            f"- {item['county']}／{item['parent_area']}：{stats['candidate_break_count']} 個月。出現最多次的領先名稱是{leader}（{leaders.get(leader, 0)} 個月）。"
        )
    lines.extend(
        [
            "",
            "雪霸與玉山跨過 0.10 的月份很多，因為這兩條序列本身起伏大。這些月份是候選標記，不是已核對的政策、閉館或災害日期。",
            "",
            "人次變動與 HHI 變動的相關，用的是相鄰可定義月份的對數總人次差，以及標準化 HHI 差。係數接近 0 表示兩種變動沒有綁在一起。雪霸是 -0.498，總人次上升時標準化 HHI 較常下降，兩者仍不是同一個序列。",
            "",
            "## 圖",
            "",
        ]
    )
    for figure in figures:
        lines.append(f"- `{figure}`")
    lines.append("")
    return "\n".join(lines)


def _exclusion_brief(rows: list[dict[str, object]]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        if row["exclusion_reason"]:
            counts[str(row["exclusion_reason"])] += 1
    if not counts:
        return "無"
    return "、".join(f"{reason} {count}" for reason, count in sorted(counts.items()))


def _sensitivity_text(summaries: list[dict[str, object]]) -> str:
    lines = [
        "# V4 Phase 11｜測量敏感度",
        "",
        "五個群維持 B 級 reported_visits_conditional。相同的方法名稱沒有升成 documented_comparable。下面的倍率是事前指定的假設情境，用來觀察 HHI 會移動多少，不是估計出來的校正係數，也不能把調整後的 HHI 當成真實人次。",
        "",
        "Phase 10 的 identity_confidence 若顯示 1.0，只表示這次規則比對有覆蓋到群內成員。這裡改記 rule_match_coverage。外部身分驗證沒有做，externally_verified_identity_status 維持 unknown。",
        "",
        "| 地理群 | 規則覆蓋 | 外部驗證 | 標籤覆蓋 | 標籤變更的 uid | 版本衝突月 | 標準化 HHI 平均 | 領先據點 ×0.8 的平均變動 | ×1.2 的平均變動 | 衝擊窗平均 | 窗外平均 |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summaries:
        measurement = item["measurement"]
        sensitivity = item["sensitivity"]
        rows = item["rows"]
        conflict_rate = sum(1 for row in rows if row["source_conflict"]) / len(rows)
        shifts = sensitivity["hypothetical_leader_scale_mean_delta"]
        lines.append(
            "| {name} | {cover} | unknown | {labels} | {changed} | {conflict} | {base} | {low} | {high} | {shock} | {outside} |".format(
                name=f"{item['county']}／{item['parent_area']}",
                cover=_fmt(item["identity"]["rule_match_coverage"]),
                labels=_fmt(measurement["label_coverage"]),
                changed=measurement["uids_with_label_change"],
                conflict=_fmt(conflict_rate),
                base=_fmt(sensitivity["baseline_mean_normalized_hhi"]),
                low=_fmt(shifts.get("0.8")),
                high=_fmt(shifts.get("1.2")),
                shock=_fmt(sensitivity["shock_window_mean"]),
                outside=_fmt(sensitivity["outside_shock_window_mean"]),
            )
        )
    lines.extend(["", "## 各群計數標籤", ""])
    for item in summaries:
        measurement = item["measurement"]
        lines.append(f"### {item['county']}／{item['parent_area']}")
        lines.append("")
        lines.append(f"Phase 10 記錄的方法欄是 {item['measurement_methods'] or '（空白）'}。可歸類次數：{measurement['family_counts']}。")
        lines.append("")
        lines.append(item["reason"])
        lines.append("")
        lines.append("假設情境只縮放該月領先據點的人次，再重算標準化 HHI。0.8 與 1.2 是事先指定的倍率。")
        lines.append("")
    lines.append("來源衝突月份沒有併入 HHI。若要看衝突版本會把集中程度改成什麼，必須先選定 CSV 或 Excel，這一輪不選。")
    lines.append("")
    return "\n".join(lines)


def _forecast_text(summaries: list[dict[str, object]], holdout: dict[str, object]) -> str:
    lines = [
        "# V4 Phase 11｜預測可行性",
        "",
        "這一輪沒有擬合模型。2025 與 2026 的人次、HHI 與模型表現都沒有用來挑選競爭群。staging 只核對年份與 used_for_group_selection。",
        "",
        f"staging 列數 {holdout['rows']}，年份 {holdout['years']}，used_for_group_selection 為 {holdout['used_for_group_selection']}。人次欄位沒有讀入。",
        "",
        "回溯窗口的長度檢查使用事先指定的 24 個訓練月加 12 個回測月，而且每個月都要是日曆上相鄰、份額可定義的月份。下面的起點數只表示窗口內放得下這樣的區段，不是模型結果。",
        "",
        "| 地理群 | 份額可定義月 | 2024 年底可凍結 | 回溯區段起點數 | 需要正式身分確認的 uid |",
        "| --- | ---: | --- | ---: | ---: |",
    ]
    for item in summaries:
        review = sum(1 for site in item["continuity"] if site["needs_formal_identity_review"])
        lines.append(
            f"| {item['county']}／{item['parent_area']} | {item['diagnostics']['defined_months']} | {'可以' if item['freeze_at_2024'] else '不可以'} | {item['diagnostics']['retrospective_origins']} | {review}/{len(item['continuity'])} |"
        )
    lines.extend(["", "## 2024 連續性", ""])
    for item in summaries:
        lines.append(f"### {item['county']}／{item['parent_area']}")
        lines.append("")
        for site in item["continuity"]:
            other = [version for version in site["versions"] if version["site_uid"] != site["site_uid"]]
            other_text = "；".join(
                f"{version['category']} {version['first_year']}–{version['last_year']} {version['match_status']}" for version in other
            ) or "沒有另一個同名 uid"
            lines.append(
                f"- {site['official_name']}（{site['site_uid']}）規則鏈最後年到 {site['last_year']}，2024 年有 {site['months_2024']} 個月。同名其他版本：{other_text}。"
            )
        lines.append("")
    lines.extend(
        [
            "五個群目前都不能在 2024 年底凍結成完整的共同回報集合。玉山的探索窗口停在 2021-12。阿里山、太魯閣、雪霸、金門的共同窗口停在 2023-12。2024 若出現在另一個類型或另一個 site_uid，只能列為待確認的身分，不能直接接上。",
            "",
            "2025 能不能當獨立測試集，要等群組集合、資料規則和模型選擇都先凍結。這一輪沒有做這三件事。",
            "",
        ]
    )
    return "\n".join(lines)


def _bridge_text(summaries: list[dict[str, object]]) -> str:
    lines = [
        "# V4 Phase 11｜模擬與觀察的概念對應",
        "",
        "這裡只寫 adapter 的邊界，沒有修改 ABM 引擎，也沒有把模擬步數換成月份。",
        "",
        "## 三個不同的量",
        "",
        "1. 模擬內的代理人選擇份額，是模型在一個決策步上的選擇結果或佔用比例。",
        "2. 真實景點群的條件式人次份額，是 B 級官方回報人次在該地理群內的比例。一個人同一個月可以出現在多個據點，所以它不是選擇機率。",
        "3. 景點群每月總需求，是該群回報人次的合計。份額模型不產生這個合計。",
        "",
        "## 對應時要分開處理的事",
        "",
        "模擬的 500 步或 2000 步是有限決策序列的長度。真實資料的一列是一個西元年月。兩者要對上，必須先寫明哪一段步數聚成一個月，而且這個規則要在看到結果之前定下來。本輪沒有這種對應。",
        "",
        "不同 J 的原始 HHI 不能直接比較。標準化 HHI 把均勻分布放到 0、單一據點放到 1，比較時仍要同時寫出 J、窗口和計數限制。本輪五群的 J 是：",
        "",
    ]
    for item in summaries:
        lines.append(f"- {item['county']}／{item['parent_area']}：J = {item['rows'][0]['expected_sites']}，窗口 {item['common_from']}–{item['common_until']}。")
    lines.extend(
        [
            "",
            "全群零人次的月份，份額與 HHI 都沒有定義。adapter 應把該月標成無定義，不能填 0，也不能改成均勻分布。阿里山與金門的 2021-06 就是這種月份。",
            "",
            "非 LLM 的統計基準若要和之後的模擬比較，應使用同一組凍結的 site_uid、同一批份額可定義的月份，以及份額向量本身的誤差。只比 HHI 會把不同的份額形狀看成一樣。基準的季節平均、去年同月或近期份額，都要在凍結之後才擬合。本輪沒有擬合。",
            "",
            "若未來要預測絕對人次，需要另外一個總需求模型，再乘上份額。模擬裡的代理人數不能直接當成遊客人次。",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt(value: object) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return ""
    return f"{number:.3f}"


def _hashes() -> dict[str, str]:
    return {relative: file_sha256(WORLD_ROOT / relative) for relative in GUARDED}


if __name__ == "__main__":
    print(json.dumps(run_phase11(), ensure_ascii=False, indent=2))
