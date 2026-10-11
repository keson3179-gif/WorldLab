from __future__ import annotations

import pandas as pd

from worldlab_data.hhi import herfindahl, normalized_herfindahl, shares_from_visits

MIN_SITES = 5
MAX_SITES = 8
MIN_COMMON_MONTHS = 24
MISSING_PARENTS = {"", "-", "—", "–", "無", "na", "n/a", "none"}


def candidate_groups(history: pd.DataFrame, visits: pd.DataFrame) -> pd.DataFrame:
    """Size and place can make a candidate. They do not prove substitution or a shared counting method."""
    if history.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    coverage = _month_sets(visits)
    for (county, parent), group in history.groupby(["county", "parent_area"]):
        members = group.to_dict(orient="records")
        if str(parent).strip().casefold() in MISSING_PARENTS:
            definition = f"{county} 的列沒有可用的觀光遊憩區名稱。"
            for member in members:
                rows.append(_row(member, "", definition, False, "來源的觀光遊憩區是空白或「-」，不與同縣其他缺區名據點合成一群"))
            continue
        common = None
        for member in members:
            months = coverage.get(str(member["site_id"]), set())
            common = months if common is None else common & months
        common = common or set()
        definition = f"{county}／{parent} 的統計細分據點。同一遊憩區不代表彼此可替代，計數方式也尚未確認。"
        too_small = len(members) < MIN_SITES
        too_large = len(members) > MAX_SITES
        short_overlap = len(common) < MIN_COMMON_MONTHS
        if too_small or too_large or short_overlap:
            reason = []
            if too_small:
                reason.append("少於 5 個據點，不併入其他區域來湊數")
            if too_large:
                reason.append("多於 8 個據點，不依任何結果挑選子集")
            if short_overlap:
                reason.append(f"共同有回報的月份 {len(common)}，少於 {MIN_COMMON_MONTHS}")
            for member in members:
                rows.append(_row(member, "", definition, False, "；".join(reason)))
            continue
        group_id = f"cand_{_slug(str(county))}_{_slug(str(parent))}"
        included_from = min(common) if common else ""
        included_until = max(common) if common else ""
        for member in members:
            rows.append(
                _row(
                    member,
                    group_id,
                    definition,
                    True,
                    "",
                    included_from,
                    included_until,
                    "same_region_substitute_unverified",
                )
            )
    return pd.DataFrame.from_records(rows)


def _row(
    member: dict[str, object],
    group_id: str,
    definition: str,
    included: bool,
    exclusion_reason: str,
    included_from: str = "",
    included_until: str = "",
    relationship: str = "not_grouped",
) -> dict[str, object]:
    return {
        "competition_group_id": group_id,
        "site_id": member["site_id"],
        "official_name": member.get("official_name", ""),
        "county": member.get("county", ""),
        "parent_area": member.get("parent_area", ""),
        "group_definition": definition,
        "included_from": included_from or member.get("first_year_month", ""),
        "included_until": included_until or member.get("last_year_month", ""),
        "included": included,
        "relationship": relationship if included else "not_grouped",
        "eligibility_reason": "同縣市且同屬一個觀光遊憩區，規模落在 5 到 8，共同月份足夠；替代關係與計數方式都未確認"
        if included
        else "",
        "exclusion_reason": exclusion_reason,
        "measurement_comparability": "unknown",
        "backtest_eligible": False,
    }


def _month_sets(visits: pd.DataFrame) -> dict[str, set[str]]:
    if visits.empty:
        return {}
    usable = visits[visits["observation_status"].isin(["reported", "reported_zero"])]
    result: dict[str, set[str]] = {}
    for site_id, group in usable.groupby("site_id"):
        result[str(site_id)] = set(group["year_month"].astype(str))
    return result


def _slug(value: str) -> str:
    return "".join(character for character in value if character.isalnum())[:24]


def build_panel(visits: pd.DataFrame, membership: pd.DataFrame, require_backtest: bool) -> pd.DataFrame:
    columns = [
        "competition_group_id",
        "site_id",
        "year_month",
        "visits",
        "group_visit_total",
        "visit_share",
        "group_hhi",
        "normalized_hhi",
        "leading_site_id",
        "site_rank",
        "lag_visits",
        "lag_share",
        "measurement_method",
        "publication_history_unknown",
        "available_at",
        "strict_feature_eligible",
        "share_meaning",
        "backtest_eligible",
    ]
    if membership.empty or visits.empty:
        return pd.DataFrame(columns=columns)
    selected = membership[membership["included"] == True].copy()  # noqa: E712
    if require_backtest:
        selected = selected[selected["backtest_eligible"] == True]  # noqa: E712
    if selected.empty:
        return pd.DataFrame(columns=columns)
    usable = visits[visits["observation_status"].isin(["reported", "reported_zero"])][
        ["site_id", "year_month", "visits", "measurement_method", "publication_history_unknown", "available_at"]
    ]
    rows: list[dict[str, object]] = []
    for group_id, members in selected.groupby("competition_group_id"):
        member_rows = members.drop_duplicates("site_id").sort_values("site_id")
        site_ids = [str(value) for value in member_rows["site_id"]]
        observed = usable[usable["site_id"].astype(str).isin(site_ids)]
        start = str(member_rows["included_from"].iloc[0]) if "included_from" in member_rows.columns else ""
        end = str(member_rows["included_until"].iloc[0]) if "included_until" in member_rows.columns else ""
        months = sorted(set(observed["year_month"].astype(str)))
        if start and end:
            months = [month for month in months if start <= month <= end]
        lookup = {
            (str(record["site_id"]), str(record["year_month"])): record
            for record in observed.to_dict(orient="records")
        }
        for year_month in months:
            records = []
            visit_values = []
            for site_id in site_ids:
                record = lookup.get((site_id, year_month))
                if record is None:
                    visit_values.append(None)
                    records.append({"site_id": site_id, "measurement_method": "unknown", "publication_history_unknown": True, "available_at": "", "backtest_eligible": False})
                else:
                    visit_values.append(float(record["visits"]))
                    records.append(record)
            share_values = shares_from_visits(visit_values)
            if share_values is None:
                hhi_value = None
                normalized = None
                leader = ""
                ranks = [None] * len(site_ids)
            else:
                hhi_value = herfindahl(share_values)
                normalized = normalized_herfindahl(hhi_value, len(share_values))
                leader_index = min(range(len(share_values)), key=lambda index: (-share_values[index], site_ids[index]))
                leader = site_ids[leader_index]
                order = sorted(range(len(share_values)), key=lambda index: (-share_values[index], site_ids[index]))
                ranks = [0] * len(share_values)
                for rank, index in enumerate(order, start=1):
                    ranks[index] = rank
            total = None if any(value is None for value in visit_values) else float(sum(visit_values))
            for position, record in enumerate(records):
                publication_unknown = bool(record.get("publication_history_unknown", True))
                available_at = "" if record.get("available_at") is None else str(record.get("available_at"))
                member_flag = bool(member_rows.iloc[position].get("backtest_eligible", False))
                rows.append(
                    {
                        "competition_group_id": group_id,
                        "site_id": site_ids[position],
                        "year_month": year_month,
                        "visits": visit_values[position],
                        "group_visit_total": total,
                        "visit_share": None if share_values is None else share_values[position],
                        "group_hhi": hhi_value,
                        "normalized_hhi": normalized,
                        "leading_site_id": leader,
                        "site_rank": ranks[position],
                        "lag_visits": None,
                        "lag_share": None,
                        "measurement_method": record.get("measurement_method", "unknown"),
                        "publication_history_unknown": publication_unknown,
                        "available_at": available_at,
                        "strict_feature_eligible": False,
                        "share_meaning": "據點回報人次分布，不是個人選擇機率",
                        "backtest_eligible": member_flag and not publication_unknown and available_at != "",
                    }
                )
    panel = pd.DataFrame.from_records(rows)
    if panel.empty:
        return pd.DataFrame(columns=columns)
    panel = panel.sort_values(["competition_group_id", "site_id", "year_month"])
    panel["lag_visits"] = panel.groupby(["competition_group_id", "site_id"])["visits"].shift(1)
    panel["lag_share"] = panel.groupby(["competition_group_id", "site_id"])["visit_share"].shift(1)
    panel.loc[panel["publication_history_unknown"] == True, "strict_feature_eligible"] = False  # noqa: E712
    return panel


def strict_features(panel: pd.DataFrame, origin: str) -> pd.DataFrame:
    """Keep only fields already known to be available at origin. Unknown publication dates drop out."""
    if panel.empty:
        return panel.copy()
    eligible = panel[panel["strict_feature_eligible"] == True].copy()  # noqa: E712
    if eligible.empty:
        return eligible
    dated = eligible[eligible["available_at"].astype(str) != ""]
    return dated[dated["available_at"].astype(str) <= origin]
