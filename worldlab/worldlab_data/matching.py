from __future__ import annotations

from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

from worldlab_data.textutil import comparison_key, nfkc


CONFIRMED = "confirmed_manual"
REVIEW = "needs_manual_review"


def candidate_matches(sites: pd.DataFrame, pois: pd.DataFrame, score_floor: float = 80) -> pd.DataFrame:
    """Propose name candidates inside the same county. Scores are not decisions."""
    rows: list[dict[str, object]] = []
    if sites.empty or pois.empty:
        return pd.DataFrame()
    poi_groups = {county: group for county, group in pois.groupby("county")}
    for site in sites.drop_duplicates("site_id").to_dict(orient="records"):
        county = nfkc(site.get("county"))
        site_key = comparison_key(site.get("official_name"))
        pool = poi_groups.get(county)
        if pool is None or not site_key:
            rows.append(_review_row(site, None, None, "no_same_county_candidate"))
            continue
        scored: list[tuple[float, dict[str, object]]] = []
        exact: list[dict[str, object]] = []
        for poi in pool.to_dict(orient="records"):
            poi_key = comparison_key(poi.get("official_name"))
            if not poi_key:
                continue
            if poi_key == site_key:
                exact.append(poi)
            else:
                scored.append((float(fuzz.WRatio(site_key, poi_key)), poi))
        if len(exact) == 1:
            rows.append(_review_row(site, exact[0], 100.0, "exact_unique_same_county"))
            continue
        if len(exact) > 1:
            for poi in exact:
                rows.append(_review_row(site, poi, 100.0, "exact_name_multiple_pois"))
            continue
        scored.sort(key=lambda item: item[0], reverse=True)
        kept = [item for item in scored if item[0] >= score_floor][:3]
        if not kept:
            rows.append(_review_row(site, None, None, "no_candidate_above_floor"))
            continue
        for score, poi in kept:
            rows.append(_review_row(site, poi, score, "fuzzy_candidate"))
    return pd.DataFrame.from_records(rows)


def _review_row(site: dict[str, object], poi: dict[str, object] | None, score: float | None, method: str) -> dict[str, object]:
    return {
        "site_id": site.get("site_id"),
        "official_name": site.get("official_name"),
        "county": site.get("county"),
        "parent_area": site.get("parent_area"),
        "candidate_poi_id": "" if poi is None else poi.get("poi_id"),
        "candidate_name": "" if poi is None else poi.get("official_name"),
        "candidate_district": "" if poi is None else poi.get("district"),
        "candidate_latitude": "" if poi is None else poi.get("latitude"),
        "candidate_longitude": "" if poi is None else poi.get("longitude"),
        "match_method": method,
        "match_score": score,
        "match_status": REVIEW,
        "match_origin": "candidate",
        "notes": "候選配對，尚未人工確認，座標不寫入景點主檔",
    }


def apply_manual_crosswalk(candidates: pd.DataFrame, manual_path: Path) -> pd.DataFrame:
    """A manual row wins over any higher fuzzy score for the same site."""
    if not manual_path.exists():
        return candidates.copy()
    manual = pd.read_csv(manual_path, dtype=str, keep_default_na=False)
    if manual.empty or "site_id" not in manual.columns:
        return candidates.copy()
    manual = manual.copy()
    manual["match_origin"] = "manual"
    if "match_status" not in manual.columns:
        manual["match_status"] = REVIEW
    locked = set(manual["site_id"].astype(str))
    if candidates.empty:
        return manual
    for column in candidates.columns:
        if column not in manual.columns:
            manual[column] = ""
    manual = manual[list(candidates.columns)]
    kept = candidates[~candidates["site_id"].astype(str).isin(locked)].copy()
    return pd.concat([manual, kept], ignore_index=True)


def attraction_master(sites: pd.DataFrame, crosswalk: pd.DataFrame) -> pd.DataFrame:
    unique_sites = sites.drop_duplicates("site_id").copy()
    confirmed = pd.DataFrame()
    if not crosswalk.empty and "match_status" in crosswalk.columns:
        confirmed = crosswalk[crosswalk["match_status"] == CONFIRMED].drop_duplicates("site_id")
    confirmed_by_site = {str(row["site_id"]): row for row in confirmed.to_dict(orient="records")} if not confirmed.empty else {}
    rows: list[dict[str, object]] = []
    for site in unique_sites.to_dict(orient="records"):
        locked = confirmed_by_site.get(str(site["site_id"]))
        rows.append(
            {
                "site_id": site["site_id"],
                "official_name": site["official_name"],
                "alternative_names": "" if locked is None else locked.get("candidate_name", ""),
                "county": site.get("county", ""),
                "district": "" if locked is None else locked.get("candidate_district", ""),
                "latitude": "" if locked is None else locked.get("candidate_latitude", ""),
                "longitude": "" if locked is None else locked.get("candidate_longitude", ""),
                "category": site.get("category", ""),
                "activity_type": "",
                "source_site_id": "",
                "source_name": site["official_name"],
                "parent_area": site.get("parent_area", ""),
                "match_status": REVIEW if locked is None else CONFIRMED,
                "match_confidence": None if locked is None else locked.get("match_score"),
                "measurement_method": "unknown",
            }
        )
    return pd.DataFrame.from_records(rows)
