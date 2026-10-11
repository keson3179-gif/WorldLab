"""Real-place cards for the observation adapter.

A 2026 attraction snapshot is not written back into 2014. Fuzzy name hits stay candidates.
"""

from __future__ import annotations

import hashlib
import math
import xml.etree.ElementTree as ET
from pathlib import Path

from worldlab_data.textutil import comparison_key, county_parts, nfkc
from worldlab_observation.adapter import (
    PHASE14_PARENTS,
    SocialSignal,
    decision_q,
    expected_share,
    logmax_scores,
    scale_tradeoff_same_probability,
    softmax_probabilities,
)

ATTRACTION_NS = "https://media.taiwan.net.tw/standard/schema/2.0"
TEMPORAL_HISTORICAL = "historically_documented"
TEMPORAL_STABLE_UNCONFIRMED = "stable_unconfirmed"
TEMPORAL_CURRENT = "current_snapshot_only"
TEMPORAL_NONE = "no_historical_validity"
STATUS_SOURCE = "source_reported"
STATUS_UNVERIFIED = "unverified"
STATUS_MISSING = "missing"
STATUS_VERIFIED = "verified"

NAME_STEMS = {
    "玉山管理處遊客服務中心": "玉山管理處",
    "塔塔加遊憩區": "塔塔加",
    "排雲山莊": "排雲",
    "太魯閣國家公園遊客中心": "太魯閣",
    "布洛灣遊憩區": "布洛灣",
    "臺八線沿線景觀區": "臺八線",
    "觀霧遊客中心": "觀霧",
    "雪見遊憩區": "雪見",
    "汶水遊客中心": "汶水",
}


def match_grade(official_name: str, county: str, candidate_name: str, candidate_county: str) -> str:
    if county_parts(candidate_county)[0] != county:
        return ""
    if comparison_key(official_name) == comparison_key(candidate_name):
        return "exact_name_county"
    stem = NAME_STEMS.get(official_name, "")
    if stem and stem in nfkc(candidate_name).replace("台", "臺"):
        return "fuzzy_candidate"
    return ""


def temporal_class_for(field: str, grade: str) -> str:
    if grade == "fuzzy_candidate":
        return TEMPORAL_NONE
    if grade != "exact_name_county":
        return TEMPORAL_NONE
    if field in {"latitude", "longitude"}:
        return TEMPORAL_STABLE_UNCONFIRMED
    return TEMPORAL_CURRENT


def field_status(value: str, grade: str) -> str:
    if grade == "fuzzy_candidate":
        return STATUS_UNVERIFIED
    if not value:
        return STATUS_MISSING
    if grade == "exact_name_county":
        return STATUS_SOURCE
    return STATUS_MISSING


def strict_historical_attributes(records: list[dict[str, str]]) -> list[dict[str, str]]:
    """Class 1 only. A current snapshot and an unconfirmed coordinate stay out."""
    return [
        record
        for record in records
        if record["temporal_class"] == TEMPORAL_HISTORICAL and record["status"] in {STATUS_VERIFIED, STATUS_SOURCE}
    ]


def m0_prediction_status(group_records: list[dict[str, str]]) -> dict[str, object]:
    """No differentiating historical attribute means the utility channel stays unavailable."""
    historical = strict_historical_attributes(group_records)
    by_field: dict[str, set[str]] = {}
    for record in historical:
        if record["field"] == "attraction_description":
            continue
        by_field.setdefault(record["field"], set()).add(record["value"])
    differentiating = [field for field, values in by_field.items() if len(values) > 1]
    if not differentiating:
        return {
            "model": "M0",
            "prediction_status": "prediction_unavailable",
            "skip_reason": "no_differentiating_historically_documented_attribute",
            "predicted_share": None,
            "constant_utility_used": False,
        }
    return {
        "model": "M0",
        "prediction_status": "prediction_unavailable",
        "skip_reason": "historical_attribute_present_but_not_a_utility_scale",
        "predicted_share": None,
        "constant_utility_used": False,
    }


def m1_prediction_status(m0: dict[str, object]) -> dict[str, object]:
    """Historical popularity is not renamed into an ABM forecast when the base utility is missing."""
    if m0["prediction_status"] != "predicted":
        return {
            "model": "M1",
            "prediction_status": "prediction_unavailable",
            "skip_reason": "base_utility_missing",
            "social_source": "reported_visit_share_conditional",
            "exposure_claim": False,
            "predicted_share": None,
        }
    return {
        "model": "M1",
        "prediction_status": "prediction_unavailable",
        "skip_reason": "not_enabled_this_round",
        "social_source": "reported_visit_share_conditional",
        "exposure_claim": False,
        "predicted_share": None,
    }


def reject_dual_use(attraction_source: str, crowding_source: str) -> None:
    if attraction_source and attraction_source == crowding_source:
        raise ValueError("the same popularity series cannot be both attraction and crowding")


def synthetic_attribute_share(attributes: list[list[float]], preference: list[float]) -> list[float]:
    """Synthetic places only. Identical attributes make preference weights drop out of the share."""
    if not attributes or any(len(row) != len(preference) for row in attributes):
        raise ValueError("attributes must align with the preference weights")
    utilities = [sum(weight * value for weight, value in zip(preference, row, strict=True)) for row in attributes]
    return softmax_probabilities(utilities)


def preference_invariance_when_attributes_match(attributes: list[list[float]], left: list[float], right: list[float]) -> bool:
    if any(len(set(column)) != 1 for column in zip(*attributes, strict=True)):
        return False
    return synthetic_attribute_share(attributes, left) == synthetic_attribute_share(attributes, right)


def haversine_km(left_lat: float, left_lon: float, right_lat: float, right_lon: float) -> float:
    radius = 6371.0
    phi1 = math.radians(left_lat)
    phi2 = math.radians(right_lat)
    d_phi = math.radians(right_lat - left_lat)
    d_lambda = math.radians(right_lon - left_lon)
    chord = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(chord))


def iter_attractions(path: Path) -> tuple[str, list[dict[str, str]]]:
    document_update = ""
    seen_attraction = False
    rows: list[dict[str, str]] = []
    tag = f"{{{ATTRACTION_NS}}}"
    for _event, element in ET.iterparse(path, events=("end",)):
        if element.tag == f"{tag}UpdateTime" and not seen_attraction and not document_update:
            document_update = nfkc(element.text)
        if element.tag != f"{tag}Attraction":
            continue
        seen_attraction = True
        name = _child_text(element, "AttractionName")
        county, _english = county_parts(_child_text(element, "PostalAddress/City"))
        stems = set(NAME_STEMS.values())
        if any(stem in name.replace("台", "臺") for stem in stems) or comparison_key(name) in {comparison_key(item) for item in NAME_STEMS}:
            rows.append(_attraction_record(element, name, county))
        element.clear()
    return document_update, rows


def provenance_row(
    site: dict[str, str],
    field: str,
    value: str,
    status: str,
    temporal_class: str,
    source_file: str,
    source_record_id: str,
    source_publication_date: str,
    match_grade_value: str,
    note: str,
    valid_from: str = "",
    valid_to: str = "",
) -> dict[str, str]:
    return {
        "group_id": site["group_id"],
        "site_uid": site["site_uid"],
        "official_name": site["official_name"],
        "county": site["county"],
        "field": field,
        "value": value,
        "value_sha256": hashlib.sha256(value.encode("utf-8")).hexdigest() if value else "",
        "status": status,
        "temporal_class": temporal_class,
        "source_file": source_file,
        "source_record_id": source_record_id,
        "source_publication_date": source_publication_date,
        "match_grade": match_grade_value,
        "valid_from": valid_from,
        "valid_to": valid_to,
        "note": note,
    }


def statistical_category_row(site: dict[str, str], category: str, first_year: str, last_year: str) -> dict[str, str]:
    return provenance_row(
        site,
        "official_category",
        category,
        STATUS_SOURCE if category else STATUS_MISSING,
        TEMPORAL_HISTORICAL if category else TEMPORAL_NONE,
        "歷年國內主要觀光遊憩據點遊客人數月別統計.csv",
        site["site_uid"],
        "",
        "statistical_chain",
        "類別來自該 site_uid 的統計鏈。公布日無法重建。2024 的另一個類型列沒有併入。",
        f"{first_year}-01" if first_year else "",
        f"{last_year}-12" if last_year else "",
    )


def snapshot_rows(site: dict[str, str], attraction: dict[str, str] | None, grade: str) -> list[dict[str, str]]:
    fields = {
        "latitude": "" if attraction is None else attraction["latitude"],
        "longitude": "" if attraction is None else attraction["longitude"],
        "activity_tags": "" if attraction is None else attraction["activity_tags"],
        "attraction_description": "" if attraction is None else attraction["description"],
        "opening_or_access_conditions": "" if attraction is None else attraction["service_time"],
        "admission_or_cost_information": "" if attraction is None else attraction["fee_info"],
        "available_facilities": "" if attraction is None else attraction["facilities"],
    }
    source_file = "" if attraction is None else attraction["source_file"]
    record_id = "" if attraction is None else attraction["attraction_id"]
    published = "" if attraction is None else attraction["record_update"][:10]
    rows = []
    for field, value in fields.items():
        status = field_status(value, grade)
        temporal = temporal_class_for(field, grade) if status != STATUS_MISSING else TEMPORAL_NONE
        if status == STATUS_MISSING:
            temporal = TEMPORAL_NONE
        note = ""
        if grade == "fuzzy_candidate":
            note = "模糊名稱只留候選，不寫進場所卡的來源值。"
            value = ""
            status = STATUS_MISSING
            temporal = TEMPORAL_NONE
        elif status == STATUS_SOURCE and temporal == TEMPORAL_CURRENT:
            note = "內容日期是景點資料庫該筆的更新日，不能回填評估年份。"
        elif status == STATUS_SOURCE and temporal == TEMPORAL_STABLE_UNCONFIRMED:
            note = "座標來自同名同縣市的當期資料庫。位置可能穩定，但沒有 2014 年的獨立座標佐證。"
        rows.append(
            provenance_row(
                site,
                field,
                "" if status == STATUS_MISSING else value,
                status,
                temporal,
                source_file if status == STATUS_SOURCE else "",
                record_id if status == STATUS_SOURCE else "",
                published if status == STATUS_SOURCE else "",
                grade if status == STATUS_SOURCE else "",
                note,
            )
        )
    return rows


def accessibility_row(
    site: dict[str, str],
    candidate_name: str,
    candidate_county: str,
    grade: str,
    latitude: str,
    longitude: str,
    distance_km: str,
    distance_kind: str,
    tdx_status: str,
    note: str,
) -> dict[str, str]:
    return {
        "group_id": site["group_id"],
        "site_uid": site["site_uid"],
        "official_name": site["official_name"],
        "candidate_name": candidate_name,
        "candidate_county": candidate_county,
        "match_grade": grade,
        "latitude": latitude if grade == "exact_name_county" else "",
        "longitude": longitude if grade == "exact_name_county" else "",
        "candidate_latitude": latitude,
        "candidate_longitude": longitude,
        "distance_km": distance_km,
        "distance_kind": distance_kind,
        "temporal_class": temporal_class_for("latitude", grade) if latitude and grade == "exact_name_county" else TEMPORAL_NONE,
        "verification_status": STATUS_SOURCE if grade == "exact_name_county" and latitude else STATUS_UNVERIFIED if grade == "fuzzy_candidate" else STATUS_MISSING,
        "tdx_status": tdx_status,
        "note": note,
    }


def social_signal_from_counts(place_ids: tuple[str, ...], counts: list[float]) -> SocialSignal:
    return SocialSignal(place_ids, logmax_scores(counts), "reported_visit_share_conditional")


def identifiability_demo() -> dict[str, object]:
    base = [1.0, 2.0, 0.5]
    scores = [0.0, 0.4, 1.0]
    same = scale_tradeoff_same_probability(base, scores, 8.0)
    q_value = decision_q(base, scores, 8.0)
    identical = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
    invariant = preference_invariance_when_attributes_match(identical, [1.0, 0.0], [0.2, 3.0])
    varied = synthetic_attribute_share([[0.2, 0.8], [0.9, 0.1], [0.4, 0.4]], [1.0, -1.0])
    return {
        "weight_scale_tradeoff": same,
        "q_is_probability_distance": q_value > 0.0,
        "identical_attributes_drop_preference": invariant,
        "synthetic_share_is_simplex": abs(sum(varied) - 1.0) <= 1e-12,
        "expected_share_matches_choice_probability_only_under_the_mapping_assumption": expected_share([varied]) == varied,
    }


def _attraction_record(element: ET.Element, name: str, county: str) -> dict[str, str]:
    classes = [nfkc(node.text) for node in element.findall(f"{{{ATTRACTION_NS}}}AttractionClasses/{{{ATTRACTION_NS}}}AttractionClass") if node.text]
    tags = [nfkc(node.text) for node in element.findall(f"{{{ATTRACTION_NS}}}Tags/{{{ATTRACTION_NS}}}Tag") if node.text]
    return {
        "attraction_id": _child_text(element, "AttractionID"),
        "name": name,
        "county": county,
        "latitude": _child_text(element, "PositionLat"),
        "longitude": _child_text(element, "PositionLon"),
        "description": _child_text(element, "Description"),
        "service_time": _child_text(element, "ServiceTimeInfo"),
        "traffic_info": _child_text(element, "TrafficInfo"),
        "facilities": _child_text(element, "Facilities"),
        "fee_info": _child_text(element, "FeeInfo"),
        "free_code": _child_text(element, "IsAccessibleForFree"),
        "activity_tags": "|".join(classes + tags),
        "website": _child_text(element, "WebsiteURL"),
        "record_update": _child_text(element, "UpdateTime"),
        "source_file": "觀光資料標準V2.1/AttractionList.xml",
    }


def _child_text(parent: ET.Element, path: str) -> str:
    node = parent.find("/".join(f"{{{ATTRACTION_NS}}}{part}" for part in path.split("/")))
    if node is None or node.text is None:
        return ""
    return nfkc(node.text)
