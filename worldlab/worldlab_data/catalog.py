from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from worldlab_data.textutil import county_parts, nfkc

NS = {"t": "https://media.taiwan.net.tw/standard/schema/2.0"}


def _text(parent: ET.Element, path: str) -> str:
    node = parent.find(path, NS)
    if node is None or node.text is None:
        return ""
    return nfkc(node.text)


def _class_codes(parent: ET.Element, collection: str, item: str) -> str:
    codes = [nfkc(node.text) for node in parent.findall(f"t:{collection}/t:{item}", NS) if node.text]
    return ",".join(code for code in codes if code)


def parse_attractions(path: Path) -> tuple[pd.DataFrame, str]:
    rows: list[dict[str, object]] = []
    document_update = ""
    for event, element in ET.iterparse(path, events=("end",)):
        if element.tag == f"{{{NS['t']}}}UpdateTime" and not document_update:
            document_update = nfkc(element.text)
        if element.tag != f"{{{NS['t']}}}Attraction":
            continue
        county, _ = county_parts(_text(element, "t:PostalAddress/t:City"))
        rows.append(
            {
                "poi_id": _text(element, "t:AttractionID"),
                "official_name": _text(element, "t:AttractionName"),
                "alternative_names": _text(element, "t:AlternateNames"),
                "county": county,
                "district": _text(element, "t:PostalAddress/t:Town"),
                "latitude": _text(element, "t:PositionLat"),
                "longitude": _text(element, "t:PositionLon"),
                "category_codes": _class_codes(element, "AttractionClasses", "AttractionClass"),
                "service_status_code": _text(element, "t:ServiceStatus"),
                "is_public_access_code": _text(element, "t:IsPublicAccess"),
                "is_accessible_for_free_code": _text(element, "t:IsAccessibleForFree"),
                "record_update_time": _text(element, "t:UpdateTime"),
                "source_file": path.name,
            }
        )
        element.clear()
    frame = pd.DataFrame.from_records(rows)
    return frame, document_update


def parse_events(path: Path) -> tuple[pd.DataFrame, str]:
    rows: list[dict[str, object]] = []
    document_update = ""
    for event, element in ET.iterparse(path, events=("end",)):
        if element.tag == f"{{{NS['t']}}}UpdateTime" and not document_update:
            document_update = nfkc(element.text)
        if element.tag != f"{{{NS['t']}}}Event":
            continue
        county, _ = county_parts(_text(element, "t:PostalAddress/t:City"))
        end_text = _text(element, "t:EndDateTime")
        end_year = int(end_text[:4]) if len(end_text) >= 4 and end_text[:4].isdigit() else None
        rows.append(
            {
                "event_id": _text(element, "t:EventID"),
                "event_name": _text(element, "t:EventName"),
                "site_id": "",
                "county": county,
                "district": _text(element, "t:PostalAddress/t:Town"),
                "latitude": _text(element, "t:PositionLat"),
                "longitude": _text(element, "t:PositionLon"),
                "event_type_code": _class_codes(element, "EventClasses", "EventClass"),
                "event_start": _text(element, "t:StartDateTime"),
                "event_end": end_text,
                "event_status": _text(element, "t:EventStatus"),
                "announcement_date": "",
                "record_update_time": _text(element, "t:UpdateTime"),
                "source": path.name,
                "implausible_end_year": bool(end_year is not None and end_year > 2035),
                "known_at_prediction_time": False,
                "usage_role": "ex_post_explanation_only",
            }
        )
        element.clear()
    frame = pd.DataFrame.from_records(rows)
    if not frame.empty:
        frame = frame.drop_duplicates(subset=["event_id"], keep="first")
    return frame, document_update


def events_monthly(frame: pd.DataFrame) -> pd.DataFrame:
    """Count each event once per overlapped month. Implausible end dates stay out of the count."""
    if frame.empty:
        return pd.DataFrame(columns=["year_month", "n_events", "announcement_known", "usage_role"])
    counts: dict[str, set[str]] = {}
    usable = frame[~frame["implausible_end_year"].astype(bool)]
    for record in usable.to_dict(orient="records"):
        start = str(record["event_start"])[:7]
        end = str(record["event_end"])[:7]
        if len(start) != 7 or len(end) != 7:
            continue
        for period in pd.period_range(start, end, freq="M"):
            counts.setdefault(str(period), set()).add(str(record["event_id"]))
    rows = [
        {
            "year_month": year_month,
            "n_events": len(event_ids),
            "announcement_known": False,
            "usage_role": "ex_post_explanation_only",
        }
        for year_month, event_ids in sorted(counts.items())
    ]
    return pd.DataFrame.from_records(rows)


def load_class_labels(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    labels: dict[str, str] = {}
    for record in frame.to_dict(orient="records"):
        code = nfkc(record.get("official_class_code"))
        label = nfkc(record.get("official_class_label"))
        if code and label and code not in labels:
            labels[code] = label
    return labels
