"""Rules for external attention sources and an immutable forecast registry.

Pageviews are not visitor exposure. A national-park article is not each site inside it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from worldlab_data.textutil import comparison_key

PARK_ARTICLES = ("玉山國家公園", "太魯閣國家公園", "雪霸國家公園")
PAGEVIEW_START = "2015-07"
PAGEVIEW_END = "2023-12"
MIN_OBSERVED_MONTHS = 36
MIN_COVERAGE = 0.80
COORDINATE_REJECT_KM = 15.0
STRIP_SUFFIXES = ("遊憩區",)
EXCLUDE_NAME_TOKENS = ("遊客中心", "遊客服務", "服務中心", "管理處", "沿線")
USER_AGENT = "WorldLabV4Phase18/1.0 (academic feasibility audit; not a bulk crawler)"
PAGEVIEW_LICENSE = "Wikimedia pageviews API, CC0"
ARTICLE_LICENSE = "Wikipedia article metadata, CC BY-SA 4.0"
SOURCE_API = "https://zh.wikipedia.org/w/api.php"
PAGEVIEW_API = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"


def month_span(start: str = PAGEVIEW_START, end: str = PAGEVIEW_END) -> list[str]:
    year, month = (int(part) for part in start.split("-"))
    end_year, end_month = (int(part) for part in end.split("-"))
    months: list[str] = []
    while (year, month) <= (end_year, end_month):
        months.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            month = 1
            year += 1
    return months


def place_stem(official_name: str) -> str:
    for suffix in STRIP_SUFFIXES:
        if official_name.endswith(suffix) and len(official_name) > len(suffix):
            return official_name[: -len(suffix)]
    return official_name


def classify_article(
    official_name: str,
    resolved_title: str,
    county: str,
    parent_area: str,
    extract: str,
    disambiguation: bool,
    site_latitude: float | None = None,
    site_longitude: float | None = None,
    article_latitude: float | None = None,
    article_longitude: float | None = None,
    redirected_from_exact_title: bool = False,
) -> dict[str, str]:
    """Accept only an article that names this site. Search rank is not evidence."""
    if disambiguation or not resolved_title:
        return {"accepted": "false", "match_evidence": "no_unique_article"}
    if resolved_title in PARK_ARTICLES and resolved_title != official_name:
        return {"accepted": "false", "match_evidence": "park_article_not_the_site"}
    exact = comparison_key(resolved_title) == comparison_key(official_name)
    stem = place_stem(official_name)
    stem_match = comparison_key(resolved_title) == comparison_key(stem) and stem != official_name
    if not exact and not stem_match and not redirected_from_exact_title:
        return {"accepted": "false", "match_evidence": "title_does_not_name_the_site"}
    if site_latitude is not None and article_latitude is not None and site_longitude is not None and article_longitude is not None:
        distance = _haversine_km(site_latitude, site_longitude, article_latitude, article_longitude)
        if distance > COORDINATE_REJECT_KM:
            return {"accepted": "false", "match_evidence": "coordinates_disagree"}
    text = extract or ""
    if county not in text and parent_area not in text and not (exact or redirected_from_exact_title):
        return {"accepted": "false", "match_evidence": "geography_not_in_extract"}
    if exact or redirected_from_exact_title:
        evidence = "exact_title" if exact else "exact_title_redirect"
    else:
        evidence = "stem_title_with_geography"
    return {"accepted": "true", "match_evidence": evidence}


def coverage_status(observed_months: list[str], start: str = PAGEVIEW_START, end: str = PAGEVIEW_END) -> dict[str, object]:
    expected = month_span(start, end)
    observed = sorted(set(observed_months) & set(expected))
    missing = [month for month in expected if month not in set(observed)]
    ratio = len(observed) / len(expected) if expected else 0.0
    usable = len(observed) >= MIN_OBSERVED_MONTHS and ratio >= MIN_COVERAGE
    return {
        "earliest_available_month": observed[0] if observed else "",
        "monthly_coverage": ratio,
        "observed_months": len(observed),
        "expected_months": len(expected),
        "missing_months": missing,
        "usable_series": usable,
    }


def select_alternate_candidates(rows: list[dict[str, str]], excluded_names: set[str], limit_sites: int = 8) -> list[dict[str, str]]:
    """One long-running official site per county. Forecast scores are not an input."""
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if row.get("match_status") not in {"provisional_rule_match", "singleton"}:
            continue
        grouped.setdefault(row["site_uid"], []).append(row)
    chosen = []
    for site_uid, chain in grouped.items():
        name = sorted(chain, key=lambda item: int(item["last_year"]))[-1]["official_name"]
        if name in excluded_names or any(token in name for token in EXCLUDE_NAME_TOKENS):
            continue
        if any(item.get("match_status") == "needs_manual_review" for item in chain):
            continue
        first_year = min(int(item["first_year"]) for item in chain)
        last_year = max(int(item["last_year"]) for item in chain)
        if first_year > 2015 or last_year < 2023:
            continue
        latest = sorted(chain, key=lambda item: int(item["last_year"]))[-1]
        chosen.append(
            {
                "site_uid": site_uid,
                "official_name": name,
                "county": latest["county"],
                "parent_area": latest.get("parent_area", ""),
                "first_year": str(first_year),
                "last_year": str(last_year),
                "selection_rule": "pre_2024_span_one_per_county_name_is_not_a_facility_counter",
            }
        )
    chosen.sort(key=lambda item: (item["county"], item["official_name"], item["site_uid"]))
    per_county: list[dict[str, str]] = []
    seen_counties: set[str] = set()
    for item in chosen:
        if item["county"] in seen_counties:
            continue
        seen_counties.add(item["county"])
        per_county.append(item)
        if len(per_county) >= limit_sites:
            break
    return per_county


def _haversine_km(left_lat: float, left_lon: float, right_lat: float, right_lon: float) -> float:
    from math import asin, cos, radians, sin, sqrt

    radius = 6371.0
    phi1 = radians(left_lat)
    phi2 = radians(right_lat)
    d_phi = radians(right_lat - left_lat)
    d_lambda = radians(right_lon - left_lon)
    chord = sin(d_phi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(d_lambda / 2) ** 2
    return 2 * radius * asin(sqrt(chord))


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")
STORAGE_GUARANTEE = "api_append_only_not_tamper_proof"
EXTERNAL_TIME_ORDER = (
    "A local filesystem does not prove external time order. "
    "Publish the manifest hashes to an RFC 3161 timestamp or a study preregistration "
    "before target_first_published_at."
)


class ProspectiveRegistry:
    """Append-only by API contract. A local disk can still be edited outside this API."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write_snapshot(self, record: dict[str, str], content: bytes) -> dict[str, str]:
        snapshot_id = _safe_id(record["snapshot_id"], "snapshot_id")
        folder = self._resolve(snapshot_id)
        if folder.exists():
            raise FileExistsError("an existing snapshot cannot be overwritten")
        folder.mkdir()
        self._mark_incomplete(folder)
        try:
            _atomic_write_bytes(folder / "content.bin", content)
            stored = dict(record)
            stored["snapshot_id"] = snapshot_id
            stored["content_sha256"] = _sha256(content)
            stored["storage_guarantee"] = STORAGE_GUARANTEE
            _assert_file_hash(folder / "content.bin", stored["content_sha256"])
            _write_json(folder / "snapshot.json", stored)
        except Exception:
            raise
        else:
            self._mark_complete(folder)
        return stored

    def revise(self, prior_id: str, new_id: str, record: dict[str, str], content: bytes) -> dict[str, str]:
        prior = self.verify_snapshot(prior_id)
        prior_bytes = (self._resolve(prior_id) / "content.bin").read_bytes()
        stored = self.write_snapshot(record | {"snapshot_id": new_id, "revises": prior["snapshot_id"]}, content)
        if (self._resolve(prior_id) / "content.bin").read_bytes() != prior_bytes:
            raise RuntimeError("a revision changed the first published snapshot")
        return stored

    def verify_snapshot(self, snapshot_id: str) -> dict[str, str]:
        folder = self._resolve(snapshot_id)
        self._require_complete(folder, "snapshot")
        stored = json.loads((folder / "snapshot.json").read_text(encoding="utf-8"))
        _assert_file_hash(folder / "content.bin", stored["content_sha256"])
        return stored

    def freeze_forecast(
        self,
        forecast_id: str,
        prediction: bytes,
        config: bytes,
        code_version: bytes,
        input_cutoff_at: str,
        frozen_at: str,
        target_first_published_at: str,
        forecast_origin: str,
        model_version: str,
        revises: str = "",
    ) -> dict[str, str]:
        forecast_id = _safe_id(forecast_id, "forecast_id")
        if revises:
            revises = _safe_id(revises, "revises")
            self.verify_forecast(revises)
        cutoff, frozen, published = _freeze_order(input_cutoff_at, frozen_at, target_first_published_at)
        folder = self._resolve("forecasts", forecast_id)
        if folder.exists():
            raise FileExistsError("a frozen forecast cannot be replaced")
        folder.mkdir(parents=True)
        self._mark_incomplete(folder)
        try:
            prediction_sha = _store_hashed(folder / "prediction.bin", prediction)
            config_sha = _store_hashed(folder / "config.bin", config)
            code_sha = _store_hashed(folder / "code_version.bin", code_version)
            stored = {
                "forecast_id": forecast_id,
                "revises": revises,
                "input_cutoff_at": cutoff,
                "frozen_at": frozen,
                "target_first_published_at": published,
                "forecast_origin": forecast_origin,
                "model_version": model_version,
                "prediction_sha256": prediction_sha,
                "config_sha256": config_sha,
                "code_version_sha256": code_sha,
                "storage_guarantee": STORAGE_GUARANTEE,
                "external_time_order": EXTERNAL_TIME_ORDER,
            }
            _write_json(folder / "forecast.json", stored)
            self._assert_forecast_hashes(folder, stored)
        except Exception:
            raise
        else:
            self._mark_complete(folder)
        return stored

    def revise_forecast(
        self,
        prior_id: str,
        new_id: str,
        prediction: bytes,
        config: bytes,
        code_version: bytes,
        input_cutoff_at: str,
        frozen_at: str,
        target_first_published_at: str,
        forecast_origin: str,
        model_version: str,
    ) -> dict[str, str]:
        prior_folder = self._resolve("forecasts", prior_id)
        prior_manifest = (prior_folder / "forecast.json").read_bytes()
        prior_prediction = (prior_folder / "prediction.bin").read_bytes()
        stored = self.freeze_forecast(
            new_id,
            prediction,
            config,
            code_version,
            input_cutoff_at,
            frozen_at,
            target_first_published_at,
            forecast_origin,
            model_version,
            revises=prior_id,
        )
        if (prior_folder / "forecast.json").read_bytes() != prior_manifest:
            raise RuntimeError("a forecast revision changed the previous manifest")
        if (prior_folder / "prediction.bin").read_bytes() != prior_prediction:
            raise RuntimeError("a forecast revision changed the previous prediction")
        return stored

    def verify_forecast(self, forecast_id: str) -> dict[str, str]:
        folder = self._resolve("forecasts", forecast_id)
        self._require_complete(folder, "forecast")
        stored = json.loads((folder / "forecast.json").read_text(encoding="utf-8"))
        self._assert_forecast_hashes(folder, stored)
        return stored

    def record_first_observation(
        self,
        forecast_id: str,
        content: bytes,
        target_first_published_at: str,
        first_retrieved_at: str,
        source_url: str,
        http_metadata: str,
    ) -> dict[str, str]:
        manifest_path = self._resolve("forecasts", forecast_id) / "forecast.json"
        manifest_before = manifest_path.read_bytes()
        forecast = self.verify_forecast(forecast_id)
        published = _parse_utc(target_first_published_at)
        if published != _parse_utc(forecast["target_first_published_at"]):
            raise ValueError("observation publication time does not match the frozen target")
        retrieved = _parse_utc(first_retrieved_at)
        if retrieved < published or retrieved <= _parse_utc(forecast["frozen_at"]):
            raise ValueError("the outcome was retrieved before it was published")
        folder = self._resolve("observations", forecast_id)
        if folder.exists():
            raise FileExistsError("the first observation snapshot cannot be sealed twice")
        folder.mkdir(parents=True)
        self._mark_incomplete(folder)
        try:
            digest = _store_hashed(folder / "content.bin", content)
            stored = {
                "forecast_id": forecast["forecast_id"],
                "target_first_published_at": _format_utc(target_first_published_at),
                "first_retrieved_at": _format_utc(first_retrieved_at),
                "source_url": source_url,
                "http_metadata": http_metadata,
                "content_sha256": digest,
                "storage_guarantee": STORAGE_GUARANTEE,
            }
            _write_json(folder / "observation.json", stored)
            _assert_file_hash(folder / "content.bin", digest)
        except Exception:
            raise
        else:
            self._mark_complete(folder)
        if manifest_path.read_bytes() != manifest_before:
            raise RuntimeError("recording an observation changed the frozen forecast")
        return stored

    def append_observation_revision(
        self,
        forecast_id: str,
        revision_id: str,
        content: bytes,
        retrieved_at: str,
        source_url: str,
        http_metadata: str,
    ) -> dict[str, str]:
        first = self.verify_observation(forecast_id)
        first_bytes = (self._resolve("observations", forecast_id) / "content.bin").read_bytes()
        revision_id = _safe_id(revision_id, "revision_id")
        folder = self._resolve("observations", forecast_id, "revisions", revision_id)
        if folder.exists():
            raise FileExistsError("an observation revision cannot be overwritten")
        folder.mkdir(parents=True)
        self._mark_incomplete(folder)
        try:
            digest = _store_hashed(folder / "content.bin", content)
            stored = {
                "forecast_id": first["forecast_id"],
                "revision_id": revision_id,
                "revises_first_retrieved_at": first["first_retrieved_at"],
                "retrieved_at": _format_utc(retrieved_at),
                "source_url": source_url,
                "http_metadata": http_metadata,
                "content_sha256": digest,
                "storage_guarantee": STORAGE_GUARANTEE,
            }
            _write_json(folder / "revision.json", stored)
        except Exception:
            raise
        else:
            self._mark_complete(folder)
        if (self._resolve("observations", forecast_id) / "content.bin").read_bytes() != first_bytes:
            raise RuntimeError("an observation revision changed the first retrieved snapshot")
        return stored

    def verify_observation(self, forecast_id: str) -> dict[str, str]:
        folder = self._resolve("observations", forecast_id)
        self._require_complete(folder, "observation")
        stored = json.loads((folder / "observation.json").read_text(encoding="utf-8"))
        _assert_file_hash(folder / "content.bin", stored["content_sha256"])
        return stored

    def replace_prediction(self, forecast_id: str, prediction_file_hash: str) -> None:
        del prediction_file_hash
        self.verify_forecast(forecast_id)
        raise ValueError("a frozen forecast cannot be changed in place")

    def record_failure(self, failure: dict[str, str]) -> dict[str, str]:
        failure_id = _safe_id(failure["failure_id"], "failure_id")
        stored = dict(failure)
        stored["failure_id"] = failure_id
        if stored.get("retrieved_at"):
            stored["retrieved_at"] = _format_utc(stored["retrieved_at"])
        stored["storage_guarantee"] = STORAGE_GUARANTEE
        folder = self._resolve("failures")
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{failure_id}.json"
        if path.exists():
            raise FileExistsError("a failure record cannot be overwritten")
        _write_json(path, stored)
        return stored

    def _resolve(self, *parts: str) -> Path:
        safe_parts = [_safe_id(part, "path") if part not in {"forecasts", "observations", "failures", "revisions"} else part for part in parts]
        root = self.root.resolve()
        folder = root.joinpath(*safe_parts).resolve()
        if folder != root and root not in folder.parents:
            raise ValueError("identifier escapes the registry")
        return folder

    def _mark_incomplete(self, folder: Path) -> None:
        (folder / ".incomplete").write_text("writing", encoding="utf-8")

    def _mark_complete(self, folder: Path) -> None:
        (folder / ".incomplete").unlink()

    def _require_complete(self, folder: Path, label: str) -> None:
        if (folder / ".incomplete").exists() or not folder.exists():
            raise RuntimeError(f"incomplete {label} is not a valid record")

    def _assert_forecast_hashes(self, folder: Path, stored: dict[str, str]) -> None:
        _assert_file_hash(folder / "prediction.bin", stored["prediction_sha256"])
        _assert_file_hash(folder / "config.bin", stored["config_sha256"])
        _assert_file_hash(folder / "code_version.bin", stored["code_version_sha256"])


def _safe_id(value: str, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise ValueError(f"{label} is not a safe identifier")
    return value


def _parse_utc(value: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError("timestamp must be a timezone-aware UTC datetime")
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError("timestamp must be a timezone-aware UTC datetime") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


def _format_utc(value: str) -> str:
    return _parse_utc(value).strftime("%Y-%m-%dT%H:%M:%SZ")


def _freeze_order(input_cutoff_at: str, frozen_at: str, target_first_published_at: str) -> tuple[str, str, str]:
    cutoff = _parse_utc(input_cutoff_at)
    frozen = _parse_utc(frozen_at)
    published = _parse_utc(target_first_published_at)
    if not (cutoff <= frozen < published):
        raise ValueError("the forecast was not frozen after the input cutoff and before the target was published")
    return _format_utc(input_cutoff_at), _format_utc(frozen_at), _format_utc(target_first_published_at)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _assert_file_hash(path: Path, expected: str) -> None:
    if _sha256(path.read_bytes()) != expected:
        raise RuntimeError("stored file hash does not match the manifest")


def _store_hashed(path: Path, content: bytes) -> str:
    digest = _sha256(content)
    _atomic_write_bytes(path, content)
    _assert_file_hash(path, digest)
    return digest


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _write_json(path: Path, payload: dict[str, str]) -> None:
    _atomic_write_bytes(path, json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"))
