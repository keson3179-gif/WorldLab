from __future__ import annotations

from pathlib import Path

import pandas as pd


def station_month_from_frozen(metrics_path: Path, registry_path: Path | None) -> pd.DataFrame:
    """Reuse the frozen v0.2 station-month file. Do not relabel daily maxima as mean temperature."""
    metrics = pd.read_csv(metrics_path, dtype=str, keep_default_na=False)
    registry = pd.read_csv(registry_path, dtype=str, keep_default_na=False) if registry_path and registry_path.exists() else pd.DataFrame()
    names = {}
    elevations = {}
    if not registry.empty:
        for record in registry.to_dict(orient="records"):
            names[str(record.get("stno"))] = record.get("name", "")
            elevations[str(record.get("stno"))] = record.get("alt_m", "")
    rows: list[dict[str, object]] = []
    for record in metrics.to_dict(orient="records"):
        temperature_ok = str(record.get("weather_valid")) == "1"
        rain_ok = str(record.get("rain_valid_90")) == "1"
        station_id = str(record.get("station_id"))
        rows.append(
            {
                "station_id": station_id,
                "station_name": names.get(station_id, ""),
                "station_elevation_m": elevations.get(station_id, ""),
                "year_month": record.get("year_month"),
                "mean_temperature": None,
                "mean_temperature_note": "來源是逐日最高溫 TX02，沒有日均溫，此欄維持缺值",
                "mean_daily_max_c": _number_if(record.get("mean_tmax"), temperature_ok),
                "max_temperature": _number_if(record.get("max_tmax"), temperature_ok),
                "hot_days": _number_if(record.get("hot_days_35"), temperature_ok),
                "hot_days_threshold_c": 35 if temperature_ok else None,
                "rainfall_total": _number_if(record.get("rainfall_total"), rain_ok),
                "rainy_days": _number_if(record.get("rain_days"), rain_ok),
                "weather_coverage": _number_if(record.get("tx02_coverage_rate"), True),
                "rain_coverage": _number_if(record.get("pp01_coverage_rate"), True),
                "source_file": metrics_path.name,
                "source_role": "frozen_v0_2_station_month_reuse",
                "elevation_difference_checked": False,
                "publication_history_unknown": True,
            }
        )
    return pd.DataFrame.from_records(rows)


def _number_if(value: object, allowed: bool) -> float | None:
    if not allowed or value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def prior_station_links(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    keep = [
        column
        for column in (
            "attraction_id",
            "county_original",
            "attraction_name_original",
            "station_primary",
            "station_primary_name",
            "station_primary_km",
            "distance_gt15km_flag",
            "coordinate_confidence",
        )
        if column in frame.columns
    ]
    out = frame[keep].copy()
    out["link_status"] = "prior_v0_2_frozen_not_revalidated"
    out["elevation_difference_checked"] = False
    out["notes"] = "沿用既有高溫研究的主站表，本次沒有重配，也沒有景點高程可檢查海拔差"
    return out
