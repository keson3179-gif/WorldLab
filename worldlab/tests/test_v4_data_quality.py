from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from worldlab_data.hhi import herfindahl, normalized_herfindahl
from worldlab_data.matching import apply_manual_crosswalk, candidate_matches
from worldlab_data.panel import build_panel, candidate_groups, strict_features
from worldlab_data.qa_runner import _missing_zero_gate, _negative_gate
from worldlab_data.tdx import TDXClient, TDXCredentialsError, TDXRateLimitError, TDXResponse, dry_run_plan
from worldlab_data.textutil import parse_calendar_year, parse_count, stable_site_id
from worldlab_data.visitors import collapse_versions, load_wide_visitor_csv, write_visitor_outputs


MONTHS = [f"{month}月" for month in range(1, 13)]


def _wide_row(year: str, name: str, county: str, values: list[str], total: str, area: str = "測試區") -> str:
    cells = [year, "國家公園", area, name, county, "63000", *values, total]
    return ",".join(cells)


def _write_csv(path: Path, rows: list[str]) -> None:
    header = ",".join(["年別", "類型", "觀光遊憩區", "細分", "縣市", "縣市別", *MONTHS, "合計"])
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


def test_date_parsing() -> None:
    assert parse_calendar_year("2012") == 2012
    assert parse_calendar_year("101") == 2012
    assert parse_calendar_year("民國114年") == 2025
    assert parse_calendar_year("") is None
    visits, status = parse_count("1,200")
    assert visits == 1200
    assert status == "reported"


def test_site_id_uniqueness(tmp_path: Path) -> None:
    path = tmp_path / "visits.csv"
    values = ["1"] * 12
    _write_csv(
        path,
        [
            _wide_row("2012", "同名館", "臺北市", values, "12"),
            _wide_row("2012", "同名館", "臺南市", values, "12"),
        ],
    )
    frame, _, _ = load_wide_visitor_csv(path)
    assert frame["site_id"].nunique() == 2
    assert stable_site_id(["國家公園", "測試區", "同名館", "臺北市"]) != stable_site_id(["國家公園", "測試區", "同名館", "臺南市"])


def test_site_crosswalk(tmp_path: Path) -> None:
    sites = pd.DataFrame(
        [
            {
                "site_id": "stat_a",
                "official_name": "甲館",
                "county": "臺北市",
                "parent_area": "測試區",
                "category": "國家公園",
            }
        ]
    )
    pois = pd.DataFrame(
        [
            {"poi_id": "poi_low", "official_name": "甲館", "county": "臺北市", "district": "中正區", "latitude": "25", "longitude": "121"},
            {"poi_id": "poi_high", "official_name": "完全不同的地方", "county": "臺北市", "district": "大同區", "latitude": "25.1", "longitude": "121.5"},
        ]
    )
    candidates = candidate_matches(sites, pois, score_floor=80)
    assert set(candidates["match_status"]) == {"needs_manual_review"}
    assert "confirmed_manual" not in set(candidates["match_status"])
    manual = tmp_path / "manual.csv"
    manual.write_text(
        "site_id,match_status,candidate_poi_id,candidate_name,match_score\nstat_a,confirmed_manual,poi_manual,人工指定,1\n",
        encoding="utf-8",
    )
    merged = apply_manual_crosswalk(candidates, manual)
    locked = merged[merged["site_id"] == "stat_a"]
    assert list(locked["candidate_poi_id"]) == ["poi_manual"]
    assert list(locked["match_origin"]) == ["manual"]


def test_monthly_duplicates() -> None:
    left = pd.DataFrame([_visit_record("stat_a", "2012-01", 10, "left.csv")])
    right = pd.DataFrame([_visit_record("stat_a", "2012-01", 12, "right.csv")])
    retained, quarantine = collapse_versions(pd.concat([left, right], ignore_index=True))
    assert retained.empty
    assert list(quarantine["observation_status"]) == ["conflicting_source_version", "conflicting_source_version"]


def test_annual_total_reconciliation(tmp_path: Path) -> None:
    path = tmp_path / "visits.csv"
    _write_csv(path, [_wide_row("101", "甲館", "臺北市", ["1"] * 12, "99")])
    frame, _, _ = load_wide_visitor_csv(path)
    assert set(frame["year"]) == {2012}
    assert set(frame["annual_reconciliation"]) == {"annual_total_mismatch"}
    assert float(frame["visits"].sum()) == 12


def test_missing_vs_zero(tmp_path: Path) -> None:
    path = tmp_path / "visits.csv"
    values = ["0", "", "停業", *["1"] * 9]
    _write_csv(path, [_wide_row("2012", "甲館", "臺北市", values, "")])
    frame, _, _ = load_wide_visitor_csv(path)
    statuses = list(frame.sort_values("month")["observation_status"])
    assert statuses[0] == "reported_zero"
    assert statuses[1] == "missing"
    assert statuses[2] == "closed_reported"
    assert pd.isna(frame.loc[frame["month"] == 2, "visits"]).all()
    assert pd.isna(frame.loc[frame["month"] == 3, "visits"]).all()
    bad = frame.copy()
    bad.loc[bad["observation_status"] == "missing", "visits"] = 0
    assert _missing_zero_gate(bad)["status"] == "FAIL"
    assert _missing_zero_gate(frame)["status"] == "PASS"


def test_group_membership() -> None:
    history, visits = _five_site_inputs()
    extra = history.iloc[0].copy()
    extra["site_id"] = "stat_other"
    extra["county"] = "臺南市"
    extra["official_name"] = "不該併入"
    history = pd.concat([history, pd.DataFrame([extra])], ignore_index=True)
    membership = candidate_groups(history, visits)
    included = membership[membership["included"] == True]
    assert included["site_id"].nunique() == 5
    assert "stat_other" not in set(included["site_id"])
    assert included["competition_group_id"].nunique() == 1
    assert set(included["backtest_eligible"]) == {False}
    dashed = history.iloc[:6].copy()
    dashed["parent_area"] = "-"
    dashed["county"] = "臺北市"
    dashed_membership = candidate_groups(dashed, visits)
    assert dashed_membership["included"].eq(False).all()
    assert dashed_membership["exclusion_reason"].str.contains("觀光遊憩區").all()


def test_hhi_formula() -> None:
    even = herfindahl([0.5, 0.5])
    assert even == pytest.approx(0.5)
    assert normalized_herfindahl(even, 2) == pytest.approx(0)
    concentrated = herfindahl([1.0, 0.0])
    assert concentrated == pytest.approx(1)
    assert normalized_herfindahl(concentrated, 2) == pytest.approx(1)
    assert normalized_herfindahl(1, 1) is None


def test_observation_availability() -> None:
    panel = pd.DataFrame(
        [
            {"strict_feature_eligible": True, "available_at": "2012-03-01", "publication_history_unknown": False, "visits": 3},
            {"strict_feature_eligible": True, "available_at": "", "publication_history_unknown": True, "visits": 9},
        ]
    )
    kept = strict_features(panel, "2012-04-01")
    assert list(kept["visits"]) == [3]


def test_no_future_leakage() -> None:
    panel = pd.DataFrame(
        [
            {"strict_feature_eligible": True, "available_at": "2012-08-01", "publication_history_unknown": False, "visits": 8},
        ]
    )
    assert strict_features(panel, "2012-04-01").empty
    history, visits = _five_site_inputs()
    membership = candidate_groups(history, visits)
    exploratory = build_panel(visits, membership, require_backtest=False)
    assert exploratory["strict_feature_eligible"].eq(False).all()
    assert exploratory["publication_history_unknown"].eq(True).all()


def test_pipeline_reproducibility(tmp_path: Path) -> None:
    path = tmp_path / "visits.csv"
    _write_csv(path, [_wide_row("2012", "甲館", "臺北市", ["1"] * 12, "12")])
    first, _, _ = load_wide_visitor_csv(path)
    second, _, _ = load_wide_visitor_csv(path)
    assert first.to_csv(index=False) == second.to_csv(index=False)
    processed = tmp_path / "processed"
    interim = tmp_path / "interim"
    reports = tmp_path / "reports"
    write_visitor_outputs(first, processed, interim, reports, [])
    digest_a = hashlib.sha256((processed / "site_month_visits.parquet").read_bytes()).hexdigest()
    write_visitor_outputs(second, processed, interim, reports, [])
    digest_b = hashlib.sha256((processed / "site_month_visits.parquet").read_bytes()).hexdigest()
    assert digest_a == digest_b


def test_tdx_adapter_offline(tmp_path: Path) -> None:
    plan = dry_run_plan()
    assert plan["will_call_network"] is False
    with pytest.raises(TDXCredentialsError):
        TDXClient("", "", _FakeTransport([]), tmp_path)

    class Sleeper:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, _seconds: float) -> None:
            self.calls += 1

    sleeper = Sleeper()
    transport = _FakeTransport(
        [
            TDXResponse(200, b'{"access_token":"token"}', {}),
            TDXResponse(429, b"", {}),
            TDXResponse(429, b"", {}),
            TDXResponse(429, b"", {}),
        ]
    )
    client = TDXClient("id", "secret", transport, tmp_path, sleeper=sleeper)
    with pytest.raises(TDXRateLimitError):
        client.get_json("v2/Bike/Station/City/Taipei")
    assert sleeper.calls == 3


class _FakeTransport:
    def __init__(self, responses: list[TDXResponse]) -> None:
        self.responses = list(responses)

    def post_form(self, url: str, form: dict[str, str]) -> TDXResponse:
        return self.responses.pop(0)

    def get(self, url: str, headers: dict[str, str]) -> TDXResponse:
        return self.responses.pop(0)


def _visit_record(site_id: str, year_month: str, visits: float, source: str) -> dict[str, object]:
    year, month = year_month.split("-")
    return {
        "site_id": site_id,
        "year_month": year_month,
        "year": int(year),
        "month": int(month),
        "visits": visits,
        "observation_status": "reported",
        "measurement_method": "unknown",
        "source_file": source,
        "source_version": "v",
        "available_at": "",
        "published_at": "",
        "downloaded_at": "",
        "revised_at": "",
        "publication_history_unknown": True,
        "revision_flag": False,
        "annual_reconciliation": "annual_total_match",
        "category": "國家公園",
        "parent_area": "測試區",
        "official_name": site_id,
        "county": "臺北市",
        "county_en": "",
        "county_code": "63000",
        "source_row_key": site_id,
    }


def _five_site_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    history_rows = []
    visit_rows = []
    for index in range(5):
        site_id = f"stat_{index}"
        history_rows.append(
            {
                "site_id": site_id,
                "official_name": f"館{index}",
                "county": "臺北市",
                "parent_area": "測試區",
                "category": "國家公園",
                "first_year_month": "2012-01",
                "last_year_month": "2013-12",
            }
        )
        for month in range(1, 13):
            for year in (2012, 2013):
                visit_rows.append(
                    {
                        "site_id": site_id,
                        "year_month": f"{year}-{month:02d}",
                        "visits": float(index + 1),
                        "observation_status": "reported",
                        "measurement_method": "unknown",
                        "publication_history_unknown": True,
                        "available_at": "",
                    }
                )
    return pd.DataFrame(history_rows), pd.DataFrame(visit_rows)


def test_negative_visits_fail_the_gate() -> None:
    frame = pd.DataFrame([_visit_record("stat_a", "2012-01", -3, "left.csv")])
    frame["observation_status"] = "reported"
    assert _negative_gate(frame)["status"] == "FAIL"
