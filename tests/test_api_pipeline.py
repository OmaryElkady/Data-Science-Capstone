"""Offline tests for the AviationStack projection.

The recorded fixture is the same one the notebooks fall back to when
USE_FIXTURE=True, so these tests exercise the exact contract used in
Databricks and give recruiters a runnable demo without a paid API key.
"""

import pandas as pd
import pytest

from src.api_pipeline import load_fixture, project_to_silver, validate_schema


@pytest.fixture(scope="module")
def payload():
    return load_fixture()


@pytest.fixture(scope="module")
def silver(payload):
    return project_to_silver(payload)


def test_fixture_has_records(payload):
    assert len(payload["data"]) >= 1


def test_silver_row_per_record(payload, silver):
    assert len(silver) == len(payload["data"])


def test_silver_contract(silver):
    required = {
        "airline_name", "airline_code", "fl_number",
        "origin_airport_code", "destination_airport_code",
        "flight_date", "flight_month", "flight_year",
        "crs_dep_time", "crs_arr_time", "crs_elapsed_time",
        "dep_delay", "arrival_delay", "distance",
        "day_of_week", "week_of_year", "day_of_month", "quarter",
        "is_weekend", "is_holiday", "is_near_holiday", "is_holiday_period",
        "season",
    }
    missing = required - set(silver.columns)
    assert not missing, f"missing columns: {missing}"


def test_season_is_derived_from_flight_month(silver):
    for _, row in silver.iterrows():
        month = row["flight_month"]
        if month in (12, 1, 2):
            assert row["season"] == "Winter"
        elif month in (3, 4, 5):
            assert row["season"] == "Spring"
        elif month in (6, 7, 8):
            assert row["season"] == "Summer"
        else:
            assert row["season"] == "Fall"


def test_validate_schema_passes(silver):
    report = validate_schema(silver)
    assert report["passed"] is True
    assert report["row_count"] == len(silver)
    assert report["missing_columns"] == []
