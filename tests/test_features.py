"""Unit tests for the pure helpers in src.features.

These do not require Spark, MLflow, or Databricks. They protect the temporal
feature contract that Silver, Gold, and the API pipeline all depend on.
"""

from datetime import date

import pandas as pd
import pytest

from src.features import (
    calculate_elapsed,
    check_holiday,
    check_holiday_period,
    check_near_holiday,
    convert_utc_to_hhmm,
    fix_timestamp_smart,
    get_season,
    hhmm_to_hour,
    build_date_dimension,
    spark_day_of_week,
)


class TestSparkDayOfWeek:
    """Spark's dayofweek is 1=Sunday..7=Saturday; Python's weekday() is 0=Monday.

    Getting this wrong shifts every day-of-week feature by one and is invisible
    in aggregate statistics, so it is pinned against known dates.
    """

    @pytest.mark.parametrize("d,expected", [
        (date(2023, 1, 1), 1),   # Sunday
        (date(2023, 1, 2), 2),   # Monday
        (date(2023, 1, 6), 6),   # Friday
        (date(2023, 1, 7), 7),   # Saturday
    ])
    def test_matches_spark_convention(self, d, expected):
        assert spark_day_of_week(d) == expected


class TestBuildDateDimension:
    """The date dimension replaces four per-row Python UDFs with a broadcast join.

    The refactor is only safe if the dimension reproduces the helpers exactly,
    so equivalence is asserted rather than assumed.
    """

    def test_covers_every_day_inclusive(self):
        rows = build_date_dimension(date(2023, 1, 1), date(2023, 1, 31))
        assert len(rows) == 31
        assert rows[0]["flight_date"] == date(2023, 1, 1)
        assert rows[-1]["flight_date"] == date(2023, 1, 31)

    def test_flight_date_stays_a_date(self):
        # A pandas offset would promote this to Timestamp and write the wrong
        # Spark column type.
        for row in build_date_dimension(date(2023, 1, 1), date(2023, 1, 5)):
            assert type(row["flight_date"]) is date

    def test_agrees_with_the_udf_helpers_it_replaces(self):
        for row in build_date_dimension(date(2019, 1, 1), date(2019, 12, 31)):
            d = row["flight_date"]
            assert row["is_holiday"] == check_holiday(d)
            assert row["is_near_holiday"] == check_near_holiday(d)
            assert row["is_holiday_period"] == check_holiday_period(d)
            assert row["season"] == get_season(d.month)

    def test_calendar_fields_match_builtins(self):
        for row in build_date_dimension(date(2021, 1, 1), date(2021, 12, 31)):
            d = row["flight_date"]
            assert row["flight_month"] == d.month
            assert row["day_of_month"] == d.day
            assert row["week_of_year"] == d.isocalendar()[1]
            assert row["quarter"] == (d.month - 1) // 3 + 1
            assert row["is_weekend"] == int(spark_day_of_week(d) in (1, 7))

    def test_known_holiday_flags(self):
        rows = {r["flight_date"]: r for r in
                build_date_dimension(date(2023, 7, 1), date(2023, 7, 10))}
        assert rows[date(2023, 7, 4)]["is_holiday"] == 1
        assert rows[date(2023, 7, 3)]["is_holiday"] == 0
        assert rows[date(2023, 7, 3)]["is_near_holiday"] == 1
        assert rows[date(2023, 7, 10)]["is_holiday_period"] == 1

    def test_rejects_reversed_range(self):
        with pytest.raises(ValueError):
            build_date_dimension(date(2023, 12, 31), date(2023, 1, 1))


class TestGetSeason:
    @pytest.mark.parametrize("month,expected", [
        (12, "Winter"), (1, "Winter"), (2, "Winter"),
        (3, "Spring"), (4, "Spring"), (5, "Spring"),
        (6, "Summer"), (7, "Summer"), (8, "Summer"),
        (9, "Fall"), (10, "Fall"), (11, "Fall"),
    ])
    def test_all_months(self, month, expected):
        assert get_season(month) == expected

    def test_invalid_month_raises(self):
        with pytest.raises(ValueError):
            get_season(13)


class TestHolidayFlags:
    def test_july_4_is_holiday(self):
        assert check_holiday(date(2025, 7, 4)) == 1

    def test_random_tuesday_is_not_holiday(self):
        assert check_holiday(date(2025, 3, 18)) == 0

    def test_day_before_july_4_is_near_holiday(self):
        assert check_near_holiday(date(2025, 7, 3)) == 1

    def test_two_days_before_july_4_is_not_near_holiday(self):
        assert check_near_holiday(date(2025, 7, 2)) == 0

    def test_week_before_thanksgiving_is_holiday_period(self):
        # Thanksgiving 2025 is Nov 27; Nov 20 is 7 days prior.
        assert check_holiday_period(date(2025, 11, 20)) == 1

    def test_far_from_holidays_is_not_holiday_period(self):
        assert check_holiday_period(date(2025, 3, 18)) == 0


class TestConvertUtcToHhmm:
    def test_iso_string(self):
        assert convert_utc_to_hhmm("2025-03-15T14:30:00+00:00") == 1430

    def test_midnight(self):
        assert convert_utc_to_hhmm("2025-03-15T00:00:00+00:00") == 0

    def test_late_evening(self):
        assert convert_utc_to_hhmm("2025-03-15T23:59:00+00:00") == 2359

    def test_none_returns_none(self):
        assert convert_utc_to_hhmm(None) is None

    def test_garbage_returns_none(self):
        assert convert_utc_to_hhmm("not-a-timestamp") is None


class TestFixTimestampSmart:
    def test_naive_new_york_summer_to_utc(self):
        # EDT is UTC-4 in mid-March post-DST, UTC-5 in mid-Feb.
        result = fix_timestamp_smart("2025-07-15 12:00:00", "America/New_York")
        # EDT (UTC-4) → 16:00 UTC
        assert result.hour == 16
        assert result.minute == 0

    def test_naive_los_angeles_winter_to_utc(self):
        # PST is UTC-8 in Jan.
        result = fix_timestamp_smart("2025-01-15 09:00:00", "America/Los_Angeles")
        assert result.hour == 17
        assert result.minute == 0

    def test_bad_timezone_returns_nat(self):
        result = fix_timestamp_smart("2025-03-15 12:00:00", "Not/A_Zone")
        assert pd.isna(result)

    def test_bad_timestamp_returns_nat(self):
        result = fix_timestamp_smart("garbage", "UTC")
        assert pd.isna(result)


class TestCalculateElapsed:
    def test_same_day(self):
        # 09:30 → 12:15 = 2h 45m = 165 min
        assert calculate_elapsed(930, 1215) == 165

    def test_midnight_rollover(self):
        # 23:30 → 01:15 (next day) = 1h 45m = 105 min
        assert calculate_elapsed(2330, 115) == 105

    def test_none_input(self):
        assert calculate_elapsed(None, 1200) is None
        assert calculate_elapsed(1200, None) is None


class TestHhmmToHour:
    def test_extracts_hour(self):
        assert hhmm_to_hour(1430) == 14

    def test_zero(self):
        assert hhmm_to_hour(0) == 0

    def test_none(self):
        assert hhmm_to_hour(None) is None
