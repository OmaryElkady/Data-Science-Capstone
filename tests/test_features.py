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
)


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
