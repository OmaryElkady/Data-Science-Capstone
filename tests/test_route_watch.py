"""Unit tests for src.route_watch."""

from datetime import date, datetime, timezone

import pytest

from src.route_watch import (
    dedupe_same_minute,
    ahead_of,
    departure_window,
    parse_routes,
    quota_exhausted,
    spread_pick,
    still_airborne,
    units_remaining,
)


class TestParseRoutes:
    def test_accepts_what_a_person_would_type(self):
        assert parse_routes("ATL-JFK, ord - lga,LAX-SFO") == [
            ("ATL", "JFK"), ("ORD", "LGA"), ("LAX", "SFO")]

    def test_ignores_empty_entries(self):
        assert parse_routes("ATL-JFK,,") == [("ATL", "JFK")]

    @pytest.mark.parametrize("bad", ["", "ATL", "ATL-JF", "ATLJFK", "ATL-ATL", "ATL-JFK,XX"])
    def test_rejects_anything_else(self, bad):
        with pytest.raises(ValueError):
            parse_routes(bad)


class TestDepartureWindow:
    """The airport query takes local time, and the origin decides what local is."""

    NOW = datetime(2026, 9, 28, 11, 0, tzinfo=timezone.utc)   # the scheduled run

    @pytest.mark.parametrize("tz,start,end", [
        ("America/New_York", datetime(2026, 9, 28, 8, 0), datetime(2026, 9, 28, 20, 0)),
        ("America/Chicago", datetime(2026, 9, 28, 7, 0), datetime(2026, 9, 28, 19, 0)),
        ("America/Los_Angeles", datetime(2026, 9, 28, 5, 0), datetime(2026, 9, 28, 17, 0)),
    ])
    def test_starts_one_lead_after_local_now(self, tz, start, end):
        assert departure_window(self.NOW, tz, lead_minutes=60, hours=12) == (start, end)

    def test_follows_daylight_saving(self):
        # January: Chicago is UTC-6, not UTC-5.
        winter = datetime(2026, 1, 15, 11, 0, tzinfo=timezone.utc)
        start, _ = departure_window(winter, "America/Chicago", lead_minutes=60, hours=12)
        assert start == datetime(2026, 1, 15, 6, 0)

    def test_returns_naive_local_times(self):
        start, end = departure_window(self.NOW, "America/New_York", 60, 12)
        assert start.tzinfo is None and end.tzinfo is None

    def test_a_negative_lead_reaches_back(self):
        at_1100 = datetime(2026, 9, 28, 11, 0, tzinfo=timezone.utc)
        start, end = departure_window(at_1100, "America/New_York", -120, 12)
        assert (start, end) == (datetime(2026, 9, 28, 5, 0), datetime(2026, 9, 28, 17, 0))

    def test_respects_the_provider_cap(self):
        with pytest.raises(ValueError):
            departure_window(self.NOW, "America/New_York", 60, hours=13)


def _row(hhmm, flight="DL1"):
    return {"flight_iata": flight, "origin_airport_code": "ATL",
            "destination_airport_code": "JFK", "flight_date": date(2026, 9, 28),
            "crs_dep_time": hhmm,
            "scheduled_departure_utc": datetime(2026, 9, 28, hhmm // 100, hhmm % 100,
                                                tzinfo=timezone.utc)}


def _flown(hhmm, dep_delay, phase="in_flight", flight="DL1"):
    return dict(_row(hhmm, flight), provider_phase=phase, dep_delay=dep_delay)


BLOCK = {("ATL", "JFK"): 140.0}
AT_1100 = datetime(2026, 9, 28, 11, 0, tzinfo=timezone.utc)


class TestAheadOf:
    def test_keeps_only_flights_past_the_lead(self):
        rows = [_row(1030), _row(1159), _row(1200), _row(1500)]
        kept = [r["crs_dep_time"] for r in ahead_of(rows, AT_1100, 60)]
        assert kept == [1200, 1500]


class TestStillAirborne:
    def test_a_flight_mid_route_is_airborne(self):
        assert len(still_airborne([_flown(1000, 5.0)], AT_1100, BLOCK)) == 1

    def test_a_flight_that_should_have_landed_is_not(self):
        # Left 08:05 on a 140-minute route: down by about 10:25.
        assert still_airborne([_flown(800, 5.0)], AT_1100, BLOCK) == []

    def test_close_to_landing_is_left_out(self):
        # Left 08:50: 130 minutes in, inside the 20-minute margin of a 140-minute block.
        assert still_airborne([_flown(845, 5.0)], AT_1100, BLOCK) == []

    def test_the_departure_delay_moves_the_clock(self):
        # Scheduled 08:00 but left at 09:30, so still flying at 11:00.
        assert len(still_airborne([_flown(800, 90.0)], AT_1100, BLOCK)) == 1

    def test_not_departed_yet(self):
        assert still_airborne([_flown(1030, 45.0)], AT_1100, BLOCK) == []

    def test_needs_an_observed_departure(self):
        assert still_airborne([_flown(1000, None)], AT_1100, BLOCK) == []
        assert still_airborne([_flown(1000, 5.0, phase="pre_departure")], AT_1100, BLOCK) == []

    def test_unknown_route_is_skipped(self):
        assert still_airborne([_flown(1000, 5.0)], AT_1100, {}) == []


class TestDedupeSameMinute:
    def test_keeps_the_first_of_a_same_minute_pair(self):
        rows = [_row(900, "DL1"), _row(900, "AF9"), _row(1000, "B6")]
        assert [r["flight_iata"] for r in dedupe_same_minute(rows)] == ["DL1", "B6"]


class TestSpreadPick:
    def test_one_pick_from_each_band_of_the_day(self):
        rows = [_row(h * 100, f"F{h}") for h in range(6, 21)]    # 15 departures, 3 bands
        picked = [r["flight_iata"] for r in spread_pick(rows, 3)]
        assert picked == ["F6", "F11", "F16"]

    def test_rotation_moves_within_each_band(self):
        rows = [_row(h * 100, f"F{h}") for h in range(6, 21)]
        picked = [r["flight_iata"] for r in spread_pick(rows, 3, rotation=2)]
        assert picked == ["F8", "F13", "F18"]

    def test_consecutive_days_watch_different_flights(self):
        rows = [_row(h * 100, f"F{h}") for h in range(6, 21)]
        days = [{r["flight_iata"] for r in spread_pick(rows, 3, rotation=d)} for d in range(5)]
        assert all(not (a & b) for a, b in zip(days, days[1:]))

    def test_rotation_wraps_inside_a_band(self):
        rows = [_row(h * 100, f"F{h}") for h in range(6, 21)]
        assert spread_pick(rows, 3, rotation=5) == spread_pick(rows, 3, rotation=0)

    def test_returns_everything_when_there_are_few(self):
        rows = [_row(1200), _row(800)]
        assert [r["crs_dep_time"] for r in spread_pick(rows, 3)] == [800, 1200]

    def test_one_pick_rotates_across_the_whole_window(self):
        rows = [_row(h * 100) for h in (6, 12, 18)]
        assert [spread_pick(rows, 1, rotation=d)[0]["crs_dep_time"] for d in range(3)] == [600, 1200, 1800]

    def test_zero_picks_nothing(self):
        assert spread_pick([_row(900)], 0) == []


class TestQuota:
    def test_parses_the_header(self):
        assert units_remaining({"units_remaining": "282"}) == 282

    @pytest.mark.parametrize("quota", [{}, None, {"units_remaining": None},
                                       {"units_remaining": "n/a"}])
    def test_unknown_is_none(self, quota):
        assert units_remaining(quota) is None

    def test_exhausted_at_the_reserve(self):
        assert quota_exhausted({"units_remaining": "40"}, reserve=40)
        assert not quota_exhausted({"units_remaining": "41"}, reserve=40)

    def test_unknown_is_not_exhausted(self):
        assert not quota_exhausted({}, reserve=40)
