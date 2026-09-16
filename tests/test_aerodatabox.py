"""Unit tests for src.aerodatabox, against recorded live responses.

Fixtures are real payloads trimmed for size, so the shape being parsed is the
shape the API actually returns rather than one invented to match the parser.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.aerodatabox import (
    AeroDataBoxClient,
    departure_to_row,
    flight_to_row,
    normalise_flight_number,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def flight_payload():
    return json.loads((FIXTURES / "aerodatabox_flight.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def airport_payload():
    return json.loads((FIXTURES / "aerodatabox_airport.json").read_text(encoding="utf-8"))


class TestNormaliseFlightNumber:
    """The user types a flight number; they should not have to type it precisely."""

    @pytest.mark.parametrize("raw", ["dl1572", "DL1572", "DL 1572", " dl-1572 ", "Dl1572"])
    def test_accepts_what_a_person_would_type(self, raw):
        assert normalise_flight_number(raw) == "DL1572"

    def test_strips_leading_zeros_on_the_numeric_part(self):
        assert normalise_flight_number("AA0100") == "AA100"

    def test_three_letter_carrier_codes(self):
        assert normalise_flight_number("uae55") == "UAE55"

    @pytest.mark.parametrize("bad", ["", "   ", "ATL", "1572", "DL", "DL15720000"])
    def test_rejects_non_flight_numbers(self, bad):
        with pytest.raises(ValueError):
            normalise_flight_number(bad)


class TestFlightToRow:
    def test_derives_the_route_from_the_flight_number(self, flight_payload):
        # The entire point: origin and destination are outputs, not user inputs.
        row = flight_to_row(flight_payload[0])
        assert row["origin_airport_code"] == "ATL"
        assert row["destination_airport_code"] == "IAH"

    def test_carries_both_opensky_join_keys(self, flight_payload):
        row = flight_to_row(flight_payload[0])
        assert row["flight_icao"] == "DAL1572"        # OpenSky callsign
        assert row["aircraft_icao24"] == "a34729"     # OpenSky icao24, lowercased
        assert row["aircraft_icao24"] == row["aircraft_icao24"].lower()

    def test_hhmm_is_the_local_scheduled_time(self, flight_payload):
        # 12:17 local, 16:17Z. This test originally asserted 1617, which encoded
        # the bug rather than the requirement: BTS CRS_DEP_TIME is local, so a
        # UTC projection silently shifts every flight along the hour-of-day risk
        # curve the model learned.
        assert flight_to_row(flight_payload[0])["crs_dep_time"] == 1217

    def test_gate_delay_is_revised_minus_scheduled(self):
        flight = {
            "departure": {"airport": {"iata": "ATL"},
                          "scheduledTime": {"utc": "2026-09-14 16:17Z"},
                          "revisedTime": {"utc": "2026-09-14 16:42Z"}},
            "arrival": {"airport": {"iata": "IAH"},
                        "scheduledTime": {"utc": "2026-09-14 18:22Z"}},
            "airline": {"iata": "DL"}, "number": "DL 1572",
        }
        assert flight_to_row(flight)["dep_delay"] == 25.0

    def test_missing_revised_time_gives_none_not_zero(self, flight_payload):
        # None routes the flight to the pre-departure model. Zero would be a
        # positive claim of punctuality the provider never made.
        flight = {
            "departure": {"airport": {"iata": "ATL"},
                          "scheduledTime": {"utc": "2026-09-14 16:17Z"}},
            "arrival": {"airport": {"iata": "IAH"},
                        "scheduledTime": {"utc": "2026-09-14 18:22Z"}},
            "airline": {"iata": "DL"}, "number": "DL 1572",
        }
        assert flight_to_row(flight)["dep_delay"] is None

    def test_negative_delay_for_an_early_departure(self):
        flight = {
            "departure": {"airport": {"iata": "ATL"},
                          "scheduledTime": {"utc": "2026-09-14 16:17Z"},
                          "revisedTime": {"utc": "2026-09-14 16:05Z"}},
            "arrival": {"airport": {"iata": "IAH"},
                        "scheduledTime": {"utc": "2026-09-14 18:22Z"}},
            "airline": {"iata": "DL"}, "number": "DL 1572",
        }
        assert flight_to_row(flight)["dep_delay"] == -12.0

    def test_elapsed_time_is_the_scheduled_block(self, flight_payload):
        # 16:17Z -> 18:22Z is 125 minutes.
        assert flight_to_row(flight_payload[0])["crs_elapsed_time"] == 125.0

    def test_fl_number_is_numeric_only(self, flight_payload):
        row = flight_to_row(flight_payload[0])
        assert row["fl_number"] == 1572
        assert row["flight_iata"] == "DL1572"

    def test_skips_records_with_no_departure_time(self):
        assert flight_to_row({"departure": {"airport": {"iata": "ATL"}},
                              "arrival": {"airport": {"iata": "IAH"}}}) is None

    def test_skips_records_with_no_route(self):
        assert flight_to_row({
            "departure": {"airport": {}, "scheduledTime": {"utc": "2026-09-14 16:17Z"}},
            "arrival": {"airport": {}},
        }) is None

    def test_every_fixture_record_projects_or_is_skipped(self, flight_payload):
        for rec in flight_payload:
            row = flight_to_row(rec)
            if row is not None:
                assert row["origin_airport_code"] and row["destination_airport_code"]
                assert row["crs_dep_time"] is not None


class TestTimezoneHandling:
    """Clock-time features are LOCAL; durations and delays are UTC.

    BTS records CRS_DEP_TIME as local time at the origin, so the model learned
    hour-of-day risk on a local clock — 02_eda measured 05:00 as the quietest
    hour and 19:00 as the worst, both local. Projecting a UTC hour puts the
    flight at a different point on that curve, and nothing about the mistake is
    visible at run time: the number is plausible, just wrong.
    """

    def test_clock_times_are_local_not_utc(self, flight_payload):
        # DL1572 leaves Atlanta at 12:17 local, 16:17Z.
        row = flight_to_row(flight_payload[0])
        assert row["crs_dep_time"] == 1217, "must be local, not 1617"
        assert row["dep_hour"] == 12
        assert row["crs_arr_time"] == 1322, "must be local at the destination"
        assert row["arr_hour"] == 13

    def test_utc_instant_is_kept_separately(self, flight_payload):
        row = flight_to_row(flight_payload[0])
        assert row["scheduled_departure_utc"].hour == 16
        assert row["origin_timezone"] == "America/New_York"

    def test_elapsed_uses_utc_so_zone_crossings_are_right(self, flight_payload):
        # ATL 12:17 EDT -> IAH 13:22 CDT looks like 65 minutes on the two local
        # clocks and is really 125. Block time has to come from UTC.
        row = flight_to_row(flight_payload[0])
        assert row["crs_elapsed_time"] == 125.0

    def test_local_date_wins_when_the_two_disagree(self):
        # 22:30 local on the 14th is 02:30Z on the 15th. BTS FL_DATE is the local
        # date; dating it in UTC moves it to the wrong day of the week too.
        flight = {
            "departure": {"airport": {"iata": "LAX", "timeZone": "America/Los_Angeles"},
                          "scheduledTime": {"utc": "2026-09-15 05:30Z",
                                            "local": "2026-09-14 22:30-07:00"}},
            "arrival": {"airport": {"iata": "JFK"},
                        "scheduledTime": {"utc": "2026-09-15 13:00Z",
                                          "local": "2026-09-15 09:00-04:00"}},
            "airline": {"iata": "DL"}, "number": "DL 99",
        }
        row = flight_to_row(flight)
        assert row["flight_date"].day == 14, "local date, not the UTC next-day"
        assert row["crs_dep_time"] == 2230
        assert row["day_of_week"] == 2, "Monday the 14th, not Tuesday the 15th"

    def test_delay_is_unaffected_by_zone(self):
        # Delay is a difference between two instants, so it must not change with
        # how either end is displayed.
        flight = {
            "departure": {"airport": {"iata": "LAX"},
                          "scheduledTime": {"utc": "2026-09-15 05:30Z",
                                            "local": "2026-09-14 22:30-07:00"},
                          "revisedTime": {"utc": "2026-09-15 05:50Z",
                                          "local": "2026-09-14 22:50-07:00"}},
            "arrival": {"airport": {"iata": "JFK"},
                        "scheduledTime": {"utc": "2026-09-15 13:00Z"}},
            "airline": {"iata": "DL"}, "number": "DL 99",
        }
        assert flight_to_row(flight)["dep_delay"] == 20.0

    def test_falls_back_to_utc_when_local_is_absent(self):
        flight = {
            "departure": {"airport": {"iata": "ATL"},
                          "scheduledTime": {"utc": "2026-09-14 16:17Z"}},
            "arrival": {"airport": {"iata": "IAH"},
                        "scheduledTime": {"utc": "2026-09-14 18:22Z"}},
            "airline": {"iata": "DL"}, "number": "DL 1572",
        }
        row = flight_to_row(flight)
        assert row["crs_dep_time"] == 1617, "no local available, so UTC is all there is"

    def test_alternatives_use_the_same_clock_as_the_flight(self, airport_payload):
        rows = [r for r in (departure_to_row(d, "ATL")
                            for d in airport_payload["departures"]) if r]
        # The recorded window was 15:00-17:00 local, so every hour must land in it.
        assert all(15 <= r["dep_hour"] <= 17 for r in rows), \
            "a UTC projection would put these at 19-21"


class TestCalendarBlock:
    """A live row and a training row must derive these identically.

    03_silver builds them from src.features; so does this. If the two ever
    diverge, the model is scored on features that mean something different from
    the ones it learned, and nothing about that failure is visible at run time.
    """

    PIPELINE_INPUTS = [
        "flight_month", "flight_year", "day_of_week", "week_of_year", "day_of_month",
        "quarter", "fl_number", "crs_elapsed_time", "distance", "dep_hour", "arr_hour",
        "dep_delay", "is_weekend", "is_holiday", "is_near_holiday", "is_holiday_period",
        "airline_name", "airline_code", "origin_airport_code",
        "destination_airport_code", "season",
    ]

    def test_row_satisfies_the_pipeline_contract(self, flight_payload):
        # dep_sequence_in_day and schedule_padding are rebuilt at scoring time
        # from Silver; everything else has to be present here.
        row = flight_to_row(flight_payload[0])
        missing = [c for c in self.PIPELINE_INPUTS if c not in row]
        assert not missing, f"missing pipeline inputs: {missing}"

    def test_day_of_week_uses_sparks_convention(self, flight_payload):
        # 2026-09-14 is a Monday; Spark numbers Monday 2, Python numbers it 0.
        assert flight_to_row(flight_payload[0])["day_of_week"] == 2

    def test_hours_derive_from_scheduled_utc(self, flight_payload):
        row = flight_to_row(flight_payload[0])
        assert row["dep_hour"] == row["crs_dep_time"] // 100
        assert row["arr_hour"] == row["crs_arr_time"] // 100

    def test_quarter_and_season_agree_with_the_month(self, flight_payload):
        row = flight_to_row(flight_payload[0])
        assert row["quarter"] == (row["flight_month"] - 1) // 3 + 1
        assert row["season"] == "Fall"          # September

    def test_weekend_flag_matches_day_of_week(self, flight_payload):
        row = flight_to_row(flight_payload[0])
        assert row["is_weekend"] == int(row["day_of_week"] in (1, 7))

    def test_holiday_flags_are_integers(self, flight_payload):
        row = flight_to_row(flight_payload[0])
        for flag in ("is_holiday", "is_near_holiday", "is_holiday_period"):
            assert row[flag] in (0, 1)


class TestDepartureToRow:
    def test_fills_the_origin_the_caller_knows(self, airport_payload):
        rows = [departure_to_row(d, "ATL") for d in airport_payload["departures"]]
        rows = [r for r in rows if r]
        assert rows, "fixture produced no rows"
        assert all(r["origin_airport_code"] == "ATL" for r in rows)

    def test_destination_comes_from_the_movement_block(self, airport_payload):
        rows = [departure_to_row(d, "ATL") for d in airport_payload["departures"]]
        dests = {r["destination_airport_code"] for r in rows if r}
        assert "IAH" in dests

    def test_the_route_pool_is_filterable(self, airport_payload):
        rows = [r for r in (departure_to_row(d, "ATL")
                            for d in airport_payload["departures"]) if r]
        to_iah = [r for r in rows if r["destination_airport_code"] == "IAH"]
        assert len(to_iah) == 2, "recorded window held exactly two ATL->IAH departures"


class TestClientConstruction:
    def test_requires_a_key(self):
        with pytest.raises(ValueError):
            AeroDataBoxClient("")

    def test_rejects_a_backwards_window(self):
        c = AeroDataBoxClient("k")
        now = datetime(2026, 9, 14, 12, 0)
        with pytest.raises(ValueError):
            c.airport_departures("ATL", now, now - timedelta(hours=1))

    def test_rejects_a_window_over_the_api_cap(self):
        c = AeroDataBoxClient("k")
        now = datetime(2026, 9, 14, 12, 0)
        with pytest.raises(ValueError):
            c.airport_departures("ATL", now, now + timedelta(hours=13))

    def test_quota_starts_empty(self):
        assert AeroDataBoxClient("k").last_quota == {}
