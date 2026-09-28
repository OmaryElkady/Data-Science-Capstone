"""Unit tests for src.opensky.

Run against a recorded snapshot in tests/fixtures/opensky_sample.json, trimmed
from a real continental-US response, so they need no network and no credentials.
The fixture deliberately retains the awkward cases: space-padded callsigns,
blank callsigns, and both phases.
"""

import json
from pathlib import Path

import pytest

from src.opensky import (
    CONUS_BBOX,
    STATE_FIELDS,
    match_to_schedule,
    normalise_callsign,
    parse_states,
    split_by_phase,
)

FIXTURE = Path(__file__).parent / "fixtures" / "opensky_sample.json"


@pytest.fixture(scope="module")
def payload():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rows(payload):
    return parse_states(payload)


class TestNormaliseCallsign:
    def test_strips_opensky_padding(self):
        assert normalise_callsign("DAL1234 ") == "DAL1234"
        assert normalise_callsign("  AAL99  ") == "AAL99"

    def test_uppercases(self):
        assert normalise_callsign("dal1234") == "DAL1234"

    def test_blank_becomes_none(self):
        # ~1.3% of live vectors carry only padding. Those aircraft cannot be
        # matched to a schedule and must not become the callsign "".
        assert normalise_callsign("        ") is None
        assert normalise_callsign("") is None
        assert normalise_callsign(None) is None


class TestParseStates:
    def test_names_every_positional_field(self, rows):
        assert rows, "fixture is empty"
        for name in STATE_FIELDS:
            assert name in rows[0]

    def test_attaches_snapshot_time(self, payload, rows):
        assert all(r["snapshot_time"] == payload["time"] for r in rows)

    def test_callsigns_are_normalised(self, rows):
        for r in rows:
            cs = r["callsign"]
            assert cs is None or (cs == cs.strip() and cs == cs.upper())

    def test_tolerates_short_vectors(self):
        # OpenSky has added fields over time; a client that hard-indexes 16
        # breaks the day a 16-element vector arrives.
        short = {"time": 1, "states": [["abc123", "DAL1 ", "United States"]]}
        row = parse_states(short)[0]
        assert row["icao24"] == "abc123"
        assert row["callsign"] == "DAL1"
        assert row["position_source"] is None

    def test_handles_empty_payload(self):
        assert parse_states({"time": 1}) == []
        assert parse_states({"time": 1, "states": None}) == []


class TestSplitByPhase:
    def test_partitions_every_row(self, rows):
        ground, airborne = split_by_phase(rows)
        assert len(ground) + len(airborne) == len(rows)

    def test_ground_is_strictly_on_ground(self, rows):
        ground, airborne = split_by_phase(rows)
        assert all(r["on_ground"] is True for r in ground)
        assert all(r["on_ground"] is not True for r in airborne)

    def test_fixture_contains_both_phases(self, rows):
        ground, airborne = split_by_phase(rows)
        assert ground and airborne, "fixture should exercise both branches"

    def test_null_on_ground_counts_as_airborne(self):
        # Safer direction: the in-flight path re-checks against the schedule,
        # while the pre-departure path would score an already-departed flight.
        rows = [{"on_ground": None, "callsign": "DAL1"}]
        ground, airborne = split_by_phase(rows)
        assert not ground and len(airborne) == 1


class TestMatchToSchedule:
    def test_matches_on_icao_designator(self, rows):
        present = [r["callsign"] for r in rows if r["callsign"]][:5]
        matched, report = match_to_schedule(rows, present)
        assert {r["callsign"] for r in matched} == set(present)
        assert report["matched"] == len(matched)

    def test_ignores_padding_and_case_on_both_sides(self, rows):
        one = next(r["callsign"] for r in rows if r["callsign"])
        matched, _ = match_to_schedule(rows, [f"  {one.lower()}  "])
        assert matched and matched[0]["callsign"] == one

    def test_rows_without_callsign_never_match(self, rows):
        matched, report = match_to_schedule(rows, ["DAL1234", "AAL99"])
        assert all(r["callsign"] for r in matched)
        assert report["without_callsign"] > 0, "fixture should include blanks"

    def test_empty_schedule_matches_nothing(self, rows):
        matched, report = match_to_schedule(rows, [])
        assert matched == []
        assert report["match_rate_of_scheduled"] == 0.0

    def test_report_counts_are_consistent(self, rows):
        _, report = match_to_schedule(rows, ["DAL1234"])
        assert report["states_seen"] == len(rows)
        assert report["with_callsign"] + report["without_callsign"] == len(rows)


class FakeClock:
    """Controllable monotonic clock so expiry can be tested without waiting."""

    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


class CountingTokenFn:
    def __init__(self, expires_in=1800):
        self.calls = 0
        self.expires_in = expires_in

    def __call__(self, client_id, client_secret):
        self.calls += 1
        return f"token-{self.calls}", self.expires_in


class TestOpenSkyClient:
    """The credentials are long-lived; only the minted token expires."""

    def _client(self, **kw):
        from src.opensky import OpenSkyClient
        clock = kw.pop("clock", FakeClock())
        fn = kw.pop("token_fn", CountingTokenFn())
        return OpenSkyClient("id", "secret", token_fn=fn, clock=clock, **kw), fn, clock

    def test_requires_both_credentials(self):
        from src.opensky import OpenSkyClient
        with pytest.raises(ValueError):
            OpenSkyClient("", "secret")
        with pytest.raises(ValueError):
            OpenSkyClient("id", "")

    def test_mints_once_and_reuses(self):
        client, fn, _ = self._client()
        assert client.token() == "token-1"
        for _ in range(5):
            assert client.token() == "token-1"
        assert fn.calls == 1, "a valid token must not be re-fetched"

    def test_refreshes_after_expiry(self):
        client, fn, clock = self._client()
        assert client.token() == "token-1"
        clock.advance(1801)
        assert client.token() == "token-2"
        assert fn.calls == 2

    def test_refreshes_inside_the_safety_margin(self):
        # Valid for another 60s, but the margin is 120s: refresh rather than risk
        # the token dying mid-request.
        client, fn, clock = self._client(refresh_margin_seconds=120)
        client.token()
        clock.advance(1800 - 60)
        assert client.expired
        assert client.token() == "token-2"

    def test_does_not_refresh_outside_the_margin(self):
        client, fn, clock = self._client(refresh_margin_seconds=120)
        client.token()
        clock.advance(1800 - 300)
        assert not client.expired
        assert client.token() == "token-1"
        assert fn.calls == 1

    def test_invalidate_forces_reauth(self):
        client, fn, _ = self._client()
        client.token()
        client.invalidate()
        assert client.token() == "token-2"
        assert fn.calls == 2

    def test_short_lived_token_refreshes_sooner(self):
        client, fn, clock = self._client(
            token_fn=CountingTokenFn(expires_in=300), refresh_margin_seconds=30
        )
        client.token()
        clock.advance(280)
        assert client.token() == "token-2"

    def test_refresh_count_is_observable(self):
        client, fn, clock = self._client()
        for _ in range(3):
            client.token()
            clock.advance(1801)
        assert client.refresh_count == 3


def test_conus_bbox_is_well_formed():
    assert CONUS_BBOX["lamin"] < CONUS_BBOX["lamax"]
    assert CONUS_BBOX["lomin"] < CONUS_BBOX["lomax"]
    # Longitudes are western-hemisphere negative; a sign slip here silently
    # returns an empty airspace rather than an error.
    assert CONUS_BBOX["lomin"] < 0 and CONUS_BBOX["lomax"] < 0


class TestDeriveSchedule:
    """OpenSky publishes no timetable, so one is derived from observed departures.

    The fixture is a recorded slice of real KATL departures.
    """

    @pytest.fixture(scope="class")
    def departures(self):
        import json
        path = Path(__file__).parent / "fixtures" / "opensky_departures.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def _records(self, callsign, times):
        """Departure records for one callsign at given UTC times (hh, mm)."""
        import calendar
        out = []
        for day, (hh, mm) in enumerate(times, start=1):
            ts = calendar.timegm((2026, 9, day, hh, mm, 0, 0, 0, 0))
            out.append({"callsign": f"{callsign} ", "firstSeen": ts,
                        "estDepartureAirport": "KATL"})
        return out

    def test_median_not_mean(self):
        from src.opensky import derive_schedule
        # Four departures near 10:00 and one three hours late. A mean would put
        # the schedule at a time the flight has never actually left.
        recs = self._records("DAL1", [(10, 0), (10, 5), (10, 10), (10, 15), (13, 0)])
        sched = derive_schedule(recs)["DAL1"]
        assert sched["median_minute"] == 10 * 60 + 10
        assert sched["median_hhmm"] == 1010

    def test_spread_is_reported(self):
        from src.opensky import derive_schedule
        recs = self._records("DAL2", [(10, 0), (10, 30), (11, 0)])
        assert derive_schedule(recs)["DAL2"]["spread_minutes"] == 60

    def test_spread_separates_reliable_from_erratic(self):
        from src.opensky import derive_schedule
        steady = self._records("DAL3", [(8, 0), (8, 5), (8, 10)])
        erratic = self._records("DAL4", [(8, 0), (9, 30), (10, 45)])
        sched = derive_schedule(steady + erratic)
        assert sched["DAL3"]["spread_minutes"] < sched["DAL4"]["spread_minutes"]

    def test_respects_the_observation_floor(self):
        from src.opensky import derive_schedule
        recs = self._records("DAL5", [(9, 0), (9, 10)])
        assert "DAL5" not in derive_schedule(recs, min_observations=3)
        assert "DAL5" in derive_schedule(recs, min_observations=2)

    def test_ignores_records_with_no_callsign_or_time(self):
        from src.opensky import derive_schedule
        recs = self._records("DAL6", [(9, 0), (9, 5), (9, 10)])
        recs += [{"callsign": "   ", "firstSeen": 1}, {"callsign": "DAL6"}]
        sched = derive_schedule(recs)
        assert sched["DAL6"]["observations"] == 3

    def test_callsigns_are_normalised(self):
        from src.opensky import derive_schedule
        recs = self._records("dal7", [(9, 0), (9, 5), (9, 10)])
        assert "DAL7" in derive_schedule(recs)

    def test_runs_on_the_recorded_fixture(self, departures):
        from src.opensky import derive_schedule
        sched = derive_schedule(departures, min_observations=1)
        assert sched, "fixture produced no schedule"
        for entry in sched.values():
            assert 0 <= entry["median_minute"] < 24 * 60
            assert entry["spread_minutes"] >= 0
            assert entry["observations"] >= 1

    def test_empty_input(self):
        from src.opensky import derive_schedule
        assert derive_schedule([]) == {}
