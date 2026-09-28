"""Unit tests for src.faa_nas, against a recorded live response."""

from pathlib import Path

import pytest

from src.faa_nas import NasStatus, parse_status, summarise

FIXTURE = Path(__file__).parent / "fixtures" / "faa_nas_status.xml"


@pytest.fixture(scope="module")
def status():
    return parse_status(FIXTURE.read_text(encoding="utf-8"))


class TestParseStatus:
    def test_reads_the_update_time(self, status):
        assert "2026" in status.updated

    def test_flattens_every_delay_type(self, status):
        kinds = {c.kind for c in status.conditions}
        assert {"ground_delay", "arrival_departure_delay", "closure"} <= kinds

    def test_every_condition_names_an_airport(self, status):
        assert status.conditions
        for c in status.conditions:
            assert c.airport and c.airport == c.airport.upper()

    def test_ground_delay_carries_average_and_max(self, status):
        gdp = [c for c in status.conditions if c.kind == "ground_delay"]
        assert gdp, "fixture should contain ground delay programmes"
        assert any("avg" in c.detail for c in gdp)

    def test_closures_carry_a_window(self, status):
        closures = [c for c in status.conditions if c.kind == "closure"]
        assert closures
        assert any("closed" in c.detail or "reopens" in c.detail for c in closures)

    def test_repeated_delay_type_blocks_are_all_kept(self, status):
        # The document repeats <Delay_type><Name>Airport Closures</Name> more
        # than once. Navigating by name and taking the first would silently drop
        # the rest, so the parser flattens instead.
        closures = [c for c in status.conditions if c.kind == "closure"]
        assert len(closures) >= 3, f"expected closures from every block, got {len(closures)}"

    def test_handles_an_empty_document(self):
        empty = parse_status("<AIRPORT_STATUS_INFORMATION></AIRPORT_STATUS_INFORMATION>")
        assert empty.conditions == []
        assert empty.airports_affected == set()


class TestLookup:
    def test_for_airport_is_case_and_space_insensitive(self, status):
        code = next(iter(status.airports_affected))
        assert status.for_airport(f"  {code.lower()}  ")

    def test_unaffected_airport_returns_nothing(self, status):
        assert status.for_airport("ZZZ") == []

    def test_describe_is_human_readable(self, status):
        line = status.conditions[0].describe()
        assert status.conditions[0].airport in line
        assert "_" not in line, "kind should be spelled out, not snake_case"


class TestSummarise:
    def test_only_reports_requested_airports(self, status):
        code = next(iter(status.airports_affected))
        lines = summarise(status, [code, "ZZZ"])
        assert lines and all(code in ln for ln in lines)

    def test_preserves_the_order_asked_for(self, status):
        codes = sorted(status.airports_affected)[:2]
        if len(codes) < 2:
            pytest.skip("fixture has fewer than two affected airports")
        lines = summarise(status, codes)
        assert lines[0].startswith(codes[0])

    def test_deduplicates_and_ignores_blanks(self, status):
        code = next(iter(status.airports_affected))
        once = summarise(status, [code])
        twice = summarise(status, [code, code, "", None])
        assert once == twice

    def test_clean_airports_produce_no_lines(self):
        assert summarise(NasStatus(), ["ATL", "IAH"]) == []
