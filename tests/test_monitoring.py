"""Unit tests for src.monitoring: what counts as a graded forecast, and how it is scored."""

import pandas as pd
import pytest

from src.monitoring import (
    breakdown, claim_status, nas_affects_airlines, reliability, roc_auc, scorecard,
    wilson, with_outcome,
)

NOW = pd.Timestamp("2026-10-05T12:00:00Z")


def _row(**kw):
    base = {
        "flight": "DL1", "flight_date": "2026-10-04", "row_kind": "watch",
        "recommended_model": "pre_departure", "flight_status": "Expected",
        "is_retrospective": False, "arrival_delay": None,
        "prediction_timestamp": "2026-10-04T09:05:00Z",
        "scheduled_departure_utc": "2026-10-04T15:00:00Z",
        "delay_probability_pct": 20.0, "will_be_delayed": 1, "advisory_flag": 0,
    }
    base.update(kw)
    return base


def _status(*rows):
    return list(claim_status(pd.DataFrame(rows), NOW, lookback_days=3))


class TestClaimStatus:
    def test_forecast_before_departure_with_an_outcome_is_graded(self):
        assert _status(_row(arrival_delay=-5.0)) == ["graded"]

    def test_recent_flight_without_an_outcome_is_awaiting(self):
        assert _status(_row()) == ["awaiting"]

    def test_old_flight_without_an_outcome_has_none(self):
        old = _row(scheduled_departure_utc="2026-09-28T15:00:00Z",
                   prediction_timestamp="2026-09-28T09:00:00Z")
        assert _status(old) == ["no_outcome"]

    def test_alternative_without_an_arrival_time_is_context(self):
        assert _status(_row(row_kind="alternative")) == ["context"]

    def test_cancelled_is_never_graded(self):
        assert _status(_row(flight_status="Canceled", arrival_delay=0.0)) == ["cancelled"]

    def test_flagged_retrospective_is_not_graded(self):
        assert _status(_row(is_retrospective=True, arrival_delay=40.0)) == ["retrospective"]

    def test_schedule_only_forecast_made_after_departure_is_retrospective(self):
        # The Oct 1 flights re-scored on Oct 4: same model, three days late.
        late = _row(prediction_timestamp="2026-10-04T16:00:00Z", arrival_delay=3.0)
        assert _status(late) == ["retrospective"]

    def test_in_flight_forecast_after_departure_is_legitimate(self):
        airborne = _row(recommended_model="in_flight", arrival_delay=9.0,
                        prediction_timestamp="2026-10-04T16:00:00Z")
        assert _status(airborne) == ["graded"]

    def test_epoch_seconds_are_read_as_utc(self):
        row = _row(arrival_delay=1.0,
                   prediction_timestamp=pd.Timestamp("2026-10-04T09:05Z").timestamp(),
                   scheduled_departure_utc=pd.Timestamp("2026-10-04T15:00Z").timestamp())
        assert _status(row) == ["graded"]


class TestScores:
    def test_auc_is_one_for_perfect_ranking(self):
        assert roc_auc(pd.Series([0.1, 0.2, 0.8, 0.9]), pd.Series([0, 0, 1, 1])) == 1.0

    def test_auc_is_a_half_when_every_score_ties(self):
        assert roc_auc(pd.Series([0.2] * 4), pd.Series([0, 1, 0, 1])) == 0.5

    def test_auc_needs_both_outcomes(self):
        assert roc_auc(pd.Series([0.1, 0.9]), pd.Series([0, 0])) is None

    def test_wilson_stays_inside_zero_and_one(self):
        lo, hi = wilson(0, 10)
        assert lo == 0.0 and 0 < hi < 0.35

    def test_wilson_is_wide_on_few_flights(self):
        lo, hi = wilson(1, 4)
        assert hi - lo > 0.5


@pytest.fixture
def graded():
    # 10 flights forecast at 20%: two late, at 30 and 15 minutes; 14 is on time.
    delays = [30.0, 15.0, 14.0, -5.0, -10.0, 0.0, 2.0, -20.0, 5.0, -1.0]
    return with_outcome(pd.DataFrame([
        _row(flight=f"DL{i}", arrival_delay=d, will_be_delayed=int(i < 4))
        for i, d in enumerate(delays)
    ]))


class TestScorecard:
    def test_the_dot_rule_is_fifteen_minutes(self, graded):
        assert graded["late"].tolist()[:3] == [1, 1, 0]

    def test_a_calibrated_constant_matches_its_own_baseline(self, graded):
        card = scorecard(graded)
        assert card["observed_rate"] == pytest.approx(0.2)
        assert card["brier"] == pytest.approx(card["brier_constant"])
        assert card["brier_skill"] == pytest.approx(0.0)

    def test_the_50_percent_call_is_compared_with_always_on_time(self, graded):
        card = scorecard(graded)
        assert card["called_late_50"] == 0
        assert card["accuracy_50"] == pytest.approx(card["accuracy_always_on_time"])

    def test_counts_what_the_f1_cut_caught(self, graded):
        card = scorecard(graded)
        assert (card["flagged_f1"], card["caught_f1"]) == (4, 2)

    def test_counts_distinct_flights_and_days(self, graded):
        card = scorecard(graded)
        assert (card["forecasts"], card["flights"], card["days"]) == (10, 10, 1)

    def test_empty_input(self):
        assert scorecard(with_outcome(pd.DataFrame([_row()]).iloc[0:0]))["forecasts"] == 0


class TestTables:
    def test_reliability_bins_hold_every_forecast(self, graded):
        bins = reliability(graded)
        assert bins["forecasts"].sum() == len(graded)
        assert bins["forecasts"].tolist() == [2] * 5    # equal counts, even when p ties
        assert (bins["ci_lo"] <= bins["observed_rate"]).all()
        assert (bins["observed_rate"] <= bins["ci_hi"]).all()

    def test_breakdown_by_route(self, graded):
        graded = graded.assign(route=["A"] * 5 + ["B"] * 5)
        table = breakdown(graded, "route").set_index("route")
        assert table.loc["A", "late"] == 2 and table.loc["B", "late"] == 0


class TestNas:
    def test_ga_only_closure_is_not_degraded_airspace(self):
        lax = ("LAX: closure (closed May 27 at 18:26 UTC. reopens May 28 at 16:00 UTC. — "
               "!LAX 05/277 LAX AD AP CLSD TO NON SKED TRANSIENT GA ACFT EXC 24HR PPR)")
        assert not nas_affects_airlines(lax)

    def test_a_ground_delay_is(self):
        assert nas_affects_airlines("LGA: ground delay (avg 43 minutes / max 1 hour)")

    def test_one_real_condition_beside_a_ga_closure_counts(self):
        assert nas_affects_airlines(
            "LAX: closure (CLSD TO NON SKED TRANSIENT GA ACFT); SFO: arrival departure delay (16 minutes)")

    def test_nothing_recorded(self):
        assert not nas_affects_airlines(None)
        assert not nas_affects_airlines(float("nan"))


def test_the_figure_draws(graded):
    import matplotlib
    matplotlib.use("Agg")
    from src.monitoring import forward_test_figure
    fig = forward_test_figure(graded, f1_cut=0.18, label="pre-departure")
    assert len(fig.axes) == 2
