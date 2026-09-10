# Live data strategy: matching the source to the flight phase

Design note for the scoring path. Nothing here is implemented yet — it is the argument for
what to build next and, as importantly, what not to.

## The problem with the current shape

`06_api_ingest` calls AviationStack's `/flights` endpoint once per origin/destination pair:

```python
params = LookupParams(dep_iata="ATL", arr_iata="LAX")
payload = fetch_flights(params, access_key)      # one call, one route
```

The dataset contains **379 origins, 379 destinations and 7,675 distinct routes**. Covering
them once costs 7,675 calls; refreshing every fifteen minutes costs that again, every
fifteen minutes. The free tier is nowhere near this, and the shape does not improve with a
paid tier — it just becomes expensive rather than impossible.

The fix is not a bigger quota. It is noticing that the two models need different things at
different rates, and that one call can answer for thousands of aircraft.

## The two models have different data needs

| | Pre-departure | In-flight |
|---|---|---|
| Needs | The published schedule | Actual departure state |
| Known | Months ahead | Only after pushback |
| Refresh | Once per day | Minutes |
| Population | Every flight not yet departed | Only flights already airborne |

**The pre-departure model needs no live data at all.** Airline, route, scheduled times,
distance — all of it is schedule. So are all four congestion features added in `03_silver`:
departures per origin-hour, bank density, sequence in day, schedule padding.

That last point is a constraint worth making explicit. Those features are computed *across*
an airport's whole day, so scoring a single flight requires the entire day's schedule at its
origin. Fetching one flight at a time cannot produce them at any price. The unit of
ingestion has to be **airport-day**, not flight.

**The in-flight model needs one number**: departure delay. It only exists once the aircraft
has pushed back, and it only matters for aircraft currently in the air.

## Where OpenSky changes the arithmetic

[OpenSky Network](https://opensky-network.org/) is a free, community ADS-B receiver network
with a REST API. Two properties matter here.

**`GET /api/states/all` returns every aircraft the network is currently tracking, in a single
call.** Not one flight — the whole tracked airspace, thousands of state vectors, one request.
That inverts the cost model: the call count stops scaling with the number of flights and
starts scaling only with how often you poll.

**Every state vector carries an `on_ground` boolean.** That is precisely the ground/air split
the two models need, and it arrives free with data already being fetched. No separate query,
no per-flight classification: `on_ground = true` is the pre-departure population,
`on_ground = false` is the in-flight population.

State vectors also carry position, altitude, and velocity, which enables something the
schedule cannot give: **observed** congestion. Counting airborne aircraft inside a radius of
a destination airport measures arrival congestion as it actually is, where
`sched_deps_origin_hour` only measures congestion as it was planned. Those are different
signals and the gap between them — planned versus actual — is itself informative.

## Proposed architecture

**Layer 1 — schedule, once per day, AviationStack.**
One call per tracked airport per day, `dep_iata` set and `flight_status=scheduled`. Tracking
the 30 busiest airports costs roughly 30–60 calls per day with pagination. Yields every
flight to score plus everything the congestion features need.

**Layer 2 — live state, one call per poll, OpenSky.**
A single `/states/all` per cycle, joined to the scheduled flights by callsign. Split on
`on_ground`. At a 10-minute cadence that is ~144 calls per day for the entire fleet.

**Layer 3 — authoritative delay, sparingly, AviationStack.**
Only for flights OpenSky shows as newly airborne, and only if the derived delay is not good
enough (see below).

The saving is structural rather than incremental: live coverage goes from
`flights × polls` calls to `polls` calls. For 5,000 tracked flights at a 10-minute cadence
that is roughly 720,000 calls per day against roughly 150 — and each of those 150 returns
thousands of records rather than one.

## The trap: gate delay is not wheels-off delay

This is the part that would quietly break the model, so it is worth being precise.

The model is trained on BTS `DEP_DELAY`, which is **gate departure delay** — actual gate-out
minus scheduled gate-out. OpenSky observes when an aircraft stops being `on_ground`, which is
**wheels-off**. The two differ by taxi-out time, and BTS records that separately as
`TAXI_OUT` precisely because it is substantial and variable: minutes at a small field, far
longer at a congested hub.

Feeding a wheels-off delay into a model trained on gate delay is train/serve skew that is
*worst exactly where delays matter most*. It would not raise an error. It would raise the
apparent delay at busy airports and the model would look like it was working.

Two ways out, in order of preference:

1. **Take the delay from AviationStack**, which reports `departure.delay` on BTS-compatible
   gate semantics, and use OpenSky only to decide *which* flights are worth asking about.
   The expensive call is then made once per departed flight instead of once per flight per
   poll.
2. **Correct the OpenSky observation** by subtracting a per-airport median taxi-out. That
   median is already available — `TAXI_OUT` is one of the 33 columns in Bronze, so it can be
   computed offline from data in hand and broadcast as a small dimension table, exactly like
   the date dimension in `03_silver`. Cheaper, and approximate; the residual error is the
   variance of taxi-out around its median, which should be reported rather than ignored.

Whichever is chosen, the README should say which delay definition is being served, because
"departure delay" names two different quantities here.

## Recommendation

Adopt layers 1 and 2. They cut live call volume by roughly three orders of magnitude, and
`on_ground` gives the phase split for free — which is the thing that makes a two-model
architecture operationally coherent rather than just conceptually neat.

Prefer option 1 for the delay value: correctness of the input matters more than the call
saving, and the OpenSky filter already reduces the AviationStack volume to departures only.

Two things not to do:

- **Do not add a weather API.** The `02_eda` §5 decomposition puts extreme weather at 5.8%
  of delay minutes. Even crediting weather with all of NAS the ceiling is ~25%, and a
  pre-departure model would have a forecast at inference time rather than the observation it
  was trained on. Late aircraft plus NAS is 57.7% and needs no external data at all.
- **Do not poll faster than the model's decision horizon.** A pre-departure prediction is
  useful hours ahead; refreshing it every minute produces churn, not information.

## Honest limitations

- OpenSky coverage is crowdsourced. It is strong over Europe and the continental US and
  patchy elsewhere; for a US-domestic BTS dataset that is acceptable, and it should be stated
  rather than assumed.
- OpenSky rate limits differ between anonymous and registered use and have changed over
  time. Confirm the current published limits before sizing a polling cadence rather than
  trusting the numbers above.
- Callsign-to-flight-number matching is not exact. Carriers do not always broadcast a
  callsign that maps cleanly to the marketed flight number, and codeshares make it
  many-to-one. Expect a match rate below 100% and measure it rather than assuming it.
- None of this is required for the project to be complete. The historical pipeline and both
  models work on BTS data alone; this is the path to a live system, and its absence is a
  scope boundary rather than a defect.
