# Data

Every number on this page is produced by `headroom data build` and was last measured on
2026-09-12 against the record as published that day. Nothing raw is committed: the loader
fetches, aggregates, and writes daily counts.

## Source

| | |
|---|---|
| Dataset | EMS Incident Dispatch Data, NYC Open Data, `76xm-jjuj` |
| Publisher | City of New York, Fire Department (FDNY) |
| Terms | NYC Open Data Terms of Use. Public domain, attribution requested, no restriction on derived aggregates. |
| Retrieved | 2026-09-12 |
| Record | 2005-01-01 to 2026-06-30, 7,851 days |
| Incidents | 29,977,935 |

The dataset is one row per emergency medical dispatch incident, with an incident
timestamp, a borough and a dispatch area. What this project uses is the daily count per
dispatch area; nothing about an individual incident is read, stored or modelled.

### How it is fetched

The aggregation runs on Socrata, not here. One grouped query per calendar year returns
the daily counts directly, about 13,000 rows a year, and each completed year is cached on
disk so a re-run does not refetch it. Downloading thirty million incident rows to count
them locally would take hours and several gigabytes to arrive at the same 279,363 grouped
rows. The current year is refetched every time, because it grows.

## What the loader does, and what each step costs

### Boroughs

`RICHMOND / STATEN ISLAND` is renamed to `STATEN ISLAND`. Rows with no usable borough or
no dispatch area are dropped.

### One borough per dispatch area

The borough is recorded on each incident and on a small number of them it disagrees with
the dispatch area: incidents in area `M9` recorded in the Bronx, `K2` in Manhattan, almost
always one or two at a time. A dispatch area is a fixed piece of geography, so these are
attribute errors on individual incidents rather than an area that moved, and the
hierarchy needs each area under exactly one borough or the summing matrix is not a matrix.

Each area is therefore assigned to the borough it is recorded under most often across the
whole record, and every incident is counted under that borough.

**Cost of the decision: 45,007 incidents move, 0.150 percent of the record.** The areas
that move most are `Q6` (10,235 incidents), `K7` (3,992), `K6` (2,869), `M7` (2,407),
`K2` (2,402) and `M1` (2,096). No incident is lost, only recounted.

### Which areas become leaves

44 dispatch area codes appear in the record. 31 become leaves. An area has to pass two
tests, because two different things disqualify one.

**Span.** Its first-to-last span must cover at least 98 percent of the window. This drops
codes that opened or closed inside the record and cannot be backtested across it. Span
rather than a count of active days, because an area with no incident on a quiet night
recorded a zero, not an absence.

**Volume.** It must average at least one incident a day. A stray code appearing once or
twice a year for twenty years spans the whole window and is still not a dispatch area.

The volume distribution makes the second test unambiguous:

| Area | Mean incidents a day | Days with none |
|---|---:|---:|
| `X1` | 0.38 | 95.8% |
| `CW` | 0.40 | 76.1% |
| `S3` (quietest real area) | 20.55 | 0% |
| `B2` (busiest area) | 270.27 | 0% |

The gap between 0.40 and 20.55 is nearly two orders of magnitude, so any threshold
between them selects the same 31 leaves. One incident a day is not a tuned number.

**Cost of the decision: 13 codes dropped, 6,736 incidents, 0.0225 percent of the
record.** The dropped codes are `BO`, `C1`, `C2`, `CW`, `MO`, `PD`, `T1`, `T2`, `X1`,
`X2`, `X3`, `X4`, `X5`.

Because those areas are dropped, **the city series is the sum of the 31 retained leaves,
not the sum of every incident.** Defining the root any other way would make the hierarchy
incoherent before a single forecast was made, and a reconciliation result on an
incoherent hierarchy is not a result. The difference is exactly the 0.0225 percent above.

## The panel

| | |
|---|---:|
| Nodes | 37 |
| City | 1 |
| Boroughs | 5 |
| Dispatch areas (leaves) | 31 |
| Days | 7,851 |
| Coherence error of the panel | 0.0 |

Leaves by borough: Bronx 5, Brooklyn 7, Manhattan 9, Queens 7, Staten Island 3.

A day on which a retained area recorded nothing is filled with zero, which is a real
count. **No day is missing from the record**: every one of the 7,851 days between the
first and last incident carries data.

City-wide daily arrivals average 3,818, with a minimum of 2,582 and a maximum of 6,526.

## The shifts

Both dates were named in `PLAN.md` before any model was fitted. They are not chosen after
seeing which method won, which matters because the headline coverage chart is plotted
through one of them.

| Marker | Date | What it is |
|---|---|---|
| `sandy` | 2012-10-29 | Hurricane Sandy landfall |
| `covid` | 2020-03-01 | The start of the March 2020 surge |

The 2020 surge is the headline. The eight busiest days in twenty-one years of record all
fall between 2020-03-26 and 2020-04-06, the busiest being 2020-03-30 at 6,526 incidents
against a long-run daily mean of 3,818. Across 2020-03-01 to 2020-04-15 the city averaged
4,912 incidents a day against 4,109 over the same calendar window in 2019, a 19.5 percent
rise sustained for six weeks.

That shape is why the outlier check uses a median and a median absolute deviation rather
than a mean and a standard deviation. Six weeks of elevated demand inflate a standard
deviation enough to pull the surge back inside a six-sigma threshold; the median absolute
deviation is not moved by them.

## Secondary dataset

NHS England monthly emergency-department attendances by provider is optional and is the
first thing dropped if the schedule tightens (`PLAN.md` section 5). It is not loaded yet.
