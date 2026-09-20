# The neural verdict

Written whichever way it falls, which is the rule this project set itself. This is where
the neural models did and did not earn their complexity. Every number is from `docs/methods.md`, which says how it was
produced; all comparisons are paired over the same 911 weekly origins (2009-01-05 to
2026-06-15), with 95 percent block-bootstrap intervals.

**Status, 2026-09-18:** N-HiTS, PatchTST and TimesFM all measured.

**The short version.** Neither model trained from scratch on this data beats the best
statistical model, and they fail in completely different ways. N-HiTS is broken here:
miscalibrated and beaten at every level. PatchTST is a good forecaster that ties ETS at the
city and the borough, loses narrowly at the dispatch areas, and costs about seventeen times
as much to fit. **The one that does not lose is the one that was never trained on this data
at all**: TimesFM, used zero-shot, beats ETS at every level on the window after its
pretraining data ends and ties the global LightGBM model there. That result comes with a
caveat the other two do not have, and the caveat is in section "What the TimesFM windows
disagree about" rather than in a footnote.

## The question, and why LightGBM is in it

"Did a neural model beat the best statistical model" confounds two things: a neural
forecaster here learns one model across all 37 series, and it is deep. A global LightGBM
model on lag and calendar features is global without being deep, so it separates the two.
It was also given something no other model had, the holiday calendar of the day being
forecast.

## The result

CRPS, paired against ETS. Negative is better than ETS.

| Level | ETS | LightGBM minus ETS | PatchTST minus ETS | N-HiTS minus ETS |
|---|---|---|---|---|
| City CRPS | 135.40 | -1.65 [-6.81, +5.98] | **-3.45 [-6.72, +2.48]** | +28.39 [+21.72, +40.03] |
| Borough CRPS | 32.39 | -0.48 [-1.25, +0.62] | -0.31 [-0.89, +0.67] | +7.19 [+5.97, +9.39] |
| Dispatch area CRPS | 8.274 | +0.007 [-0.067, +0.098] | +0.103 [+0.041, +0.203] | +1.98 [+1.82, +2.29] |
| Coverage at 90% | 0.904 to 0.918 | 0.897 to 0.903 | 0.896 to 0.901 | 0.581 to 0.676 |
| Fitting, 964 weekly origins | about 1.6 h | about 2.0 h | about 27.3 h, refitted every 13 | about 18.4 h, refitted every 4 |

## What the numbers say

1. **At the city and the borough, three unrelated model families are indistinguishable.**
   ETS, a global LightGBM and a patch transformer all land within a couple of CRPS of each
   other, and every paired interval straddles zero. PatchTST has the lowest city CRPS any
   model in this project has produced, 131.95, and it still cannot be claimed as a win.
2. **At the dispatch areas the cheapest model wins.** PatchTST is behind ETS by +0.103
   [+0.041, +0.203], an interval clear of zero. Small, but it is the one level where the
   ranking is decided, and it goes to the statistical model.
3. **PatchTST is the best-calibrated model here, ETS included.** 0.896, 0.901, 0.900
   against a 0.90 target, from its own quantiles with no conformal step, while being
   narrower than ETS at the city (778 against 842) and the borough (190 against 194). This
   is a real and slightly surprising result: the architecture that lost produced better
   raw uncertainty than the one that won.
4. **Learning across series bought nothing, and depth bought nothing on top of it.**
   LightGBM ties ETS; PatchTST ties LightGBM at the top two levels and loses to it at the
   leaves (+0.096 [+0.005, +0.185], winning on 44.5 percent of origins). Each added layer
   of machinery left the answer where it was.
5. **N-HiTS, as configured here, is simply worse.** 21 percent behind ETS at the city, it
   does not beat seasonal naive there, and its quantiles cover 58 to 68 percent of
   outcomes where they claim 90. It is not a calibration problem alone: given
   well-calibrated conformal intervals its median is still clearly behind (+18.56
   [+12.78, +29.28] at the city). It is not the refit schedule, and it is not the seed.
6. **The compute is the verdict.** PatchTST costs about seventeen times ETS and fourteen
   times LightGBM to reach a tie at two levels and a loss at the third.

## TimesFM, zero-shot: the only model that did not lose

Reported separately because it is scored on different origins. TimesFM's weights postdate
most of this backtest, so it is judged on the **clean window**: 133 weekly origins,
2023-12-04 to 2026-06-15, every forecast day after the latest documented pretraining data.
The full-backtest numbers exist, are better still, and are not a result.

| Level | TimesFM CRPS | minus ETS | minus LightGBM |
|---|---|---|---|
| City | 125.34 | **-24.34 [-29.84, -12.74]** | -6.85 [-12.37, +1.32] |
| Borough | 31.79 | **-3.23 [-4.20, -1.43]** | -0.49 [-1.43, +0.70] |
| Dispatch area | 8.571 | **-0.171 [-0.296, -0.040]** | +0.021 [-0.044, +0.109] |

7. **A model trained on none of this data beats the model fitted to each series.** That is
   the single most surprising number in the project. It is also not the claim it looks
   like, because LightGBM beats ETS on this window too (city -17.49 [-30.15, -1.58]).
   What separates both from ETS here is the conformal distribution, rebuilt from the last
   52 origins: ETS's own quantiles cover 0.889 at the city on this window and 0.791 on the
   last 40 origins, while the conformal models hold 0.89 to 0.90 throughout.
8. **Against the fitted global model, zero-shot is a tie.** Every TimesFM minus LightGBM
   interval on the clean window straddles zero. Pretraining on other people's series
   bought as much as a gradient-boosted model trained on these ones, at no training cost
   and with no calendar. That is the finding worth taking to another dataset.
9. **Its own uncertainty is good without any help.** 80 percent nominal, from its own
   deciles with no conformal step: 0.784, 0.784, 0.788 on the clean window, 0.792 at every
   level on the full backtest. Only PatchTST does better, and PatchTST was trained here.

### What the TimesFM windows disagree about

`PLAN.md` section 2.7a set a test in advance: if TimesFM's skill is materially higher
before the weights were published than after, that points to undocumented overlap. It is.
Paired against LightGBM, at the city:

| Window | Origins | TimesFM minus LightGBM | Origins won |
|---|---:|---|---|
| Full backtest, exposed | 911 | -12.82 [-18.34, -8.74] | 64.1% |
| Between the pretraining cutoff and the release | 93 | -10.45 | 72.0% |
| After the weights were published | 40 | +1.51 | 42.5% |

The advantage shrinks as the origins move further from anything the weights could have
absorbed, and on the last stretch it is gone. **Two explanations fit and 40 origins cannot
separate them.** One is a leak: overlap with data up to the release that Google's
documentation does not cover and the corpus check could not rule out. The other is
staleness: TimesFM is frozen where LightGBM refits weekly, so any change in demand after
September 2025 is something one model follows and the other cannot, and every model in this
project scores worse on that stretch. **The clean-window result stands as reported, and the
reader is entitled to know it is the average of a better stretch and a worse one.**

Those last two rows carry no interval on purpose. The block bootstrap here lays each
resample out as whole blocks of 28 origins, and on 40 origins it returned an interval for
ETS's skill that did not contain its own point estimate. `docs/methods.md` has that
diagnostic; the report now refuses to print an interval below four blocks.

**As a rota decision** (illustrative costs, `inputs/decision.toml`), at the level the costs
imply PatchTST staffs 448.8 units a day, the same as ETS to a tenth of a unit, and costs
66.84 against ETS's 65.76: **1.6 percent more**, +1.08 [+0.43, +2.13]. It is a usable
forecaster in a way N-HiTS is not, which costs 81.92, 25 percent more than ETS. The gap
widens with the service level, to +2.09 at 90 percent and +4.53 at 95.

## What this verdict does not claim

* **Not that neural forecasting cannot work here.** Two architectures, library defaults
  deliberately not tuned on the backtest, a 112-day input window, CPU training, one seed
  for PatchTST. A tuned model, or one given the calendar features LightGBM had, might do
  better; tuning it on these origins would have made any win untrustworthy. PatchTST
  coming this close on defaults is the strongest argument in the neural models' favour
  that this project produced.
* **Not that the seed decided N-HiTS.** Measured 2026-09-15: 20 refits with a second seed,
  spread from 2009 to 2026, each forecasting the 4 origins it serves. A single fit moves a
  lot (the city CRPS of one refit changed by 10.5 at the median and 56 at most), but the
  second seed is no better on average, and it loses to ETS just the same:

  | Level | Seed 1 minus seed 0 | Seed 1 minus ETS | Refits where seed 1 is behind ETS |
  |---|---|---|---|
  | City | -6.44 [-15.35, +2.38] | +23.85 [+13.76, +33.74] | 18 of 20 |
  | Borough | -0.41 [-1.35, +0.63] | +6.02 [+4.55, +7.46] | 19 of 20 |
  | Dispatch area | -0.03 [-0.15, +0.08] | +1.83 [+1.73, +1.94] | 20 of 20 |

  Intervals are an ordinary bootstrap over the 20 refits, which are about eleven months
  apart. Seed 1's 90 percent coverage is 0.61 to 0.69, as poor as seed 0's.
* **PatchTST's seed was not tested.** N-HiTS's was, and one fit moved by up to 56 CRPS at
  the city. PatchTST is refitted 75 times rather than 241, so single-fit noise averages
  down less, and its city interval is the widest in the table. A second PatchTST seed is
  28 hours that has not been spent; the city column is where it would matter and the city
  column is already a tie.
* **Not that TimesFM is clean of the leak.** It is clean of the *documented* pretraining
  data, which is what the clean window is built from. Google does not publish the Wikipedia
  and Trends extracts, so the November 2023 cutoff for those rests on their word, and the
  post-release window is the only stretch where nothing could have leaked by any reading.
  On that stretch TimesFM is behind LightGBM. Forty origins cannot make that a finding, and
  it is not treated as one in either direction.
* **Not that TimesFM would hold up under a shift.** Both its windows fall entirely after
  March 2020, so it contributes nothing to this project's coverage-through-a-shift claim,
  and the single most likely reading of its decay on the last 40 origins is that a frozen
  model cannot follow a change that a weekly refit can. That is the thing to test before
  anyone puts a foundation model on a rota.
* **Not that any of this is tuned.** TimesFM ran on the model card's own settings with the
  project's window and horizon, one pass, no second configuration tried. That is the same
  bar the other models were held to and it cuts both ways: a tuned TimesFM might be better,
  and there is no evidence here that it would be.

## For `PLAN.md` section 9

`PLAN.md` section 9's candidate 3, "a global neural model as the default forecaster",
predicted the neural
models would be "competitive at the top level, no better or worse at the leaves, at many
times the compute". **That is an accurate description of PatchTST** and a poor one of
N-HiTS, which was not competitive anywhere. The plan was right about the architecture it
did not name and wrong about the one it did, which is worth recording as written rather
than quietly rounded to "the plan was right".

The sharper form of the candidate holds: LightGBM matches the neural models at the top two
levels, so whatever gain was available came from learning across series and not from depth,
and even that gain was zero against a well-fitted ETS.

**TimesFM complicates the candidate rather than confirming it,** and that is worth saying
plainly. The candidate is about *training* a neural model on your own series, and the
project's answer to that is a clear no. The model that did not lose was not trained here at
all. If there is a successor claim it is narrower and better supported: on this panel,
weights learned from other people's series matched a gradient-boosted model fitted to these
ones, at no training cost, on the origins that model could not have seen.
