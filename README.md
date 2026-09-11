# F1 Lap Time Predictor

Predicting lap times within a single Formula 1 race, and testing whether tire
age explains pace beyond what fuel burn and grid position already explain.

GCSRM Recruitment 2026, Technical Track: AI & Machine Learning (Year 1),
Option A.

> **Question.** Within one dry Grand Prix, does a lap time model improve when it
> knows how old the tires are, and is that improvement real or an artifact of
> how the data was split?

## The answer

**Yes, and it survives an honest split.** On held-out final stints, adding tire
age cuts RMSE from 1.071s to 0.907s, a **15.4%** improvement.

The linear model separates the two effects that mask each other:

| Effect | Coefficient |
| :--- | ---: |
| Tire wear | **+0.127 s/lap** |
| Fuel burn and track evolution | **-0.085 s/lap** |
| Grid position | +0.211 s/place |

They nearly cancel, which is why they have to be modelled together. Measured
raw degradation is 1.10s across 21 laps of tire age, matching the coefficients.

**Three findings beyond the brief:**

1. **A random split reverses the verdict, it does not merely flatter it.** Under
   a random split RandomForest looks 31% better than linear regression; under
   the stint split linear regression is 20% better. A careless split leads to
   shipping the wrong model, not just to quoting an optimistic number.
2. **The leak is measured, not asserted.** 95.1% of randomly-split test laps
   have an immediately adjacent lap in the training set. Under the stint split,
   0.0%.
3. **A planned feature was deleted after being measured.** The intended fuel
   proxy correlates with lap number at exactly -1.000, because within one race
   they are the same quantity. Lap number already *is* the fuel control, which
   is what makes the baseline comparison fair.

## Results

Two feature sets, two algorithms, two splits. Full table in
[`results/model_comparison.csv`](results/model_comparison.csv).

| Split | Features | Algorithm | RMSE | MAE |
| :--- | :--- | :--- | ---: | ---: |
| **stint** | baseline | LinearRegression | 1.071 | 0.866 |
| **stint** | **enhanced** | **LinearRegression** | **0.907** | **0.703** |
| stint | baseline | RandomForest | 1.579 | 1.300 |
| stint | enhanced | RandomForest | 1.137 | 0.938 |
| random | baseline | LinearRegression | 0.915 | 0.742 |
| random | enhanced | LinearRegression | 0.708 | 0.543 |
| random | baseline | RandomForest | 0.637 | 0.417 |
| random | enhanced | RandomForest | 0.488 | 0.379 |

Tire age improves every pairing, by 15% to 28%. **Only the stint rows can be
believed**; the random rows are shown to expose the leak.

![Model comparison](figures/05_model_comparison.png)

### Predicted against actual

![Predicted vs actual](figures/06_predicted_vs_actual_stint.png)

Hamilton's race. Shaded region is training data; everything right of the red
line was held out.

- **Baseline, dotted grey, slopes the wrong way.** With no tire information it
  follows only fuel burn, predicting the car getting faster while it slows.
- **Enhanced linear, blue, gets the direction right** across 31 unseen laps.
- **RandomForest, gold, flattens from lap 44.** Beyond the training tire-age
  range a tree returns its nearest leaf and stops following the trend.

Across all eight drivers the enhanced linear model wins for six, with mean RMSE
0.926s against 1.104s for the baseline. See
[`figures/06_all_drivers_held_out.png`](figures/06_all_drivers_held_out.png).

## Method

**Race.** 2019 United States Grand Prix, 56 laps. Every 2019 race was scored on
wetness, disruption, lap-time spikes and mid-race attrition rather than picked
by reputation. Austin has the lowest lap-time spike of the season at 1.073x the
race median, so no safety car flattened the pace. Bahrain, the obvious choice,
fails on a 1.345x spike and three retirements.

**Drivers.** Eight, the highest finishers with at least 10 laps in the final
stint and 20 before it. 447 laps.

**Cleaning.** 37 of 447 laps removed, 8.3%, each attributable to a named rule.

| Rule | Laps |
| :--- | ---: |
| Lap the car entered the pits | 15 |
| Lap after a stop, on cold tires | 15 |
| Slower than 1.5x that driver's median | 0 |
| Lap 1, standing start | 8 |
| **Total** | **37** |

The 1.5x rule removing zero is the correct result, not a misapplication: it
catches safety cars, and this race has none. Lap 1 is an addition to the brief,
removed because it is 4.6s slow for starting-procedure reasons while always
sitting at tire age 0, which would teach the model that fresh tires are slow.

**Features.** `tire_age` (laps on the current set, resetting to 0 at each stop),
`tire_age_sq`, and `stint_number` as a proxy for compound.

**Split.** Each driver's final stint held out, earlier stints train. A random
80/20 split is also built, solely to measure the leak.

**Models.** LinearRegression for interpretability and its ability to
extrapolate; RandomForest as a contrasting non-linear learner. RMSE and MAE on
both splits.

Full reasoning, including two scoring bugs that changed the race choice, is in
[METHODOLOGY.md](METHODOLOGY.md). Each phase also has its own notebook.

## Limitations

- **One race.** 0.127s per lap is a property of that surface, not a constant.
- **Compound is never observed.** `stint_number` stands in for it, conflating
  compound with race progress and fuel load.
- **Fuel cannot be isolated within one race.** Fuel burn and track evolution are
  both monotonic in lap number, so -0.085s is their combined effect.
- **The stint split forces extrapolation.** 15 of 162 test laps exceed the
  training tire-age range, penalising trees and flattering linear extension.
- **Eight drivers, 410 laps.** The enhanced model wins for six of eight. The
  aggregate gain is real and not uniform.
- **No traffic feature.** Likely the reason Leclerc has the worst RMSE at 1.404s.

**Next:** repeat across a full season to separate fuel from track evolution;
join a compound source; add a per-driver intercept and a traffic feature.

## Setup

```bash
pip install -r requirements.txt
```

### Data

The dataset is the Kaggle "Formula 1 World Championship (1950 to 2020)" by Rohan
Rao, specifically `lap_times.csv`, `pit_stops.csv`, `results.csv` and
`races.csv`. **No API is used anywhere in this project.** Every stage reads
static CSV files from disk.

Download the dataset from Kaggle and point the script at it:

```bash
python scripts/download_data.py --kaggle ~/Downloads/archive.zip
```

A folder works too. If you would rather not sign in to Kaggle, running it with
no arguments fetches the same static CSV export from a public mirror of the
Ergast database, which is the upstream source the Kaggle dataset republishes:

```bash
python scripts/download_data.py
```

Both routes produce identical tables and identical results.

### Running it

```bash
python src/race_selection.py
python src/cleaning.py
python src/models.py
```

Or run the notebooks in order, `00` through `06`.

### Pinned versions and seeds

Dependencies are pinned in [`requirements.txt`](requirements.txt).

Every stochastic step is seeded, so the numbers in this README reproduce exactly:

| Setting | Value | Where |
| :--- | :--- | :--- |
| `RANDOM_SEED` | `42` | `src/splits.py`, the random-split control |
| `RANDOM_SEED` | `42` | `src/models.py`, `RandomForestRegressor(random_state=...)` |
| `RANDOM_TEST_FRACTION` | `0.2` | `src/splits.py` |
| `n_estimators` | `300` | `src/models.py` |
| `min_samples_leaf` | `2` | `src/models.py` |

The stint split is deterministic by construction and uses no seed.

`scikit-learn` is pinned to 1.6.1 rather than the latest release: 1.9.1 ships an
unsigned compiled extension that Windows Smart App Control blocks, which breaks
every `sklearn` submodule import.

**A note on `races.csv`:** the published archive ships a header listing 8
columns while every data row carries 18. Pandas' default read silently shifts
every field and turns `year` into timestamp strings, so filtering on `year`
returns zero rows. `src/data_loader.py` restores the true schema.

## Layout

```
scripts/      download_data.py
src/          data_loader, race_selection, dataset, cleaning, features, splits, models
notebooks/    00 environment .. 06 figures, one per phase, all executed
figures/      seven plots
results/      twelve CSV tables
```

Verified by deleting every figure and table and re-running all seven notebooks
from scratch.

---

Data: Kaggle "Formula 1 World Championship (1950 to 2020)" by Rohan Rao, which
republishes the Ergast Motor Racing Database. Mirror used by the fallback path:
[rubenv/ergast-mrd](https://github.com/rubenv/ergast-mrd). Non-commercial use.
