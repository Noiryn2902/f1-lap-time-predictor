# F1 Lap Time Predictor

Predicting lap times within a single Formula 1 race, and testing whether tire
age explains pace beyond what fuel burn and starting position already explain.

Submitted for **GitHub Community SRM (GCSRM) Recruitment 2026**, Technical
Track: AI & Machine Learning (Year 1), Option A.

---

## Research question

> Within one dry Grand Prix, does a lap time model improve when it knows how
> old the tires are, after fuel burn and grid position are already accounted
> for, and is that improvement real or an artifact of how the data was split?

The second half of that question matters as much as the first. A random
train/test split on lap data leaks: consecutive laps are nearly identical, so a
model tested on lap 24 has almost certainly trained on laps 23 and 25. This
project reports both splits side by side to show the size of that illusion.

---

## Status

| Phase | Description | State |
| :---- | :---------- | :---- |
| 0 | Environment, repository scaffold, dataset | Done |
| 1 | Race selection and lap extraction | Done |
| 2 | Cleaning: pit laps, out-laps, outliers | Done |
| 3 | Feature engineering: tire age, stint, fuel proxy | Pending |
| 4 | Stint-based split plus random-split control | Pending |
| 5 | Model comparison: baseline vs tire-aware | Pending |
| 6 | Figures | Pending |
| 7 | Write-up | Pending |

---

## Setup

```bash
pip install -r requirements.txt
python scripts/download_data.py
```

The dataset is the Ergast Motor Racing Database CSV export, which is the
upstream source of the Kaggle dataset "Formula 1 World Championship (1950 to
2020)" named in the brief. Ergast retired its own download endpoint, so
`scripts/download_data.py` pulls the identical archive from a public mirror.
This keeps the repository reproducible without Kaggle credentials.

Verify the loader:

```bash
python src/data_loader.py
```

### A note on `races.csv`

The published archive ships a `races.csv` whose header row lists 8 columns
while every data row carries 18. Reading it with pandas' default header
silently shifts every field and turns `year` into a column of timestamp
strings. `src/data_loader.py` restores the true schema. Any analysis that
filters on `year` without this fix returns zero rows.

---

## Repository layout

```
data/raw/        Ergast CSV tables (not committed; see download script)
scripts/         Dataset download
src/             Reusable loading, cleaning and feature code
notebooks/       Analysis notebook
figures/         Generated plots
results/         Model comparison tables
```

---

## Methodology

_Filled in as each phase completes._

### Race selection

**Chosen: the 2019 United States Grand Prix, Circuit of the Americas, 56 laps.**

The brief asks for a dry race with no red flags. Rather than picking one from
reputation, every 2019 race was scored on four signals visible in the lap data
itself: lap-time variation as a wetness proxy, the share of laps beyond 1.5x
the field median, the slowest field-median lap relative to the race median, and
mid-race attrition. `src/race_selection.py` does the scoring and
`results/race_selection_2019.csv` holds the full table.

Austin wins on the measurements that matter:

| Signal | Austin | Season position |
| :--- | ---: | :--- |
| Slowest lap, as a multiple of race median | 1.073 | lowest of 21 |
| Lap-time variation (wetness proxy) | 0.0373 | 2nd lowest |
| Laps beyond 1.5x the field median | 0.0000 | joint lowest |
| Mid-race retirements | 1 | joint lowest |

Eight races survive the filter. The tiebreaker is experimental design. Most
Austin finishers ran three stints, which gives two training stints and one
held-out final stint per driver at roughly a 60/40 split of laps. The 2019
Austrian Grand Prix scores just as clean, but almost every driver ran a single
stop, leaving one training stint each and a test set larger than the training
set.

**Two measurement traps, both of which changed the answer.** A first pass
measured red flags as the largest drop in cars circulating between consecutive
laps. That flags nearly every clean race, because the leader finishes before
the cars they lapped, so the field appears to collapse at the flag; attrition
is now measured only up to three laps from the end. Separately, with wetness
used only to rank rather than to filter, the 2019 German Grand Prix passed:
its disruption and attrition are low, yet it was the wettest race of the
season, with 78 pit stops.

**Why not Bahrain.** It is the natural first choice and it fails the filter, on
a 1.345x lap spike and three mid-race retirements. That is the late safety car
and the leader's engine failure, and those laps would carry neutralised running
that has nothing to do with tire wear.

### Drivers

Eight, the highest finishers who were classified, ran within one lap of the
full distance, made at least two stints, and had at least 10 laps in the final
stint and 20 before it.

| Code | Driver | Grid | Finish | Stints | Train laps | Test laps |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| BOT | Bottas | 1 | 1 | 3 | 35 | 21 |
| HAM | Hamilton | 5 | 2 | 2 | 24 | 32 |
| VER | Verstappen | 3 | 3 | 3 | 34 | 22 |
| LEC | Leclerc | 4 | 4 | 3 | 42 | 14 |
| ALB | Albon | 6 | 5 | 4 | 40 | 16 |
| RIC | Ricciardo | 9 | 6 | 2 | 21 | 35 |
| NOR | Norris | 8 | 7 | 3 | 42 | 14 |
| HUL | Hülkenberg | 11 | 9 | 3 | 39 | 17 |

447 laps in total, uncleaned median 100.861s.

![Raw lap times](figures/01_raw_lap_times.png)

The raw trace shows the expected sawtooth: pace decaying through each stint,
a tall spike on the lap the car pitted, then the pattern restarting on fresh
tires. The spikes are staggered across drivers rather than aligned, which is
the visual confirmation that no safety car neutralised the race.

### Cleaning

37 of 447 laps removed, 8.3%, each attributable to a named rule.
`results/cleaning_report.csv` holds the table.

| Rule | Why | Laps |
| :--- | :--- | ---: |
| `pit_in` | The lap the car entered the pit lane, which contains the stop | 15 |
| `pit_out` | The lap after, begun at pit-lane speed on cold tires | 15 |
| `outlier` | Slower than 1.5x that driver's own median | 0 |
| `lap_one` | The opening lap, begun from a standstill | 8 |
| | **Total unique laps removed** | **37** |

The outlier test uses each driver's own median rather than the field median, so
a slower car is not penalised for being slower.

**The outlier rule removes nothing, and that is the correct result.** It exists
to catch safety cars and off-track moments. This race has neither, which is
exactly why Phase 1 selected it. The threshold sits near 151s and the slowest
green-flag lap is 47s clear of it. A count of zero is evidence the race
selection worked, not evidence the rule was misapplied.

**Lap 1 is an addition to the brief, not part of it.** It begins from a
standstill and includes the first-corner scramble, making it 4.6s slower than a
green-flag lap. That is a starting-procedure effect, not a tire effect. It
matters more than an ordinary outlier because lap 1 is always run on brand new
tires: keeping it would place a very slow lap at a tire age of zero for every
driver and teach the model that fresh tires are slow, which inverts the effect
being measured. Removing it also pulls the slowest remaining lap from 110.5s
down to 104.6s.

![Cleaning, before and after](figures/02_cleaning_before_after.png)

The lower panel is what the models see. The sawtooth is gone. What remains is a
gentle downward drift, which is fuel burn, overlaid with a repeating rise inside
each stint, which is tire wear. Separating those two is the whole problem.

### Features

Pending.

### Splitting strategy

Pending.

### Models and metrics

Pending.

---

## Results

Pending.

---

## Limitations

Pending.

---

## Data source

Ergast Motor Racing Database, covering 1950 to March 2022, mirrored at
[rubenv/ergast-mrd](https://github.com/rubenv/ergast-mrd). Used under the
Ergast terms for non-commercial purposes.
