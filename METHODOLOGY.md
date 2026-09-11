# Methodology

Full reasoning behind each decision. The [README](README.md) carries the
summary; this document carries the argument, including the places where the
first attempt was wrong.

---

## 1. Race selection

**Chosen: the 2019 United States Grand Prix, Circuit of the Americas, 56 laps.**

The brief asks for a dry race with no red flags. Rather than picking one from
reputation, every 2019 race was scored on four signals visible in the lap data
itself: lap-time variation as a wetness proxy, the share of laps beyond 1.5x
the field median, the slowest field-median lap relative to the race median, and
mid-race attrition. `src/race_selection.py` does the scoring;
`results/race_selection_2019.csv` holds the full table.

| Signal | Austin | Season position |
| :--- | ---: | :--- |
| Slowest lap, as a multiple of race median | 1.073 | lowest of 21 |
| Lap-time variation (wetness proxy) | 0.0373 | 2nd lowest |
| Laps beyond 1.5x the field median | 0.0000 | joint lowest |
| Mid-race retirements | 1 | joint lowest |

Eight races survive the filter. The tiebreaker is experimental design. Most
Austin finishers ran three stints, giving two training stints and one held-out
final stint per driver at roughly a 60/40 split of laps. The 2019 Austrian
Grand Prix scores just as clean, but almost every driver ran a single stop,
leaving one training stint each and a test set larger than the training set.

### Two measurement traps, both of which changed the answer

**Finishing is not retirement.** A first pass measured red flags as the largest
drop in the number of cars circulating between consecutive laps. That flags
nearly every clean race, because the leader completes more laps than the cars
they lapped, so the field appears to collapse at the flag. Attrition is now
measured only up to three laps from the end.

**Wetness has to be a gate, not a tiebreaker.** With wetness used only for
ranking, the 2019 German Grand Prix passed the filter: its disruption and
attrition are low. It was the wettest race of the season, with 78 pit stops.
`wet_score` is now a hard threshold.

### Why not Bahrain

Bahrain is the natural first choice and fails the filter, on a 1.345x lap spike
and three mid-race retirements. That is the late safety car and the leader's
engine failure. Those laps would carry neutralised running that has nothing to
do with tire wear.

---

## 2. Drivers

Eight, the highest finishers who were classified, ran within one lap of the full
distance, made at least two stints, and had at least 10 laps in the final stint
and 20 before it.

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

447 laps, uncleaned median 100.861s.

![Raw lap times](figures/01_raw_lap_times.png)

The raw trace shows the expected sawtooth: pace decaying through each stint, a
tall spike on the lap the car pitted, then the pattern restarting on fresh
tires. The spikes are staggered across drivers rather than aligned, which is
visual confirmation that no safety car neutralised the race.

---

## 3. Cleaning

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
exactly why it was selected. The threshold sits near 151s and the slowest
green-flag lap is 47s clear of it. A count of zero is evidence the race
selection worked, not evidence the rule was misapplied.

**Lap 1 is an addition to the brief, not part of it.** It begins from a
standstill and includes the first-corner scramble, making it 4.6s slower than a
green-flag lap. That is a starting-procedure effect, not a tire effect. It
matters more than an ordinary outlier because lap 1 is always run on brand new
tires: keeping it would place a very slow lap at a tire age of zero for every
driver and teach the model that fresh tires are slow, inverting the effect being
measured. Removing it also pulls the slowest remaining lap from 110.5s to
104.6s.

![Cleaning, before and after](figures/02_cleaning_before_after.png)

The lower panel is what the models see. The sawtooth is gone. What remains is a
gentle downward drift, which is fuel burn, overlaid with a repeating rise inside
each stint, which is tire wear. Separating those two is the whole problem.

---

## 4. Features

| Feature | Meaning |
| :--- | :--- |
| `stint_number` | Which set of tires, counting from 1. Teams run different compounds in different stints, so this carries compound information the dataset does not expose directly. |
| `tire_age` | Laps completed on the current set, resetting to 0 at each stop. The feature the project exists to test. |
| `tire_age_sq` | Tire age squared. A set loses little early and falls away faster once the surface is gone; a linear term alone cannot bend to that. |

### The fuel proxy was dropped, and that is the finding

The plan was to add `fuel_proxy = total_laps - lap`, so the model could separate
fuel burn from tire wear. Cars start heavy and get lighter, which makes them
faster; tires get older, which makes them slower. Two effects pulling opposite
ways.

Within a single race that feature is worthless. `total_laps` is constant, so
`total_laps - lap` is an exact linear transform of `lap`. Their measured
correlation is **-1.000**. The two columns carry identical information, and in a
linear model the pair is perfectly collinear, so the coefficients stop being
identifiable. `results/fuel_collinearity.csv` records the check.

The useful conclusion runs the other way:

> Within one race, **lap number already is the fuel proxy.** It also absorbs
> track evolution as rubber goes down. Both effects are monotonic in lap number
> and cannot be separated from it without data from more than one race.

This is what makes the model comparison fair. The baseline already has `lap`, so
it already controls for fuel burn and track evolution. Any improvement from
`tire_age` is attributable to the tires rather than to a confound the baseline
was missing.

### Correlations with lap time

| Feature | Correlation |
| :--- | ---: |
| `stint_number` | -0.724 |
| `lap` | -0.623 |
| `grid` | +0.467 |
| `tire_age` | +0.172 |
| `tire_age_sq` | +0.144 |

The signs are the story. Lap number is negative, because fuel burn and track
evolution make the cars faster as the race runs. Tire age is positive, because
degradation makes them slower.

![Degradation by tire age](figures/03_degradation_by_tire_age.png)

Averaged across every stint, a set of tires costs **1.10s** between its first lap
and its twenty-first. The right-hand panel separates the stints: each sits lower
than the one before, which is fuel burn, and each still rises within itself,
which is tire wear.

**Tire age resets correctly.** It is 0 on the out-lap and 1 on the lap after.
Cleaning removes every out-lap, and removes lap 1 which is the opening stint's
equivalent, so all 23 surviving stints begin at 1. Albon has no first stint at
all: he pitted on lap 2, so his opening stint is lap 1 alone and cleaning removes
it. That is the data behaving correctly.

---

## 5. Splitting

**Stint split.** Each driver's final stint is held out; everything before it
trains. A stint is a contiguous block of laps on one set of tires, so holding
one out asks the model to predict a run of the race it has never seen.

**Random split.** Laps shuffled and cut 80/20, which is what a default
`train_test_split` call produces. Kept only so its failure can be measured.

| Split | Train | Test | Test share | Test laps with a neighbour in training |
| :--- | ---: | ---: | ---: | ---: |
| Stint | 248 | 162 | 39.5% | **0.0%** |
| Random | 328 | 82 | 20.0% | **95.1%** |

**The last column names the mechanism.** Consecutive laps by the same driver on
the same set of tires differ by a few tenths of a second. If laps 23 and 25 sit
in training, predicting lap 24 is not prediction, it is interpolation between two
nearly identical points, and any model will look excellent at it. Under the
random split that situation holds for 95.1% of test laps.

The stint split scores exactly zero by construction. A held-out stint is bounded
by a pit stop on one side and the end of the race on the other, and cleaning
already removed the laps at those boundaries, so nothing adjacent survives in
training.

![Split structure](figures/04_split_structure.png)

**Two costs the stint split pays, honestly.** It puts 39.5% of laps in the test
set rather than the usual 20%, because a final stint is simply a large share of a
race. And it forces extrapolation: held-out stints reach a tire age of 34, beyond
much of what the training stints cover. Both make the stint split look worse on
paper, and both are the reason its score can be believed.

Neither split lets a lap appear in both sets; `assert_disjoint` checks it.

---

## 6. Models and findings

| Feature set | Columns |
| :--- | :--- |
| `baseline` | `grid`, `lap` |
| `enhanced` | `grid`, `lap`, `tire_age`, `tire_age_sq`, `stint_number` |

`LinearRegression` was chosen for interpretability: its coefficients read
directly in seconds, and it extrapolates, which matters because held-out stints
reach tire ages beyond the training range. `RandomForestRegressor` was chosen as
a contrasting non-linear learner.

### Finding 1 — tire age helps in every pairing

| Split | Algorithm | Baseline RMSE | Enhanced RMSE | Gain |
| :--- | :--- | ---: | ---: | ---: |
| stint | LinearRegression | 1.071 | 0.907 | **15.4%** |
| stint | RandomForest | 1.579 | 1.137 | 28.0% |
| random | LinearRegression | 0.915 | 0.708 | 22.7% |
| random | RandomForest | 0.637 | 0.488 | 23.4% |

**The headline number is 15.4%**, from the stint split with linear regression.
That is the one measured without leakage.

### Finding 2 — the leak reverses the verdict

| Split | LinearRegression | RandomForest | Apparent winner |
| :--- | ---: | ---: | :--- |
| Random (leaky) | 0.708 | **0.488** | RandomForest, by 31% |
| Stint (honest) | **0.907** | 1.137 | LinearRegression, by 20% |

A random split does not just report an over-optimistic score. It would lead to
shipping the **wrong model**.

The mechanism is direct. A tree predicts by averaging the training points that
fall in the same leaf. When 95.1% of test laps have an immediate neighbour in
training, the nearest leaf already contains nearly the answer, so memorisation
scores extremely well. RandomForest's score is inflated **57%** by the leak; the
linear model's by **22%**. The more flexible learner is the one the leak rewards
most, which is what makes a flexible model plus a careless split so dangerous.

### Finding 3 — RandomForest loses honestly because it cannot extrapolate

| Training max tire age | Test max tire age | Test laps beyond training range |
| ---: | ---: | ---: |
| 25 | 34 | 15 of 162 |

Final stints run longer than earlier ones, so the stint split necessarily asks
for extrapolation. A tree cannot extend a trend: past the edge of its training
range every branch returns its nearest leaf, so the prediction flattens exactly
where degradation is steepest. A linear model keeps extending its slope. This is
a limitation of the split as much as of the tree, and it is reported rather than
engineered away.

### What the linear model learned

| Effect | Coefficient | Reading |
| :--- | ---: | :--- |
| `tire_age` | **+0.127 s/lap** | each extra lap on a set costs about an eighth of a second |
| `lap` | **-0.085 s/lap** | fuel burn and track evolution give back about a twelfth of a second per lap |
| `grid` | +0.211 s/place | a car starting one place further back is about a fifth of a second slower |

Tire wear and fuel burn nearly cancel: net visible degradation is about 0.042s
per lap, roughly 0.8s over a 20-lap stint. Section 4 measured raw observed
degradation at 1.10s across 21 laps of tire age, so the two agree.

That near-cancellation is exactly why the baseline needs `lap`. Without it, tire
wear and fuel burn would be confounded and the measured degradation would be
wrong by a factor of three.

**The quadratic term did almost nothing.** `tire_age_sq` comes out at -0.0002, so
the marginal cost of one more lap falls only from 0.126s at age 1 to 0.119s at
age 20. Degradation at this race is very nearly linear across the range observed.
The term was worth including to test for a curve, and the honest result is that
there is barely one.

---

## 7. Per-driver results

![All eight drivers](figures/06_all_drivers_held_out.png)

`results/per_driver_rmse.csv` holds the numbers.

| Code | Held-out stint | Test laps | Linear RMSE | Forest RMSE | Baseline RMSE |
| :--- | ---: | ---: | ---: | ---: | ---: |
| RIC | 2 | 34 | **0.590** | 0.510 | 0.673 |
| NOR | 3 | 13 | **0.639** | 1.369 | 1.191 |
| VER | 3 | 21 | **0.744** | 1.377 | 1.006 |
| HAM | 2 | 31 | **0.790** | 1.046 | 1.146 |
| ALB | 4 | 15 | **1.024** | 1.129 | 1.064 |
| HUL | 3 | 15 | 1.043 | **0.981** | 1.334 |
| BOT | 3 | 20 | 1.171 | 1.360 | **1.071** |
| LEC | 3 | 13 | 1.404 | 1.567 | **1.344** |
| | | **mean** | **0.926** | 1.167 | 1.104 |

The enhanced linear model wins for six of eight drivers against both the
baseline and the forest. It loses to the baseline for Bottas and Leclerc, whose
held-out stints are among the noisiest. Reporting six of eight rather than an
aggregate alone is the honest form: the aggregate improvement is real, and it is
not uniform.

---

## 8. Reproducibility

Verified by deleting every figure and CSV and re-running all seven notebooks
from scratch. All pass, all 7 figures and 12 tables regenerate, and every
headline number was checked against the regenerated outputs.

Pinned versions and seed configs are tabled in the [README](README.md#pinned-versions-and-seeds).
