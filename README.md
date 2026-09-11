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
| 1 | Race selection and lap extraction | Pending |
| 2 | Cleaning: pit laps, out-laps, outliers | Pending |
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

Pending.

### Cleaning

Pending.

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
