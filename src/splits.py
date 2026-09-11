"""Split laps into training and test sets, two ways.

Stint split (the honest one)
----------------------------
Each driver's final stint is held out; everything before it trains. A stint is
a contiguous block of laps on one set of tires, so holding one out means the
model is asked to predict a run of the race it has never seen, on a tire age
range it may never have seen either.

Random split (the control, kept only to expose it)
--------------------------------------------------
Laps shuffled and cut 80/20. This is what a default ``train_test_split`` call
produces, and it leaks badly. Consecutive laps by the same driver on the same
set of tires differ by a few tenths of a second, so a lap drawn into the test
set almost always has its immediate neighbours sitting in the training set.
The model can interpolate between them rather than predict anything.

Both splits are run through identical models in Phase 5. The gap between their
scores is the size of the illusion, and reporting it is the point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RANDOM_SEED = 42
RANDOM_TEST_FRACTION = 0.2


def stint_split(featured: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out each driver's final stint; train on everything before it."""
    final_stint = featured.groupby("driverId")["stint_number"].transform("max")
    is_test = featured["stint_number"] == final_stint
    return featured[~is_test].copy(), featured[is_test].copy()


def random_split(
    featured: pd.DataFrame,
    test_fraction: float = RANDOM_TEST_FRACTION,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shuffle laps and cut them, the way a default split would."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(featured))
    n_test = int(round(len(featured) * test_fraction))
    test_idx = order[:n_test]
    mask = np.zeros(len(featured), dtype=bool)
    mask[test_idx] = True
    return featured[~mask].copy(), featured[mask].copy()


def assert_disjoint(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Fail loudly if any lap appears in both sets."""
    key = ["driverId", "lap"]
    overlap = train[key].merge(test[key], on=key, how="inner")
    if len(overlap):
        raise AssertionError(f"{len(overlap)} laps appear in both train and test")


def neighbour_leakage(train: pd.DataFrame, test: pd.DataFrame) -> float:
    """Share of test laps whose immediately adjacent lap is in the training set.

    This is the mechanism, measured. A high value means the model can reach a
    test lap by interpolating between the laps either side of it, which is not
    prediction.
    """
    train_keys = set(zip(train["driverId"], train["lap"]))
    adjacent = [
        ((d, lap - 1) in train_keys) or ((d, lap + 1) in train_keys)
        for d, lap in zip(test["driverId"], test["lap"])
    ]
    return float(np.mean(adjacent))


def split_summary(featured: pd.DataFrame) -> pd.DataFrame:
    """Compare the two splits on size, tire-age coverage and neighbour leakage."""
    rows = []
    for name, (train, test) in {
        "stint": stint_split(featured),
        "random": random_split(featured),
    }.items():
        assert_disjoint(train, test)
        rows.append({
            "split": name,
            "train_laps": len(train),
            "test_laps": len(test),
            "test_share": round(len(test) / len(featured), 3),
            "test_tire_age_min": int(test["tire_age"].min()),
            "test_tire_age_max": int(test["tire_age"].max()),
            "neighbour_in_train": round(neighbour_leakage(train, test), 3),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from cleaning import clean
    from data_loader import load_pit_stops
    from dataset import build_lap_table
    from features import add_features

    race, selected, laps = build_lap_table()
    stops = load_pit_stops()
    stops = stops[stops["raceId"] == race["raceId"]]
    featured = add_features(clean(laps, stops)[0], stops)

    print(split_summary(featured).to_string(index=False))

    train, test = stint_split(featured)
    print("\nper driver, stint split")
    per = pd.DataFrame({
        "train": train.groupby("code").size(),
        "test": test.groupby("code").size(),
        "held_out_stint": test.groupby("code")["stint_number"].max(),
    }).fillna(0).astype(int)
    print(per.to_string())
