"""Train and score the baseline and tire-aware models on both splits.

The comparison is a 2x2x2 grid: two feature sets, two algorithms, two splits.

Feature sets
------------
``baseline``
    ``grid`` and ``lap``. This already controls for fuel burn and track
    evolution, because within one race both are monotonic in lap number (see
    ``features.py``). It knows nothing about tires.
``enhanced``
    Adds ``tire_age``, ``tire_age_sq`` and ``stint_number``.

Because the baseline already carries the fuel control, any improvement the
enhanced set shows is attributable to tire information rather than to a
confound the baseline was missing.

Algorithms
----------
``LinearRegression``
    Chosen for interpretability. Its ``tire_age`` coefficient reads directly as
    seconds lost per lap of tire age, which is the quantity the project is
    about. It also extrapolates, which matters here: held-out stints reach tire
    ages beyond the training range.
``RandomForestRegressor``
    Chosen as a contrasting, non-linear learner. It cannot extrapolate: beyond
    the training range every tree returns its nearest leaf, so predictions
    flatten out. That limitation is reported rather than hidden.

Splits
------
Both the stint split and the random split, so the gap between them measures the
leakage rather than merely describing it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from splits import random_split, stint_split

TARGET = "seconds"

FEATURE_SETS: dict[str, list[str]] = {
    "baseline": ["grid", "lap"],
    "enhanced": ["grid", "lap", "tire_age", "tire_age_sq", "stint_number"],
}

RANDOM_SEED = 42


def _algorithms() -> dict[str, object]:
    return {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
    }


def evaluate(featured: pd.DataFrame) -> pd.DataFrame:
    """Return one row per feature set, algorithm and split."""
    splits = {
        "stint": stint_split(featured),
        "random": random_split(featured),
    }

    rows = []
    for split_name, (train, test) in splits.items():
        for fs_name, cols in FEATURE_SETS.items():
            for algo_name, model in _algorithms().items():
                model.fit(train[cols], train[TARGET])
                pred = model.predict(test[cols])
                rows.append({
                    "split": split_name,
                    "features": fs_name,
                    "algorithm": algo_name,
                    "rmse": float(np.sqrt(mean_squared_error(test[TARGET], pred))),
                    "mae": float(mean_absolute_error(test[TARGET], pred)),
                    "n_train": len(train),
                    "n_test": len(test),
                })

    table = pd.DataFrame(rows)
    return table.sort_values(["split", "features", "algorithm"]).reset_index(drop=True)


def improvement_table(results: pd.DataFrame) -> pd.DataFrame:
    """Baseline versus enhanced RMSE, per split and algorithm."""
    wide = results.pivot_table(
        index=["split", "algorithm"], columns="features", values="rmse"
    ).reset_index()
    wide["rmse_gain"] = wide["baseline"] - wide["enhanced"]
    wide["rmse_gain_pct"] = 100 * wide["rmse_gain"] / wide["baseline"]
    return wide.round(4)


def tire_age_coefficient(featured: pd.DataFrame) -> pd.DataFrame:
    """Read the linear model's coefficients on the honest split.

    ``tire_age`` and ``tire_age_sq`` act together, so the marginal cost of one
    more lap on a set is reported at several ages rather than as one number.
    """
    train, _ = stint_split(featured)
    cols = FEATURE_SETS["enhanced"]
    model = LinearRegression().fit(train[cols], train[TARGET])
    coefs = dict(zip(cols, model.coef_))

    rows = [
        {"feature": c, "coefficient_s": round(float(coefs[c]), 5)} for c in cols
    ]
    # d(time)/d(age) = b_age + 2 * b_age_sq * age
    for age in (1, 5, 10, 15, 20):
        marginal = coefs["tire_age"] + 2 * coefs["tire_age_sq"] * age
        rows.append({
            "feature": f"marginal cost at tire age {age}",
            "coefficient_s": round(float(marginal), 5),
        })
    return pd.DataFrame(rows)


def extrapolation_check(featured: pd.DataFrame) -> pd.DataFrame:
    """How far the held-out stints run beyond the training tire-age range."""
    train, test = stint_split(featured)
    return pd.DataFrame([{
        "train_tire_age_max": int(train["tire_age"].max()),
        "test_tire_age_max": int(test["tire_age"].max()),
        "test_laps_beyond_train_range": int(
            (test["tire_age"] > train["tire_age"].max()).sum()
        ),
        "test_laps": len(test),
    }])


if __name__ == "__main__":
    from cleaning import clean
    from data_loader import load_pit_stops
    from dataset import build_lap_table
    from features import add_features

    race, selected, laps = build_lap_table()
    stops = load_pit_stops()
    stops = stops[stops["raceId"] == race["raceId"]]
    featured = add_features(clean(laps, stops)[0], stops)

    results = evaluate(featured)
    print(results.to_string(index=False))

    print("\nBaseline vs enhanced")
    print(improvement_table(results).to_string(index=False))

    print("\nLinear coefficients, stint split")
    print(tire_age_coefficient(featured).to_string(index=False))

    print("\nExtrapolation demanded by the stint split")
    print(extrapolation_check(featured).to_string(index=False))
