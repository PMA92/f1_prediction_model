"""Honest evaluation harness for the finishing-position model.

Two rules this enforces that a plain train_test_split does not:
  1. Split by round, never shuffled. A shuffled split puts drivers from the SAME
     race on both sides, and trains on races that happened after the test races.
  2. Fit the DictVectorizer on train only. Calling fit_transform on the full X
     before splitting lets the encoder see the test set.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_absolute_error

CATEGORICAL = ['EventName', 'Abbreviation', 'TeamName', 'Year', 'KnockedOutIn']

# Known before lights out -- safe to use when predicting an unraced Grand Prix.
PRE_RACE = ['StartingGridPosition', 'QualPosition', 'QualGapToPolePct', 'QualNoTime',
            'AirTemp', 'TrackTemp', 'Humidity', 'Pressure', 'WindSpeed', 'Rainfall']

# Measured DURING the race being predicted. Including these leaks the answer.
IN_RACE = ['AvgLapTime', 'MedianLapTime', 'FastestLapTime']

TARGET = 'FinishingPosition'


def to_records(df, categorical, numeric):
    """One dict per driver-per-race. str -> one-hot, float -> raw value."""
    missing = [c for c in categorical + numeric if c not in df.columns]
    if missing:
        raise KeyError(f"not in dataframe: {missing}")

    out = df.copy()
    for c in categorical:
        out[c] = out[c].astype(str)
    for c in numeric:
        out[c] = out[c].astype(float)

    return [{k: v for k, v in row.items() if not (isinstance(v, float) and np.isnan(v))}
            for row in out[categorical + numeric].to_dict(orient='records')]


def evaluate(train_df, test_df, categorical, numeric, label):
    """Fit on train, score on test. Returns (MAE, mean within-race Spearman)."""
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(to_records(train_df, categorical, numeric))
    X_test = dv.transform(to_records(test_df, categorical, numeric))  # transform, not fit

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, train_df[TARGET])
    pred = model.predict(X_test)

    mae = mean_absolute_error(test_df[TARGET], pred)

    # Spearman per race, then averaged: "did I get the running order right?"
    scored = test_df.assign(pred=pred)
    rho = np.mean([spearmanr(g[TARGET], g['pred']).statistic
                   for _, g in scored.groupby(['Year', 'Round'])])

    print(f"  {label:<40} feats={X_train.shape[1]:>3}  MAE={mae:.2f}  rho={rho:+.3f}")
    return mae, rho


def main():
    df = pd.concat([pd.read_excel('f1_2024_results.xlsx'),
                    pd.read_excel('f1_2025_results.xlsx')], ignore_index=True)
    df = df.sort_values(['Year', 'Round']).reset_index(drop=True)

    print(f"{len(df)} driver-races | 2024 through round "
          f"{df[df.Year == 2024].Round.max()}, 2025 through round "
          f"{df[df.Year == 2025].Round.max()}")

    # Time-ordered: everything up to 2025 R12 trains, 2025 R13+ tests.
    train_df = df[(df.Year == 2024) | ((df.Year == 2025) & (df.Round <= 12))]
    test_df = df[(df.Year == 2025) & (df.Round > 12)]
    print(f"train {len(train_df)} rows ({train_df.Round.nunique()} rounds)  ->  "
          f"test {len(test_df)} rows ({test_df.Round.nunique()} rounds)\n")

    print("Honest (pre-race features only):")
    evaluate(train_df, test_df, [], ['StartingGridPosition'], "grid position alone")
    evaluate(train_df, test_df, [], PRE_RACE, "all pre-race numeric")
    evaluate(train_df, test_df, CATEGORICAL, PRE_RACE, "+ driver/team/track/knockout one-hots")

    print("\nLeaky (adds lap times measured during the race being predicted):")
    evaluate(train_df, test_df, CATEGORICAL, PRE_RACE + IN_RACE, "same + in-race lap times")

    print("\nMAE is 'how many places off on average'. rho is 'did I get the order")
    print("right within each race' (1.0 = perfect order, 0.0 = random).")


if __name__ == '__main__':
    main()
