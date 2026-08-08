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


def fit_predict(train_df, test_df, categorical, numeric):
    """Fit on train, predict test. Returns (scored test rows, n_features)."""
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(to_records(train_df, categorical, numeric))
    X_test = dv.transform(to_records(test_df, categorical, numeric))  # transform, not fit

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, train_df[TARGET])

    # The raw prediction is a float (11.94) and two drivers in the same race can
    # land on nearly the same value, so it is not a finishing order by itself.
    # Ranking it WITHIN each race turns it into a clean 1..N with no ties.
    scored = test_df.assign(pred=model.predict(X_test))
    scored['PredictedFinish'] = (scored.groupby(['Year', 'Round'])['pred']
                                       .rank(method='first').astype(int))
    scored['Error'] = scored['PredictedFinish'] - scored[TARGET]
    return scored, X_train.shape[1]


def score(scored):
    """(MAE on the raw prediction, mean within-race Spearman)."""
    mae = mean_absolute_error(scored[TARGET], scored['pred'])
    rho = np.mean([spearmanr(g[TARGET], g['pred']).statistic
                   for _, g in scored.groupby(['Year', 'Round'])])
    return mae, rho


def evaluate(train_df, test_df, categorical, numeric, label):
    """One train/test split, printed as a single comparison row."""
    scored, n_feats = fit_predict(train_df, test_df, categorical, numeric)
    mae, rho = score(scored)
    print(f"  {label:<40} feats={n_feats:>3}  MAE={mae:.2f}  rho={rho:+.3f}")
    return mae, rho, scored


def backtest_year(df, year, categorical, numeric, min_train_rounds=5):
    """Out-of-sample predictions for a whole season, walk-forward.

    For each round N, the model is refit on every race that happened strictly
    before N and then predicts N. So every prediction returned is for a race the
    model had not seen -- which is the only way to get honest 2024 numbers when
    2024 is otherwise entirely inside the training set.

    The first `min_train_rounds` rounds are skipped: with no prior season loaded
    there is nothing to train round 1 on.
    """
    rounds = sorted(df.loc[df.Year == year, 'Round'].unique())
    frames = []
    for rnd in rounds[min_train_rounds:]:
        train = df[(df.Year < year) | ((df.Year == year) & (df.Round < rnd))]
        test = df[(df.Year == year) & (df.Round == rnd)]
        scored, _ = fit_predict(train, test, categorical, numeric)
        frames.append(scored)
    return pd.concat(frames, ignore_index=True)


def season_table(scored, to_csv=None):
    """Track / driver / grid / actual / predicted, every race in order."""
    cols = ['EventName', 'Abbreviation', 'StartingGridPosition',
            TARGET, 'PredictedFinish']
    out = (scored.sort_values(['Year', 'Round', TARGET])[['Round'] + cols]
                 .rename(columns={'EventName': 'Track', 'Abbreviation': 'Driver',
                                  'StartingGridPosition': 'Grid',
                                  TARGET: 'Actual', 'PredictedFinish': 'Predicted'}))
    if to_csv:
        out.to_csv(to_csv, index=False)
        print(f"wrote {to_csv}  ({len(out)} rows)")
    return out


def show_predictions(scored, to_csv=None):
    """Every test-set prediction, one race at a time, sorted by actual finish.

    Grouping on ['Year', 'Round'] rather than EventName matters: track names
    repeat across seasons, so grouping by name would fuse 2024 and 2025 Monza
    into one 40-driver 'race' and wreck the per-race rho.
    """
    cols = ['Abbreviation', 'TeamName', 'StartingGridPosition',
            TARGET, 'PredictedFinish', 'pred', 'Error']
    names = {'Abbreviation': 'Driver', 'TeamName': 'Team',
             'StartingGridPosition': 'Grid', TARGET: 'Actual',
             'PredictedFinish': 'Predicted', 'pred': 'RawScore'}

    for (year, rnd), g in scored.groupby(['Year', 'Round']):
        race_rho = spearmanr(g[TARGET], g['pred']).statistic
        race_mae = g['Error'].abs().mean()
        print(f"\n=== {year} R{rnd}  {g['EventName'].iloc[0]}   "
              f"rho={race_rho:+.3f}  mean |error|={race_mae:.1f} places ===")
        # Sorted by actual finish so the winner is the top row and you read down
        # the real classification with the model's guess beside it.
        print(g.sort_values(TARGET)[cols].rename(columns=names)
               .to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    if to_csv:
        (scored.sort_values(['Year', 'Round', TARGET])
               [['Year', 'Round', 'EventName'] + cols]
               .to_csv(to_csv, index=False))
        print(f"\nwrote {to_csv}")


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
    # ngroups, not Round.nunique(): round numbers repeat across seasons, so
    # nunique() would call 2024 R1-24 plus 2025 R1-12 "24 rounds" instead of 36.
    print(f"train {len(train_df)} rows "
          f"({train_df.groupby(['Year', 'Round']).ngroups} races)  ->  "
          f"test {len(test_df)} rows "
          f"({test_df.groupby(['Year', 'Round']).ngroups} races)\n")

    print("Honest (pre-race features only):")
    evaluate(train_df, test_df, [], ['StartingGridPosition'], "grid position alone")
    evaluate(train_df, test_df, [], PRE_RACE, "all pre-race numeric")
    _, _, scored = evaluate(train_df, test_df, CATEGORICAL, PRE_RACE,
                            "+ driver/team/track/knockout one-hots")

    print("\nLeaky (adds lap times measured during the race being predicted):")
    evaluate(train_df, test_df, CATEGORICAL, PRE_RACE + IN_RACE, "same + in-race lap times")

    print("\nMAE is 'how many places off on average'. rho is 'did I get the order")
    print("right within each race' (1.0 = perfect order, 0.0 = random).")

    # Race-by-race detail for the honest full-feature model.
    show_predictions(scored, to_csv='predictions.csv')


if __name__ == '__main__':
    main()
