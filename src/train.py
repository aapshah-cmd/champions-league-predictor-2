"""Train and evaluate match-outcome models with a strict time-based split.

    python -m src.train            # real data (downloads on first run)
    python -m src.train --demo     # synthetic data, works offline
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config, evaluate, models
from src.data_loader import load_all
from src.features import feature_columns, make_dataset


def warmup_end(df: pd.DataFrame) -> pd.Timestamp:
    first = int(df["season"].min())
    return pd.Timestamp(first + config.WARMUP_SEASONS, 7, 1)


def split(df: pd.DataFrame):
    df = df[df["Date"] >= warmup_end(df)]              # drop the warm-up season(s)
    return df[df["Date"] < config.TEST_START], df[df["Date"] >= config.TEST_START]


def run(matches: pd.DataFrame | None = None, out_dir: Path | None = None,
        model_names: list[str] | None = None, use_clubelo: bool = True) -> pd.DataFrame:
    out_dir = Path(out_dir or config.OUTPUT_DIR)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if matches is None:
        matches, _, _ = load_all(use_clubelo=use_clubelo)
    df, _ = make_dataset(matches)
    feats = feature_columns(df)
    # drop feature columns that are entirely missing (e.g. ClubElo unavailable)
    feats = [c for c in feats if df[c].notna().any()]
    train, test = split(df)
    print(f"Matches: train {len(train)}  test {len(test)}  features {len(feats)}")
    print(f"Train: {train.Date.min():%Y-%m-%d} -> {train.Date.max():%Y-%m-%d} | "
          f"Test: {test.Date.min():%Y-%m-%d} -> {test.Date.max():%Y-%m-%d}")
    if len(test) < 30:
        print("WARNING: very few test matches; metrics will be noisy.")

    y_test = test["y"].astype(int).to_numpy()
    preds: dict[str, np.ndarray] = {}

    # ---- baselines -----------------------------------------------------------
    prior = np.bincount(train["y"].astype(int), minlength=3) / len(train)
    preds["Baseline: class frequencies"] = np.tile(prior, (len(test), 1))
    elo_col = "ce_diff" if "ce_diff" in feats else "elo_diff"
    elo_only = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                         ("clf", LogisticRegression())])
    elo_only.fit(train[[elo_col]], train["y"].astype(int))
    label = "Baseline: ClubElo only" if elo_col == "ce_diff" else "Baseline: Elo only"
    preds[label] = elo_only.predict_proba(test[[elo_col]])

    # ---- machine learning models ---------------------------------------------
    fitted = {}
    for name in (model_names or models.ALL_MODELS):
        if name == "xgb" and not models.xgb_available():
            print("  (skipping XGBoost: not installed)")
            continue
        print(f"  training {models.LABELS[name]} ...")
        model = models.fit_model(name, train, feats, val_start=config.VAL_START)
        fitted[name] = model
        preds[models.LABELS[name]] = models.predict_proba(model, test[feats])

    results = [evaluate.summarize(n, y_test, P) for n, P in preds.items()]
    table = pd.DataFrame(results).sort_values("log_loss").reset_index(drop=True)
    table.to_csv(out_dir / "results.csv", index=False)
    print(f"\nTest results on {len(test)} matches (lower log loss / Brier / RPS is better):")
    print(table.round(4).to_string(index=False))

    ml_labels = [models.LABELS[n] for n in fitted]
    best_label = min(ml_labels, key=lambda l: next(r["log_loss"] for r in results
                                                   if r["model"] == l))
    best_name = next(n for n in fitted if models.LABELS[n] == best_label)
    P = preds[best_label]
    evaluate.plot_model_comparison(table.to_dict("records"), fig_dir / "model_comparison.png")
    evaluate.plot_calibration(y_test, P, fig_dir / "calibration.png",
                              title=f"Calibration: {best_label}")
    evaluate.plot_confusion(y_test, P, fig_dir / "confusion_matrix.png", title=best_label)
    imp = models.importances(fitted[best_name], best_name, feats)
    if imp is not None:
        evaluate.plot_importance(*imp, fig_dir / "feature_importance.png")
    out = test[["Date", "stage", "HomeTeam", "AwayTeam", "FTR"]].copy()
    out[["p_home", "p_draw", "p_away"]] = P
    out.to_csv(out_dir / "test_predictions.csv", index=False)
    print(f"\nBest ML model by log loss: {best_label}. Figures saved to {fig_dir}")
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="use synthetic data (offline)")
    ap.add_argument("--no-clubelo", action="store_true", help="skip ClubElo ratings")
    args = ap.parse_args()
    if args.demo:
        from src.synthetic import make_synthetic
        run(make_synthetic()[0], out_dir=config.OUTPUT_DIR / "demo")
    else:
        run(use_clubelo=not args.no_clubelo)
