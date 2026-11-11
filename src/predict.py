"""Predict the next matches, simulate the whole tournament, and name the likely winner.

    python -m src.predict            # real data
    python -m src.predict --demo     # synthetic data, works offline

Outputs (in outputs/):
    upcoming_predictions.csv      win/draw/loss % + expected goals for the next matchday
    tournament_probabilities.csv  every team's chance of reaching each round / winning
    projected_bracket.csv         one readable bracket (favourite advances in every tie)
    figures/title_odds.png, figures/advancement.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config, models, tournament
from src.data_loader import load_all
from src.features import feature_columns, make_dataset
from src.train import warmup_end


def _pick_model(preferred: str) -> str:
    if preferred == "xgb" and not models.xgb_available():
        print("xgboost not installed -> using sklearn gradient boosting for match predictions")
        return "hgb"
    return preferred


def main(matches=None, fixtures=None, ext_now=None, out_dir: Path | None = None,
         model_name: str = "xgb", n_sims: int = 20000, use_clubelo: bool = True) -> None:
    out_dir = Path(out_dir or config.OUTPUT_DIR)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    if matches is None:
        matches, fixtures, ext_now = load_all(use_clubelo=use_clubelo)
    ext_now = ext_now or {}
    model_name = _pick_model(model_name)

    df, builder = make_dataset(matches)             # builder now holds every team's CURRENT state
    feats = [c for c in feature_columns(df) if df[c].notna().any()]
    train = df[df["Date"] >= warmup_end(df)]
    val_start = f"{int(train['season'].max()) - 1}-07-01"      # last full season: early stopping
    print(f"Training on {len(train)} matches ({len(feats)} features) ...")
    clf = models.fit_model(model_name, train, feats, val_start=val_start)
    poisson = models.fit_model("poisson", train, feats)

    ext = lambda t: ext_now.get(t, np.nan)

    def feat_row(r):
        return builder.features_for(r.HomeTeam, r.AwayTeam, r.Date, r.is_knockout, r.leg,
                                    r.neutral, ext(r.HomeTeam), ext(r.AwayTeam))

    # ---- next matchday ----------------------------------------------------------
    if fixtures is not None and len(fixtures):
        nxt = fixtures.sort_values("Date")
        nxt = nxt[nxt["Date"] <= nxt["Date"].min() + pd.Timedelta(days=3)]
        X = pd.DataFrame([feat_row(r) for r in nxt.itertuples()])[feats]
        P = models.predict_proba(clf, X)
        lh, la = poisson.lambdas(X)
        up = nxt[["Date", "stage", "HomeTeam", "AwayTeam"]].copy()
        up["home_win_%"], up["draw_%"], up["away_win_%"] = (P * 100).round(1).T
        up["exp_goals"] = [f"{a:.1f}-{b:.1f}" for a, b in zip(lh, la)]
        up["pick"] = np.array(config.CLASSES)[P.argmax(axis=1)]
        up.to_csv(out_dir / "upcoming_predictions.csv", index=False)
        print(f"\nNext matches ({models.LABELS[model_name]}):")
        print(up.to_string(index=False))
    else:
        print("No scheduled fixtures found (skipping next-match predictions).")

    # ---- tournament simulation ---------------------------------------------------
    cur = config.CURRENT_SEASON if (matches["season"] == config.CURRENT_SEASON).any() \
        else int(matches["season"].max())
    season_played = matches[matches["season"] == cur]
    fx = fixtures if fixtures is not None else matches.iloc[0:0]
    fx_league = fx[(fx["season"] == cur) & (fx["stage"] == "league")] if len(fx) else fx
    teams = sorted(set(season_played[season_played.stage == "league"].HomeTeam)
                   | set(season_played[season_played.stage == "league"].AwayTeam)
                   | set(fx_league.HomeTeam) | set(fx_league.AwayTeam))
    if len(teams) != config.N_TEAMS:
        print(f"WARNING: found {len(teams)} teams in the {cur} league phase "
              f"(expected {config.N_TEAMS}). Check the data / team names.")
    if len(teams) < 24:
        print("Not enough teams to simulate the tournament.")
        return
    idx = {t: i for i, t in enumerate(teams)}
    table = tournament.standings(season_played, teams)
    rem_home = np.array([idx[t] for t in fx_league.HomeTeam], dtype=int)
    rem_away = np.array([idx[t] for t in fx_league.AwayTeam], dtype=int)

    print(f"\nSimulating {n_sims} tournaments ({len(rem_home)} league-phase matches left) ...")
    LH, LA = tournament.lambda_matrices(builder, poisson, feats, teams, ext_now, knockout=0)
    KH, KA = tournament.lambda_matrices(builder, poisson, feats, teams, ext_now, knockout=1)
    sim = tournament.simulate(teams, table, rem_home, rem_away, LH, LA, KH, KA, n_sims=n_sims)
    sim.round(1).to_csv(out_dir / "tournament_probabilities.csv", index_label="team")
    show = ["played", "points", "exp_points", "top8_%", "playoff_%", "r16_%", "qf_%", "sf_%",
            "final_%", "winner_%"]
    print(sim[show].head(16).round(1).to_string())

    # ---- projected bracket --------------------------------------------------------
    proj_order = [idx[t] for t in sim.sort_values(["exp_points", "gd"], ascending=False).index]
    bracket, champion = tournament.projected_bracket(teams, proj_order, KH, KA)
    bracket.to_csv(out_dir / "projected_bracket.csv", index=False)
    print("\nProjected bracket (seeds from projected final league-phase table; the favourite "
          "advances in every tie):")
    for rnd, g in bracket.groupby("round", sort=False):
        print(f"\n  {rnd}")
        for a, b, pa, adv in zip(g["team_a"], g["team_b"], g["p_a_advances_%"], g["advances"]):
            print(f"    {a} vs {b}  ->  {adv}  ({a} advances {pa:.0f}%)")
    top = sim.index[0]
    print(f"\nProjected champion (bracket): {champion}")
    print(f"Highest title probability (simulation): {top} ({sim.iloc[0]['winner_%']:.1f}%)")

    # ---- figures ---------------------------------------------------------------------
    t8 = sim.sort_values("winner_%", ascending=False).head(10)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.barh(t8.index, t8["winner_%"], color="#0a1c4e")
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_xlabel("Probability of winning the Champions League (%)")
    ax.set_title(f"Title odds, {cur}-{str(cur + 1)[-2:]}")
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "title_odds.png", dpi=150)
    plt.close(fig)

    h = sim.head(16)[["r16_%", "qf_%", "sf_%", "final_%", "winner_%"]]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(h.to_numpy(), cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(5), ["Round of 16", "Quarter", "Semi", "Final", "Winner"])
    ax.set_yticks(range(len(h)), h.index)
    for i in range(len(h)):
        for j in range(5):
            v = h.iloc[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    color="white" if v > 55 else "black", fontsize=8)
    ax.set_title("Chance (%) of reaching each round")
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "advancement.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="use synthetic data (offline)")
    ap.add_argument("--model", default="xgb", choices=["logreg", "rf", "hgb", "xgb"])
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--no-clubelo", action="store_true", help="skip ClubElo ratings")
    args = ap.parse_args()
    if args.demo:
        from src.synthetic import make_synthetic
        m, f, e = make_synthetic()
        main(m, f, e, out_dir=config.OUTPUT_DIR / "demo", model_name=args.model, n_sims=args.sims)
    else:
        main(model_name=args.model, n_sims=args.sims, use_clubelo=not args.no_clubelo)
