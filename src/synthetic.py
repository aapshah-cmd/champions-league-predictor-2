"""Synthetic Champions League-like data for the offline demo and the tests.

Hidden attack/defence strengths drive Poisson goals. Each season 36 clubs from a pool of 60
play the 36-team league phase (8 games each) followed by playoffs, round of 16, quarter-finals,
semi-finals and a final. The current season is only partly played. A noisy "ClubElo-like"
rating is attached so the full feature set is exercised.

This is NOT real data: never report results from it.
"""
from __future__ import annotations

from math import gcd

import numpy as np
import pandas as pd

from src import config
from src.data_loader import finalize_cl

POOL, N = 60, config.N_TEAMS
OFFSETS = (1, 2, 3, 5)                       # team i hosts teams i+k (mod 36): 4 home, 4 away
MD_DAYS = [0, 35, 42, 56, 77, 91, 133, 140]  # days after the first matchday


def make_synthetic(seed: int = 7, n_seasons: int = 11, current_matchdays_played: int = 4):
    """Returns (matches, fixtures, ext_now)."""
    rng = np.random.default_rng(seed)
    pool = [f"Club{i:02d}" for i in range(POOL)]
    att = dict(zip(pool, rng.normal(0, 0.30, POOL)))
    dfn = dict(zip(pool, rng.normal(0, 0.30, POOL)))
    records, fixtures, ext_now = [], [], {}
    first = config.CURRENT_SEASON - n_seasons + 1

    for season in range(first, config.CURRENT_SEASON + 1):
        for t in pool:                                   # strengths drift between seasons
            att[t] = 0.9 * att[t] + rng.normal(0, 0.08)
            dfn[t] = 0.9 * dfn[t] + rng.normal(0, 0.08)
        strength = np.array([att[t] + dfn[t] for t in pool])
        p = np.exp(2 * strength)
        teams = list(rng.choice(pool, N, replace=False, p=p / p.sum()))
        rng.shuffle(teams)
        elo = {t: 1700 + 350 * (att[t] + dfn[t]) + rng.normal(0, 25) for t in teams}
        start = pd.Timestamp(season, 9, 8)
        is_current = season == config.CURRENT_SEASON

        def play(h, a, when, stage, neutral=False):
            lam_h = np.exp(0.28 + att[h] - dfn[a]) if not neutral else np.exp(0.14 + att[h] - dfn[a])
            lam_a = np.exp(0.0 + att[a] - dfn[h]) if not neutral else np.exp(0.14 + att[a] - dfn[h])
            return {"Date": when, "season": season, "HomeTeam": h, "AwayTeam": a,
                    "FTHG": rng.poisson(lam_h), "FTAG": rng.poisson(lam_a), "stage": stage,
                    "ext_elo_home": elo[h], "ext_elo_away": elo[a]}

        # ---- league phase
        table = {t: [0, 0, 0] for t in teams}                  # pts, gd, gf
        for j, k in enumerate(OFFSETS):
            g = gcd(N, k)
            for i in range(N):
                md = 2 * j + (i // g) % 2
                h, a = teams[i], teams[(i + k) % N]
                when = start + pd.Timedelta(days=MD_DAYS[md])
                if is_current and md >= current_matchdays_played:
                    fixtures.append({"Date": when, "season": season, "HomeTeam": h,
                                     "AwayTeam": a, "FTHG": np.nan, "FTAG": np.nan,
                                     "stage": "league", "ext_elo_home": elo[h],
                                     "ext_elo_away": elo[a]})
                    continue
                r = play(h, a, when, "league")
                records.append(r)
                hg, ag = r["FTHG"], r["FTAG"]
                table[h][0] += 3 if hg > ag else int(hg == ag)
                table[a][0] += 3 if ag > hg else int(hg == ag)
                table[h][1] += hg - ag; table[a][1] += ag - hg
                table[h][2] += hg; table[a][2] += ag
        if is_current:
            ext_now = dict(elo)
            continue

        # ---- knockouts (simple fixed bracket: this only generates training data)
        rank = sorted(teams, key=lambda t: (-table[t][0], -table[t][1], -table[t][2], rng.random()))
        seed_t = lambda s: rank[s - 1]

        def two_legs(low, high, rnd_day, stage):
            """low-ranked team hosts leg 1; returns winner by aggregate (coin flip if level)."""
            d1 = pd.Timestamp(season + 1, *rnd_day)
            r1, r2 = play(low, high, d1, stage), play(high, low, d1 + pd.Timedelta(days=7), stage)
            records.extend([r1, r2])
            a_low, a_high = r1["FTHG"] + r2["FTAG"], r1["FTAG"] + r2["FTHG"]
            if a_low == a_high:
                return low if rng.random() < 0.5 else high
            return low if a_low > a_high else high

        po = {s: two_legs(seed_t(33 - s), seed_t(s), (2, 17), "playoff") for s in range(9, 17)}
        r16 = {k: two_legs(po[17 - k], seed_t(k), (3, 10), "r16") for k in range(1, 9)}
        qf = {1: (r16[1], r16[8]), 2: (r16[2], r16[7]), 3: (r16[3], r16[6]), 4: (r16[4], r16[5])}
        qfw = {k: two_legs(a, b, (4, 7), "qf") for k, (a, b) in qf.items()}
        s1 = two_legs(qfw[1], qfw[4], (4, 28), "sf")
        s2 = two_legs(qfw[2], qfw[3], (4, 28), "sf")
        records.append(play(s1, s2, pd.Timestamp(season + 1, 6, 5), "final", neutral=True))

    matches = finalize_cl(pd.DataFrame(records))
    fixtures = finalize_cl(pd.DataFrame(fixtures))
    return matches, fixtures, ext_now
