"""Monte Carlo simulation of the Champions League (36-team league phase + knockouts).

Format simulated
  * League phase: 36 teams, 8 games each. Table order = points, goal difference, goals scored.
  * Top 8 -> round of 16.  9th-24th -> two-legged knockout PLAYOFFS.  25th-36th out.
  * Playoff pairs: 9/10 vs 23/24, 11/12 vs 21/22, 13/14 vs 19/20, 15/16 vs 17/18.
    Their winners meet seeds 7/8, 5/6, 3/4 and 1/2 in the round of 16.
  * Round of 16, quarter-finals, semi-finals: two legs, better league-phase finisher hosts
    leg 2. Aggregate score, then extra time (in leg 2's stadium), then a 50/50 penalty shootout
    (no away-goals rule). Final: one neutral-venue match.

Simplifications (also stated in the README)
  * UEFA draws the playoff / round-of-16 / quarter-final / semi-final pairings; the draw has
    extra rules (e.g. clubs from one country can't meet early). We draw pairings RANDOMLY
    within each seeded pair and ignore the country rules.
  * Team strength is frozen at today's values (no updating during the simulation).
  * Goals are independent Poisson draws.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

STAGES = ["top8", "playoff", "r16", "qf", "sf", "final", "winner"]


# ------------------------------------------------------------------ inputs
def standings(season_matches: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    """League-phase table so far."""
    t = pd.DataFrame(0, index=teams, columns=["played", "points", "gd", "gf"], dtype=float)
    lg = season_matches[(season_matches["stage"] == "league") & season_matches["FTHG"].notna()]
    for m in lg.itertuples():
        for team, gf, ga in ((m.HomeTeam, m.FTHG, m.FTAG), (m.AwayTeam, m.FTAG, m.FTHG)):
            if team in t.index:
                t.loc[team, "played"] += 1
                t.loc[team, "gf"] += gf
                t.loc[team, "gd"] += gf - ga
                t.loc[team, "points"] += 3 if gf > ga else 1 if gf == ga else 0
    return t


def lambda_matrices(builder, model, feats, teams, ext_now, knockout: int):
    """Expected goals for every ordered pair (i at home vs j): matrices LH[i,j], LA[i,j]."""
    T = len(teams)
    rows, idx = [], []
    for i, h in enumerate(teams):
        for j, a in enumerate(teams):
            if i != j:
                rows.append(builder.features_for(
                    h, a, None, knockout=knockout, leg2=0, neutral=0,
                    ext_home=ext_now.get(h, np.nan), ext_away=ext_now.get(a, np.nan)))
                idx.append((i, j))
    X = pd.DataFrame(rows)[feats]
    lh, la = model.lambdas(X)
    ii, jj = np.array(idx).T
    LH, LA = np.ones((T, T)), np.ones((T, T))
    LH[ii, jj], LA[ii, jj] = lh, la
    return LH, LA


# ------------------------------------------------------------------ league phase
def simulate_league_phase(table, rem_home, rem_away, LH, LA, n, rng):
    T = len(table)
    M = len(rem_home)
    pts = np.tile(table["points"].to_numpy(float), (n, 1))
    gd = np.tile(table["gd"].to_numpy(float), (n, 1))
    gf = np.tile(table["gf"].to_numpy(float), (n, 1))
    if M:
        hg = rng.poisson(LH[rem_home, rem_away][None, :], size=(n, M))
        ag = rng.poisson(LA[rem_home, rem_away][None, :], size=(n, M))
        H, A = np.eye(T)[rem_home], np.eye(T)[rem_away]          # (M, T) one-hot
        ph = 3 * (hg > ag) + (hg == ag)
        pa = 3 * (ag > hg) + (hg == ag)
        pts += ph @ H + pa @ A
        gd += (hg - ag) @ H + (ag - hg) @ A
        gf += hg @ H + ag @ A
    score = pts * 1e6 + gd * 1e3 + gf + rng.random((n, T)) * 1e-3   # random last-resort tiebreak
    order = np.argsort(-score, axis=1)                               # (n, T) team index by rank
    return order, pts


# ------------------------------------------------------------------ knockout matches
def knockout_tie(a, b, KH, KA, rank_of, rng):
    """Two-legged tie between arrays of team indices a, b. Returns winners."""
    rows = np.arange(len(a))
    ra, rb = rank_of[rows, a], rank_of[rows, b]
    second_home = np.where(ra < rb, a, b)            # better finisher hosts leg 2
    first_home = np.where(ra < rb, b, a)
    g1h = rng.poisson(KH[first_home, second_home])
    g1a = rng.poisson(KA[first_home, second_home])
    g2h = rng.poisson(KH[second_home, first_home])
    g2a = rng.poisson(KA[second_home, first_home])
    agg_first, agg_second = g1h + g2a, g1a + g2h
    level = agg_first == agg_second
    et_first = np.where(level, rng.poisson(KA[second_home, first_home] / 3.0), 0)
    et_second = np.where(level, rng.poisson(KH[second_home, first_home] / 3.0), 0)
    tot_first, tot_second = agg_first + et_first, agg_second + et_second
    coin = rng.random(len(a)) < 0.5                  # penalty shootout
    first_wins = np.where(tot_first != tot_second, tot_first > tot_second, coin)
    return np.where(first_wins, first_home, second_home)


def final_match(a, b, KH, KA, rng):
    """Single neutral-venue match: home advantage averaged out."""
    lam_a = 0.5 * (KH[a, b] + KA[b, a])
    lam_b = 0.5 * (KH[b, a] + KA[a, b])
    ga, gb = rng.poisson(lam_a), rng.poisson(lam_b)
    level = ga == gb
    ga = ga + np.where(level, rng.poisson(lam_a / 3.0), 0)
    gb = gb + np.where(level, rng.poisson(lam_b / 3.0), 0)
    coin = rng.random(len(a)) < 0.5
    return np.where(ga != gb, np.where(ga > gb, a, b), np.where(coin, a, b))


# ------------------------------------------------------------------ full simulation
def simulate(teams, table, rem_home, rem_away, LH, LA, KH, KA, n_sims=20000, seed=0):
    rng = np.random.default_rng(seed)
    T, n = len(teams), n_sims
    order, pts = simulate_league_phase(table, rem_home, rem_away, LH, LA, n, rng)
    rank_of = np.empty_like(order)
    rank_of[np.arange(n)[:, None], order] = np.arange(T)[None, :]

    counts = {s: np.zeros(T) for s in STAGES}

    def tally(stage, idx):
        counts[stage] += np.bincount(np.asarray(idx).ravel(), minlength=T)

    seed_of = lambda k: order[:, k]
    tally("top8", order[:, :config.N_DIRECT])
    tally("playoff", order[:, config.N_DIRECT:config.N_PLAYOFF])

    # --- playoffs: (seeded pair) vs (unseeded pair); unseeded side drawn at random
    groups = [((8, 9), (22, 23)), ((10, 11), (20, 21)), ((12, 13), (18, 19)), ((14, 15), (16, 17))]
    top_pairs = [(6, 7), (4, 5), (2, 3), (0, 1)]       # group g's winners face these seeds
    po_winners = []
    for (s1, s2), (u1, u2) in groups:
        sa, sb, ua, ub = seed_of(s1), seed_of(s2), seed_of(u1), seed_of(u2)
        flip = rng.random(n) < 0.5
        opp_a, opp_b = np.where(flip, ub, ua), np.where(flip, ua, ub)
        po_winners.append((knockout_tie(sa, opp_a, KH, KA, rank_of, rng),
                           knockout_tie(sb, opp_b, KH, KA, rank_of, rng)))
    tally("r16", np.column_stack([order[:, :config.N_DIRECT]]
                                 + [w for pair in po_winners for w in pair]))

    # --- round of 16 -> T1 (seeds 1/2), T2 (3/4), T3 (5/6), T4 (7/8)
    r16 = {}
    for g, (p, q) in enumerate(top_pairs):
        x, y = seed_of(p), seed_of(q)
        wa, wb = po_winners[g]
        flip = rng.random(n) < 0.5
        r16[(p, q)] = (knockout_tie(x, np.where(flip, wb, wa), KH, KA, rank_of, rng),
                       knockout_tie(y, np.where(flip, wa, wb), KH, KA, rank_of, rng))
    T1, T2, T3, T4 = r16[(0, 1)], r16[(2, 3)], r16[(4, 5)], r16[(6, 7)]
    tally("qf", np.column_stack(list(T1) + list(T2) + list(T3) + list(T4)))

    # --- quarter-finals: T1 vs T4 and T2 vs T3 (random pairing inside each half)
    def half(Ta, Tb):
        flip = rng.random(n) < 0.5
        return (knockout_tie(Ta[0], np.where(flip, Tb[1], Tb[0]), KH, KA, rank_of, rng),
                knockout_tie(Ta[1], np.where(flip, Tb[0], Tb[1]), KH, KA, rank_of, rng))
    qa, qb = half(T1, T4), half(T2, T3)
    tally("sf", np.column_stack(list(qa) + list(qb)))

    # --- semi-finals: one QF winner from each half meets one from the other
    flip = rng.random(n) < 0.5
    s1 = knockout_tie(qa[0], np.where(flip, qb[1], qb[0]), KH, KA, rank_of, rng)
    s2 = knockout_tie(qa[1], np.where(flip, qb[0], qb[1]), KH, KA, rank_of, rng)
    tally("final", np.column_stack([s1, s2]))
    tally("winner", final_match(s1, s2, KH, KA, rng))

    out = pd.DataFrame({"played": table["played"].astype(int), "points": table["points"].astype(int),
                        "gd": table["gd"].astype(int)}, index=teams)
    out["exp_points"] = pts.mean(axis=0)
    out["avg_rank"] = rank_of.mean(axis=0) + 1
    out["top8_%"] = counts["top8"] / n * 100
    out["playoff_%"] = counts["playoff"] / n * 100
    out["out_%"] = 100 - out["top8_%"] - out["playoff_%"]
    for s in ["r16", "qf", "sf", "final", "winner"]:
        out[f"{s}_%"] = counts[s] / n * 100
    return out.sort_values(["winner_%", "exp_points"], ascending=False)


# ------------------------------------------------------------------ "most likely" bracket
def projected_bracket(teams, proj_order, KH, KA, n=20000, seed=1):
    """A single readable bracket: seeds from projected finishing order, favourite advances.

    proj_order: team indices sorted by projected league-phase finish (best first).
    """
    rng = np.random.default_rng(seed)
    T = len(teams)
    rank_of = np.tile(np.argsort(proj_order), (n, 1))        # rank_of[i, team]
    seed_team = lambda s: proj_order[s - 1]                   # 1-based seed -> team index
    rows = []

    def tie(rnd, a, b):
        wa = np.full(n, a)
        wb = np.full(n, b)
        winners = knockout_tie(wa, wb, KH, KA, rank_of, rng)
        p = float((winners == a).mean())
        adv = a if p >= 0.5 else b
        rows.append({"round": rnd, "team_a": teams[a], "team_b": teams[b],
                     "p_a_advances_%": round(p * 100, 1), "advances": teams[adv]})
        return adv

    po = {s: tie("Playoff", seed_team(s), seed_team(33 - s)) for s in range(9, 17)}
    r16 = {k: tie("Round of 16", seed_team(k), po[17 - k]) for k in range(1, 9)}
    qf = {1: tie("Quarter-final", r16[1], r16[8]), 2: tie("Quarter-final", r16[2], r16[7]),
          3: tie("Quarter-final", r16[3], r16[6]), 4: tie("Quarter-final", r16[4], r16[5])}
    sf1 = tie("Semi-final", qf[1], qf[4])
    sf2 = tie("Semi-final", qf[2], qf[3])
    fa, fb = np.full(n, sf1), np.full(n, sf2)
    winners = final_match(fa, fb, KH, KA, rng)
    p = float((winners == sf1).mean())
    champ = sf1 if p >= 0.5 else sf2
    rows.append({"round": "Final", "team_a": teams[sf1], "team_b": teams[sf2],
                 "p_a_advances_%": round(p * 100, 1), "advances": teams[champ]})
    return pd.DataFrame(rows), teams[champ]
