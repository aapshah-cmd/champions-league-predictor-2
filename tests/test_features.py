"""The tests that matter most: no leakage, and a tournament simulator that obeys the format."""
import numpy as np
import pandas as pd

from src import config, evaluate, tournament
from src.features import FeatureBuilder, feature_columns, make_dataset
from src.synthetic import make_synthetic

MATCHES, FIXTURES, EXT_NOW = make_synthetic(seed=1, n_seasons=4)


def test_no_leakage_from_own_or_future_results():
    k = 300
    base = FeatureBuilder().build(MATCHES)
    altered = MATCHES.copy()
    altered.loc[k, "FTHG"] += 5                    # pretend the home side scored 5 more
    changed = FeatureBuilder().build(altered)
    # every match up to and INCLUDING k must be unaffected by k's result
    pd.testing.assert_frame_equal(base.iloc[: k + 1], changed.iloc[: k + 1])
    # later matches should be affected, proving the test is sensitive
    assert not base.iloc[k + 1:].equals(changed.iloc[k + 1:])


def test_dataset_shape_and_target():
    df, _ = make_dataset(MATCHES)
    feats = feature_columns(df)
    assert len(df) == len(MATCHES) and len(feats) > 20
    assert set(df["y"].unique()) == {0, 1, 2}
    assert not any(c in feats for c in ["FTHG", "FTAG", "FTR", "y"])


def test_stage_and_leg_labels():
    ko = MATCHES[MATCHES["is_knockout"] == 1]
    assert set(ko["stage"]) == set(config.KNOCKOUT_STAGES)
    assert set(ko[ko.stage == "final"]["leg"]) == {0}
    assert set(ko[ko.stage == "qf"]["leg"]) == {1, 2}
    assert (MATCHES[MATCHES.stage == "final"]["neutral"] == 1).all()


def test_tournament_obeys_format():
    T = config.N_TEAMS
    teams = [f"T{i}" for i in range(T)]
    table = pd.DataFrame(0.0, index=teams, columns=["played", "points", "gd", "gf"])
    # identical teams: every outcome should be equally likely for every team
    LH, LA = np.full((T, T), 1.4), np.full((T, T), 1.1)
    # remaining matches: each team hosts the next four teams
    hi = np.array([i for i in range(T) for _ in range(4)])
    ai = np.array([(i + k) % T for i in range(T) for k in range(1, 5)])
    n = 20000
    sim = tournament.simulate(teams, table, hi, ai, LH, LA, LH, LA, n_sims=n, seed=3)
    assert abs(sim["top8_%"].sum() - 800) < 1e-6
    assert abs(sim["playoff_%"].sum() - 1600) < 1e-6
    assert abs(sim["r16_%"].sum() - 1600) < 1e-6
    assert abs(sim["qf_%"].sum() - 800) < 1e-6
    assert abs(sim["sf_%"].sum() - 400) < 1e-6
    assert abs(sim["final_%"].sum() - 200) < 1e-6
    assert abs(sim["winner_%"].sum() - 100) < 1e-6
    # identical teams -> each wins about 1/36 of the time
    assert sim["winner_%"].between(100 / T - 1.2, 100 / T + 1.2).all()
    # reaching the next round never gets easier
    assert (sim["qf_%"] <= sim["r16_%"] + 1e-9).all() and (sim["winner_%"] <= sim["final_%"] + 1e-9).all()


def test_stronger_team_wins_more_often():
    T = config.N_TEAMS
    teams = [f"T{i}" for i in range(T)]
    table = pd.DataFrame(0.0, index=teams, columns=["played", "points", "gd", "gf"])
    LH, LA = np.full((T, T), 1.4), np.full((T, T), 1.1)
    # LH[i, j] / LA[i, j] = goals by the home / away side when team i hosts team j
    LH[0, :], LA[0, :] = 2.6, 0.5          # team 0 at home: scores a lot, concedes little
    LA[:, 0], LH[:, 0] = 2.2, 0.6          # team 0 away: same
    hi = np.array([i for i in range(T) for _ in range(4)])
    ai = np.array([(i + k) % T for i in range(T) for k in range(1, 5)])
    sim = tournament.simulate(teams, table, hi, ai, LH, LA, LH, LA, n_sims=10000, seed=5)
    assert sim.index[0] == "T0" and sim.loc["T0", "winner_%"] > 15


def test_projected_bracket_shape():
    T = config.N_TEAMS
    teams = [f"T{i}" for i in range(T)]
    LH, LA = np.full((T, T), 1.4), np.full((T, T), 1.1)
    bracket, champ = tournament.projected_bracket(teams, list(range(T)), LH, LA, n=2000)
    assert bracket.groupby("round").size().to_dict() == {
        "Playoff": 8, "Round of 16": 8, "Quarter-final": 4, "Semi-final": 2, "Final": 1}
    assert champ in teams


def test_metrics_basic():
    y = np.array([0, 1, 2])
    assert evaluate.rps(y, np.eye(3)) == 0 and evaluate.brier(y, np.eye(3)) == 0
    assert abs(evaluate.log_loss(y, np.full((3, 3), 1 / 3)) - np.log(3)) < 1e-9


# ---------------------------------------------------------------- data-source cleaning
def _fake_rows(n, uefa=True):
    real = ["Real Madrid", "FC Barcelona", "FC Bayern München", "Manchester City", "Juventus Turin",
            "Chelsea", "Liverpool", "Arsenal", "Paris Saint-Germain", "AC Mailand", "Dortmund",
            "Benfica", "Porto", "Ajax"]
    fake = [f"Hobby Team {i}" for i in range(14)]
    pool = real if uefa else fake
    return pd.DataFrame({"HomeTeam": [pool[i % 14] for i in range(n)],
                         "AwayTeam": [pool[(i + 3) % 14] for i in range(n)]})


def test_choose_competitions_on_real_opendata_listing():
    """The exact (season, shortcut, matches) list OpenLigaDB returned, incl. junk and duplicates."""
    from src.data_loader import choose_competitions
    listing = [
        (2008, "cl", 93, 1), (2009, "CL", 32, 1), (2010, "cl1011", 125, 1), (2010, "hcl1011", 148, 0),
        (2012, "hcl", 148, 0), (2012, "cl", 125, 1), (2014, "hcl", 148, 0), (2015, "hcl", 200, 0),
        (2015, "cl", 125, 1), (2015, "cl15", 43, 0), (2016, "cl1617", 96, 1), (2016, "mbchamp", 81, 0),
        (2016, "ucl2016", 125, 1), (2016, "hcl", 172, 0), (2017, "cl1718", 125, 1),
        (2017, "ucl2017", 125, 1), (2018, "cl1819", 68, 1), (2018, "cl2018", 125, 1),
        (2022, "uefacl22", 95, 1), (2023, "champion1", 125, 1), (2023, "cl1", 3, 1),
        (2024, "ucl2024", 187, 1), (2024, "ucl24", 189, 1), (2024, "cl24de", 52, 0),
        (2025, "ucl2025", 76, 1), (2025, "ucl", 189, 1),
        (config.CURRENT_SEASON, "ucl2026", 32, 1), (config.CURRENT_SEASON, "ucl", 144, 1),
    ]
    cands = [{"shortcut": sc, "season": se, "rows": _fake_rows(n, bool(u))}
             for se, sc, n, u in listing]
    chosen, report = choose_competitions(cands)
    picked = {c["season"]: c["shortcut"] for c in chosen}
    assert picked == {2010: "cl1011", 2012: "cl", 2015: "cl", 2016: "ucl2016", 2017: "cl1718",
                      2018: "cl2018", 2023: "champion1", 2024: "ucl24", 2025: "ucl",
                      config.CURRENT_SEASON: "ucl"}
    assert len(picked) == len(chosen)                      # exactly one per season


def test_missing_seasons_reset_stale_form():
    # drop 2024 and 2025 so 2023 is followed directly by the current season
    gap = MATCHES[~MATCHES["season"].isin([2024, 2025])].reset_index(drop=True)
    feats = FeatureBuilder().build(gap)
    new_season = gap["season"] == config.CURRENT_SEASON
    first_day = new_season & (gap["Date"] == gap.loc[new_season, "Date"].min())
    veterans = first_day & (feats["h_games"] > 0)           # clubs seen before the gap
    assert veterans.sum() > 0
    assert feats.loc[veterans, "h_gf5"].isna().all()         # old form was forgotten
