"""Leak-free feature engineering for Champions League matches.

Matches are processed in date order. For each match we compute features from what the
teams had done BEFORE kickoff, and only then update histories and Elo ratings
(proved in tests/test_features.py).

Features (h_ = home team, a_ = away team, diff_ = home minus away):
  * rolling form over the team's last 5 / 10 Champions League matches: goals for/against,
    points per game, goal difference, days of rest, number of CL matches so far (experience)
  * elo_*  : a Champions League-only Elo rating built from the matches in this dataset
  * ce_*   : ClubElo rating (covers domestic + European results of every club)
  * ctx_*  : knockout round? second leg?
Neutral venue (the final) removes home advantage from the Elo win expectancy.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

from src import config

SHORT, LONG = 5, 10
ELO_NEW, ELO_MEAN, ELO_K, ELO_HOME_ADV, ELO_SEASON_REGRESSION = 1500.0, 1500.0, 25.0, 60.0, 0.15
CE_HOME_ADV = 60.0
GF, GA, PTS = range(3)
DIFF_STATS = ["gf5", "ga5", "ppg5", "ppg10", "gd10"]


def _avg(rows, idx: int, k: int) -> float:
    vals = [r[idx] for r in list(rows)[-k:]]
    return float(np.mean(vals)) if vals else np.nan


class TeamState:
    def __init__(self) -> None:
        self.recent = deque(maxlen=LONG)
        self.last_date: pd.Timestamp | None = None
        self.n = 0


def _team_features(s: TeamState, date) -> dict:
    gf10, ga10 = _avg(s.recent, GF, LONG), _avg(s.recent, GA, LONG)
    if s.last_date is None:
        rest = np.nan
    elif date is None:
        rest = 7.0
    else:
        rest = float(min((date - s.last_date).days, 60))
    return {"gf5": _avg(s.recent, GF, SHORT), "ga5": _avg(s.recent, GA, SHORT),
            "ppg5": _avg(s.recent, PTS, SHORT), "ppg10": _avg(s.recent, PTS, LONG),
            "gd10": gf10 - ga10, "rest": rest, "games": float(s.n)}


def _expectancy(r_home: float, r_away: float, neutral: bool, adv: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(r_home + (0.0 if neutral else adv) - r_away) / 400.0))


class FeatureBuilder:
    def __init__(self) -> None:
        self.states: dict[str, TeamState] = defaultdict(TeamState)
        self.elo: dict[str, float] = {}
        self._season: int | None = None

    def features_for(self, home: str, away: str, date=None, knockout: int = 0,
                     leg2: int = 0, neutral: int = 0, ext_home=np.nan, ext_away=np.nan) -> dict:
        h, a = _team_features(self.states[home], date), _team_features(self.states[away], date)
        f = {f"h_{k}": v for k, v in h.items()}
        f.update({f"a_{k}": v for k, v in a.items()})
        for stat in DIFF_STATS:
            f[f"diff_{stat}"] = h[stat] - a[stat]
        eh, ea = self.elo.get(home, ELO_NEW), self.elo.get(away, ELO_NEW)
        f.update({"elo_home": eh, "elo_away": ea, "elo_diff": eh - ea,
                  "elo_p_home": _expectancy(eh, ea, bool(neutral), ELO_HOME_ADV)})
        ext_home = np.nan if ext_home is None else ext_home
        ext_away = np.nan if ext_away is None else ext_away
        f.update({"ce_home": ext_home, "ce_away": ext_away, "ce_diff": ext_home - ext_away,
                  "ce_p_home": (_expectancy(ext_home, ext_away, bool(neutral), CE_HOME_ADV)
                                if not (np.isnan(ext_home) or np.isnan(ext_away)) else np.nan)})
        f.update({"ctx_knockout": float(knockout), "ctx_leg2": float(leg2 == 2)})
        return f

    def _new_season(self, gap: int = 1) -> None:
        """Regress ratings to the mean; if whole seasons are missing from the data,
        regress more and forget the (now stale) recent form."""
        keep = (1 - ELO_SEASON_REGRESSION) ** gap
        for t, r in self.elo.items():
            self.elo[t] = ELO_MEAN + (r - ELO_MEAN) * keep
        if gap > 1:
            for state in self.states.values():
                state.recent.clear()

    def _update(self, row) -> None:
        hg, ag = float(row.FTHG), float(row.FTAG)
        hp, ap = (3, 0) if hg > ag else (0, 3) if hg < ag else (1, 1)
        for team, rec in ((row.HomeTeam, (hg, ag, hp)), (row.AwayTeam, (ag, hg, ap))):
            s = self.states[team]
            s.recent.append(rec)
            s.n += 1
            s.last_date = row.Date
        eh, ea = self.elo.get(row.HomeTeam, ELO_NEW), self.elo.get(row.AwayTeam, ELO_NEW)
        exp = _expectancy(eh, ea, bool(row.neutral), ELO_HOME_ADV)
        score = 1.0 if hg > ag else 0.0 if hg < ag else 0.5
        gd = abs(hg - ag)
        mult = 1.0 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8
        delta = ELO_K * mult * (score - exp)
        self.elo[row.HomeTeam], self.elo[row.AwayTeam] = eh + delta, ea - delta

    def build(self, matches: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for row in matches.itertuples(index=False):
            if self._season is not None and row.season != self._season:
                self._new_season(max(1, int(row.season) - int(self._season)))
            self._season = row.season
            rows.append(self.features_for(
                row.HomeTeam, row.AwayTeam, row.Date, row.is_knockout, row.leg, row.neutral,
                row.ext_elo_home, row.ext_elo_away))
            self._update(row)               # only AFTER the features were computed
        return pd.DataFrame(rows, index=matches.index)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(("h_", "a_", "diff_", "elo_", "ce_", "ctx_"))]


def make_dataset(matches: pd.DataFrame):
    builder = FeatureBuilder()
    feats = builder.build(matches)
    df = pd.concat([matches, feats], axis=1)
    df["y"] = df["FTR"].map(config.LABEL_MAP)
    return df, builder
