"""ClubElo ratings: one strength number that is comparable across all European leagues.

Why: Champions League teams play only ~8-15 games a year against each other, so form from
those games alone is thin. ClubElo ratings use every match a club plays (domestic + Europe).

For every match we use the rating as of the DAY BEFORE kickoff (no leakage).
"""
from __future__ import annotations

import difflib
import time
from datetime import date, timedelta
from io import StringIO
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from src import config
from src.team_names import DEFAULT_ALIASES, norm

CACHE = config.RAW_DIR / "clubelo"
HEADERS = {"User-Agent": "champions-league-predictor (student project)"}


def _fetch(url: str, cache_file, refresh: bool = False) -> str | None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    if cache_file.exists() and not refresh:
        return cache_file.read_text(encoding="utf-8")
    try:
        with urlopen(Request(url, headers=HEADERS), timeout=15) as resp:
            text = resp.read().decode("utf-8")
        cache_file.write_text(text, encoding="utf-8")
        time.sleep(0.3)                      # be polite to a free service
        return text
    except Exception as exc:                 # network down, bad name, ...
        print(f"  ClubElo: could not fetch {url} ({exc})")
        return cache_file.read_text(encoding="utf-8") if cache_file.exists() else None


def snapshot(day: date) -> pd.DataFrame | None:
    """Ratings of every club on one day."""
    txt = _fetch(f"{config.CLUBELO_API}/{day.isoformat()}", CACHE / f"snapshot_{day}.csv",
                 refresh=False)
    return pd.read_csv(StringIO(txt)) if txt else None


def history(club: str) -> pd.DataFrame | None:
    txt = _fetch(f"{config.CLUBELO_API}/{quote(club)}",
                 CACHE / f"history_{norm(club).replace(' ', '_')}.csv",
                 refresh=False)
    if not txt:
        return None
    df = pd.read_csv(StringIO(txt))
    df["From"] = pd.to_datetime(df["From"])
    df["To"] = pd.to_datetime(df["To"])
    return df.sort_values("From").reset_index(drop=True)


def load_aliases() -> dict[str, str]:
    aliases = dict(DEFAULT_ALIASES)
    if config.ALIAS_FILE.exists():                       # your edits win
        user = pd.read_csv(config.ALIAS_FILE)
        aliases.update({norm(t): c for t, c in zip(user["team"], user["clubelo"])})
    return aliases


def match_names(teams: list[str], club_names: list[str]) -> pd.DataFrame:
    """Map our team names to ClubElo club names. Review the result file once!"""
    aliases = load_aliases()
    by_norm = {norm(c): c for c in club_names}
    rows = []
    for team in sorted(set(teams)):
        key = norm(team)
        club, method = None, "none"
        if key in aliases and aliases[key] in club_names:
            club, method = aliases[key], "alias"
        elif key in by_norm:
            club, method = by_norm[key], "exact"
        else:
            hit = difflib.get_close_matches(key, list(by_norm), n=1, cutoff=0.75)
            if hit:
                club, method = by_norm[hit[0]], "fuzzy"
        rows.append({"team": team, "clubelo": club, "method": method})
    return pd.DataFrame(rows)


def _rating_on(hist: pd.DataFrame, day: pd.Timestamp) -> float:
    """Rating valid at the end of the day before `day`."""
    i = np.searchsorted(hist["From"].to_numpy(), np.datetime64(day - pd.Timedelta(days=1)),
                        side="right") - 1
    return float(hist["Elo"].iloc[i]) if i >= 0 else np.nan


def add_external_elo(matches: pd.DataFrame, fixtures: pd.DataFrame):
    """Returns (matches, fixtures, ext_now) with ext_elo_home / ext_elo_away filled in."""
    matches, fixtures = matches.copy(), fixtures.copy()
    for df in (matches, fixtures):
        df["ext_elo_home"] = np.nan
        df["ext_elo_away"] = np.nan

    snap = snapshot(date.today())
    if snap is None:
        print("ClubElo unavailable: continuing with the Champions League-only Elo.")
        return matches, fixtures, {}

    teams = sorted(set(matches.HomeTeam) | set(matches.AwayTeam)
                   | set(fixtures.get("HomeTeam", [])) | set(fixtures.get("AwayTeam", [])))
    mapping = match_names(teams, list(snap["Club"]))
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(config.PROCESSED_DIR / "clubelo_name_matches.csv", index=False)
    missing = mapping[mapping["clubelo"].isna()]["team"].tolist()
    fuzzy = mapping[mapping["method"] == "fuzzy"]
    print(f"ClubElo name matching: {len(mapping) - len(missing)}/{len(mapping)} matched "
          f"({len(fuzzy)} fuzzy). Review data/processed/clubelo_name_matches.csv")
    if missing:
        print("  NOT matched (add them to data/team_aliases.csv):", ", ".join(missing))

    name_to_club = dict(zip(mapping["team"], mapping["clubelo"]))
    hists = {}
    for club in sorted({c for c in name_to_club.values() if c}):
        h = history(club)
        if h is not None:
            hists[club] = h

    def lookup(team, day):
        h = hists.get(name_to_club.get(team))
        return _rating_on(h, day) if h is not None else np.nan

    for df in (matches, fixtures):
        if len(df):
            df["ext_elo_home"] = [lookup(t, d) for t, d in zip(df.HomeTeam, df.Date)]
            df["ext_elo_away"] = [lookup(t, d) for t, d in zip(df.AwayTeam, df.Date)]

    ratings = dict(zip(snap["Club"], snap["Elo"]))
    ext_now = {t: float(ratings[c]) for t, c in name_to_club.items() if c in ratings}
    return matches, fixtures, ext_now
