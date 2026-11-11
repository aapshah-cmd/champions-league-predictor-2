"""Download Champions League results + schedule from OpenLigaDB (free, no API key).

    python -m src.data_loader
"""
from __future__ import annotations

import json
import sys
import time
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from src import config, team_names

OPENLIGA_DIR = config.RAW_DIR / "openligadb"
HEADERS = {"User-Agent": "champions-league-predictor (student project)"}
BAD_WORDS = ("women", "frauen", "qualif", "youth", "junior", "asia", "afc", "caf",
             "concacaf", "futsal")


def _get_json(url: str, cache_name: str, refresh: bool):
    OPENLIGA_DIR.mkdir(parents=True, exist_ok=True)
    path = OPENLIGA_DIR / cache_name
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        with urlopen(Request(url, headers=HEADERS), timeout=40) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        path.write_text(json.dumps(data), encoding="utf-8")
        time.sleep(0.3)
        return data
    except Exception as exc:
        print(f"  could not fetch {url}: {exc}")
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def stage_from_group(name: str) -> str:
    """OpenLigaDB labels rounds in German or English; map them to our stage codes."""
    s = (name or "").lower()
    if "qualif" in s or "quali" in s:
        return "qualifying"
    if "achtel" in s or "round of 16" in s or "last 16" in s:
        return "r16"
    if "viertel" in s or "quarter" in s:
        return "qf"
    if "halb" in s or "semi" in s:
        return "sf"
    if "playoff" in s or "play-off" in s or "zwischenrunde" in s:
        return "playoff"
    if "finale" in s or s.strip() == "final":
        return "final"
    return "league"             # "3. Spieltag", "Gruppenphase", ...


def available_competitions(refresh: bool = False) -> list[tuple[str, int]]:
    leagues = _get_json(f"{config.OPENLIGA_API}/getavailableleagues", "leagues.json", refresh)
    if not leagues:
        return []
    out = set()
    for lg in leagues:
        name = (lg.get("leagueName") or "").lower()
        if "champions league" in name and not any(w in name for w in BAD_WORDS):
            try:
                out.add((lg["leagueShortcut"], int(lg["leagueSeason"])))
            except (KeyError, ValueError, TypeError):
                pass
    return sorted(out, key=lambda x: x[1])


def _parse(matches: list[dict], season: int) -> list[dict]:
    rows = []
    for m in matches:
        try:
            when = pd.to_datetime(m.get("matchDateTimeUTC") or m.get("matchDateTime"), utc=True)
            when = when.tz_localize(None)
            results = m.get("matchResults") or []
            final = next((r for r in results if r.get("resultTypeID") == 2), None)
            if final is None and results:
                final = max(results, key=lambda r: r.get("resultOrderID", 0))
            finished = bool(m.get("matchIsFinished")) and final is not None
            rows.append({
                "Date": when, "season": season,
                "HomeTeam": m["team1"]["teamName"], "AwayTeam": m["team2"]["teamName"],
                "FTHG": final["pointsTeam1"] if finished else np.nan,
                "FTAG": final["pointsTeam2"] if finished else np.nan,
                "stage": stage_from_group((m.get("group") or {}).get("groupName", "")),
                "finished": finished,
            })
        except (KeyError, TypeError, ValueError):
            continue                          # skip malformed rows
    return rows


def finalize_cl(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw rows (Date, season, HomeTeam, AwayTeam, FTHG, FTAG, stage) -> project schema."""
    df = raw.copy()
    df = df[df["stage"] != "qualifying"]
    df = team_names.canonicalize(df)
    df["Date"] = pd.to_datetime(df["Date"])
    df["season"] = df["Date"].apply(lambda d: d.year if d.month >= 7 else d.year - 1)
    df = df.sort_values("Date", kind="mergesort").reset_index(drop=True)
    for col in ("ext_elo_home", "ext_elo_away"):
        if col not in df:
            df[col] = np.nan
    df["is_knockout"] = df["stage"].isin(config.KNOCKOUT_STAGES).astype(int)
    df["neutral"] = (df["stage"] == "final").astype(int)

    # leg 1 / leg 2 = order of the two games between the same teams in the same round
    df["leg"] = 0
    ko = df[df["is_knockout"] == 1]
    key = ko.apply(lambda r: (r["season"], r["stage"], frozenset((r["HomeTeam"], r["AwayTeam"]))),
                   axis=1)
    order = ko.groupby(key).cumcount() + 1
    size = ko.groupby(key)["Date"].transform("size")
    df.loc[ko.index, "leg"] = np.where(size == 2, order, 0)

    played = df["FTHG"].notna()
    df["FTR"] = np.where(~played, None,
                         np.where(df["FTHG"] > df["FTAG"], "H",
                                  np.where(df["FTHG"] < df["FTAG"], "A", "D")))
    df["match_id"] = np.arange(len(df))
    cols = ["match_id", "Date", "season", "stage", "is_knockout", "neutral", "leg",
            "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "ext_elo_home", "ext_elo_away"]
    return df[cols]


BIG_CLUBS = ("real madrid", "barcelona", "bayern", "manchester", "juventus", "chelsea",
             "liverpool", "arsenal", "paris", "inter", "milan", "atletico", "dortmund")


def _expected_matches(season: int) -> int:
    return (config.FULL_SEASON_MATCHES_NEW if season >= 2024
            else config.FULL_SEASON_MATCHES_OLD)


def _looks_like_uefa(rows: pd.DataFrame) -> bool:
    """A real Champions League season contains several of Europe's biggest clubs."""
    names = {team_names.norm(n) for n in pd.concat([rows["HomeTeam"], rows["AwayTeam"]])}
    return sum(any(b in n for n in names) for b in BIG_CLUBS) >= 4


def choose_competitions(cands: list[dict]):
    """OpenLigaDB mixes in hobby leagues, other sports and duplicate copies of a season.
    Keep exactly ONE competition per season: right size for the format AND real UEFA clubs.

    cands: [{"shortcut", "season", "rows": DataFrame}]. Returns (chosen, report DataFrame).
    """
    by_season: dict[int, list[dict]] = {}
    for c in cands:
        by_season.setdefault(c["season"], []).append(c)
    chosen, report = [], []
    for season, group in sorted(by_season.items()):
        expected = _expected_matches(season)
        forced = config.FORCE_COMPETITIONS.get(season)
        valid = []
        for c in group:
            n = len(c["rows"])
            if forced is not None:
                ok, why = c["shortcut"] == forced, "forced" if c["shortcut"] == forced else "not forced"
            else:
                if season == config.CURRENT_SEASON:     # partly played: schedule must be there
                    size_ok = config.N_TEAMS * 8 // 2 <= n <= expected * 1.03
                else:
                    size_ok = expected * 0.97 <= n <= expected * 1.03
                uefa = _looks_like_uefa(c["rows"])
                ok = size_ok and uefa
                why = "ok" if ok else ("wrong size" if not size_ok else "not UEFA clubs")
            report.append({"season": season, "shortcut": c["shortcut"], "n_matches": n,
                           "status": why, "used": False})
            if ok:
                valid.append(c)
        if valid:
            best = min(valid, key=lambda c: (abs(len(c["rows"]) - expected), c["shortcut"]))
            chosen.append(best)
            for r in report:
                if r["season"] == season and r["shortcut"] == best["shortcut"]:
                    r["used"], r["status"] = True, "USED"
                elif r["season"] == season and r["status"] == "ok":
                    r["status"] = "duplicate of the used one"
    return chosen, pd.DataFrame(report)


def load_cl_matches(refresh_current: bool = True):
    """Returns (played matches, scheduled fixtures of the current season)."""
    comps = available_competitions(refresh=refresh_current)
    if not comps:
        sys.exit("Could not list Champions League seasons from OpenLigaDB. "
                 "Check your internet connection (and the SSL certificate note in the README).")
    cands = []
    for shortcut, season in comps:
        is_current = season == config.CURRENT_SEASON
        data = _get_json(f"{config.OPENLIGA_API}/getmatchdata/{shortcut}/{season}",
                         f"{shortcut}_{season}.json", refresh=is_current and refresh_current)
        rows = _parse(data, season) if data else []
        if rows:
            cands.append({"shortcut": shortcut, "season": season, "rows": pd.DataFrame(rows)})

    chosen, report = choose_competitions(cands)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(config.PROCESSED_DIR / "competitions_used.csv", index=False)
    if not chosen:
        sys.exit("No competition looked like the real Champions League. See "
                 "data/processed/competitions_used.csv and set FORCE_COMPETITIONS in config.py.")
    print("\nChampions League competitions used (one per season):")
    for c in chosen:
        print(f"  {c['season']}: {c['shortcut']:<10} {len(c['rows'])} matches")
    ignored = report[~report["used"]]
    print(f"Ignored {len(ignored)} other OpenLigaDB competitions (hobby leagues, other sports, "
          f"duplicates, partial seasons). Details: data/processed/competitions_used.csv\n")

    raw = pd.concat([c["rows"] for c in chosen], ignore_index=True)
    done = raw[raw["finished"]].drop(columns="finished")
    todo = raw[(~raw["finished"]) & (raw["season"] == config.CURRENT_SEASON)]
    todo = todo.drop(columns=["finished", "FTHG", "FTAG"]).assign(FTHG=np.nan, FTAG=np.nan)
    played = finalize_cl(done)
    fixtures = finalize_cl(todo) if len(todo) else played.iloc[0:0].copy()
    return played, fixtures


def load_all(use_clubelo: bool = True):
    """(matches, fixtures, ext_now): matches with ClubElo ratings attached."""
    matches, fixtures = load_cl_matches()
    ext_now: dict = {}
    if use_clubelo:
        from src import clubelo
        matches, fixtures, ext_now = clubelo.add_external_elo(matches, fixtures)
    return matches, fixtures, ext_now


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-clubelo", action="store_true", help="skip ClubElo ratings")
    args = ap.parse_args()
    print("Downloading Champions League data (OpenLigaDB) ...")
    m, f, _ = load_all(use_clubelo=not args.no_clubelo)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    m.to_csv(config.PROCESSED_DIR / "matches.csv", index=False)
    print(f"\n{len(m)} played matches, seasons {m.season.min()}-{m.season.max()}, "
          f"{len(f)} scheduled fixtures this season")
    print(m.groupby("stage").size().to_string())
