"""Team-name cleaning shared by the match data and the ClubElo ratings."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

import pandas as pd

STOP = {"fc", "afc", "cf", "sc", "ac", "as", "ss", "ssc", "us", "fk", "sk", "bk", "sv",
        "sl", "cd", "ud", "rcd", "gnk", "nk", "1", "04", "1899", "1909", "bsc", "ogc",
        "rc", "ca", "kv", "cp", "de", "the"}

# normalised team name -> ClubElo spelling (extend freely, or use data/team_aliases.csv)
DEFAULT_ALIASES = {
    "manchester city": "Man City", "manchester united": "Man United",
    "paris saint germain": "Paris SG", "paris sg": "Paris SG", "psg": "Paris SG",
    "bayern munchen": "Bayern", "bayern munich": "Bayern", "bayern": "Bayern",
    "borussia dortmund": "Dortmund", "bayer leverkusen": "Leverkusen",
    "bayer 04 leverkusen": "Leverkusen", "eintracht frankfurt": "Frankfurt",
    "atletico madrid": "Atletico", "atletico de madrid": "Atletico",
    "tottenham hotspur": "Tottenham", "newcastle united": "Newcastle",
    "sporting lissabon": "Sporting", "sporting": "Sporting", "benfica lissabon": "Benfica",
    "inter mailand": "Inter", "inter milan": "Inter", "mailand": "Milan",
    "milan": "Milan", "neapel": "Napoli", "rom": "Roma", "juventus turin": "Juventus",
    "marseille": "Marseille", "olympique marseille": "Marseille",
    "olympique lyon": "Lyon", "lyon": "Lyon", "nizza": "Nice",
    "club brugge": "Brugge", "fc brugge": "Brugge", "brugge": "Brugge",
    "slavia prag": "Slavia Praha", "sparta prag": "Sparta Praha",
    "schachtar donezk": "Shakhtar", "shakhtar donetsk": "Shakhtar",
    "roter stern belgrad": "Crvena Zvezda", "red star belgrade": "Crvena Zvezda",
    "dynamo kiew": "Dynamo Kyiv", "kopenhagen": "Kobenhavn", "copenhagen": "Kobenhavn",
    "salzburg": "Salzburg", "red bull salzburg": "Salzburg", "rb salzburg": "Salzburg",
    "olympiakos piraus": "Olympiakos", "olympiacos": "Olympiakos",
    "paok saloniki": "PAOK", "young boys bern": "Young Boys", "celtic glasgow": "Celtic",
    "glasgow rangers": "Rangers", "galatasaray istanbul": "Galatasaray",
    "fenerbahce istanbul": "Fenerbahce", "besiktas istanbul": "Besiktas",
    "bodo glimt": "Bodo/Glimt", "real betis": "Betis", "betis sevilla": "Betis",
    "slovan bratislava": "Slovan Bratislava", "rb leipzig": "RB Leipzig",
    "leipzig": "RB Leipzig", "stuttgart": "Stuttgart", "vfb stuttgart": "Stuttgart",
}


def norm(name: str) -> str:
    """lower-case, accent-free, punctuation-free, without 'FC'/'AC'-style tokens."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(t for t in s.split() if t not in STOP)


def canonicalize(df: pd.DataFrame, cols=("HomeTeam", "AwayTeam")) -> pd.DataFrame:
    """Give every club ONE display name even if the source spells it differently by season."""
    df = df.copy()
    counts: dict[str, Counter] = {}
    for col in cols:
        for raw in df[col]:
            counts.setdefault(norm(raw), Counter())[raw] += 1
    display = {k: c.most_common(1)[0][0] for k, c in counts.items()}
    for col in cols:
        df[col] = df[col].map(lambda r: display[norm(r)])
    return df
