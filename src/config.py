"""Project-wide settings. Change these in one place."""
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
ALIAS_FILE = ROOT / "data" / "team_aliases.csv"   # optional: your own name fixes
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"

OPENLIGA_API = "https://api.openligadb.de"
CLUBELO_API = "http://api.clubelo.com"


def current_season_start(today: date | None = None) -> int:
    """The Champions League season starts in July/September. 2026-27 -> 2026."""
    today = today or date.today()
    return today.year if today.month >= 7 else today.year - 1


CURRENT_SEASON = current_season_start()

# Time-based split: train < VAL_START <= validation < TEST_START <= test
# 2024-25 is the first season of the new 36-team league-phase format.
VAL_START = "2023-07-01"     # validation = 2023-24 (used for early stopping)
TEST_START = "2024-07-01"    # test = 2024-25 onward (never touched while tuning)
WARMUP_SEASONS = 1           # the first season in the data only builds up form/Elo

# Label encoding: order matters for the Ranked Probability Score
CLASSES = ["H", "D", "A"]
LABEL_MAP = {"H": 0, "D": 1, "A": 2}
RANDOM_STATE = 42

# Format of the current Champions League
N_TEAMS = 36
N_DIRECT = 8           # top 8 -> round of 16
N_PLAYOFF = 24         # 9th-24th -> knockout playoffs
KNOCKOUT_STAGES = ["playoff", "r16", "qf", "sf", "final"]

# OpenLigaDB is community-edited and contains non-UEFA "Champions League" competitions and
# duplicates. If the automatic choice for a season is wrong, force the right shortcut here,
# e.g. {2024: "ucl24"}.
FORCE_COMPETITIONS: dict[int, str] = {}
FULL_SEASON_MATCHES_OLD = 125      # 32-team format: 96 group + 29 knockout matches
FULL_SEASON_MATCHES_NEW = 189      # 36-team format (2024-25 onward)
