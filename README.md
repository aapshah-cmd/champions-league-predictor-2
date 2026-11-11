# Champions League Predictor

Predicts **UEFA Champions League** matches (win / draw / loss probabilities), simulates the
**entire 2026-27 tournament** (league phase, knockout playoffs, round of 16, quarter-finals,
semi-finals, final), and estimates every club's chance of reaching each round and **winning the
trophy**.

Models: logistic regression, random forest, gradient boosting, **XGBoost**, and a **Poisson goals
model** (needed to simulate scorelines, aggregates, extra time and penalties). Evaluated with log
loss, Brier score, Ranked Probability Score, accuracy and calibration, on seasons the models never
saw during training.

## Results

> Run `python -m src.train` and `python -m src.predict`, then paste your numbers and charts here.

| Model | Log loss | RPS | Accuracy |
|---|---|---|---|
| Baseline: class frequencies | | | |
| Baseline: ClubElo only | | | |
| Logistic regression | | | |
| Random forest | | | |
| XGBoost | | | |
| Poisson goals model | | | |

## Quick start

```bash
git clone https://github.com/<your-username>/<repo>.git
cd <repo>
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m src.data_loader     # download results + schedule (OpenLigaDB) and ratings (ClubElo)
python -m src.train           # train + evaluate on held-out seasons
python -m src.predict         # next matchday, full-tournament simulation, projected bracket
python -m pytest              # tests
```

No internet? `python -m src.train --demo` and `python -m src.predict --demo` run everything on
synthetic data (to check the code only, never to report results).

### First-run checklist (important)

1. **Team names.** Open `data/processed/clubelo_name_matches.csv`. Any club with `method = none`
   (or a wrong `fuzzy` match) should be fixed in `data/team_aliases.csv` (columns: `team,clubelo`),
   then re-run. Unmatched clubs just lose their ClubElo feature; nothing crashes.
2. **Competitions.** OpenLigaDB is community-edited and contains hobby leagues, other sports and duplicate copies of seasons. The loader keeps ONE competition per season (right match count for the format and real UEFA clubs) and prints which ones it used. Check that list.
3. **Stages.** `python -m src.data_loader` prints match counts per stage (league, playoff, r16, qf,
   sf, final). If a stage is missing or everything is `league`, adjust `stage_from_group()` in
   `src/data_loader.py` to the round names OpenLigaDB uses.
4. **Fixtures.** Predictions for upcoming matches and the simulation need the remaining schedule
   from OpenLigaDB. If none is found, the script says so.

## Troubleshooting

| Problem | Fix |
|---|---|
| `CERTIFICATE_VERIFY_FAILED` (Mac) | `pip install certifi`, then `export SSL_CERT_FILE=$(python -m certifi)` before running |
| `XGBoost ... libomp.dylib could not be loaded` (Mac) | `brew install libomp`. Until then the project runs without XGBoost |
| ClubElo times out / is down | Add `--no-clubelo` to any command, e.g. `python -m src.train --no-clubelo` |
| Wrong or duplicate competitions picked | See `data/processed/competitions_used.csv`; override a season with `FORCE_COMPETITIONS` in `src/config.py` |

## Data

| Source | Used for |
|---|---|
| [OpenLigaDB](https://www.openligadb.de) (free, no key) | Champions League results and schedule |
| [ClubElo](http://clubelo.com) (free CSV API) | Strength rating of every club, comparable across leagues |

Not available from these free sources: shots, possession, xG, bookmaker odds for the Champions
League. Features therefore use results, goals, Elo ratings and context (see roadmap).

## Features (all computed from information available before kickoff)

- Rolling form over each team's last 5 / 10 Champions League matches: goals for/against, points
  per game, goal difference, rest days, number of CL matches played (experience)
- A Champions League-only Elo rating (goal-difference weighted, home advantage, summer regression)
- **ClubElo** rating as of the day before the match (domestic + European strength)
- Context: knockout round?, second leg?; the final is treated as a neutral venue
- Home-minus-away difference features

**No leakage:** features are computed first, histories/Elo updated afterwards, and
`tests/test_features.py` proves a match's own result never changes its features.

## Validation

Train on 2016-17 to 2023-24 (XGBoost early stopping uses 2023-24); test on 2024-25 onward (the new
36-team format), never used for tuning. The Champions League has far fewer matches than a domestic
league (~190 per season), so **test metrics are noisy**: differences of ~0.01 in log loss between
models are not meaningful. The honest benchmark is the ClubElo-only baseline.

## Tournament simulation (`src/tournament.py`)

20,000 Monte Carlo runs. Each run: simulates the remaining league-phase games (Poisson goals),
ranks the 36 teams (points, goal difference, goals scored), then plays the whole knockout:

- Top 8 -> round of 16. 9th-24th -> two-legged playoffs (9/10 vs 23/24, 11/12 vs 21/22,
  13/14 vs 19/20, 15/16 vs 17/18). Winners meet seeds 7/8, 5/6, 3/4 and 1/2.
- Round of 16, quarter-finals, semi-finals: two legs, better league-phase finisher hosts leg 2;
  aggregate score, then extra time, then a 50/50 penalty shootout (no away-goals rule).
- Final: one neutral-venue match.

Outputs: per-team probabilities of reaching each round and winning
(`outputs/tournament_probabilities.csv`), plus a readable **projected bracket** where the favourite
advances in every tie (`outputs/projected_bracket.csv`).

**Simplifications:** UEFA's real draw has extra rules (e.g. same-country clubs can't meet early)
and is made after each round; we draw pairings randomly within each seeded pair. Team strength is
frozen at today's values. Goals are independent Poisson draws. The simulation always starts from
the league phase, so once the playoffs begin it does not yet lock in the real pairings.

## Project structure

```
src/
  config.py        seasons, split dates, format constants
  data_loader.py   OpenLigaDB download + cleaning
  clubelo.py       ClubElo ratings + team-name matching
  team_names.py    name normalisation and aliases
  features.py      leak-free rolling features + Elo
  models.py        logistic regression, random forest, gradient boosting, XGBoost, Poisson goals
  evaluate.py      metrics and plots
  train.py         time-based evaluation
  tournament.py    Monte Carlo tournament + projected bracket
  predict.py       next matches + tournament + figures
  synthetic.py     offline demo data
tests/             leakage, bracket-format and metric tests
```

## Roadmap

- [ ] Domestic-league shots/xG features (football-data.co.uk, FBref)
- [ ] Squad market value (Transfermarkt) and injuries
- [ ] Lock in real playoff/round-of-16 pairings once drawn
- [ ] Bootstrap confidence intervals on the metrics
- [ ] Calibration (isotonic) and walk-forward validation
- [ ] Streamlit app with a bracket diagram
