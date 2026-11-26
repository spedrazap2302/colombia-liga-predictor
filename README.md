# Colombian Primera A Championship Predictor

Who will win the Liga BetPlay (Colombia's top football league), and how sure can we be? This Streamlit app rates every team with an Elo system, turns ratings into match probabilities with a Poisson goals model, and simulates the rest of the season 100,000 times, including the cuadrangular group stage and the two-legged final, to get each team's title odds.

**Live app:** _coming soon_

## What's in the app

- **Championship odds:** each team's chance of winning the title, from 100,000 simulated seasons.
- **Upcoming matches:** win, draw and loss chances for every remaining game.
- **Elo ratings:** the current strength of all 20 teams.
- **Simulate one season:** one random, complete version of how the season could end.
- **Standings:** the real regular-season table.

## Data

- **2015 to Aug 2026:** every Primera A match (about 4,750), collected by hand from Flashscore into one cleaned file, `data/matches.csv`. I cross-checked it against an external season-totals reference and fixed mislabeled home/away teams, missing matches and duplicates.
- **Since Aug 2026: updated automatically.** `src/update_results.py` reads the same public JSON data that ESPN's Liga BetPlay scoreboard page loads, converts kickoff times to Colombian time, maps ESPN's team names to ours, and fills new results into the CSV. A GitHub Action runs it every morning and commits any new results, and the app refreshes itself.

## How the model works

1. **Elo ratings.** Every match moves the two teams' ratings toward the result, more for a bigger winning margin. Ratings drift a quarter of the way back to average between tournaments. Home teams get a bonus, plus a little more when their city sits higher than the visitors' (Bogotá, Tunja and Pasto are above 2,500 m).
2. **Poisson goals model.** The rating gap gives each side's expected goals, and a Poisson distribution turns those into the chance of every scoreline. Draw chances are scaled up 10%, because backtesting showed the model underestimated draws.
3. **Monte Carlo season.** The remaining regular season, the cuadrangular groups (1st and 2nd seeded, the other six drawn at random, seeds winning ties on points) and the final are simulated 100,000 times. Once the real groups are set, the simulation uses them and the results already played.

## Does it work?

I tested it walk-forward, the way the app is used: go through every match since 2015 in date order, predict it from the ratings as they stood before kickoff, then update.

| | Picked the right result | Log loss (lower is better) |
|---|---|---|
| **Model** | **47.4%** | **1.030** |
| Naive baseline (home 45%, draw 28%, away 27%) | 46.6% | 1.061 |

- The model beats the baseline on log loss in **21 of 22 seasons** (2015-II to 2026-II). Log loss scores the probabilities themselves, which is what the title odds are built from.
- Picking the single most likely result is hard in this league: about 30% of matches are draws, and a draw is almost never the most likely single outcome.
- `src/backtest.py` gives a stricter version that freezes ratings at the start of each season: 46.9% vs 46.4%, log loss 1.036 vs 1.061.

### What I tested and rejected

Each idea was tested with the same backtest. Only these made the model better:

- **Kept:** draw calibration (the model underestimated draws), winning margin in the Elo update (better on both 2015–2022 and the held-out 2023–2026 seasons; see `src/compare_margin_of_victory.py`), and two corrected altitudes (Alianza's Valledupar was listed at 1,720 m instead of about 170 m, and Águilas Doradas' Rionegro was missing).
- **Rejected:** per-team home advantage, recent form, coaching changes, shorter history windows, a Dixon-Coles correction, separate attack and defense ratings, ensembles, and Random Forests (alone and with all the extra features).

A simple, well-tested model beat every fancier version I tried.

## Files

| File | What it does |
|---|---|
| `app.py` | The Streamlit app |
| `src/update_results.py` | Gets new results from ESPN and adds them to `data/matches.csv` |
| `.github/workflows/update-results.yml` | Runs the updater every morning |
| `src/load_data.py` | Loads the matches and splits played games from fixtures |
| `src/elo_model.py` | Elo ratings |
| `src/predict_match.py` | Expected goals, scoreline grid and win/draw/loss chances |
| `src/simulate_season.py` | Monte Carlo season, cuadrangulares and final |
| `src/standings.py` | Regular-season table |
| `src/altitude.py`, `src/team_home_advantage.py` | Home-field adjustments |
| `src/backtest.py`, `src/compare_*.py`, `src/tune_*.py` | The tests behind every modeling decision |

## How to run it

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

To get the latest results by hand: `python src/update_results.py` (add `--dry-run` to preview).

## Limits

- ESPN's scoreboard data is public but unofficial, so its format could change. If it does, the updater stops with an error and the app keeps the last good data.
- A newly promoted team stops the updater until its name is added to `TEAM_MAP` in `src/update_results.py`, so nothing is saved under the wrong name.
- Elo only knows results. It can't see injuries, transfers or a new coach until the results show it.

## AI acknowledgment

I used Claude (Anthropic) as a coding assistant: to plan, debug and test ideas, and later to build the automatic updater and polish the project for my portfolio.

## Author

Sebastián Pedraza · [GitHub](https://github.com/spedrazap2302) · [LinkedIn](https://www.linkedin.com/in/sebastianpedraza2302)
