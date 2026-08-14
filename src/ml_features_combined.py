import pandas as pd
from elo_model import expected_score, INITIAL_RATING, K_FACTOR, HOME_ADVANTAGE, SEASON_REGRESSION
from altitude import altitude_boost
from coaches import load_coaches, get_coach_tenure_days

FORM_WINDOW = 5
coaches_df = load_coaches()


def build_combined_training_dataset(matches: pd.DataFrame):
    """Same point-in-time discipline as ml_features.py, but now also
    tracks recent form and a running per-team home-record signal
    incrementally as it walks through history -- and looks up coach
    tenure from coaches.csv (which is safe to do directly, since it's
    calendar-based and never depends on match RESULTS)."""
    ratings = {}
    current_season = None
    rows = []

    recent_results = {}   # team -> list of 1/0.5/0 results, most recent last
    home_results = {}     # team -> list of 1/0.5/0 results, HOME matches only

    def ensure(team):
        ratings.setdefault(team, INITIAL_RATING)
        recent_results.setdefault(team, [])
        home_results.setdefault(team, [])

    for _, m in matches.iterrows():
        if m["season"] != current_season:
            if current_season is not None:
                for team in ratings:
                    ratings[team] += (INITIAL_RATING - ratings[team]) * SEASON_REGRESSION
            current_season = m["season"]

        home, away = m["home_team"], m["away_team"]
        ensure(home)
        ensure(away)

        home_rating = ratings[home]
        away_rating = ratings[away]
        alt = altitude_boost(home, away)

        # Recent form: average of last FORM_WINDOW results, or 0.5 (neutral) if not enough history
        home_form = sum(recent_results[home][-FORM_WINDOW:]) / len(recent_results[home][-FORM_WINDOW:]) if len(recent_results[home]) >= FORM_WINDOW else 0.5
        away_form = sum(recent_results[away][-FORM_WINDOW:]) / len(recent_results[away][-FORM_WINDOW:]) if len(recent_results[away]) >= FORM_WINDOW else 0.5

        # Per-team home record so far this walk (proxy for "this team's own home advantage")
        home_record = sum(home_results[home]) / len(home_results[home]) if len(home_results[home]) >= 5 else 0.5

        # Coach tenure -- calendar-based, safe to look up directly
        home_tenure = get_coach_tenure_days(home, m["date"], coaches_df)
        away_tenure = get_coach_tenure_days(away, m["date"], coaches_df)
        home_tenure = home_tenure if home_tenure is not None else -1  # -1 flags "unknown" for the model
        away_tenure = away_tenure if away_tenure is not None else -1

        if m["home_goals"] > m["away_goals"]:
            outcome = "home"
            actual_home = 1.0
        elif m["home_goals"] < m["away_goals"]:
            outcome = "away"
            actual_home = 0.0
        else:
            outcome = "draw"
            actual_home = 0.5

        rows.append({
            "elo_diff": home_rating - away_rating,
            "effective_diff": (home_rating + HOME_ADVANTAGE + alt) - away_rating,
            "altitude_effect": alt,
            "home_form": home_form,
            "away_form": away_form,
            "home_record": home_record,
            "home_coach_tenure": home_tenure,
            "away_coach_tenure": away_tenure,
            "outcome": outcome,
        })

        # Update ratings and tracking, same order as always
        expected_home = expected_score(home_rating + HOME_ADVANTAGE + alt, away_rating)
        ratings[home] = home_rating + K_FACTOR * (actual_home - expected_home)
        ratings[away] = away_rating + K_FACTOR * ((1 - actual_home) - (1 - expected_home))

        recent_results[home].append(actual_home)
        recent_results[away].append(1 - actual_home)
        home_results[home].append(actual_home)

    return pd.DataFrame(rows), ratings, recent_results, home_results