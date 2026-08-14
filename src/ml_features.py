import pandas as pd
from elo_model import expected_score, INITIAL_RATING, K_FACTOR, HOME_ADVANTAGE, SEASON_REGRESSION
from altitude import altitude_boost


def build_training_dataset(matches: pd.DataFrame):
    """Walks through matches chronologically, exactly like build_ratings
    does -- but instead of just updating ratings, it also RECORDS each
    match's features using the ratings as they stood BEFORE that match
    was played. This guarantees no lookahead: every training row only
    ever sees information that was actually available at the time."""
    ratings = {}
    current_season = None
    rows = []

    def get_rating(team):
        return ratings.setdefault(team, INITIAL_RATING)

    for _, m in matches.iterrows():
        if m["season"] != current_season:
            if current_season is not None:
                for team in ratings:
                    ratings[team] += (INITIAL_RATING - ratings[team]) * SEASON_REGRESSION
            current_season = m["season"]

        home, away = m["home_team"], m["away_team"]
        home_rating = get_rating(home)
        away_rating = get_rating(away)
        alt = altitude_boost(home, away)

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
            "outcome": outcome,
        })

        # Update ratings the same way build_ratings does, so later
        # matches see the effect of this one -- same as real Elo
        expected_home = expected_score(home_rating + HOME_ADVANTAGE + alt, away_rating)
        ratings[home] = home_rating + K_FACTOR * (actual_home - expected_home)
        ratings[away] = away_rating + K_FACTOR * ((1 - actual_home) - (1 - expected_home))

    return pd.DataFrame(rows), ratings