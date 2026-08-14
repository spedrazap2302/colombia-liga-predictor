import pandas as pd
from load_data import load_data
from team_home_advantage import compute_team_home_adjustments

INITIAL_RATING = 1500
K_FACTOR = 20             # how much each match result moves a team's rating
HOME_ADVANTAGE = 100      # rating-point bonus for playing at home
SEASON_REGRESSION = 0.25  # ratings drift back toward average between tournaments


def expected_score(rating_a, rating_b):
    """Standard Elo formula: A's expected result against B (0 to 1)."""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def build_ratings(matches: pd.DataFrame, use_per_team_home_adv=False):
    ratings = {}
    history = []
    current_season = None
    home_adjustments = compute_team_home_adjustments(matches) if use_per_team_home_adv else {}

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

        if m["home_goals"] > m["away_goals"]:
            actual_home = 1.0
        elif m["home_goals"] < m["away_goals"]:
            actual_home = 0.0
        else:
            actual_home = 0.5

        team_home_advantage = HOME_ADVANTAGE + home_adjustments.get(home, 0)
        expected_home = expected_score(home_rating + team_home_advantage, away_rating)

        ratings[home] = home_rating + K_FACTOR * (actual_home - expected_home)
        ratings[away] = away_rating + K_FACTOR * ((1 - actual_home) - (1 - expected_home))

    return ratings


if __name__ == "__main__":
    played, _ = load_data()
    ratings = build_ratings(played)

    ranked = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
    print("Current Elo ratings:\n")
    for team, rating in ranked:
        print(f"{team:20s} {rating:.1f}")