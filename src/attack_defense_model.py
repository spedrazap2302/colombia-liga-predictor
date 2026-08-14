import pandas as pd

INITIAL_RATING = 1.0          # average attack/defense strength
LEARNING_RATE = 0.10          # how much each match result moves ratings -- needs tuning
SEASON_REGRESSION = 0.25      # same idea as elo_model.py -- reset toward average between tournaments


def compute_league_averages(matches: pd.DataFrame):
    """The league-wide baseline goals-per-game, home and away separately --
    same numbers your Elo model already calibrated against."""
    avg_home_goals = matches["home_goals"].mean()
    avg_away_goals = matches["away_goals"].mean()
    return avg_home_goals, avg_away_goals


def build_attack_defense_ratings(matches: pd.DataFrame):
    """Every team gets TWO ratings instead of one:
    - attack[team]: >1.0 means they score more than a league-average team would
    - defense[team]: >1.0 means they CONCEDE more than average (i.e. a WEAKER
      defense); <1.0 means a stronger defense
    Both start at 1.0 (perfectly average) and update after every match,
    the same way Elo updates after every match -- just tracking goals
    instead of win/loss."""
    avg_home_goals, avg_away_goals = compute_league_averages(matches)

    attack = {}
    defense = {}
    current_season = None

    def ensure(team):
        attack.setdefault(team, INITIAL_RATING)
        defense.setdefault(team, INITIAL_RATING)

    for _, m in matches.iterrows():
        if m["season"] != current_season:
            if current_season is not None:
                for team in attack:
                    attack[team] += (INITIAL_RATING - attack[team]) * SEASON_REGRESSION
                    defense[team] += (INITIAL_RATING - defense[team]) * SEASON_REGRESSION
            current_season = m["season"]

        home, away = m["home_team"], m["away_team"]
        ensure(home)
        ensure(away)

        expected_home_goals = avg_home_goals * attack[home] * defense[away]
        expected_away_goals = avg_away_goals * attack[away] * defense[home]

        actual_home_goals = m["home_goals"]
        actual_away_goals = m["away_goals"]

        # Relative error: positive means "scored more than expected"
        home_error = (actual_home_goals - expected_home_goals) / max(expected_home_goals, 0.1)
        away_error = (actual_away_goals - expected_away_goals) / max(expected_away_goals, 0.1)

        attack[home] += LEARNING_RATE * home_error * attack[home]
        defense[away] += LEARNING_RATE * home_error * defense[away]  # conceded more than expected -> weaker defense

        attack[away] += LEARNING_RATE * away_error * attack[away]
        defense[home] += LEARNING_RATE * away_error * defense[home]

    return attack, defense, avg_home_goals, avg_away_goals


if __name__ == "__main__":
    from load_data import load_data

    played, _ = load_data()
    attack, defense, avg_home, avg_away = build_attack_defense_ratings(played)

    print(f"League baseline: {avg_home:.2f} home goals/game, {avg_away:.2f} away goals/game\n")
    print(f"{'Team':20s} {'Attack':>8s} {'Defense':>8s}")
    for team in sorted(attack.keys(), key=lambda t: attack[t], reverse=True):
        print(f"{team:20s} {attack[team]:>8.2f} {defense[team]:>8.2f}")