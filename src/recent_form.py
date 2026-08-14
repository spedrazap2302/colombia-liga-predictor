import pandas as pd

FORM_WINDOW = 5  # how many recent matches count as "recent form"
FORM_WEIGHT = 15  # max Elo-equivalent points a team's form can add/subtract


def compute_form_adjustments(history: pd.DataFrame, as_of_date):
    """For every team, looks at their last FORM_WINDOW matches strictly
    before as_of_date, and returns an Elo-equivalent adjustment based on
    how those results compare to a neutral 50% win rate."""
    past_matches = history[history["date"] < as_of_date].sort_values("date")

    team_recent_results = {}

    for _, m in past_matches.iterrows():
        home, away = m["home_team"], m["away_team"]

        if m["home_goals"] > m["away_goals"]:
            home_result, away_result = 1.0, 0.0
        elif m["home_goals"] < m["away_goals"]:
            home_result, away_result = 0.0, 1.0
        else:
            home_result, away_result = 0.5, 0.5

        team_recent_results.setdefault(home, []).append(home_result)
        team_recent_results.setdefault(away, []).append(away_result)

    adjustments = {}
    for team, results in team_recent_results.items():
        recent = results[-FORM_WINDOW:]
        if len(recent) < FORM_WINDOW:
            adjustments[team] = 0  # not enough recent history to judge form yet
            continue
        avg_result = sum(recent) / len(recent)
        adjustments[team] = (avg_result - 0.5) * 2 * FORM_WEIGHT

    return adjustments

if __name__ == "__main__":
    from load_data import load_data
    import pandas as pd

    played, _ = load_data()
    today = pd.Timestamp.now()
    adjustments = compute_form_adjustments(played, today)

    ranked = sorted(adjustments.items(), key=lambda x: x[1], reverse=True)
    print(f"{'Team':20s} {'Form Adjustment':>16s}")
    for team, adj in ranked[:10]:
        print(f"{team:20s} {adj:>+15.1f}")
    print("...")
    for team, adj in ranked[-5:]:
        print(f"{team:20s} {adj:>+15.1f}")