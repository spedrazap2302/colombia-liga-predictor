import pandas as pd
from load_data import load_data

GLOBAL_HOME_ADVANTAGE = 100  # the current shared value, used as the baseline every team is measured against
SHRINKAGE_K = 15  # higher = more caution for teams with few home matches


def compute_team_home_adjustments(played: pd.DataFrame, min_season: str = None):
    if min_season is not None:
        played = played[played["season"] >= min_season]
    """For each team, returns how many EXTRA Elo points (positive or
    negative) should be added on top of the global home advantage,
    based on how that team specifically performs at home versus the
    league-wide home win rate. Shrunk toward 0 for smaller samples."""
    league_home_win_rate = (played["home_goals"] > played["away_goals"]).mean()

    adjustments = {}
    for team, group in played.groupby("home_team"):
        n = len(group)
        team_home_win_rate = (group["home_goals"] > group["away_goals"]).mean()
        diff = team_home_win_rate - league_home_win_rate
        raw_adjustment = diff * 400  # rough win-rate-to-Elo-points conversion
        shrink_factor = n / (n + SHRINKAGE_K)
        adjustments[team] = raw_adjustment * shrink_factor

    return adjustments


if __name__ == "__main__":
    played, _ = load_data()
    adjustments = compute_team_home_adjustments(played)

    ranked = sorted(adjustments.items(), key=lambda x: x[1], reverse=True)
    print(f"{'Team':20s} {'Adjustment':>12s} {'Effective Home Adv':>20s}")
    for team, adj in ranked:
        print(f"{team:20s} {adj:>+11.1f} {GLOBAL_HOME_ADVANTAGE + adj:>19.1f}")