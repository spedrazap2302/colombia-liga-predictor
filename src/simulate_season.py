import random
import numpy as np
import pandas as pd

from load_data import load_data
from elo_model import build_ratings
from predict_match import expected_goals
from standings import compute_standings, CURRENT_SEASON

N_SIMULATIONS = 100000


def get_all_current_season_teams(played, upcoming, season):
    """Every team that's part of this season, whether they've played yet or not."""
    teams = set()
    for df in (played[played["season"] == season], upcoming[upcoming["season"] == season]):
        teams.update(df["home_team"].unique())
        teams.update(df["away_team"].unique())
    return teams


def simulate_score(home_team, away_team, ratings):
    """Rolls one random scoreline for a match, weighted by each team's
    Elo-based expected goals."""
    home_xg, away_xg = expected_goals(ratings[home_team], ratings[away_team], home_team, away_team)
    home_goals = np.random.poisson(home_xg)
    away_goals = np.random.poisson(away_xg)
    return home_goals, away_goals


def simulate_regular_season(played, upcoming, ratings, season):
    """Combines real results so far with simulated results for every
    remaining regular-season fixture, and returns a final ranked table."""
    points = {}
    goal_diff = {}

    def ensure(team):
        points.setdefault(team, 0)
        goal_diff.setdefault(team, 0)

    for team in get_all_current_season_teams(played, upcoming, season):
        ensure(team)

    # Real results already played
    real = played[played["season"] == season]
    for _, m in real.iterrows():
        home, away, hg, ag = m["home_team"], m["away_team"], m["home_goals"], m["away_goals"]
        goal_diff[home] += hg - ag
        goal_diff[away] += ag - hg
        if hg > ag:
            points[home] += 3
        elif hg < ag:
            points[away] += 3
        else:
            points[home] += 1
            points[away] += 1

    # Remaining fixtures, simulated
    remaining = upcoming[(upcoming["season"] == season) & (upcoming["phase_type"] == "regular")]
    for _, m in remaining.iterrows():
        home, away = m["home_team"], m["away_team"]
        hg, ag = simulate_score(home, away, ratings)
        goal_diff[home] += hg - ag
        goal_diff[away] += ag - hg
        if hg > ag:
            points[home] += 3
        elif hg < ag:
            points[away] += 3
        else:
            points[home] += 1
            points[away] += 1

    table = sorted(points.keys(), key=lambda t: (points[t], goal_diff[t]), reverse=True)
    return table  # just the ranked list of team names, 1st to last
def draw_cuadrangular_groups(top_8):
    """Applies the real Dimayor draw rule: 1st place leads Group A,
    2nd place leads Group B, and the other 6 teams (3rd-8th) are
    randomly shuffled into the remaining slots -- re-drawn fresh
    every simulation, since that's genuinely random each real season."""
    seed_a, seed_b = top_8[0], top_8[1]
    remaining = top_8[2:].copy()
    random.shuffle(remaining)

    group_a = [seed_a] + remaining[0:3]
    group_b = [seed_b] + remaining[3:6]
    return group_a, group_b


def simulate_group(group, ratings, seed_team):
    """Simulates a double round-robin (home and away) within a group
    of 4 teams. The seed_team gets the tiebreaker edge on level points,
    matching the real regulation."""
    points = {team: 0 for team in group}
    goal_diff = {team: 0 for team in group}

    for home in group:
        for away in group:
            if home == away:
                continue
            hg, ag = simulate_score(home, away, ratings)
            goal_diff[home] += hg - ag
            goal_diff[away] += ag - hg
            if hg > ag:
                points[home] += 3
            elif hg < ag:
                points[away] += 3
            else:
                points[home] += 1
                points[away] += 1

    def sort_key(team):
        # seed_team wins ties over everyone else; otherwise sort by
        # points then goal difference like a normal table
        is_seed = 1 if team == seed_team else 0
        return (points[team], is_seed, goal_diff[team])

    ranked = sorted(group, key=sort_key, reverse=True)
    return ranked[0]  # group winner


def simulate_final(team_a, team_b, ratings):
    """Two-legged final. Away goals don't count as a tiebreaker in
    this league's regulations, so on aggregate ties we go to penalties
    -- modeled as a coin flip, which is the standard assumption for
    penalty shootouts (they carry no reliable skill signal)."""
    hg1, ag1 = simulate_score(team_a, team_b, ratings)
    hg2, ag2 = simulate_score(team_b, team_a, ratings)

    agg_a = hg1 + ag2
    agg_b = ag1 + hg2

    if agg_a > agg_b:
        return team_a
    elif agg_b > agg_a:
        return team_b
    else:
        return random.choice([team_a, team_b])


def simulate_one_season(played, upcoming, ratings, season):
    """Runs one complete simulated season end-to-end: regular season
    -> top 8 -> cuadrangular groups -> final. Returns the champion."""
    final_table = simulate_regular_season(played, upcoming, ratings, season)
    top_8 = final_table[:8]

    group_a, group_b = draw_cuadrangular_groups(top_8)
    winner_a = simulate_group(group_a, ratings, seed_team=top_8[0])
    winner_b = simulate_group(group_b, ratings, seed_team=top_8[1])

    champion = simulate_final(winner_a, winner_b, ratings)
    return champion
def run_monte_carlo(played, upcoming, ratings, season, n_simulations=N_SIMULATIONS):
    """Runs the whole season simulation many times and counts how often
    each team wins the title."""
    np.random.seed(42)
    random.seed(42)
    champion_counts = {}

    for i in range(n_simulations):
        champion = simulate_one_season(played, upcoming, ratings, season)
        champion_counts[champion] = champion_counts.get(champion, 0) + 1

        if (i + 1) % 500 == 0:
            print(f"  ...{i + 1}/{n_simulations} simulations done")

    results = pd.DataFrame({
        "team": list(champion_counts.keys()),
        "title_probability_%": [round(count / n_simulations * 100, 1)
                                 for count in champion_counts.values()],
    })
    results = results.sort_values("title_probability_%", ascending=False).reset_index(drop=True)
    results.index += 1
    return results

def simulate_group_detailed(group, ratings, seed_team):
    """Same as simulate_group, but also returns the full group table
    (for display), not just the winner."""
    points = {team: 0 for team in group}
    goal_diff = {team: 0 for team in group}

    for home in group:
        for away in group:
            if home == away:
                continue
            hg, ag = simulate_score(home, away, ratings)
            goal_diff[home] += hg - ag
            goal_diff[away] += ag - hg
            if hg > ag:
                points[home] += 3
            elif hg < ag:
                points[away] += 3
            else:
                points[home] += 1
                points[away] += 1

    def sort_key(team):
        is_seed = 1 if team == seed_team else 0
        return (points[team], is_seed, goal_diff[team])

    ranked = sorted(group, key=sort_key, reverse=True)
    table = [{"team": t, "points": points[t], "goal_diff": goal_diff[t]} for t in ranked]
    return ranked[0], table


def simulate_final_detailed(team_a, team_b, ratings):
    """Same as simulate_final, but also returns the actual scorelines
    of both legs."""
    hg1, ag1 = simulate_score(team_a, team_b, ratings)
    hg2, ag2 = simulate_score(team_b, team_a, ratings)

    agg_a = hg1 + ag2
    agg_b = ag1 + hg2

    if agg_a > agg_b:
        champion = team_a
    elif agg_b > agg_a:
        champion = team_b
    else:
        champion = random.choice([team_a, team_b])

    legs = {
        "leg1": {"home": team_a, "away": team_b, "home_goals": hg1, "away_goals": ag1},
        "leg2": {"home": team_b, "away": team_a, "home_goals": hg2, "away_goals": ag2},
    }
    return champion, legs


def simulate_one_season_detailed(played, upcoming, ratings, season):
    """Runs one full simulated season and returns every stage of detail,
    for display purposes -- not used in the Monte Carlo loop, which only
    needs the champion and runs this thousands of times faster without detail."""
    np.random.seed(42)
    random.seed(42)
    final_table = simulate_regular_season(played, upcoming, ratings, season)
    top_8 = final_table[:8]

    group_a, group_b = draw_cuadrangular_groups(top_8)
    winner_a, table_a = simulate_group_detailed(group_a, ratings, seed_team=top_8[0])
    winner_b, table_b = simulate_group_detailed(group_b, ratings, seed_team=top_8[1])

    champion, final_legs = simulate_final_detailed(winner_a, winner_b, ratings)

    return {
        "final_table": final_table,
        "group_a_table": table_a,
        "group_b_table": table_b,
        "winner_a": winner_a,
        "winner_b": winner_b,
        "final_legs": final_legs,
        "champion": champion,
    }

if __name__ == "__main__":
    played, upcoming = load_data()
    ratings = build_ratings(played)

    print(f"Running {N_SIMULATIONS} simulations of the rest of the {CURRENT_SEASON} season...\n")
    probabilities = run_monte_carlo(played, upcoming, ratings, CURRENT_SEASON)
    print("\nChampionship probabilities:\n")
    print(probabilities.to_string())