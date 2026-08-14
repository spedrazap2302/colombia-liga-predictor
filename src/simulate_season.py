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
def _build_team_index(played, upcoming, ratings, season):
    """Fixed team <-> integer index mapping, plus a dense (T, T) lookup table
    of expected home/away goals for every ordered team pair. expected_goals()
    is a pure function of team identity (not match outcome), so precomputing
    it once and doing a numpy fancy-index lookup per simulated match avoids
    recomputing the same formula millions of times inside the Monte Carlo loop."""
    teams = sorted(get_all_current_season_teams(played, upcoming, season))
    team_to_idx = {team: i for i, team in enumerate(teams)}
    T = len(teams)

    xg_home = np.zeros((T, T))
    xg_away = np.zeros((T, T))
    for i, home in enumerate(teams):
        for j, away in enumerate(teams):
            if i == j:
                continue
            h, a = expected_goals(ratings[home], ratings[away], home, away)
            xg_home[i, j] = h
            xg_away[i, j] = a

    return teams, team_to_idx, xg_home, xg_away


def _base_points_and_goal_diff(played, team_to_idx, season):
    """Points/goal difference from already-played matches, computed once
    (not per simulation) -- a plain loop mirroring simulate_regular_season's
    real-match logic is fine here since it doesn't run inside the hot loop."""
    T = len(team_to_idx)
    points = np.zeros(T)
    goal_diff = np.zeros(T)

    real = played[played["season"] == season]
    for _, m in real.iterrows():
        hi, ai = team_to_idx[m["home_team"]], team_to_idx[m["away_team"]]
        hg, ag = m["home_goals"], m["away_goals"]
        goal_diff[hi] += hg - ag
        goal_diff[ai] += ag - hg
        if hg > ag:
            points[hi] += 3
        elif hg < ag:
            points[ai] += 3
        else:
            points[hi] += 1
            points[ai] += 1

    return points, goal_diff


def _row_argsort_desc(*arrays):
    """Row-wise argsort ranking each row's columns by multiple keys, all
    descending, earlier arrays taking priority on ties. Uses a structured
    dtype for an exact lexicographic comparison -- unlike combining keys
    into one weighted numeric score, this can't silently misorder rows if
    one key's range turns out wider than expected."""
    N, K = arrays[0].shape
    dtype = [(f"k{i}", "i8") for i in range(len(arrays))]
    key = np.empty((N, K), dtype=dtype)
    for i, arr in enumerate(arrays):
        key[f"k{i}"] = (-arr).astype(np.int64)
    return np.argsort(key, axis=1, order=[f"k{i}" for i in range(len(arrays))])


def run_monte_carlo(played, upcoming, ratings, season, n_simulations=N_SIMULATIONS):
    """Runs the whole season simulation many times and counts how often each
    team wins the title. Vectorized with numpy: every trial is simulated as
    one batched array operation instead of a Python loop calling
    simulate_one_season n_simulations times."""
    rng = np.random.default_rng(42)
    N = n_simulations

    teams, team_to_idx, xg_home, xg_away = _build_team_index(played, upcoming, ratings, season)
    T = len(teams)
    base_points, base_gd = _base_points_and_goal_diff(played, team_to_idx, season)

    # ---- Regular season: remaining fixtures, simulated for all N trials at once ----
    remaining = upcoming[(upcoming["season"] == season) & (upcoming["phase_type"] == "regular")]
    home_idx = remaining["home_team"].map(team_to_idx).to_numpy(dtype=np.int64)
    away_idx = remaining["away_team"].map(team_to_idx).to_numpy(dtype=np.int64)
    R = len(home_idx)

    home_xg = xg_home[home_idx, away_idx]
    away_xg = xg_away[home_idx, away_idx]
    # int16 keeps these (N, R) arrays small -- goal counts never come close to
    # its range, and this matters at N=100,000 on a memory-constrained host.
    home_goals = rng.poisson(home_xg[None, :], size=(N, R)).astype(np.int16)
    away_goals = rng.poisson(away_xg[None, :], size=(N, R)).astype(np.int16)

    pts_home = np.where(home_goals > away_goals, 3, np.where(home_goals == away_goals, 1, 0)).astype(np.float64)
    pts_away = np.where(away_goals > home_goals, 3, np.where(home_goals == away_goals, 1, 0)).astype(np.float64)
    gd_delta = (home_goals - away_goals).astype(np.float64)

    # One-hot fixture->team indicator matrices let a single matmul do what
    # would otherwise be a per-fixture scatter-add into each team's totals.
    onehot_home = np.zeros((R, T))
    onehot_home[np.arange(R), home_idx] = 1
    onehot_away = np.zeros((R, T))
    onehot_away[np.arange(R), away_idx] = 1

    points = base_points[None, :] + pts_home @ onehot_home + pts_away @ onehot_away
    goal_diff = base_gd[None, :] + gd_delta @ onehot_home - gd_delta @ onehot_away
    points = points.astype(np.int64)  # exact -- these are integer-valued floats, no rounding involved
    goal_diff = goal_diff.astype(np.int64)

    # ---- Top 8: rank all T teams per trial by (points desc, goal_diff desc, tiebreak) ----
    idx_bcast = np.broadcast_to(np.arange(T), (N, T))
    order = _row_argsort_desc(points, goal_diff, -idx_bcast)
    top8 = order[:, :8]

    # ---- Cuadrangular draw: seed 1 leads group A, seed 2 leads group B, the
    # other 6 teams are shuffled into the remaining slots, independently per trial ----
    seed_a, seed_b = top8[:, 0], top8[:, 1]
    remaining6 = top8[:, 2:8]
    perm = np.argsort(rng.random((N, 6)), axis=1)
    shuffled = np.take_along_axis(remaining6, perm, axis=1)
    group_a = np.concatenate([seed_a[:, None], shuffled[:, 0:3]], axis=1)  # (N, 4), slot 0 = seed
    group_b = np.concatenate([seed_b[:, None], shuffled[:, 3:6]], axis=1)

    def simulate_group_vectorized(group):
        """Double round-robin among the 4 teams in `group` (an (N, 4) array
        of team indices, slot 0 always the seed), for all N trials at once."""
        gp = np.zeros((N, 4))
        gd = np.zeros((N, 4))
        for home_slot in range(4):
            for away_slot in range(4):
                if home_slot == away_slot:
                    continue
                ht, at = group[:, home_slot], group[:, away_slot]
                hg = rng.poisson(xg_home[ht, at])
                ag = rng.poisson(xg_away[ht, at])
                ph = np.where(hg > ag, 3, np.where(hg == ag, 1, 0))
                pa = np.where(ag > hg, 3, np.where(hg == ag, 1, 0))
                gp[:, home_slot] += ph
                gp[:, away_slot] += pa
                gd[:, home_slot] += hg - ag
                gd[:, away_slot] += ag - hg

        is_seed = np.zeros((N, 4))
        is_seed[:, 0] = 1  # tiebreak order: points -> is_seed -> goal_diff, matching simulate_group
        winner_slot = _row_argsort_desc(gp, is_seed, gd)[:, 0]
        return np.take_along_axis(group, winner_slot[:, None], axis=1)[:, 0]

    winner_a = simulate_group_vectorized(group_a)
    winner_b = simulate_group_vectorized(group_b)

    # ---- Final: two legs, aggregate score, coin flip on an exact tie ----
    hg1 = rng.poisson(xg_home[winner_a, winner_b])
    ag1 = rng.poisson(xg_away[winner_a, winner_b])
    hg2 = rng.poisson(xg_home[winner_b, winner_a])
    ag2 = rng.poisson(xg_away[winner_b, winner_a])
    agg_a, agg_b = hg1 + ag2, ag1 + hg2
    coin = rng.random(N) < 0.5
    champion = np.where(
        agg_a > agg_b, winner_a,
        np.where(agg_b > agg_a, winner_b, np.where(coin, winner_a, winner_b)),
    )

    counts = np.bincount(champion, minlength=T)
    results = pd.DataFrame({
        "team": teams,
        "title_probability_%": np.round(counts / n_simulations * 100, 1),
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
    import time

    played, upcoming = load_data()
    ratings = build_ratings(played)

    print(f"Running {N_SIMULATIONS} simulations of the rest of the {CURRENT_SEASON} season...\n")
    start = time.time()
    probabilities = run_monte_carlo(played, upcoming, ratings, CURRENT_SEASON)
    elapsed = time.time() - start
    print(f"\nChampionship probabilities ({elapsed:.2f}s):\n")
    print(probabilities.to_string())