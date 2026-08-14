import math
from altitude import altitude_boost
from elo_model import expected_score, HOME_ADVANTAGE

AVG_GOALS_PER_TEAM = 1.14   # average of home (1.33) and away (0.95) goals -- shared baseline now, since HOME_ADVANTAGE already carries the home-field effect through the tilt below
MAX_GOALS = 8               # cap when building the scoreline grid (10+ goal games are ~never)
DRAW_INFLATION = 1.10  # calibrated via backtesting -- model was underestimating draw rates

def poisson_pmf(k, lam):
    """Probability of scoring exactly k goals, given an expected-goals rate lam."""
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


def expected_goals(home_rating, away_rating, home_team=None, away_team=None, home_adjustment=0):
    """Converts an Elo rating gap (plus an altitude adjustment, if team
    names are given) into expected goals for each team. home_adjustment
    optionally overrides the shared HOME_ADVANTAGE with a per-team value."""
    effective_home_rating = home_rating
    if home_team and away_team:
        effective_home_rating += altitude_boost(home_team, away_team)

    expected_result = expected_score(effective_home_rating + HOME_ADVANTAGE + home_adjustment, away_rating)
    tilt = (expected_result - 0.5) * 2
    home_xg = AVG_GOALS_PER_TEAM * (1 + tilt * 0.6)
    away_xg = AVG_GOALS_PER_TEAM * (1 - tilt * 0.6)
    return max(home_xg, 0.1), max(away_xg, 0.1)

def dixon_coles_tau(home_goals, away_goals, lam, mu, rho):
    """The Dixon-Coles low-score correction factor. Only the four
    scorelines where the independent-Poisson assumption breaks down
    (0-0, 1-0, 0-1, 1-1) get adjusted; everything else is untouched."""
    if home_goals == 0 and away_goals == 0:
        return 1 - (lam * mu * rho)
    elif home_goals == 1 and away_goals == 0:
        return 1 + (lam * rho)
    elif home_goals == 0 and away_goals == 1:
        return 1 + (mu * rho)
    elif home_goals == 1 and away_goals == 1:
        return 1 - rho
    else:
        return 1.0


def scoreline_probabilities(home_rating, away_rating, home_team=None, away_team=None,
                             home_adjustment=0, draw_inflation=DRAW_INFLATION, dixon_coles_rho=None):
    """Returns {(home_goals, away_goals): probability} for every scoreline up
    to MAX_GOALS a side, with the same Dixon-Coles / draw-inflation
    calibration applied as match_probabilities -- the single source of truth
    both match_probabilities and the season simulator draw from, so the
    displayed win/draw/loss odds and the simulated scorelines never drift
    apart. Exactly one of draw_inflation or dixon_coles_rho should be active
    -- pass dixon_coles_rho to use the more precise correction instead of
    the blanket draw_inflation multiplier (pass draw_inflation=1.0 when
    using dixon_coles_rho, to avoid stacking both)."""
    home_xg, away_xg = expected_goals(home_rating, away_rating, home_team, away_team, home_adjustment)

    grid = {}
    for h in range(MAX_GOALS + 1):
        for a in range(MAX_GOALS + 1):
            p = poisson_pmf(h, home_xg) * poisson_pmf(a, away_xg)
            if dixon_coles_rho is not None:
                p *= dixon_coles_tau(h, a, home_xg, away_xg, dixon_coles_rho)
            grid[(h, a)] = p

    # Dixon-Coles doesn't guarantee the grid sums to exactly 1 -- renormalize to be safe
    if dixon_coles_rho is not None:
        total = sum(grid.values())
        if total > 0:
            grid = {k: v / total for k, v in grid.items()}

    if draw_inflation != 1.0:
        draw_mass = sum(v for (h, a), v in grid.items() if h == a)
        new_draw_mass = min(draw_mass * draw_inflation, 0.95)
        old_remaining = 1 - draw_mass
        new_remaining = 1 - new_draw_mass
        draw_scale = (new_draw_mass / draw_mass) if draw_mass > 0 else 1.0
        remaining_scale = (new_remaining / old_remaining) if old_remaining > 0 else 1.0
        grid = {(h, a): v * (draw_scale if h == a else remaining_scale) for (h, a), v in grid.items()}

    return grid


def match_probabilities(home_rating, away_rating, home_team=None, away_team=None,
                         home_adjustment=0, draw_inflation=DRAW_INFLATION, dixon_coles_rho=None):
    """Returns (P(home win), P(draw), P(away win)), summarized from scoreline_probabilities()."""
    grid = scoreline_probabilities(home_rating, away_rating, home_team, away_team,
                                    home_adjustment, draw_inflation, dixon_coles_rho)
    home_win = sum(v for (h, a), v in grid.items() if h > a)
    draw = sum(v for (h, a), v in grid.items() if h == a)
    away_win = sum(v for (h, a), v in grid.items() if h < a)
    return home_win, draw, away_win


if __name__ == "__main__":
    from elo_model import build_ratings
    from load_data import load_data

    played, _ = load_data()
    ratings = build_ratings(played)

    # quick manual test: pick two real teams from your ranking
    team_a, team_b = "Cali", "Millonarios"
    hw, d, aw = match_probabilities(ratings[team_a], ratings[team_b], team_a, team_b)
    print(f"{team_a} (home) vs {team_b} (away)")
    print(f"  {team_a} win: {hw:.1%}")
    print(f"  Draw:          {d:.1%}")
    print(f"  {team_b} win: {aw:.1%}")