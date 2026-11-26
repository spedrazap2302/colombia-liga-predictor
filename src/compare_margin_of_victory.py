"""Does counting the winning margin make the Elo ratings better?

Walk-forward test, the same way the app is used: go through every match in
date order, predict it with the ratings as they stood before kickoff, then
update the ratings with the result. Tuned on 2015-2022, then checked on
2023 onward, which the choice never saw.
"""
import math

from load_data import load_data
import elo_model as E
from predict_match import match_probabilities
from team_home_advantage import compute_team_home_adjustments


def walk_forward(played, use_margin):
    ratings, season, adj, rows = {}, None, {}, []
    for m in played.itertuples():
        if m.season != season:
            if season is not None:
                for t in ratings:
                    ratings[t] += (E.INITIAL_RATING - ratings[t]) * E.SEASON_REGRESSION
            season = m.season
            history = played[played["date"] < m.date]
            adj = compute_team_home_adjustments(history) if len(history) else {}
        h, a = m.home_team, m.away_team
        rh, ra = ratings.setdefault(h, E.INITIAL_RATING), ratings.setdefault(a, E.INITIAL_RATING)
        if m.season != played["season"].iloc[0]:  # first season has no history to predict from
            hw, d, aw = match_probabilities(rh, ra, h, a, adj.get(h, 0))
            out = "home" if m.home_goals > m.away_goals else "away" if m.home_goals < m.away_goals else "draw"
            p = {"home": hw, "draw": d, "away": aw}
            rows.append((m.season, max(p, key=p.get) == out, -math.log(max(p[out], 1e-3))))
        res = 1.0 if m.home_goals > m.away_goals else 0.0 if m.home_goals < m.away_goals else 0.5
        exp = E.expected_score(rh + E.HOME_ADVANTAGE, ra)
        k = E.K_FACTOR * (E.margin_multiplier(m.home_goals - m.away_goals) if use_margin else 1.0)
        ratings[h] = rh + k * (res - exp)
        ratings[a] = ra - k * (res - exp)
    return rows


def summary(rows, test):
    part = [r for r in rows if (r[0] >= "2023") == test]
    return sum(r[1] for r in part) / len(part) * 100, sum(r[2] for r in part) / len(part)


if __name__ == "__main__":
    played, _ = load_data()
    print(f"{'Elo version':22s} {'2015-22 acc':>11s} {'log loss':>9s}   {'2023+ acc':>9s} {'log loss':>9s}")
    for name, use in [("result only", False), ("with winning margin", True)]:
        rows = walk_forward(played, use)
        (a1, l1), (a2, l2) = summary(rows, False), summary(rows, True)
        print(f"{name:22s} {a1:>10.1f}% {l1:>9.4f}   {a2:>8.1f}% {l2:>9.4f}")
