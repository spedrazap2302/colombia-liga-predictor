import math
from load_data import load_data
from backtest import get_data_before_season, ALL_SEASONS
from elo_model import build_ratings
from predict_match import match_probabilities, poisson_pmf, MAX_GOALS
from attack_defense_model import build_attack_defense_ratings


def match_probabilities_attack_defense(home, away, attack, defense, avg_home_goals, avg_away_goals):
    """Same Poisson scoreline grid as predict_match.py, but expected
    goals come from multiplying attack/defense ratings instead of an
    Elo rating gap."""
    home_xg = avg_home_goals * attack.get(home, 1.0) * defense.get(away, 1.0)
    away_xg = avg_away_goals * attack.get(away, 1.0) * defense.get(home, 1.0)

    home_win = draw = away_win = 0.0
    for h in range(MAX_GOALS + 1):
        for a in range(MAX_GOALS + 1):
            p = poisson_pmf(h, home_xg) * poisson_pmf(a, away_xg)
            if h > a:
                home_win += p
            elif h == a:
                draw += p
            else:
                away_win += p

    return home_win, draw, away_win


def evaluate_elo(history, test_matches):
    ratings = build_ratings(history)
    correct = 0
    log_loss_sum = 0.0
    total = 0

    for _, m in test_matches.iterrows():
        home, away = m["home_team"], m["away_team"]
        if home not in ratings or away not in ratings:
            continue

        hw, d, aw = match_probabilities(ratings[home], ratings[away], home, away)
        actual = "home" if m["home_goals"] > m["away_goals"] else "away" if m["home_goals"] < m["away_goals"] else "draw"
        predicted = max([("home", hw), ("draw", d), ("away", aw)], key=lambda x: x[1])[0]
        if predicted == actual:
            correct += 1
        actual_prob = {"home": hw, "draw": d, "away": aw}[actual]
        log_loss_sum += -1 * math.log(max(actual_prob, 0.001))
        total += 1

    return correct / total * 100, log_loss_sum / total


def evaluate_attack_defense(history, test_matches):
    attack, defense, avg_home, avg_away = build_attack_defense_ratings(history)
    correct = 0
    log_loss_sum = 0.0
    total = 0

    for _, m in test_matches.iterrows():
        home, away = m["home_team"], m["away_team"]
        if home not in attack or away not in attack:
            continue

        hw, d, aw = match_probabilities_attack_defense(home, away, attack, defense, avg_home, avg_away)
        actual = "home" if m["home_goals"] > m["away_goals"] else "away" if m["home_goals"] < m["away_goals"] else "draw"
        predicted = max([("home", hw), ("draw", d), ("away", aw)], key=lambda x: x[1])[0]
        if predicted == actual:
            correct += 1
        actual_prob = {"home": hw, "draw": d, "away": aw}[actual]
        log_loss_sum += -1 * math.log(max(actual_prob, 0.001))
        total += 1

    return correct / total * 100, log_loss_sum / total


played, _ = load_data()

results_elo = []
results_ad = []

for season in ALL_SEASONS:
    history = get_data_before_season(played, season)
    test_matches = played[played["season"] == season]
    if len(test_matches) == 0 or len(history) == 0:
        continue

    acc_elo, ll_elo = evaluate_elo(history, test_matches)
    acc_ad, ll_ad = evaluate_attack_defense(history, test_matches)

    results_elo.append((acc_elo, ll_elo))
    results_ad.append((acc_ad, ll_ad))

avg_acc_elo = sum(r[0] for r in results_elo) / len(results_elo)
avg_ll_elo = sum(r[1] for r in results_elo) / len(results_elo)
avg_acc_ad = sum(r[0] for r in results_ad) / len(results_ad)
avg_ll_ad = sum(r[1] for r in results_ad) / len(results_ad)

print(f"{'Method':25s} {'Avg Accuracy':>14s} {'Avg Log Loss':>14s}")
print(f"{'Elo + Poisson (current)':25s} {avg_acc_elo:>13.1f}% {avg_ll_elo:>14.3f}")
print(f"{'Attack/Defense':25s} {avg_acc_ad:>13.1f}% {avg_ll_ad:>14.3f}")