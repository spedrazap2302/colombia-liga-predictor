import math
from load_data import load_data
from backtest import get_data_before_season, ALL_SEASONS
from elo_model import build_ratings
from predict_match import match_probabilities
from attack_defense_model import build_attack_defense_ratings
from compare_attack_defense_methods import match_probabilities_attack_defense

# How much weight to give Elo+Poisson vs Attack/Defense.
# 1.0 = pure Elo (today's baseline), 0.0 = pure Attack/Defense
ELO_WEIGHT_CANDIDATES = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]


def evaluate_ensemble(history, test_matches, elo_weight):
    ratings = build_ratings(history)
    attack, defense, avg_home, avg_away = build_attack_defense_ratings(history)

    correct = 0
    log_loss_sum = 0.0
    total = 0

    for _, m in test_matches.iterrows():
        home, away = m["home_team"], m["away_team"]
        if home not in ratings or away not in ratings:
            continue

        hw1, d1, aw1 = match_probabilities(ratings[home], ratings[away], home, away)
        hw2, d2, aw2 = match_probabilities_attack_defense(home, away, attack, defense, avg_home, avg_away)

        hw = elo_weight * hw1 + (1 - elo_weight) * hw2
        d = elo_weight * d1 + (1 - elo_weight) * d2
        aw = elo_weight * aw1 + (1 - elo_weight) * aw2

        actual = "home" if m["home_goals"] > m["away_goals"] else "away" if m["home_goals"] < m["away_goals"] else "draw"
        predicted = max([("home", hw), ("draw", d), ("away", aw)], key=lambda x: x[1])[0]
        if predicted == actual:
            correct += 1
        actual_prob = {"home": hw, "draw": d, "away": aw}[actual]
        log_loss_sum += -1 * math.log(max(actual_prob, 0.001))
        total += 1

    return correct / total * 100, log_loss_sum / total


played, _ = load_data()

print(f"{'Elo Weight':>12s} {'Avg Accuracy':>14s} {'Avg Log Loss':>14s}")

for weight in ELO_WEIGHT_CANDIDATES:
    accs, lls = [], []
    for season in ALL_SEASONS:
        history = get_data_before_season(played, season)
        test_matches = played[played["season"] == season]
        if len(test_matches) == 0 or len(history) == 0:
            continue
        acc, ll = evaluate_ensemble(history, test_matches, weight)
        accs.append(acc)
        lls.append(ll)
    avg_acc = sum(accs) / len(accs)
    avg_ll = sum(lls) / len(lls)
    print(f"{weight:>12.1f} {avg_acc:>13.1f}% {avg_ll:>14.3f}")