import math
from load_data import load_data
from backtest import get_data_before_season, ALL_SEASONS
from elo_model import build_ratings
from predict_match import match_probabilities

CANDIDATE_VALUES = [1.0, 1.05, 1.10, 1.15, 1.20, 1.25]

played, _ = load_data()

print(f"{'draw_inflation':>15s} {'Avg Accuracy':>14s} {'Avg Log Loss':>14s}")

for inflation in CANDIDATE_VALUES:
    accuracies = []
    log_losses = []

    for season in ALL_SEASONS:
        history = get_data_before_season(played, season)
        test_matches = played[played["season"] == season]
        if len(test_matches) == 0 or len(history) == 0:
            continue

        ratings = build_ratings(history)
        correct = 0
        log_loss_sum = 0.0
        total = 0

        for _, m in test_matches.iterrows():
            home, away = m["home_team"], m["away_team"]
            if home not in ratings or away not in ratings:
                continue

            hw, d, aw = match_probabilities(ratings[home], ratings[away], home, away, draw_inflation=inflation)

            if m["home_goals"] > m["away_goals"]:
                actual = "home"
            elif m["home_goals"] < m["away_goals"]:
                actual = "away"
            else:
                actual = "draw"

            predicted = max([("home", hw), ("draw", d), ("away", aw)], key=lambda x: x[1])[0]
            if predicted == actual:
                correct += 1

            actual_prob = {"home": hw, "draw": d, "away": aw}[actual]
            log_loss_sum += -1 * math.log(max(actual_prob, 0.001))
            total += 1

        accuracies.append(correct / total * 100)
        log_losses.append(log_loss_sum / total)

    avg_acc = sum(accuracies) / len(accuracies)
    avg_ll = sum(log_losses) / len(log_losses)
    print(f"{inflation:>15.2f} {avg_acc:>13.1f}% {avg_ll:>14.3f}")