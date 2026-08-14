import math
import pandas as pd
from load_data import load_data
from backtest import get_data_before_season, ALL_SEASONS
from elo_model import build_ratings
from predict_match import match_probabilities
from recent_form import compute_form_adjustments


def evaluate(history, test_matches, played_full, use_form):
    ratings = build_ratings(history)

    correct = 0
    log_loss_sum = 0.0
    total = 0

    test_matches = test_matches.sort_values("date")

    for _, m in test_matches.iterrows():
        home, away = m["home_team"], m["away_team"]
        if home not in ratings or away not in ratings:
            continue

        if use_form:
            # form as of THIS match's date, using all real results before it
            played_so_far = played_full[played_full["date"] < m["date"]]
            form = compute_form_adjustments(played_so_far, m["date"])
            home_form = form.get(home, 0)
            away_form = form.get(away, 0)
        else:
            home_form = away_form = 0

        hw, d, aw = match_probabilities(
            ratings[home] + home_form, ratings[away] + away_form, home, away
        )

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

    return correct / total * 100, log_loss_sum / total


played, _ = load_data()

results_no_form = []
results_with_form = []

for season in ALL_SEASONS:
    history = get_data_before_season(played, season)
    test_matches = played[played["season"] == season]
    if len(test_matches) == 0 or len(history) == 0:
        continue

    acc_no, ll_no = evaluate(history, test_matches, played, use_form=False)
    acc_yes, ll_yes = evaluate(history, test_matches, played, use_form=True)

    results_no_form.append((acc_no, ll_no))
    results_with_form.append((acc_yes, ll_yes))

avg_acc_no = sum(r[0] for r in results_no_form) / len(results_no_form)
avg_ll_no = sum(r[1] for r in results_no_form) / len(results_no_form)
avg_acc_yes = sum(r[0] for r in results_with_form) / len(results_with_form)
avg_ll_yes = sum(r[1] for r in results_with_form) / len(results_with_form)

print(f"{'Method':20s} {'Avg Accuracy':>14s} {'Avg Log Loss':>14s}")
print(f"{'Without form':20s} {avg_acc_no:>13.1f}% {avg_ll_no:>14.3f}")
print(f"{'With form':20s} {avg_acc_yes:>13.1f}% {avg_ll_yes:>14.3f}")