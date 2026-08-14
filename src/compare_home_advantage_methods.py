import math
from load_data import load_data
from backtest import get_data_before_season, ALL_SEASONS
from elo_model import build_ratings
from predict_match import match_probabilities
from team_home_advantage import compute_team_home_adjustments


def evaluate(history, test_matches, use_per_team):
    ratings = build_ratings(history, use_per_team_home_adv=use_per_team)
    home_adjustments = compute_team_home_adjustments(history) if use_per_team else {}

    correct = 0
    log_loss_sum = 0.0
    total = 0

    for _, m in test_matches.iterrows():
        home, away = m["home_team"], m["away_team"]
        if home not in ratings or away not in ratings:
            continue

        adj = home_adjustments.get(home, 0) if use_per_team else 0
        hw, d, aw = match_probabilities(ratings[home], ratings[away], home, away, adj)

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

results_shared = []
results_per_team = []

for season in ALL_SEASONS:
    history = get_data_before_season(played, season)
    test_matches = played[played["season"] == season]
    if len(test_matches) == 0 or len(history) == 0:
        continue

    acc_shared, ll_shared = evaluate(history, test_matches, use_per_team=False)
    acc_team, ll_team = evaluate(history, test_matches, use_per_team=True)

    results_shared.append((acc_shared, ll_shared))
    results_per_team.append((acc_team, ll_team))

avg_acc_shared = sum(r[0] for r in results_shared) / len(results_shared)
avg_ll_shared = sum(r[1] for r in results_shared) / len(results_shared)
avg_acc_team = sum(r[0] for r in results_per_team) / len(results_per_team)
avg_ll_team = sum(r[1] for r in results_per_team) / len(results_per_team)

print(f"{'Method':25s} {'Avg Accuracy':>14s} {'Avg Log Loss':>14s}")
print(f"{'Shared HOME_ADVANTAGE':25s} {avg_acc_shared:>13.1f}% {avg_ll_shared:>14.3f}")
print(f"{'Per-team adjustment':25s} {avg_acc_team:>13.1f}% {avg_ll_team:>14.3f}")

sample_history = get_data_before_season(played, "2025-II")
sample_adjustments = compute_team_home_adjustments(sample_history)
print("\n--- Diagnostic: sample adjustments for 2025-II history ---")
for team in ["Nacional", "Magdalena", "Junior"]:
    print(f"{team}: {sample_adjustments.get(team, 'NOT FOUND')}")