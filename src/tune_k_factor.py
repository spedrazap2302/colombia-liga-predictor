import elo_model
from load_data import load_data
from backtest import ALL_SEASONS, get_data_before_season, evaluate_match_predictions

played, _ = load_data()

candidate_values = [10, 15, 20, 25, 30, 35, 40]
current_k_factor = elo_model.K_FACTOR

results = []

for k in candidate_values:
    elo_model.K_FACTOR = k

    total_log_loss = 0.0
    total_matches = 0

    for season in ALL_SEASONS:
        history = get_data_before_season(played, season)
        test_matches = played[played["season"] == season]

        if len(test_matches) == 0 or len(history) == 0:
            continue

        _, avg_log_loss, n_checked = evaluate_match_predictions(history, test_matches)
        total_log_loss += avg_log_loss * n_checked
        total_matches += n_checked

    overall_avg = total_log_loss / total_matches
    results.append((k, overall_avg))

elo_model.K_FACTOR = current_k_factor  # restore original before printing

print(f"{'K_FACTOR':>10s} {'Avg LogLoss':>12s}")
for k, ll in results:
    marker = "  <- current" if k == current_k_factor else ""
    print(f"{k:>10d} {ll:>12.4f}{marker}")

best_k, best_ll = min(results, key=lambda x: x[1])
print(f"\nBest: K_FACTOR={best_k} (log loss {best_ll:.4f})")