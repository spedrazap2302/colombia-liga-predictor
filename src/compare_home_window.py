from load_data import load_data
from backtest import ALL_SEASONS, get_data_before_season, evaluate_match_predictions

played, _ = load_data()

configs = {
    "full_history": None,
    "capped_2021": "2021-I",
}

results = {name: {"total_ll": 0.0, "total_n": 0} for name in configs}

for season in ALL_SEASONS:
    history = get_data_before_season(played, season)
    test_matches = played[played["season"] == season]

    if len(test_matches) == 0 or len(history) == 0:
        continue

    for name, min_season in configs.items():
        _, avg_log_loss, n_checked = evaluate_match_predictions(history, test_matches, home_adv_min_season=min_season)
        results[name]["total_ll"] += avg_log_loss * n_checked
        results[name]["total_n"] += n_checked

print(f"{'Config':>15s} {'Avg LogLoss':>12s}")
for name, r in results.items():
    overall = r["total_ll"] / r["total_n"]
    print(f"{name:>15s} {overall:>12.4f}")
    