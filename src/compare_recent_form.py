from load_data import load_data
from elo_model import build_ratings
from predict_match import match_probabilities
from team_home_advantage import compute_team_home_adjustments
from recent_form import compute_form_adjustments
from backtest import ALL_SEASONS, get_data_before_season, evaluate_match_predictions
import math

played, _ = load_data()

baseline_ll_total = 0.0
form_ll_total = 0.0
total_n = 0

for season in ALL_SEASONS:
    history = get_data_before_season(played, season)
    test_matches = played[played["season"] == season]

    if len(test_matches) == 0 or len(history) == 0:
        continue

    print(f"Processing {season}: history={len(history)} rows, test={len(test_matches)} matches...")

    # Baseline (no form) -- reuse the existing backtest function as-is
    _, baseline_avg_ll, n_checked = evaluate_match_predictions(history, test_matches)
    baseline_ll_total += baseline_avg_ll * n_checked
    total_n += n_checked

    # With form adjustment
    ratings = build_ratings(history)
    home_adjustments = compute_team_home_adjustments(history)

    match_count = 0
    for _, m in test_matches.iterrows():
        home, away = m["home_team"], m["away_team"]
        if home not in ratings or away not in ratings:
            continue

        form_adj = compute_form_adjustments(history, as_of_date=m["date"])
        home_rating = ratings[home] + form_adj.get(home, 0)
        away_rating = ratings[away] + form_adj.get(away, 0)

        hw, d, aw = match_probabilities(home_rating, away_rating, home, away, home_adjustments.get(home, 0))

        if m["home_goals"] > m["away_goals"]:
            actual_outcome = "home"
        elif m["home_goals"] < m["away_goals"]:
            actual_outcome = "away"
        else:
            actual_outcome = "draw"

        actual_prob = {"home": hw, "draw": d, "away": aw}[actual_outcome]
        form_ll_total += -1 * math.log(max(actual_prob, 0.001))

        match_count += 1
        if match_count % 50 == 0:
            print(f"  ...{match_count}/{len(test_matches)} matches done")

    print(f"  {season} complete.")

print(f"\n{'Config':>15s} {'Avg LogLoss':>12s}")
print(f"{'baseline':>15s} {baseline_ll_total / total_n:>12.4f}")
print(f"{'with_form':>15s} {form_ll_total / total_n:>12.4f}")