import math
from load_data import load_data
from backtest import get_data_before_season, ALL_SEASONS
from elo_model import build_ratings
from predict_match import match_probabilities
from coaches import load_coaches, get_coach_tenure_days

NEW_COACH_THRESHOLD_DAYS = 60  # under this many days in charge counts as "new"

coaches_df = load_coaches()


def evaluate(history, test_matches, adjustment):
    ratings = build_ratings(history)

    correct = 0
    log_loss_sum = 0.0
    total = 0

    for _, m in test_matches.iterrows():
        home, away = m["home_team"], m["away_team"]
        if home not in ratings or away not in ratings:
            continue

        home_tenure = get_coach_tenure_days(home, m["date"], coaches_df)
        away_tenure = get_coach_tenure_days(away, m["date"], coaches_df)

        home_adj = adjustment if (home_tenure is not None and home_tenure < NEW_COACH_THRESHOLD_DAYS) else 0
        away_penalty = adjustment if (away_tenure is not None and away_tenure < NEW_COACH_THRESHOLD_DAYS) else 0

        effective_home_rating = ratings[home] + home_adj
        effective_away_rating = ratings[away] + away_penalty

        hw, d, aw = match_probabilities(effective_home_rating, effective_away_rating, home, away)

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

CANDIDATE_ADJUSTMENTS = [-30, -15, 0, 15, 30]

print(f"{'Adjustment':>12s} {'Avg Accuracy':>14s} {'Avg Log Loss':>14s}")

for adjustment in CANDIDATE_ADJUSTMENTS:
    accuracies = []
    log_losses = []

    for season in ALL_SEASONS:
        history = get_data_before_season(played, season)
        test_matches = played[played["season"] == season]
        if len(test_matches) == 0 or len(history) == 0:
            continue

        acc, ll = evaluate(history, test_matches, adjustment)
        accuracies.append(acc)
        log_losses.append(ll)

    avg_acc = sum(accuracies) / len(accuracies)
    avg_ll = sum(log_losses) / len(log_losses)
    print(f"{adjustment:>12d} {avg_acc:>13.1f}% {avg_ll:>14.3f}")