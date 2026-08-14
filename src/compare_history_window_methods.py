import math
from load_data import load_data
from backtest import get_data_before_season, ALL_SEASONS
from elo_model import build_ratings
from predict_match import match_probabilities

# How many seasons of history to feed into build_ratings, counting
# backward from the test season. None = use everything available.
CANDIDATE_WINDOWS = [2, 4, 6, None]


def trim_to_window(history, n_seasons):
    """Keeps only the most recent n_seasons worth of tournaments from
    history (None = keep everything)."""
    if n_seasons is None:
        return history
    seasons_in_history = history["season"].unique().tolist()
    # seasons_in_history is already in chronological order since
    # get_data_before_season only includes matches before the test season
    recent_seasons = seasons_in_history[-n_seasons:]
    return history[history["season"].isin(recent_seasons)]


def evaluate(history, test_matches):
    ratings = build_ratings(history)

    correct = 0
    log_loss_sum = 0.0
    total = 0

    for _, m in test_matches.iterrows():
        home, away = m["home_team"], m["away_team"]
        if home not in ratings or away not in ratings:
            continue

        hw, d, aw = match_probabilities(ratings[home], ratings[away], home, away)

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

print(f"{'Seasons of history':>20s} {'Avg Accuracy':>14s} {'Avg Log Loss':>14s}")

for window in CANDIDATE_WINDOWS:
    accuracies = []
    log_losses = []

    for season in ALL_SEASONS:
        full_history = get_data_before_season(played, season)
        history = trim_to_window(full_history, window)
        test_matches = played[played["season"] == season]
        if len(test_matches) == 0 or len(history) == 0:
            continue

        acc, ll = evaluate(history, test_matches)
        accuracies.append(acc)
        log_losses.append(ll)

    avg_acc = sum(accuracies) / len(accuracies)
    avg_ll = sum(log_losses) / len(log_losses)
    label = "All available" if window is None else str(window)
    print(f"{label:>20s} {avg_acc:>13.1f}% {avg_ll:>14.3f}")