import math
import pandas as pd
from load_data import load_data
from elo_model import build_ratings
from predict_match import match_probabilities
from team_home_advantage import compute_team_home_adjustments

ALL_SEASONS = ["2015-I", "2015-II", "2016-I", "2016-II", "2017-I", "2017-II", "2018-I", "2018-II",
                "2019-I", "2019-II", "2020",
                "2021-I", "2021-II", "2022-I", "2022-II", "2023-I", "2023-II", "2024-I", "2024-II", "2025-I", "2025-II"]
# Note: 2015-I is skipped as a test target since there's no earlier
# history to build ratings from -- it's still used as history for later seasons.


def get_data_before_season(played, test_season):
    """Returns only the matches that happened strictly before the
    test season started -- simulating 'what the model would have
    known at the time.'"""
    season_start_date = played[played["season"] == test_season]["date"].min()
    return played[played["date"] < season_start_date]


def evaluate_match_predictions(history, test_matches, home_adv_min_season=None):
    """Builds Elo ratings using only the history, then checks how often
    the model's favorite actually won each match in the test season."""
    ratings = build_ratings(history)
    home_adjustments = compute_team_home_adjustments(history, min_season=home_adv_min_season)

    correct_favorite = 0
    total_checked = 0
    log_loss_sum = 0.0

    for _, m in test_matches.iterrows():
        home, away = m["home_team"], m["away_team"]
        if home not in ratings or away not in ratings:
            continue

        hw, d, aw = match_probabilities(ratings[home], ratings[away], home, away, home_adjustments.get(home, 0))

        if m["home_goals"] > m["away_goals"]:
            actual_outcome = "home"
        elif m["home_goals"] < m["away_goals"]:
            actual_outcome = "away"
        else:
            actual_outcome = "draw"

        predicted_favorite = max([("home", hw), ("draw", d), ("away", aw)], key=lambda x: x[1])[0]
        if predicted_favorite == actual_outcome:
            correct_favorite += 1

        actual_prob = {"home": hw, "draw": d, "away": aw}[actual_outcome]
        log_loss_sum += -1 * math.log(max(actual_prob, 0.001))

        total_checked += 1

    accuracy = correct_favorite / total_checked * 100
    avg_log_loss = log_loss_sum / total_checked
    return accuracy, avg_log_loss, total_checked


def naive_baseline(test_matches):
    """What accuracy/log-loss would we get by always picking the home
    team, with no model at all? This is the bar our model needs to clear."""
    correct = 0
    log_loss_sum = 0.0
    total = 0

    for _, m in test_matches.iterrows():
        if m["home_goals"] > m["away_goals"]:
            actual_outcome = "home"
        elif m["home_goals"] < m["away_goals"]:
            actual_outcome = "away"
        else:
            actual_outcome = "draw"

        if actual_outcome == "home":
            correct += 1

        naive_probs = {"home": 0.45, "draw": 0.28, "away": 0.27}
        log_loss_sum += -1 * math.log(naive_probs[actual_outcome])
        total += 1

    return correct / total * 100, log_loss_sum / total


def check_favorite_distribution(history, test_matches):
    """Counts how often the model's predicted favorite is home/draw/away,
    versus how often each actually happens -- to catch a model that's
    defaulting to 'always pick home' regardless of team strength."""
    ratings = build_ratings(history)
    predicted_counts = {"home": 0, "draw": 0, "away": 0}
    actual_counts = {"home": 0, "draw": 0, "away": 0}

    for _, m in test_matches.iterrows():
        home, away = m["home_team"], m["away_team"]
        if home not in ratings or away not in ratings:
            continue

        hw, d, aw = match_probabilities(ratings[home], ratings[away], home, away)
        predicted_favorite = max([("home", hw), ("draw", d), ("away", aw)], key=lambda x: x[1])[0]
        predicted_counts[predicted_favorite] += 1

        if m["home_goals"] > m["away_goals"]:
            actual_counts["home"] += 1
        elif m["home_goals"] < m["away_goals"]:
            actual_counts["away"] += 1
        else:
            actual_counts["draw"] += 1

    return predicted_counts, actual_counts


if __name__ == "__main__":
    played, _ = load_data()

    all_results = []

    for season in ALL_SEASONS:
        history = get_data_before_season(played, season)
        test_matches = played[played["season"] == season]

        if len(test_matches) == 0 or len(history) == 0:
            continue

        accuracy, avg_log_loss, total_checked = evaluate_match_predictions(history, test_matches)
        naive_accuracy, naive_log_loss = naive_baseline(test_matches)

        all_results.append({
            "season": season,
            "matches": total_checked,
            "our_accuracy": accuracy,
            "naive_accuracy": naive_accuracy,
            "our_log_loss": avg_log_loss,
            "naive_log_loss": naive_log_loss,
        })

    print("\n--- Favorite distribution check (using full history, all seasons combined) ---")
    testable_seasons = ALL_SEASONS[1:]  # skip the very first season -- no earlier history exists to test it against
    all_test_matches = played[played["season"].isin(testable_seasons)]
    full_history = get_data_before_season(played, testable_seasons[0])
    predicted_counts, actual_counts = check_favorite_distribution(full_history, all_test_matches)
    print(f"Model's predicted favorites: {predicted_counts}")
    print(f"What actually happened:      {actual_counts}")

    print(f"\n{'Season':10s} {'N':>5s} {'Our Acc%':>9s} {'Naive Acc%':>11s} {'Our LogLoss':>12s} {'Naive LogLoss':>14s}")
    for r in all_results:
        print(f"{r['season']:10s} {r['matches']:>5d} {r['our_accuracy']:>8.1f}% {r['naive_accuracy']:>10.1f}% "
              f"{r['our_log_loss']:>12.3f} {r['naive_log_loss']:>14.3f}")

    avg_our_acc = sum(r["our_accuracy"] for r in all_results) / len(all_results)
    avg_naive_acc = sum(r["naive_accuracy"] for r in all_results) / len(all_results)
    avg_our_ll = sum(r["our_log_loss"] for r in all_results) / len(all_results)
    avg_naive_ll = sum(r["naive_log_loss"] for r in all_results) / len(all_results)

    print(f"\n{'AVERAGE':10s} {'':>5s} {avg_our_acc:>8.1f}% {avg_naive_acc:>10.1f}% {avg_our_ll:>12.3f} {avg_naive_ll:>14.3f}")