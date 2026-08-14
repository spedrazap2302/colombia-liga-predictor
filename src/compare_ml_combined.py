import math
from sklearn.ensemble import RandomForestClassifier
from load_data import load_data
from backtest import get_data_before_season, ALL_SEASONS
from elo_model import build_ratings, HOME_ADVANTAGE
from predict_match import match_probabilities
from ml_features_combined import build_combined_training_dataset, coaches_df
from altitude import altitude_boost
from coaches import get_coach_tenure_days

FEATURE_COLUMNS = ["elo_diff", "effective_diff", "altitude_effect", "home_form", "away_form",
                    "home_record", "home_coach_tenure", "away_coach_tenure"]


def evaluate_elo(history, test_matches):
    ratings = build_ratings(history)
    correct = 0
    log_loss_sum = 0.0
    total = 0
    for _, m in test_matches.iterrows():
        home, away = m["home_team"], m["away_team"]
        if home not in ratings or away not in ratings:
            continue
        hw, d, aw = match_probabilities(ratings[home], ratings[away], home, away)
        actual = "home" if m["home_goals"] > m["away_goals"] else "away" if m["home_goals"] < m["away_goals"] else "draw"
        predicted = max([("home", hw), ("draw", d), ("away", aw)], key=lambda x: x[1])[0]
        if predicted == actual:
            correct += 1
        actual_prob = {"home": hw, "draw": d, "away": aw}[actual]
        log_loss_sum += -1 * math.log(max(actual_prob, 0.001))
        total += 1
    return correct / total * 100, log_loss_sum / total


def evaluate_ml_combined(history, test_matches):
    train_df, final_ratings, recent_results, home_results = build_combined_training_dataset(history)
    if len(train_df) < 100 or train_df["outcome"].nunique() < 2:
        return None, None

    clf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=15, random_state=42)
    clf.fit(train_df[FEATURE_COLUMNS], train_df["outcome"])
    class_order = list(clf.classes_)

    correct = 0
    log_loss_sum = 0.0
    total = 0

    for _, m in test_matches.iterrows():
        home, away = m["home_team"], m["away_team"]
        if home not in final_ratings or away not in final_ratings:
            continue

        alt = altitude_boost(home, away)
        elo_diff = final_ratings[home] - final_ratings[away]
        effective_diff = (final_ratings[home] + HOME_ADVANTAGE + alt) - final_ratings[away]

        home_form = sum(recent_results.get(home, [])[-5:]) / len(recent_results.get(home, [])[-5:]) if len(recent_results.get(home, [])) >= 5 else 0.5
        away_form = sum(recent_results.get(away, [])[-5:]) / len(recent_results.get(away, [])[-5:]) if len(recent_results.get(away, [])) >= 5 else 0.5
        home_record = sum(home_results.get(home, [])) / len(home_results.get(home, [])) if len(home_results.get(home, [])) >= 5 else 0.5

        home_tenure = get_coach_tenure_days(home, m["date"], coaches_df)
        away_tenure = get_coach_tenure_days(away, m["date"], coaches_df)
        home_tenure = home_tenure if home_tenure is not None else -1
        away_tenure = away_tenure if away_tenure is not None else -1

        features = [[elo_diff, effective_diff, alt, home_form, away_form, home_record, home_tenure, away_tenure]]
        probs = clf.predict_proba(features)[0]
        prob_by_class = dict(zip(class_order, probs))

        actual = "home" if m["home_goals"] > m["away_goals"] else "away" if m["home_goals"] < m["away_goals"] else "draw"
        predicted = max(prob_by_class.items(), key=lambda x: x[1])[0]
        if predicted == actual:
            correct += 1
        actual_prob = prob_by_class.get(actual, 0.001)
        log_loss_sum += -1 * math.log(max(actual_prob, 0.001))
        total += 1

    return correct / total * 100, log_loss_sum / total


played, _ = load_data()

results_elo = []
results_ml = []

for season in ALL_SEASONS:
    history = get_data_before_season(played, season)
    test_matches = played[played["season"] == season]
    if len(test_matches) == 0 or len(history) == 0:
        continue

    acc_elo, ll_elo = evaluate_elo(history, test_matches)
    acc_ml, ll_ml = evaluate_ml_combined(history, test_matches)

    results_elo.append((acc_elo, ll_elo))
    if acc_ml is not None:
        results_ml.append((acc_ml, ll_ml))

avg_acc_elo = sum(r[0] for r in results_elo) / len(results_elo)
avg_ll_elo = sum(r[1] for r in results_elo) / len(results_elo)
avg_acc_ml = sum(r[0] for r in results_ml) / len(results_ml)
avg_ll_ml = sum(r[1] for r in results_ml) / len(results_ml)

print(f"{'Method':30s} {'Avg Accuracy':>14s} {'Avg Log Loss':>14s}")
print(f"{'Elo + Poisson (current)':30s} {avg_acc_elo:>13.1f}% {avg_ll_elo:>14.3f}")
print(f"{'RF with all combined signals':30s} {avg_acc_ml:>13.1f}% {avg_ll_ml:>14.3f}")