import math
from load_data import load_data
from backtest import get_data_before_season, ALL_SEASONS
from elo_model import build_ratings
from predict_match import match_probabilities

# rho candidates to test -- Dixon-Coles literature typically finds
# values in the -0.2 to 0 range for football
RHO_CANDIDATES = [-0.20, -0.15, -0.10, -0.05, 0.0]


def evaluate(history, test_matches, rho=None, draw_inflation=1.10):
    ratings = build_ratings(history)

    correct = 0
    log_loss_sum = 0.0
    total = 0

    for _, m in test_matches.iterrows():
        home, away = m["home_team"], m["away_team"]
        if home not in ratings or away not in ratings:
            continue

        if rho is not None:
            hw, d, aw = match_probabilities(ratings[home], ratings[away], home, away,
                                             draw_inflation=1.0, dixon_coles_rho=rho)
        else:
            hw, d, aw = match_probabilities(ratings[home], ratings[away], home, away,
                                             draw_inflation=draw_inflation)

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

print(f"{'Method':30s} {'Avg Accuracy':>14s} {'Avg Log Loss':>14s}")

# baseline: current draw_inflation approach
accs, lls = [], []
for season in ALL_SEASONS:
    history = get_data_before_season(played, season)
    test_matches = played[played["season"] == season]
    if len(test_matches) == 0 or len(history) == 0:
        continue
    acc, ll = evaluate(history, test_matches, rho=None)
    accs.append(acc)
    lls.append(ll)
print(f"{'Current (draw_inflation=1.10)':30s} {sum(accs)/len(accs):>13.1f}% {sum(lls)/len(lls):>14.3f}")

# Dixon-Coles at each rho candidate
for rho in RHO_CANDIDATES:
    accs, lls = [], []
    for season in ALL_SEASONS:
        history = get_data_before_season(played, season)
        test_matches = played[played["season"] == season]
        if len(test_matches) == 0 or len(history) == 0:
            continue
        acc, ll = evaluate(history, test_matches, rho=rho)
        accs.append(acc)
        lls.append(ll)
    label = f"Dixon-Coles (rho={rho})"
    print(f"{label:30s} {sum(accs)/len(accs):>13.1f}% {sum(lls)/len(lls):>14.3f}")