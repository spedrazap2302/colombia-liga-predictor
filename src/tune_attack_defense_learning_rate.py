import attack_defense_model
from load_data import load_data
from backtest import get_data_before_season, ALL_SEASONS
from compare_attack_defense_methods import evaluate_attack_defense

CANDIDATE_VALUES = [0.01, 0.02, 0.05, 0.10, 0.20]

played, _ = load_data()

print(f"{'LEARNING_RATE':>15s} {'Avg Accuracy':>14s} {'Avg Log Loss':>14s}")

for candidate in CANDIDATE_VALUES:
    attack_defense_model.LEARNING_RATE = candidate

    accs, lls = [], []
    for season in ALL_SEASONS:
        history = get_data_before_season(played, season)
        test_matches = played[played["season"] == season]
        if len(test_matches) == 0 or len(history) == 0:
            continue
        acc, ll = evaluate_attack_defense(history, test_matches)
        accs.append(acc)
        lls.append(ll)

    avg_acc = sum(accs) / len(accs)
    avg_ll = sum(lls) / len(lls)
    print(f"{candidate:>15.2f} {avg_acc:>13.1f}% {avg_ll:>14.3f}")