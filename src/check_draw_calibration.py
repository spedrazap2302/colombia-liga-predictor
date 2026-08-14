import pandas as pd
from load_data import load_data
from backtest import get_data_before_season, ALL_SEASONS
from elo_model import build_ratings
from predict_match import match_probabilities

played, _ = load_data()

records = []

for season in ALL_SEASONS:
    history = get_data_before_season(played, season)
    test_matches = played[played["season"] == season]
    if len(test_matches) == 0 or len(history) == 0:
        continue

    ratings = build_ratings(history)

    for _, m in test_matches.iterrows():
        home, away = m["home_team"], m["away_team"]
        if home not in ratings or away not in ratings:
            continue

        _, draw_prob, _ = match_probabilities(ratings[home], ratings[away], home, away)
        actual_draw = 1 if m["home_goals"] == m["away_goals"] else 0

        records.append({"predicted_draw_prob": draw_prob, "actual_draw": actual_draw})

df = pd.DataFrame(records)

# Bucket predictions into ranges and compare predicted vs actual rate in each
df["bucket"] = pd.cut(df["predicted_draw_prob"], bins=[0, 0.20, 0.24, 0.28, 0.32, 1.0])

summary = df.groupby("bucket", observed=True).agg(
    avg_predicted=("predicted_draw_prob", "mean"),
    actual_draw_rate=("actual_draw", "mean"),
    n=("actual_draw", "count"),
)

print(summary.to_string())