import pandas as pd
from load_data import load_data
from elo_model import build_ratings
from predict_match import match_probabilities

played, upcoming = load_data()
ratings = build_ratings(played)

results = []
for _, row in upcoming.iterrows():
    home, away = row["home_team"], row["away_team"]

    if home not in ratings or away not in ratings:
        continue  # skip if a team name doesn't match our ratings (name mismatch)

    hw, d, aw = match_probabilities(ratings[home], ratings[away], home, away)
    results.append({
        "date": row["date"].date(),
        "home_team": home,
        "away_team": away,
        "home_win_%": round(hw * 100, 1),
        "draw_%": round(d * 100, 1),
        "away_win_%": round(aw * 100, 1),
    })

predictions = pd.DataFrame(results)
print(predictions.to_string(index=False))