import pandas as pd
from load_data import load_data

played, upcoming = load_data()
all_matches = pd.concat([played, upcoming])

regular = all_matches[all_matches["phase_type"] == "regular"]

pairing_counts = regular.groupby(["season", "home_team", "away_team"]).size().reset_index(name="count")
repeated = pairing_counts[pairing_counts["count"] > 1].sort_values(["season", "count"], ascending=[True, False])

print(f"Found {len(repeated)} home/away pairings that appear more than once in the regular season:\n")
print(repeated.to_string(index=False))