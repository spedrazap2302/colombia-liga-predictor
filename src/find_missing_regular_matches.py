import pandas as pd
from itertools import combinations
from load_data import load_data

played, upcoming = load_data()
all_matches = pd.concat([played, upcoming])

SEASONS_TO_CHECK = ["2023-I", "2023-II", "2024-I", "2024-II", "2025-II"]

for season in SEASONS_TO_CHECK:
    season_matches = all_matches[(all_matches["season"] == season) & (all_matches["phase_type"] == "regular")]
    teams = sorted(set(season_matches["home_team"]) | set(season_matches["away_team"]))

    existing_pairs = set(zip(season_matches["home_team"], season_matches["away_team"]))

    missing = []
    for team_a, team_b in combinations(teams, 2):
        if (team_a, team_b) not in existing_pairs and (team_b, team_a) not in existing_pairs:
            missing.append((team_a, team_b))

    print(f"\n{season}: {len(missing)} completely missing pairing(s) (neither team hosted the other)")
    for pair in missing:
        print(f"  {pair[0]} vs {pair[1]} (not entered at all, in either direction)")