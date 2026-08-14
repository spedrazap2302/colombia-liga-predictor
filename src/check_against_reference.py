import pandas as pd
from load_data import load_data

# Reference totals you found -- (regular, playoff/cuadrangular, final)
REFERENCE = {
    "2021-I":  (171, 12, 2),
    "2021-II": (200, 24, 2),
    "2022-I":  (200, 24, 2),
    "2022-II": (200, 24, 2),
    "2023-I":  (200, 24, 2),
    "2023-II": (200, 24, 2),
    "2024-I":  (190, 24, 2),
    "2024-II": (190, 24, 2),
    "2025-I":  (200, 24, 2),
    "2025-II": (200, 24, 2),
}

played, upcoming = load_data()
all_matches = pd.concat([played, upcoming])

print(f"{'Season':10s} {'Ours: Reg/Playoff/Final':>26s} {'Reference: Reg/Playoff/Final':>30s} {'Match?':>8s}")

for season, (ref_reg, ref_playoff, ref_final) in REFERENCE.items():
    season_matches = all_matches[all_matches["season"] == season]

    our_reg = len(season_matches[season_matches["phase_type"] == "regular"])
    our_cuad = len(season_matches[(season_matches["phase_type"] == "cuadrangular") & (season_matches["stage"] != "final")])
    our_knockout = len(season_matches[(season_matches["phase_type"] == "knockout_top8") & (season_matches["stage"] != "final")])
    our_playoff = our_cuad + our_knockout
    our_final = len(season_matches[season_matches["stage"] == "final"])

    matches = (our_reg == ref_reg) and (our_playoff == ref_playoff) and (our_final == ref_final)
    flag = "OK" if matches else "DIFFERS"

    print(f"{season:10s} {our_reg:>3d}/{our_playoff:<3d}/{our_final:<3d}{'':>15s} {ref_reg:>3d}/{ref_playoff:<3d}/{ref_final:<3d}{'':>19s} {flag:>8s}")