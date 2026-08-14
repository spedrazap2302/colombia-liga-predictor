import pandas as pd
from load_data import load_data

CURRENT_SEASON = "2026-II"

def compute_standings(played: pd.DataFrame, season=CURRENT_SEASON):
    season_matches = played[played["season"] == season]

    stats = {}

    def ensure(team):
        if team not in stats:
            stats[team] = {"played": 0, "won": 0, "drawn": 0, "lost": 0,
                            "goals_for": 0, "goals_against": 0, "points": 0, "form": []}

    # Sort by date so "form" (recent results) comes out in the right order
    season_matches = season_matches.sort_values("date")

    for _, m in season_matches.iterrows():
        home, away = m["home_team"], m["away_team"]
        ensure(home)
        ensure(away)
        hg, ag = m["home_goals"], m["away_goals"]

        stats[home]["played"] += 1
        stats[away]["played"] += 1
        stats[home]["goals_for"] += hg
        stats[home]["goals_against"] += ag
        stats[away]["goals_for"] += ag
        stats[away]["goals_against"] += hg

        if hg > ag:
            stats[home]["won"] += 1
            stats[home]["points"] += 3
            stats[away]["lost"] += 1
            stats[home]["form"].append("W")
            stats[away]["form"].append("L")
        elif hg < ag:
            stats[away]["won"] += 1
            stats[away]["points"] += 3
            stats[home]["lost"] += 1
            stats[away]["form"].append("W")
            stats[home]["form"].append("L")
        else:
            stats[home]["drawn"] += 1
            stats[away]["drawn"] += 1
            stats[home]["points"] += 1
            stats[away]["points"] += 1
            stats[home]["form"].append("D")
            stats[away]["form"].append("D")

    rows = []
    for team, s in stats.items():
        goal_diff = s["goals_for"] - s["goals_against"]
        last_5_form = " ".join(s["form"][-5:])  # most recent 5 results
        rows.append({
            "team": team,
            "PJ": s["played"],
            "G": s["won"],
            "E": s["drawn"],
            "P": s["lost"],
            "Goals": f"{int(s['goals_for'])}:{int(s['goals_against'])}",
            "DG": goal_diff,
            "PTS": s["points"],
            "Form": last_5_form,
        })

    table = pd.DataFrame(rows)
    table = table.sort_values(["PTS", "DG"], ascending=False).reset_index(drop=True)
    table.index += 1
    return table