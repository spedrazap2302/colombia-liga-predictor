import pandas as pd


def load_coaches(filepath="data/coaches.xlsx"):
    df = pd.read_excel(filepath)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    return df


def get_coach_tenure_days(team, match_date, coaches_df):
    """Returns how many days the team's coach had been in charge as of
    match_date, or None if that team/date combination isn't covered by
    real data -- either a genuine gap, or a row explicitly marked
    UNKNOWN, which should never produce a fake number."""
    team_coaches = coaches_df[coaches_df["team"] == team]

    for _, row in team_coaches.iterrows():
        if "UNKNOWN" in str(row["coach_name"]):
            continue

        start = row["start_date"]
        end = row["end_date"] if pd.notna(row["end_date"]) else pd.Timestamp.max
        if start <= match_date <= end:
            return (match_date - start).days

    return None


if __name__ == "__main__":
    coaches = load_coaches()
    print(f"Loaded {len(coaches)} coaching-tenure records across {coaches['team'].nunique()} teams")

    # quick sanity check on a few known cases
    test_cases = [
        ("Nacional", "2026-08-01"),
        ("Millonarios", "2026-08-01"),
        ("Aguilas", "2025-06-01"),  # should be None -- the known gap
    ]
    for team, date in test_cases:
        tenure = get_coach_tenure_days(team, pd.Timestamp(date), coaches)
        print(f"{team} on {date}: {tenure} days" if tenure is not None else f"{team} on {date}: no data (expected for the known gap)")