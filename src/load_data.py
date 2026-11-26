import pandas as pd

DATA_PATH = "data/matches.csv"


def load_data(filepath=DATA_PATH):
    """Reads every match (played and upcoming) and splits them in two.
    The file is a plain CSV so the daily updater (update_results.py) can add
    new results automatically and GitHub can show exactly what changed."""
    df = pd.read_csv(filepath, dtype={"season": str})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date", kind="stable").reset_index(drop=True)

    played = df.dropna(subset=["home_goals", "away_goals"]).copy()
    upcoming = df[df["home_goals"].isna() | df["away_goals"].isna()].copy()

    return played, upcoming


if __name__ == "__main__":
    played, upcoming = load_data()
    print(f"Played matches: {len(played)}")
    print(f"Date range: {played['date'].min()} to {played['date'].max()}")
    print(f"\nUpcoming fixtures: {len(upcoming)}")
    print("\nFirst few played matches:")
    print(played.head())
