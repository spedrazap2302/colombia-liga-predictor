import pandas as pd

def load_data(filepath="data/matches.csv.xlsx"):
    df = pd.read_excel(filepath)
    df["date"] = pd.to_datetime(df["date"])
    df["season"] = df["season"].astype(str)
    df = df.sort_values("date").reset_index(drop=True)

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