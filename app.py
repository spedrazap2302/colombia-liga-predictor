import sys
sys.path.append("src")

import os
import pandas as pd
import numpy as np
import streamlit as st
from load_data import load_data
from elo_model import build_ratings
from predict_match import match_probabilities
from standings import compute_standings, CURRENT_SEASON
from simulate_season import run_monte_carlo, N_SIMULATIONS, simulate_one_season_detailed, simulate_score
import zlib
from team_home_advantage import compute_team_home_adjustments

st.set_page_config(page_title="Liga BetPlay Predictor", layout="wide")
st.title("🇨🇴 Liga BetPlay Colombia — Predictor")

@st.cache_data
def get_data(file_mtime):
    return load_data(data_path)

data_path = "data/matches.csv.xlsx"
played, upcoming = get_data(os.path.getmtime(data_path))

@st.cache_data
def get_ratings(played):
    return build_ratings(played)

ratings = get_ratings(played)
home_adjustments = compute_team_home_adjustments(played)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Championship Odds", "Upcoming Match Predictions", "Current Elo Ratings", "Simulate One Season", "Current Standings"])

@st.cache_data
def get_championship_probabilities(_played, _upcoming, _ratings, cache_key):
    return run_monte_carlo(_played, _upcoming, _ratings, CURRENT_SEASON)

with tab1:
    st.header("Probability of winning the 2026-II title")
    st.caption("Updates automatically whenever new results are added to the spreadsheet.")

    with st.spinner("Calculating..."):
        probabilities = get_championship_probabilities(played, upcoming, ratings, os.path.getmtime(data_path))
    st.dataframe(probabilities, width="stretch")
    st.bar_chart(probabilities.set_index("team")["title_probability_%"])

with tab2:
    st.header("Win / Draw / Loss probabilities for every upcoming fixture")
    st.caption("The 'Simulated Score' column is one random possible result, not a prediction of the exact score -- click the button below to roll new ones.")

    reroll = st.button("Re-roll simulated scores")

    rows = []
    for _, row in upcoming.iterrows():
        home, away = row["home_team"], row["away_team"]
        if home not in ratings or away not in ratings:
            continue
        hw, d, aw = match_probabilities(ratings[home], ratings[away], home, away, home_adjustments.get(home, 0))
        match_seed = zlib.crc32((home + away).encode()) % (2**32)
        np.random.seed(match_seed)
        sim_hg, sim_ag = simulate_score(home, away, ratings)
        rows.append({
            "Date": row["date"].date(),
            "Home": home, "Away": away,
            "Home Win %": round(hw * 100, 1),
            "Draw %": round(d * 100, 1),
            "Away Win %": round(aw * 100, 1),
            "Simulated Score": f"{sim_hg} - {sim_ag}",
        })

    fixtures_df = pd.DataFrame(rows)
    st.dataframe(fixtures_df, width="stretch", height=600)

with tab3:
    st.header("Current team Elo ratings")
    ranked = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
    ratings_df = pd.DataFrame(ranked, columns=["Team", "Elo Rating"])
    ratings_df.index += 1
    st.dataframe(ratings_df, width="stretch")

with tab4:
    st.header("One possible way the season could play out")
    st.caption("Each click simulates one random version of the rest of the season -- a different, equally plausible outcome every time.")

    if st.button("Simulate one full season"):
        result = simulate_one_season_detailed(played, upcoming, ratings, CURRENT_SEASON)

        st.subheader("Final regular-season table (top 8 qualify)")
        table_df = pd.DataFrame({"Team": result["final_table"]})
        table_df.index += 1
        st.dataframe(table_df.head(8), width="stretch")

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Group A")
            df_a = pd.DataFrame(result["group_a_table"])
            st.dataframe(df_a, width="stretch")
            st.success(f"Group A winner: {result['winner_a']}")
        with col_b:
            st.subheader("Group B")
            df_b = pd.DataFrame(result["group_b_table"])
            st.dataframe(df_b, width="stretch")
            st.success(f"Group B winner: {result['winner_b']}")

        st.subheader("Final (two legs)")
        leg1 = result["final_legs"]["leg1"]
        leg2 = result["final_legs"]["leg2"]
        st.write(f"Leg 1: **{leg1['home']}** {leg1['home_goals']} - {leg1['away_goals']} **{leg1['away']}**")
        st.write(f"Leg 2: **{leg2['home']}** {leg2['home_goals']} - {leg2['away_goals']} **{leg2['away']}**")

        st.header(f"🏆 Champion: {result['champion']}")

with tab5:
    st.header("Current 2026-II Standings")
    st.caption("Based on matches actually played so far this season.")

    table = compute_standings(played)
    st.dataframe(table, width="stretch")

#streamlit run app.py
