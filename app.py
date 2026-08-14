import sys
sys.path.append("src")

import os
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
from load_data import load_data
from elo_model import build_ratings
from predict_match import match_probabilities
from standings import compute_standings, CURRENT_SEASON
from simulate_season import run_monte_carlo, N_SIMULATIONS, simulate_one_season_detailed, simulate_score
import zlib
from team_home_advantage import compute_team_home_adjustments

# Colors picked to hold up against the dataviz palette validator (contrast +
# CVD checks), not eyeballed -- gold marks the standout/leader, blue is the
# default series color, green/red are reserved status colors for W/L badges.
GOLD = "#eda100"
BLUE = "#2a78d6"
GOOD = "#0ca30c"
MUTED = "#898781"

st.set_page_config(page_title="Liga BetPlay Predictor", page_icon="⚽", layout="wide")
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

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏆 Championship Odds",
    "⚽ Upcoming Matches",
    "📊 Elo Ratings",
    "🎲 Simulate One Season",
    "📋 Standings",
])

@st.cache_data
def get_championship_probabilities(_played, _upcoming, _ratings, cache_key):
    return run_monte_carlo(_played, _upcoming, _ratings, CURRENT_SEASON)

with tab1:
    st.header("Probability of winning the 2026-II title")
    st.caption("Updates automatically whenever new results are added to the spreadsheet.")

    with st.spinner("Calculating..."):
        probabilities = get_championship_probabilities(played, upcoming, ratings, os.path.getmtime(data_path))

    leader = probabilities.iloc[0]
    runner_up = probabilities.iloc[1] if len(probabilities) > 1 else None
    col1, col2 = st.columns(2)
    col1.metric("Favorite to win the title", leader["team"], f"{leader['title_probability_%']}%")
    if runner_up is not None:
        gap = leader["title_probability_%"] - runner_up["title_probability_%"]
        col2.metric("Closest contender", runner_up["team"], f"{gap:+.1f}pp behind", delta_color="off")

    chart_df = probabilities.copy()
    chart_df["is_leader"] = chart_df["team"] == leader["team"]
    chart = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("title_probability_%:Q", title="Title probability (%)"),
            y=alt.Y("team:N", sort="-x", title=None),
            color=alt.condition(alt.datum.is_leader, alt.value(GOLD), alt.value(BLUE)),
            tooltip=[alt.Tooltip("team:N", title="Team"), alt.Tooltip("title_probability_%:Q", title="Probability (%)")],
        )
        .properties(height=alt.Step(24))
    )
    labels = chart.mark_text(align="left", dx=4, color=MUTED).encode(text="title_probability_%:Q")
    st.altair_chart(chart + labels, width="stretch")

    st.dataframe(
        probabilities,
        width="stretch",
        column_config={
            "title_probability_%": st.column_config.ProgressColumn(
                "Title probability", format="%.1f%%", min_value=0, max_value=max(100, leader["title_probability_%"]),
            ),
        },
    )

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

    def highlight_favorite(row):
        pct_cols = ["Home Win %", "Draw %", "Away Win %"]
        favorite_col = row[pct_cols].astype(float).idxmax()
        return [f"background-color: {GOLD}33; font-weight: 600" if c == favorite_col else "" for c in row.index]

    st.caption("Highlighted cell = the model's predicted favorite for that match.")
    st.dataframe(fixtures_df.style.apply(highlight_favorite, axis=1), width="stretch", height=600)

with tab3:
    st.header("Current team Elo ratings")
    ranked = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
    medals = {0: "🥇 ", 1: "🥈 ", 2: "🥉 "}
    ratings_df = pd.DataFrame(
        [(medals.get(i, "") + team, round(rating)) for i, (team, rating) in enumerate(ranked)],
        columns=["Team", "Elo Rating"],
    )
    ratings_df.index += 1
    st.dataframe(
        ratings_df,
        width="stretch",
        column_config={
            "Elo Rating": st.column_config.ProgressColumn(
                "Elo Rating", format="%d", min_value=int(ratings_df["Elo Rating"].min()), max_value=int(ratings_df["Elo Rating"].max()),
            ),
        },
    )

with tab4:
    st.header("One possible way the season could play out")
    st.caption("Each click simulates one random version of the rest of the season -- a different, equally plausible outcome every time.")

    def highlight_winner_row(row, winner):
        style = f"background-color: {GOLD}33; font-weight: 600"
        return [style if row["team"] == winner else "" for _ in row.index]

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
            st.dataframe(df_a.style.apply(highlight_winner_row, winner=result["winner_a"], axis=1), width="stretch")
            st.success(f"Group A winner: {result['winner_a']}")
        with col_b:
            st.subheader("Group B")
            df_b = pd.DataFrame(result["group_b_table"])
            st.dataframe(df_b.style.apply(highlight_winner_row, winner=result["winner_b"], axis=1), width="stretch")
            st.success(f"Group B winner: {result['winner_b']}")

        st.subheader("Final (two legs)")
        leg1 = result["final_legs"]["leg1"]
        leg2 = result["final_legs"]["leg2"]
        leg_col1, leg_col2 = st.columns(2)
        leg_col1.metric("Leg 1", f"{leg1['home_goals']} - {leg1['away_goals']}", f"{leg1['home']} vs {leg1['away']}", delta_color="off")
        leg_col2.metric("Leg 2", f"{leg2['home_goals']} - {leg2['away_goals']}", f"{leg2['home']} vs {leg2['away']}", delta_color="off")

        st.markdown(
            f"""<div style="background-color:{GOLD}33; border:1px solid {GOLD};
            border-radius:8px; padding:1.25rem; text-align:center; margin-top:1rem;">
            <span style="font-size:1.6rem; font-weight:700;">🏆 Champion: {result['champion']}</span>
            </div>""",
            unsafe_allow_html=True,
        )
        st.balloons()

with tab5:
    st.header("Current 2026-II Standings")
    st.caption("Based on matches actually played so far this season. Highlighted rows = top 8, who'd qualify for the cuadrangular stage today.")

    table = compute_standings(played)
    # Letter kept alongside the color dot so form is never conveyed by color alone
    # (green/red read the same to deuteranopic viewers).
    badge = {"W": "🟢W", "D": "⚪D", "L": "🔴L"}
    table["Form"] = table["Form"].apply(lambda f: " ".join(badge.get(c, c) for c in f.split()))

    def highlight_qualifiers(row):
        style = f"background-color: {GOOD}1a" if row.name <= 8 else ""
        return [style] * len(row)

    st.dataframe(table.style.apply(highlight_qualifiers, axis=1), width="stretch")

#streamlit run app.py
