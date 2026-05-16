# pages/2_🗺️_Destination_Recommender.py
# ─────────────────────────────────────────────
# Part A Extension: Destination Recommender
# Recommends Sri Lankan destinations based on
# user keyword preferences and district choice
# using keyword density scoring from Spark
# ─────────────────────────────────────────────

import streamlit as st
import plotly.express as px
import pandas as pd
from utils.spark_session import get_spark_session
from utils.part_a_analytics import (
    engineer_popularity_score,
    get_tfidf_keywords,
    get_district_distribution,
    get_keyword_scored_destinations
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Destination Recommender",
    page_icon="🗺️",
    layout="wide"
)

# ─────────────────────────────────────────────
# SPARK SESSION
# ─────────────────────────────────────────────

@st.cache_resource
def init_spark():
    return get_spark_session()

spark = init_spark()

# ─────────────────────────────────────────────
# CACHED DATA LOADERS
# ─────────────────────────────────────────────

@st.cache_data
def load_popularity():
    return engineer_popularity_score(spark)

@st.cache_data
def load_keywords():
    return get_tfidf_keywords(spark, top_n=200)

@st.cache_data
def load_districts():
    return get_district_distribution(spark)

# ─────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────

st.title("🗺️ Destination Recommender")
st.markdown("**Find the best Sri Lankan destinations based on your preferences**")
st.markdown("Powered by **keyword density scoring** and **popularity ranking** via Apache Spark.")
st.divider()

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

with st.spinner("Loading destination data from Spark..."):
    popularity_df = load_popularity()
    keywords_df = load_keywords()
    district_df = load_districts()

# ─────────────────────────────────────────────
# USER PREFERENCES
# ─────────────────────────────────────────────

st.subheader("🎯 Set Your Preferences")

col1, col2, col3 = st.columns(3)

with col1:
    districts = ["All Districts"] + sorted(
        popularity_df["District"].unique().tolist()
    )
    selected_district = st.selectbox(
        "📍 Preferred District",
        districts,
        help="Select a specific district or search all"
    )

with col2:
    min_popularity = st.slider(
        "⭐ Minimum Popularity Score",
        min_value=0,
        max_value=100,
        value=10,
        help="0 = show all, 100 = only the most popular"
    )

with col3:
    top_n = st.slider(
        "🔢 Number of Recommendations",
        min_value=5,
        max_value=30,
        value=10
    )

# Keyword search
st.markdown("#### 🔑 Keyword Preferences")
st.markdown("Type keywords that match your travel interests:")

col1, col2 = st.columns(2)
with col1:
    keyword_input = st.text_input(
        "Enter keywords (comma separated)",
        placeholder="e.g. nature, waterfall, hiking, beach, temple",
        help="Destinations are ranked by how frequently these keywords appear in their reviews"
    )

with col2:
    top_keywords = keywords_df["word"].head(20).tolist()
    st.markdown("**Popular keywords from reviews:**")
    st.markdown(" · ".join([f"`{k}`" for k in top_keywords]))

st.divider()

# ─────────────────────────────────────────────
# GENERATE RECOMMENDATIONS
# ─────────────────────────────────────────────

if st.button("🔍 Find Destinations", type="primary", use_container_width=True):

    with st.spinner("Matching destinations using Spark analytics..."):

        if keyword_input.strip():
            keywords = [k.strip().lower() for k in keyword_input.split(",") if k.strip()]

            # Get keyword density scored results from Spark
            result = get_keyword_scored_destinations(
                spark, keywords, top_n=top_n
            )

            # Apply district filter
            if selected_district != "All Districts":
                result = result[result["District"] == selected_district]

            # Apply minimum popularity filter if column exists
            if "final_score" in result.columns:
                result = result[result["final_score"] >= 0]

        else:
            # No keywords — use popularity ranking only
            filtered = popularity_df.copy()

            if selected_district != "All Districts":
                filtered = filtered[filtered["District"] == selected_district]

            filtered = filtered[filtered["popularity_score"] >= min_popularity]
            result = filtered.sort_values(
                "popularity_score", ascending=False
            ).head(top_n)

    # ─────────────────────────────────────────
    # DISPLAY RESULTS
    # ─────────────────────────────────────────

    if result.empty:
        st.warning("""
        No destinations found matching your preferences.
        Try different keywords or adjust your filters.
        """)
    else:
        st.success(f"✅ Found **{len(result)}** destinations matching your preferences!")
        st.divider()

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Destinations Found", len(result))

        if "relevance_score" in result.columns:
            col2.metric("Avg Relevance Score", f"{result['relevance_score'].mean():.1f}")
        elif "popularity_score" in result.columns:
            col2.metric("Avg Popularity Score", f"{result['popularity_score'].mean():.1f}")

        col3.metric("Districts Covered", result["District"].nunique())

        st.divider()

        # Results table
        st.markdown("#### 📋 Recommended Destinations")

        if "final_score" in result.columns:
            display_df = result[[
                "Destination", "District",
                "review_count", "keyword_freq",
                "relevance_score", "final_score"
            ]].copy()
            display_df.columns = [
                "Destination", "District",
                "Total Reviews", "Keyword Mentions",
                "Relevance Score", "Final Score"
            ]
        else:
            display_df = result[[
                "Destination", "District",
                "review_count", "popularity_score", "district_rank"
            ]].copy()
            display_df.columns = [
                "Destination", "District",
                "Total Reviews", "Popularity Score", "District Rank"
            ]

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.divider()

        # Visualisations
        col1, col2 = st.columns(2)

        with col1:
            # Pick best available score column
            score_col = "final_score" if "final_score" in result.columns \
                else "relevance_score" if "relevance_score" in result.columns \
                else "popularity_score"

            fig = px.bar(
                result,
                x=score_col,
                y="Destination",
                orientation="h",
                color="District",
                title="Recommended Destinations by Score",
                labels={score_col: "Score"},
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.pie(
                result,
                names="District",
                values=score_col,
                title="Recommendations by District",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig, use_container_width=True)

        # Top recommendation highlight
        st.divider()
        top = result.iloc[0]
        st.markdown("### 🏆 Top Recommendation")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📍 Destination", top["Destination"])
        col2.metric("🗺️ District", top["District"])
        col3.metric(
            "⭐ Score",
            f"{top.get('final_score', top.get('relevance_score', top.get('popularity_score', 0))):.1f}"
        )
        col4.metric("📝 Total Reviews", f"{int(top['review_count']):,}")

else:
    # Default view — show all top destinations
    st.markdown("#### 🌟 Most Popular Sri Lankan Destinations")
    st.markdown("Set your preferences above and click **Find Destinations** to get personalised recommendations.")

    top_all = popularity_df.sort_values(
        "popularity_score", ascending=False
    ).head(15)

    fig = px.bar(
        top_all,
        x="popularity_score",
        y="Destination",
        orientation="h",
        color="District",
        title="Top 15 Destinations by Popularity Score",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(
            district_df,
            values="Review Count",
            names="District",
            title="Tourism Distribution by District",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            popularity_df.groupby("District")["popularity_score"]
            .mean().reset_index().sort_values("popularity_score", ascending=False),
            x="District",
            y="popularity_score",
            color="popularity_score",
            color_continuous_scale="Teal",
            title="Average Popularity Score by District"
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

st.divider()
st.markdown("""
<div style='text-align: center; color: grey; font-size: 13px;'>
    Destination recommendations powered by Apache Spark · Keyword Density Scoring · Popularity Ranking
</div>
""", unsafe_allow_html=True)
