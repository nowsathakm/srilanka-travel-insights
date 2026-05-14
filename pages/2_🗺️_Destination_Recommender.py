# pages/2_🗺️_Destination_Recommender.py
# ─────────────────────────────────────────────
# Part A Extension: Destination Recommender
# Recommends Sri Lankan destinations based on
# user keyword preferences and district choice
# using TF-IDF similarity from Spark MLlib
# ─────────────────────────────────────────────

import streamlit as st
import plotly.express as px
import pandas as pd
from utils.spark_session import get_spark_session
from utils.part_a_analytics import (
    engineer_popularity_score,
    get_tfidf_keywords,
    get_district_distribution
)
from pyspark.sql import functions as F

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
st.markdown("Powered by **TF-IDF keyword matching** and **Popularity Scoring** via Apache Spark.")
st.divider()

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

with st.spinner("Loading destination data from Spark..."):
    popularity_df = load_popularity()
    keywords_df = load_keywords()
    district_df = load_districts()

# ─────────────────────────────────────────────
# SECTION 1 — USER PREFERENCES
# ─────────────────────────────────────────────

st.subheader("🎯 Set Your Preferences")

col1, col2, col3 = st.columns(3)

with col1:
    # District filter
    districts = ["All Districts"] + sorted(
        popularity_df["District"].unique().tolist()
    )
    selected_district = st.selectbox(
        "📍 Preferred District",
        districts,
        help="Select a specific district or search all"
    )

with col2:
    # Minimum popularity score
    min_popularity = st.slider(
        "⭐ Minimum Popularity Score",
        min_value=0,
        max_value=100,
        value=30,
        help="0 = show all, 100 = only the most popular"
    )

with col3:
    # Number of recommendations
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
        help="We'll match these against TF-IDF keywords from real tourist reviews"
    )

with col2:
    # Show available top keywords as chips
    top_keywords = keywords_df["word"].head(20).tolist()
    st.markdown("**Popular keywords from reviews:**")
    st.markdown(" · ".join([f"`{k}`" for k in top_keywords]))

st.divider()

# ─────────────────────────────────────────────
# SECTION 2 — GENERATE RECOMMENDATIONS
# ─────────────────────────────────────────────

if st.button("🔍 Find Destinations", type="primary", use_container_width=True):

    with st.spinner("Matching destinations using Spark analytics..."):

        # Start with full popularity dataframe
        filtered = popularity_df.copy()

        # Apply district filter
        if selected_district != "All Districts":
            filtered = filtered[filtered["District"] == selected_district]

        # Apply popularity score filter
        filtered = filtered[filtered["popularity_score"] >= min_popularity]

        # Apply keyword matching if provided
        if keyword_input.strip():
            keywords = [k.strip().lower() for k in keyword_input.split(",")]

            # Load reviews for keyword matching
            reviews_spark = spark.read.csv(
                "data/reviews_final.csv",
                header=True,
                inferSchema=True
            )

            # For each keyword find matching destinations
            matched_destinations = set()
            for keyword in keywords:
                matches = reviews_spark.filter(
                    F.lower(F.col("Review")).contains(keyword)
                ).select("Destination").distinct()
                matched_list = [r["Destination"] for r in matches.collect()]
                matched_destinations.update(matched_list)

            if matched_destinations:
                filtered = filtered[
                    filtered["Destination"].isin(matched_destinations)
                ]

        # Sort by popularity score and get top N
        result = filtered.sort_values(
            "popularity_score", ascending=False
        ).head(top_n)

    # ─────────────────────────────────────────
    # DISPLAY RESULTS
    # ─────────────────────────────────────────

    if result.empty:
        st.warning("No destinations found matching your preferences. Try adjusting your filters.")
    else:
        st.success(f"✅ Found **{len(result)}** destinations matching your preferences!")
        st.divider()

        # Results metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Destinations Found", len(result))
        col2.metric("Avg Popularity Score", f"{result['popularity_score'].mean():.1f}")
        col3.metric("Districts Covered", result["District"].nunique())

        st.divider()

        # Results table
        st.markdown("#### 📋 Recommended Destinations")
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

        # Visualisation
        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(
                result,
                x="popularity_score",
                y="Destination",
                orientation="h",
                color="District",
                title="Recommended Destinations by Popularity Score",
                labels={"popularity_score": "Popularity Score"},
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.pie(
                result,
                names="District",
                values="popularity_score",
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
        col3.metric("⭐ Popularity Score", f"{top['popularity_score']:.1f}")
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

    # District overview
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
    Destination recommendations powered by Apache Spark TF-IDF · Popularity Scoring · Window Functions
</div>
""", unsafe_allow_html=True)