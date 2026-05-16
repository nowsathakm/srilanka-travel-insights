# pages/3_🏠_Accommodation_Recommender.py
# ─────────────────────────────────────────────
# Part B: Recommendation System Using Big Data
# Hybrid accommodation recommender combining
# content-based filtering + popularity ranking
# ─────────────────────────────────────────────

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from utils.spark_session import get_spark_session
from utils.part_b_recommender import (
    get_hybrid_recommendations,
    get_content_based_recommendations,
    get_popularity_rankings,
    evaluate_recommendations,
    get_room_types,
    get_airbnb_overview
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Accommodation Recommender",
    page_icon="🏠",
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
def load_room_types():
    return get_room_types(spark)

@st.cache_data
def load_overview():
    return get_airbnb_overview(spark)

@st.cache_data
def load_popularity_rankings():
    return get_popularity_rankings(spark)

# ─────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────

st.title("🏠 Accommodation Recommender")
st.markdown("**Part B — Hybrid Recommendation System Using Big Data**")
st.markdown("""
A **hybrid recommendation engine** combining content-based filtering and
popularity-based ranking to suggest the best Sri Lankan accommodations.
""")
st.divider()

# ─────────────────────────────────────────────
# DATASET OVERVIEW
# ─────────────────────────────────────────────

with st.spinner("Loading accommodation data..."):
    overview = load_overview()
    room_types = load_room_types()

col1, col2, col3, col4 = st.columns(4)
col1.metric("🏠 Total Listings", f"{overview['total_listings']:,}")
col2.metric("🛏️ Room Types", f"{overview['unique_room_types']:,}")
col3.metric("⭐ Average Stars", f"{overview['avg_stars']}")
col4.metric("👥 Max Guests", f"{overview['max_guests']}")

st.divider()

# ─────────────────────────────────────────────
# HOW IT WORKS
# ─────────────────────────────────────────────

with st.expander("ℹ️ How the Hybrid Recommendation System Works"):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        #### 1️⃣ Content-Based Filtering
        Matches your preferences:
        - Room type preference
        - Number of guests
        - Minimum star rating
        Filters listings that fit
        your exact requirements.
        """)
    with col2:
        st.markdown("""
        #### 2️⃣ Popularity-Based Ranking
        Ranks filtered listings by:
        - Star rating (quality score)
        - Guest capacity match
        Ensures you only see
        highly rated options.
        """)
    with col3:
        st.markdown("""
        #### 3️⃣ Hybrid Scoring Formula
        ```
        hybrid_score =
          (stars_norm × 0.7) +
          (guest_match × 0.3)
        ```
        70% quality weight +
        30% relevance weight.
        """)

st.divider()

# ─────────────────────────────────────────────
# USER PREFERENCES
# ─────────────────────────────────────────────

st.subheader("🎯 Your Accommodation Preferences")

col1, col2, col3 = st.columns(3)

with col1:
    selected_room_type = st.selectbox(
        "🛏️ Room Type",
        room_types,
        help="Select your preferred accommodation type"
    )

with col2:
    num_guests = st.number_input(
        "👥 Number of Guests",
        min_value=1,
        max_value=16,
        value=2,
        help="Minimum guest capacity needed"
    )

with col3:
    min_stars = st.slider(
        "⭐ Minimum Star Rating",
        min_value=1.0,
        max_value=5.0,
        value=4.0,
        step=0.5,
        help="Minimum acceptable star rating"
    )

col1, col2 = st.columns(2)
with col1:
    top_n = st.slider(
        "🔢 Number of Recommendations",
        min_value=5,
        max_value=20,
        value=10
    )

with col2:
    recommendation_mode = st.radio(
        "🔀 Recommendation Mode",
        ["Hybrid (Recommended)", "Content-Based Only", "Popularity Only"],
        horizontal=True
    )

st.divider()

# ─────────────────────────────────────────────
# GENERATE RECOMMENDATIONS
# ─────────────────────────────────────────────

if st.button("🏨 Get Recommendations", type="primary", use_container_width=True):

    with st.spinner("Running recommendation engine on Spark..."):

        if recommendation_mode == "Hybrid (Recommended)":
            results = get_hybrid_recommendations(
                spark, selected_room_type,
                num_guests, min_stars, top_n
            )
            mode_label = "Hybrid"

        elif recommendation_mode == "Content-Based Only":
            results = get_content_based_recommendations(
                spark, selected_room_type,
                num_guests, top_n
            )
            mode_label = "Content-Based"

        else:
            results = get_popularity_rankings(spark, top_n)
            mode_label = "Popularity"

    # ─────────────────────────────────────────
    # DISPLAY RESULTS
    # ─────────────────────────────────────────

    if results.empty:
        st.warning("""
        No accommodations found matching your preferences.
        Try lowering the minimum star rating or
        selecting 'Any' for room type.
        """)
    else:
        st.success(f"✅ Found **{len(results)}** recommendations using **{mode_label}** mode!")
        st.divider()

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Recommendations", len(results))

        if "stars" in results.columns:
            col2.metric("Avg Stars", f"{results['stars'].mean():.2f} ⭐")
            col3.metric("Top Stars", f"{results['stars'].max():.2f} ⭐")

        if "hybrid_score" in results.columns:
            col4.metric("Avg Hybrid Score", f"{results['hybrid_score'].mean():.3f}")

        st.divider()

        # Results table
        st.markdown("#### 📋 Your Recommendations")
        st.dataframe(results, use_container_width=True, hide_index=True)

        st.divider()

        # Visualisations
        col1, col2 = st.columns(2)

        with col1:
            if "stars" in results.columns and "name" in results.columns:
                fig = px.bar(
                    results.head(10),
                    x="stars",
                    y="name",
                    orientation="h",
                    color="stars",
                    color_continuous_scale="Viridis",
                    title="Top Recommendations by Star Rating",
                    labels={"stars": "Star Rating", "name": "Property"}
                )
                fig.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if "roomType" in results.columns:
                room_dist = results["roomType"].value_counts().reset_index()
                room_dist.columns = ["Room Type", "Count"]
                fig = px.pie(
                    room_dist,
                    values="Count",
                    names="Room Type",
                    title="Recommendations by Room Type",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig, use_container_width=True)

        # Hybrid score chart
        if "hybrid_score" in results.columns:
            st.markdown("#### 📐 Hybrid Score Breakdown")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="Stars (70% weight)",
                x=results["name"],
                y=results["stars"] * 0.7,
                marker_color="#636EFA"
            ))
            fig.add_trace(go.Bar(
                name="Guest Match (30% weight)",
                x=results["name"],
                y=results.get("guest_match_score", pd.Series([0]*len(results))) * 0.3,
                marker_color="#EF553B"
            ))
            fig.update_layout(
                barmode="stack",
                title="Hybrid Score Components per Recommendation",
                xaxis_tickangle=-45,
                legend=dict(orientation="h", y=-0.3)
            )
            st.plotly_chart(fig, use_container_width=True)

        # Top recommendation highlight
        st.divider()
        st.markdown("### 🏆 Top Recommendation")
        top = results.iloc[0]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🏠 Property", top.get("name", "N/A"))
        col2.metric("🛏️ Room Type", top.get("roomType", "N/A"))
        col3.metric("👥 Guests", top.get("numberOfGuests", "N/A"))
        col4.metric("⭐ Stars", top.get("stars", "N/A"))

        # ─────────────────────────────────────
        # EVALUATION METRICS
        # ─────────────────────────────────────

        st.divider()
        st.subheader("📊 System Evaluation — Precision@K")
        st.markdown("""
        Evaluating recommendation quality using **Precision@K** —
        the standard metric for content-based recommendation systems.
        > *Precision@K = Relevant recommendations in top K / K*
        """)

        with st.spinner("Evaluating recommendations..."):
            eval_results = evaluate_recommendations(
                spark, selected_room_type, num_guests, top_n
            )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "Precision@K",
            f"{eval_results['precision_at_k']:.2f}",
            help="Proportion of relevant recommendations"
        )
        col2.metric(
            "Relevant Results",
            f"{eval_results['relevant_count']} / {eval_results['total_recommended']}"
        )
        col3.metric(
            "Coverage",
            f"{eval_results['coverage']}%",
            help="% of total listings considered"
        )
        col4.metric(
            "Total Recommended",
            eval_results["total_recommended"]
        )

        # Precision gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=eval_results["precision_at_k"] * 100,
            title={"text": "Precision@K (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#00CC96"},
                "steps": [
                    {"range": [0, 40], "color": "#FFCDD2"},
                    {"range": [40, 70], "color": "#FFF9C4"},
                    {"range": [70, 100], "color": "#C8E6C9"}
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 70
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

else:
    # Default view — popularity rankings
    st.markdown("#### 🌟 Top Rated Room Types in Sri Lanka")
    st.markdown("Set your preferences above and click **Get Recommendations** for personalised results.")

    with st.spinner("Loading popularity rankings..."):
        pop_df = load_popularity_rankings()

    fig = px.bar(
        pop_df,
        x="avg_stars",
        y="roomType",
        orientation="h",
        color="avg_stars",
        color_continuous_scale="Viridis",
        title="Average Star Rating by Room Type",
        labels={"avg_stars": "Average Stars", "roomType": "Room Type"}
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(
            pop_df,
            x="listing_count",
            y="avg_stars",
            size="listing_count",
            hover_name="roomType",
            color="avg_stars",
            color_continuous_scale="Teal",
            title="Listing Count vs Average Stars by Room Type"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            pop_df,
            x="roomType",
            y="listing_count",
            color="listing_count",
            color_continuous_scale="Blues",
            title="Number of Listings per Room Type"
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

st.divider()
st.markdown("""
<div style='text-align: center; color: grey; font-size: 13px;'>
    Hybrid Recommendation System powered by Apache Spark · Content-Based Filtering · Popularity Ranking · Precision@K Evaluation
</div>
""", unsafe_allow_html=True)
