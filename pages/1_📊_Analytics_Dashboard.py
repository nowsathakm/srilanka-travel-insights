# pages/1_📊_Analytics_Dashboard.py
# ─────────────────────────────────────────────
# Part A: Big Data Analytics Dashboard
# Shows all Spark analytics results visually
# ─────────────────────────────────────────────

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from utils.spark_session import get_spark_session
from utils.part_a_analytics import (
    get_cleaning_comparison,
    get_top_destinations,
    get_district_distribution,
    get_review_trends,
    get_district_destination_breakdown,
    engineer_popularity_score,
    get_tfidf_keywords,
    cluster_destinations
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Analytics Dashboard",
    page_icon="📊",
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
# PAGE HEADER
# ─────────────────────────────────────────────

st.title("📊 Analytics Dashboard")
st.markdown("**Part A — Big Data Analytics Using Apache Spark**")
st.markdown("Exploring 35,000+ Sri Lankan tourist destination reviews using PySpark.")
st.divider()

# ─────────────────────────────────────────────
# SECTION 1 — DATA CLEANING COMPARISON
# ─────────────────────────────────────────────

st.subheader("🧹 Step 1 — Data Cleaning")
st.markdown("Comparing the raw dataset against the cleaned dataset to demonstrate preprocessing.")

with st.spinner("Running data cleaning comparison on Spark..."):
    cleaning_df = get_cleaning_comparison(spark)

col1, col2 = st.columns(2)

with col1:
    st.dataframe(cleaning_df, use_container_width=True, hide_index=True)

with col2:
    fig = go.Figure(data=[
        go.Bar(
            name="Raw Dataset",
            x=cleaning_df["Metric"],
            y=cleaning_df["Raw Dataset"],
            marker_color="#EF553B"
        ),
        go.Bar(
            name="Clean Dataset",
            x=cleaning_df["Metric"],
            y=cleaning_df["Clean Dataset"],
            marker_color="#00CC96"
        )
    ])
    fig.update_layout(
        barmode="group",
        title="Raw vs Cleaned Dataset Comparison",
        xaxis_title="Metric",
        yaxis_title="Count",
        legend=dict(orientation="h", y=-0.2)
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────
# SECTION 2 — TOP DESTINATIONS
# ─────────────────────────────────────────────

st.subheader("🏆 Step 2 — Top Destinations by Review Volume")
st.markdown("Using **Spark groupBy + orderBy** to rank destinations by number of reviews.")

with st.spinner("Aggregating destination data on Spark..."):
    top_n = st.slider("Number of destinations to show", 5, 30, 15)
    top_dest_df = get_top_destinations(spark, top_n)

fig = px.bar(
    top_dest_df,
    x="Review Count",
    y="Destination",
    orientation="h",
    color="Review Count",
    color_continuous_scale="Teal",
    title=f"Top {top_n} Sri Lankan Destinations by Review Count"
)
fig.update_layout(yaxis=dict(autorange="reversed"))
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────
# SECTION 3 — DISTRICT ANALYSIS
# ─────────────────────────────────────────────

st.subheader("🗺️ Step 3 — District-wise Tourism Analysis")
st.markdown("Identifying which Sri Lankan districts attract the most tourist reviews.")

col1, col2 = st.columns(2)

with col1:
    with st.spinner("Analysing district distribution..."):
        district_df = get_district_distribution(spark)

    fig = px.pie(
        district_df,
        values="Review Count",
        names="District",
        title="Review Distribution by District",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    with st.spinner("Counting unique destinations per district..."):
        breakdown_df = get_district_destination_breakdown(spark)

    fig = px.bar(
        breakdown_df,
        x="District",
        y="Unique Destinations",
        color="Unique Destinations",
        color_continuous_scale="Viridis",
        title="Unique Destinations per District"
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────
# SECTION 4 — REVIEW TRENDS OVER TIME
# ─────────────────────────────────────────────

st.subheader("📈 Step 4 — Review Trends Over Time")
st.markdown("Analysing tourism activity patterns using the **Timespan** column.")

with st.spinner("Analysing review trends on Spark..."):
    trends_df = get_review_trends(spark)

fig = px.line(
    trends_df,
    x="Timespan",
    y="Review Count",
    markers=True,
    title="Tourist Review Volume Over Time",
    color_discrete_sequence=["#636EFA"]
)
fig.update_layout(
    xaxis_title="Time Period",
    yaxis_title="Number of Reviews",
    xaxis_tickangle=-45
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────
# SECTION 5 — FEATURE ENGINEERING
# ─────────────────────────────────────────────

st.subheader("⚙️ Step 5 — Feature Engineering — Popularity Score")
st.markdown("""
Engineering a **Popularity Score (0–100)** for each destination using:
- Review count normalised between 0 and 100
- District ranking using **Spark Window Functions**
""")

with st.spinner("Engineering popularity scores on Spark..."):
    popularity_df = engineer_popularity_score(spark)

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Top 15 Destinations by Popularity Score")
    st.dataframe(
        popularity_df[["Destination", "District", "review_count", "popularity_score", "district_rank"]]
        .head(15),
        use_container_width=True,
        hide_index=True
    )

with col2:
    fig = px.scatter(
        popularity_df.head(50),
        x="review_count",
        y="popularity_score",
        color="District",
        hover_name="Destination",
        title="Review Count vs Popularity Score",
        labels={
            "review_count": "Review Count",
            "popularity_score": "Popularity Score (0-100)"
        }
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────
# SECTION 6 — TF-IDF KEYWORD ANALYSIS
# ─────────────────────────────────────────────

st.subheader("🔑 Step 6 — TF-IDF Keyword Analysis (Spark MLlib)")
st.markdown("""
Applying **TF-IDF via Spark MLlib Pipeline** to extract the most meaningful
keywords from 35,000+ tourist reviews.
""")

with st.spinner("Running TF-IDF pipeline on Spark MLlib..."):
    keywords_df = get_tfidf_keywords(spark, top_n=100)

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Top 20 Keywords")
    st.dataframe(
        keywords_df.head(20),
        use_container_width=True,
        hide_index=True
    )

with col2:
    st.markdown("#### Keyword Word Cloud")
    word_freq = dict(zip(
        keywords_df["word"],
        keywords_df["frequency"]
    ))
    wordcloud = WordCloud(
        width=600,
        height=400,
        background_color="black",
        colormap="cool",
        max_words=80
    ).generate_from_frequencies(word_freq)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

st.divider()

# ─────────────────────────────────────────────
# SECTION 7 — K-MEANS CLUSTERING
# ─────────────────────────────────────────────

st.subheader("🗂️ Step 7 — K-Means Destination Clustering (Spark MLlib)")
st.markdown("""
Clustering Sri Lankan destinations into groups using **K-Means from Spark MLlib**
based on review volume and popularity score.
""")

with st.spinner("Running K-Means clustering on Spark MLlib..."):
    k = st.slider("Number of clusters (K)", 2, 6, 4)
    clustered_df = cluster_destinations(spark, k=k)

col1, col2 = st.columns(2)

with col1:
    fig = px.scatter(
        clustered_df,
        x="review_count",
        y="popularity_score",
        color="cluster_label",
        hover_name="Destination",
        title=f"K-Means Clustering (K={k})",
        labels={
            "review_count": "Review Count",
            "popularity_score": "Popularity Score"
        },
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("#### Cluster Summary")
    cluster_summary = clustered_df.groupby("cluster_label").agg(
        Destinations=("Destination", "count"),
        Avg_Popularity=("popularity_score", "mean"),
        Avg_Reviews=("review_count", "mean")
    ).round(2).reset_index()
    cluster_summary.columns = ["Cluster", "Destinations", "Avg Popularity", "Avg Reviews"]
    st.dataframe(cluster_summary, use_container_width=True, hide_index=True)

    st.markdown("#### Sample Destinations per Cluster")
    for label in clustered_df["cluster_label"].unique():
        with st.expander(f"{label}"):
            sample = clustered_df[clustered_df["cluster_label"] == label][
                ["Destination", "District", "review_count", "popularity_score"]
            ].head(5)
            st.dataframe(sample, use_container_width=True, hide_index=True)

st.divider()

# Footer
st.markdown("""
<div style='text-align: center; color: grey; font-size: 13px;'>
    Part A Complete — All analytics powered by Apache Spark & Spark MLlib
</div>
""", unsafe_allow_html=True)