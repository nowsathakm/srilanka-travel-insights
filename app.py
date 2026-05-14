# app.py
# ─────────────────────────────────────────────
# Sri Lanka Travel Insights
# Main Streamlit Application Entry Point
#
# This is the home page of the app.
# Streamlit automatically creates navigation
# from the pages/ folder.
# ─────────────────────────────────────────────

import streamlit as st
from utils.spark_session import get_spark_session
from utils.part_a_analytics import get_dataset_overview
from utils.part_b_recommender import get_airbnb_overview

# ─────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Sri Lanka Travel Insights",
    page_icon="🇱🇰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# INITIALISE SPARK SESSION
# Store in session state so it persists across
# all pages without restarting
# ─────────────────────────────────────────────

@st.cache_resource
def init_spark():
    """
    Initialises Spark once and caches it.
    @st.cache_resource means Spark starts only
    once even when user navigates between pages.
    """
    return get_spark_session()

spark = init_spark()

# ─────────────────────────────────────────────
# HOME PAGE CONTENT
# ─────────────────────────────────────────────

# Header
st.title("🇱🇰 Sri Lanka Travel Insights")
st.subheader("Big Data Analytics & Recommendation System")
st.markdown("""
A comprehensive big data analytics platform built with **Apache Spark** and **PySpark**,
analyzing Sri Lankan tourist destinations and providing personalized accommodation recommendations.
""")

st.divider()

# ─────────────────────────────────────────────
# LIVE DATASET STATS
# ─────────────────────────────────────────────

st.subheader("📊 Dataset Overview")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🗺️ Destination Reviews Dataset")
    with st.spinner("Loading analytics..."):
        try:
            overview_a = get_dataset_overview(spark)
            m1, m2 = st.columns(2)
            m3, m4 = st.columns(2)
            m1.metric("Total Reviews", f"{overview_a['total_reviews']:,}")
            m2.metric("Destinations", f"{overview_a['total_destinations']:,}")
            m3.metric("Districts", f"{overview_a['total_districts']:,}")
            m4.metric("Time Periods", f"{overview_a['total_timespans']:,}")
        except Exception as e:
            st.error(f"Error loading reviews data: {e}")

with col2:
    st.markdown("#### 🏠 Airbnb Accommodations Dataset")
    with st.spinner("Loading accommodations..."):
        try:
            overview_b = get_airbnb_overview(spark)
            m1, m2 = st.columns(2)
            m3, m4 = st.columns(2)
            m1.metric("Total Listings", f"{overview_b['total_listings']:,}")
            m2.metric("Room Types", f"{overview_b['unique_room_types']:,}")
            m3.metric("Average Stars", f"{overview_b['avg_stars']} ⭐")
            m4.metric("Max Guests", f"{overview_b['max_guests']} 👥")
        except Exception as e:
            st.error(f"Error loading Airbnb data: {e}")

st.divider()

# ─────────────────────────────────────────────
# PROJECT DESCRIPTION
# ─────────────────────────────────────────────

st.subheader("🎯 About This Project")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    #### Part A — Big Data Analytics
    Analyses **35,000+ tourist destination reviews**
    across Sri Lanka using Apache Spark to uncover:
    - 🏆 Most popular destinations by district
    - 📈 Tourism trends over time
    - 🔑 Key travel keywords via TF-IDF
    - 🗂️ Destination clusters via K-Means
    """)

with col2:
    st.markdown("""
    #### Part B — Recommendation System
    A **hybrid recommendation engine** for Sri Lankan
    Airbnb accommodations combining:
    - 🎯 Content-based filtering (room type + guests)
    - ⭐ Popularity-based ranking (star ratings)
    - 📐 Hybrid scoring (70% quality + 30% relevance)
    - 📊 Precision@K evaluation
    """)

st.divider()

# ─────────────────────────────────────────────
# TECH STACK
# ─────────────────────────────────────────────

st.subheader("🛠️ Technology Stack")

col1, col2, col3, col4 = st.columns(4)
col1.info("⚡ Apache Spark\nPySpark 3.x\nDistributed Processing")
col2.info("🤖 Spark MLlib\nTF-IDF · K-Means\nFeature Engineering")
col3.info("🖥️ Streamlit\nInteractive UI\nReal-time Analytics")
col4.info("🐍 Python\nPandas · Plotly\nMatplotlib · Seaborn")

st.divider()

# ─────────────────────────────────────────────
# NAVIGATION GUIDE
# ─────────────────────────────────────────────

st.subheader("🧭 How to Navigate")

col1, col2, col3 = st.columns(3)

with col1:
    st.success("""
    **📊 Analytics Dashboard**
    Explore data cleaning,
    EDA, TF-IDF keywords,
    and K-Means clusters
    """)

with col2:
    st.success("""
    **🗺️ Destination Recommender**
    Find top Sri Lankan
    destinations based on
    popularity and keywords
    """)

with col3:
    st.success("""
    **🏠 Accommodation Recommender**
    Get personalised hotel
    recommendations based on
    your group size and preferences
    """)

st.divider()

# Footer
st.markdown("""
<div style='text-align: center; color: grey; font-size: 13px;'>
    Built with Apache Spark · PySpark · Streamlit &nbsp;|&nbsp;
    Sri Lanka Tourism Analytics Project &nbsp;|&nbsp;
    Big Data Analytics Assignment
</div>
""", unsafe_allow_html=True)