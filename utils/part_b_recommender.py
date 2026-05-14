# utils/part_b_recommender.py
# ─────────────────────────────────────────────
# Part B: Recommendation System Using Big Data
# This file contains all recommendation logic for
# the Sri Lanka Airbnb accommodation dataset.
#
# Approach: Hybrid Recommendation System
# combining Content-Based Filtering +
# Popularity-Based Ranking
#
# Why hybrid?
# - Content-based: matches user preferences
#   (room type + guest capacity)
# - Popularity-based: ranks by real star ratings
# - Together: personalised + quality assured
# ─────────────────────────────────────────────

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer, VectorAssembler, Normalizer
from pyspark.ml import Pipeline


# ─────────────────────────────────────────────
# 1. LOAD & PREPROCESS AIRBNB DATA
# ─────────────────────────────────────────────

def load_airbnb(spark: SparkSession):
    """
    Loads the Airbnb listings dataset and performs
    initial preprocessing:
    - Strips BOM and whitespace from column names
    - Reads everything as string first for safe casting
    - Handles empty strings, nulls, and dirty values
    - Returns only rows with valid stars and guest count
    """
    df = spark.read.csv(
        "data/airbnb_listings.csv",
        header=True,
        inferSchema=False  # Read everything as string first
    )

    # Clean BOM and whitespace from column names
    for col_name in df.columns:
        clean_name = col_name.strip() \
            .replace('\ufeff', '') \
            .replace('"', '')
        df = df.withColumnRenamed(col_name, clean_name)

    # Trim whitespace from all values
    for col_name in df.columns:
        df = df.withColumn(col_name, F.trim(F.col(col_name)))

    # Safely cast stars — empty string or text becomes null
    df = df.withColumn(
        "stars",
        F.when(
            F.col("stars").rlike(r"^\d+(\.\d+)?$"),
            F.col("stars").cast("float")
        ).otherwise(F.lit(None).cast("float"))
    )

    # Safely cast numberOfGuests — extract digits only
    df = df.withColumn(
        "numberOfGuests",
        F.when(
            F.regexp_extract(F.col("numberOfGuests"), r"(\d+)", 1) != "",
            F.regexp_extract(F.col("numberOfGuests"), r"(\d+)", 1).cast("integer")
        ).otherwise(F.lit(None).cast("integer"))
    )

    # Keep only rows where both stars and guests are valid
    df = df.filter(
        F.col("stars").isNotNull() &
        F.col("numberOfGuests").isNotNull() &
        (F.col("numberOfGuests") > 0)
    )

    return df


def get_airbnb_overview(spark: SparkSession):
    """
    Returns high level stats about the Airbnb dataset
    for display on the accommodation page header.
    """
    df = load_airbnb(spark)

    return {
        "total_listings": df.count(),
        "unique_room_types": df.select("roomType").distinct().count(),
        "avg_stars": round(df.agg(F.avg("stars")).collect()[0][0], 2),
        "max_guests": df.agg(F.max("numberOfGuests")).collect()[0][0]
    }


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_features(spark: SparkSession):
    """
    Engineers features for the recommendation model:

    1. roomType → numeric index using StringIndexer
       (converts categories like 'Entire villa' → 0.0)

    2. numberOfGuests → kept as numeric

    3. stars → normalised 0-1 for fair comparison

    4. guest_bucket → groups guests into buckets:
       Solo (1), Couple (2), Small Group (3-4),
       Large Group (5+)

    5. feature_vector → combines all features into
       one vector for similarity calculation
    """
    df = load_airbnb(spark)

    # Engineer guest bucket feature
    df = df.withColumn(
        "guest_bucket",
        F.when(F.col("numberOfGuests") == 1, 1)
         .when(F.col("numberOfGuests") == 2, 2)
         .when(F.col("numberOfGuests").between(3, 4), 3)
         .when(F.col("numberOfGuests") >= 5, 4)
         .otherwise(1)
    )

    # Normalise stars to 0-1 range
    stats = df.agg(
        F.min("stars").alias("min_stars"),
        F.max("stars").alias("max_stars")
    ).collect()[0]

    df = df.withColumn(
        "stars_normalised",
        (F.col("stars") - stats["min_stars"]) /
        (stats["max_stars"] - stats["min_stars"])
    )

    # StringIndexer — converts roomType string to number
    indexer = StringIndexer(
        inputCol="roomType",
        outputCol="roomType_index",
        handleInvalid="keep"
    )

    # VectorAssembler — combines features into one vector
    assembler = VectorAssembler(
        inputCols=["roomType_index", "guest_bucket", "stars_normalised"],
        outputCol="feature_vector"
    )

    # Normalizer — scales vector to unit length
    # This ensures fair cosine similarity calculation
    normalizer = Normalizer(
        inputCol="feature_vector",
        outputCol="normalised_features",
        p=2.0
    )

    # Build and run pipeline
    pipeline = Pipeline(stages=[indexer, assembler, normalizer])
    model = pipeline.fit(df)
    featured_df = model.transform(df)

    return featured_df, model


# ─────────────────────────────────────────────
# 3. CONTENT-BASED FILTERING
# ─────────────────────────────────────────────

def get_content_based_recommendations(
    spark: SparkSession,
    preferred_room_type: str,
    num_guests: int,
    top_n: int = 10
):
    """
    Content-Based Filtering:
    Recommends accommodations that match the user's
    stated preferences for room type and group size.

    How it works:
    1. Filter listings matching room type preference
    2. Filter listings with sufficient guest capacity
    3. Rank by star rating (popularity score)
    4. Return top N recommendations

    This is a valid content-based recommendation
    approach as per assignment guidelines.
    """
    df = load_airbnb(spark)

    # Filter by room type preference
    if preferred_room_type != "Any":
        filtered = df.filter(
            F.lower(F.col("roomType")).contains(
                preferred_room_type.lower()
            )
        )
    else:
        filtered = df

    # Filter by guest capacity
    filtered = filtered.filter(
        F.col("numberOfGuests") >= num_guests
    )

    # Rank by star rating — popularity scoring
    recommendations = filtered \
        .orderBy(F.desc("stars"), F.desc("numberOfGuests")) \
        .limit(top_n)

    return recommendations.select(
        "name", "roomType", "numberOfGuests", "stars"
    ).toPandas()


# ─────────────────────────────────────────────
# 4. POPULARITY-BASED RANKING
# ─────────────────────────────────────────────

def get_popularity_rankings(spark: SparkSession, top_n: int = 10):
    """
    Popularity-Based Recommendation:
    Ranks all accommodations purely by star rating.
    Used as a baseline and as part of hybrid scoring.

    This answers: "What are the best rated
    accommodations in Sri Lanka overall?"
    """
    df = load_airbnb(spark)

    # Group by room type and get average stars
    result = df.groupBy("roomType") \
        .agg(
            F.round(F.avg("stars"), 2).alias("avg_stars"),
            F.count("*").alias("listing_count"),
            F.round(F.max("stars"), 2).alias("max_stars")
        ) \
        .orderBy(F.desc("avg_stars")) \
        .limit(top_n)

    return result.toPandas()


# ─────────────────────────────────────────────
# 5. HYBRID RECOMMENDATION
# ─────────────────────────────────────────────

def get_hybrid_recommendations(
    spark: SparkSession,
    preferred_room_type: str,
    num_guests: int,
    min_stars: float = 4.0,
    top_n: int = 10
):
    """
    Hybrid Recommendation System:
    Combines content-based filtering with
    popularity-based ranking into one hybrid score.

    Hybrid Score Formula:
    hybrid_score = (stars_normalised * 0.7) +
                   (guest_match_score * 0.3)

    - 70% weight on star rating (quality)
    - 30% weight on guest capacity match (relevance)

    This is the main recommendation function used
    in the Streamlit UI.
    """
    df = load_airbnb(spark)

    # Step 1 — Apply content filter
    if preferred_room_type != "Any":
        filtered = df.filter(
            F.lower(F.col("roomType")).contains(
                preferred_room_type.lower()
            )
        )
    else:
        filtered = df

    # Step 2 — Filter by minimum stars
    filtered = filtered.filter(F.col("stars") >= min_stars)

    # Step 3 — Filter by guest capacity
    filtered = filtered.filter(
        F.col("numberOfGuests") >= num_guests
    )

    # Step 4 — Normalise stars for scoring
    stats = filtered.agg(
        F.min("stars").alias("min_s"),
        F.max("stars").alias("max_s")
    ).collect()[0]

    min_s = stats["min_s"] if stats["min_s"] else 0
    max_s = stats["max_s"] if stats["max_s"] else 5

    filtered = filtered.withColumn(
        "stars_norm",
        (F.col("stars") - min_s) / (max_s - min_s + 0.001)
    )

    # Step 5 — Guest match score
    # Closer to requested number = higher score
    filtered = filtered.withColumn(
        "guest_match_score",
        F.when(
            F.col("numberOfGuests") == num_guests, 1.0
        ).when(
            F.col("numberOfGuests") <= num_guests + 2, 0.8
        ).otherwise(0.5)
    )

    # Step 6 — Calculate hybrid score
    filtered = filtered.withColumn(
        "hybrid_score",
        F.round(
            (F.col("stars_norm") * 0.7) +
            (F.col("guest_match_score") * 0.3), 4
        )
    )

    # Step 7 — Return top N by hybrid score
    result = filtered \
        .orderBy(F.desc("hybrid_score")) \
        .limit(top_n) \
        .select(
            "name",
            "roomType",
            "numberOfGuests",
            "stars",
            "hybrid_score"
        )

    return result.toPandas()


# ─────────────────────────────────────────────
# 6. RECOMMENDATION EVALUATION
# ─────────────────────────────────────────────

def evaluate_recommendations(
    spark: SparkSession,
    preferred_room_type: str,
    num_guests: int,
    top_n: int = 10
):
    """
    Evaluates the recommendation system using
    Precision@K metric.

    Precision@K = (relevant recommendations in top K)
                  / K

    A recommendation is considered RELEVANT if:
    - Room type matches preference AND
    - Guest capacity >= requested guests AND
    - Stars >= 4.0

    This is the standard evaluation approach for
    content-based recommendation systems where
    no user-item matrix exists.
    """
    recommendations = get_hybrid_recommendations(
        spark, preferred_room_type, num_guests, top_n=top_n
    )

    if recommendations.empty:
        return {
            "precision_at_k": 0,
            "total_recommended": 0,
            "relevant_count": 0,
            "coverage": 0
        }

    # Count relevant recommendations
    relevant = recommendations[
        (recommendations["stars"] >= 4.0) &
        (recommendations["numberOfGuests"] >= num_guests)
    ]

    precision_at_k = round(len(relevant) / len(recommendations), 2)

    # Coverage — what % of total listings were considered
    df = load_airbnb(spark)
    total_listings = df.count()
    coverage = round(len(recommendations) / total_listings * 100, 2)

    return {
        "precision_at_k": precision_at_k,
        "total_recommended": len(recommendations),
        "relevant_count": len(relevant),
        "coverage": coverage
    }


# ─────────────────────────────────────────────
# 7. AVAILABLE FILTER OPTIONS
# ─────────────────────────────────────────────

def get_room_types(spark: SparkSession):
    """
    Returns all unique room types for the
    Streamlit dropdown filter.
    """
    df = load_airbnb(spark)
    types = df.select("roomType") \
        .distinct() \
        .orderBy("roomType") \
        .rdd.flatMap(lambda x: x) \
        .collect()
    return ["Any"] + types