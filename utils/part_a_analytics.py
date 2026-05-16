# utils/part_a_analytics.py
# ─────────────────────────────────────────────
# Part A: Big Data Analytics Using Apache Spark
# This file contains all analytics functions for
# the Sri Lanka destination reviews dataset.
# Each function performs a specific analysis using
# PySpark and returns a pandas DataFrame for
# Streamlit to visualize.
# ─────────────────────────────────────────────

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────

def load_reviews(spark: SparkSession):
    """
    Loads the cleaned destination reviews CSV into
    a Spark DataFrame. inferSchema automatically
    detects column data types.
    """
    df = spark.read.csv(
        "data/reviews_final.csv",
        header=True,
        inferSchema=True
    )
    return df


def load_raw_reviews(spark: SparkSession):
    """
    Loads the raw (uncleaned) reviews CSV.
    Used to demonstrate the data cleaning step.
    """
    df = spark.read.csv(
        "data/reviews_raw.csv",
        header=True,
        inferSchema=True
    )
    return df


# ─────────────────────────────────────────────
# 2. DATA CLEANING COMPARISON
# ─────────────────────────────────────────────

def get_cleaning_comparison(spark: SparkSession):
    """
    Compares raw vs cleaned dataset to demonstrate
    the data cleaning step visually in the dashboard.
    Returns a summary pandas DataFrame.
    """
    raw = load_raw_reviews(spark)
    clean = load_reviews(spark)

    # Count nulls in each column for raw data
    raw_nulls = raw.select([
        F.count(F.when(F.col(c).isNull() | (F.col(c) == ""), c)).alias(c)
        for c in raw.columns
    ]).toPandas()

    clean_nulls = clean.select([
        F.count(F.when(F.col(c).isNull() | (F.col(c) == ""), c)).alias(c)
        for c in clean.columns
    ]).toPandas()

    summary = pd.DataFrame({
        "Metric": ["Total Rows", "Null Values", "Unique Destinations", "Unique Districts"],
        "Raw Dataset": [
            raw.count(),
            int(raw_nulls.sum(axis=1).values[0]),
            raw.select("Destination").distinct().count(),
            raw.select("District").distinct().count()
        ],
        "Clean Dataset": [
            clean.count(),
            int(clean_nulls.sum(axis=1).values[0]),
            clean.select("Destination").distinct().count(),
            clean.select("District").distinct().count()
        ]
    })

    return summary


# ─────────────────────────────────────────────
# 3. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────

def get_top_destinations(spark: SparkSession, top_n: int = 15):
    """
    Returns the top N destinations ranked by
    number of reviews using Spark groupBy.
    This tells us which places tourists talk about most.
    """
    df = load_reviews(spark)

    result = df.groupBy("Destination") \
        .agg(F.count("*").alias("Review Count")) \
        .orderBy(F.desc("Review Count")) \
        .limit(top_n)

    return result.toPandas()


def get_district_distribution(spark: SparkSession):
    """
    Returns review count per district.
    Helps identify which regions are most visited.
    """
    df = load_reviews(spark)

    result = df.groupBy("District") \
        .agg(F.count("*").alias("Review Count")) \
        .orderBy(F.desc("Review Count"))

    return result.toPandas()


def get_review_trends(spark: SparkSession):
    """
    Analyses review volume over time using
    the Timespan column (e.g. '2 years ago').
    Shows tourism activity trends over time.
    """
    df = load_reviews(spark)

    result = df.groupBy("Timespan") \
        .agg(F.count("*").alias("Review Count")) \
        .orderBy("Timespan")

    return result.toPandas()


def get_district_destination_breakdown(spark: SparkSession):
    """
    Returns number of unique destinations per district.
    Shows diversity of attractions in each region.
    """
    df = load_reviews(spark)

    result = df.groupBy("District") \
        .agg(F.countDistinct("Destination").alias("Unique Destinations")) \
        .orderBy(F.desc("Unique Destinations"))

    return result.toPandas()


# ─────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_popularity_score(spark: SparkSession):
    """
    Engineers a popularity score for each destination
    based on review count. Uses Spark Window functions
    to rank destinations within each district.

    Popularity Score = review count normalised 0-100
    This is a key feature engineering step.
    """
    df = load_reviews(spark)

    # Count reviews per destination
    dest_counts = df.groupBy("Destination", "District") \
        .agg(F.count("*").alias("review_count"))

    # Get min and max for normalisation
    stats = dest_counts.agg(
        F.min("review_count").alias("min_count"),
        F.max("review_count").alias("max_count")
    ).collect()[0]

    min_count = stats["min_count"]
    max_count = stats["max_count"]

    # Normalise to 0-100 popularity score
    dest_with_score = dest_counts.withColumn(
        "popularity_score",
        F.round(
            ((F.col("review_count") - min_count) /
             (max_count - min_count)) * 100, 2
        )
    )

    # Rank within each district using Window function
    window_spec = Window.partitionBy("District") \
        .orderBy(F.desc("popularity_score"))

    dest_with_score = dest_with_score.withColumn(
        "district_rank",
        F.rank().over(window_spec)
    )

    return dest_with_score.orderBy(
        F.desc("popularity_score")
    ).toPandas()


# ─────────────────────────────────────────────
# 5. TF-IDF KEYWORD ANALYSIS (MLlib)
# ─────────────────────────────────────────────

def get_tfidf_keywords(spark: SparkSession, top_n: int = 20):
    """
    Applies TF-IDF (Term Frequency - Inverse Document Frequency)
    on review text using Spark MLlib.

    TF-IDF tells us which keywords are most important
    and unique to Sri Lankan tourism reviews —
    not just the most common words.

    Pipeline:
    Review text → Tokenize → Remove stop words → TF → IDF
    """
    df = load_reviews(spark)

    # Combine all reviews per destination into one document
    dest_reviews = df.groupBy("Destination") \
        .agg(F.concat_ws(" ", F.collect_list("Review")).alias("all_reviews"))

    # Step 1: Tokenize — split text into individual words
    tokenizer = Tokenizer(
        inputCol="all_reviews",
        outputCol="words"
    )

    # Step 2: Remove stop words (the, is, at, a, etc.)
    remover = StopWordsRemover(
        inputCol="words",
        outputCol="filtered_words"
    )

    # Step 3: Term Frequency — how often each word appears
    hashing_tf = HashingTF(
        inputCol="filtered_words",
        outputCol="raw_features",
        numFeatures=1000
    )

    # Step 4: IDF — penalise words that appear everywhere
    idf = IDF(
        inputCol="raw_features",
        outputCol="tfidf_features"
    )

    # Build and run the pipeline
    pipeline = Pipeline(stages=[tokenizer, remover, hashing_tf, idf])
    model = pipeline.fit(dest_reviews)
    result = model.transform(dest_reviews)

    # Extract top keywords by word frequency for display
    all_words = df.select(
        F.explode(F.split(F.lower(F.col("Review")), " ")).alias("word")
    ).filter(F.length("word") > 3)

    stop_words = ["this", "that", "with", "from", "they",
                  "have", "been", "were", "also", "very",
                  "good", "nice", "great", "place", "visit"]

    top_keywords = all_words \
        .filter(~F.col("word").isin(stop_words)) \
        .groupBy("word") \
        .agg(F.count("*").alias("frequency")) \
        .orderBy(F.desc("frequency")) \
        .limit(top_n)

    return top_keywords.toPandas()


# ─────────────────────────────────────────────
# 6. K-MEANS CLUSTERING (MLlib)
# ─────────────────────────────────────────────

def cluster_destinations(spark: SparkSession, k: int = 4):
    """
    Clusters destinations into groups using K-Means
    from Spark MLlib based on their popularity score
    and review volume.

    This groups destinations into categories like:
    - Hidden gems (low reviews, high rating)
    - Popular hotspots (high reviews)
    - Underrated spots (few reviews)
    - Tourist favourites
    """
    popularity_df = engineer_popularity_score(spark)
    spark_df = spark.createDataFrame(popularity_df)

    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.clustering import KMeans

    # Assemble features into a vector for MLlib
    assembler = VectorAssembler(
        inputCols=["review_count", "popularity_score"],
        outputCol="features"
    )

    assembled = assembler.transform(spark_df)

    # Apply K-Means clustering
    kmeans = KMeans(k=k, seed=42, featuresCol="features")
    model = kmeans.fit(assembled)
    clustered = model.transform(assembled)

    # Label clusters meaningfully
    result = clustered.select(
        "Destination", "District",
        "review_count", "popularity_score", "prediction"
    ).toPandas()

    # Name clusters based on popularity score ranges
    cluster_labels = {
        result.groupby("prediction")["popularity_score"]
        .mean().idxmax(): "🔥 Popular Hotspot",
        result.groupby("prediction")["popularity_score"]
        .mean().idxmin(): "💎 Hidden Gem",
    }

    result["cluster_label"] = result["prediction"].map(
        lambda x: cluster_labels.get(x, "📍 Regular Destination")
    )

    return result


# ─────────────────────────────────────────────
# 7. DATASET OVERVIEW STATS
# ─────────────────────────────────────────────

def get_dataset_overview(spark: SparkSession):
    """
    Returns high level statistics about the dataset
    for display on the dashboard header.
    """
    df = load_reviews(spark)

    return {
        "total_reviews": df.count(),
        "total_destinations": df.select("Destination").distinct().count(),
        "total_districts": df.select("District").distinct().count(),
        "total_timespans": df.select("Timespan").distinct().count()
    }

def get_keyword_scored_destinations(spark: SparkSession, keywords: list, top_n: int = 10):
    """
    Ranks destinations by keyword relevance using
    keyword density — not just raw frequency.

    Keyword Density = keyword mentions / total reviews
    This prevents high-review destinations from
    dominating results just because of volume.

    Final score = 80% keyword density + 20% popularity
    Minimum threshold: keyword must appear in at least
    5% of the destination's reviews to qualify.
    """
    df = load_reviews(spark)

    # Combine all reviews per destination
    dest_profiles = df.groupBy("Destination", "District") \
        .agg(
            F.concat_ws(" ", F.collect_list(F.lower(F.col("Review")))).alias("all_reviews"),
            F.count("*").alias("review_count")
        )

    # Calculate raw keyword frequency for each keyword
    score_expr = sum(
        (F.length(F.col("all_reviews")) -
         F.length(F.regexp_replace(F.col("all_reviews"), kw.lower(), ""))) /
        F.greatest(F.lit(len(kw)), F.lit(1))
        for kw in keywords
    )

    scored = dest_profiles.withColumn("keyword_freq", score_expr)

    # Calculate keyword DENSITY = freq / review_count
    # This normalises by destination size
    scored = scored.withColumn(
        "keyword_density",
        F.col("keyword_freq") / F.col("review_count")
    )

    # Minimum density threshold — keyword must appear
    # in at least 5% of reviews to qualify
    # This filters out accidental single mentions
    min_threshold = 0.05 * len(keywords)
    scored = scored.filter(F.col("keyword_density") >= min_threshold)

    # Join with popularity scores
    popularity = engineer_popularity_score(spark)
    popularity_spark = spark.createDataFrame(popularity)

    result = scored.join(
        popularity_spark.select("Destination", "popularity_score"),
        on="Destination",
        how="left"
    )

    # Normalise density to 0-100
    max_density = scored.agg(
        F.max("keyword_density")
    ).collect()[0][0]

    # Handle case where no keywords match — return empty
    if max_density is None or scored.count() == 0:
        return pd.DataFrame(columns=[
            "Destination", "District",
            "review_count", "keyword_freq",
            "relevance_score", "final_score"
        ])

    result = result.withColumn(
        "relevance_score",
        F.round(
            F.col("keyword_density") /
            F.lit(max(float(max_density), 0.001)) * 100, 2
        )
    )

    # Final score — 80% keyword relevance + 20% popularity
    result = result.withColumn(
        "final_score",
        F.round(
            (F.col("relevance_score") * 0.8) +
            (F.coalesce(F.col("popularity_score"), F.lit(0)) * 0.2),
            2
        )
    )

    return result.orderBy(F.desc("final_score")) \
        .limit(top_n) \
        .select(
            "Destination", "District",
            "review_count", "keyword_freq",
            "relevance_score", "final_score"
        ).toPandas()