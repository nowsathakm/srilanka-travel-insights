# utils/spark_session.py
# ─────────────────────────────────────────────
# This file creates a single shared Spark session
# that all other parts of the project will use.
# We use a singleton pattern so Spark starts once
# and stays running while the app is open.
# ─────────────────────────────────────────────

from pyspark.sql import SparkSession

def get_spark_session():
    """
    Creates or retrieves an existing Spark session.
    Using local[*] means Spark will use all available
    CPU cores on your Mac for processing.
    """
    spark = SparkSession.builder \
        .appName("SriLankaTravelInsights") \
        .master("local[*]") \
        .config("spark.driver.memory", "2g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.ui.showConsoleProgress", "false") \
        .getOrCreate()

    # Suppress noisy Spark logs — only show errors
    spark.sparkContext.setLogLevel("ERROR")

    return spark


def stop_spark_session():
    """
    Cleanly stops the Spark session.
    Called when the app shuts down.
    """
    spark = SparkSession.getActiveSession()
    if spark:
        spark.stop()