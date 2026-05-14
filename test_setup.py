# Test 1 — Python version
import sys
print(f"Python: {sys.version}")

# Test 2 — PySpark
from pyspark.sql import SparkSession
spark = SparkSession.builder \
    .appName("SetupTest") \
    .master("local[*]") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
print(f"PySpark: working ✅")

# Test 3 — Load both datasets
reviews = spark.read.csv("data/reviews_final.csv", header=True, inferSchema=True)
airbnb = spark.read.csv("data/airbnb_listings.csv", header=True, inferSchema=True)
print(f"Reviews dataset: {reviews.count()} rows ✅")
print(f"Airbnb dataset: {airbnb.count()} rows ✅")

# Test 4 — Other libraries
import streamlit
import pandas
import matplotlib
import seaborn
import plotly
import sklearn
print(f"Streamlit: {streamlit.__version__} ✅")
print(f"All libraries loaded ✅")

spark.stop()
print("\n✅ Setup complete — ready to build!")