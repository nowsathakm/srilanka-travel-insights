# 🇱🇰 Sri Lanka Travel Insights

Big data analytics and recommendation system for Sri Lanka tourism using Apache Spark and PySpark. Built with Streamlit for interactive visualization.

---

## 📋 Project Structure

```
srilanka-travel-insights/
├── app.py                                    ← Home page
├── requirements.txt                          ← Dependencies
├── README.md                                 ← This file
├── data/
│   ├── reviews_raw.csv                       ← Raw reviews (cleaning demo)
│   ├── reviews_final.csv                     ← Cleaned reviews (analysis)
│   └── airbnb_listings.csv                   ← Airbnb accommodations
├── utils/
│   ├── spark_session.py                      ← Spark setup
│   ├── part_a_analytics.py                   ← Spark analytics functions
│   └── part_b_recommender.py                 ← Recommendation functions
└── pages/
    ├── 1_📊_Analytics_Dashboard.py           ← Part A dashboard
    ├── 2_🗺️_Destination_Recommender.py       ← Destination finder
    └── 3_🏠_Accommodation_Recommender.py     ← Part B recommender
```

---

## 🚀 How to Run Locally

### Prerequisites
- Python 3.9+
- Java 11 or 17 (required for PySpark)
- Mac / Linux / Windows

### Step 1 — Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/srilanka-travel-insights.git
cd srilanka-travel-insights
```

### Step 2 — Create and activate virtual environment
```bash
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Set JAVA_HOME (Mac/Linux)
```bash
export JAVA_HOME=$(/usr/libexec/java_home)
```

### Step 5 — Add datasets
Place these files in the `data/` folder:
- `reviews_raw.csv`
- `reviews_final.csv`
- `airbnb_listings.csv`

### Step 6 — Run the app
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 📊 Features

### Part A — Big Data Analytics
- Data cleaning comparison (raw vs cleaned)
- Top destinations by review volume
- District-wise tourism analysis
- Review trends over time
- Feature engineering — popularity score
- TF-IDF keyword analysis (Spark MLlib)
- K-Means destination clustering (Spark MLlib)

### Part B — Recommendation System
- Content-based filtering (room type + guests)
- Popularity-based ranking (star ratings)
- Hybrid recommendation (70% quality + 30% relevance)
- Precision@K evaluation metric

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Apache Spark / PySpark | Big data processing |
| Spark MLlib | TF-IDF, K-Means, feature engineering |
| Streamlit | Interactive web UI |
| Plotly | Interactive charts |
| Pandas | Data manipulation |
| WordCloud | Keyword visualisation |

---

## 📁 Datasets

- **Destination Reviews** — 35,434 Sri Lankan tourist destination reviews
  - Source: https://www.kaggle.com/datasets/nethumdperera/travel-destinations-reviews-in-sir-lanka

- **Airbnb Listings** — 5,000 Sri Lankan Airbnb accommodation listings
  - Source: https://www.kaggle.com/datasets/kanchana1990/destination-sri-lanka/data

---
