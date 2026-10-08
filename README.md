# Netflix Content Analysis, Recommendation System & Trend Forecasting

An end-to-end data science project analyzing Netflix content patterns, building a content-based recommendation engine, and forecasting content growth trends using historical data.

## Project Overview

This repository contains four key data science workflows applied to the Netflix Movies & TV Shows dataset:
- **Task 1: Data Cleaning & Preprocessing** — Handling missing values, standardizing categorical text, and parsing dates/durations.
- **Task 2: Exploratory Data Analysis (EDA)** — Visualizing content type distributions, top producing countries, and annual additions.
- **Task 3: Recommendation System** — Building a TF-IDF and Cosine Similarity content-based recommendation engine.
- **Task 4: Trend Prediction Analysis** — Forecasting future content growth trends using time-series linear regression.

## Setup and run

Use Python 3.11 for a compatible environment, then install the dashboard and
notebook dependencies and start Streamlit:

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\python.exe -m pip install -r requirements.txt
.\.venv311\Scripts\python.exe -m streamlit run app.py
```

The dashboard reads `data/cleaned_netflix_data.csv` when available and falls
back to `data/netflix_titles.csv`. Run commands from the project root or use
the configured virtual environment; dataset paths are resolved relative to
`app.py`.

## Repository Structure

```text
├── app.py                    # Streamlit dashboard
├── data/
│   ├── netflix_titles.csv    # Original dataset
│   └── cleaned_netflix_data.csv
├── notebooks/
│   ├── data_cleaning_&_EDA.ipynb
│   ├── recommendation_system_analysis.ipynb
│   └── trend_prediction_analysis.ipynb
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation