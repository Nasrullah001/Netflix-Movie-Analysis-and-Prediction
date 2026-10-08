# Netflix Content Analysis, Recommendation System & Trend Forecasting

An end-to-end data science project analyzing Netflix content patterns, building a content-based recommendation engine, and forecasting content growth trends using historical data.

## Project Overview

This repository contains four key data science workflows applied to the Netflix Movies & TV Shows dataset:
- **Task 1: Data Cleaning & Preprocessing** — Handling missing values, standardizing categorical text, and parsing dates/durations.
- **Task 2: Exploratory Data Analysis (EDA)** — Visualizing content type distributions, top producing countries, and annual additions.
- **Task 3: Recommendation System** — Building a TF-IDF and Cosine Similarity content-based recommendation engine.
- **Task 4: Trend Prediction Analysis** — Forecasting future content growth trends using time-series linear regression.

## Repository Structure

```text
├── data/
│   ├── raw/                  # Original raw dataset
│   └── processed/            # Cleaned CSV output
├── notebooks/
│   ├── 01_data_cleaning_eda.ipynb
│   ├── 02_recommendation_system.ipynb
│   └── 03_trend_prediction.ipynb
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation