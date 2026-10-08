from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import linear_kernel

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

st.set_page_config(
    page_title="Netflix Intelligence Dashboard",
    page_icon="🎬",
    layout="wide",
)


@st.cache_data
def load_and_preprocess_data():
    cleaned_path = DATA_DIR / "cleaned_netflix_data.csv"
    raw_path = DATA_DIR / "netflix_titles.csv"
    dataset_path = cleaned_path if cleaned_path.is_file() else raw_path

    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"No Netflix dataset found in {DATA_DIR}. "
            "Expected cleaned_netflix_data.csv or netflix_titles.csv."
        )

    data = pd.read_csv(dataset_path)
    required_columns = {"title", "type", "release_year"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{dataset_path.name} is missing required columns: {missing}")

    defaults = {
        "title": "Unknown",
        "type": "Unknown",
        "director": "Unknown",
        "cast": "",
        "country": "Unknown",
        "rating": "Unknown",
        "listed_in": "",
        "description": "",
        "date_added": "",
    }
    for column, default in defaults.items():
        if column not in data.columns:
            data[column] = default
        data[column] = data[column].fillna(default)

    for column in defaults:
        data[column] = data[column].astype(str).str.strip()

    data["release_year"] = pd.to_numeric(data["release_year"], errors="coerce")
    data["date_added_clean"] = pd.to_datetime(data["date_added"], errors="coerce")
    data["year_added"] = data["date_added_clean"].dt.year
    data["month_added"] = data["date_added_clean"].dt.month_name()
    return data.reset_index(drop=True)


df = load_and_preprocess_data()

st.sidebar.title("🎬 Navigation")
page = st.sidebar.radio(
    "Select analysis module:",
    ["Overview & EDA", "Recommendation System", "Trend Prediction"],
)

if page == "Overview & EDA":
    st.title("📊 Netflix dataset overview & insights")
    st.write("Explore content distributions, top countries, and genre breakdown.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total titles", f"{len(df):,}")
    col2.metric("Total movies", f"{df['type'].eq('Movie').sum():,}")
    col3.metric("Total TV shows", f"{df['type'].eq('TV Show').sum():,}")

    valid_years = df["release_year"].dropna()
    year_range = (
        f"{int(valid_years.min())} - {int(valid_years.max())}"
        if not valid_years.empty
        else "Unavailable"
    )
    col4.metric("Release year range", year_range)
    st.markdown("---")

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Content type")
        type_counts = df["type"].value_counts()
        if type_counts.empty:
            st.info("No content type data is available.")
        else:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            colors = sns.color_palette("Set2", n_colors=len(type_counts))
            ax.bar(type_counts.index, type_counts.values, color=colors)
            ax.set_ylabel("Count")
            st.pyplot(fig, clear_figure=True)

    with col_right:
        st.subheader("Top 10 content-producing countries")
        countries = (
            df.loc[df["country"].str.casefold() != "unknown", "country"]
            .str.split(",")
            .explode()
            .str.strip()
        )
        top_countries = countries[countries.ne("")].value_counts().head(10)
        if top_countries.empty:
            st.info("No country data is available.")
        else:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            colors = sns.color_palette("viridis", n_colors=len(top_countries))
            ax.barh(top_countries.index[::-1], top_countries.values[::-1], color=colors)
            ax.set_xlabel("Number of titles")
            st.pyplot(fig, clear_figure=True)

    st.markdown("---")
    st.subheader("Filter content by genre")
    genres = (
        df["listed_in"]
        .str.split(",")
        .explode()
        .str.strip()
    )
    genres = sorted(genre for genre in genres.unique() if genre)
    if not genres:
        st.info("No genre data is available.")
    else:
        selected_genre = st.selectbox("Select a genre:", genres)
        filtered_df = df[
            df["listed_in"].str.contains(re.escape(selected_genre), case=False, na=False)
        ]
        st.write(
            f"Displaying up to 10 titles for **{selected_genre}** "
            f"({len(filtered_df)} total titles found):"
        )
        st.dataframe(
            filtered_df[["title", "type", "release_year", "rating", "director"]].head(10),
            width="stretch",
        )

elif page == "Recommendation System":
    st.title("🤖 Content recommendation engine")
    st.write("Find similar content using TF-IDF and cosine similarity.")

    @st.cache_resource
    def build_recommendation_engine(data):
        feature_columns = [
            "title",
            "type",
            "director",
            "cast",
            "listed_in",
            "description",
        ]
        features = data.reindex(columns=feature_columns, fill_value="").fillna("")
        combined_features = features.astype(str).agg(" ".join, axis=1)
        tfidf_matrix = TfidfVectorizer(
            stop_words="english",
            max_features=5000,
        ).fit_transform(combined_features)

        title_indices = {}
        for index, title in enumerate(features["title"].astype(str)):
            title_indices.setdefault(title.casefold(), index)
        return tfidf_matrix, title_indices

    tfidf_matrix, title_indices = build_recommendation_engine(df)
    available_titles = sorted(df.loc[df["title"].ne(""), "title"].unique())
    if len(available_titles) < 2:
        st.info("At least two titles are required to make recommendations.")
    else:
        selected_title = st.selectbox("Select a movie or TV show:", available_titles)
        top_n = st.slider(
            "Number of recommendations:",
            min_value=1,
            max_value=min(10, len(available_titles) - 1),
            value=min(5, len(available_titles) - 1),
        )

        if st.button("Get recommendations"):
            selected_index = title_indices[selected_title.casefold()]
            similarities = linear_kernel(
                tfidf_matrix[selected_index],
                tfidf_matrix,
            ).ravel()
            similarities[selected_index] = -1
            recommended_indices = np.argsort(similarities)[::-1][:top_n]
            results = df.iloc[recommended_indices][
                ["title", "type", "listed_in", "description"]
            ].copy()
            results["similarity_score"] = np.round(
                similarities[recommended_indices] * 100,
                2,
            )

            st.subheader(f"Recommendations for “{selected_title}”")
            for _, row in results.iterrows():
                with st.expander(
                    f"{row['title']} ({row['type']}) — Match: "
                    f"{row['similarity_score']}%"
                ):
                    st.write(f"**Genre:** {row['listed_in'] or 'Unavailable'}")
                    st.write(f"**Description:** {row['description'] or 'Unavailable'}")

elif page == "Trend Prediction":
    st.title("📈 Content growth trend forecasting")
    st.write("Forecast future title counts from historical release-year data.")

    valid_years = df["release_year"].dropna()
    if valid_years.empty:
        st.warning("No valid release years are available for trend analysis.")
    else:
        min_year = int(valid_years.min())
        max_year = int(valid_years.max())
        if min_year < max_year:
            default_start = min(max(min_year, 2000), max_year - 1)
            start_year = st.slider(
                "Start year for historical analysis:",
                min_value=min_year,
                max_value=max_year,
                value=default_start,
            )
        else:
            start_year = min_year
            st.info(f"Only {min_year} is available as a release year.")

        forecast_years = st.slider(
            "Forecast horizon (years):",
            min_value=1,
            max_value=10,
            value=5,
        )
        yearly_counts = (
            df.loc[df["release_year"] >= start_year, "release_year"]
            .value_counts()
            .sort_index()
        )

        if len(yearly_counts) < 2:
            st.warning("At least two distinct release years are required for a forecast.")
        else:
            last_year = int(yearly_counts.index.max())
            historical_years = np.arange(start_year, last_year + 1)
            history = yearly_counts.reindex(historical_years, fill_value=0)
            training_data = pd.DataFrame({"release_year": history.index})
            model = LinearRegression().fit(training_data, history.values)

            future_years = np.arange(
                last_year + 1,
                last_year + forecast_years + 1,
            )
            future_data = pd.DataFrame({"release_year": future_years})
            future_predictions = np.maximum(
                np.rint(model.predict(future_data)),
                0,
            ).astype(int)
            forecast_df = pd.DataFrame(
                {
                    "Release Year": future_years,
                    "Predicted Titles": future_predictions,
                }
            )

            fig, ax = plt.subplots(figsize=(10, 4.5))
            ax.plot(history.index, history.values, marker="o", label="Historical titles")
            ax.plot(
                forecast_df["Release Year"],
                forecast_df["Predicted Titles"],
                marker="s",
                linestyle="--",
                color="red",
                label="Forecast",
            )
            ax.set_xlabel("Release year")
            ax.set_ylabel("Title count")
            ax.legend()
            st.pyplot(fig, clear_figure=True)

            st.subheader("Forecast data")
            st.dataframe(forecast_df, width="stretch")
