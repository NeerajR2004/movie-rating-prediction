import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import zipfile
import urllib.request
import certifi
import ssl
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Movie Rating Prediction", page_icon="🎬", layout="wide")

st.title("🎬 Movie Rating Prediction System")
st.write("Predict a movie's average user rating using a Machine Learning model.")

@st.cache_resource
def load_model():
    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    with urllib.request.urlopen(url, timeout=60, context=ssl.create_default_context(cafile=certifi.where())) as response:
        content = response.read()

    with zipfile.ZipFile(BytesIO(content)) as z:
        ratings = pd.read_csv(z.open("ml-latest-small/ratings.csv"))
        movies = pd.read_csv(z.open("ml-latest-small/movies.csv"))

    data = ratings.merge(movies, on="movieId")
    data["year"] = data["title"].str.extract(r"\((\d{4})\)").astype(float)
    data["year"] = data["year"].fillna(data["year"].median()).astype(int)

    genre_columns = data["genres"].str.get_dummies(sep="|")
    features = pd.concat([data[["year"]], genre_columns], axis=1)
    target = data["rating"]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=60,
        max_depth=18,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "R2": r2_score(y_test, predictions)
    }

    return model, list(features.columns), list(genre_columns.columns), metrics, y_test, predictions

with st.spinner("Downloading movie data and training the model. Please wait..."):
    try:
        model, feature_columns, genres, metrics, y_test, predictions = load_model()
    except Exception as e:
        st.error("Could not load the dataset or train the model.")
        st.write("Check your internet connection and try again.")
        st.code(str(e))
        st.stop()

st.success("Machine Learning model trained successfully!")

st.subheader("Predict a Movie Rating")

year = st.slider("Select movie release year", 1900, 2026, 2020)
selected_genres = st.multiselect(
    "Select movie genres",
    options=genres,
    default=["Comedy"] if "Comedy" in genres else []
)

if st.button("Predict Movie Rating"):
    input_data = {column: 0 for column in feature_columns}
    input_data["year"] = year

    for genre in selected_genres:
        if genre in input_data:
            input_data[genre] = 1

    input_df = pd.DataFrame([input_data], columns=feature_columns)
    rating = float(model.predict(input_df)[0])

    st.metric("Predicted Average Rating", f"{rating:.2f} / 5")
    st.progress(min(max(rating / 5, 0.0), 1.0))
    st.caption("This is an estimate learned from historical MovieLens user ratings.")

st.divider()
st.subheader("Model Evaluation")

col1, col2, col3 = st.columns(3)
col1.metric("Mean Absolute Error (MAE)", f"{metrics['MAE']:.3f}")
col2.metric("Root Mean Squared Error (RMSE)", f"{metrics['RMSE']:.3f}")
col3.metric("R² Score", f"{metrics['R2']:.3f}")

st.caption("Lower MAE and RMSE generally indicate smaller prediction errors.")

st.subheader("Actual vs Predicted Ratings")
fig1, ax1 = plt.subplots()
sample_size = min(300, len(y_test))
ax1.scatter(np.array(y_test)[:sample_size], np.array(predictions)[:sample_size], alpha=0.5)
ax1.set_xlabel("Actual Rating")
ax1.set_ylabel("Predicted Rating")
ax1.set_title("Actual vs Predicted Ratings")
ax1.plot([0.5, 5], [0.5, 5], linestyle="--")
st.pyplot(fig1)
plt.close(fig1)

st.subheader("Distribution of Ratings")
fig2, ax2 = plt.subplots()
ax2.hist(y_test, bins=10, edgecolor="black")
ax2.set_xlabel("Rating")
ax2.set_ylabel("Number of test ratings")
ax2.set_title("Distribution of Actual Ratings")
st.pyplot(fig2)
plt.close(fig2)

st.subheader("Feature Importance")
importance = pd.Series(model.feature_importances_, index=feature_columns)
importance = importance.drop(labels=["year"], errors="ignore").nlargest(10).sort_values()
fig3, ax3 = plt.subplots()
importance.plot(kind="barh", ax=ax3)
ax3.set_xlabel("Importance")
ax3.set_title("Top Genre Features")
st.pyplot(fig3)
plt.close(fig3)

st.info("Dataset: MovieLens Latest Small. Algorithm: Random Forest Regressor.")