# streamlit_app.py
# ============================================================
# Project 4 Bonus: Interactive Airbnb Price Predictor
# Uses the trained final model (final_model.pkl)
# Run with: streamlit run streamlit_app.py
# ============================================================

import streamlit as st

# ------------------------------
# CRITICAL: set_page_config MUST be the first Streamlit command
# ------------------------------
st.set_page_config(page_title="NYC Airbnb Price Predictor", layout="wide")

# Now import other libraries
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# ------------------------------
# 1. Load the final model (pipeline)
# ------------------------------
@st.cache_resource  # Cache the model to avoid reloading
def load_model():
    return joblib.load("final_model.pkl")

model = load_model()

# ------------------------------
# 2. App title and description
# ------------------------------
st.title("🏙️ NYC Airbnb Price Predictor")
st.markdown("""
This interactive dashboard predicts the nightly price of an Airbnb listing in New York City.
It is based on a **Tuned Random Forest** model trained on the NYC Airbnb Open Data (2019).
Adjust the parameters on the left sidebar to see the predicted price.
""")

# ------------------------------
# 3. Sidebar – User Input
# ------------------------------
st.sidebar.header("Listing Details")

# ---- Location ----
neighbourhood_group = st.sidebar.selectbox(
    "Borough (Neighbourhood Group)",
    ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
)

# We need the actual neighbourhood names that exist in the training data.
# For simplicity, provide common options (you can expand this list).
neighbourhood = st.sidebar.selectbox(
    "Neighbourhood",
    ["Upper West Side", "Harlem", "East Village", "Midtown",
     "Williamsburg", "Astoria", "Long Island City", "Bushwick",
     "Financial District", "Hell's Kitchen", "Chelsea", "Crown Heights"]
)

room_type = st.sidebar.selectbox(
    "Room Type",
    ["Entire home/apt", "Private room", "Shared room"]
)

# ---- Coordinates ----
st.sidebar.subheader("Coordinates")
latitude = st.sidebar.number_input("Latitude", min_value=40.5, max_value=40.9, value=40.75, step=0.001)
longitude = st.sidebar.number_input("Longitude", min_value=-74.3, max_value=-73.7, value=-73.98, step=0.001)

# ---- Numeric features ----
st.sidebar.subheader("Listing Attributes")
minimum_nights = st.sidebar.slider("Minimum Nights", 1, 30, 3)
number_of_reviews = st.sidebar.number_input("Number of Reviews", 0, 500, 10)
reviews_per_month = st.sidebar.number_input("Reviews per Month", 0.0, 20.0, 1.5, step=0.1)
calculated_host_listings_count = st.sidebar.number_input("Host Listings Count", 1, 100, 5)
availability_365 = st.sidebar.slider("Availability (days per year)", 0, 365, 180)

# ---- Derived / engineered features ----
# The model expects all the features we created during training.
# We'll compute them automatically from the inputs above.
st.sidebar.subheader("Listing Name Keywords")
# Display checkboxes for a few keywords (simplified; all others default to 0)
keywords_list = ["luxury", "cozy", "spacious", "modern", "private", "studio",
                 "loft", "central", "park", "view", "garden", "rooftop", "penthouse"]
keyword_flags = {}
st.sidebar.caption("Check any that apply:")
for kw in keywords_list[:5]:  # Show only 5 for cleaner UI
    keyword_flags[kw] = st.sidebar.checkbox(kw.title())
# For unchecked or not shown, set 0
for kw in keywords_list:
    if kw not in keyword_flags:
        keyword_flags[kw] = 0

# ------------------------------
# 4. Feature Engineering (same as training)
# ------------------------------
def engineer_features(inputs):
    # Create a DataFrame with one row
    df = pd.DataFrame([inputs])
    
    # Distances to landmarks (computed from coordinates)
    landmarks = {
        "times_sq": (40.7580, -73.9855),
        "central_park": (40.7855, -73.9632),
        "empire_state": (40.7484, -73.9857),
        "jfk": (40.6413, -73.7781)
    }
    for name, (lat, lon) in landmarks.items():
        df[f"dist_{name}"] = np.sqrt((df["latitude"] - lat)**2 + (df["longitude"] - lon)**2)
    
    # Temporal features (we don't have last_review date, so use defaults)
    df["days_since_review"] = 180  # median assumption
    df["has_reviews"] = 1 if df["number_of_reviews"].values[0] > 0 else 0
    
    # Interaction features
    df["activity_index"] = df["number_of_reviews"] * df["reviews_per_month"]
    df["avail_ratio"] = df["availability_365"] / 365
    df["host_listings_log"] = np.log1p(df["calculated_host_listings_count"])
    df["popularity_proxy"] = df["number_of_reviews"] * df["avail_ratio"]
    
    # Keyword features
    for kw in keywords_list:
        df[f"kw_{kw}"] = keyword_flags[kw]
    
    # geo_cluster: we cannot run KMeans here, so assign a default cluster (e.g., 2)
    # In the original training, we had a 'geo_cluster' column.
    # To avoid missing column errors, we set a default. The model will still work,
    # but the prediction might be slightly less accurate.
    df["geo_cluster"] = 2
    
    return df

# ------------------------------
# 5. Prepare input and predict
# ------------------------------
input_data = {
    "neighbourhood_group": neighbourhood_group,
    "neighbourhood": neighbourhood,
    "room_type": room_type,
    "latitude": latitude,
    "longitude": longitude,
    "minimum_nights": minimum_nights,
    "number_of_reviews": number_of_reviews,
    "reviews_per_month": reviews_per_month,
    "calculated_host_listings_count": calculated_host_listings_count,
    "availability_365": availability_365
}

input_df = engineer_features(input_data)

# Make prediction (log scale)
log_pred = model.predict(input_df)[0]
# Convert back to dollars
price_pred = np.expm1(log_pred)

# ------------------------------
# 6. Display results
# ------------------------------
st.markdown("---")
col1, col2 = st.columns(2)
col1.metric("📌 Predicted Nightly Price", f"${price_pred:.2f}")
col2.metric("🏷️ Log-Price", f"{log_pred:.4f}")

st.markdown("### Input Summary")
st.dataframe(input_df.transpose(), use_container_width=True)

# Optional: Show feature importance or other insights
st.info("Note: This prediction is based on a Random Forest model trained on 2019 data. Actual current prices may vary.")

# Footer
st.markdown("---")
st.caption("Built as a bonus deliverable for Project 4 – End-to-End Machine Learning.")