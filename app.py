import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time

# ------------------------------------------------
# Page Configuration
# ------------------------------------------------

st.set_page_config(
    page_title="VisaTime AI",
    page_icon="🛂",
    layout="wide"
)

# ------------------------------------------------
# Load Model + Data
# ------------------------------------------------

model = joblib.load("visa_processing_model.pkl")
features = joblib.load("model_features.pkl")

df = pd.read_csv("visa_dataset_training_ready.csv")

# ------------------------------------------------
# Sidebar
# ------------------------------------------------

st.sidebar.title("🛂 VisaTime AI")

st.sidebar.write("""
AI-powered system for predicting visa processing time.

Trained using:
- XGBoost Regressor
- Fine-tuned with GridSearchCV
""")

st.sidebar.write("Estimate **visa processing duration** using machine learning trained on historical visa data.")

# ------------------------------------------------
# Title
# ------------------------------------------------

st.title("🛂 Visa Processing Time Prediction System")

st.markdown("""
Fill in the details below to generate an **AI-based prediction**.
""")

st.divider()

# ------------------------------------------------
# Layout Columns
# ------------------------------------------------

col1, col2 = st.columns(2)

# ------------------------------------------------
# User Inputs
# ------------------------------------------------

with col1:

    st.subheader("Visa Information")

    visa_type = st.selectbox(
        "Visa Type",
        ["Select Visa Type", "H1B", "L1", "F1"]
    )

    occupation_category = st.selectbox(
        "Occupation Category",
        ["Select Occupation", "IT", "Engineering", "Finance", "Healthcare"]
    )

    wage = st.number_input(
        "Annual Wage (USD)",
        min_value=20000,
        max_value=300000,
        value=None,
        placeholder="Enter wage"
    )

with col2:

    st.subheader("Application Details")

    worksite_state = st.selectbox(
        "Worksite State",
        ["Select State", "CA", "TX", "NY", "WA", "FL"]
    )

    processing_center = st.selectbox(
        "Processing Center",
        ["Select Center", "WEST_CENTER", "EAST_CENTER", "SOUTH_CENTER"]
    )

    month = st.slider(
        "Application Month",
        1, 12
    )

    day_of_week = st.slider(
        "Day of Week (0 = Monday)",
        0, 6
    )

st.divider()

# ------------------------------------------------
# Prediction Engine
# ------------------------------------------------

if st.button("🔍 Predict Processing Time"):

    if (
        visa_type == "Select Visa Type" or
        occupation_category == "Select Occupation" or
        worksite_state == "Select State" or
        processing_center == "Select Center" or
        wage is None
    ):
        st.error("⚠ Please fill all input fields before prediction.")

    else:

        with st.spinner("🔍 Analyzing visa application... Please wait"):
            time.sleep(3)   # ⏳ adds delay

            input_data = pd.DataFrame({
                "visa_type": [visa_type],
                "occupation_category": [occupation_category],
                "worksite_state": [worksite_state],
                "processing_center": [processing_center],
                "wage_scaled": [wage],
                "month": [month],
                "day_of_week": [day_of_week]
            })

            input_encoded = pd.get_dummies(input_data)
            input_encoded = input_encoded.reindex(columns=features, fill_value=0)

            prediction = model.predict(input_encoded)[0]
        # ------------------------------------------------
        # Prediction Result
        # ------------------------------------------------

        st.subheader("Prediction Result")

        st.metric(
            label="Estimated Processing Time",
            value=f"{prediction:.2f} days",
            delta="AI Prediction"
        )

        # Risk Indicator
        if prediction <= 7:
            st.success("Low Processing Delay Risk")
        elif prediction <= 30:
            st.warning("Moderate Processing Delay Risk")
        else:
            st.error("High Processing Delay Risk")


# ------------------------------------------------
# Data Insights Section
# ------------------------------------------------

st.divider()

# Toggle Button
if "show_data" not in st.session_state:
    st.session_state.show_data = False

if st.button("📊 Show Historical Data Insights"):
    st.session_state.show_data = not st.session_state.show_data


# Show only when clicked
if st.session_state.show_data:

    st.subheader("📊 Historical Data Insights")

    col3, col4 = st.columns(2)

    # Histogram
    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Processing Time Distribution")

        fig, ax = plt.subplots()
        sns.histplot(df["processing_days"], bins=40, kde=True, ax=ax)

        ax.set_xlabel("Processing Days")
        ax.set_ylabel("Frequency")

        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    # Feature Importance
    with col4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Model Feature Importance")

        importance = model.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": features,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False).head(10)

        fig, ax = plt.subplots()
        ax.barh(importance_df["Feature"], importance_df["Importance"])
        ax.invert_yaxis()

        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
        
# ------------------------------------------------
# AI Recommendation Section
# ------------------------------------------------

st.divider()
st.subheader("Insights:")

st.write("""
Based on historical visa data analysis:

• Applications submitted early in the year often experience faster processing.

• Processing times may vary by administrative processing centers.

• Higher wage applications may receive faster processing due to priority categories.

These insights are derived from patterns observed in the training dataset.
""")

# ------------------------------------------------
# Footer
# ------------------------------------------------

st.markdown("---")
st.caption("VisaTime AI • AI Enabled Visa Processing Time Estimator • Infosys Springboard Virtual Internship 6.0")