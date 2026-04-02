import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time
import base64

def get_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read()).decode()


img = get_base64("Bg.jpg")   



st.set_page_config(
    page_title="VisaTime AI",
    page_icon="🛂",
    layout="wide"
)

st.markdown(f"""
<style>

/* BACKGROUND */
.stApp {{
    background-image: url("data:image/jpg;base64,{img}");
    background-size: cover;
    background-attachment: fixed;
}}

.stApp::before {{
    content: "";
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.35);
    z-index: -1;
}}

/* SIDEBAR */
section[data-testid="stSidebar"] {{
    background-color: #000 !important;
}}

section[data-testid="stSidebar"] * {{
    color: white !important;
}}

/* MAIN TEXT */
.block-container h1,
.block-container h2,
.block-container h3,
.block-container p,
.block-container label {{
    color: white !important;
}}

/* INPUTS */
input, textarea {{
    color: white !important;
    background-color: rgba(0,0,0,0.6) !important;
    border: 1px solid white !important;
}}

/* SELECT */
div[data-baseweb="select"] > div {{
    background-color: black !important;
    color: white !important;
}}

div[data-baseweb="select"] span {{
    color: white !important;
}}

ul[role="listbox"] {{
    background-color: black !important;
}}

ul[role="listbox"] li {{
    color: white !important;
}}

ul[role="listbox"] li:hover {{
    background-color: #333 !important;
}}

/* BUTTON (FIXED NO WHITE HOVER) */
button {{
    background-color: black !important;
    color: white !important;
    border: 1px solid white !important;
}}

button:hover {{
    background-color: black !important;
    color: white !important;
}}

/* METRIC */
[data-testid="stMetricValue"] {{
    color: white !important;
}}

/* SLIDER */
.stSlider * {{
    color: white !important;
}}

/* TITLE POSITION */
.title-container {{
    margin-top: -40px;
    margin-left: -50px;
}}

.title-container h1 {{
    color: white !important;
    margin-bottom: 0;
}}

.title-container .subtitle-text {{
    color: white !important;
    font-weight: 700;
}}

</style>
""", unsafe_allow_html=True)


model = joblib.load("visa_processing_model.pkl")
features = joblib.load("model_features.pkl")
df = pd.read_csv("visa_dataset_training_ready.csv")


st.sidebar.image("US_flag.png", width=120)  
st.sidebar.title("🛂 VisaTime AI")

st.sidebar.write("""
AI-powered system for predicting visa processing time.

Trained using:
- XGBoost Regressor
- Fine-tuned with GridSearchCV
""")
st.sidebar.write("""This system is based on U.S. visa processing data""")

st.markdown("""
<div class="title-container">
    <h1>VisaTime AI</h1>
    <p class="subtitle-text">
        AI-Based Visa Processing System (United States)
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

col1, col2 = st.columns(2)

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
    if visa_type == "F1":
        wage = 0
        st.info("ℹ️ Annual wage is not applicable for F1 (student visa)")
    else:
        wage = st.number_input( "Annual Wage (USD)",min_value=20000,max_value=300000,value=None,placeholder="Enter wage")

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

    month = st.slider("Application Month", 1, 12)
    day_of_week = st.slider("Day of Week (0 = Monday)", 0, 6)

st.divider()

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
            time.sleep(3)

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

        st.subheader("📊 Prediction Result")

        st.metric("Estimated Processing Time", f"{round(prediction)} days")
        

        if prediction <= 7:
            st.success("Low Processing Delay Risk")
        elif prediction <= 30:
            st.warning("Moderate Processing Delay Risk")
        else:
            st.error("High Processing Delay Risk")

st.divider()

if "show_data" not in st.session_state:
    st.session_state.show_data = False

if st.button("📊 Show Historical Data Insights"):
    st.session_state.show_data = not st.session_state.show_data

if st.session_state.show_data:

    st.subheader("📊 Historical Data Insights")

    col3, col4 = st.columns(2)

    with col3:
        fig, ax = plt.subplots()
        sns.histplot(df["processing_days"], bins=40, kde=True, ax=ax)
        ax.set_xlabel("Processing Days")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    with col4:
        importance = model.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": features,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False).head(10)

        fig, ax = plt.subplots()
        ax.barh(importance_df["Feature"], importance_df["Importance"])
        ax.invert_yaxis()
        st.pyplot(fig)

st.divider()
st.subheader("Insights")

st.write("""
• Applications submitted early in the year often experience faster processing  
• Processing varies across administrative centers  
• Higher wage applications may receive priority  
""")

st.markdown("---")
st.caption("VisaTime AI • AI Enabled Visa Processing Time Estimator • Infosys Springboard Internship")
