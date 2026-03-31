# 🛂 VisaTime AI  
### AI-Enabled Visa Processing Time Estimator

VisaTime AI is an end-to-end machine learning system designed to estimate visa processing time using historical H-1B disclosure data. The project combines data preprocessing, exploratory analysis, predictive modeling, and an interactive Streamlit-based user interface.

---

# 📌 Project Overview

Visa processing systems handle large volumes of applications, leading to delays and uncertainty. This project leverages machine learning to:

- Predict visa processing duration  
- Provide data-driven insights  
- Improve transparency for applicants  

The system is built as a **complete AI pipeline**, from raw data to a deployed interactive interface.

---

# 🧱 Project Architecture
Raw Dataset → Preprocessing → EDA → Model Training → UI → Prediction

---

# ⚙️ Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost  
- Matplotlib, Seaborn  
- Streamlit  

---

# 📊 Dataset

- Source: H-1B Disclosure Data (Public Dataset)  
- Initial Size: 79,999 × 11  
- Final Size: 78,400 × 8  

### Features Used

- visa_type  
- occupation_category  
- worksite_state  
- processing_center (proxy)  
- wage_scaled  
- month  
- day_of_week  

### Target Variable

- processing_days → Visa processing duration  

---

# 🟢 Milestone 1 – Data Preprocessing

### Key Steps

- Column standardization  
- Date conversion and validation  
- Target variable creation  
- Removal of unrealistic records (1–240 days)  
- Missing value handling  
- Outlier removal (wage filtering)  
- Feature engineering (month, day_of_week)  
- Processing center proxy creation  
- Encoding and normalization  

### Outcome

- Clean, structured dataset  
- Reduced noise and anomalies  
- Training-ready dataset (`visa_training_ready.csv`)  

---

# 🟡 Milestone 2 – Exploratory Data Analysis

### Key Insights

- Processing time is **right-skewed**  
- Majority of applications processed within a few days  
- No strong linear correlation between individual features  
- Processing depends on **complex feature interactions**  
- Seasonal patterns exist but are mild  

### Conclusion

- Linear models are insufficient  
- Nonlinear ensemble models are required  

---

# 🔵 Milestone 3 – Model Development

### Models Implemented

- Linear Regression  
- Random Forest Regressor  
- XGBoost Regressor  
- Fine-Tuned XGBoost (GridSearchCV)  

### Evaluation Metrics

- MAE (Mean Absolute Error)  
- RMSE (Root Mean Squared Error)  
- R² Score  

### Best Model

**Fine-Tuned XGBoost Regressor**

| Model | MAE | RMSE | R² |
|------|------|------|------|
| Linear Regression | ~7.62 | ~23.43 | ~0.0006 |
| Random Forest | ~7.55 | ~23.27 | ~0.013 |
| XGBoost | ~7.59 | ~23.12 | ~0.026 |
| **XGBoost (Tuned)** | **~7.45** | **~23.03** | **~0.033** |

### Outcome

- Nonlinear models outperform linear models  
- XGBoost selected as final model  

---

# 🟣 Milestone 4 – Streamlit UI & Prediction Engine

### Overview

A user-friendly **Streamlit web application** was developed to interact with the trained model.

### Features

- Interactive input fields  
- Real-time prediction engine  
- Risk classification (Low / Moderate / High)  
- AI-based analysis simulation (spinner)  
- Dark-themed professional UI  
- On-demand historical data visualization  

### User Workflow
User Input → Data Encoding → Model Prediction → Output Display

### Output

- Estimated processing time (in days)  
- Risk level indicator  
- Visual insights (optional toggle)  

---

# 📊 Visualizations

- Processing Time Distribution (Histogram)  
- Feature Importance (Model Explainability)  

---

# 🎯 Key Observations

- Most visa applications are processed quickly  
- Delays occur in a small subset of cases  
- Processing depends on multiple interacting factors  
- Model captures nonlinear relationships effectively  

---

# 🚀 Features of the System

- End-to-end ML pipeline  
- Data-driven predictions  
- Interactive UI  
- Explainable insights  
- Scalable architecture  

---

# ⚠ Limitations

- Dataset lacks real-time USCIS processing data  
- External factors (policy changes, workload spikes) are not captured  
- Moderate R² due to complexity of real-world systems  

---

# 🔮 Future Scope

- Integration with real-time immigration APIs  
- Explainable AI (SHAP) for per-user prediction insights  
- Multi-country visa prediction system  
- AI-based conversational visa assistant  

---

# ▶️ How to Run the Project

```bash
pip install -r requirements.txt
streamlit run app.py

# Deployed link:

https://visatime-ai-data-driven-visa-processing-time-estimator-feptkuc.streamlit.app/
