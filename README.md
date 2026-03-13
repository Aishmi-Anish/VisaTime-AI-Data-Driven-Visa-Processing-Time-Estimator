#  VisaTime AI  
## AI-Enabled Visa Processing Time Estimator  

# Milestone 1: Data Collection & Preprocessing

##  Project Overview

**VisaTime AI** is an AI-powered system designed to estimate visa processing times using historical H-1B disclosure data.

Milestone 1 focuses on transforming raw disclosure records into a structured, realistic, and machine learning–ready dataset.

## Milestone 1 Objectives

- Clean raw dataset  
- Handle missing values  
- Engineer meaningful features  
- Remove unrealistic and abnormal records  
- Normalize and encode data  
- Generate final training-ready dataset  

#  Dataset Description

- **Source:** H-1B Disclosure Data (Public Government Dataset)  
- **Initial Dataset Size:** 79,999 rows × 11 columns  
- **Final Dataset Size:** 78,400 rows × 8 columns  

##  Key Features Used

- `application_date`
- `decision_date`
- `visa_type`
- `occupation_category`
- `wage`
- `worksite_state`
- `processing_center` *(derived proxy)*
- `processing_days` *(target variable)*

# Preprocessing Pipeline

## 1️⃣ Column Standardization

- Converted column names to lowercase  
- Removed extra spaces  
- Ensured naming consistency  

 Prevents case-sensitive errors and improves readability.


## 2️⃣ Date Standardization

- Converted `application_date` and `decision_date` into datetime format  
- Invalid or corrupted values were converted to null and removed  

 Ensures accurate time-based calculations.

## 3️⃣ Target Variable Creation

Processing time calculated as:

```python
processing_days = decision_date - application_date
```

 This becomes the regression target variable.

## 4️⃣ Removal of Unrealistic Durations

Records retained where:

- Processing days ≥ 1  
- Processing days ≤ 240  

 Removes appeal cases and abnormal records.  
 Ensures realistic adjudication window.

## 5️⃣ Missing Value Handling

### Numerical Columns
- `wage` filled using **median imputation**
- Median is robust against outliers.

### Categorical Columns
Missing values replaced with `"UNKNOWN"` for:

- `job_title`
- `occupation_category`
- `worksite_state`
- `visa_type`

 Prevents unnecessary row deletion.  
 Preserves dataset size.

## 6️⃣ Outlier Treatment (Wage)

- Removed extreme wage values using 1st and 99th percentile filtering.

 Improves regression stability.  
 Reduces noise from abnormal salary entries.

## 7️⃣ Feature Engineering

### Seasonal Features

Extracted:
- `month`
- `day_of_week`

 Captures seasonal and workload patterns.

### Processing Center Proxy

Since real USCIS service center information is not available in disclosure data, a proxy was created using geographic state mapping.

Example grouping:
- WEST_CENTER
- CENTRAL_CENTER
- EAST_CENTER
- SOUTH_CENTER

 Captures regional processing variation.

## 8️⃣ Encoding

Applied **Label Encoding** to categorical variables:

- `visa_type`
- `occupation_category`
- `worksite_state`
- `processing_center`

 Converts text features into numeric format for machine learning models.

## 9️⃣ Normalization

Applied **Standard Scaling** to the `wage` feature.

 Prevents large numerical values from dominating regression models.  
 Improves performance for linear models.


## 🔟 Feature Selection

Final selected features for training:

- `visa_type`
- `occupation_category`
- `wage_scaled`
- `worksite_state`
- `processing_center`
- `month`
- `day_of_week`
- `processing_days` (Target)

 Removed redundant or non-predictive columns.  
 Reduced dimensionality from 11 to 8 columns.

# Final Training Dataset

| Metric | Value |
|--------|--------|
| Rows | 78,400 |
| Columns | 8 |
| Target Variable | processing_days |
| Data Type | Fully numeric |

Final dataset saved as:

```
visa_training_ready.csv
```

# Milestone 1 Outcome

- Cleaned and validated dataset  
- Removed unrealistic and abnormal cases  
- Handled missing values effectively  
- Engineered seasonality and regional features  
- Applied encoding and normalization  
- Generated ML-ready dataset  

The dataset is now ready for:

-  Exploratory Data Analysis (Milestone 2)  
-  Regression Model Training  
-  Performance Evaluation  


# Tech Stack Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn

#  Milestone 2: Exploratory Data Analysis (EDA)

##  Overview

Milestone 2 focuses on performing **Exploratory Data Analysis (EDA)** on the preprocessed dataset generated in Milestone 1.  
The objective of this phase is to understand data behavior, identify hidden patterns, analyze feature relationships, and derive insights that guide model selection and prediction strategy.

EDA helps transform cleaned data into actionable knowledge before building machine learning models.

##  Objectives of Milestone 2

- Understand distribution of visa processing time
- Identify trends and seasonal patterns
- Analyze feature influence on processing duration
- Detect correlations between variables
- Validate preprocessing decisions
- Support model selection using data-driven insights

##  Dataset Used

- **Input Dataset:** `visa_training_ready.csv`
- **Records:** 78,400
- **Features:** 8 (Fully numeric and ML-ready)

### Features Analyzed

- `visa_type`
- `occupation_category`
- `wage_scaled`
- `worksite_state`
- `processing_center`
- `month`
- `day_of_week`
- `processing_days` (Target Variable)

#  Exploratory Analysis Performed


## 1️⃣ Processing Time Distribution

### Observation
- Distribution is highly **right-skewed**.
- Majority of applications processed within **4–8 days**.
- Small number of cases extend to longer durations.

### Conclusion
Visa adjudication is dominated by fast approvals, while complex cases create a long-tail delay pattern.

## 2️⃣ State-wise Processing Analysis

### Observation
Average processing time varies slightly across states.

### Conclusion
Geographical location alone does not strongly influence processing speed, suggesting centralized or standardized processing workflows.

## 3️⃣ Processing Center Analysis

### Observation
Processing distributions across centers are similar.

### Conclusion
Regional workload differences exist but are not dominant predictors individually.

## 4️⃣ Visa Type Analysis

### Observation
Processing time distributions overlap across visa types.

### Conclusion
Visa category has limited standalone impact on processing duration.

## 5️⃣ Wage vs Processing Time

### Observation
Scatter analysis shows no strong linear relationship.

### Conclusion
Higher wages do not directly result in faster processing decisions.

## 6️⃣ Seasonal Trend Analysis

### Observation
Moderate variation observed across months.

### Conclusion
Visa processing exhibits mild seasonal workload patterns influenced by application cycles.

## 7️⃣ Day-of-Week Analysis

### Observation
Processing times remain consistent across weekdays.

### Conclusion
Submission day has minimal operational effect due to batch processing systems.

## 8️⃣ Correlation Analysis

A correlation heatmap revealed:

- Very weak linear correlations between individual features and processing time.
- No single dominant predictor.

### Key Insight
Processing duration depends on **nonlinear interactions between multiple features** rather than isolated variables.

#  Key Findings

- Processing time is strongly right-skewed.
- Most applications are processed rapidly.
- Delayed cases form a small but important subset.
- Individual features show weak linear relationships.
- Complex interactions drive prediction behavior.

#  Implications for Modeling

Based on EDA findings:

| Observation | Modeling Decision |
|-------------|------------------|
| Skewed target distribution | Avoid purely linear assumptions |
| Weak correlations | Use nonlinear models |
| Complex feature interactions | Apply ensemble learning |
| Presence of rare delays | Robust regression required |

### Recommended Models
- Random Forest Regressor
- Gradient Boosting Regressor
- Ensemble-based approaches

# Milestone 2 Outcome

- Comprehensive understanding of dataset behavior
- Identification of temporal and regional trends
- Validation of preprocessing pipeline
- Data-driven justification for model selection

The project is now ready to proceed to:

➡ **Milestone 3 – Predictive Modeling**


## Tools Used

- Python
- Pandas
- Matplotlib
- Seaborn

# Milestone 3 – Model Development, Evaluation & Fine-Tuning

## Overview

In this milestone, multiple regression models were trained to predict **visa processing time (in days)** using the preprocessed dataset generated in Milestone 1 and analyzed in Milestone 2. The goal was to evaluate different machine learning models and identify the most suitable model for estimating processing duration.

The following models were implemented and compared:

- Linear Regression (Baseline Model)
- Random Forest Regressor
- XGBoost Regressor
- Fine-Tuned XGBoost Regressor (using GridSearchCV)

Model performance was evaluated using the following metrics:

- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **Coefficient of Determination (R² Score)**

# Dataset Used

**Input Dataset:** `visa_training_ready.csv`

| Property | Value |
|--------|------|
| Total Records | 78,400 |
| Features | Visa attributes, location, wage, temporal features |
| Target Variable | `processing_days` |

### Features Used

- visa_type  
- occupation_category  
- worksite_state  
- processing_center  
- wage_scaled  
- month  
- day_of_week  

### Target Variable
- processing_days: Represents the number of days taken for visa application processing.

# Models Implemented

## Linear Regression

Linear Regression was used as a baseline model to understand whether a simple linear relationship exists between the features and the target variable.

However, the dataset exhibited weak linear correlations and a highly skewed distribution, which limited the performance of this model.


## Random Forest Regressor

Random Forest is an ensemble learning method that builds multiple decision trees and aggregates their predictions.

Advantages:
- Handles nonlinear relationships
- Robust to outliers
- Works well with high-dimensional tabular data

Random Forest improved prediction performance compared to Linear Regression.

## XGBoost Regressor

XGBoost (Extreme Gradient Boosting) is a powerful ensemble algorithm based on gradient boosting.

Advantages:
- Captures complex nonlinear relationships
- Handles feature interactions effectively
- Provides strong performance on structured datasets

XGBoost achieved the best performance among the initial models.



# Hyperparameter Fine-Tuning

To further improve the model performance, **GridSearchCV** was used to optimize XGBoost hyperparameters.

### Parameters Tuned

- Number of estimators (`n_estimators`)
- Tree depth (`max_depth`)
- Learning rate (`learning_rate`)
- Subsample ratio (`subsample`)
- Feature sampling (`colsample_bytree`)

Cross-validation was used to evaluate parameter combinations and identify the best configuration.

# Model Performance Comparison

| Model | MAE | RMSE | R² Score |
|------|------|------|------|
| Linear Regression | ~7.62 | ~23.43 | ~0.0006 |
| Random Forest | ~7.55 | ~23.27 | ~0.013 |
| XGBoost (Baseline) | 7.5978 | 23.12 | 0.0268 |
| **XGBoost (Fine-Tuned)** | **7.4493** | **23.03** | **0.0339** |


# Best Model Selection

The **Fine-Tuned XGBoost Regressor** was selected as the final model because it achieved:

- Lowest **MAE**
- Lowest **RMSE**
- Highest **R² score**

Although the dataset exhibited weak feature correlations and a highly skewed distribution, XGBoost demonstrated better ability to capture nonlinear relationships among visa attributes.

# Key Observations

- The dataset shows a strong dominance of applications processed within a few days.
- Processing delays are influenced by complex operational factors not fully captured by available features.
- Tree-based ensemble models outperform linear models for this type of structured data.

# Outcome of Milestone 3

By the end of this milestone:

✔ Multiple regression models were trained  
✔ Model performance was evaluated using MAE, RMSE, and R²  
✔ Hyperparameter tuning was applied using GridSearchCV  
✔ The **Fine-Tuned XGBoost model** was selected as the final predictive model  

# Next Step

The selected model will be integrated into a **user-facing application interface** in **Milestone 4**, enabling users to input visa attributes and receive estimated processing time predictions.

