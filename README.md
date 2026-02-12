#  VisaTime AI  
## AI-Enabled Visa Processing Time Estimator  

# 📘 Milestone 1: Data Collection & Preprocessing

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

✔ Ensures accurate time-based calculations.

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

- 📊 Exploratory Data Analysis (Milestone 2)  
- 🤖 Regression Model Training  
- 📈 Performance Evaluation  


# 🛠 Tech Stack Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
