📘 VisaTime AI – Milestone 1
Data Collection & Preprocessing
📌 Project Overview

VisaTime AI is an AI-powered system designed to estimate visa processing times using historical H-1B disclosure data.

Milestone 1 focuses on:

Data cleaning

Missing value handling

Feature engineering

Outlier treatment

Preparing a training-ready dataset

This stage transforms raw disclosure records into a structured dataset suitable for machine learning.

📂 Dataset Description

Source: H-1B Disclosure Data (Public Government Dataset)
Initial Size: 79,999 records × 11 columns
After Preprocessing: 78,400 records × 8 columns

Key Features Used

application_date

decision_date

visa_type

occupation_category

wage

worksite_state

processing_center (derived proxy)

processing_days (target variable)

🎯 Objective of Milestone 1

To create a clean, realistic, and model-ready dataset for predicting visa processing time.

🔧 Preprocessing Steps Performed
1️⃣ Date Standardization

Converted application_date and decision_date into datetime format.

Removed invalid or corrupted date records.

✔ Ensures correct time-based calculations.

2️⃣ Target Variable Creation
processing_days = decision_date - application_date

✔ This represents the number of days taken for visa decision.

3️⃣ Removal of Unrealistic Durations

Kept records where:

Processing days ≥ 1

Processing days ≤ 240

✔ Removes appeal cases and abnormal records.

4️⃣ Missing Value Handling
Numerical

Wage values filled using median imputation

Categorical

Missing values replaced with "UNKNOWN"

✔ Prevents unnecessary row deletion.

5️⃣ Outlier Treatment

Removed extreme wage values using 1st and 99th percentile filtering.

✔ Improves regression stability.

6️⃣ Feature Engineering
Seasonal Features

Extracted month

Extracted day_of_week

Processing Center Proxy

Since real USCIS service center information is not available in disclosure data, a proxy was created using geographic state mapping.

✔ Captures regional workload variation.

7️⃣ Encoding

Applied Label Encoding to categorical variables:

visa_type

occupation_category

worksite_state

processing_center

✔ Converts text features into numeric format for ML models.

8️⃣ Normalization

Applied Standard Scaling to wage feature.

✔ Prevents scale dominance in regression models.

📊 Final Training Dataset

After preprocessing:

Metric	Value
Rows	78,400
Columns	8
Target Variable	processing_days
Data Type	Fully numeric

Final dataset saved as:

visa_training_ready.csv
✅ Milestone 1 Outcome

Cleaned and validated dataset

Removed unrealistic records

Handled missing values

Engineered relevant features

Created ML-ready training dataset

The dataset is now ready for:

Exploratory Data Analysis (Milestone 2)

Regression modeling

Performance evaluation
