import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def map_processing_center(state):
    west = ['CA','WA','OR','NV','AZ','UT','CO','ID','MT','WY','AK','HI']
    central = ['TX','OK','KS','NE','IA','MO','MN','ND','SD']
    east = ['NY','NJ','MA','CT','PA','VA','MD','DC','RI','NH','ME','VT']

    if state in west:
        return "WEST_CENTER"
    elif state in central:
        return "CENTRAL_CENTER"
    elif state in east:
        return "EAST_CENTER"
    else:
        return "SOUTH_CENTER"


def main():

    df = pd.read_csv("visa_dataset.csv")
    print("Initial shape:", df.shape)

    
    df.columns = df.columns.str.lower().str.strip()

    
    df['application_date'] = pd.to_datetime(df['application_date'], errors='coerce')
    df['decision_date'] = pd.to_datetime(df['decision_date'], errors='coerce')

    
    df['processing_days'] = (df['decision_date'] - df['application_date']).dt.days

   
    df = df[(df['processing_days'] >= 1) & (df['processing_days'] <= 240)]

    df['wage'] = df['wage'].fillna(df['wage'].median())

   
    for col in ['job_title','occupation_category','worksite_state','visa_type']:
        df[col] = df[col].fillna("UNKNOWN")

  
    df['month'] = df['application_date'].dt.month
    df['day_of_week'] = df['application_date'].dt.dayofweek

   
    df['processing_center'] = df['worksite_state'].apply(map_processing_center)

    q1 = df['wage'].quantile(0.01)
    q99 = df['wage'].quantile(0.99)
    df = df[(df['wage'] >= q1) & (df['wage'] <= q99)]

    encoder = LabelEncoder()
    for col in ['visa_type','occupation_category','worksite_state','processing_center']:
        df[col] = encoder.fit_transform(df[col])

    scaler = StandardScaler()
    df['wage_scaled'] = scaler.fit_transform(df[['wage']])


    training_df = df[[
        'visa_type',
        'occupation_category',
        'wage_scaled',
        'worksite_state',
        'processing_center',
        'month',
        'day_of_week',
        'processing_days'
    ]]


    training_df.to_csv("visa_training_ready.csv", index=False)

    print("Final shape:", training_df.shape)
    print("\nProcessing Days Summary:")
    print(training_df['processing_days'].describe())


if __name__ == "__main__":
    main()