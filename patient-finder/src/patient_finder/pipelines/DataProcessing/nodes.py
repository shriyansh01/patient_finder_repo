"""
This is a boilerplate pipeline 'DataProcessing'
generated using Kedro 1.0.0
"""

import pandas as pd
import numpy as np
import random 
import pandas as pd
from datetime import datetime, timedelta
import numpy as np


def create_patient_pivot(df, values_to_pivot, current_date=None):
    df["service_date"] = pd.to_datetime(df["service_date"])

    if current_date is None:
        current_date = datetime.now()
    

    pivot_data = []

    for patient_id, group in df.groupby("patient_id"):
        patient_row = {"patient_id": patient_id, "target": group["target"].iloc[0]}
        patient_max_date = group["service_date"].max()
        for val in values_to_pivot:
            val_group = group[group["code"] == val].sort_values("service_date")
            freq = len(val_group)
            gap = (val_group["service_date"].diff().dt.days.mean()
                   if freq > 1 else np.nan)
            duration = ((patient_max_date  - val_group["service_date"]).dt.days.mean()
                        if freq > 0 else np.nan)
            patient_row[f"{val}_frequency"] = freq
            patient_row[f"{val}_gap"] = gap
            patient_row[f"{val}_duration"] = duration
        pivot_data.append(patient_row)
    pivot_df = pd.DataFrame(pivot_data)
    return pivot_df





import pandas as pd

def create_patient_sequence(claims_table):
    # Sort by patient and service_date
    claims_table = claims_table.sort_values(["patient_id", "service_date"])

    # Group by patient and aggregate codes into list (ordered sequence)
    seq_df = claims_table.groupby("patient_id").agg({
        "code": lambda x: list(x),
        "target": "first"   # or 'first' depending on definition
    }).reset_index()
    seq_df.columns = ["patient_id", "codes", "target"]
    return seq_df


def data_processing_pipeline(positive_class, negative_class,n_important_features=10):
    unique_codes = pd.concat([positive_class['code'], negative_class['code']]).unique()


# Count distinct patient IDs for each code
    positive_counts = positive_class.groupby('code')['patient_id'].nunique()
    negative_counts = negative_class.groupby('code')['patient_id'].nunique()

    # Combine counts into a DataFrame
    code_counts = pd.DataFrame({
        'positive_patient_count': positive_counts,
        'negative_patient_count': negative_counts
    }).fillna(0).astype(int)

    total_positive_patients_count = positive_class.patient_id.nunique()
    total_negative_patients_count = negative_class.patient_id.nunique()


    code_counts['positive_patient_relative'] = code_counts['positive_patient_count'] / total_positive_patients_count
    code_counts['negative_patient_relative'] = code_counts['negative_patient_count'] / total_negative_patients_count

    code_counts['variance'] = np.var([code_counts['positive_patient_relative'], code_counts['negative_patient_relative']], axis=0)
    important_features = code_counts.sort_values(by='variance', ascending=False).head(n_important_features).index.tolist()
    claims_table = pd.concat([positive_class, negative_class], ignore_index=True)

    pivot_df = create_patient_pivot(claims_table, important_features)
    sequence_df = create_patient_sequence(claims_table)

    return pivot_df, sequence_df
