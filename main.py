
import pandas as pd
data = pd.read_parquet("data/07_model_output/positive_patient_journeys.parquet")
print(data.head())