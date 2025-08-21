"""
This is a boilerplate pipeline 'DataIngestion'
generated using Kedro 1.0.0
"""

# src/<your_project>/pipelines/data_ingestion/nodes.py
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_claims_data(n_patients: int = 200, seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)

    # Patient demographics
    genders = ["M", "F"]
    ages = np.random.randint(18, 80, size=n_patients)

    # Diagnosis Codes (ICD-10)
    dx_codes_igan = ["N02.8", "N03.2", "N04.9"]
    dx_codes_potential = ["N18.9", "R80.9", "N17.9"]
    all_dx_codes = dx_codes_igan + dx_codes_potential

    # Procedure Codes (CPT)
    px_codes = ["81001", "80069", "36415", "50360"]

    # Prescription Codes (NDC)
    rx_codes = ["00093-7424", "00781-5184", "00054-0450", "00093-3109"]

    # Service date range
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)

    # Patient IDs
    patient_ids = [f"PT{str(i).zfill(4)}" for i in range(1, n_patients + 1)]

    records = []
    for idx, pid in enumerate(patient_ids):
        age = ages[idx]
        gender = random.choice(genders)
        n_claims = random.randint(3, 9)
        target_patient = random.random() < 0.5  # 50% prevalence

        for _ in range(n_claims):
            service_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))

            # Weighted sampling of diagnosis codes
            if target_patient:
                dx_code = random.choices(
                    all_dx_codes,
                    weights=[0.6 if code in dx_codes_igan else 0.4 for code in all_dx_codes],
                    k=1
                )[0]
            else:
                dx_code = random.choices(
                    all_dx_codes,
                    weights=[0.3 if code in dx_codes_igan else 0.7 for code in all_dx_codes],
                    k=1
                )[0]

            # Pick ONE code type per claim
            code_type = random.choice(["DX", "PX", "RX"])
            code = dx_code if code_type == "DX" else random.choice(px_codes if code_type == "PX" else rx_codes)

            records.append({
                "patient_id": pid,
                "age": age,
                "gender": gender,
                "service_date": service_date.strftime("%Y-%m-%d"),
                "code": code,
                "code_type": code_type,
                "target": target_patient
            })

    claims_df = pd.DataFrame(records)
    positive_cohort = claims_df[claims_df["target"] == True]
    negative_cohort = claims_df[claims_df["target"] == False]

    return positive_cohort, negative_cohort
