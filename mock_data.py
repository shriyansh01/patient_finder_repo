import pandas as pd 
import numpy as np
import random
from datetime import datetime, timedelta

# ---------------------------
# 1. Configuration
# ---------------------------
N_PATIENTS = 200
np.random.seed(42)
random.seed(42)

# Patient demographics
genders = ["M", "F"]
ages = np.random.randint(18, 80, size=N_PATIENTS)

# Diagnosis Codes (ICD-10)
dx_codes_igan = ["N02.8", "N03.2", "N04.9"]   # IgA nephropathy & chronic nephritis
dx_codes_potential = ["N18.9", "R80.9", "N17.9"]  # CKD, proteinuria, acute renal failure
all_dx_codes = dx_codes_igan + dx_codes_potential

# Procedure Codes (CPT)
px_codes = ["81001", "80069", "36415", "50360"]

# Prescription Codes (NDC examples for CKD/IgAN related drugs)
rx_codes = ["00093-7424", "00781-5184", "00054-0450", "00093-3109"]

# Service date range
start_date = datetime(2020, 1, 1)
end_date = datetime(2023, 12, 31)

# ---------------------------
# 2. Patient list
# ---------------------------
patient_ids = [f"PT{str(i).zfill(4)}" for i in range(1, N_PATIENTS + 1)]

# ---------------------------
# 3. Generate claims
# ---------------------------
records = []

for idx, pid in enumerate(patient_ids):
    age = ages[idx]
    gender = random.choice(genders)
    
    # Each patient has 3–9 claims
    n_claims = random.randint(3, 9)
    
    # Define whether this patient is IgAN-positive (class 1)
    is_igan_patient = random.random() < 0.5   # 50% prevalence
    
    for _ in range(n_claims):
        service_date = start_date + timedelta(
            days=random.randint(0, (end_date - start_date).days)
        )

        # Weighted sampling of diagnosis codes
        if is_igan_patient:
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

        # Pick ONE code type per claim (like real claims: either Dx, Px, or Rx)
        code_type = random.choice(["DX", "PX", "RX"])
        if code_type == "DX":
            code = dx_code
        elif code_type == "PX":
            code = random.choice(px_codes)
        else:
            code = random.choice(rx_codes)

        # Append claim record
        records.append({
            "patient_id": pid,
            "age": age,
            "gender": gender,
            "service_date": service_date.strftime("%Y-%m-%d"),
            "code": code,
            "code_type": code_type,
            "is_igan": is_igan_patient
        })

# ---------------------------
# 4. Create DataFrame
# ---------------------------
claims_df = pd.DataFrame(records)

# ---------------------------
# 5. Split into cohorts
# ---------------------------
positive_cohort = claims_df[claims_df["is_igan"] == True]
negative_cohort = claims_df[claims_df["is_igan"] == False]

# ---------------------------
# 6. Save to CSV files
# ---------------------------
positive_cohort.to_csv("patient-finder/data/01_raw/positive_cohort.csv", index=False)
negative_cohort.to_csv("patient-finder/data/01_raw/negative_cohort.csv", index=False)

(len_positive, len_negative) = (len(positive_cohort), len(negative_cohort))
(len_positive, len_negative)

