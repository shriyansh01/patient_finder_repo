import random
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

def generate_mock_claims(
    num_patients=100,
    records_per_patient=20,
    num_hcps=200,
    num_payers=8,
    start_date="2022-01-01",
    end_date="2025-08-01"
):
    random.seed(42)
    
    # ----- ICD-10 Diagnosis Codes (IgAN related + other kidney diseases) -----
    diagnosis_codes = {
        "N02.8": "Recurrent and persistent hematuria with other morphologic changes",
        "N02.9": "Recurrent and persistent hematuria, unspecified",
        "N03.8": "Chronic nephritic syndrome with other morphologic changes",
        "N03.9": "Chronic nephritic syndrome, unspecified",
        "N04.0": "Nephrotic syndrome with minor glomerular abnormality",
        "N04.1": "Nephrotic syndrome with focal and segmental glomerular lesions",
        "N04.2": "Nephrotic syndrome with membranous glomerulonephritis",
        "N04.3": "Nephrotic syndrome with diffuse mesangial proliferative GN",
        "N04.8": "Nephrotic syndrome with other morphologic changes",
        "N04.9": "Nephrotic syndrome, unspecified",
        "N05.8": "Unspecified nephritic syndrome with other morphologic changes",
        "N05.9": "Unspecified nephritic syndrome, unspecified",
        "N07.8": "Hereditary nephropathy, not elsewhere classified",
        "Q61.9": "Cystic kidney disease, unspecified",
        "E85.4": "Organ-limited amyloidosis",
        "I12.9": "Hypertensive CKD without heart failure",
        "I13.10": "Hypertensive heart and CKD without HF, stage 1–4",
        "N18.1": "Chronic kidney disease stage 1",
        "N18.2": "Chronic kidney disease stage 2",
        "N18.3": "Chronic kidney disease stage 3",
        "N18.4": "Chronic kidney disease stage 4",
        "N18.5": "Chronic kidney disease stage 5",
        "N18.6": "End stage renal disease",
        "N25.0": "Renal osteodystrophy",
        "N02.B9": "Recurrent IgA nephropathy",
        "N02.B1": "IgA nephropathy with glomerular lesion",
    }
    # Expand to 100+ by adding mock rare CKD codes
    while len(diagnosis_codes) < 100:
        code = f"N{random.randint(10,99)}.{random.randint(0,9)}"
        diagnosis_codes[code] = f"Rare kidney-related condition {len(diagnosis_codes)+1}"
    
    # ----- CPT Procedure Codes -----
    procedure_codes = {
        "36415": "Collection of venous blood by venipuncture",
        "80053": "Comprehensive metabolic panel",
        "81001": "Urinalysis, automated with microscopy",
        "82565": "Creatinine; blood",
        "84156": "Protein, total, urine",
        "85025": "Complete blood count (CBC)",
        "50360": "Renal allotransplantation",
        "50370": "Donor nephrectomy, open",
        "76770": "Ultrasound, retroperitoneal, complete",
        "76937": "Ultrasound guidance for vascular access",
        "85027": "Complete blood count (CBC), automated",
        "84155": "Protein, total, serum",
        "86803": "Hepatitis C antibody test",
        "87340": "Hepatitis B surface antigen test",
        "87522": "HCV RNA quantification",
        "90935": "Hemodialysis, single evaluation",
        "90937": "Hemodialysis, repeated evaluation",
        "90945": "Dialysis, single evaluation (home)",
        "50380": "Renal autotransplantation"
    }
    # Expand to 100+
    while len(procedure_codes) < 100:
        code = str(random.randint(10000,99999))
        procedure_codes[code] = f"Mock renal procedure {len(procedure_codes)+1}"
    
    # ----- Drug Names (Rx) -----
    drug_names = {
        "Prednisone": "Corticosteroid for inflammation",
        "Cyclophosphamide": "Immunosuppressant for IgAN",
        "Azathioprine": "Immunosuppressant for kidney disease",
        "Mycophenolate mofetil": "Immunosuppressant for IgAN",
        "Tacrolimus": "Immunosuppressant to prevent rejection",
        "Lisinopril": "ACE inhibitor for CKD",
        "Losartan": "ARB for CKD",
        "Enalapril": "ACE inhibitor for proteinuria",
        "Furosemide": "Diuretic for fluid retention",
        "Hydrochlorothiazide": "Thiazide diuretic",
        "Omega-3 fatty acids": "Supplement for inflammation",
        "Eculizumab": "Monoclonal antibody for IgAN",
        "Sparsentan": "Dual endothelin/angiotensin receptor antagonist",
        "Budesonide": "Targeted corticosteroid",
        "Rituximab": "B-cell depleting agent",
        "Atorvastatin": "Statin for dyslipidemia",
        "Rosuvastatin": "Statin for dyslipidemia",
        "Vitamin D": "Supplement for CKD",
        "Calcium carbonate": "Phosphate binder",
        "Sevelamer": "Phosphate binder",
        "Cinacalcet": "Calcimimetic",
        "Erythropoietin": "Anemia treatment in CKD",
        "Iron sucrose": "Iron supplement for anemia",
        "Ferrous sulfate": "Iron supplement",
        "N-acetylcysteine": "Antioxidant"
    }
    while len(drug_names) < 30:
        name = f"Drug_{len(drug_names)+1}"
        drug_names[name] = f"Mock drug for CKD treatment {len(drug_names)+1}"
    
    # ----- Payers -----
    payers = [f"Payer_{i+1}" for i in range(num_payers)]
    
    # ----- HCP IDs -----
    hcps = [f"HCP_{i+1}" for i in range(num_hcps)]
    
    # ----- Generate Data -----
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    rows = []
    for pid in range(1, num_patients+1):
        patient_id = f"PAT_{pid}"
        
        # Simulate realistic journey: dx → px → rx
        dates = sorted(
            [start_dt + timedelta(days=random.randint(0, (end_dt-start_dt).days))
             for _ in range(records_per_patient)]
        )
        
        for i, date in enumerate(dates):
            hcp = random.choice(hcps)
            payer = random.choice(payers)
            
            if i < records_per_patient * 0.3:  # Early stage: diagnosis
                val, desc = random.choice(list(diagnosis_codes.items()))
                type_ = "dx"
            elif i < records_per_patient * 0.6:  # Mid stage: procedures
                val, desc = random.choice(list(procedure_codes.items()))
                type_ = "px"
            else:  # Late stage: medications
                val, desc = random.choice(list(drug_names.items()))
                type_ = "rx"
            
            rows.append([
                patient_id, date.strftime("%Y-%m-%d"), val, desc, type_, hcp, payer
            ])
    
    df = pd.DataFrame(rows, columns=[
        "patient_id", "service_date", "value", "description", "type", "hcp_id", "payer_name"
    ])
    
    return df

def create_patient_pivot(df, values_to_pivot, current_date=None):
    if current_date is None:
        current_date = datetime.now()

    pivot_data = []

    for patient_id, group in df.groupby("patient_id"):
        patient_row = {"patient_id": patient_id}
        for val in values_to_pivot:
            val_group = group[group["value"] == val].sort_values("service_date")
            freq = len(val_group)
            gap = (val_group["service_date"].diff().dt.days.mean()
                   if freq > 1 else np.nan)
            duration = ((current_date - val_group["service_date"]).dt.days.mean()
                        if freq > 0 else np.nan)
            patient_row[f"{val}_frequency"] = freq
            patient_row[f"{val}_gap"] = gap
            patient_row[f"{val}_duration"] = duration
        pivot_data.append(patient_row)
    pivot_df = pd.DataFrame(pivot_data)
    return pivot_df

# Example usage

claims_df = generate_mock_claims(num_patients=1000, records_per_patient=5000, num_hcps=200)
claims_df["service_date"] = pd.to_datetime(claims_df["service_date"])
values_list = claims_df["value"].unique()[:100]  # first 5 codes to pivot
print('---------------------------',values_list)
pivot_df = create_patient_pivot(claims_df, values_list)
claims_df.to_csv("data/01_raw/claims_data_mock.csv", index=False)
pivot_df.to_csv("data/02_intermediate/claims_pivot_mock.csv", index=False)
# print(claims_df.head())
# print(pivot_df.head())

