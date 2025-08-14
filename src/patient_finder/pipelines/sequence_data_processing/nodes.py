import pandas as pd
import json
def build_patient_journeys(df):
    """
    Create a patient journey for each patient by concatenating
    diagnosis, procedure, and product (RX) codes in chronological order.
    Returns a dictionary {patient_id: [sequence_of_codes]}.
    """

    # Ensure date type
    df["TRANSACTION_DATE"] = pd.to_datetime(df["TRANSACTION_DATE"], errors="coerce")

    # Sort by patient and date
    df = df.sort_values(["PATIENT_ID", "TRANSACTION_DATE"])

    patient_journeys = {}

    for patient_id, group in df.groupby("PATIENT_ID"):
        sequence = []

        for _, row in group.iterrows():
            # Add diagnosis codes (skip NaN)
            for col in ["DIAGNOSIS_CODE_1", "DIAGNOSIS_CODE_2", "DIAGNOSIS_CODE_3", "DIAGNOSIS_CODE_4"]:
                if pd.notna(row[col]):
                    sequence.append(f"DX_{row[col]}")

            # Add procedure codes if present
            if "PROCEDURE_CODE" in row and pd.notna(row["PROCEDURE_CODE"]):
                sequence.append(f"PX_{row['PROCEDURE_CODE']}")

            # Add product (medication) IDs if present
            if "PRODUCT_NAME" in row and pd.notna(row["PRODUCT_NAME"]):
                sequence.append(f"RX_{row['PRODUCT_NAME']}")

        patient_journeys[patient_id] = sequence
    return patient_journeys


def generate_sequences(df_positive, df_negative):
    """
    Build patient journeys from the input DataFrames and pad them to the same length.
    Returns two DataFrames: positive_df, negative_df
    """
    # Build patient journeys
    positive_journeys = build_patient_journeys(df_positive)
    negative_journeys = build_patient_journeys(df_negative)

    # Determine max sequence length across both groups
    max_length = max(
        len(seq)
        for seq in list(positive_journeys.values()) + list(negative_journeys.values())
    )

    # Pad sequences to match max_length
    for patient_id in positive_journeys:
        positive_journeys[patient_id] += ["PAD"] * (max_length - len(positive_journeys[patient_id]))
    for patient_id in negative_journeys:
        negative_journeys[patient_id] += ["PAD"] * (max_length - len(negative_journeys[patient_id]))

    # Convert to DataFrames (patients as columns, sequence positions as rows)
    positive_df = pd.DataFrame({k: pd.Series(v) for k, v in positive_journeys.items()})
    negative_df = pd.DataFrame({k: pd.Series(v) for k, v in negative_journeys.items()})

    return positive_df, negative_df
