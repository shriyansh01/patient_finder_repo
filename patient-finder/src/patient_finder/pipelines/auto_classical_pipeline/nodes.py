import pandas as pd
from typing import Dict, Tuple

from pycaret.classification import (
    setup,
    models,
    create_model,
    tune_model,
    finalize_model,
    predict_model,
    pull,
)


def train_all_pycaret_models(
    features: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Dict[str, object], pd.DataFrame, Dict[str, Dict[str, float]]]:
    """Train all available PyCaret classification models and collect metrics."""

    target = "target"
    if "patient_id" in features.columns:
        features = features.drop(columns=["patient_id"])
    features = features.fillna(0)

    # Initialize PyCaret
    setup(
        data=features,
        target=target,
        train_size=1 - test_size,
        session_id=random_state,
        silent=True,
        verbose=False,
    )

    # Get all classification models available in PyCaret
    available_models = models().index.tolist()

    models_dict: Dict[str, object] = {}
    metrics: Dict[str, Dict[str, float]] = {}
    preds = {"y_true": features[target].reset_index(drop=True)}

    for model_name in available_models:
        try:
            # Train, tune, and finalize
            model = create_model(model_name)
            tuned_model = tune_model(model, optimize="AUC")
            final_model = finalize_model(tuned_model)

            models_dict[model_name] = final_model

            # Predictions on holdout set
            pred_df = predict_model(final_model)

            # Save predictions
            preds[model_name] = pred_df["Label"].reset_index(drop=True)

            # Extract metrics from PyCaret
            results = pull()  # last table produced
            if not results.empty:
                metrics[model_name] = {
                    "accuracy": results.get("Accuracy", [None])[-1],
                    "f1": results.get("F1", [None])[-1],
                    "roc_auc": results.get("AUC", [None])[-1],
                }
        except Exception as e:
            print(f"⚠️ Skipping model {model_name}: {e}")
            continue

    predictions = pd.DataFrame(preds)

    return models_dict, predictions, metrics
