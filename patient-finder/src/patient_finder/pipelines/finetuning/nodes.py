import pandas as pd
from typing import Dict, Tuple
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import mlflow.xgboost

import json


def finetuning(
    models: Dict[str, object],
    pivot_data: pd.DataFrame,
    param_grid: Dict[str, dict],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Dict[str, object], pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Fine-tune ML models using hyperparameters from a YAML file.
    Logs parameters, metrics, and tuned models to MLflow.

    Args:
        models (dict): Dictionary of pre-initialized models.
        pivot_data (pd.DataFrame): Training data including 'target' column.
        param_grid (dict): Hyperparameter grid from YAML file.
        test_size (float): Fraction of data to hold out for validation.
        random_state (int): Random seed for reproducibility.

    Returns:
        Tuple: (tuned_models, predictions, metrics)
    """
    data = pivot_data.copy()
    y = data.pop("target")
    
    if "patient_id" in data.columns:
        data.drop(columns=["patient_id"], inplace=True)
    
    data.fillna(0, inplace=True)

    # Split data into train and validation to avoid overfitting
    X_train, X_val, y_train, y_val = train_test_split(
        data, y, test_size=test_size, random_state=random_state, stratify=y
    )

    tuned_models = {}
    metrics = {}
    preds = {"y_true": y_val.reset_index(drop=True)}

    with mlflow.start_run(run_name="finetuning", nested=True):
        for name, model in models.items():
            if name not in param_grid:
                tuned_models[name] = model
                continue

            # Convert YAML null -> Python None
            grid_params = {
                k: [None if v2 is None else v2 for v2 in v]
                for k, v in param_grid[name].items()
            }

            grid = GridSearchCV(model, grid_params, cv=3, scoring="roc_auc", n_jobs=-1)

            with mlflow.start_run(run_name=f"{name}_finetune", nested=True):
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_
                tuned_models[name] = best_model

                # Log best parameters and CV score
                mlflow.log_params(grid.best_params_)
                mlflow.log_metric("cv_best_score", grid.best_score_)

                # Predictions on validation set
                y_pred = best_model.predict(X_val)
                y_prob = (
                    best_model.predict_proba(X_val)[:, 1]
                    if hasattr(best_model, "predict_proba")
                    else y_pred
                )
                preds[name] = y_pred
                metrics[name] = {
                    "accuracy": accuracy_score(y_val, y_pred),
                    "f1": f1_score(y_val, y_pred),
                    "roc_auc": roc_auc_score(y_val, y_prob),
                }

                # Log metrics
                mlflow.log_metrics(metrics[name])

                # Log the tuned model
                if name.lower() == "lightgbm":
                    mlflow.lightgbm.log_model(best_model, f"{name}_finetuned")
                elif name.lower() == "xgboost":
                    mlflow.xgboost.log_model(best_model, f"{name}_finetuned")
                else:
                    mlflow.sklearn.log_model(best_model, f"{name}_finetuned")

    # Save metrics locally
    with open("data/02_intermediate/finetuning_metrics.json", "w") as f:
        json.dump(metrics, f)

    predictions = pd.DataFrame(preds)
    return tuned_models, predictions, metrics
