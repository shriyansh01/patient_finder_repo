import pandas as pd
from typing import Dict, Tuple
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb

import pandas as pd
from typing import Dict, Tuple
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def finetuning(
    models: Dict[str, object],
    pivot_data: pd.DataFrame,
    param_grid: Dict[str, dict],
) -> Tuple[Dict[str, object], pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Fine-tune ML models using hyperparameters from a YAML file.
    
    Args:
        models (dict): Dictionary of pre-initialized models.
        pivot_data (pd.DataFrame): Training data including 'target' column.
        param_grid (dict): Hyperparameter grid from YAML file.
    
    Returns:
        Tuple: (tuned_models, predictions, metrics)
    """
    data = pivot_data.copy()
    y_train = data.pop("target")
    if "patient_id" in data.columns:
        data.drop(columns=["patient_id"], inplace=True)
    data.fillna(0, inplace=True)

    tuned_models = {}
    metrics = {}
    preds = {"y_true": y_train.reset_index(drop=True)}

    for name, model in models.items():
        if name not in param_grid:
            tuned_models[name] = model
            continue

        # Convert YAML null to Python None if present
        grid_params = {k: [None if v2 is None else v2 for v2 in v] for k, v in param_grid[name].items()}

        grid = GridSearchCV(model, grid_params, cv=3, scoring="roc_auc", n_jobs=-1)
        grid.fit(data, y_train)
        best_model = grid.best_estimator_
        tuned_models[name] = best_model

        # Predictions
        y_pred = best_model.predict(data)
        y_prob = best_model.predict_proba(data)[:, 1] if hasattr(best_model, "predict_proba") else y_pred
        preds[name] = y_pred
        metrics[name] = {
            "accuracy": accuracy_score(y_train, y_pred),
            "f1": f1_score(y_train, y_pred),
            "roc_auc": roc_auc_score(y_train, y_prob),
        }
    import json
    with open("data/02_intermediate/finetuning_metrics.json", "w") as f:
        json.dump(metrics, f)
    predictions = pd.DataFrame(preds)
    return tuned_models, predictions, metrics
