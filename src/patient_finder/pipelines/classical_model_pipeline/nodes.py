import pandas as pd
from typing import Dict, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import lightgbm as lgb
import xgboost as xgb


def train_classical_models(
    features: pd.DataFrame,
    params: dict,
) -> Tuple[Dict[str, object], pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Train multiple classical ML models for binary classification with params from YAML.
    
    Args:
        features (pd.DataFrame): Input dataset with 'target' and 'patient_id'.
        params (dict): Parameters from YAML (test_size, random_state, model hyperparameters).
    
    Returns:
        Tuple: (trained_models, predictions, metrics)
    """
    test_size = params.get("test_size", 0.2)
    random_state = params.get("random_state", 42)
    model_params = params.get("models", {})

    data = features.copy()
    target = data.pop("target")
    if "patient_id" in data.columns:
        data.drop(columns=["patient_id"], inplace=True)
    data.fillna(0, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=test_size, random_state=random_state, stratify=target
    )

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=model_params.get("logistic_regression", {}).get("max_iter", 1000),
            random_state=random_state
        ),
        "decision_tree": DecisionTreeClassifier(
            random_state=random_state
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=model_params.get("random_forest", {}).get("n_estimators", 200),
            random_state=random_state
        ),
        "lightgbm": lgb.LGBMClassifier(random_state=random_state),
        "xgboost": xgb.XGBClassifier(
            use_label_encoder=model_params.get("xgboost", {}).get("use_label_encoder", False),
            eval_metric=model_params.get("xgboost", {}).get("eval_metric", "logloss"),
            random_state=random_state,
        ),
    }

    metrics = {}
    preds = {"y_true": y_test.reset_index(drop=True)}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        preds[name] = y_pred
        metrics[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob),
        }

    predictions = pd.DataFrame(preds)
    return models, predictions, metrics
