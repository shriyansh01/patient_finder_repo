import pandas as pd
from typing import Dict, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import lightgbm as lgb
import xgboost as xgb

import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import mlflow.xgboost
import json


def train_classical_models(
    features: pd.DataFrame,
    params: dict,
) -> Tuple[Dict[str, object], pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Train multiple classical ML models for binary classification with params from YAML.
    Logs metrics and models to MLflow.
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

    # Log the whole experiment under one MLflow run
    with mlflow.start_run(run_name="train_classical_models", nested=True):
        mlflow.log_params({"test_size": test_size, "random_state": random_state})

        for name, model in models.items():
            with mlflow.start_run(run_name=name, nested=True):
                # log model-specific params if available
                mlflow.log_params(model.get_params())

                # train model
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = (
                    model.predict_proba(X_test)[:, 1]
                    if hasattr(model, "predict_proba")
                    else y_pred
                )

                preds[name] = y_pred
                metrics[name] = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "f1": f1_score(y_test, y_pred),
                    "roc_auc": roc_auc_score(y_test, y_prob),
                }

                # log metrics
                mlflow.log_metrics(metrics[name])
                input_example = X_train.head(3)  # first 3 rows

                # log model artifact
                if name == "lightgbm":
                    mlflow.lightgbm.log_model(model, name)
                elif name == "xgboost":
                    mlflow.xgboost.log_model(model, name)
                else:
                    mlflow.sklearn.log_model(model, name)

    # Save metrics locally as JSON (if you still want them in data/02_intermediate)
    with open("data/02_intermediate/metrics.json", "w") as f:
        json.dump(metrics, f)

    predictions = pd.DataFrame(preds)
    return models, predictions, metrics
