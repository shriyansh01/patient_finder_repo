"""
This is a boilerplate pipeline 'auto_classical_pipeline'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Node, Pipeline  # noqa
from .nodes import train_pycaret_models

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(
            func=train_pycaret_models,
            inputs=["pivot_table", "params:auto_test_size", "params:auto_random_state"],
            outputs=["pycaret_models", "pycaret_predictions", "pycaret_metrics"],
            name="train_pycaret_models_node"
        )
    ])
