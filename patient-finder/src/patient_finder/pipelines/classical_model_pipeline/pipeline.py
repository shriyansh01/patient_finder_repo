"""
This is a boilerplate pipeline 'classical_model_pipeline'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Node, Pipeline  # noqa
from .nodes import train_classical_models

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
            Node(
                func=train_classical_models,
                inputs=["pivot_table","params:classical_training"],
                outputs=["models", "predictions", "metrics"],
                name="train_classical_model_node"
            )

    ])
