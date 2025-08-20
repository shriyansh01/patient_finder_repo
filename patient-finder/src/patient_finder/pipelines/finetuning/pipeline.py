"""
This is a boilerplate pipeline 'finetuning'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Node, Pipeline  # noqa
from .nodes import finetuning

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(
            func=finetuning,
            inputs=["models", "pivot_table","params:finetuning"],
            outputs=["finetuned_models", "finetuned_predictions", "finetuned_metrics"],
            name="finetuning_node",
        )
    ])



