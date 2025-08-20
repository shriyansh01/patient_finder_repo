"""
This is a boilerplate pipeline 'explainAI'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Node, Pipeline  # noqa
from .nodes import explain_models  

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(
            func=explain_models,
            inputs=["models", "pivot_table"],
            outputs="shap_explanations",
            name="explain_models_node"
        )
    ])
