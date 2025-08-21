"""
This is a boilerplate pipeline 'DataIngestion'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Node, Pipeline  # noqa
from .nodes import generate_claims_data 

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(
            func=generate_claims_data,
            inputs=["params:n_patients", "params:seed"],
            outputs=["positive_cohort", "negative_cohort"],
            name="generate_claims_data_node"
        )
    ])
