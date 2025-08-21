"""
This is a boilerplate pipeline 'DataProcessing'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Node, Pipeline  # noqa
from .nodes import data_processing_pipeline

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([

        Node(
            func=data_processing_pipeline,
            inputs=["positive_cohort", "negative_cohort", "params:n_important_features"],
            outputs=["pivot_table", "sequence_table"],
            name="data_processing_node"
        )
    ])
