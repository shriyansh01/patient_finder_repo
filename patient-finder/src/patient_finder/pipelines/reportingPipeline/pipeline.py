"""
This is a boilerplate pipeline 'reportingPipeline'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Node, Pipeline  # noqa
from .nodes import generate_report  # noqa

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(
            func=generate_report,
            inputs=[
                "code_variance_table",
                "metrics",
                "finetuned_metrics",
            ],
            outputs="business_report_json",
            name="generate_report_node",
        )
    ])
