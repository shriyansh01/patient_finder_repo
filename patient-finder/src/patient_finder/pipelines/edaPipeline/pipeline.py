"""
This is a boilerplate pipeline 'edaPipeline'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Node, Pipeline  # noqa
from .nodes import create_eda_pdf_report

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(
            func=create_eda_pdf_report,
            inputs=["positive_cohort", "negative_cohort"],
            outputs="eda_report_json",
            name="create_eda_pdf_report_node"
        )
    ])
