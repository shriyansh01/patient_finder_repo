# """Project pipelines."""

# from kedro.framework.project import find_pipelines
# from kedro.pipeline import Pipeline

# def register_pipelines() -> dict[str, Pipeline]:
#     """Register the project's pipelines.

#     Returns:
#         A mapping from pipeline names to ``Pipeline`` objects.
#     """
#     pipelines = find_pipelines()
#     print(pipelines)
#     pipelines["__default__"] = sum(pipelines.values())
#     return pipelines


"""Project pipelines."""
from __future__ import annotations

from typing import Dict
from kedro.pipeline import Pipeline

# from patient_finder.pipelines.auto_classical_pipeline import pipeline as acp
from patient_finder.pipelines.classical_model_pipeline import pipeline as cmp
from patient_finder.pipelines.DataProcessing import pipeline as dp
from patient_finder.pipelines.explainAI import pipeline as xai
from patient_finder.pipelines.finetuning import pipeline as ft
from patient_finder.pipelines.reportingPipeline import pipeline as rp
from patient_finder.pipelines.DataIngestion import pipeline as di
from patient_finder.pipelines.edaPipeline import pipeline as eda

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = {
        "DataIngestion": di.create_pipeline(),
        "edaPipeline" : eda.create_pipeline(),
        "DataProcessing": dp.create_pipeline(),
        # "auto_classical": acp.create_pipeline(),
        "classical_model_pipeline": cmp.create_pipeline(),
        "finetuning": ft.create_pipeline(),
        "explainAI": xai.create_pipeline(),
        "reportingPipeline": rp.create_pipeline(),
    }

    pipelines["__default__"] = sum(pipelines.values(), Pipeline([]))

    return pipelines
