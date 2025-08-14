"""Project pipelines."""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()
    from .pipelines.sequence_data_processing import pipeline as sequence_data_pipeline
    pipelines["sequence_data_processing"] = sequence_data_pipeline.create_pipeline()
    pipelines["__default__"] = sum(pipelines.values())
    return pipelines
