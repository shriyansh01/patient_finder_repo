from kedro.pipeline import Pipeline, node
from .nodes import generate_sequences, build_patient_journeys
def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=generate_sequences,
            inputs=["positive_sequence_data", "negative_sequence_data"],
            outputs=["positive_patient_journeys", "negative_patient_journeys"],
            name="generate_sequences_node",
        ),
    ])