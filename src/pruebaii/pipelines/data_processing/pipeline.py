from kedro.pipeline import Pipeline, node, pipeline
from .nodes import build_student_features

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=build_student_features,
                inputs=["estudiantes", "calificaciones", "asistencia", "inscripciones"],
                outputs="preprocessed_data",
                name="build_student_features_node",
            )
        ]
    )
