import pandas as pd
from typing import Tuple

def build_student_features(estudiantes: pd.DataFrame, calificaciones: pd.DataFrame, asistencia: pd.DataFrame, inscripciones: pd.DataFrame) -> pd.DataFrame:
    """
    Construye un dataset a nivel de estudiante agregando las calificaciones, asistencia e inscripciones.
    """
    estudiantes = estudiantes.dropna(subset=['id_estudiante']).copy()
    estudiantes['id_estudiante'] = estudiantes['id_estudiante'].astype(int)
    estudiantes['estado_matricula'] = estudiantes['estado_matricula'].astype(str).str.strip().str.upper()
    
    calificaciones['nota'] = pd.to_numeric(calificaciones['nota'], errors='coerce')
    calificaciones_agg = calificaciones.groupby('id_inscripcion')['nota'].agg(['mean', 'count']).reset_index()
    calificaciones_agg = calificaciones_agg.rename(columns={'mean': 'nota_promedio', 'count': 'num_evaluaciones'})
    
    asistencia['presente'] = asistencia['estado_asistencia'].apply(lambda x: 1 if str(x).strip().lower() in ['presente', 'justificado'] else 0)
    asistencia_agg = asistencia.groupby('id_inscripcion')['presente'].agg(['mean', 'count']).reset_index()
    asistencia_agg = asistencia_agg.rename(columns={'mean': 'porcentaje_asistencia', 'count': 'clases_totales'})
    
    inscripciones_full = inscripciones.merge(calificaciones_agg, on='id_inscripcion', how='left')
    inscripciones_full = inscripciones_full.merge(asistencia_agg, on='id_inscripcion', how='left')
    
    student_agg = inscripciones_full.groupby('id_estudiante').agg(
        total_asignaturas=('id_inscripcion', 'count'),
        promedio_general=('nota_promedio', 'mean'),
        asistencia_general=('porcentaje_asistencia', 'mean'),
        evaluaciones_totales=('num_evaluaciones', 'sum')
    ).reset_index()
    
    df_final = estudiantes.merge(student_agg, on='id_estudiante', how='left')
    df_final['target'] = (df_final['estado_matricula'] == 'DESERTOR').astype(int)
    
    # Rellenar nulos con medianas simples por ahora, para tener dataset íntegro. 
    # El preprocessor de Sklearn también lo hará, pero Kedro prefiere datasets limpios en preprocesamiento si es simple.
    return df_final
