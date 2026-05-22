# Predicción y Prevención de Deserción Estudiantil (Prueba II)

Este proyecto desarrolla un ecosistema integral de Machine Learning para predecir el riesgo de deserción y el rendimiento académico de los estudiantes. Utiliza técnicas avanzadas de Aprendizaje Supervisado, No Supervisado y Aprendizaje por Refuerzo (RL), todo orquestado bajo el framework de ingeniería de datos **Kedro**.

## 1. Organización del Proyecto

Aunque la pauta recomienda una estructura plana en `src/`, este proyecto ha optado por buenas prácticas de la industria (MLOps) utilizando **Kedro**, lo que garantiza una modularidad, reproducibilidad y evitar el *data leakage* de manera muy superior.

*   `data/`: Contiene los conjuntos de datos sin procesar, intermedios y finales. (Excluidos de git por seguridad).
*   `notebooks/`:
    *   `01_exploratory_analysis.ipynb`: Análisis exploratorio y limpieza inicial.
    *   `02_supervised_modeling.ipynb`: Implementación interactiva.
    *   `03_model_evaluation.ipynb`: Evaluación y carga del catálogo de modelos.
    *   `04_hyperparameter_optimization.ipynb`: Optimización (GridSearch y RandomizedSearch).
    *   `05_final_analysis.ipynb`: Discusión y cierre.
    *   `06_metrics_visualization.ipynb`: Gráficos comparativos de los 16 modelos.
*   `conf/base/catalog.yml`: El diccionario de datos y modelos del proyecto.
*   `src/pruebaii/pipelines/data_science/`:
    *   `nodes.py`: Contiene las funciones modulares (`data_preprocessing`, `model_training`, `model_evaluation` y `hyperparameter_tuning`).
    *   `pipeline.py`: Enruta la secuencia de ejecución (DAG).
*   `models/trained_models/`: Archivos `.pkl` serializados de los 16 algoritmos entrenados.
*   `results/metrics/`: Exportación de métricas en `.csv` para visualización.
*   `informe_tecnico.md`: Documento formal de justificación y resultados (>15 páginas).

## 2. Dependencias del Proyecto

El proyecto requiere un entorno virtual con Python 3.10 o superior (recomendado vía Anaconda).

Para instalar las dependencias exactas, abre tu terminal en la raíz del proyecto y ejecuta:

```bash
pip install -r requirements.txt
```

Las dependencias clave incluyen:
*   `kedro`: Orquestador del pipeline.
*   `scikit-learn`: Para los modelos Supervisados y No Supervisados.
*   `xgboost` y `lightgbm`: Modelos basados en Gradient Boosting.
*   `tpot`: Algoritmos genéticos para AutoML.
*   `stable-baselines3` y `gymnasium`: Para los Agentes de Aprendizaje por Refuerzo (RL).
*   `pandas`, `numpy`, `matplotlib`, `seaborn`: Stack clásico de datos y visualización.
*   `ipykernel`: Para enlazar el entorno virtual con los Jupyter Notebooks.

## 3. Instrucciones de Uso y Reproducibilidad

Este proyecto es **100% reproducible**. 

### 3.1 Ejecutar el Pipeline Completo
No necesitas correr scripts individuales. El framework Kedro se encarga de cargar los datos, transformarlos, entrenar los 16 modelos de manera secuencial y exportar las métricas de rendimiento en un solo paso.

Ejecuta en tu terminal (con tu entorno virtual activado):
```bash
kedro run
```
*Tiempo estimado de ejecución: ~45 segundos.*

### 3.2 Visualizar Resultados
Una vez que el pipeline finaliza, puedes revisar las métricas ejecutando el cuaderno final:
```bash
jupyter notebook notebooks/06_metrics_visualization.ipynb
```
*(Asegúrate de seleccionar el kernel del entorno virtual `Python (PruebaII)`).*

## 4. Modelos Implementados (16 en total)
1.  **Supervisados:** Random Forest, XGBoost, SVM, Regresión Logística, KNN, TPOT (AutoML).
2.  **No Supervisados:** K-Means, DBSCAN, Agglomerative Clustering, Gaussian Mixture (GMM), Isolation Forest.
3.  **Por Refuerzo (RL):** PPO, A2C, DQN, SAC, DDPG (vía entornos continuos y discretos de Gymnasium).

---
*Desarrollado para la Evaluación Parcial N°2 - Programación para la Ciencia de Datos (SCY1101).*
