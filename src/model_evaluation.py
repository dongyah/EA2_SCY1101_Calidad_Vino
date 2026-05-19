"""Módulo para la auditoría de rendimiento de modelos, extracción de métricas y validación cruzada."""

from typing import Dict
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


class ModelAuditor:
    """Evalúa los modelos entrenados usando métricas estadísticas profundas y bucles de validación."""

    @staticmethod
    def run_cross_validation(
        pipelines: Dict[str, Pipeline],
        x_train: pd.DataFrame,
        y_train: pd.Series,
        cv: int = 5,
    ) -> Dict[str, np.ndarray]:
        """Aplica bucles de Validación Cruzada Estratificada (K-Fold) sobre las matrices de entrenamiento.

        Args:
            pipelines (Dict[str, Pipeline]): Modelos de producción configurados.
            x_train (pd.DataFrame): Matriz de características de entrenamiento.
            y_train (pd.Series): Vector objetivo de entrenamiento.
            cv (int): Número de pliegues (folds) a particionar.

        Returns:
            Dict[str, np.ndarray]: Colección de puntajes (scores) obtenidos para cada pliegue.
        """
        cv_scores: Dict[str, np.ndarray] = {}
        for name, pipeline in pipelines.items():
            scores: np.ndarray = cross_val_score(
                pipeline, x_train, y_train, cv=cv
            )
            cv_scores[name] = scores
        return cv_scores

    @staticmethod
    def compute_test_metrics(
        pipelines: Dict[str, Pipeline], x_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, Dict[str, any]]:
        """Extrae reportes completos de rendimiento para los entornos de prueba/validación.

        Args:
            pipelines (Dict[str, Pipeline]): Pipelines de los modelos evaluados.
            x_test (pd.DataFrame): Matriz de características de validación.
            y_test (pd.Series): Valores objetivo de validación.

        Returns:
            Dict[str, Dict[str, any]]: Métricas y matrices de confusión detalladas por modelo.
        """
        evaluation_results: Dict[str, Dict[str, any]] = {}

        for name, pipeline in pipelines.items():
            predictions: np.ndarray = pipeline.predict(x_test)
            accuracy: float = accuracy_score(y_test, predictions)
            f1: float = f1_score(y_test, predictions, average="weighted")
            matrix: np.ndarray = confusion_matrix(y_test, predictions)

            evaluation_results[name] = {
                "accuracy": accuracy,
                "f1_score_weighted": f1,
                "confusion_matrix": matrix,
            }

        return evaluation_results
