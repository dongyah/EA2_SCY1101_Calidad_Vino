"""Módulo para la optimización de hiperparámetros y flujos de análisis de aprendizaje no supervisado."""

from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class HyperparameterOptimizer:
    """Gestiona los bucles de optimización mediante búsqueda en rejilla (Grid Search) para producción."""

    @staticmethod
    def optimize_random_forest(
        x_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Dict[str, List[Union[int, str, None]]],
        cv: int = 5,
    ) -> Tuple[Pipeline, Dict[str, any]]:
        """Ejecuta una búsqueda en rejilla sobre el espacio de hiperparámetros de un Random Forest.

        Args:
            x_train (pd.DataFrame): Matrices del conjunto de entrenamiento.
            y_train (pd.Series): Valores objetivo del conjunto de entrenamiento.
            param_grid (Dict): Espacio de búsqueda estructurado.
            cv (int): Índice de pliegues para la validación cruzada.

        Returns:
            Tuple[Pipeline, Dict]: El pipeline óptimo ajustado y el diccionario
            con los mejores parámetros encontrados.
        """
        # Se define la estructura base del pipeline de forma nativa
        base_pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "forest",
                    None,  # Se instancia mediante la referencia en el espacio de parámetros del grid
                ),
            ]
        )

        grid_search = GridSearchCV(
            estimator=base_pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
        )
        grid_search.fit(x_train, y_train)

        return grid_search.best_estimator_, grid_search.best_params_


class UnsupervisedAnalyzer:
    """Ejecuta análisis de clustering y reducciones estructurales de dimensionalidad."""

    def __init__(self, dataframe: pd.DataFrame) -> None:
        """Inicializa el analizador abstrayendo y eliminando la columna objetivo.

        Args:
            dataframe (pd.DataFrame): Registros completos de la fuente de datos.
        """
        # Aplica restricciones de aprendizaje no supervisado eliminando la etiqueta objetivo
        self.data: pd.DataFrame = dataframe.drop(
            columns=["quality"], errors="ignore"
        )
        self.scaled_data: np.ndarray = np.empty((0, 0))

    def scale_features(self) -> np.ndarray:
        """Escala las variaciones numéricas para obtener media cero y varianza unitaria.

        Returns:
            np.ndarray: Matriz que contiene las filas de características escaladas.
        """
        scaler = StandardScaler()
        self.scaled_data = scaler.fit_transform(self.data)
        return self.scaled_data

    def compute_elbow_inertia(self, max_k: int = 10) -> List[float]:
        """Calculates los errores de distancia (inercia) desde los centros de los clusters para distintos K.

        Args:
            max_k (int): Límite superior para la iteración del número de clusters.

        Returns:
            List[float]: Resultados de inercia calculados para las curvas de validación (Método del Codo).
        """
        inertia_scores: List[float] = []
        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
            kmeans.fit(self.scaled_data)
            inertia_scores.append(kmeans.inertia_)
        return inertia_scores

    def apply_pca_reduction(
        self, n_components: int = 2
    ) -> Tuple[PCA, pd.DataFrame]:
        """Realiza la compresión del espacio mediante Análisis de Componentes Principales (PCA).

        Args:
            n_components (int): Número objetivo de componentes principales.

        Returns:
            Tuple[PCA, pd.DataFrame]: El objeto PCA ajustado y las dimensiones reducidas
            estructuradas en un DataFrame de Pandas.
        """
        pca = PCA(n_components=n_components, random_state=42)
        reduced_array = pca.fit_transform(self.scaled_data)

        columns = [f"PC{i+1}" for i in range(n_components)]
        reduced_df = pd.DataFrame(reduced_array, columns=columns)

        return pca, reduced_df
