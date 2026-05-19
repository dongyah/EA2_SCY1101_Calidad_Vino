"""Módulo para la definición de pipelines de machine learning supervisado y ajuste de modelos."""

from typing import Dict, Tuple
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


class WineQualityTrainer:
    """Gestiona la división de datos, pipelines de ingeniería y ajuste de modelos de clasificación."""

    def __init__(
        self, dataframe: pd.DataFrame, target_column: str = "quality"
    ) -> None:
        """Inicializa el entrenador con los datos y la especificación del objetivo.

        Args:
            dataframe (pd.DataFrame): Datos fuente limpios.
            target_column (str): Columna objetivo para la clasificación (ej. 'quality').
        """
        self.dataframe: pd.DataFrame = dataframe
        self.target_column: str = target_column
        self.pipelines: Dict[str, Pipeline] = {}

    def prepare_splits(
        self, test_size: float = 0.3, random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Separa los vectores endógenos y exógenos en matrices de entrenamiento y prueba.

        Args:
            test_size (float): Proporción del dataset que se incluirá en el split de prueba.
            random_state (int): Controla la aleatoriedad aplicada a la división de los datos.

        Returns:
            Tuple: Estructuras correspondientes a X_train, X_test, y_train, y_test.
        """
        x: pd.DataFrame = self.dataframe.drop(columns=[self.target_column])
        y: pd.Series = self.dataframe[self.target_column]

        return train_test_split(
            x, y, test_size=test_size, random_state=random_state
        )

    def build_pipelines(self, random_state: int = 42) -> None:
        """Construye los pasos de machine learning encapsulados dentro de contenedores Pipeline."""
        # Pipeline para el Árbol de Decisión
        self.pipelines["decision_tree"] = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "tree",
                    DecisionTreeClassifier(random_state=random_state),
                ),
            ]
        )

        # Pipeline para el Random Forest
        self.pipelines["random_forest"] = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "forest",
                    RandomForestClassifier(random_state=random_state),
                ),
            ]
        )

    def fit_models(
        self, x_train: pd.DataFrame, y_train: pd.Series
    ) -> Dict[str, Pipeline]:
        """Ajusta y entrena todas las arquitecturas configuradas con las particiones de entrenamiento.

        Args:
            x_train (pd.DataFrame): Matriz de características de entrenamiento.
            y_train (pd.Series): Vector objetivo de entrenamiento.

        Returns:
            Dict[str, Pipeline]: Flujos de trabajo de ejecución (pipelines) ya entrenados.
        """
        for name, pipeline in self.pipelines.items():
            pipeline.fit(x_train, y_train)
        return self.pipelines
