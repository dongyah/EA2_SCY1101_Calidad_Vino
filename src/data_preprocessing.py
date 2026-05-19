"""Módulo para la carga, diagnóstico, limpieza y auditoría de valores atípicos (outliers)."""

from typing import Tuple
import numpy as np
import pandas as pd


class DataPreprocessor:
    """Gestiona el pipeline de extremo a extremo para el preprocesamiento del dataset de vino."""

    def __init__(self, filepath: str) -> None:
        """Inicializa el preprocesador con la ruta del archivo de datos.

        Args:
            filepath (str): Ruta al archivo CSV crudo.
        """
        self.filepath: str = filepath
        self.df: pd.DataFrame = pd.DataFrame()

    def load_data(self) -> pd.DataFrame:
        """Carga el dataset desde la ruta del archivo CSV especificada.

        Returns:
            pd.DataFrame: El dataframe de Pandas con los datos crudos.
        """
        self.df = pd.read_csv(self.filepath)
        return self.df

    def check_nulls(self) -> pd.Series:
        """Realiza una verificación estricta de valores faltantes o nulos en el dataset.

        Returns:
            pd.Series: Cantidad de valores nulos por cada columna de características.
        """
        return self.df.isnull().sum()

    def remove_duplicates(self) -> Tuple[pd.DataFrame, int]:
        """Mitiga el sobreajuste eliminando las filas idénticas del dataset.

        Returns:
            Tuple[pd.DataFrame, int]: El DataFrame limpio y el número total de
            filas eliminadas.
        """
        initial_rows: int = self.df.shape[0]
        self.df = self.df.drop_duplicates()
        removed_count: int = initial_rows - self.df.shape[0]
        return self.df, removed_count

    @staticmethod
    def audit_outliers_iqr(dataframe: pd.DataFrame) -> pd.DataFrame:
        """Ejecución vectorizada del método de Rango Intercuartílico (IQR).

        Cuantifica la visibilidad absoluta y porcentual de valores extremos en las
        características, excluyendo la etiqueta objetivo (quality).

        Args:
            dataframe (pd.DataFrame): El DataFrame que se va a auditar.

        Returns:
            pd.DataFrame: Matriz de resumen ordenada con los outliers por columna numérica.
        """
        df_num: pd.DataFrame = dataframe.select_dtypes(include=[np.number]).drop(
            columns=["quality"], errors="ignore"
        )

        q1: pd.Series = df_num.quantile(0.25)
        q3: pd.Series = df_num.quantile(0.75)
        iqr: pd.Series = q3 - q1

        lower_bound: pd.Series = q1 - 1.5 * iqr
        upper_bound: pd.Series = q3 + 1.5 * iqr

        mask_outliers: pd.DataFrame = (df_num < lower_bound) | (
            df_num > upper_bound
        )

        summary: pd.DataFrame = mask_outliers.sum().reset_index()
        summary.columns = ["Variable", "Total_Outliers"]
        summary["Porcentaje"] = (
            (summary["Total_Outliers"] / len(dataframe)) * 100
        ).round(2)

        return summary.sort_values(by="Total_Outliers", ascending=False)


if __name__ == "__main__":
    # Punto de entrada para la verificación del script
    preprocessor = DataPreprocessor(filepath="data/raw/winequality-red.csv")
    raw_df = preprocessor.load_data()
    print("Resumen de entradas nulas:\n", preprocessor.check_nulls())
    cleaned_df, duplicates = preprocessor.remove_duplicates()
    print(f"Se eliminaron {duplicates} entradas duplicadas.")
    outlier_summary = preprocessor.audit_outliers_iqr(cleaned_df)
    print("Resultado de la auditoría de outliers:\n", outlier_summary)
