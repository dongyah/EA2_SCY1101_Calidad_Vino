# 🍷 Proyecto: Predicción y Segmentación de la Calidad del Vino Tinto (Vinho Verde)

**Asignatura:** SCY1101 - Programación para la Ciencia de Datos (Duoc UC)
**Estudiante:** Belén Toloza
**Docente:** Mauricio González V.

## 📝 Sobre este Proyecto
Este repositorio contiene mi Evaluación Parcial 2. El objetivo principal de este proyecto es analizar y modelar un dataset de vinos tintos de la variante "Vinho Verde" del norte de Portugal. 

El proyecto aborda un problema de negocio real utilizando Machine Learning y está dividido en dos grandes enfoques:
1. **Aprendizaje Supervisado:** Creación y optimización de un modelo predictivo (Random Forest ajustado con `GridSearchCV`) capaz de automatizar la clasificación de la calidad del vino (notas del 3 al 8) basándose en sus 11 componentes químicos, superando el gran desafío del desbalanceo de clases.
2. **Aprendizaje No Supervisado:** Un experimento "a ciegas" utilizando `K-Means` y `PCA` (reducción de dimensionalidad) para descubrir perfiles comerciales de forma automática, logrando aislar matemáticamente el perfil del "Vino Premium" frente al "Económico".

## 💻 Justificación del Entorno de Trabajo
Para el desarrollo de este proyecto decidí utilizar **Jupyter Notebook** (a través de Google Colab). Elegí este entorno por tres grandes razones:
* **Interactividad visual:** Me permite combinar bloques de código con celdas de texto (Markdown) y ver mis gráficos (como los Boxplots, Heatmaps y el mapa 2D del PCA) inmediatamente debajo del código.
* **Experimentación paso a paso:** Es ideal para la Ciencia de Datos porque puedo limpiar y transformar los datos celda por celda sin tener que correr todo el programa desde cero cada vez.
* **Integración con GitHub:** Facilita muchísimo la subida y el control de versiones directamente al repositorio, garantizando que mi código sea 100% reproducible.

## 📁 Estructura del Repositorio
Para garantizar la modularidad y reproducibilidad que exige el proyecto, las carpetas están organizadas de la siguiente manera:

* **`data/`**: Contiene los datos crudos (`raw/winequality-red.csv`) y los datos limpios tras la Fase 1 (`processed/winequality_clean.csv`).
* **`notebooks/`**: Contiene los 5 cuadernos de análisis paso a paso:
  * `01_exploratory_analysis.ipynb` (Limpieza y EDA)
  * `02_supervised_modeling.ipynb` (Modelos base)
  * `03_model_evaluation.ipynb` (Métricas y evaluación)
  * `04_hyperparameter_optimization.ipynb` (Optimización de hiperparámetros)
  * `05_final_analysis.ipynb` (K-Means, PCA y conclusiones)
* **`src/`**: Scripts modulares de Python (`.py`) para automatizar el preprocesamiento, entrenamiento y evaluación.
* **`models/`**: Carpeta designada para exportar y guardar los modelos entrenados serializados.
* **`results/`**: Contiene los gráficos clave (`plots/`), las métricas (`metrics/`) y el informe técnico final en formato documento (`reports/`).

## 🚀 Cómo reproducir este proyecto
Todo el código fue diseñado para ejecutarse de principio a fin sin errores. 
1. Clona este repositorio en tu entorno local o ábrelo en Google Colab.
2. Asegúrate de instalar las librerías necesarias (`pandas`, `numpy`, `matplotlib`, `seaborn` y `scikit-learn`).
