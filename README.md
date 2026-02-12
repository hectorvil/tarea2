# Pronóstico de demanda mensual con LightGBM

Este repositorio implementa un pipeline para pronosticar la **demanda mensual** por combinación **producto–tienda** usando el dataset de 1C Company (Kaggle: *Predict Future Sales*). El objetivo es apoyar decisiones de inventario: **cuánto pedir y dónde colocarlo con anticipación**, buscando reducir sobrestock y disminuir ventas perdidas, además de automatizar un proceso que podría depender de promedios móviles y ajustes manuales.

El modelo principal es un enfoque en **dos etapas**: primero estima si habrá venta con una clasificación y después estima cuántas unidades se venderán con una regresión. Esto es especialmente útil cuando la demanda es **esporádica e intermitente**.

## Resultados
- El desempeño agregado cumple la meta operativa: **RMSE cercano a 1**, lo cual está por debajo del umbral de 5 unidades.
- Aun así, el análisis por segmentos muestra que el modelo puede **subestimar picos de demanda**. En producción conviene complementar con criterio de negocio y/o otros criterios, tales como inventario de seguridad para productos prioritarios.

---

## Estructura del repositorio

```text
.
├── artifacts
│   └── model.joblib
├── data
│   ├── inference
│   │   └── test_features.parquet
│   ├── predictions
│   │   └── submission.csv
│   ├── prep
│   │   ├── meta.json
│   │   ├── test_features.parquet
│   │   ├── test_pairs.parquet
│   │   ├── train.parquet
│   │   └── valid.parquet
│   └── raw
│       ├── sales_train.csv
│       └── test.csv
├── notebooks
│   ├── Entendimientodelos_datosEDA.ipynb
│   ├── FeatureEngineering.ipynb
│   ├── Modeling.ipynb
│   └── SimulationComparation.ipynb
├── pyproject.toml
├── src
│   ├── __init__.py
│   ├── inference.py
│   ├── prep.py
│   └── train.py
└── tree.txt
---
## Detalle

### Notebooks

- **`notebooks/Entendimientodelos_datosEDA.ipynb`**  
  Exploración del dataset: nulos, rangos, outliers, devoluciones, agregación mensual, estacionalidad e intermitencia (recency, meses con venta).

- **`notebooks/FeatureEngineering.ipynb`**  
  Construcción de features para series de tiempo (lags, ventanas recientes, recency, intermitencia, señales de precio y agregados laggeados) y guardado de base intermedia en `data/prep/`.

- **`notebooks/Modeling.ipynb`**  
  Entrenamiento del modelo final (clasificación + regresión), generación de predicciones y guardado de `artifacts/model.joblib` y `data/predictions/submission.csv`.

- **`notebooks/SimulationComparation.ipynb`**  
  Evaluación: calibración por deciles, análisis, comparación contra baseline y simulación operativa (sobrestock vs stockouts) con análisis de sensibilidad.

---

### Scripts (pipeline automatizable)

Los scripts se ejecutan desde la raíz del repo y siguen la estructura antes mencionada:

- **`src/prep.py`**  
  - Entrada: `data/raw/`  
  - Salida: `data/prep/` (train/valid/test_features + meta)

- **`src/train.py`**  
  - Entrada: `data/prep/`  
  - Salida: `artifacts/model.joblib`

- **`src/inference.py`**  
  - Entrada: `data/inference/` + `artifacts/model.joblib`  
  - Salida: `data/predictions/submission.csv`

---

## Cómo ejecutar el pipeline con uv

### Preprocesamiento y features
```bash
uv run python -m src.prep

### Entrenamiento
uv run python -m src.train
### Inference batch
uv run python -m src.inference
### Outputs esperados
-data/prep/train.parquet, data/prep/valid.parquet, data/prep/test_features.parquet, data/prep/test_pairs.parquet, data/prep/meta.json

-artifacts/model.joblib

-data/predictions/submission.csv

Referencias

Manokhin, V. (n.d.). Mastering modern time series forecasting: A comprehensive guide to statistical, machine learning, and deep learning models in Python (Early Access). Leanpub.

OpenAI. (2023). ChatGPT (Mar 14 version) [Large language model versión 5.2]. https://chat.openai.com/



