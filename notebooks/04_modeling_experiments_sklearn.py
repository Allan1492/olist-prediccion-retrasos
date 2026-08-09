# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Experimentos de modelado (sklearn + pandas, optimizado para memoria)
# MAGIC
# MAGIC Entrada: `big_data_2026.olist.gold_ml_features`
# MAGIC Salida: runs en MLflow + modelo campeón registrado en Unity Catalog
# MAGIC
# MAGIC Optimizaciones: Uso de tipos de datos reducidos (float32), liberación de memoria (gc), 
# MAGIC y modelos más ligeros (menos árboles, menor profundidad) para evitar OOM en clusters gratuitos.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuracion

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import gc
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

CATALOG, SCHEMA = "big_data_2026", "olist"
NOMBRE_MODELO_UC = f"{CATALOG}.{SCHEMA}.olist_delay_predictor"
SEED = 42

# Función auxiliar para optimizar memoria del DataFrame
def optimize_memory(df):
    for col in df.columns:
        if col not in ['is_late', 'split_temporal']: # Evitar tocar la target o el split
            col_type = df[col].dtype
            if col_type == 'object':
                df[col] = df[col].astype('category')
            elif pd.api.types.is_numeric_dtype(col_type):
                c_min = df[col].min()
                c_max = df[col].max()
                if pd.api.types.is_integer_dtype(col_type):
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.float32)
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Cargar Gold a pandas + contrato

# COMMAND ----------

pdf = spark.table(f"{CATALOG}.{SCHEMA}.gold_ml_features").toPandas()
assert {"order_id", "is_late", "split_temporal"} <= set(pdf.columns)

FUGA = {"order_delivered_customer_date", "days_delay", "order_status",
        "review_score", "order_delivered_carrier_date"}
assert not (FUGA & set(pdf.columns)), f"Fuga detectada: {FUGA & set(pdf.columns)}"

NO_FEATURE = {"order_id", "is_late", "split_temporal", "order_purchase_timestamp"}
cat_cols = [c for c in pdf.columns if c not in NO_FEATURE and pdf[c].dtype == object]
num_cols = [c for c in pdf.columns if c not in NO_FEATURE and c not in cat_cols
            and pd.api.types.is_numeric_dtype(pdf[c])]

# Optimizar memoria antes de separar
pdf = optimize_memory(pdf)

train = pdf[pdf.split_temporal == "train"]
test  = pdf[pdf.split_temporal == "test"]
X_train, y_train = train[cat_cols + num_cols], train.is_late
X_test,  y_test  = test[cat_cols + num_cols],  test.is_late

# Liberar memoria del dataframe completo original
del pdf, train, test
gc.collect()

print(f"Features: {len(num_cols)} num + {len(cat_cols)} cat | train={len(X_train):,} | test={len(X_test):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Evaluacion

# COMMAND ----------

def evaluar(y_true, proba, thr):
    pred = (proba >= thr).astype(int)
    return {"pr_auc": average_precision_score(y_true, proba),
            "roc_auc": roc_auc_score(y_true, proba),
            "f1": f1_score(y_true, pred)}

def mejor_threshold(y_true, proba):
    return max([round(0.05 * i, 2) for i in range(2, 19)],
               key=lambda t: evaluar(y_true, proba, t)["f1"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Entrenar los 3 modelos con MLflow (Version Ligera)

# COMMAND ----------

from sklearn.preprocessing import StandardScaler
from mlflow.models import infer_signature

prepro = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
])

MODELOS = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=SEED, n_jobs=-1),
    "random_forest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=SEED, n_jobs=-1),
    "gbt": HistGradientBoostingClassifier(max_iter=100, max_depth=None, random_state=SEED),
}

resultados = []
input_example = X_train.head(1)

for nombre, est in MODELOS.items():
    with mlflow.start_run(run_name=nombre) as run:
        # Entrenar pipeline
        pipe = Pipeline([("pre", prepro), ("modelo", est)]).fit(X_train, y_train)
        
        # Predicciones y métricas
        proba_tr = pipe.predict_proba(X_train)[:, 1]
        proba_te = pipe.predict_proba(X_test)[:, 1]

        thr = mejor_threshold(y_train, proba_tr)
        m_tr = evaluar(y_train, proba_tr, thr)
        m_te = evaluar(y_test, proba_te, thr)

        # Logging de parámetros y métricas
        mlflow.log_params({"modelo": nombre, "n_features": len(num_cols) + len(cat_cols), "seed": SEED})
        for k, v in m_tr.items(): mlflow.log_metric(f"train_{k}", v)
        for k, v in m_te.items(): mlflow.log_metric(f"test_{k}", v)
        mlflow.log_metric("threshold", thr)
        
        # Generar firma y registrar modelo con artifact_path correcto
        signature = infer_signature(X_train, pipe.predict(X_train))
        mlflow.sklearn.log_model(
            pipe, 
            artifact_path="sklearn-model", 
            input_example=input_example,
            signature=signature
        )

        resultados.append({
            "nombre": nombre, 
            "run_id": run.info.run_id,
            "test": m_te, 
            "thr": thr
        })
        print(f"{nombre} | test PR-AUC={m_te['pr_auc']:.4f} | F1={m_te['f1']:.4f}")

    # Limpieza de memoria por iteración
    del pipe, proba_tr, proba_te, est
    gc.collect()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Seleccion del Campeon y Registro en Unity Catalog

# COMMAND ----------

# Seleccionar al campeón basándose en métricas
campeon = max(resultados, key=lambda r: (r["test"]["pr_auc"], r["test"]["f1"]))
print(f"CAMPEON: {campeon['nombre']} | PR-AUC={campeon['test']['pr_auc']:.4f} | threshold={campeon['thr']}")

if "confusion" in campeon:
    display(campeon["confusion"])

# Intentar registrar en Unity Catalog probando los nombres de artefacto posibles
for art in ["sklearn-model", "model"]:
    try:
        reg = mlflow.register_model(f"runs:/{campeon['run_id']}/{art}", NOMBRE_MODELO_UC)
        print(f"Registrado en Unity Catalog exitosamente: {NOMBRE_MODELO_UC} | version {reg.version}")
        print("Asignale el alias 'champion' desde la pestaña Models.")
        break
    except Exception as e:
        print(f"Con artefacto '{art}' no se pudo: {e}")