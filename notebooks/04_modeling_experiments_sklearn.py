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
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_score, recall_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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

NO_FEATURE = {
    "order_id", "customer_id", "customer_unique_id", 
    "seller_id_principal", "product_id_principal", 
    "is_late", "split_temporal", "order_purchase_timestamp"
}
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


def mejor_threshold_precision(y_true, proba, min_precision=0.80):
    best_thr = 0.5
    best_f1 = 0
    for i in range(1, 20):
        thr = round(0.05 * i, 2)
        pred = (proba >= thr).astype(int)
        p = precision_score(y_true, pred, zero_division=0)
        r = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        
        # Filtrar umbrales que cumplan con la precisión mínima deseada
        if p >= min_precision and f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return best_thr

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Entrenar los 3 modelos con MLflow

# COMMAND ----------

from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, precision_recall_curve, f1_score, auc
from mlflow.models import infer_signature
import mlflow
import pandas as pd
import numpy as np
import gc

# 1. Carga directa desde Spark
pdf = spark.table("big_data_2026.olist.gold_ml_features").toPandas()

# 2. Exclusión estricta de IDs, fechas y variables objetivo/fuga
NO_FEATURE_FINAL = {
    "order_id", "customer_id", "customer_unique_id", 
    "seller_id_principal", "product_id_principal", 
    "is_late", "split_temporal", "order_purchase_timestamp",
    "order_estimated_delivery_date", "order_delivered_customer_date"
}

cat_cols = [c for c in pdf.columns if c not in NO_FEATURE_FINAL and pdf[c].dtype == object]
num_cols = [c for c in pdf.columns if c not in NO_FEATURE_FINAL and c not in cat_cols and pd.api.types.is_numeric_dtype(pdf[c])]

# 3. División de datos en train y test
train = pdf[pdf.split_temporal == "train"]
test  = pdf[pdf.split_temporal == "test"]
X_train, y_train = train[cat_cols + num_cols], train.is_late
X_test,  y_test  = test[cat_cols + num_cols],  test.is_late

prepro = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
])

# 4. Configuración de modelos y UMBRAL FIJO a 0.40
SEED = 42
MODELOS = {
    "logistic_regression": LogisticRegression(max_iter=5000, C=0.1, class_weight="balanced", random_state=SEED, n_jobs=-1),
    "random_forest": RandomForestClassifier(n_estimators=150, max_depth=10, random_state=SEED, n_jobs=-1, class_weight="balanced"),
    "gbt": HistGradientBoostingClassifier(max_iter=150, max_depth=6, random_state=SEED),
}

UMBRAL_FIJO = 0.40

def evaluar_local(y_true, probas, umbral):
    preds = (probas >= umbral).astype(int)
    prec, rec, _ = precision_recall_curve(y_true, probas)
    pr_auc = auc(rec, prec)
    return {
        "f1": f1_score(y_true, preds, zero_division=0),
        "pr_auc": pr_auc
    }

resultados = []
input_example = X_train.head(1)

for nombre, est in MODELOS.items():
    with mlflow.start_run(run_name=nombre) as run:
        pipe = Pipeline([("pre", prepro), ("modelo", est)]).fit(X_train, y_train)
        
        proba_tr = pipe.predict_proba(X_train)[:, 1]
        proba_te = pipe.predict_proba(X_test)[:, 1]

        thr = UMBRAL_FIJO
        m_tr = evaluar_local(y_train, proba_tr, thr)
        m_te = evaluar_local(y_test, proba_te, thr)

        pred_te = (proba_te >= thr).astype(int)
        matriz_te = confusion_matrix(y_test, pred_te)

        mlflow.log_params({"modelo": nombre, "n_features": len(num_cols) + len(cat_cols), "seed": SEED, "threshold": thr})
        for k, v in m_tr.items(): mlflow.log_metric(f"train_{k}", v)
        for k, v in m_te.items(): mlflow.log_metric(f"test_{k}", v)
        mlflow.log_metric("threshold", thr)
        
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
            "thr": thr,
            "confusion": matriz_te
        })
        
        print(f"--- MODELO: {nombre} ---")
        print(f"PR-AUC={m_te['pr_auc']:.4f} | F1={m_te['f1']:.4f} | Threshold={thr}")
        print("Matriz de confusión:")
        print(matriz_te)
        print("\n")

    del pipe, proba_tr, proba_te, est
    gc.collect()

del pdf, train, test
gc.collect()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Seleccion del Campeon y Registro en Unity Catalog

# COMMAND ----------

# Seleccionar al campeón basándose en métricas
campeon = max(resultados, key=lambda r: r["test"]["f1"])
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

# COMMAND ----------

# MAGIC %md
# MAGIC ### Matriz confusion

# COMMAND ----------

# Visualizar la matriz del campeón
disp = ConfusionMatrixDisplay(confusion_matrix=campeon["confusion"], 
                              display_labels=["A tiempo (0)", "Retraso (1)"])
disp.plot(cmap=plt.cm.Blues)
plt.title(f"Matriz de Confusión - {campeon['nombre']} (Umbral: {campeon['thr']})")
plt.show()