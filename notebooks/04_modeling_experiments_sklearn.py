# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 04 — Experimentos de modelado (sklearn + pandas, sin Spark ML)
# MAGIC
# MAGIC Entrada: `big_data_2026.olist.gold_ml_features` (proveniente de 03_2)
# MAGIC Salida: runs en MLflow + modelo campeon registrado en Unity Catalog
# MAGIC
# MAGIC Por que sklearn: el cache ML de Spark Connect (1 GB) se llena en este
# MAGIC workspace; con 96k filas todo cabe en pandas y sklearn no usa ese cache.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuracion

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

CATALOG, SCHEMA = "big_data_2026", "olist"
NOMBRE_MODELO_UC = f"{CATALOG}.{SCHEMA}.olist_delay_predictor"
SEED = 42

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

train = pdf[pdf.split_temporal == "train"]
test  = pdf[pdf.split_temporal == "test"]
X_train, y_train = train[cat_cols + num_cols], train.is_late
X_test,  y_test  = test[cat_cols + num_cols],  test.is_late
print(f"Features: {len(num_cols)} num + {len(cat_cols)} cat | train={len(train):,} | test={len(test):,}")

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
# MAGIC ## 4. Entrenar los 3 modelos con MLflow

# COMMAND ----------

prepro = ColumnTransformer([
    ("num", "passthrough", num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
], sparse_threshold=0)

MODELOS = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "random_forest":       RandomForestClassifier(n_estimators=20, max_depth=5, random_state=SEED),
    "gbt":                 HistGradientBoostingClassifier(max_iter=100, random_state=SEED),
}

resultados = []
for nombre, est in MODELOS.items():
    with mlflow.start_run(run_name=nombre) as run:
        pipe = Pipeline([("pre", prepro), ("modelo", est)]).fit(X_train, y_train)
        proba_tr = pipe.predict_proba(X_train)[:, 1]
        proba_te = pipe.predict_proba(X_test)[:, 1]

        thr = mejor_threshold(y_train, proba_tr)
        m_tr = evaluar(y_train, proba_tr, thr)
        m_te = evaluar(y_test, proba_te, thr)

        mlflow.log_params({"modelo": nombre, "n_features": len(num_cols) + len(cat_cols), "seed": SEED})
        for k, v in m_tr.items(): mlflow.log_metric(f"train_{k}", v)
        for k, v in m_te.items(): mlflow.log_metric(f"test_{k}", v)
        mlflow.log_metric("threshold", thr)
        mlflow.sklearn.log_model(pipe, artifact_path="model")

        confusion = pd.crosstab(test.is_late, (proba_te >= thr).astype(int),
                                rownames=["is_late"], colnames=["pred"]).reset_index()
        resultados.append({"nombre": nombre, "run_id": run.info.run_id,
                           "test": m_te, "thr": thr, "confusion": confusion})
        print(f"{nombre} | test PR-AUC={m_te['pr_auc']:.4f} | F1={m_te['f
