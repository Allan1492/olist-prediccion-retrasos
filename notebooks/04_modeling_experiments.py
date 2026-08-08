# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 04 — Experimentos de modelado (fuera de la capa Gold)
# MAGIC
# MAGIC Proyecto: Predicción de Retrasos en Entregas — Olist E-commerce
# MAGIC Entrada: `big_data_2026.olist.gold_ml_features` (proveniente de 03_2)
# MAGIC Salida: runs en MLflow + modelo campeón registrado en Unity Catalog
# MAGIC
# MAGIC Algoritmos: Logistic Regression (baseline), Random Forest y GBT.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuración

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import StringType, DateType, TimestampType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.functions import vector_to_array
import mlflow
import mlflow.spark

CATALOG, SCHEMA = "big_data_2026", "olist"
NOMBRE_MODELO_UC = f"{CATALOG}.{SCHEMA}.olist_delay_predictor"
SEED = 42
T = lambda n: f"{CATALOG}.{SCHEMA}.{n}"

# Por defecto usa el experimento del propio notebook (pestana Experiments).
# Para un experimento compartido, descomenta:
# mlflow.set_experiment("/Shared/olist/retrasos-entrega")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Cargar tabla Gold + contrato de entrada

# COMMAND ----------

df = spark.table(T("gold_ml_features"))
assert {"order_id", "is_late", "split_temporal"} <= set(df.columns)
assert df.count() == df.select("order_id").distinct().count(), "grano roto"

FUGA = {"order_delivered_customer_date", "days_delay", "order_status",
        "review_score", "order_delivered_carrier_date"}
assert not (FUGA & set(df.columns)), f"Fuga detectada: {FUGA & set(df.columns)}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Features + split temporal

# COMMAND ----------

NO_FEATURE = {"order_id", "is_late", "split_temporal", "order_purchase_timestamp"}
cat_cols = [f.name for f in df.schema.fields
            if f.name not in NO_FEATURE and isinstance(f.dataType, StringType)]
num_cols = [f.name for f in df.schema.fields
            if f.name not in NO_FEATURE and f.name not in cat_cols
            and not isinstance(f.dataType, (DateType, TimestampType))]

train = df.filter(F.col("split_temporal") == "train")
test  = df.filter(F.col("split_temporal") == "test")
print(f"Features: {len(num_cols)} num + {len(cat_cols)} cat | train={train.count():,} | test={test.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Preprocesamiento

# COMMAND ----------

prepro = [StringIndexer(inputCol=c, outputCol=f"{c}__idx", handleInvalid="keep") for c in cat_cols]
prepro += [OneHotEncoder(inputCols=[f"{c}__idx" for c in cat_cols],
                         outputCols=[f"{c}__ohe" for c in cat_cols], handleInvalid="keep"),
           VectorAssembler(inputCols=num_cols + [f"{c}__ohe" for c in cat_cols],
                           outputCol="features", handleInvalid="keep")]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Funciones de evaluación

# COMMAND ----------

def con_prob(d):
    return d.withColumn("prob_late", vector_to_array(F.col("probability"))[1])

def evaluar(d, thr):
    p = d.withColumn("pred", (F.col("prob_late") >= thr).cast("double"))
    pr  = BinaryClassificationEvaluator(labelCol="is_late", rawPredictionCol="probability", metricName="areaUnderPR").evaluate(p)
    roc = BinaryClassificationEvaluator(labelCol="is_late", rawPredictionCol="probability", metricName="areaUnderROC").evaluate(p)
    f1  = MulticlassClassificationEvaluator(labelCol="is_late", predictionCol="pred", metricName="f1").evaluate(p)
    return {"pr_auc": pr, "roc_auc": roc, "f1": f1}, p

def mejor_threshold(train_pred):
    return max([round(0.05 * i, 2) for i in range(2, 19)],
               key=lambda t: evaluar(train_pred, t)[0]["f1"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Entrenar los 3 modelos con MLflow

# COMMAND ----------

MODELOS = {
    "logistic_regression": LogisticRegression(labelCol="is_late", seed=SEED),
    "random_forest":       RandomForestClassifier(labelCol="is_late", numTrees=50, seed=SEED),
    "gbt":                 GBTClassifier(labelCol="is_late", maxIter=20, seed=SEED),
}

resultados = []
for nombre, est in MODELOS.items():
    with mlflow.start_run(run_name=nombre) as run:
        fitted = Pipeline(stages=prepro + [est]).fit(train)
        tr_pred = con_prob(fitted.transform(train))
        te_pred = con_prob(fitted.transform(test))

        thr = mejor_threshold(tr_pred)
        m_tr, _ = evaluar(tr_pred, thr)
        m_te, te_scored = evaluar(te_pred, thr)

        mlflow.log_params({"modelo": nombre, "n_features": len(num_cols) + len(cat_cols), "seed": SEED})
        for k, v in m_tr.items(): mlflow.log_metric(f"train_{k}", v)
        for k, v in m_te.items(): mlflow.log_metric(f"test_{k}", v)
        mlflow.log_metric("threshold", thr)
        mlflow.spark.log_model(fitted, artifact_path="model")

        resultados.append({"nombre": nombre, "run_id": run.info.run_id,
                           "test": m_te, "thr": thr, "test_pred": te_scored})
        print(f"{nombre} | test PR-AUC={m_te['pr_auc']:.4f} | F1={m_te['f1']:.4f}")
        print(f"  carpeta de artefactos: {run.info.artifact_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Campeón + registro en Unity Catalog

# COMMAND ----------

campeon = max(resultados, key=lambda r: (r["test"]["pr_auc"], r["test"]["f1"]))
print(f"CAMPEON: {campeon['nombre']} | PR-AUC={campeon['test']['pr_auc']:.4f} | threshold={campeon['thr']}")

display(campeon["test_pred"].groupBy("is_late", "pred").count().orderBy("is_late", "pred"))

try:
    reg = mlflow.register_model(f"runs:/{campeon['run_id']}/model", NOMBRE_MODELO_UC)
    print(f"Registrado en Unity Catalog: {NOMBRE_MODELO_UC} | version {reg.version}")
    print("Asignale el alias 'champion' desde la pestana Models.")
except Exception as e:
    print("No se pudo registrar en Unity Catalog; el modelo queda como artefacto del run.")
    print(f"Detalle: {e}")
