# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # CAPA GOLD 2 — Tabla de features para el modelo predictivo
# MAGIC
# MAGIC **Proyecto:** Predicción de Retrasos en Entregas — Olist E-commerce
# MAGIC **Responsable:** Marlon
# MAGIC **Entrada:** `silver_enriched`
# MAGIC **Salida:** `gold_ml_features` (grano: 1 fila = 1 pedido entregado)
# MAGIC
# MAGIC ```
# MAGIC 03.1_gold_business_kpis
# MAGIC        ▼
# MAGIC 03.2_gold_ml_features          <-- estás aquí
# MAGIC        ▼
# MAGIC 04_gold_mlflow_modeling  (entrenamiento, fuera de este notebook)
# MAGIC ```
# MAGIC
# MAGIC ## Alcance
# MAGIC
# MAGIC A diferencia de `03.1`, esta tabla **sí alimenta un modelo**, así que hereda el
# MAGIC contrato anti-fuga documentado en la sección 6 de `02.2_silver_geolocation_target`
# MAGIC y lo aplica en código: acá no hay margen de que alguien copie `days_delay` a un
# MAGIC notebook de entrenamiento "por error".
# MAGIC
# MAGIC Este notebook hace tres cosas y nada más:
# MAGIC 1. **Selecciona** solo columnas seguras (+ identificadores + target).
# MAGIC 2. **Imputa** los nulos de las features numéricas, dejando un flag de auditoría.
# MAGIC 3. **Particiona temporalmente** en train/test — nunca al azar.
# MAGIC
# MAGIC No hace `StringIndexer`/`OneHotEncoder` ni escala nada: eso es responsabilidad del
# MAGIC *pipeline* de `04_gold_mlflow_modeling`, que necesita `fit` solo sobre train para no
# MAGIC filtrar información de test. Hacerlo acá lo haría irreversible y opaco.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Configuración

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import Window

CATALOG = "big_data_2026"
SCHEMA = "olist"

TABLA_ENTRADA = "silver_enriched"
TABLA_SALIDA = "gold_ml_features"

# ¿Se predice en el checkout o al confirmarse el pago? Ver la "zona gris" documentada
# en 02.2 sección 6. False = postura conservadora (checkout): la más segura por defecto.
# Cambiar a True solo si el equipo decide explícitamente que el modelo se sirve después
# de aprobado el pago.
INCLUIR_COLUMNAS_CONDICIONALES = False

# Proporción de pedidos (ordenados por fecha) que va a train. El resto, a test.
FRACCION_TRAIN = 0.8


def T(nombre: str) -> str:
    return f"{CATALOG}.{SCHEMA}.{nombre}"


print(f"Entrada                        : {T(TABLA_ENTRADA)}")
print(f"Salida                         : {T(TABLA_SALIDA)}")
print(f"Incluir columnas condicionales : {INCLUIR_COLUMNAS_CONDICIONALES}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Contrato de entrada

# COMMAND ----------

existentes = {
    fila.tableName for fila in spark.sql(f"SHOW TABLES IN {CATALOG}.{SCHEMA}").collect()
}
assert TABLA_ENTRADA in existentes, (
    f"No existe {T(TABLA_ENTRADA)}. Ejecuta primero 02.2_silver_geolocation_target."
)

silver = spark.table(T(TABLA_ENTRADA))
n_pedidos = silver.count()

assert n_pedidos == silver.select("order_id").distinct().count(), (
    "silver_enriched no está a grano de pedido."
)
assert "is_late" in silver.columns, "Falta la columna target 'is_late' en silver_enriched."

print(f"Contrato de entrada OK — {n_pedidos:,} pedidos, grano verificado.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Contrato anti-fuga (espejo de `02.2`, sección 6)
# MAGIC
# MAGIC Se redefine acá — en vez de importarse de otro notebook, porque Databricks no
# MAGIC comparte estado entre notebooks salvo tablas — para que este archivo sea
# MAGIC **autocontenido y auditable**: alguien puede leer este notebook solo y confirmar
# MAGIC qué entra al modelo, sin tener que abrir `02.2` primero. Las listas deben
# MAGIC mantenerse sincronizadas entre ambos notebooks; si `02.2` agrega una columna nueva
# MAGIC con fuga, hay que replicarla acá.

# COMMAND ----------

COLUMNAS_CON_FUGA = [
    "order_delivered_customer_date",
    "fecha_entrega_real",
    "days_delay",
    "dias_entrega_real",
    "severidad_retraso",
    "order_delivered_carrier_date",
    "fecha_envio_transportista",
    "dias_hasta_transportista",
    "order_status",
    "review_score",
    "fecha_review",
    "tiene_review",
]

COLUMNAS_CONDICIONALES = [
    "order_approved_at",
    "horas_hasta_aprobacion",
]

COLUMNAS_IDENTIFICADORAS = [
    "order_id",
    "customer_id",
    "customer_unique_id",
    "seller_id_principal",
    "product_id_principal",
]

# Columnas de apoyo temporal / geográfico que no son features de entrenamiento pero sí
# se necesitan para construir la tabla (split temporal, texto legible de ubicación).
COLUMNAS_AUXILIARES = [
    "order_purchase_timestamp",
    "fecha_compra",
    "cliente_ciudad",
    "vendedor_ciudad",
    "shipping_limit_date",
    "cliente_zip_prefix",
    "vendedor_zip_prefix",
    "cliente_lat",
    "cliente_lng",
    "vendedor_lat",
    "vendedor_lng",
]

TARGET = ["is_late"]

columnas_excluidas = (
    set(COLUMNAS_CON_FUGA)
    | set(COLUMNAS_IDENTIFICADORAS)
    | set(COLUMNAS_AUXILIARES)
    | set(TARGET)
)
if not INCLUIR_COLUMNAS_CONDICIONALES:
    columnas_excluidas |= set(COLUMNAS_CONDICIONALES)

features = sorted(set(silver.columns) - columnas_excluidas)

print(f"FEATURES SELECCIONADAS ({len(features)}):")
for c in features:
    print(f"   {c}")

assert not set(features) & set(COLUMNAS_CON_FUGA), "Una columna con fuga se coló en features."
assert "is_late" not in features, "El target se coló en la lista de features."

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Selección de columnas

# COMMAND ----------

df = silver.select(
    *COLUMNAS_IDENTIFICADORAS,
    "order_purchase_timestamp",
    *TARGET,
    *features,
)

print(f"Columnas seleccionadas: {len(df.columns)} "
      f"({len(COLUMNAS_IDENTIFICADORAS)} ids + 1 timestamp + 1 target + {len(features)} features)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Imputación de nulos
# MAGIC
# MAGIC Silver reporta la realidad (perfil de nulos en `02.1`), Gold decide la estrategia:
# MAGIC
# MAGIC | Columna | Estrategia | Por qué |
# MAGIC |---|---|---|
# MAGIC | `distancia_km` | mediana por `cliente_estado`, con fallback a mediana global | La distancia varía mucho por región; el estado del cliente es la mejor señal disponible sin la coordenada. |
# MAGIC | `peso_producto_g`, `volumen_producto_cm3`, `n_fotos_producto` | mediana por `categoria_producto`, con fallback a mediana global | Productos de la misma categoría se parecen entre sí; mejor que una mediana única para todo el catálogo. |
# MAGIC | `ratio_flete` | mediana global | Nulo solo cuando `valor_total_pedido = 0`, un caso raro sin señal categórica útil. |
# MAGIC | `valor_pagado`, `max_cuotas`, `n_tipos_pago`, `n_transacciones_pago` | `0`, con flag único `sin_registro_pago` | Estas quedan nulas porque el pedido no tiene fila en `order_payments` (el `left join` de 02.1 las dejó pasar). No es un valor faltante al azar: es "no hay pago registrado", así que un solo flag booleano es más honesto que imputar cada columna por separado. |
# MAGIC
# MAGIC Cada imputación deja un flag `*_imputado` (o `sin_registro_pago` para el grupo de
# MAGIC pago) — el modelo puede usarlo como feature adicional (a veces "faltaba el dato" es
# MAGIC información en sí misma) y permite auditar cuánta de la tabla es dato real vs. imputado.
# MAGIC
# MAGIC Las categóricas con nulos (`cliente_estado`, `vendedor_estado`, `tipo_pago_principal`)
# MAGIC se imputan con la etiqueta literal `"desconocido"`, mismo criterio que `02.1` usó
# MAGIC para `categoria_producto`.

# COMMAND ----------

mediana_distancia_global = df.select(
    F.expr("percentile_approx(distancia_km, 0.5)").alias("m")
).collect()[0]["m"]

mediana_distancia_estado = df.groupBy("cliente_estado").agg(
    F.expr("percentile_approx(distancia_km, 0.5)").alias("mediana_distancia_estado")
)

df = df.withColumn("distancia_km_imputado", F.col("distancia_km").isNull().cast("int"))
df = (
    df.join(mediana_distancia_estado, on="cliente_estado", how="left")
    .withColumn(
        "distancia_km",
        F.coalesce(
            F.col("distancia_km"),
            F.col("mediana_distancia_estado"),
            F.lit(mediana_distancia_global),
        ),
    )
    .drop("mediana_distancia_estado")
)

mediana_peso_global = df.select(
    F.expr("percentile_approx(peso_producto_g, 0.5)").alias("m")
).collect()[0]["m"]
mediana_volumen_global = df.select(
    F.expr("percentile_approx(volumen_producto_cm3, 0.5)").alias("m")
).collect()[0]["m"]

medianas_categoria = df.groupBy("categoria_producto").agg(
    F.expr("percentile_approx(peso_producto_g, 0.5)").alias("mediana_peso_categoria"),
    F.expr("percentile_approx(volumen_producto_cm3, 0.5)").alias("mediana_volumen_categoria"),
)

df = (
    df.withColumn("peso_producto_g_imputado", F.col("peso_producto_g").isNull().cast("int"))
    .withColumn(
        "volumen_producto_cm3_imputado", F.col("volumen_producto_cm3").isNull().cast("int")
    )
    .join(medianas_categoria, on="categoria_producto", how="left")
    .withColumn(
        "peso_producto_g",
        F.coalesce(
            F.col("peso_producto_g"),
            F.col("mediana_peso_categoria"),
            F.lit(mediana_peso_global),
        ),
    )
    .withColumn(
        "volumen_producto_cm3",
        F.coalesce(
            F.col("volumen_producto_cm3"),
            F.col("mediana_volumen_categoria"),
            F.lit(mediana_volumen_global),
        ),
    )
    .drop("mediana_peso_categoria", "mediana_volumen_categoria")
)

mediana_ratio_flete = df.select(
    F.expr("percentile_approx(ratio_flete, 0.5)").alias("m")
).collect()[0]["m"]

df = df.withColumn(
    "ratio_flete_imputado", F.col("ratio_flete").isNull().cast("int")
).withColumn(
    "ratio_flete", F.coalesce(F.col("ratio_flete"), F.lit(mediana_ratio_flete))
)

# `pagos_agg` se unió con LEFT JOIN en 02.1: los pedidos sin registro de pago llegan acá
# con valor_pagado/max_cuotas/n_tipos_pago/n_transacciones_pago en null. El null no es
# "dato faltante que hay que adivinar", es "no hubo pago registrado" — un solo flag
# booleano (`sin_registro_pago`) captura eso mejor que imputar cada columna por separado
# con estrategias distintas que fingirían tener una señal que no existe.
df = df.withColumn(
    "sin_registro_pago", F.col("valor_pagado").isNull().cast("int")
)
for columna in ["valor_pagado", "max_cuotas", "n_tipos_pago", "n_transacciones_pago"]:
    df = df.withColumn(columna, F.coalesce(F.col(columna), F.lit(0)))

# n_fotos_producto tiene el mismo origen que peso_producto_g/volumen_producto_cm3: fichas
# de producto incompletas en el dataset de Olist. Misma estrategia: mediana por categoría
# con fallback global.
mediana_fotos_global = df.select(
    F.expr("percentile_approx(n_fotos_producto, 0.5)").alias("m")
).collect()[0]["m"]
mediana_fotos_categoria = df.groupBy("categoria_producto").agg(
    F.expr("percentile_approx(n_fotos_producto, 0.5)").alias("mediana_fotos_categoria")
)

df = (
    df.withColumn(
        "n_fotos_producto_imputado", F.col("n_fotos_producto").isNull().cast("int")
    )
    .join(mediana_fotos_categoria, on="categoria_producto", how="left")
    .withColumn(
        "n_fotos_producto",
        F.coalesce(
            F.col("n_fotos_producto"),
            F.col("mediana_fotos_categoria"),
            F.lit(mediana_fotos_global),
        ),
    )
    .drop("mediana_fotos_categoria")
)

df = (
    df.withColumn("cliente_estado", F.coalesce(F.col("cliente_estado"), F.lit("desconocido")))
    .withColumn(
        "vendedor_estado", F.coalesce(F.col("vendedor_estado"), F.lit("desconocido"))
    )
    .withColumn(
        "tipo_pago_principal",
        F.coalesce(F.col("tipo_pago_principal"), F.lit("desconocido")),
    )
)

print("Imputación aplicada.")
print(f"  Mediana global distancia_km          : {mediana_distancia_global:.2f} km")
print(f"  Mediana global peso_producto_g       : {mediana_peso_global:.0f} g")
print(f"  Mediana global volumen_producto_cm3  : {mediana_volumen_global:.0f} cm3")
print(f"  Mediana global ratio_flete            : {mediana_ratio_flete:.4f}")
print(f"  Mediana global n_fotos_producto      : {mediana_fotos_global:.0f}")
print("  valor_pagado / max_cuotas / n_tipos_pago / n_transacciones_pago -> 0 si sin_registro_pago")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Split temporal train/test
# MAGIC
# MAGIC **Nunca aleatorio.** Este es el punto que `02.2` deja como advertencia explícita
# MAGIC para Gold: un split al azar entrena con pedidos posteriores a otros de test y
# MAGIC sobrestima el desempeño real, porque en producción el modelo solo va a ver el
# MAGIC pasado. El corte es el percentil `FRACCION_TRAIN` (80% por defecto) de
# MAGIC `order_purchase_timestamp`: todo lo anterior es train, todo lo posterior es test.

# COMMAND ----------

fecha_corte = df.select(
    F.expr(f"percentile_approx(order_purchase_timestamp, {FRACCION_TRAIN})").alias("f")
).collect()[0]["f"]

df = df.withColumn(
    "split_temporal",
    F.when(F.col("order_purchase_timestamp") <= F.lit(fecha_corte), "train").otherwise(
        "test"
    ),
)

resumen_split = (
    df.groupBy("split_temporal")
    .agg(
        F.count("*").alias("pedidos"),
        F.min("order_purchase_timestamp").alias("desde"),
        F.max("order_purchase_timestamp").alias("hasta"),
        F.round(100 * F.avg("is_late"), 2).alias("tasa_retraso_pct"),
    )
    .orderBy("split_temporal")
)

print(f"Fecha de corte (percentil {FRACCION_TRAIN:.0%}): {fecha_corte}")
display(resumen_split)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Control de calidad antes de escribir

# COMMAND ----------

n_filas = df.count()
n_unicos = df.select("order_id").distinct().count()

assert n_filas == n_unicos == n_pedidos, (
    f"FAN-OUT o pérdida de filas: {n_filas:,} filas, {n_unicos:,} order_id únicos, "
    f"{n_pedidos:,} pedidos de entrada."
)

assert df.filter(~F.col("is_late").isin(0, 1)).count() == 0, "is_late fuera de {0,1}."

columnas_features_finales = [
    c for c in df.columns
    if c not in COLUMNAS_IDENTIFICADORAS
    and c not in ["order_purchase_timestamp", "is_late", "split_temporal"]
]
nulos_restantes = {
    c: n
    for c, n in (
        df.select(
            [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in columnas_features_finales]
        )
        .collect()[0]
        .asDict()
        .items()
    )
    if n > 0
}
assert not nulos_restantes, (
    f"Quedaron nulos sin imputar en columnas de features: {nulos_restantes}"
)

train_count = df.filter(F.col("split_temporal") == "train").count()
test_count = df.filter(F.col("split_temporal") == "test").count()
assert train_count > 0 and test_count > 0, "Uno de los dos splits quedó vacío."

max_fecha_train = df.filter(F.col("split_temporal") == "train").select(
    F.max("order_purchase_timestamp")
).collect()[0][0]
min_fecha_test = df.filter(F.col("split_temporal") == "test").select(
    F.min("order_purchase_timestamp")
).collect()[0][0]
assert max_fecha_train <= min_fecha_test, (
    "El split temporal se mezcló: hay pedidos de train posteriores a pedidos de test."
)

print(f"Filas               : {n_filas:,}")
print(f"Train               : {train_count:,} ({100.0*train_count/n_filas:.1f}%)")
print(f"Test                : {test_count:,} ({100.0*test_count/n_filas:.1f}%)")
print(f"Sin nulos en features: OK")
print(f"Orden temporal train <= test: OK")
print("\nTodas las aserciones de calidad pasaron.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Escritura de `gold_ml_features`

# COMMAND ----------

PRIMERAS = ["order_id", "is_late", "split_temporal", "order_purchase_timestamp"]
columnas_finales = PRIMERAS + [c for c in df.columns if c not in PRIMERAS]

gold_ml_features = df.select(*columnas_finales)

(
    gold_ml_features.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(T(TABLA_SALIDA))
)

print(f"Tabla escrita: {T(TABLA_SALIDA)}")
print(f"Filas        : {spark.table(T(TABLA_SALIDA)).count():,}")
print(f"Columnas     : {len(spark.table(T(TABLA_SALIDA)).columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Documentación en Unity Catalog

# COMMAND ----------

comentario_tabla = (
    "Capa Gold. Tabla lista para entrenar el modelo de is_late: solo features conocidas "
    "al momento de la compra (o de la aprobacion de pago si INCLUIR_COLUMNAS_CONDICIONALES=True), "
    "nulos imputados con flags de auditoria (*_imputado), y particionada temporalmente "
    "via split_temporal (train/test). Cero columnas con fuga de datos. "
    "Insumo de 04_gold_mlflow_modeling. Responsable: Marlon."
).replace("'", "\\'")

spark.sql(f"COMMENT ON TABLE {T(TABLA_SALIDA)} IS '{comentario_tabla}'")

COMENTARIOS_COLUMNAS = {
    "is_late": "TARGET. 1 si la entrega fue posterior a la fecha estimada.",
    "split_temporal": "train o test, asignado por corte temporal sobre order_purchase_timestamp. No mezclar ni re-splitear al azar.",
    "distancia_km_imputado": "1 si distancia_km fue imputada (mediana por cliente_estado o global).",
    "peso_producto_g_imputado": "1 si peso_producto_g fue imputado (mediana por categoria_producto o global).",
    "volumen_producto_cm3_imputado": "1 si volumen_producto_cm3 fue imputado.",
    "ratio_flete_imputado": "1 si ratio_flete fue imputado (mediana global).",
    "n_fotos_producto_imputado": "1 si n_fotos_producto fue imputado (mediana por categoria_producto o global).",
    "sin_registro_pago": "1 si el pedido no tenia registro en order_payments (valor_pagado/max_cuotas/n_tipos_pago/n_transacciones_pago quedaron en 0).",
    "dias_promesa": "Dias prometidos al cliente en el checkout. Feature de mayor senal esperada.",
    "distancia_km": "Distancia Haversine vendedor-cliente en km, imputada si faltaba.",
}

for columna, comentario in COMENTARIOS_COLUMNAS.items():
    if columna in gold_ml_features.columns:
        comentario_sql = comentario.replace("'", "\\'")
        spark.sql(
            f"ALTER TABLE {T(TABLA_SALIDA)} ALTER COLUMN {columna} COMMENT '{comentario_sql}'"
        )

print("Metadatos documentados en Unity Catalog.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Optimización del layout físico

# COMMAND ----------

try:
    spark.sql(f"OPTIMIZE {T(TABLA_SALIDA)}")
    print("OPTIMIZE ejecutado.")
except Exception as e:
    print(f"OPTIMIZE no disponible en este cómputo (no es un error del pipeline): {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Vista previa

# COMMAND ----------

display(
    spark.table(T(TABLA_SALIDA)).select(
        "order_id",
        "is_late",
        "split_temporal",
        "dias_promesa",
        "distancia_km",
        "distancia_km_imputado",
        "valor_total_pedido",
        "ratio_flete",
        "cliente_estado",
        "vendedor_estado",
        "mismo_estado",
        "categoria_producto",
    ).limit(20)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resumen de la capa Gold — ML features
# MAGIC
# MAGIC | Aspecto | Decisión |
# MAGIC |---|---|
# MAGIC | Grano | 1 fila = 1 pedido entregado (idéntico a `silver_enriched`) |
# MAGIC | Features | Solo columnas seguras según el contrato de `02.2` (sin fuga) |
# MAGIC | Condicionales (`order_approved_at`, etc.) | Excluidas por defecto (`INCLUIR_COLUMNAS_CONDICIONALES = False`) |
# MAGIC | Imputación | Mediana por grupo con fallback global; flags `*_imputado` conservados |
# MAGIC | Split | Temporal (percentil 80% de `order_purchase_timestamp`), nunca aleatorio |
# MAGIC | Target | `is_late`, desbalanceado ~1:12 — usar PR-AUC / F1 / recall, no accuracy |
# MAGIC | Salida | `big_data_2026.olist.gold_ml_features` |
# MAGIC
# MAGIC **Siguiente paso:** `04_gold_mlflow_modeling` — construir el pipeline de
# MAGIC entrenamiento (encoding de categóricas, escalado si corresponde, y registro del
# MAGIC modelo en MLflow) usando `split_temporal` para separar train/test.