# Databricks notebook source
# MAGIC %md
# MAGIC # CAPA SILVER 1 — Transformación y JOINs
# MAGIC
# MAGIC **Proyecto:** Predicción de Retrasos en Entregas — Olist E-commerce
# MAGIC **Responsable:** ESTEBAN
# MAGIC **Entrada:** 9 tablas `bronze_*` en `big_data_2026.olist`
# MAGIC **Salida:** `silver_orders_joined` (grano: 1 fila = 1 pedido entregado)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Decisión de diseño central: el problema del grano
# MAGIC
# MAGIC Las tablas de Olist **no están al mismo grano**:
# MAGIC
# MAGIC | Tabla | Grano | Filas por pedido |
# MAGIC |---|---|---|
# MAGIC | `orders` | pedido | 1 |
# MAGIC | `order_items` | ítem del pedido | 1..N |
# MAGIC | `order_payments` | transacción de pago | 1..N |
# MAGIC | `order_reviews` | reseña | 0..N |
# MAGIC
# MAGIC Unir estas tablas directamente con `JOIN` produce un **producto cartesiano parcial**
# MAGIC (*fan-out*): un pedido con 3 ítems y 2 pagos genera 6 filas. Eso infla el dataset,
# MAGIC duplica el target y sesga cualquier modelo entrenado encima.
# MAGIC
# MAGIC **Solución aplicada:** cada tabla transaccional se **agrega a grano de pedido ANTES**
# MAGIC de unirse. El resultado es una tabla estrictamente 1 fila = 1 pedido, verificado
# MAGIC con un `assert` al final del notebook.
# MAGIC
# MAGIC ## Alcance
# MAGIC
# MAGIC Este notebook hace limpieza, agregación y unión. La **geolocalización, el cálculo de
# MAGIC distancia y la construcción del target** viven en `02.2_silver_geolocation_target`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Configuración e imports

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql import DataFrame

CATALOG = "big_data_2026"
SCHEMA = "olist"

TABLA_SALIDA = "silver_orders_joined"


def T(nombre: str) -> str:
    """Devuelve el nombre completamente calificado de una tabla."""
    return f"{CATALOG}.{SCHEMA}.{nombre}"


print(f"Catálogo destino : {CATALOG}.{SCHEMA}")
print(f"Tabla de salida  : {T(TABLA_SALIDA)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Contrato de entrada: validar que la capa Bronze existe
# MAGIC
# MAGIC Silver depende de Bronze. En vez de fallar 200 líneas más abajo con un error críptico
# MAGIC de Spark, verificamos el contrato de entrada de una vez y con un mensaje claro.

# COMMAND ----------

TABLAS_BRONZE = [
    "bronze_olist_orders_dataset",
    "bronze_olist_order_items_dataset",
    "bronze_olist_order_payments_dataset",
    "bronze_olist_order_reviews_dataset",
    "bronze_olist_products_dataset",
    "bronze_olist_sellers_dataset",
    "bronze_olist_customers_dataset",
    "bronze_olist_geolocation_dataset",
    "bronze_product_category_name_translation",
]

existentes = {
    fila.tableName for fila in spark.sql(f"SHOW TABLES IN {CATALOG}.{SCHEMA}").collect()
}
faltantes = [t for t in TABLAS_BRONZE if t not in existentes]

assert not faltantes, (
    "La capa Bronze está incompleta. Faltan estas tablas: "
    f"{faltantes}. Ejecuta primero 01_bronze_ingestion."
)

print("Contrato de entrada OK — las 9 tablas Bronze están disponibles.")
for t in TABLAS_BRONZE:
    print(f"   {t:<45} {spark.table(T(t)).count():>9,} filas")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Carga y normalización de tipos
# MAGIC
# MAGIC Bronze se ingirió con `inferSchema`, lo que casi siempre acierta pero **no es un
# MAGIC contrato**: basta con que un CSV llegue con una fecha mal formateada para que la
# MAGIC columna caiga a `string` y todas las restas de fechas devuelvan `null` en silencio.
# MAGIC Casteamos explícitamente para que Silver no dependa de la inferencia.

# COMMAND ----------

orders_raw = spark.table(T("bronze_olist_orders_dataset"))
items_raw = spark.table(T("bronze_olist_order_items_dataset"))
pagos_raw = spark.table(T("bronze_olist_order_payments_dataset"))
reviews_raw = spark.table(T("bronze_olist_order_reviews_dataset"))
productos_raw = spark.table(T("bronze_olist_products_dataset"))
vendedores_raw = spark.table(T("bronze_olist_sellers_dataset"))
clientes_raw = spark.table(T("bronze_olist_customers_dataset"))
traduccion_raw = spark.table(T("bronze_product_category_name_translation"))


def castear_timestamps(df: DataFrame, columnas: list) -> DataFrame:
    """Castea a timestamp de forma idempotente (no-op si ya es timestamp)."""
    for c in columnas:
        if c in df.columns:
            df = df.withColumn(c, F.to_timestamp(F.col(c)))
    return df


orders = castear_timestamps(
    orders_raw,
    [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
)

items = castear_timestamps(items_raw, ["shipping_limit_date"]).withColumn(
    "price", F.col("price").cast("double")
).withColumn("freight_value", F.col("freight_value").cast("double"))

pagos = (
    pagos_raw.withColumn("payment_value", F.col("payment_value").cast("double"))
    .withColumn("payment_installments", F.col("payment_installments").cast("int"))
    .withColumn("payment_sequential", F.col("payment_sequential").cast("int"))
)

reviews = castear_timestamps(
    reviews_raw, ["review_creation_date", "review_answer_timestamp"]
).withColumn("review_score", F.col("review_score").cast("int"))

clientes = clientes_raw.withColumn(
    "customer_zip_code_prefix", F.col("customer_zip_code_prefix").cast("int")
)
vendedores = vendedores_raw.withColumn(
    "seller_zip_code_prefix", F.col("seller_zip_code_prefix").cast("int")
)

print("Tipos normalizados.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Filtrado de pedidos: por qué solo `delivered`
# MAGIC
# MAGIC El target `is_late` compara la fecha de entrega real contra la estimada. Un pedido
# MAGIC `shipped`, `canceled` o `unavailable` **no tiene fecha de entrega real**, así que no
# MAGIC puede etiquetarse sin inventar el dato. Entrenar con esos registros implicaría
# MAGIC imputar el target, que es la peor forma posible de contaminar un modelo.
# MAGIC
# MAGIC Filtros aplicados, en orden, con conteo de descarte documentado:
# MAGIC
# MAGIC 1. `order_status = 'delivered'`
# MAGIC 2. Fecha de entrega real no nula
# MAGIC 3. Fecha estimada no nula
# MAGIC 4. Coherencia temporal: la entrega no puede ser anterior a la compra

# COMMAND ----------

n_inicial = orders.count()

paso1 = orders.filter(F.col("order_status") == "delivered")
n1 = paso1.count()

paso2 = paso1.filter(F.col("order_delivered_customer_date").isNotNull())
n2 = paso2.count()

paso3 = paso2.filter(F.col("order_estimated_delivery_date").isNotNull())
n3 = paso3.count()

paso4 = paso3.filter(
    F.col("order_delivered_customer_date") >= F.col("order_purchase_timestamp")
)
n4 = paso4.count()

orders_entregados = paso4

print("EMBUDO DE FILTRADO DE PEDIDOS")
print(f"  Pedidos en Bronze                        : {n_inicial:>8,}")
print(f"  (1) status = 'delivered'                 : {n1:>8,}  (-{n_inicial - n1:,})")
print(f"  (2) con fecha de entrega real            : {n2:>8,}  (-{n1 - n2:,})")
print(f"  (3) con fecha estimada                   : {n3:>8,}  (-{n2 - n3:,})")
print(f"  (4) coherencia temporal entrega >= compra: {n4:>8,}  (-{n3 - n4:,})")
print(f"  Retención total: {100.0 * n4 / n_inicial:.2f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Distribución de estados descartados (para la documentación)

# COMMAND ----------

display(
    orders.groupBy("order_status")
    .agg(F.count("*").alias("pedidos"))
    .withColumn(
        "porcentaje", F.round(100 * F.col("pedidos") / F.lit(n_inicial), 2)
    )
    .orderBy(F.col("pedidos").desc())
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Agregación de `order_items` a grano de pedido
# MAGIC
# MAGIC Dos salidas distintas de la misma tabla:
# MAGIC
# MAGIC - **`items_agg`**: métricas agregadas del pedido completo (cuántos ítems, valor total,
# MAGIC   flete total, cuántos vendedores distintos participan).
# MAGIC - **`item_principal`**: el ítem de mayor precio del pedido. Su `seller_id` es el que
# MAGIC   usaremos para geolocalizar el origen del envío en `02.2`.
# MAGIC
# MAGIC ### Por qué "el ítem más caro" y no "el primero"
# MAGIC
# MAGIC El 3% de los pedidos involucra más de un vendedor, o sea más de un punto de origen.
# MAGIC Como el modelo necesita **una** distancia por pedido, hay que elegir un vendedor
# MAGIC representativo. El ítem de mayor precio es el que más peso tiene en el flete y en la
# MAGIC logística del pedido, así que es la mejor aproximación de un solo punto.
# MAGIC
# MAGIC El desempate (`order_item_id`, luego `seller_id`) es explícito para que el notebook
# MAGIC sea **determinista**: sin él, dos ejecuciones sobre los mismos datos podrían elegir
# MAGIC vendedores distintos y los resultados no serían reproducibles.

# COMMAND ----------

items_agg = items.groupBy("order_id").agg(
    F.count("*").alias("n_items"),
    F.countDistinct("product_id").alias("n_productos_distintos"),
    F.countDistinct("seller_id").alias("n_vendedores"),
    F.round(F.sum("price"), 2).alias("valor_productos"),
    F.round(F.sum("freight_value"), 2).alias("valor_flete"),
    F.round(F.avg("price"), 2).alias("precio_promedio_item"),
    F.round(F.max("price"), 2).alias("precio_max_item"),
    F.max("shipping_limit_date").alias("shipping_limit_date"),
)

ventana_item = Window.partitionBy("order_id").orderBy(
    F.col("price").desc(),
    F.col("order_item_id").asc(),
    F.col("seller_id").asc(),
)

item_principal = (
    items.withColumn("_rn", F.row_number().over(ventana_item))
    .filter(F.col("_rn") == 1)
    .select(
        "order_id",
        F.col("seller_id").alias("seller_id_principal"),
        F.col("product_id").alias("product_id_principal"),
    )
)

print(f"items_agg      : {items_agg.count():,} pedidos")
print(f"item_principal : {item_principal.count():,} pedidos")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Agregación de `order_payments` a grano de pedido
# MAGIC
# MAGIC Un pedido puede pagarse con varios instrumentos (por ejemplo, voucher + tarjeta).
# MAGIC Agregamos el total y quedamos con el método de pago que aportó el mayor monto como
# MAGIC método principal.

# COMMAND ----------

pagos_agg = pagos.groupBy("order_id").agg(
    F.round(F.sum("payment_value"), 2).alias("valor_pagado"),
    F.max("payment_installments").alias("max_cuotas"),
    F.countDistinct("payment_type").alias("n_tipos_pago"),
    F.count("*").alias("n_transacciones_pago"),
)

ventana_pago = Window.partitionBy("order_id").orderBy(
    F.col("payment_value").desc(),
    F.col("payment_sequential").asc(),
)

pago_principal = (
    pagos.withColumn("_rn", F.row_number().over(ventana_pago))
    .filter(F.col("_rn") == 1)
    .select("order_id", F.col("payment_type").alias("tipo_pago_principal"))
)

print(f"pagos_agg : {pagos_agg.count():,} pedidos")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Deduplicación de `order_reviews`
# MAGIC
# MAGIC ### Aviso de fuga de datos (leakage)
# MAGIC
# MAGIC `review_score` se conoce **después** de que el pedido fue entregado. Si se usa como
# MAGIC feature para predecir el retraso, el modelo está mirando el futuro: los pedidos
# MAGIC atrasados reciben reseñas de 1 estrella, así que la métrica se dispararía en
# MAGIC validación y colapsaría en producción.
# MAGIC
# MAGIC Se conserva en Silver porque Silver es la **vista de negocio confiable** y el equipo
# MAGIC de BI la necesita para análisis descriptivo. Pero queda registrada en la lista
# MAGIC `COLUMNAS_CON_FUGA` que se documenta en `02.2` y se entrega a la capa Gold.
# MAGIC
# MAGIC Un pedido puede tener más de una reseña; conservamos la **más reciente**.

# COMMAND ----------

ventana_review = Window.partitionBy("order_id").orderBy(
    F.col("review_answer_timestamp").desc_nulls_last(),
    F.col("review_creation_date").desc_nulls_last(),
    F.col("review_id").asc(),
)

reviews_dedup = (
    reviews.withColumn("_rn", F.row_number().over(ventana_review))
    .filter(F.col("_rn") == 1)
    .select(
        "order_id",
        "review_score",
        F.col("review_creation_date").alias("fecha_review"),
    )
)

n_reviews_total = reviews.count()
n_reviews_dedup = reviews_dedup.count()
print(f"Reseñas en Bronze          : {n_reviews_total:,}")
print(f"Reseñas tras deduplicación : {n_reviews_dedup:,}")
print(f"Duplicados eliminados      : {n_reviews_total - n_reviews_dedup:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Dimensión de producto: traducción de categoría y volumen
# MAGIC
# MAGIC Se une la tabla de traducción para tener la categoría en inglés (más legible para el
# MAGIC reporte) y se calcula el volumen físico, que es un predictor logístico más informativo
# MAGIC que las tres dimensiones por separado.
# MAGIC
# MAGIC Las categorías nulas se etiquetan como `desconocida` en vez de dejarse en `null`:
# MAGIC así la capa Gold puede codificarlas como una categoría legítima en vez de perder filas.

# COMMAND ----------

productos = (
    productos_raw.join(traduccion_raw, on="product_category_name", how="left")
    .withColumn(
        "categoria_producto",
        F.coalesce(
            F.col("product_category_name_english"),
            F.col("product_category_name"),
            F.lit("desconocida"),
        ),
    )
    .withColumn(
        "volumen_producto_cm3",
        F.col("product_length_cm") * F.col("product_height_cm") * F.col("product_width_cm"),
    )
    .select(
        F.col("product_id").alias("product_id_principal"),
        "categoria_producto",
        F.col("product_weight_g").alias("peso_producto_g"),
        F.col("volumen_producto_cm3"),
        F.col("product_photos_qty").alias("n_fotos_producto"),
    )
)

display(productos.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. JOIN maestro
# MAGIC
# MAGIC Orden y tipo de cada unión, con su justificación:
# MAGIC
# MAGIC | Unión | Tipo | Por qué |
# MAGIC |---|---|---|
# MAGIC | `items_agg` | **inner** | Un pedido sin ítems no tiene contenido logístico que modelar. |
# MAGIC | `item_principal` | left | Derivada de la misma tabla; el inner anterior ya garantiza cobertura. |
# MAGIC | `pagos_agg` / `pago_principal` | left | Hay pedidos sin registro de pago; se conservan y se marcan. |
# MAGIC | `reviews_dedup` | left | ~1% de pedidos no tiene reseña. No es motivo para descartarlos. |
# MAGIC | `clientes` | left | Debería ser 1:1; el left permite auditar si no lo es. |
# MAGIC | `vendedores` | left | Vía `seller_id_principal`. |
# MAGIC | `productos` | left | Vía `product_id_principal`. |
# MAGIC
# MAGIC Todas las dimensiones van con `left` a propósito: si una falla, quiero **verlo en el
# MAGIC perfil de nulos**, no perder filas silenciosamente.

# COMMAND ----------

vendedores_join = vendedores.select(
    F.col("seller_id").alias("seller_id_principal"),
    F.col("seller_zip_code_prefix").alias("vendedor_zip_prefix"),
    F.col("seller_city").alias("vendedor_ciudad"),
    F.col("seller_state").alias("vendedor_estado"),
)

clientes_join = clientes.select(
    "customer_id",
    "customer_unique_id",
    F.col("customer_zip_code_prefix").alias("cliente_zip_prefix"),
    F.col("customer_city").alias("cliente_ciudad"),
    F.col("customer_state").alias("cliente_estado"),
)

silver = (
    orders_entregados.select(
        "order_id",
        "customer_id",
        "order_status",
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    )
    .join(items_agg, on="order_id", how="inner")
    .join(item_principal, on="order_id", how="left")
    .join(pagos_agg, on="order_id", how="left")
    .join(pago_principal, on="order_id", how="left")
    .join(reviews_dedup, on="order_id", how="left")
    .join(clientes_join, on="customer_id", how="left")
    .join(vendedores_join, on="seller_id_principal", how="left")
    .join(productos, on="product_id_principal", how="left")
)

print(f"Filas tras el JOIN maestro: {silver.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Variables derivadas conocidas al momento de la compra
# MAGIC
# MAGIC Regla que se respeta en todo el notebook: **solo se derivan variables cuyo valor es
# MAGIC conocido en el instante en que el pedido se realiza**. Ese es el momento en el que el
# MAGIC modelo tiene que predecir, así que cualquier cosa posterior sería fuga.
# MAGIC
# MAGIC - `dias_promesa`: cuántos días prometió Olist. Es el predictor más fuerte del problema
# MAGIC   y se conoce en el checkout.
# MAGIC - `ratio_flete`: qué proporción del ticket es transporte. Proxy de dificultad logística.
# MAGIC - Atributos de calendario: la estacionalidad brasileña (Carnaval, Black Friday, Navidad)
# MAGIC   afecta fuertemente los tiempos de entrega.
# MAGIC - `mismo_estado`: bandera cruda de proximidad. La distancia real se calcula en `02.2`.

# COMMAND ----------

silver = (
    silver.withColumn(
        "valor_total_pedido",
        F.round(F.coalesce(F.col("valor_productos"), F.lit(0.0))
                + F.coalesce(F.col("valor_flete"), F.lit(0.0)), 2),
    )
    .withColumn(
        "ratio_flete",
        F.round(
            F.when(
                F.col("valor_total_pedido") > 0,
                F.col("valor_flete") / F.col("valor_total_pedido"),
            ).otherwise(F.lit(None)),
            4,
        ),
    )
    .withColumn(
        "dias_promesa",
        F.datediff(
            F.to_date("order_estimated_delivery_date"),
            F.to_date("order_purchase_timestamp"),
        ),
    )
    .withColumn(
        "horas_hasta_aprobacion",
        F.round(
            (
                F.col("order_approved_at").cast("long")
                - F.col("order_purchase_timestamp").cast("long")
            )
            / 3600.0,
            2,
        ),
    )
    .withColumn("anio_compra", F.year("order_purchase_timestamp"))
    .withColumn("mes_compra", F.month("order_purchase_timestamp"))
    .withColumn("trimestre_compra", F.quarter("order_purchase_timestamp"))
    .withColumn("dia_semana_compra", F.dayofweek("order_purchase_timestamp"))
    .withColumn("hora_compra", F.hour("order_purchase_timestamp"))
    .withColumn(
        "es_fin_de_semana",
        F.when(F.col("dia_semana_compra").isin(1, 7), 1).otherwise(0),
    )
    .withColumn(
        "mismo_estado",
        F.when(F.col("cliente_estado") == F.col("vendedor_estado"), 1)
        .when(F.col("cliente_estado").isNull() | F.col("vendedor_estado").isNull(), None)
        .otherwise(0),
    )
    .withColumn(
        "multi_vendedor", F.when(F.col("n_vendedores") > 1, 1).otherwise(0)
    )
    .withColumn("tiene_review", F.when(F.col("review_score").isNotNull(), 1).otherwise(0))
)

print("Variables derivadas creadas.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Control de calidad antes de escribir
# MAGIC
# MAGIC Las aserciones fallan ruidosamente si algo salió mal. Es preferible que el notebook
# MAGIC reviente aquí a que escriba una tabla corrupta que Gold consuma sin darse cuenta.

# COMMAND ----------

n_filas = silver.count()
n_pedidos_unicos = silver.select("order_id").distinct().count()

print(f"Filas totales      : {n_filas:,}")
print(f"order_id distintos : {n_pedidos_unicos:,}")

assert n_filas == n_pedidos_unicos, (
    f"FAN-OUT DETECTADO: {n_filas:,} filas para {n_pedidos_unicos:,} pedidos únicos. "
    "Alguna unión no está a grano de pedido."
)

assert silver.filter(F.col("order_id").isNull()).count() == 0, "Hay order_id nulos."
assert silver.filter(F.col("dias_promesa") < 0).count() == 0, (
    "Hay pedidos con fecha estimada anterior a la compra."
)

print("\nTodas las aserciones de calidad pasaron. Grano correcto: 1 fila = 1 pedido.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Perfil de nulos
# MAGIC
# MAGIC Cobertura esperada: las columnas de cliente y vendedor deben rondar el 100%. Los
# MAGIC nulos en `peso_producto_g` y `volumen_producto_cm3` vienen del propio dataset de Olist
# MAGIC (productos sin ficha completa) y se imputan en Gold, no aquí: Silver reporta la
# MAGIC realidad, Gold decide la estrategia de imputación.

# COMMAND ----------

perfil = (
    silver.select(
        [
            F.round(100 * F.sum(F.col(c).isNull().cast("int")) / F.lit(n_filas), 2).alias(c)
            for c in silver.columns
        ]
    )
    .collect()[0]
    .asDict()
)

filas_perfil = sorted(
    [(col, pct) for col, pct in perfil.items()], key=lambda x: -x[1]
)

print(f"{'COLUMNA':<32} {'% NULOS':>8}")
print("-" * 42)
for col, pct in filas_perfil:
    marca = "  <-- revisar" if pct > 5 else ""
    print(f"{col:<32} {pct:>7.2f}%{marca}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Escritura de `silver_orders_joined`
# MAGIC
# MAGIC `overwriteSchema` permite reejecutar el notebook tras cambiar columnas sin tener que
# MAGIC borrar la tabla a mano.

# COMMAND ----------

(
    silver.write.format("delta")
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
# MAGIC
# MAGIC Los comentarios quedan visibles en el Catalog Explorer, lo que hace la tabla
# MAGIC autoexplicativa para el resto del equipo.

# COMMAND ----------

spark.sql(
    f"""
    COMMENT ON TABLE {T(TABLA_SALIDA)} IS
    'Capa Silver 1. Pedidos Olist entregados, con tablas transaccionales agregadas a grano
     de pedido (1 fila = 1 pedido). Insumo de 02.2_silver_geolocation_target.
     Responsable: Esteban.'
    """
)

COMENTARIOS_COLUMNAS = {
    "n_vendedores": "Vendedores distintos en el pedido. >1 implica varios orígenes de envío.",
    "seller_id_principal": "Vendedor del ítem más caro. Origen usado para la distancia en 02.2.",
    "dias_promesa": "Días entre la compra y la fecha estimada. Conocido en el checkout.",
    "ratio_flete": "valor_flete / valor_total_pedido. Proxy de dificultad logística.",
    "review_score": "FUGA DE DATOS: posterior a la entrega. No usar como feature de entrenamiento.",
    "shipping_limit_date": "Fecha límite pactada con el vendedor para despachar.",
}

columnas_escritas = spark.table(T(TABLA_SALIDA)).columns
for columna, comentario in COMENTARIOS_COLUMNAS.items():
    if columna in columnas_escritas:
        spark.sql(
            f"ALTER TABLE {T(TABLA_SALIDA)} ALTER COLUMN {columna} COMMENT '{comentario}'"
        )

print("Metadatos documentados en Unity Catalog.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Vista previa

# COMMAND ----------

display(
    spark.table(T(TABLA_SALIDA)).select(
        "order_id",
        "order_purchase_timestamp",
        "order_estimated_delivery_date",
        "order_delivered_customer_date",
        "dias_promesa",
        "n_items",
        "n_vendedores",
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
# MAGIC ## Resumen
# MAGIC
# MAGIC | Aspecto | Resultado |
# MAGIC |---|---|
# MAGIC | Grano | 1 fila = 1 pedido entregado (verificado con `assert`) |
# MAGIC | Fan-out | Evitado agregando antes de unir |
# MAGIC | Determinismo | Desempates explícitos en todas las funciones de ventana |
# MAGIC | Fuga de datos | `review_score` conservado pero marcado en el catálogo |
# MAGIC | Salida | `big_data_2026.olist.silver_orders_joined` |
# MAGIC
# MAGIC **Siguiente paso:** ejecutar `02.2_silver_geolocation_target`.
