# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # CAPA GOLD 1 — KPIs de negocio y tablas analíticas
# MAGIC
# MAGIC **Proyecto:** Predicción de Retrasos en Entregas — Olist E-commerce
# MAGIC **Responsable:** Marlon
# MAGIC **Entrada:** `silver_enriched`
# MAGIC **Salida:** 7 tablas `gold_*`, una por eje de análisis
# MAGIC
# MAGIC ```
# MAGIC 02.2_silver_geolocation_target
# MAGIC        ▼
# MAGIC 03.1_gold_business_kpis        <-- estás aquí
# MAGIC        ▼
# MAGIC 03.2_gold_ml_features
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Alcance
# MAGIC
# MAGIC Este notebook **no entrena nada**. Construye las tablas analíticas que responden a
# MAGIC los objetivos específicos 2 y 3 del proyecto: comportamiento de ventas, clientes,
# MAGIC productos y vendedores, e indicadores para BI/visualización. Cada tabla es un
# MAGIC `groupBy` agregado a un grano distinto (mes, categoría, estado, vendedor, cliente),
# MAGIC así que **no hay riesgo de fuga de datos**: Gold-BI puede usar `review_score`,
# MAGIC `days_delay` o cualquier columna de `silver_enriched` libremente, porque estas
# MAGIC tablas describen lo que **ya pasó**, no alimentan un modelo predictivo.
# MAGIC
# MAGIC La tabla lista para entrenar (con el contrato anti-fuga aplicado) vive en
# MAGIC `03.2_gold_ml_features`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Configuración e imports

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql import DataFrame

CATALOG = "big_data_2026"
SCHEMA = "olist"

TABLA_ENTRADA = "silver_enriched"


def T(nombre: str) -> str:
    """Devuelve el nombre completamente calificado de una tabla."""
    return f"{CATALOG}.{SCHEMA}.{nombre}"


def escribir_tabla(df: DataFrame, nombre: str, comentario: str) -> DataFrame:
    """Escribe una tabla Gold, la documenta en Unity Catalog e imprime un resumen.

    Centralizado acá porque este notebook escribe 7 tablas con el mismo patrón
    (overwrite + overwriteSchema + COMMENT ON TABLE); repetir el bloque 7 veces
    solo agregaría ruido sin agregar información.
    """
    (
        df.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(T(nombre))
    )
    comentario_sql = comentario.replace("'", "\\'")
    spark.sql(f"COMMENT ON TABLE {T(nombre)} IS '{comentario_sql}'")
    tabla = spark.table(T(nombre))
    print(f"   -> {nombre:<28} {tabla.count():>7,} filas   {len(tabla.columns)} columnas")
    return tabla


print(f"Catálogo/esquema : {CATALOG}.{SCHEMA}")
print(f"Entrada          : {T(TABLA_ENTRADA)}")

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
    "silver_enriched no está a grano de pedido. Revisar la capa Silver antes de continuar."
)

print(f"Contrato de entrada OK — {n_pedidos:,} pedidos, grano verificado.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. `gold_kpis_mensuales` — evolución temporal del negocio
# MAGIC
# MAGIC Serie mensual para el dashboard ejecutivo: volumen, ingresos, ticket promedio y
# MAGIC tasa de retraso mes a mes. Es la tabla que responde "¿estamos vendiendo más y
# MAGIC entregando peor, o al revés?".

# COMMAND ----------

gold_kpis_mensuales = (
    silver.groupBy("anio_compra", "mes_compra")
    .agg(
        F.count("*").alias("pedidos"),
        F.round(F.sum("valor_total_pedido"), 2).alias("ingresos_totales"),
        F.round(F.avg("valor_total_pedido"), 2).alias("ticket_promedio"),
        F.round(F.avg("valor_flete"), 2).alias("flete_promedio"),
        F.round(F.avg("n_items"), 2).alias("items_promedio_por_pedido"),
        F.round(100 * F.avg("is_late"), 2).alias("tasa_retraso_pct"),
        F.round(F.avg(F.when(F.col("is_late") == 1, F.col("days_delay"))), 2).alias(
            "retraso_promedio_dias"
        ),
        F.round(F.avg("review_score"), 2).alias("review_score_promedio"),
        F.countDistinct("customer_unique_id").alias("clientes_distintos"),
        F.countDistinct("seller_id_principal").alias("vendedores_activos"),
    )
    .withColumn(
        "periodo", F.format_string("%04d-%02d", F.col("anio_compra"), F.col("mes_compra"))
    )
    .orderBy("anio_compra", "mes_compra")
)

escribir_tabla(
    gold_kpis_mensuales,
    "gold_kpis_mensuales",
    "Capa Gold. Serie mensual de KPIs de negocio: volumen, ingresos, ticket promedio, "
    "tasa de retraso y actividad de clientes/vendedores. Grano: 1 fila = 1 mes. "
    "Fuente: silver_enriched. Responsable: Marlon.",
)

display(gold_kpis_mensuales)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. `gold_ventas_categoria` — productos
# MAGIC
# MAGIC Qué categorías venden más, cuál es su ticket, y —cruzando con logística— cuáles
# MAGIC son estructuralmente más difíciles de entregar a tiempo (peso, distancia).

# COMMAND ----------

gold_ventas_categoria = (
    silver.groupBy("categoria_producto")
    .agg(
        F.count("*").alias("pedidos"),
        F.round(F.sum("valor_total_pedido"), 2).alias("ingresos_totales"),
        F.round(F.avg("valor_total_pedido"), 2).alias("ticket_promedio"),
        F.round(F.avg("peso_producto_g"), 0).alias("peso_promedio_g"),
        F.round(F.avg("n_fotos_producto"), 1).alias("fotos_promedio"),
        F.round(100 * F.avg("is_late"), 2).alias("tasa_retraso_pct"),
        F.round(F.avg("distancia_km"), 1).alias("distancia_promedio_km"),
        F.round(F.avg("dias_promesa"), 1).alias("dias_promesa_promedio"),
        F.round(F.avg("review_score"), 2).alias("review_score_promedio"),
    )
    .orderBy(F.col("ingresos_totales").desc())
)

escribir_tabla(
    gold_ventas_categoria,
    "gold_ventas_categoria",
    "Capa Gold. Ventas, logística y retraso agregados por categoria_producto. "
    "Grano: 1 fila = 1 categoria. Fuente: silver_enriched. Responsable: Marlon.",
)

display(gold_ventas_categoria)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Geografía: `gold_ventas_geografia` y `gold_corredores_envio`
# MAGIC
# MAGIC Dos vistas geográficas complementarias:
# MAGIC
# MAGIC - **`gold_ventas_geografia`**: negocio por estado del **cliente** (demanda).
# MAGIC - **`gold_corredores_envio`**: por par vendedor→cliente (**corredor** de envío).
# MAGIC   Es la tabla que expone si el problema de retraso es de ciertos *corredores*
# MAGIC   específicos (p. ej. Norte→Sudeste) y no solo de la distancia en abstracto.
# MAGIC
# MAGIC Se filtran corredores con menos de 20 pedidos: con muestras chicas la tasa de
# MAGIC retraso es ruido estadístico, no señal, y mezclarlos en el ranking distorsiona
# MAGIC el análisis.

# COMMAND ----------

gold_ventas_geografia = (
    silver.groupBy("cliente_estado")
    .agg(
        F.count("*").alias("pedidos"),
        F.round(F.sum("valor_total_pedido"), 2).alias("ingresos_totales"),
        F.round(F.avg("valor_total_pedido"), 2).alias("ticket_promedio"),
        F.round(100 * F.avg("is_late"), 2).alias("tasa_retraso_pct"),
        F.round(F.avg("distancia_km"), 1).alias("distancia_promedio_km"),
        F.round(F.avg("dias_promesa"), 1).alias("dias_promesa_promedio"),
        F.countDistinct("customer_unique_id").alias("clientes_distintos"),
    )
    .orderBy(F.col("ingresos_totales").desc())
)

escribir_tabla(
    gold_ventas_geografia,
    "gold_ventas_geografia",
    "Capa Gold. Ventas, ticket y tasa de retraso por estado del cliente (demanda). "
    "Grano: 1 fila = 1 estado. Fuente: silver_enriched. Responsable: Marlon.",
)

display(gold_ventas_geografia)

MIN_PEDIDOS_CORREDOR = 20

gold_corredores_envio = (
    silver.groupBy("vendedor_estado", "cliente_estado")
    .agg(
        F.count("*").alias("pedidos"),
        F.round(F.sum("valor_total_pedido"), 2).alias("ingresos_totales"),
        F.round(100 * F.avg("is_late"), 2).alias("tasa_retraso_pct"),
        F.round(F.avg("distancia_km"), 1).alias("distancia_promedio_km"),
        F.round(F.avg("dias_promesa"), 1).alias("dias_promesa_promedio"),
    )
    .filter(F.col("pedidos") >= MIN_PEDIDOS_CORREDOR)
    .orderBy(F.col("tasa_retraso_pct").desc())
)

escribir_tabla(
    gold_corredores_envio,
    "gold_corredores_envio",
    f"Capa Gold. Tasa de retraso por corredor vendedor_estado -> cliente_estado, "
    f"filtrado a corredores con >= {MIN_PEDIDOS_CORREDOR} pedidos para evitar ruido "
    "estadístico. Grano: 1 fila = 1 corredor. Fuente: silver_enriched. Responsable: Marlon.",
)

display(gold_corredores_envio.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. `gold_performance_vendedores`
# MAGIC
# MAGIC Un `seller_id_principal` con muy pocos pedidos tiene una tasa de retraso inestable
# MAGIC (un solo pedido tarde ya es 100%). Se marca con `muestra_pequena` en vez de
# MAGIC descartarse, para que el consumidor del dashboard decida si lo filtra.

# COMMAND ----------

UMBRAL_MUESTRA_PEQUENA = 5

gold_performance_vendedores = (
    silver.groupBy("seller_id_principal", "vendedor_estado", "vendedor_ciudad")
    .agg(
        F.count("*").alias("pedidos"),
        F.round(F.sum("valor_productos"), 2).alias("ingresos_generados"),
        F.round(F.avg("valor_total_pedido"), 2).alias("ticket_promedio"),
        F.round(100 * F.avg("is_late"), 2).alias("tasa_retraso_pct"),
        F.round(F.avg("distancia_km"), 1).alias("distancia_promedio_km"),
        F.countDistinct("categoria_producto").alias("categorias_distintas"),
        F.round(100 * F.avg("multi_vendedor"), 2).alias("pct_pedidos_multivendedor"),
        F.round(F.avg("review_score"), 2).alias("review_score_promedio"),
    )
    .withColumn(
        "muestra_pequena",
        F.when(F.col("pedidos") < UMBRAL_MUESTRA_PEQUENA, 1).otherwise(0),
    )
    .orderBy(F.col("ingresos_generados").desc())
)

escribir_tabla(
    gold_performance_vendedores,
    "gold_performance_vendedores",
    "Capa Gold. Performance comercial y logística por vendedor principal. Grano: 1 fila "
    "= 1 seller_id_principal. muestra_pequena=1 si pedidos < "
    f"{UMBRAL_MUESTRA_PEQUENA} (tasa_retraso_pct poco confiable). "
    "Fuente: silver_enriched. Responsable: Marlon.",
)

display(gold_performance_vendedores.orderBy(F.col("tasa_retraso_pct").desc()).limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. `gold_comportamiento_clientes`
# MAGIC
# MAGIC En Olist la enorme mayoría de los `customer_unique_id` compra **una sola vez**
# MAGIC (es un patrón conocido del dataset, no un error del pipeline). Esta tabla lo
# MAGIC deja explícito con `cliente_recurrente` en vez de que alguien lo asuma mal en
# MAGIC un dashboard de retención.

# COMMAND ----------

gold_comportamiento_clientes = (
    silver.groupBy("customer_unique_id")
    .agg(
        F.count("*").alias("n_pedidos"),
        F.round(F.sum("valor_total_pedido"), 2).alias("gasto_total"),
        F.round(F.avg("valor_total_pedido"), 2).alias("ticket_promedio"),
        F.countDistinct("categoria_producto").alias("categorias_distintas"),
        F.round(100 * F.avg("is_late"), 2).alias("tasa_retraso_propia_pct"),
        F.min("order_purchase_timestamp").alias("primera_compra"),
        F.max("order_purchase_timestamp").alias("ultima_compra"),
        F.first("cliente_estado").alias("cliente_estado"),
    )
    .withColumn(
        "cliente_recurrente", F.when(F.col("n_pedidos") > 1, 1).otherwise(0)
    )
    .orderBy(F.col("gasto_total").desc())
)

n_recurrentes = gold_comportamiento_clientes.filter(
    F.col("cliente_recurrente") == 1
).count()
n_clientes_total = gold_comportamiento_clientes.count()
print(
    f"Clientes recurrentes: {n_recurrentes:,} de {n_clientes_total:,} "
    f"({100.0 * n_recurrentes / n_clientes_total:.2f}%)"
)

escribir_tabla(
    gold_comportamiento_clientes,
    "gold_comportamiento_clientes",
    "Capa Gold. Gasto, frecuencia y tasa de retraso propia por customer_unique_id. "
    "Grano: 1 fila = 1 cliente único. cliente_recurrente=1 si tuvo mas de 1 pedido "
    "(la mayoria de clientes Olist compra una sola vez). "
    "Fuente: silver_enriched. Responsable: Marlon.",
)

display(gold_comportamiento_clientes.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. `gold_factores_retraso` — tabla larga para el objetivo específico 1
# MAGIC
# MAGIC Las tablas anteriores están cada una a un grano distinto (mes, categoría, estado...),
# MAGIC lo que las hace excelentes para sus propios dashboards pero incómodas para comparar
# MAGIC "¿qué factor mueve más la tasa de retraso?" de un vistazo. Esta tabla resuelve eso:
# MAGIC **formato largo** (`dimension`, `segmento`, `pedidos`, `tasa_retraso_pct`), con una
# MAGIC fila por segmento de cada dimensión analizada, pensada para alimentar un único
# MAGIC gráfico de barras comparativo en el dashboard de BI.
# MAGIC
# MAGIC No repite factores ya cubiertos por `gold_ventas_categoria` / `gold_ventas_geografia`
# MAGIC (categoría y estado): agrega las dimensiones operativas que todavía no tienen tabla
# MAGIC propia.

# COMMAND ----------


def resumen_factor(df: DataFrame, columna: str, dimension: str) -> DataFrame:
    """Agrega `df` por `columna` y devuelve el resultado en formato largo estandarizado.

    Reutilizado para cada dimensión analizada; evita repetir el mismo groupBy con
    columnas renombradas siete veces.
    """
    return (
        df.filter(F.col(columna).isNotNull())
        .groupBy(F.col(columna).alias("segmento"))
        .agg(
            F.count("*").alias("pedidos"),
            F.round(100 * F.avg("is_late"), 2).alias("tasa_retraso_pct"),
            F.round(
                F.avg(F.when(F.col("is_late") == 1, F.col("days_delay"))), 2
            ).alias("retraso_promedio_dias"),
        )
        .withColumn("dimension", F.lit(dimension))
        .withColumn("segmento", F.col("segmento").cast("string"))
        .select("dimension", "segmento", "pedidos", "tasa_retraso_pct", "retraso_promedio_dias")
    )


# Dimensiones categóricas / binarias: agregación directa.
factores_binarios = [
    resumen_factor(silver, "dia_semana_compra", "dia_semana_compra"),
    resumen_factor(silver, "es_fin_de_semana", "es_fin_de_semana"),
    resumen_factor(silver, "mismo_estado", "mismo_estado"),
    resumen_factor(silver, "multi_vendedor", "multi_vendedor"),
    resumen_factor(silver, "trimestre_compra", "trimestre_compra"),
    resumen_factor(silver, "tipo_pago_principal", "metodo_pago"),
]

# Dimensiones continuas: se discretizan en cuartiles antes de agregar. El orden del
# cuartil es el propio dato (ntile), así que "Q1" siempre es "más corto/más cercano".
silver_dist = silver.filter(F.col("distancia_km").isNotNull()).withColumn(
    "cuartil_distancia",
    F.concat(F.lit("Q"), F.ntile(4).over(Window.orderBy("distancia_km"))),
)
silver_promesa = silver.filter(F.col("dias_promesa").isNotNull()).withColumn(
    "cuartil_dias_promesa",
    F.concat(F.lit("Q"), F.ntile(4).over(Window.orderBy("dias_promesa"))),
)

factores_continuos = [
    resumen_factor(silver_dist, "cuartil_distancia", "cuartil_distancia_km"),
    resumen_factor(silver_promesa, "cuartil_dias_promesa", "cuartil_dias_promesa"),
]

gold_factores_retraso = factores_binarios[0]
for tabla in factores_binarios[1:] + factores_continuos:
    gold_factores_retraso = gold_factores_retraso.unionByName(tabla)

gold_factores_retraso = gold_factores_retraso.orderBy(
    "dimension", F.col("tasa_retraso_pct").desc()
)

escribir_tabla(
    gold_factores_retraso,
    "gold_factores_retraso",
    "Capa Gold. Tabla larga (dimension, segmento, pedidos, tasa_retraso_pct) para "
    "comparar factores de retraso en un solo gráfico. Dimensiones: dia_semana_compra, "
    "es_fin_de_semana, mismo_estado, multi_vendedor, trimestre_compra, metodo_pago, "
    "cuartil_distancia_km, cuartil_dias_promesa. No incluye categoria_producto ni "
    "estado (ver gold_ventas_categoria / gold_ventas_geografia). "
    "Fuente: silver_enriched. Responsable: Marlon.",
)

display(gold_factores_retraso)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Control de calidad
# MAGIC
# MAGIC Cada tabla Gold debe ser 1 fila por combinación de sus columnas de agrupación
# MAGIC (sin fan-out) y los ingresos totales agregados no pueden superar el total de
# MAGIC `silver_enriched` (solo pueden ser iguales o menores si hay nulos en la dimensión).

# COMMAND ----------

ingresos_silver_total = silver.select(
    F.round(F.sum("valor_total_pedido"), 2).alias("t")
).collect()[0]["t"]

tablas_a_validar = {
    "gold_kpis_mensuales": (["anio_compra", "mes_compra"], "ingresos_totales"),
    "gold_ventas_categoria": (["categoria_producto"], "ingresos_totales"),
    "gold_ventas_geografia": (["cliente_estado"], "ingresos_totales"),
    "gold_performance_vendedores": (["seller_id_principal"], "ingresos_generados"),
    "gold_comportamiento_clientes": (["customer_unique_id"], "gasto_total"),
}

print("VALIDACIÓN DE GRANO E INGRESOS POR TABLA GOLD")
print("=" * 65)
for nombre, (claves, columna_ingreso) in tablas_a_validar.items():
    tabla = spark.table(T(nombre))
    n_filas = tabla.count()
    n_claves_unicas = tabla.select(*claves).distinct().count()
    assert n_filas == n_claves_unicas, (
        f"FAN-OUT en {nombre}: {n_filas:,} filas para {n_claves_unicas:,} combinaciones "
        f"únicas de {claves}."
    )
    ingresos_tabla = tabla.select(F.sum(columna_ingreso).alias("t")).collect()[0]["t"] or 0.0
    assert ingresos_tabla <= ingresos_silver_total * 1.001, (
        f"{nombre} reporta más ingresos ({ingresos_tabla:,.2f}) que silver_enriched "
        f"({ingresos_silver_total:,.2f}). Revisar la agregación."
    )
    print(f"  OK  {nombre:<28} grano correcto, ingresos <= total silver_enriched")

print(f"\nIngresos totales en silver_enriched: {ingresos_silver_total:,.2f}")
print("\nTodas las validaciones pasaron.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resumen de la capa Gold — KPIs de negocio
# MAGIC
# MAGIC | Tabla | Grano | Responde |
# MAGIC |---|---|---|
# MAGIC | `gold_kpis_mensuales` | 1 mes | ¿Cómo evoluciona el negocio y el retraso en el tiempo? |
# MAGIC | `gold_ventas_categoria` | 1 categoría | ¿Qué se vende y qué categorías son logísticamente difíciles? |
# MAGIC | `gold_ventas_geografia` | 1 estado (cliente) | ¿Dónde está la demanda y dónde el retraso? |
# MAGIC | `gold_corredores_envio` | 1 corredor vendedor→cliente | ¿Qué rutas específicas fallan más? |
# MAGIC | `gold_performance_vendedores` | 1 vendedor | ¿Qué vendedores generan más ingreso y cuáles entregan peor? |
# MAGIC | `gold_comportamiento_clientes` | 1 cliente único | ¿Cuánto gasta y compra cada cliente? |
# MAGIC | `gold_factores_retraso` | 1 segmento (formato largo) | Comparativo único de todos los factores no geográficos/de producto. |
# MAGIC
# MAGIC **Siguiente paso:** ejecutar `03.2_gold_ml_features` para la tabla de entrenamiento.