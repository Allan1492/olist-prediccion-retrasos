# Databricks notebook source
# MAGIC %md
# MAGIC # CAPA SILVER 2 — Geolocalización y Variable Objetivo
# MAGIC
# MAGIC **Proyecto:** Predicción de Retrasos en Entregas — Olist E-commerce
# MAGIC **Responsable:** ESTEBAN
# MAGIC **Entrada:** `silver_orders_joined` + `bronze_olist_geolocation_dataset`
# MAGIC **Salida:** `silver_enriched` (grano: 1 fila = 1 pedido entregado)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Qué resuelve este notebook
# MAGIC
# MAGIC 1. **Geolocalización.** `olist_geolocation_dataset` tiene ~1M de filas y es la tabla
# MAGIC    más sucia del dataset: múltiples coordenadas por prefijo postal y outliers fuera de
# MAGIC    Brasil. Requiere limpieza y agregación antes de poder usarse.
# MAGIC 2. **Distancia vendedor → cliente** mediante la fórmula de Haversine.
# MAGIC 3. **Variable objetivo** `is_late` y `days_delay`.
# MAGIC 4. **Auditoría de fuga de datos**: qué columnas puede usar Gold y cuáles no.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Configuración

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import Window

CATALOG = "big_data_2026"
SCHEMA = "olist"

TABLA_ENTRADA = "silver_orders_joined"
TABLA_SALIDA = "silver_enriched"

RADIO_TIERRA_KM = 6371.0

# Bounding box continental de Brasil.
# Fuente: extremos geográficos oficiales (IBGE), con un margen de seguridad.
BRASIL_LAT_MIN, BRASIL_LAT_MAX = -33.75, 5.27
BRASIL_LNG_MIN, BRASIL_LNG_MAX = -73.99, -34.79

# Umbrales de validación de distancia. Ver la sección 4 para la justificación.
# Cota dura  = diagonal del bounding box de arriba (5.982 km), redondeada hacia arriba.
#              Superarla es imposible; si pasa, el error está en el código.
# Cota blanda = mayor distancia real dentro de Brasil (Roraima -> Chuí, 4.399 km).
#              Superarla es posible pero sospechoso: se avisa, no se falla.
COTA_DURA_KM = 6000.0
COTA_BLANDA_KM = 4400.0


def T(nombre: str) -> str:
    return f"{CATALOG}.{SCHEMA}.{nombre}"


print(f"Entrada : {T(TABLA_ENTRADA)}")
print(f"Salida  : {T(TABLA_SALIDA)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Contrato de entrada

# COMMAND ----------

existentes = {
    fila.tableName for fila in spark.sql(f"SHOW TABLES IN {CATALOG}.{SCHEMA}").collect()
}

assert TABLA_ENTRADA in existentes, (
    f"No existe {T(TABLA_ENTRADA)}. Ejecuta primero 02.1_silver_transformation."
)
assert "bronze_olist_geolocation_dataset" in existentes, (
    "No existe la tabla de geolocalización en Bronze."
)

pedidos = spark.table(T(TABLA_ENTRADA))
n_pedidos_entrada = pedidos.count()

# El grano de entrada tiene que ser 1 fila = 1 pedido. Si 02.1 se corrompió, se detecta aquí.
assert n_pedidos_entrada == pedidos.select("order_id").distinct().count(), (
    "La tabla de entrada no está a grano de pedido."
)

print(f"Contrato de entrada OK — {n_pedidos_entrada:,} pedidos, grano verificado.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Limpieza de la tabla de geolocalización
# MAGIC
# MAGIC ### Los dos problemas de esta tabla
# MAGIC
# MAGIC **Problema 1 — cardinalidad.** No hay una fila por prefijo postal, hay una por cada
# MAGIC dirección registrada. Un prefijo urbano de São Paulo puede tener miles de coordenadas.
# MAGIC Unirla directo contra los pedidos multiplicaría las filas por miles.
# MAGIC
# MAGIC **Problema 2 — outliers.** Hay coordenadas fuera de Brasil: errores de digitación,
# MAGIC signos invertidos y ceros. Contaminan cualquier estadístico que se calcule.
# MAGIC
# MAGIC ### Por qué mediana y no promedio
# MAGIC
# MAGIC Un solo punto mal digitado (por ejemplo, longitud positiva cuando toda Brasil es
# MAGIC negativa) arrastra el **promedio** del prefijo cientos de kilómetros. La **mediana**
# MAGIC tiene punto de ruptura del 50%: hace falta que la mitad de los puntos estén mal para
# MAGIC moverla. Con datos de direcciones capturadas por usuarios, es la elección correcta.
# MAGIC
# MAGIC El filtro de bounding box se aplica **antes** de agregar, para que los outliers ni
# MAGIC siquiera entren al cálculo.

# COMMAND ----------

geo_raw = spark.table(T("bronze_olist_geolocation_dataset"))
n_geo_raw = geo_raw.count()

geo_limpio = (
    geo_raw.withColumn("lat", F.col("geolocation_lat").cast("double"))
    .withColumn("lng", F.col("geolocation_lng").cast("double"))
    .withColumn("zip_prefix", F.col("geolocation_zip_code_prefix").cast("int"))
    .filter(F.col("lat").isNotNull() & F.col("lng").isNotNull())
    .filter(F.col("lat").between(BRASIL_LAT_MIN, BRASIL_LAT_MAX))
    .filter(F.col("lng").between(BRASIL_LNG_MIN, BRASIL_LNG_MAX))
    .filter(F.col("zip_prefix").isNotNull())
)

n_geo_limpio = geo_limpio.count()

print("LIMPIEZA DE GEOLOCALIZACIÓN")
print(f"  Filas en Bronze                  : {n_geo_raw:>9,}")
print(f"  Filas dentro del territorio      : {n_geo_limpio:>9,}")
print(f"  Outliers descartados             : {n_geo_raw - n_geo_limpio:>9,}"
      f"  ({100.0 * (n_geo_raw - n_geo_limpio) / n_geo_raw:.3f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Agregación a un punto por prefijo postal

# COMMAND ----------

geo_zip = geo_limpio.groupBy("zip_prefix").agg(
    F.expr("percentile_approx(lat, 0.5)").alias("zip_lat"),
    F.expr("percentile_approx(lng, 0.5)").alias("zip_lng"),
    F.count("*").alias("zip_n_puntos"),
)

# Centroide por estado: red de seguridad para prefijos sin cobertura.
geo_estado = geo_limpio.groupBy(
    F.col("geolocation_state").alias("estado")
).agg(
    F.expr("percentile_approx(lat, 0.5)").alias("estado_lat"),
    F.expr("percentile_approx(lng, 0.5)").alias("estado_lng"),
)

print(f"Prefijos postales únicos : {geo_zip.count():,}")
print(f"Estados con centroide    : {geo_estado.count():,}")

display(geo_zip.orderBy(F.col("zip_n_puntos").desc()).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Asignación de coordenadas con fallback en cascada
# MAGIC
# MAGIC No todos los prefijos postales de los pedidos aparecen en la tabla de geolocalización.
# MAGIC Descartar esos pedidos perdería datos válidos; dejarlos en `null` rompería el cálculo
# MAGIC de distancia. La estrategia es una **cascada de precisión decreciente**, con una
# MAGIC bandera que registra qué nivel se usó en cada fila:
# MAGIC
# MAGIC | Nivel | Fuente | Precisión | Bandera |
# MAGIC |---|---|---|---|
# MAGIC | 1 | Mediana del prefijo postal | ~1–5 km | `zip` |
# MAGIC | 2 | Centroide del estado | ~100–300 km | `estado` |
# MAGIC | 3 | Sin dato | — | `sin_dato` |
# MAGIC
# MAGIC La bandera es tan importante como la coordenada: permite que Gold entrene solo con
# MAGIC nivel `zip` si la imputación por estado resulta demasiado ruidosa, y deja la decisión
# MAGIC documentada en vez de escondida.

# COMMAND ----------

geo_cliente = geo_zip.select(
    F.col("zip_prefix").alias("cliente_zip_prefix"),
    F.col("zip_lat").alias("_cli_zip_lat"),
    F.col("zip_lng").alias("_cli_zip_lng"),
)
geo_vendedor = geo_zip.select(
    F.col("zip_prefix").alias("vendedor_zip_prefix"),
    F.col("zip_lat").alias("_ven_zip_lat"),
    F.col("zip_lng").alias("_ven_zip_lng"),
)
centroide_cliente = geo_estado.select(
    F.col("estado").alias("cliente_estado"),
    F.col("estado_lat").alias("_cli_est_lat"),
    F.col("estado_lng").alias("_cli_est_lng"),
)
centroide_vendedor = geo_estado.select(
    F.col("estado").alias("vendedor_estado"),
    F.col("estado_lat").alias("_ven_est_lat"),
    F.col("estado_lng").alias("_ven_est_lng"),
)

df = (
    pedidos.join(geo_cliente, on="cliente_zip_prefix", how="left")
    .join(geo_vendedor, on="vendedor_zip_prefix", how="left")
    .join(centroide_cliente, on="cliente_estado", how="left")
    .join(centroide_vendedor, on="vendedor_estado", how="left")
)

df = (
    df.withColumn("cliente_lat", F.coalesce("_cli_zip_lat", "_cli_est_lat"))
    .withColumn("cliente_lng", F.coalesce("_cli_zip_lng", "_cli_est_lng"))
    .withColumn("vendedor_lat", F.coalesce("_ven_zip_lat", "_ven_est_lat"))
    .withColumn("vendedor_lng", F.coalesce("_ven_zip_lng", "_ven_est_lng"))
    .withColumn(
        "geo_cliente_nivel",
        F.when(F.col("_cli_zip_lat").isNotNull(), F.lit("zip"))
        .when(F.col("_cli_est_lat").isNotNull(), F.lit("estado"))
        .otherwise(F.lit("sin_dato")),
    )
    .withColumn(
        "geo_vendedor_nivel",
        F.when(F.col("_ven_zip_lat").isNotNull(), F.lit("zip"))
        .when(F.col("_ven_est_lat").isNotNull(), F.lit("estado"))
        .otherwise(F.lit("sin_dato")),
    )
    .drop(
        "_cli_zip_lat", "_cli_zip_lng", "_cli_est_lat", "_cli_est_lng",
        "_ven_zip_lat", "_ven_zip_lng", "_ven_est_lat", "_ven_est_lng",
    )
)

print("COBERTURA GEOGRÁFICA")
display(
    df.groupBy("geo_cliente_nivel", "geo_vendedor_nivel")
    .agg(F.count("*").alias("pedidos"))
    .withColumn(
        "porcentaje",
        F.round(100 * F.col("pedidos") / F.lit(n_pedidos_entrada), 2),
    )
    .orderBy(F.col("pedidos").desc())
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Distancia Haversine
# MAGIC
# MAGIC La fórmula calcula la distancia de círculo máximo entre dos puntos de una esfera:
# MAGIC
# MAGIC $$a = \sin^2\left(\frac{\Delta\varphi}{2}\right) + \cos\varphi_1 \cdot \cos\varphi_2 \cdot \sin^2\left(\frac{\Delta\lambda}{2}\right)$$
# MAGIC $$d = 2R \cdot \arcsin\left(\sqrt{a}\right)$$
# MAGIC
# MAGIC donde \\(\varphi\\) es latitud, \\(\lambda\\) longitud y \\(R = 6371\\) km.
# MAGIC
# MAGIC ### Por qué funciones nativas y no una UDF de Python
# MAGIC
# MAGIC Es tentador escribir `@udf` con el módulo `math`. Sería un error en tres frentes:
# MAGIC
# MAGIC - **Rendimiento.** Una UDF de Python serializa cada fila entre la JVM y el intérprete
# MAGIC   de Python. Sobre 100k filas es tolerable; sobre millones, órdenes de magnitud más lento.
# MAGIC - **Optimización.** Catalyst trata las UDF como cajas negras: no puede reordenar ni
# MAGIC   simplificar la expresión. Con funciones nativas la fórmula entera se compila.
# MAGIC - **Compatibilidad.** El cómputo serverless de Databricks Free Edition impone
# MAGIC   restricciones sobre UDFs de Python. Las funciones nativas siempre funcionan.
# MAGIC
# MAGIC ### Nota sobre precisión
# MAGIC
# MAGIC Haversine asume una Tierra esférica; el error frente al elipsoide real es < 0.5%.
# MAGIC Para predecir retrasos logísticos eso es irrelevante — la distancia por carretera
# MAGIC introduce un error mucho mayor que el modelo esférico.

# COMMAND ----------

lat1 = F.radians(F.col("vendedor_lat"))
lng1 = F.radians(F.col("vendedor_lng"))
lat2 = F.radians(F.col("cliente_lat"))
lng2 = F.radians(F.col("cliente_lng"))

a_haversine = F.pow(F.sin((lat2 - lat1) / 2), 2) + F.cos(lat1) * F.cos(lat2) * F.pow(
    F.sin((lng2 - lng1) / 2), 2
)

# least(a, 1) protege contra que el error de punto flotante empuje `a` por encima de 1,
# lo que haría que asin() devuelva NaN en pedidos donde origen y destino coinciden.
distancia_km = 2 * F.lit(RADIO_TIERRA_KM) * F.asin(
    F.sqrt(F.least(a_haversine, F.lit(1.0)))
)

df = df.withColumn(
    "distancia_km",
    F.when(
        F.col("cliente_lat").isNotNull() & F.col("vendedor_lat").isNotNull(),
        F.round(distancia_km, 3),
    ).otherwise(F.lit(None).cast("double")),
).withColumn(
    "distancia_confiable",
    F.when(
        (F.col("geo_cliente_nivel") == "zip") & (F.col("geo_vendedor_nivel") == "zip"), 1
    ).otherwise(0),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Validación de la distancia
# MAGIC
# MAGIC Se usan **dos umbrales distintos**, y la diferencia entre ambos importa:
# MAGIC
# MAGIC - **Cota dura (`assert`, 6.000 km).** Es la diagonal del bounding box con el que se
# MAGIC   filtró la geolocalización: 5.982 km. Ninguna distancia calculada a partir de puntos
# MAGIC   que pasaron el filtro puede superarla. Si el `assert` falla, hay un error de código,
# MAGIC   no de datos. Poner aquí un umbral más bajo haría que el notebook reventara con
# MAGIC   coordenadas perfectamente válidas.
# MAGIC - **Cota blanda (aviso, 4.400 km).** Es la extensión real del territorio brasileño
# MAGIC   (Roraima → Chuí ≈ 4.399 km). Superarla no es imposible —el bounding box es un
# MAGIC   rectángulo y Brasil no—, pero sí es sospechoso y merece revisión manual.
# MAGIC
# MAGIC Confundir las dos cotas es un error común: convierte una advertencia razonable en un
# MAGIC fallo del pipeline.

# COMMAND ----------

df_con_distancia = df.filter(F.col("distancia_km").isNotNull())

# Sin esta guarda, F.min sobre cero filas devuelve null y el formateo numérico de abajo
# revienta con un TypeError que no explica nada sobre la causa real.
assert df_con_distancia.count() > 0, (
    "Ningún pedido tiene distancia calculable. Revisar la unión con geolocalización: "
    "probablemente los tipos de zip_prefix no coinciden entre las tablas."
)

estadisticas_distancia = df_con_distancia.select(
    F.count("*").alias("con_distancia"),
    F.round(F.min("distancia_km"), 2).alias("minimo_km"),
    F.round(F.expr("percentile_approx(distancia_km, 0.25)"), 2).alias("p25_km"),
    F.round(F.expr("percentile_approx(distancia_km, 0.50)"), 2).alias("mediana_km"),
    F.round(F.expr("percentile_approx(distancia_km, 0.75)"), 2).alias("p75_km"),
    F.round(F.expr("percentile_approx(distancia_km, 0.99)"), 2).alias("p99_km"),
    F.round(F.max("distancia_km"), 2).alias("maximo_km"),
).collect()[0]

print("DISTRIBUCIÓN DE DISTANCIA VENDEDOR -> CLIENTE")
for campo, valor in estadisticas_distancia.asDict().items():
    print(f"  {campo:<16}: {valor:>12,.2f}")

maximo_km = estadisticas_distancia["maximo_km"]

assert estadisticas_distancia["minimo_km"] >= 0, "Distancia negativa: fórmula incorrecta."
assert maximo_km <= COTA_DURA_KM, (
    f"Distancia máxima de {maximo_km:.0f} km, imposible con el bounding box aplicado "
    f"(diagonal = {COTA_DURA_KM:.0f} km). Hay un error en la fórmula o en el filtro."
)

if maximo_km > COTA_BLANDA_KM:
    print(
        f"\nAVISO: la distancia máxima ({maximo_km:.0f} km) supera la extensión real de "
        f"Brasil ({COTA_BLANDA_KM:.0f} km). Es geométricamente posible dentro del "
        "bounding box, pero conviene revisar esos pedidos."
    )
else:
    print("\nValidación geométrica superada: todas las distancias caben en el territorio.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Variable objetivo
# MAGIC
# MAGIC ### La sutileza que decide la calidad del target
# MAGIC
# MAGIC `order_estimated_delivery_date` es una **fecha** y llega con hora `00:00:00`.
# MAGIC `order_delivered_customer_date` es un **timestamp** con hora real de entrega.
# MAGIC
# MAGIC Comparar los dos directamente:
# MAGIC
# MAGIC ```python
# MAGIC F.col("order_delivered_customer_date") > F.col("order_estimated_delivery_date")   # INCORRECTO
# MAGIC ```
# MAGIC
# MAGIC marca como **atrasado** un pedido entregado a las 15:00 del mismo día prometido,
# MAGIC porque `2018-03-10 15:00:00 > 2018-03-10 00:00:00`. Comercialmente ese pedido llegó
# MAGIC a tiempo: Olist prometió un *día*, no una hora.
# MAGIC
# MAGIC Ese error infla la tasa de retraso en varios puntos porcentuales y produce un target
# MAGIC que no corresponde a la promesa real del negocio.
# MAGIC
# MAGIC **Solución:** truncar ambas a fecha antes de comparar.
# MAGIC
# MAGIC ### Definiciones
# MAGIC
# MAGIC - `days_delay` = `fecha_entrega - fecha_estimada`. Negativo = adelantado.
# MAGIC - `is_late` = 1 si `days_delay > 0`.
# MAGIC
# MAGIC Se conserva `days_delay` además de `is_late` porque permite reformular el problema
# MAGIC como regresión sin rehacer la capa Silver, y porque un retraso de 1 día y uno de 30
# MAGIC no cuestan lo mismo.

# COMMAND ----------

df = (
    df.withColumn("fecha_compra", F.to_date("order_purchase_timestamp"))
    .withColumn("fecha_entrega_real", F.to_date("order_delivered_customer_date"))
    .withColumn("fecha_entrega_estimada", F.to_date("order_estimated_delivery_date"))
    .withColumn("fecha_envio_transportista", F.to_date("order_delivered_carrier_date"))
)

df = (
    df.withColumn(
        "days_delay",
        F.datediff(F.col("fecha_entrega_real"), F.col("fecha_entrega_estimada")),
    )
    .withColumn("is_late", (F.col("days_delay") > 0).cast("int"))
    .withColumn(
        "dias_entrega_real",
        F.datediff(F.col("fecha_entrega_real"), F.col("fecha_compra")),
    )
    .withColumn(
        "dias_hasta_transportista",
        F.datediff(F.col("fecha_envio_transportista"), F.col("fecha_compra")),
    )
    .withColumn(
        "severidad_retraso",
        F.when(F.col("days_delay") <= 0, F.lit("a_tiempo"))
        .when(F.col("days_delay") <= 3, F.lit("leve_1_3d"))
        .when(F.col("days_delay") <= 7, F.lit("moderado_4_7d"))
        .when(F.col("days_delay") <= 30, F.lit("grave_8_30d"))
        .otherwise(F.lit("critico_30d_mas")),
    )
)

print("Variable objetivo construida.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Distribución del target
# MAGIC
# MAGIC En Olist la tasa de retraso ronda el **6–9%**. Es un problema de **clases
# MAGIC desbalanceadas**, un hecho que la capa Gold tiene que manejar explícitamente:
# MAGIC un modelo que prediga siempre "a tiempo" alcanza ~92% de accuracy y es inútil.
# MAGIC La métrica a optimizar debe ser PR-AUC, F1 o recall sobre la clase positiva,
# MAGIC no accuracy.

# COMMAND ----------

distribucion = (
    df.groupBy("is_late")
    .agg(F.count("*").alias("pedidos"))
    .withColumn(
        "porcentaje", F.round(100 * F.col("pedidos") / F.lit(n_pedidos_entrada), 2)
    )
    .orderBy("is_late")
)
display(distribucion)

tasa_retraso = (
    df.select(F.round(100 * F.avg("is_late"), 2).alias("tasa")).collect()[0]["tasa"]
)
print(f"Tasa de retraso: {tasa_retraso}%")
print(f"Ratio de desbalance ~ 1:{round((100 - tasa_retraso) / max(tasa_retraso, 0.01))}")

assert 2.0 <= tasa_retraso <= 20.0, (
    f"Tasa de retraso de {tasa_retraso}%, fuera del rango esperado para Olist (6-9%). "
    "Probable error en la construcción del target."
)

display(
    df.groupBy("severidad_retraso")
    .agg(
        F.count("*").alias("pedidos"),
        F.round(F.avg("days_delay"), 1).alias("retraso_promedio_dias"),
        F.round(F.avg("distancia_km"), 1).alias("distancia_promedio_km"),
    )
    .orderBy(F.col("pedidos").desc())
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Auditoría de fuga de datos
# MAGIC
# MAGIC **El momento de predicción es la compra.** Cualquier columna cuyo valor solo se
# MAGIC conoce después de ese instante es fuga y no puede entrenar el modelo, por más que
# MAGIC mejore las métricas de validación.
# MAGIC
# MAGIC Esta sección genera el contrato explícito que se entrega a la capa Gold.

# COMMAND ----------

COLUMNAS_CON_FUGA = [
    # Definen el target: usarlas sería predecir con la respuesta.
    "order_delivered_customer_date",
    "fecha_entrega_real",
    "days_delay",
    "dias_entrega_real",
    "severidad_retraso",
    # Ocurren después de la compra.
    "order_delivered_carrier_date",
    "fecha_envio_transportista",
    "dias_hasta_transportista",
    "order_status",
    # El cliente reseña después de recibir el pedido.
    "review_score",
    "fecha_review",
    "tiene_review",
]


# Zona gris: disponibles al aprobarse el pago, no en el checkout.
# El pago se aprueba entre minutos y horas después de la compra, siempre antes del envío.
# Si el modelo se despliega para predecir en el momento del checkout, estas columnas NO
# existen todavía y usarlas es fuga. Si se despliega al confirmarse el pago —que es lo
# operativamente razonable, porque es cuando Olist puede actuar sobre el pedido— son
# legítimas. La decisión es del equipo; lo que no es aceptable es no tomarla.
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

TARGET = ["is_late"]

columnas_disponibles = set(df.columns)
features_seguras = sorted(
    columnas_disponibles
    - set(COLUMNAS_CON_FUGA)
    - set(COLUMNAS_CONDICIONALES)
    - set(COLUMNAS_IDENTIFICADORAS)
    - set(TARGET)
)

print("CONTRATO PARA LA CAPA GOLD")
print("=" * 60)
print("\nTarget: is_late  (auxiliar para regresión: days_delay)")
print(f"\nFEATURES SEGURAS ({len(features_seguras)}) — conocidas al momento de la compra:")
for c in features_seguras:
    print(f"   {c}")
print(f"\nCONDICIONALES ({len(COLUMNAS_CONDICIONALES)}) — solo si se predice tras aprobar el pago:")
for c in COLUMNAS_CONDICIONALES:
    print(f"   {c}")
print(f"\nPROHIBIDAS POR FUGA ({len(COLUMNAS_CON_FUGA)}):")
for c in COLUMNAS_CON_FUGA:
    print(f"   {c}")
print("\nIDENTIFICADORES — excluir del entrenamiento, útiles para trazabilidad:")
for c in COLUMNAS_IDENTIFICADORAS:
    print(f"   {c}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Verificación empírica: ¿las features tienen señal real?
# MAGIC
# MAGIC Si la tasa de retraso no varía entre cuartiles de distancia o de días prometidos,
# MAGIC entonces las variables construidas no aportan nada y hay que revisar la lógica.
# MAGIC Es una comprobación barata de que el trabajo sirve para lo que se hizo.

# COMMAND ----------

df_con_dist = df.filter(F.col("distancia_km").isNotNull())

print("TASA DE RETRASO POR CUARTIL DE DISTANCIA")
display(
    df_con_dist.withColumn(
        "cuartil_distancia", F.ntile(4).over(Window.orderBy("distancia_km"))
    )
    .groupBy("cuartil_distancia")
    .agg(
        F.round(F.min("distancia_km"), 0).alias("desde_km"),
        F.round(F.max("distancia_km"), 0).alias("hasta_km"),
        F.count("*").alias("pedidos"),
        F.round(100 * F.avg("is_late"), 2).alias("tasa_retraso_pct"),
    )
    .orderBy("cuartil_distancia")
)

# COMMAND ----------

print("TASA DE RETRASO POR CUARTIL DE DÍAS PROMETIDOS")
display(
    df.filter(F.col("dias_promesa").isNotNull())
    .withColumn("cuartil_promesa", F.ntile(4).over(Window.orderBy("dias_promesa")))
    .groupBy("cuartil_promesa")
    .agg(
        F.min("dias_promesa").alias("desde_dias"),
        F.max("dias_promesa").alias("hasta_dias"),
        F.count("*").alias("pedidos"),
        F.round(100 * F.avg("is_late"), 2).alias("tasa_retraso_pct"),
    )
    .orderBy("cuartil_promesa")
)

# COMMAND ----------

print("TOP 10 ESTADOS POR TASA DE RETRASO (mínimo 500 pedidos)")
display(
    df.groupBy("cliente_estado")
    .agg(
        F.count("*").alias("pedidos"),
        F.round(100 * F.avg("is_late"), 2).alias("tasa_retraso_pct"),
        F.round(F.avg("distancia_km"), 0).alias("distancia_promedio_km"),
        F.round(F.avg("dias_promesa"), 1).alias("dias_prometidos_promedio"),
    )
    .filter(F.col("pedidos") >= 500)
    .orderBy(F.col("tasa_retraso_pct").desc())
    .limit(10)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Control de calidad final

# COMMAND ----------

n_final = df.count()
n_unicos = df.select("order_id").distinct().count()
n_target_nulo = df.filter(F.col("is_late").isNull()).count()
n_sin_distancia = df.filter(F.col("distancia_km").isNull()).count()
pct_confiable = (
    df.select(F.round(100 * F.avg("distancia_confiable"), 2).alias("p")).collect()[0]["p"]
)

print("CONTROL DE CALIDAD — silver_enriched")
print("=" * 60)
print(f"  Filas                                : {n_final:>9,}")
print(f"  order_id únicos                      : {n_unicos:>9,}")
print(f"  Filas perdidas vs. entrada           : {n_pedidos_entrada - n_final:>9,}")
print(f"  Target nulo                          : {n_target_nulo:>9,}")
print(f"  Sin distancia calculable             : {n_sin_distancia:>9,}"
      f"  ({100.0 * n_sin_distancia / n_final:.2f}%)")
print(f"  Distancia de precisión postal        : {pct_confiable:>8.2f}%")
print(f"  Tasa de retraso                      : {tasa_retraso:>8.2f}%")

assert n_final == n_unicos, "FAN-OUT: la unión con geolocalización duplicó filas."
assert n_final == n_pedidos_entrada, (
    f"Se perdieron {n_pedidos_entrada - n_final:,} filas. Las uniones geográficas "
    "deben ser LEFT y no deben descartar pedidos."
)
assert n_target_nulo == 0, "Hay filas con el target nulo."
assert df.filter(~F.col("is_late").isin(0, 1)).count() == 0, (
    "is_late tiene valores fuera de {0, 1}."
)

print("\nTodas las aserciones pasaron.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Escritura de `silver_enriched`

# COMMAND ----------

PRIMERAS = ["order_id", "is_late", "days_delay", "severidad_retraso"]
COLUMNAS_ORDENADAS = PRIMERAS + [c for c in df.columns if c not in PRIMERAS]

silver_enriched = df.select(*COLUMNAS_ORDENADAS)

(
    silver_enriched.write.format("delta")
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

spark.sql(
    f"""
    COMMENT ON TABLE {T(TABLA_SALIDA)} IS
    'Capa Silver final. Un pedido Olist entregado por fila, con distancia Haversine
     vendedor-cliente y la variable objetivo is_late. Insumo de 03_gold_mlflow_modeling.
     ATENCION: contiene columnas con fuga de datos (days_delay, review_score,
     fechas de entrega). Ver la seccion 6 del notebook 02.2 para el contrato de features.
     Responsable: Esteban.'
    """
)

COMENTARIOS_COLUMNAS = {
    "is_late": "TARGET. 1 si la entrega fue posterior a la fecha estimada (comparacion a nivel de fecha).",
    "days_delay": "FUGA. Dias de diferencia entre entrega real y estimada. Negativo = adelantado.",
    "distancia_km": "Distancia Haversine vendedor-cliente en km. Nula si falta geolocalizacion.",
    "distancia_confiable": "1 si ambas coordenadas vienen de prefijo postal; 0 si alguna es centroide estatal.",
    "geo_cliente_nivel": "Precision de la geolocalizacion del cliente: zip | estado | sin_dato.",
    "geo_vendedor_nivel": "Precision de la geolocalizacion del vendedor: zip | estado | sin_dato.",
    "dias_promesa": "Dias prometidos al cliente en el checkout. Feature segura y de alta senal.",
    "review_score": "FUGA. El cliente resena despues de recibir el pedido.",
    "severidad_retraso": "FUGA. Bucket derivado de days_delay, para analisis de BI.",
}

for columna, comentario in COMENTARIOS_COLUMNAS.items():
    if columna in silver_enriched.columns:
        spark.sql(
            f"ALTER TABLE {T(TABLA_SALIDA)} ALTER COLUMN {columna} COMMENT '{comentario}'"
        )

print("Metadatos documentados en Unity Catalog.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Optimización del layout físico
# MAGIC
# MAGIC `OPTIMIZE` compacta los archivos pequeños que genera Spark al escribir en paralelo.
# MAGIC Va envuelto en `try` porque el comando no está disponible en todas las
# MAGIC configuraciones de cómputo, y no vale la pena que falle el notebook por una mejora
# MAGIC que es puramente de rendimiento.

# COMMAND ----------

try:
    spark.sql(f"OPTIMIZE {T(TABLA_SALIDA)}")
    print("OPTIMIZE ejecutado.")
except Exception as e:
    print(f"OPTIMIZE no disponible en este computo (no es un error del pipeline): {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Vista previa del entregable

# COMMAND ----------

display(
    spark.table(T(TABLA_SALIDA)).select(
        "order_id",
        "is_late",
        "days_delay",
        "dias_promesa",
        "distancia_km",
        "distancia_confiable",
        "cliente_estado",
        "vendedor_estado",
        "mismo_estado",
        "valor_total_pedido",
        "ratio_flete",
        "n_items",
        "categoria_producto",
    ).limit(20)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resumen de la capa Silver
# MAGIC
# MAGIC | Aspecto | Decisión |
# MAGIC |---|---|
# MAGIC | Grano | 1 fila = 1 pedido entregado, verificado con `assert` en ambos notebooks |
# MAGIC | Geolocalización | Mediana por prefijo postal, filtrada al bounding box de Brasil |
# MAGIC | Cobertura faltante | Fallback a centroide estatal, marcado con `geo_*_nivel` |
# MAGIC | Distancia | Haversine con funciones nativas de Spark, validada contra la geometría de Brasil |
# MAGIC | Target | `is_late` comparando a nivel de **fecha**, no de timestamp |
# MAGIC | Fuga de datos | 12 columnas identificadas y documentadas en el catálogo |
# MAGIC | Salida | `big_data_2026.olist.silver_enriched` |
# MAGIC
# MAGIC ### Para Marlon (capa Gold)
# MAGIC
# MAGIC 1. Usar solo la lista de **features seguras** que imprime la sección 6.
# MAGIC 2. El target está desbalanceado (~1:12). Optimizar **PR-AUC o F1**, no accuracy.
# MAGIC 3. Hacer el split **temporal** por `order_purchase_timestamp`, no aleatorio: un split
# MAGIC    aleatorio entrena con pedidos futuros y sobrestima el desempeño real.
# MAGIC 4. `distancia_km` y `peso_producto_g` tienen nulos. Imputar en Gold, con
# MAGIC    `distancia_confiable` disponible como bandera de calidad.
