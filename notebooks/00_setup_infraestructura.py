# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 00 — Setup de infraestructura
# MAGIC
# MAGIC Crea el catálogo, el esquema y el volumen que necesita `01_bronze_ingestion`.
# MAGIC
# MAGIC **Ejecutar una sola vez, antes que cualquier otro notebook.**
# MAGIC
# MAGIC ```
# MAGIC 00_setup_infraestructura   <-- estás aquí
# MAGIC        │
# MAGIC        ▼  (subir los 9 CSVs al volumen)
# MAGIC 01_bronze_ingestion
# MAGIC        ▼
# MAGIC 02.1_silver_transformation
# MAGIC        ▼
# MAGIC 02.2_silver_geolocation_target
# MAGIC ```

# COMMAND ----------

CATALOG = "big_data_2026"
SCHEMA = "olist"
VOLUME = "raw_csv_files"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Crear el catálogo
# MAGIC
# MAGIC Algunas cuentas de Databricks no permiten crear catálogos nuevos. Si falla, la celda
# MAGIC no revienta: cae al catálogo `workspace`, que existe siempre, y te avisa que hay que
# MAGIC cambiar la constante `CATALOG` en los demás notebooks.

# COMMAND ----------

try:
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
    print(f"Catálogo '{CATALOG}' disponible.")
except Exception as e:
    print(f"No se pudo crear el catálogo '{CATALOG}':\n   {e}\n")
    CATALOG = "workspace"
    print(f"Se usará el catálogo por defecto: '{CATALOG}'")
    print("\n>>> IMPORTANTE: cambiá CATALOG = \"workspace\" en los notebooks")
    print("    01_bronze_ingestion, 02.1_silver_transformation y 02.2_silver_geolocation_target.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Crear el esquema y el volumen
# MAGIC
# MAGIC El **volumen** es donde viven los archivos crudos. Es el equivalente en Unity Catalog
# MAGIC a una carpeta: guarda los CSVs tal como los subís, sin convertirlos en tablas.

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
print(f"Esquema '{CATALOG}.{SCHEMA}' listo.")

spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")
print(f"Volumen '{CATALOG}.{SCHEMA}.{VOLUME}' listo.")

RUTA_VOLUMEN = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
print(f"\nRuta para subir los CSVs:\n   {RUTA_VOLUMEN}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Ahora subí los 9 CSVs
# MAGIC
# MAGIC Descarga: **Brazilian E-Commerce Public Dataset by Olist** en Kaggle
# MAGIC → `https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce`
# MAGIC
# MAGIC En Databricks: menú izquierdo → **Catalog** → navegá hasta el volumen
# MAGIC `big_data_2026 > olist > raw_csv_files` → botón **Upload to this volume**
# MAGIC → arrastrá los 9 archivos.
# MAGIC
# MAGIC Los nombres tienen que quedar **exactamente** así (el notebook de Bronze los busca
# MAGIC por nombre; si alguno viene con `(1)` o renombrado, falla):
# MAGIC
# MAGIC ```
# MAGIC olist_orders_dataset.csv
# MAGIC olist_order_items_dataset.csv
# MAGIC olist_order_payments_dataset.csv
# MAGIC olist_order_reviews_dataset.csv
# MAGIC olist_products_dataset.csv
# MAGIC olist_sellers_dataset.csv
# MAGIC olist_customers_dataset.csv
# MAGIC olist_geolocation_dataset.csv
# MAGIC product_category_name_translation.csv
# MAGIC ```
# MAGIC
# MAGIC El zip de Kaggle trae exactamente esos 9 archivos, así que basta con descomprimirlo
# MAGIC y subir todo el contenido.
# MAGIC
# MAGIC **Cuando termines de subirlos, ejecutá la celda de abajo para verificar.**

# COMMAND ----------

ARCHIVOS_ESPERADOS = [
    "olist_orders_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "olist_customers_dataset.csv",
    "olist_geolocation_dataset.csv",
    "product_category_name_translation.csv",
]

try:
    presentes = {f.name for f in dbutils.fs.ls(RUTA_VOLUMEN)}
except Exception as e:
    presentes = set()
    print(f"No se pudo listar el volumen: {e}")

faltantes = [a for a in ARCHIVOS_ESPERADOS if a not in presentes]
sobrantes = sorted(presentes - set(ARCHIVOS_ESPERADOS))

print("VERIFICACIÓN DE ARCHIVOS")
print("=" * 55)
for archivo in ARCHIVOS_ESPERADOS:
    print(f"  {'OK   ' if archivo in presentes else 'FALTA'}  {archivo}")

if sobrantes:
    print(f"\nArchivos extra en el volumen (se ignoran): {sobrantes}")

if faltantes:
    print(f"\nFaltan {len(faltantes)} archivos. Subilos y volvé a ejecutar esta celda.")
else:
    print("\nLos 9 archivos están. Ya podés ejecutar 01_bronze_ingestion.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Prueba de lectura
# MAGIC
# MAGIC Antes de correr el Bronze completo, confirmamos que Spark puede leer un CSV del
# MAGIC volumen. Es una prueba de 5 segundos que evita descubrir un problema de permisos
# MAGIC después de esperar la ingesta de las 9 tablas.

# COMMAND ----------

if not faltantes:
    prueba = (
        spark.read.option("header", "true")
        .option("inferSchema", "true")
        .csv(f"{RUTA_VOLUMEN}/olist_orders_dataset.csv")
    )
    print(f"Lectura correcta: {prueba.count():,} filas, {len(prueba.columns)} columnas.")
    print("Esperado: 99.441 filas, 8 columnas.\n")
    display(prueba.limit(5))
else:
    print("Subí primero los archivos que faltan.")