# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # CAPA BRONZE - INGESTA DE CSVs A TABLAS DELTA
# MAGIC
# MAGIC Proyecto: Predicción de Retrasos en Entregas - Olist E-commerce
# MAGIC Responsable: Allan
# MAGIC
# MAGIC Este notebook lee los 9 CSVs del volumen y los guarda como tablas Delta
# MAGIC con prefijo "bronze_" en Unity Catalog.

# COMMAND ----------

# Configuración de rutas
CATALOG = "big_data_2026"
SCHEMA = "olist"
VOLUME_NAME = "raw_csv_files"

# Lista de los 9 archivos CSV (sin la extensión .csv)
archivos_csv = [
    "olist_orders_dataset",
    "olist_order_items_dataset",
    "olist_order_payments_dataset",
    "olist_order_reviews_dataset",
    "olist_products_dataset",
    "olist_sellers_dataset",
    "olist_customers_dataset",
    "olist_geolocation_dataset",
    "product_category_name_translation"
]

# Ruta del volumen (para Spark)
ruta_volumen = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}/"
print(f"Ruta del volumen: {ruta_volumen}")
print("Iniciando ingesta a capa Bronze...\n")
print("Iniciando ingestion a capa Bronze...\n")

# COMMAND ----------

# Procesar cada CSV
for archivo in archivos_csv:
    # 1. Leer CSV desde el volumen
    #    escape='"' usa el estándar CSV (comillas dobles) en vez del default de Spark,
    #    que es la barra invertida. Sin esto, las reseñas cuyo texto contiene "\"
    #    rompen el parseo.
    #    multiline=true es CRÍTICO para order_reviews porque los comentarios tienen saltos de línea.
    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .option("multiline", "true")
        .option("escape", '"')
        .csv(f"{ruta_volumen}{archivo}.csv")
    )
    
    # 2. Guardar como tabla Delta con prefijo "bronze_"
    #    overwriteSchema permite reejecutar el notebook aunque cambien los tipos inferidos
    nombre_tabla = f"bronze_{archivo}"
    (
        df.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(f"{CATALOG}.{SCHEMA}.{nombre_tabla}")
    )
    print(f"   -> {nombre_tabla} creado con {df.count():,} registros")

print(f"\n¡CAPA BRONZE COMPLETADA!")
print(f"9 tablas creadas en: {CATALOG}.{SCHEMA}")
print("\nVerificá en el Catálogo ejecutando:")
print(f"   spark.sql('SELECT * FROM {CATALOG}.{SCHEMA}.bronze_olist_orders_dataset').show()")