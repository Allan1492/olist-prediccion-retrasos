# CAPA BRONZE - INGESTA DE CSVs A TABLAS DELTA
# Proyecto: Predicción de Retrasos en Entregas - Olist E-commerce
# Responsable: Persona  (Allan)

# Configuracion de rutas
CATALOG = "big_data_2026"
SCHEMA = "olist"
VOLUME_NAME = "raw_csv_files"

# Lista de los 9 archivos CSV (sin la extension .csv)
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
print("Iniciando ingestion a capa Bronze...\n")

# Procesar cada CSV
for archivo in archivos_csv:
    # 1. Leer CSV desde el volumen
    df = spark.read.option("header", "true") \
                   .option("inferSchema", "true") \
                   .option("multiline", "true") \
                   .csv(f"{ruta_volumen}{archivo}.csv")
    
    # 2. Guardar como tabla Delta con prefijo "bronze_"
    nombre_tabla = f"bronze_{archivo}"
    df.write.format("delta") \
      .mode("overwrite") \
      .saveAsTable(f"{CATALOG}.{SCHEMA}.{nombre_tabla}")
    
    print(f"   -> {nombre_tabla} creado con {df.count()} registros")

print(f"\n¡CAPA BRONZE COMPLETADA!")
print(f"9 tablas creadas en: {CATALOG}.{SCHEMA}")
print("\nVerifica en el Catalogo ejecutando:")
print(f"   spark.sql('SELECT * FROM {CATALOG}.{SCHEMA}.bronze_olist_orders_dataset').show()")