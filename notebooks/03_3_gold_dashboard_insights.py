# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # CAPA GOLD 3 — Visualizaciones e insights para toma de decisiones
# MAGIC
# MAGIC **Proyecto:** Predicción de Retrasos en Entregas — Olist E-commerce
# MAGIC **Responsable:** Marlon
# MAGIC **Entrada:** las 7 tablas `gold_*` de `03.1_gold_business_kpis`
# MAGIC **Salida:** gráficos inline + `gold_resumen_ejecutivo`
# MAGIC
# MAGIC ```
# MAGIC 03.1_gold_business_kpis
# MAGIC        ▼
# MAGIC 03.2_gold_ml_features
# MAGIC        ▼
# MAGIC 03.3_gold_dashboard_insights   <-- estás aquí
# MAGIC        ▼
# MAGIC 04_gold_mlflow_modeling
# MAGIC ```
# MAGIC
# MAGIC ## Por qué existe este notebook
# MAGIC
# MAGIC `03.1` deja los **indicadores** (tablas agregadas, listas para graficar). Pero una
# MAGIC tabla agregada no es una decisión: "el corredor SP→RR tiene 24% de retraso" es un
# MAGIC dato; "recomendamos renegociar el plazo prometido en ese corredor" es una decisión.
# MAGIC Este notebook agrega esa última capa — la que el objetivo específico 3 del proyecto
# MAGIC pide explícitamente ("generar indicadores **y visualizaciones** para apoyar la toma
# MAGIC de decisiones") — y la deja escrita en una tabla, no solo en el output de una celda
# MAGIC que nadie vuelve a abrir.
# MAGIC
# MAGIC Cada sección sigue el mismo patrón: **gráfico → hallazgo → recomendación**. Las
# MAGIC tablas de origen son pequeñas (ya vienen agregadas), así que se leen con `.toPandas()`
# MAGIC sin costo real, y se grafican con `matplotlib` — no hace falta Spark para dibujar
# MAGIC 20 barras.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Configuración e imports

# COMMAND ----------

from pyspark.sql import functions as F
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

CATALOG = "big_data_2026"
SCHEMA = "olist"

TABLAS_GOLD_NEGOCIO = [
    "gold_kpis_mensuales",
    "gold_ventas_categoria",
    "gold_ventas_geografia",
    "gold_corredores_envio",
    "gold_performance_vendedores",
    "gold_comportamiento_clientes",
    "gold_factores_retraso",
]

TABLA_SALIDA = "gold_resumen_ejecutivo"


def T(nombre: str) -> str:
    return f"{CATALOG}.{SCHEMA}.{nombre}"


plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

hallazgos = []  # (hallazgo, recomendacion, tabla_fuente, prioridad) — se completa en cada sección


def registrar_hallazgo(hallazgo: str, recomendacion: str, tabla_fuente: str, prioridad: str):
    """Guarda un hallazgo con su recomendación para escribirlo en gold_resumen_ejecutivo.

    Centralizado en una función para que el resumen ejecutivo final sea, literalmente,
    la lista de todo lo que se fue concluyendo sección a sección — no un texto aparte
    que alguien puede olvidarse de actualizar.
    """
    hallazgos.append(
        {
            "hallazgo": hallazgo,
            "recomendacion": recomendacion,
            "tabla_fuente": tabla_fuente,
            "prioridad": prioridad,
        }
    )


print(f"Catálogo/esquema: {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Contrato de entrada

# COMMAND ----------

existentes = {
    fila.tableName for fila in spark.sql(f"SHOW TABLES IN {CATALOG}.{SCHEMA}").collect()
}
faltantes = [t for t in TABLAS_GOLD_NEGOCIO if t not in existentes]

assert not faltantes, (
    f"Faltan tablas Gold de negocio: {faltantes}. Ejecuta primero 03.1_gold_business_kpis."
)

print("Contrato de entrada OK — las 7 tablas Gold de negocio están disponibles.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Tendencia mensual: ¿crecemos vendiendo, o crecemos entregando mal?
# MAGIC
# MAGIC Ingresos y tasa de retraso en el mismo gráfico, con dos ejes. Si ambas líneas suben
# MAGIC juntas, el crecimiento está estresando la logística — la señal que un ejecutivo
# MAGIC necesita ver primero.

# COMMAND ----------

pdf_kpis = spark.table(T("gold_kpis_mensuales")).orderBy("anio_compra", "mes_compra").toPandas()

fig, ax1 = plt.subplots(figsize=(11, 4.5))
ax1.plot(pdf_kpis["periodo"], pdf_kpis["ingresos_totales"], color="#1f77b4", marker="o", label="Ingresos")
ax1.set_ylabel("Ingresos totales (R$)", color="#1f77b4")
ax1.tick_params(axis="y", labelcolor="#1f77b4")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
ax1.set_xticks(range(len(pdf_kpis)))
ax1.set_xticklabels(pdf_kpis["periodo"], rotation=60, ha="right", fontsize=7)

ax2 = ax1.twinx()
ax2.plot(pdf_kpis["periodo"], pdf_kpis["tasa_retraso_pct"], color="#d62728", marker="s", label="Tasa de retraso")
ax2.set_ylabel("Tasa de retraso (%)", color="#d62728")
ax2.tick_params(axis="y", labelcolor="#d62728")

plt.title("Ingresos vs. tasa de retraso, mes a mes")
fig.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Hallazgo y recomendación

# COMMAND ----------

pico_retraso = pdf_kpis.loc[pdf_kpis["tasa_retraso_pct"].idxmax()]
pico_ingresos = pdf_kpis.loc[pdf_kpis["ingresos_totales"].idxmax()]

print(
    f"Mes con mayor tasa de retraso: {pico_retraso['periodo']} "
    f"({pico_retraso['tasa_retraso_pct']:.2f}%)"
)
print(
    f"Mes con mayor volumen de ingresos: {pico_ingresos['periodo']} "
    f"(R$ {pico_ingresos['ingresos_totales']:,.0f})"
)

registrar_hallazgo(
    hallazgo=(
        f"El pico de retraso ({pico_retraso['periodo']}, {pico_retraso['tasa_retraso_pct']:.1f}%) "
        f"{'coincide' if pico_retraso['periodo'] == pico_ingresos['periodo'] else 'NO coincide'} "
        f"con el pico de ingresos ({pico_ingresos['periodo']})."
    ),
    recomendacion=(
        "Si coinciden: reforzar capacidad logística (transportistas, personal de despacho) "
        "en los meses de alta demanda, ya conocidos por estacionalidad (Black Friday, Navidad). "
        "Si no coinciden: investigar la causa puntual del mes con más retraso — no es volumen."
    ),
    tabla_fuente="gold_kpis_mensuales",
    prioridad="alta",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Categorías: ingreso vs. riesgo de retraso
# MAGIC
# MAGIC No alcanza con saber qué categoría vende más — hace falta saber si esa categoría
# MAGIC además entrega mal. Un cuadrante "alto ingreso + alto retraso" es la prioridad
# MAGIC número uno para optimizar embalaje, peso o transportista.

# COMMAND ----------

pdf_cat = spark.table(T("gold_ventas_categoria")).orderBy(F.col("ingresos_totales").desc()).limit(15).toPandas()

fig, ax = plt.subplots(figsize=(10, 5))
colores = ["#d62728" if t > pdf_cat["tasa_retraso_pct"].median() else "#1f77b4" for t in pdf_cat["tasa_retraso_pct"]]
ax.barh(pdf_cat["categoria_producto"][::-1], pdf_cat["ingresos_totales"][::-1], color=colores[::-1])
ax.set_xlabel("Ingresos totales (R$)")
ax.set_title("Top 15 categorías por ingreso — rojo = tasa de retraso sobre la mediana")
fig.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Hallazgo y recomendación

# COMMAND ----------

categorias_riesgo = pdf_cat[pdf_cat["tasa_retraso_pct"] > pdf_cat["tasa_retraso_pct"].median()].nlargest(
    3, "ingresos_totales"
)
print("Categorías de alto ingreso con retraso sobre la mediana del top 15:")
print(categorias_riesgo[["categoria_producto", "ingresos_totales", "tasa_retraso_pct"]].to_string(index=False))

registrar_hallazgo(
    hallazgo=(
        "Entre las categorías de mayor ingreso, "
        f"{', '.join(categorias_riesgo['categoria_producto'].tolist())} tienen tasa de "
        "retraso por encima de la mediana del top 15."
    ),
    recomendacion=(
        "Priorizar auditoría logística (empaque, peso declarado vs. real, transportista "
        "asignado) sobre esas categorías primero: es donde una mejora de 1 punto porcentual "
        "de puntualidad tiene más impacto en ingresos protegidos."
    ),
    tabla_fuente="gold_ventas_categoria",
    prioridad="alta",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Geografía: estados y corredores problemáticos
# MAGIC
# MAGIC Dos vistas: el estado del cliente (dónde se siente el problema) y el corredor
# MAGIC vendedor→cliente (dónde se origina). Un estado con retraso alto puede deberse a
# MAGIC uno o dos corredores puntuales, no a toda la región.

# COMMAND ----------

pdf_geo = (
    spark.table(T("gold_ventas_geografia"))
    .filter(F.col("pedidos") >= 200)
    .orderBy(F.col("tasa_retraso_pct").desc())
    .limit(10)
    .toPandas()
)

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar(pdf_geo["cliente_estado"], pdf_geo["tasa_retraso_pct"], color="#d62728")
ax.set_ylabel("Tasa de retraso (%)")
ax.set_title("Top 10 estados de cliente con mayor tasa de retraso (min. 200 pedidos)")
fig.tight_layout()
plt.show()

pdf_corredores = spark.table(T("gold_corredores_envio")).orderBy(F.col("tasa_retraso_pct").desc()).limit(10).toPandas()
pdf_corredores["corredor"] = pdf_corredores["vendedor_estado"] + " → " + pdf_corredores["cliente_estado"]

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.barh(pdf_corredores["corredor"][::-1], pdf_corredores["tasa_retraso_pct"][::-1], color="#ff7f0e")
ax.set_xlabel("Tasa de retraso (%)")
ax.set_title("Top 10 corredores vendedor→cliente con mayor tasa de retraso")
fig.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Hallazgo y recomendación

# COMMAND ----------

peor_estado = pdf_geo.iloc[0]
peor_corredor = pdf_corredores.iloc[0]

print(f"Peor estado (cliente): {peor_estado['cliente_estado']} — {peor_estado['tasa_retraso_pct']:.2f}%")
print(f"Peor corredor: {peor_corredor['corredor']} — {peor_corredor['tasa_retraso_pct']:.2f}%")

registrar_hallazgo(
    hallazgo=(
        f"El estado {peor_estado['cliente_estado']} concentra la mayor tasa de retraso "
        f"({peor_estado['tasa_retraso_pct']:.1f}%) entre estados con volumen relevante, y el "
        f"corredor {peor_corredor['corredor']} es el peor específico "
        f"({peor_corredor['tasa_retraso_pct']:.1f}%)."
    ),
    recomendacion=(
        "Evaluar si el corredor peor ubicado explica la mayor parte del retraso del estado. "
        "Si es así, la corrección es puntual (renegociar SLA con esos vendedores o "
        "ajustar dias_promesa para ese corredor) y no requiere una campaña regional completa."
    ),
    tabla_fuente="gold_ventas_geografia / gold_corredores_envio",
    prioridad="media",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Vendedores: ¿quién genera ingreso y a qué costo de puntualidad?
# MAGIC
# MAGIC Dispersión de ingreso generado vs. tasa de retraso, con el tamaño del punto
# MAGIC proporcional al volumen de pedidos (más pedidos = más confiable la tasa). El
# MAGIC cuadrante "ingreso alto + retraso alto" son los vendedores con más para perder
# MAGIC si no se corrigen — y con más peso si se corrigen.

# COMMAND ----------

pdf_vend = (
    spark.table(T("gold_performance_vendedores"))
    .filter(F.col("muestra_pequena") == 0)
    .orderBy(F.col("ingresos_generados").desc())
    .limit(200)
    .toPandas()
)

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.scatter(
    pdf_vend["tasa_retraso_pct"],
    pdf_vend["ingresos_generados"],
    s=pdf_vend["pedidos"].clip(upper=100),
    alpha=0.5,
    color="#1f77b4",
)
ax.axvline(pdf_vend["tasa_retraso_pct"].median(), color="gray", linestyle="--", linewidth=1)
ax.set_xlabel("Tasa de retraso (%)")
ax.set_ylabel("Ingresos generados (R$)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
ax.set_title("Vendedores: ingreso generado vs. tasa de retraso (línea = mediana)")
fig.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Hallazgo y recomendación

# COMMAND ----------

mediana_retraso_vend = pdf_vend["tasa_retraso_pct"].median()
vendedores_riesgo = pdf_vend[pdf_vend["tasa_retraso_pct"] > mediana_retraso_vend].nlargest(
    5, "ingresos_generados"
)
ingreso_en_riesgo = vendedores_riesgo["ingresos_generados"].sum()

print(f"Vendedores de alto ingreso con retraso sobre la mediana ({mediana_retraso_vend:.1f}%):")
print(vendedores_riesgo[["seller_id_principal", "ingresos_generados", "tasa_retraso_pct"]].to_string(index=False))
print(f"\nIngreso combinado de esos 5 vendedores: R$ {ingreso_en_riesgo:,.0f}")

registrar_hallazgo(
    hallazgo=(
        f"Los 5 vendedores de mayor ingreso con retraso sobre la mediana concentran "
        f"R$ {ingreso_en_riesgo:,.0f} en ventas."
    ),
    recomendacion=(
        "Contactar a esos vendedores puntualmente para revisar su proceso de despacho "
        "(shipping_limit_date vs. cumplimiento real) antes de aplicar cualquier penalización "
        "o campaña de mejora a todo el marketplace."
    ),
    tabla_fuente="gold_performance_vendedores",
    prioridad="alta",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Clientes: ¿retenemos o vendemos una sola vez?
# MAGIC
# MAGIC Define si las decisiones de negocio deben enfocarse en **adquisición** (si casi
# MAGIC nadie vuelve a comprar) o en **retención** (si hay una base recurrente que vale la
# MAGIC pena cuidar con mejor puntualidad).

# COMMAND ----------

pdf_clientes = spark.table(T("gold_comportamiento_clientes")).toPandas()
recurrencia = pdf_clientes["cliente_recurrente"].value_counts(normalize=True) * 100

fig, ax = plt.subplots(figsize=(5, 5))
etiquetas = ["Compra única" if v == 0 else "Recurrente" for v in recurrencia.index]
ax.pie(recurrencia.values, labels=etiquetas, autopct="%1.1f%%", colors=["#1f77b4", "#2ca02c"])
ax.set_title("Clientes: compra única vs. recurrente")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Hallazgo y recomendación

# COMMAND ----------

pct_recurrente = recurrencia.get(1, 0.0)
print(f"Clientes recurrentes: {pct_recurrente:.2f}%")

registrar_hallazgo(
    hallazgo=f"Solo el {pct_recurrente:.1f}% de los clientes únicos volvió a comprar.",
    recomendacion=(
        "Con recurrencia tan baja, invertir en cumplir la promesa de entrega en la primera "
        "compra (dias_promesa realista, no optimista) importa más para la reputación/reseñas "
        "que un programa de fidelización: la mayoría del negocio depende de que la primera "
        "impresión sea buena."
    ),
    tabla_fuente="gold_comportamiento_clientes",
    prioridad="media",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Factores de retraso: comparativo único
# MAGIC
# MAGIC Un solo gráfico con todas las dimensiones no geográficas/de producto analizadas en
# MAGIC `03.1`, para ver de un vistazo cuál pesa más.

# COMMAND ----------

pdf_factores = spark.table(T("gold_factores_retraso")).toPandas()

dimensiones = pdf_factores["dimension"].unique()
fig, axes = plt.subplots(len(dimensiones), 1, figsize=(9, 2.6 * len(dimensiones)))
for ax, dim in zip(axes, dimensiones):
    sub = pdf_factores[pdf_factores["dimension"] == dim].sort_values("tasa_retraso_pct")
    ax.barh(sub["segmento"].astype(str), sub["tasa_retraso_pct"], color="#9467bd")
    ax.set_title(dim, fontsize=9, loc="left")
    ax.tick_params(labelsize=7)
fig.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Hallazgo y recomendación

# COMMAND ----------

fila_max = pdf_factores.loc[pdf_factores["tasa_retraso_pct"].idxmax()]
print(
    f"Mayor tasa de retraso observada: dimensión '{fila_max['dimension']}', "
    f"segmento '{fila_max['segmento']}' ({fila_max['tasa_retraso_pct']:.2f}%)"
)

registrar_hallazgo(
    hallazgo=(
        f"El segmento con peor desempeño de todo el análisis es '{fila_max['segmento']}' "
        f"dentro de la dimensión '{fila_max['dimension']}' ({fila_max['tasa_retraso_pct']:.1f}%)."
    ),
    recomendacion=(
        "Usar esta dimensión como primer filtro de priorización operativa: es la variable "
        "individual con mayor poder discriminante encontrada en el análisis exploratorio, y "
        "debería aparecer entre las features de mayor importancia del modelo en "
        "04_gold_mlflow_modeling."
    ),
    tabla_fuente="gold_factores_retraso",
    prioridad="alta",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Escritura de `gold_resumen_ejecutivo`
# MAGIC
# MAGIC Esta es la tabla que cierra el objetivo específico 3: no un gráfico que se pierde
# MAGIC al cerrar el notebook, sino una tabla Delta con cada hallazgo, su recomendación y
# MAGIC su prioridad, consultable desde cualquier herramienta de BI conectada al catálogo.

# COMMAND ----------

gold_resumen_ejecutivo = spark.createDataFrame(hallazgos).withColumn(
    "orden_prioridad",
    F.when(F.col("prioridad") == "alta", 1)
    .when(F.col("prioridad") == "media", 2)
    .otherwise(3),
).orderBy("orden_prioridad").drop("orden_prioridad")

(
    gold_resumen_ejecutivo.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(T(TABLA_SALIDA))
)

spark.sql(
    f"""
    COMMENT ON TABLE {T(TABLA_SALIDA)} IS
    'Capa Gold. Hallazgos y recomendaciones de negocio derivados de las tablas gold_* de
     03.1, con prioridad asignada. Es el entregable directo del objetivo especifico 3
     del proyecto (indicadores y visualizaciones para apoyar la toma de decisiones).
     Responsable: Marlon.'
    """
)

print(f"Tabla escrita: {T(TABLA_SALIDA)}")
display(spark.table(T(TABLA_SALIDA)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resumen del notebook
# MAGIC
# MAGIC | Sección | Tabla fuente | Pregunta de negocio |
# MAGIC |---|---|---|
# MAGIC | Tendencia mensual | `gold_kpis_mensuales` | ¿Crecer nos está costando puntualidad? |
# MAGIC | Categorías | `gold_ventas_categoria` | ¿Qué categorías de alto ingreso son de alto riesgo? |
# MAGIC | Geografía | `gold_ventas_geografia`, `gold_corredores_envio` | ¿El problema es regional o de rutas puntuales? |
# MAGIC | Vendedores | `gold_performance_vendedores` | ¿Qué vendedores de alto ingreso entregan peor? |
# MAGIC | Clientes | `gold_comportamiento_clientes` | ¿Enfocar esfuerzo en retención o en adquisición? |
# MAGIC | Factores | `gold_factores_retraso` | ¿Cuál es la variable individual más influyente? |
# MAGIC
# MAGIC **Salida consultable:** `big_data_2026.olist.gold_resumen_ejecutivo`
# MAGIC **Siguiente paso:** `04_gold_mlflow_modeling`, usando `gold_ml_features` de `03.2` y
# MAGIC confirmando si las variables señaladas acá como más influyentes también resultan las
# MAGIC de mayor importancia en el modelo entrenado.