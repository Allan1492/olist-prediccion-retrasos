# Predicción de Retrasos en Entregas — Olist E-commerce

Análisis del comercio electrónico brasileño y predicción de retrasos en la entrega
utilizando Apache Spark, Python y Databricks.

## Objetivo general

Analizar el comportamiento de las ventas del comercio electrónico brasileño y
desarrollar un modelo para predecir retrasos en la entrega.

## Cómo usar este README

Si no te queda claro **dónde se responde cada objetivo del proyecto**, andá directo a
la sección [Objetivos → dónde se responden](#objetivos--dónde-se-responden). El resto
del documento es el mapa completo del pipeline por si necesitás más contexto.

---

## Arquitectura (Medallion)

```
00_setup_infraestructura
       │  (crea catálogo/esquema/volumen; subís los 9 CSVs)
       ▼
01_bronze_ingestion
       │  (9 CSVs -> 9 tablas bronze_*, sin modificar)
       ▼
02.1_silver_transformation
       │  (limpieza + agregación a grano de pedido + JOIN maestro)
       ▼  silver_orders_joined
02.2_silver_geolocation_target
       │  (distancia Haversine + variable objetivo is_late + contrato anti-fuga)
       ▼  silver_enriched
       │
       ├──► 03.1_gold_business_kpis     ──► 7 tablas gold_* (BI / indicadores)
       │
       └──► 03.2_gold_ml_features       ──► gold_ml_features (para el modelo)
                                              ▼
                                        04_gold_mlflow_modeling  (próximo paso)
```

| Capa | Qué hace | Responsable |
|---|---|---|
| Bronze | Carga los 9 CSV tal cual, sin modificaciones | Allan |
| Silver | Limpieza, agregación a grano de pedido, geolocalización, target, contrato anti-fuga | Esteban |
| Gold | Tablas analíticas de negocio (KPIs) + tabla de features para el modelo | Marlon |

---

## Tablas por notebook

| Notebook | Tabla(s) de salida | Grano |
|---|---|---|
| `00_setup_infraestructura` | (crea catálogo/esquema/volumen, no tablas) | — |
| `01_bronze_ingestion` | 9× `bronze_*` | 1 fila = 1 registro del CSV original |
| `02.1_silver_transformation` | `silver_orders_joined` | 1 fila = 1 pedido entregado |
| `02.2_silver_geolocation_target` | `silver_enriched` | 1 fila = 1 pedido entregado |
| `03.1_gold_business_kpis` | `gold_kpis_mensuales`, `gold_ventas_categoria`, `gold_ventas_geografia`, `gold_corredores_envio`, `gold_performance_vendedores`, `gold_comportamiento_clientes`, `gold_factores_retraso` | Uno distinto por tabla (mes, categoría, estado, corredor, vendedor, cliente, segmento) |
| `03.2_gold_ml_features` | `gold_ml_features` | 1 fila = 1 pedido entregado |

---

## Objetivos → dónde se responden

### Objetivo específico 1 — Identificar los factores que influyen en los retrasos de entrega

| Pregunta | Dónde se responde |
|---|---|
| ¿Qué variable mueve más la tasa de retraso (distancia, día de semana, fin de semana, mismo estado, multi-vendedor, trimestre, método de pago, días prometidos)? | `gold_factores_retraso` (tabla larga, un solo gráfico comparativo) |
| ¿Qué rutas específicas vendedor→cliente fallan más? | `gold_corredores_envio` |
| ¿Qué categorías de producto son logísticamente más difíciles? | `gold_ventas_categoria` |
| ¿Qué tan fuerte es la señal real de estas variables? | Sección "Verificación empírica" de `02.2` (cuartiles de distancia y de días prometidos) |

### Objetivo específico 2 — Analizar el comportamiento de ventas, clientes, productos y vendedores

| Pregunta | Dónde se responde |
|---|---|
| ¿Cómo evoluciona el negocio (volumen, ingresos, ticket, retraso) mes a mes? | `gold_kpis_mensuales` |
| ¿Qué se vende más y con qué ticket promedio? | `gold_ventas_categoria` |
| ¿Dónde está la demanda por región? | `gold_ventas_geografia` |
| ¿Qué vendedores generan más ingreso y cómo entregan? | `gold_performance_vendedores` |
| ¿Cuánto gasta cada cliente? ¿Compra una vez o es recurrente? | `gold_comportamiento_clientes` |

### Objetivo específico 3 — Generar indicadores y visualizaciones para apoyar la toma de decisiones

Las 7 tablas `gold_*` de `03.1_gold_business_kpis` **son** los indicadores: cada una
ya viene agregada al grano correcto para graficarse directo (Power BI, `display()` en
Databricks, o cualquier herramienta de BI), sin tocar Spark de nuevo.

### Objetivo específico 4 — Construir un modelo predictivo para estimar si un pedido será entregado con retraso

| Pregunta | Dónde se responde |
|---|---|
| ¿Qué columnas puede usar el modelo sin hacer trampa (fuga de datos)? | Sección 6 de `02.2` (contrato anti-fuga) y sección 2 de `03.2` (aplicado en código) |
| ¿Cuál es la tabla lista para entrenar? | `gold_ml_features`, con nulos imputados y `split_temporal` ya resuelto |
| ¿Cómo se entrena y registra el modelo? | `04_gold_mlflow_modeling` (**próximo paso, no construido todavía**) |

---

## Puntos importantes para no perderte

- **`is_late`** es el target: 1 si el pedido llegó después de la fecha estimada
  (comparando a nivel de *fecha*, no de hora). Vive en `silver_enriched` y en
  `gold_ml_features`.
- **Fuga de datos (leakage):** `days_delay`, `review_score`, `order_status` y las
  fechas de entrega real NO pueden usarse para entrenar el modelo — se conocen
  después de que el pedido ya se entregó. `silver_enriched` las conserva para
  análisis de BI (por eso aparecen en las tablas `gold_*` de negocio), pero
  `gold_ml_features` las excluye por diseño.
- **Desbalance de clases:** la tasa de retraso ronda 6–9% (~1:12). El modelo de
  `04_gold_mlflow_modeling` debe optimizar PR-AUC, F1 o recall — no accuracy.
- **Split temporal, no aleatorio:** `gold_ml_features` ya viene particionado en
  `split_temporal` (`train`/`test`) cortado por fecha de compra, para que el modelo
  no "vea el futuro" durante el entrenamiento.

---

## Cómo correr el pipeline desde cero

1. `00_setup_infraestructura` → subir los 9 CSVs de Kaggle al volumen indicado.
2. `01_bronze_ingestion`
3. `02.1_silver_transformation`
4. `02.2_silver_geolocation_target`
5. `03.1_gold_business_kpis` (independiente de `03.2`, se puede correr en cualquier orden entre sí)
6. `03.2_gold_ml_features`
7. `04_gold_mlflow_modeling` (pendiente de construir)
