# Valores esperados — validación de la capa Silver

Números calculados sobre el dataset real de Olist (descarga de Kaggle, versión de 2021).
Sirven para verificar que la ejecución en Databricks es correcta: si tu salida coincide
con esta tabla, el pipeline está bien.

Si algún número difiere de forma significativa, la columna **"Qué significa si no coincide"**
indica dónde buscar el problema.

---

## Notebook 02.1 — celda 1, contrato de entrada

| Tabla | Filas |
|---|---:|
| `bronze_olist_orders_dataset` | 99.441 |
| `bronze_olist_order_items_dataset` | 112.650 |
| `bronze_olist_order_payments_dataset` | 103.886 |
| `bronze_olist_order_reviews_dataset` | 99.224 |
| `bronze_olist_products_dataset` | 32.951 |
| `bronze_olist_sellers_dataset` | 3.095 |
| `bronze_olist_customers_dataset` | 99.441 |
| `bronze_olist_geolocation_dataset` | 1.000.163 |
| `bronze_product_category_name_translation` | 71 |

> **Ojo con `order_reviews`.** El archivo tiene 104.719 líneas de texto pero solo
> **99.224 registros**: los comentarios de los clientes contienen saltos de línea. Si te
> aparece 104.719, el notebook de Bronze perdió la opción `multiline` y las reseñas están
> partidas en varias filas.

---

## Notebook 02.1 — embudo de filtrado

| Paso | Pedidos | Descarte |
|---|---:|---:|
| Pedidos en Bronze | 99.441 | — |
| (1) `status = 'delivered'` | 96.478 | −2.963 |
| (2) con fecha de entrega real | 96.470 | −8 |
| (3) con fecha estimada | 96.470 | −0 |
| (4) coherencia temporal | 96.470 | −0 |

**Retención total: 97,01%**

Los 8 pedidos del paso 2 son un detalle conocido del dataset: figuran como `delivered`
pero no tienen fecha de entrega registrada.

---

## Notebook 02.1 — agregaciones

| Métrica | Valor |
|---|---:|
| Pedidos con ítems | 98.666 |
| Pedidos con pago | 99.440 |
| Reseñas antes de deduplicar | 99.224 |
| Reseñas después de deduplicar | 98.673 |
| **Duplicados eliminados** | **551** |
| **Filas de `silver_orders_joined`** | **96.470** |
| `order_id` únicos | 96.470 |

El `assert` de grano tiene que pasar: 96.470 = 96.470.

---

## Notebook 02.2 — geolocalización

| Métrica | Valor |
|---|---:|
| Filas en Bronze | 1.000.163 |
| Dentro del bounding box | 1.000.121 |
| **Outliers descartados** | **42 (0,004%)** |
| Prefijos postales únicos | 19.010 |

El filtro descarta muy poco, y eso está bien: es una red de seguridad barata. Lo relevante
es que esos 42 puntos están repartidos en unos pocos prefijos, y en esos prefijos sí
habrían desplazado el promedio cientos de kilómetros. Por eso la agregación usa mediana.

**Si el descarte te supera el 5%, revisá que `lat` y `lng` no vengan invertidas.**

### Cobertura geográfica

| Nivel cliente | Nivel vendedor | Pedidos | % |
|---|---|---:|---:|
| `zip` | `zip` | 95.993 | 99,51% |
| `estado` | `zip` | 264 | 0,27% |
| `zip` | `estado` | 212 | 0,22% |
| `estado` | `estado` | 1 | 0,00% |

Ningún pedido queda en `sin_dato`. El fallback estatal se usa en 477 pedidos (0,49%).

---

## Notebook 02.2 — distancia

| Estadístico | km |
|---|---:|
| Mínimo | 0,00 |
| P25 | 187,16 |
| **Mediana** | **435,19** |
| P75 | 800,58 |
| P99 | 2.482,11 |
| **Máximo** | **3.398,49** |

- `assert` de cota dura (6.000 km): **pasa**
- Aviso de cota blanda (4.400 km): **no se dispara**
- Distancia de precisión postal: **99,51%**

---

## Notebook 02.2 — variable objetivo

| Clase | Pedidos | % |
|---|---:|---:|
| A tiempo (0) | 89.936 | 93,23% |
| **Atrasado (1)** | **6.534** | **6,77%** |

**Ratio de desbalance: ~1:14.**

### Por qué importa truncar a fecha

| Método | Tasa de retraso |
|---|---:|
| Comparando **fechas** (correcto) | **6,77%** |
| Comparando **timestamps** (incorrecto) | 8,11% |

La diferencia son **1.288 pedidos mal etiquetados**, es decir 1,34 puntos porcentuales de
falsos positivos: entregas que llegaron el día prometido pero después de medianoche.
Si te da 8,11%, el `to_date()` no se está aplicando.

---

## Notebook 02.2 — verificación de señal

Estas tablas confirman que las features construidas sirven. **Si los cuartiles dan todos
la misma tasa, algo está mal en el cálculo.**

### Por cuartil de distancia

| Cuartil | Rango | Pedidos | Tasa de retraso |
|---|---|---:|---:|
| Q1 | 0–187 km | 24.118 | 4,50% |
| Q2 | 187–435 km | 24.117 | 6,45% |
| Q3 | 435–801 km | 24.117 | 7,12% |
| Q4 | 801–3.398 km | 24.118 | **9,02%** |

Monotónica y creciente: el doble de retraso en el cuartil más lejano frente al más cercano.
La distancia es un predictor real.

### Por cuartil de días prometidos

| Cuartil | Rango | Pedidos | Tasa de retraso |
|---|---|---:|---:|
| Q1 | 3–19 días | 26.000 | 6,28% |
| Q2 | 20–24 días | 25.933 | **8,31%** |
| Q3 | 25–29 días | 21.871 | 7,25% |
| Q4 | 30–156 días | 22.666 | **5,12%** |

Esta no es monotónica, y el patrón en U invertida tiene una lectura de negocio interesante:
cuando Olist promete plazos muy largos (Q4) casi siempre cumple, porque el margen absorbe
cualquier imprevisto. El riesgo se concentra en los plazos intermedios (Q2), donde la
promesa es ajustada pero el envío no es trivial.

---

## QA final de `silver_enriched`

| Control | Valor | Estado |
|---|---:|---|
| Filas | 96.470 | — |
| `order_id` únicos | 96.470 | sin fan-out |
| Filas perdidas contra 02.1 | 0 | correcto |
| Target nulo | 0 | correcto |
| Sin distancia calculable | 0 (0,00%) | cobertura total |

---

## Nota metodológica

Estos valores se obtuvieron reproduciendo la lógica de ambos notebooks en pandas sobre los
CSV originales, no ejecutando Spark. Las agregaciones, filtros, la fórmula de Haversine y
la construcción del target son idénticos, así que los resultados deben coincidir.

Puede haber diferencias mínimas en los percentiles de distancia: Spark usa
`percentile_approx`, que es una aproximación, mientras que el cálculo de referencia usa la
mediana exacta. En un dataset de este tamaño la diferencia es despreciable, pero si ves
variaciones de pocos decimales en P25/P75, es esperado y no indica un error.
