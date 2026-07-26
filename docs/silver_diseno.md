# Capa Silver — Diseño, decisiones y contrato de salida

**Responsable:** Esteban
**Notebooks:** `02.1_silver_transformation.py`, `02.2_silver_geolocation_target.py`
**Salida final:** `big_data_2026.olist.silver_enriched`

---

## 1. Propósito de la capa

Silver convierte nueve tablas Bronze crudas y a distintos granos en **una tabla analítica
confiable, a grano de pedido, con la variable objetivo construida y auditada**.

Todo lo que Gold necesita para modelar sale de aquí. Todo lo que Gold *no debe usar*
también queda identificado aquí.

```
Bronze (9 tablas, granos mixtos)
        │
        ▼
02.1 ── silver_orders_joined ── 1 fila = 1 pedido entregado
        │                       agregaciones + JOINs + variables de calendario
        ▼
02.2 ── silver_enriched ─────── + geolocalización, distancia Haversine,
                                  is_late / days_delay, auditoría de fuga
        │
        ▼
Gold (03_gold_mlflow_modeling)
```

---

## 2. Las cinco decisiones que definen la calidad de esta capa

### 2.1 Agregar antes de unir (evitar el fan-out)

Las tablas de Olist no comparten grano:

| Tabla | Grano | Filas por pedido |
|---|---|---|
| `orders` | pedido | 1 |
| `order_items` | ítem | 1..N |
| `order_payments` | transacción | 1..N |
| `order_reviews` | reseña | 0..N |

Un `JOIN` directo de las cuatro genera un producto cartesiano parcial: un pedido con 3
ítems y 2 pagos produce 6 filas. El dataset se infla, el target se duplica y cualquier
modelo entrenado encima queda sesgado hacia los pedidos grandes.

**Decisión:** cada tabla transaccional se agrega a grano de pedido *antes* del JOIN. El
grano resultante se verifica con `assert` al cierre de cada notebook, así que un fan-out
accidental hace fallar el pipeline en vez de propagarse silenciosamente hasta Gold.

### 2.2 Comparar fechas, no timestamps, al construir el target

`order_estimated_delivery_date` viene con hora `00:00:00`.
`order_delivered_customer_date` trae la hora real de entrega.

Compararlas directo marca como atrasado un pedido entregado a las 15:00 del mismo día
prometido, porque `2018-03-10 15:00:00 > 2018-03-10 00:00:00`. Comercialmente ese pedido
llegó a tiempo: Olist prometió un día, no una hora.

**Decisión:** ambas se truncan con `to_date()` antes de comparar. Sin este paso la tasa de
retraso sale inflada en varios puntos porcentuales y el target no representa la promesa
real del negocio.

### 2.3 Mediana, no promedio, al agregar geolocalización

`olist_geolocation_dataset` tiene ~1M de filas con múltiples coordenadas por prefijo
postal y outliers fuera de Brasil (signos invertidos, ceros, errores de digitación).

Un solo punto con longitud positiva —imposible en Brasil— arrastra el **promedio** del
prefijo cientos de kilómetros. La **mediana** tiene punto de ruptura del 50%: haría falta
que la mitad de los registros estuvieran mal para moverla.

**Decisión:** filtro de bounding box continental *antes* de agregar, y mediana
(`percentile_approx(col, 0.5)`) como estadístico de agregación.

| Límite | Valor |
|---|---|
| Latitud | −33.75 a 5.27 |
| Longitud | −73.99 a −34.79 |

### 2.4 Haversine con funciones nativas, no con UDF de Python

Escribir la fórmula como `@udf` con el módulo `math` falla en tres frentes:

- **Rendimiento.** Cada fila se serializa entre la JVM y el intérprete de Python.
- **Optimización.** Catalyst trata las UDF como cajas negras y no puede optimizar dentro.
- **Compatibilidad.** El cómputo serverless de Databricks Free Edition restringe las UDFs
  de Python; las funciones nativas siempre están disponibles.

La implementación usa `F.radians`, `F.sin`, `F.cos`, `F.asin`, `F.sqrt` y `F.pow`, con un
`F.least(a, 1.0)` que evita que el error de punto flotante produzca `NaN` cuando origen y
destino coinciden.

**Validación con dos umbrales.** La distinción importa y es fácil equivocarse:

| Umbral | Valor | Origen | Acción |
|---|---|---|---|
| Cota dura | 6.000 km | Diagonal del bounding box (5.982 km) | `assert` — falla el notebook |
| Cota blanda | 4.400 km | Extensión real de Brasil (Roraima → Chuí, 4.399 km) | Aviso impreso |

Superar la cota dura es geométricamente imposible con puntos que pasaron el filtro, así
que solo puede deberse a un error de código. Superar la blanda es posible —el bounding box
es un rectángulo y Brasil no— pero merece revisión. Poner el `assert` en la cota blanda
haría reventar el pipeline con coordenadas perfectamente válidas.

### 2.5 Fallback geográfico explícito y marcado

No todos los prefijos postales de los pedidos existen en la tabla de geolocalización.
Descartar esos pedidos pierde datos válidos; dejarlos nulos rompe el cálculo de distancia.

**Decisión:** cascada de precisión decreciente, con la fuente registrada en una bandera.

| Nivel | Fuente | Precisión aproximada | Valor de `geo_*_nivel` |
|---|---|---|---|
| 1 | Mediana del prefijo postal | 1–5 km | `zip` |
| 2 | Centroide del estado | 100–300 km | `estado` |
| 3 | Sin dato disponible | — | `sin_dato` |

La columna `distancia_confiable` vale 1 solo cuando **ambos** extremos vienen de prefijo
postal. Gold puede entrenar únicamente con esas filas si la imputación estatal resulta
demasiado ruidosa. La decisión queda documentada en los datos, no escondida en el código.

---

## 3. Filtrado de pedidos

Solo entran a Silver los pedidos que pueden etiquetarse honestamente:

1. `order_status = 'delivered'` — los estados `shipped`, `canceled` y `unavailable` no
   tienen fecha de entrega real, así que no pueden etiquetarse sin inventar el target.
2. `order_delivered_customer_date` no nula.
3. `order_estimated_delivery_date` no nula.
4. Coherencia temporal: la entrega no puede ser anterior a la compra.

El notebook `02.1` imprime el embudo completo con el conteo de descarte en cada paso.
La retención esperada ronda el 96–97% de los pedidos de Bronze.

---

## 4. Contrato de features para la capa Gold

**El momento de predicción es la compra.** Cualquier columna cuyo valor solo se conoce
después de ese instante es fuga de datos, sin importar cuánto mejore las métricas de
validación.

### Prohibidas — fuga de datos

| Columna | Motivo |
|---|---|
| `order_delivered_customer_date`, `fecha_entrega_real` | Definen el target |
| `days_delay`, `dias_entrega_real`, `severidad_retraso` | Derivadas del target |
| `order_delivered_carrier_date`, `fecha_envio_transportista`, `dias_hasta_transportista` | Posteriores a la compra |
| `order_status` | Constante `delivered` tras el filtrado; no aporta y en general es posterior |
| `review_score`, `fecha_review`, `tiene_review` | El cliente reseña después de recibir |

`review_score` es la trampa más peligrosa del dataset: los pedidos atrasados reciben
reseñas de 1 estrella, así que la métrica se dispara en validación y colapsa en producción.
Se conserva en Silver porque BI la necesita, pero está marcada como fuga en el comentario
de columna de Unity Catalog.

### Condicionales — depende de dónde se ponga el momento de predicción

`order_approved_at` y `horas_hasta_aprobacion`.

El pago se aprueba entre minutos y horas después de la compra, siempre antes del envío.
Si el modelo predice en el **checkout**, esas columnas todavía no existen y usarlas es
fuga. Si predice al **confirmarse el pago** —que es lo operativamente razonable, porque es
cuando Olist puede actuar sobre el pedido— son legítimas y aportan señal.

La decisión es del equipo. Lo que no es aceptable es dejarlas en el dataset sin haberla
tomado, que es como se cuelan la mayoría de las fugas reales.

### Identificadores — excluir del entrenamiento

`order_id`, `customer_id`, `customer_unique_id`, `seller_id_principal`,
`product_id_principal`. Sirven para trazabilidad, no como predictores.

### Features seguras — conocidas en el checkout

Agrupadas por familia:

- **Promesa:** `dias_promesa`, `shipping_limit_date`
- **Geografía:** `distancia_km`, `distancia_confiable`, `cliente_estado`, `vendedor_estado`,
  `mismo_estado`, `cliente_zip_prefix`, `vendedor_zip_prefix`, coordenadas,
  `geo_cliente_nivel`, `geo_vendedor_nivel`
- **Composición del pedido:** `n_items`, `n_productos_distintos`, `n_vendedores`,
  `multi_vendedor`
- **Económicas:** `valor_productos`, `valor_flete`, `valor_total_pedido`, `ratio_flete`,
  `precio_promedio_item`, `precio_max_item`, `valor_pagado`, `max_cuotas`,
  `tipo_pago_principal`, `n_tipos_pago`
- **Producto:** `categoria_producto`, `peso_producto_g`, `volumen_producto_cm3`,
  `n_fotos_producto`
- **Calendario:** `anio_compra`, `mes_compra`, `trimestre_compra`, `dia_semana_compra`,
  `hora_compra`, `es_fin_de_semana`

La sección 6 del notebook `02.2` imprime la lista exacta y actualizada en tiempo de
ejecución, así que no hay riesgo de que esta documentación se desactualice respecto al
código.

---

## 5. Advertencias para el modelado

1. **Clases desbalanceadas.** La tasa de retraso ronda el 6–9%, o sea ~1:12. Un modelo que
   prediga siempre "a tiempo" alcanza ~92% de accuracy y es completamente inútil. Optimizar
   **PR-AUC, F1 o recall** sobre la clase positiva.
2. **Split temporal, no aleatorio.** Un `randomSplit` entrena con pedidos futuros y prueba
   con pasados, lo que sobrestima el desempeño real. Dividir por
   `order_purchase_timestamp`.
3. **Nulos remanentes.** `distancia_km`, `peso_producto_g` y `volumen_producto_cm3` tienen
   nulos legítimos del dataset original. Silver los reporta; Gold decide la imputación.
   Se recomienda añadir banderas de "valor imputado" en vez de rellenar en silencio.
4. **Cardinalidad de `categoria_producto`.** ~74 categorías. Conviene agrupar la cola larga
   antes de aplicar One-Hot Encoding.

---

## 6. Controles de calidad implementados

Ambos notebooks fallan ruidosamente en vez de escribir tablas corruptas.

| Control | Notebook | Qué previene |
|---|---|---|
| Existencia de las 9 tablas Bronze | 02.1 | Errores crípticos por dependencias faltantes |
| `filas == order_id distintos` | 02.1, 02.2 | Fan-out en los JOINs |
| `dias_promesa >= 0` | 02.1 | Fechas incoherentes |
| `0 <= distancia <= 6000 km` | 02.2 | Error en la fórmula de Haversine o en el filtro |
| `2% <= tasa de retraso <= 20%` | 02.2 | Target mal construido |
| `is_late` sin nulos y solo en {0,1} | 02.2 | Target inválido |
| Filas de salida == filas de entrada | 02.2 | Un JOIN geográfico que descarte pedidos |

Además se documentan en Unity Catalog el comentario de tabla y los comentarios de las
columnas críticas, de modo que la advertencia de fuga sea visible desde el Catalog
Explorer sin necesidad de leer el código.

---

## 7. Cómo ejecutar

1. Ejecutar `01_bronze_ingestion` (responsable: Allan).
2. Ejecutar `02.1_silver_transformation` → crea `silver_orders_joined`.
3. Ejecutar `02.2_silver_geolocation_target` → crea `silver_enriched`.

Ambos notebooks son **idempotentes**: se pueden reejecutar sin borrar tablas a mano
(`mode("overwrite")` + `overwriteSchema`), y las funciones de ventana llevan desempates
explícitos para que dos ejecuciones sobre los mismos datos produzcan resultados idénticos.
