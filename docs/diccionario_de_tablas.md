#  Diccionario de Datos (Olist Brazilian E-commerce)

## Tablas Principales
| Tabla | Descripción |
|-------|-------------|
| `olist_orders_dataset` | Pedidos principales con fechas de compra, aprobación, envío y entrega estimada/real. |
| `olist_order_items_dataset` | Detalle de productos por pedido, incluyendo precio y valor del flete (`freight_value`). |
| `olist_order_payments_dataset` | Método de pago, número de cuotas y monto total pagado. |
| `olist_order_reviews_dataset` | Calificación (1-5) y comentarios de los clientes. |
| `olist_products_dataset` | Dimensiones, peso y categoría del producto. |
| `olist_geolocation_dataset` | Coordenadas (lat/lng) basadas en el prefijo del código postal (zip_code_prefix). |

## Variable Objetivo (Target)
- **`is_late`**: 1 si `order_delivered_customer_date` > `order_estimated_delivery_date`, 0 en caso contrario.
- **`days_delay`**: Diferencia en días entre la entrega real y la estimada.
