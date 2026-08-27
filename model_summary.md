# Resumen del modelo

## 1. Arquitectura

**Backbone:** ResNet18 preentrenado en ImageNet, sin su capa `fc` final. Se conserva todo el extractor convolucional (incluye el global average pooling nativo de ResNet), produciendo un vector de **512** características por imagen.

**Head de regresión (MLP):**

```
Entrada (512)
  → Linear(512, 256) → ReLU
  → Dropout(0.5)
  → Linear(256, 128) → ReLU
  → Linear(128, 7)      # salida lineal, sin activación
```

- Total de salidas: **7** valores = 3 de posición + 4 de orientación (cuaternión).
- El dropout (0.5) es la única regularización explícita dentro del head; no hay BatchNorm en el head.

---

## 2. Entrada de la red

|  |  |
|---|---|
| Fuente | Imagen RGB de la cámara frontal del dron |
| Resize | 224 × 224 px |
| Normalización | Media/std de ImageNet: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]` |
| Canal de color | RGB (conversión BGR→RGB explícita antes de PIL en el nodo de inferencia) |
| Augmentation (solo entrenamiento) | ColorJitter (brillo/contraste/saturación/matiz), GaussianBlur, ruido gaussiano aditivo, RandomErasing |
| Augmentation (inferencia/validación) | Ninguna — solo resize + normalización (`eval_transform`) |

---

## 3. Salida de la red

Vector de 7 valores: `[x, y, z, q0, q1, q2, q3]`

**Posición (x, y, z):**
- La red predice valores normalizados en el rango **[-1, 1]**.
- Normalización min-max usando los límites físicos del cuarto:
  - x: [-1.25, 8.75] m
  - y: [-8.0, 2.0] m
  - z: [0.1347, 4.3449] m
- Denormalización: `pos = (pos_norm + 1) / 2 * (POS_MAX - POS_MIN) + POS_MIN`

**Orientación (cuaternión, q0, q1, q2, q3):**
- La red produce 4 valores crudos, sin garantía de norma unitaria.
- Se normalizan a norma unitaria **después** de la inferencia: `q = q / ||q||`.
- Orden confirmado: `q0=x, q1=y, q2=z, q3=w`, coincide directamente con `geometry_msgs/Quaternion` de ROS2.

---

## 4. Función de pérdida

```
loss = t_loss + BETA_ROT * q_loss        (BETA_ROT = 10.0)
```

- `t_loss`: MSE entre posición predicha y real (ya normalizadas).
- `q_loss`: MSE entre cuaternión predicho (normalizado) y el ground truth.
- **Manejo de doble cobertura del cuaternión:** antes del MSE se alinea el signo del cuaternión ground-truth con el predicho:
  ```
  dot = q_pred · q_gt
  q_gt_aligned = -q_gt  si dot < 0,  si no  q_gt
  ```
  Esto evita penalizar como "totalmente incorrecta" una predicción geométricamente correcta que cayó en el hemisferio opuesto de la esfera de cuaterniones (q y -q representan la misma rotación).

---

## 5. Algoritmo de entrenamiento

| Componente | Configuración |
|---|---|
| Optimizador | Adam, `lr=1e-4`, `weight_decay=1e-4` |
| Scheduler | `ReduceLROnPlateau` (factor 0.5, patience 3 épocas) |
| Precisión mixta | Sí (`torch.autocast` + `GradScaler`), para GPU de 4GB VRAM |
| Batch size | 8 |
| Épocas máximas | 60 |
| Early stopping | Patience = 20 épocas sin mejora en val_loss |
| Muestreo | `WeightedRandomSampler` — pondera inversamente por frecuencia de `route_type` para balancear tipos de trayectoria (A: Sweep, B: Around obstacles, C: Through obstacles, D: Room perimeter, E: Random) |
| Checkpointing | Se guarda `last` (para reanudar: incluye optimizador, scheduler, scaler, contador de paciencia) y `best` (solo mejor val_loss, usado para evaluación final) por separado |

**Dataset:** Euler (RPY) en el CSV original, convertido a cuaterniones (w, x, y, z) vía `scipy.spatial.transform.Rotation.from_euler('xyz', ...)`, verificado contra la convención de ROS2.

---

## 6. Evaluación

- Se carga el checkpoint **best** (no el último) para evaluación en test.
- **Error de traslación:** norma L2 entre posición predicha y real (metros), en espacio denormalizado.
- **Error de rotación:** ángulo geodésico real entre cuaterniones (grados), no el MSE de entrenamiento:
  ```
  angulo = 2 * arccos(|q1_norm · q2_norm|)
  ```
- Resultados desglosados por `route_type` para detectar fallas sistemáticas en tipos de trayectoria específicos.

**Mejor resultado hasta ahora (Training 6 y 7)**

---

## 7. Consistencia entrenamiento ↔ inferencia (ROS2)

El nodo `nn_pose_estimator.py` replica exactamente:
- Arquitectura `PoseNetLight`.
- `eval_transform` (sin augmentation).
- Constantes de normalización espacial (`X/Y/Z_MIN/MAX`).
- Orden y mapeo del cuaternión a `geometry_msgs/Quaternion`.
- Conversión BGR→RGB antes de pasar la imagen a PIL/torchvision.

---

## 8. Posibles mejoras

**Mejoras a considerar**
1. Reemplazar el MSE de cuaterniones por una pérdida geodésica directa (más proporcional al error angular real reportado en la evaluación).
2. Evaluar aprender `BETA_ROT` (uncertainty weighting) en vez de fijarlo en 10.0.
3. Añadir throttling/desacoplo de la inferencia respecto a la tasa de llegada de imágenes en el nodo ROS2.
