# Análisis de resultados

Se compararon 6 corridas del experimento `diabetes-cdc-logreg` en MLflow (01-baseline a 06-saga-1-iteracion), variando un hiperparámetro a la vez respecto a la configuración base (semilla 42; umbral elegido en validación para un recall ≥ 0.80). Las métricas de prueba se calculan sobre 37,965 registros (13.9 % positivos).

| Corrida | Cambio respecto a la base | ROC-AUC val | ROC-AUC prueba | PR-AUC | Recall | Precisión |
|---|---|---|---|---|---|---|
| 01-baseline | búsqueda de C (elige 0.001), balanced, lbfgs | **0.8236** | **0.8229** | 0.409 | 0.796 | 0.299 |
| 02-C1e-5-regularizacion-fuerte | C = 0.00001 | 0.8148 | 0.8130 | 0.392 | 0.796 | 0.289 |
| 03-C10-regularizacion-debil | C = 10 | 0.8234 | 0.8228 | 0.409 | 0.794 | 0.299 |
| 04-sin-balanceo | C = 1, class_weight = none | 0.8235 | 0.8228 | 0.410 | 0.795 | 0.298 |
| 05-solver-liblinear | C = 1, solver liblinear | 0.8234 | 0.8228 | 0.409 | 0.794 | 0.299 |
| 06-saga-1-iteracion | C = 1, saga, max_iter = 1 | 0.8110 | 0.8099 | 0.382 | 0.795 | 0.287 |

**Mejor configuración.** El criterio de selección es el ROC-AUC en **validación**, no en prueba, para no sesgar la evaluación final. La mejor es **01-baseline**: regresión logística con `C = 0.001` (elegido por la búsqueda interna), `class_weight="balanced"` y `lbfgs`. Obtuvo un ROC-AUC de 0.8236 en validación y 0.8229 en prueba, con PR-AUC de 0.409 y recall de 0.796. Las corridas 03, 04 y 05 quedan a menos de 0.0002 de ella. Esa diferencia está muy por debajo del error estándar del ROC-AUC en prueba (≈ 0.0036, IC 95 % ± 0.007), así que en la práctica son **equivalentes**. Se conserva la base por ser la más simple y porque su C se eligió con evidencia de validación.

**Hiperparámetro más influyente.** Fue **C (la regularización)**, y solo fuera del rango razonable. Entre C = 0.001 y C = 10 el ROC-AUC prácticamente no cambia (≈ 0.8228–0.8229), pero con C = 0.00001 cae a 0.8130 (−0.010) y la PR-AUC baja de 0.409 a 0.392. Una regularización tan fuerte encoge los coeficientes y el modelo pierde capacidad de ordenar a los pacientes por riesgo. `max_iter` también degradó el modelo (corrida 06, −0.013), pero por falta de convergencia (`ConvergenceWarning`): es una falla de optimización, no una propiedad del modelo. `class_weight` no alteró la discriminación; solo desplazó el umbral óptimo de 0.466 a 0.125, y el criterio de recall mínimo lo compensó. El solver, cuando converge, es irrelevante.

**Limitaciones.**
1. **Diferencias dentro del ruido.** La mayoría de las configuraciones difieren menos que la incertidumbre estadística. Una sola partición no basta para distinguirlas; haría falta validación cruzada o varias semillas.
2. **Techo del modelo.** El ROC-AUC en entrenamiento (0.825) y en prueba (0.823) son casi iguales: no hay sobreajuste, sino sesgo (subajuste). Con ~177 mil registros y 21 variables, la regularización tiene poco efecto. Las mejoras vendrán de otra familia de modelos (p. ej. gradient boosting) o de nuevas variables, no de afinar la regresión logística.
3. **Precisión baja.** Con recall de 0.80, solo ~30 % de las alertas son casos reales, por la prevalencia de 14 %. Que esto sea aceptable depende del costo de un falso positivo en el tamizaje, que no se modeló.
4. **Exploración limitada.** Se varió un factor a la vez y con pocas corridas, sin explorar interacciones. Además, la corrida 06 es artificial (fuerza la no convergencia).
5. **Datos.** Son encuestas autodeclaradas (BRFSS 2015), transversales y con duplicados conservados, lo que limita la generalización.
6. **Una sola semilla.** Las seis corridas usan la semilla 42, de modo que no se midió cuánto varían los resultados con otra partición de los datos.
