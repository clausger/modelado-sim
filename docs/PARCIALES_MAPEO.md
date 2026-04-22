# Mapeo de ejercicios de parciales → módulos del script

Documenta los 4 parciales encontrados en `~/Downloads/Modelado examenes /`
y cruza cada ítem con la cobertura actual del script Streamlit.

Última revisión: 2026-04-21.

---

## 1. Inventario de parciales

### Parcial A — `20250430_114503` + `20250430_114511`

1. **f(x) = e^x − 3x², Bolzano en [0,1]**
   - a) Aitken acelerado + condición de Lipschitz en compactos, x₀=0.5, 6 cifras
   - b) Newton-Raphson, x₀=0.5, tol < 10⁻⁸, tabla resumen + análisis comparativo
     (velocidad, precisión, dificultad)
2. **f(x) = Ln(x+1), nodos x₀=0, x₁=0.5, x₂=1**
   - a) Lagrange n−1, error local en ξ=0.45, cota de error global
   - b) f'(1.5) por diferencias finitas centrales desde nodos discretos
3. **I = ∫₀¹ √2·e^x dx**
   - a) Newton-Cotes trapecio n=4, n=10, error de truncamiento en ξ=0.5
   - b) Simpson 1/3 con n=4
4. **I = ∫₀²∫₁³ 2^(x−1)·y dy dx** (integral doble)
   - a) Montecarlo en rectángulo [0,1]×[1,2] con `np.random.seed(0)`, n=10000, **10 repeticiones**, tabla con desviación estándar, varianza, error estándar, media muestral, **IC 95%**
   - b) Demostrar error ∝ 1/√n — duplicar n baja el error a la mitad
5. **EDO dy/dx = y·sin(t), y(0)=1, [0,π]**
   - a) Euler o RK precisión 10⁻¹, h=π/10, **comparar 1ª y 2ª iteración con datos de salida**
   - b) RK4 con k₁, k₂, k₃, k₄ **explícitos por paso**, tol 10⁻⁶

### Parcial B — `20250702_1027582` + `Euler y edo.jpg`

1. a) f(x)=x³−3x−4, Newton, [0,3], 6 cifras significativas
   b) f(x)=cos(x)−x, Aitken, x₀=0.5, [0,1]
2. **I = ∫₀² e^(x²) dx**
   a) Rectángulo **punto medio** n=10 → n=20 si error > 1%
   b) Simpson n=10, comparar convergencia con rectángulo
3. Montecarlo
   a) Aproximar **π** (círculo unidad en [−1,1]²)
   b) Área intersección entre y=x² y y=√x
4. Tabla x=[0,1,3], y=[1,3,0]
   a) Lagrange grado n−1
   b) Diferencias centrales en x=2 del P(x) hallado (error < 1%)
5. EDO
   a) dy/dx = (x³−1)/y², y(1)=3, h=0.1, **Euler mejorado (Heun)**, y(1.5)
   b) dy/dt = 2t√y, y(1)=3, h=0.1, **RK4**, y(1.2)

### Parcial C — `Hoja_1` + `Hoja_2`

Estructura idéntica al Parcial B, con detalles menores en cifras pedidas
(8 decimales en algunos ítems).

### Parcial D — `IMG_5755` + `IMG_5756` (el que estuviste resolviendo)

1. f(x) = x³ − sin(x) − 5, Bolzano en [0,2]
   a) Steffensen-Aitken + Lipschitz, x₀=2, 6 cifras
   b) Newton x₀=2, tol < 10⁻⁸, análisis comparativo
2. f(x) = sin(πx), nodos 0, 0.5, 1, 1.5
   a) Lagrange + error local ξ=0.45 + cota global
   b) Centrales f'(0.75) desde nodos
3. I = ∫₀¹ ln(x+1)/x dx
   a) Trapecio n=4, error ξ=0.5
   b) Simpson n=4 mejora ξ=0.5
4. I = ∫₀¹∫₁³ x·e^y dy dx — Montecarlo rectángulo, `np.random.seed(0)`, n=10000
5. EDO dy/dx = cos(x)+x, y(0)=1, [0, π/2]
   a) Euler precisión 10⁻¹, h=π/8, 1ª y 2ª iter
   b) RK4 con k₁..k₄, tol 10⁻⁶

---

## 2. Cobertura por tema

| Tema | Módulo | Estado | Observaciones |
|---|---|---|---|
| Bolzano | `modules/biseccion.py` (+ `utils/ui/bolzano.py`) | ✅ | Tab dedicada, verificación f(a)·f(b), visual |
| Bisección | `modules/biseccion.py` | ✅ | 4 criterios cátedra, tabla iteraciones |
| Newton-Raphson | `modules/newton_raphson.py` | ✅ | Expander estructura examen formato alumno (2026-04-21), orden empírico de convergencia |
| Punto fijo | `modules/punto_fijo.py` | ✅ | 5 estrategias de g(x): Picard ±, Newton, despeje, **inversión trascendente** (2026-04-21), Lipschitz auto |
| Aitken | `modules/punto_fijo.py` | ✅ | Post-proceso Δ², expander examen |
| Steffensen | `modules/punto_fijo.py` | ✅ | Reinicia desde x*, expander **formato alumno** (2026-04-21) con verificación Lipschitz en semilla |
| Lagrange | `modules/lagrange.py` | ✅ | Nodos simbólicos π (2026-04-21), error local ξ, cota global, maximización simbólica de ∏(x−xᵢ), expander estructura alumno |
| Diferencias finitas — f(x) | `modules/derivacion.py` | ✅ | Progresiva/regresiva/central, tolerancia %, input simbólico π (2026-04-21) |
| Diferencias finitas — tabla t/x | `modules/derivacion.py` | ✅ | Cinemática (velocidad/aceleración) |
| Diferencias finitas — desde nodos Lagrange | `modules/derivacion.py` | ✅ | Central no-uniforme, importa P(x) por session_state |
| Integración — rectángulo | `modules/integracion.py` | ✅ | Punto medio, tabla convergencia |
| Integración — trapecio | `modules/integracion.py` | ✅ | Compuesto |
| Integración — Simpson 1/3 | `modules/integracion.py` | ✅ | Compuesto |
| Integración — Simpson 3/8 | `modules/integracion.py` | ✅ | Compuesto |
| Montecarlo 1D | `modules/montecarlo.py` → `_integracion_1d` | ✅ | Semilla, IC configurable |
| Montecarlo multidim (2D/3D) | `modules/montecarlo.py` → `_integracion_multidimensional` | ✅ | Rectángulo n-dim, IC |
| Montecarlo rechazo 2D | `modules/montecarlo.py` → `_muestreo_rechazo_2d` | ✅ | Para áreas de regiones |
| Montecarlo Vista Cátedra | `modules/montecarlo_catedra.py` | ✅ | Hit-or-miss + promedio + Gauss comparativos |
| EDO Euler | `modules/edo.py` | ✅ | Explícito e implícito |
| EDO Heun (Euler mejorado) | `modules/edo.py` | ✅ | |
| EDO RK4 | `modules/edo.py` | ✅ | |

---

## 3. Gaps detectados — lo que falta hacer a medida

Pensado para que el script resuelva **exactamente** los ejercicios del profe.
Prioridad según frecuencia de aparición en los 4 parciales.

### 🔴 Prioridad alta (aparece en todos los parciales)

1. **Integral doble por Montecarlo + 10 repeticiones + IC 95%**
   Aparece en Parciales A-4 y D-4. El módulo actual (`_integracion_multidimensional`)
   hace **una sola corrida**. El enunciado pide:
   > "repita el experimento 10 veces, promedie los resultados y presente una tabla
   > resumen con desviación estándar, varianza, error estándar, media muestral,
   > intervalos de confianza para 95%"

   **Falta**: bloque de *repeticiones múltiples* en Montecarlo multidim.
   - Input `n_repeticiones` (default 10).
   - Correr n_reps veces con `seed=0+k` o seed=0 para la primera y acumular.
   - Tabla por corrida: seed, media, desvío, varianza, err std, IC low/high.
   - Fila final: promedio de medias, desvío entre corridas, IC 95% sobre las medias.

2. **Demostración error ∝ 1/√n** (Parcial A-4b)
   > "Demuestre que el error se reduce a medida que aumenta la muestra n, en una
   > relación inversa dada por 1/√n. Simule duplicando n y muestre que el error
   > se redujo a la mitad."

   **Falta**: bloque pedagógico que corra Montecarlo con n y 2n, muestre ambos
   errores y el ratio err(2n)/err(n) → ½. Plot log-log de err vs n con pendiente
   teórica −½.

3. **RK4 con k₁, k₂, k₃, k₄ explícitos por iteración**
   Aparece en Parciales A-5b, B-5b, D-5b. El enunciado pide:
   > "estimar cuatro pendientes k₁, k₂, k₃, k₄ en cada extremo del intervalo a
   > analizar y dos en el punto medio. Presente los cálculos en una tabla donde
   > al final de cada iteración se obtenga y."

   **Verificar**: si `modules/edo.py` ya muestra los 4 k's por paso en el paso
   a paso. Si no, agregar columna/bloque por iteración con k₁, k₂, k₃, k₄, y_new.

4. **Estructura de respuesta para examen (formato alumno) en EDO**
   Ya existe en Lagrange, Newton, Steffensen. **Falta** en:
   - Euler / Heun: tabla t_i, y_i, f(t_i, y_i), y_{i+1} con valores sustituidos;
     comparación con analítica si existe; análisis de error vs h.
   - RK4: planteo, f(t,y), k₁..k₄ por paso, conclusión.

### 🟡 Prioridad media

5. **Estructura de respuesta para examen en Integración**
   Aparece en Parciales A-3, B-2, C-2, D-3. Debería incluir:
   - Planteo: f(x), [a,b], n, h = (b−a)/n
   - Nodos x_i y valores f(x_i) tabulados
   - Fórmula del método con pesos
   - Sustitución numérica
   - **Error de truncamiento** con f'' (o f⁽⁴⁾) evaluada en ξ específico (0.5)
   - Resultado y criterio de tolerancia

6. **Montecarlo — intersección de curvas** (Parcial B-3b)
   > "El área que contiene la intersección de las curvas y=x² con y=√x"

   **Verificar**: si `_muestreo_rechazo_2d` permite dos funciones que definan
   la región (y < f₁(x) ∧ y > f₂(x)). Si solo acepta una función, agregar
   preset "área entre dos curvas" con entrada de f_sup(x) y f_inf(x).

7. **Montecarlo π con círculo unidad en [−1,1]²** (Parcial B-3a)
   **Verificar** que Vista Cátedra o rechazo 2D tenga un preset "Aproximar π"
   directo con el círculo x²+y² ≤ 1 en cuadrado [−1,1]².

### 🟢 Prioridad baja (ya cubierto pero mejorable)

8. **Error de truncamiento en ξ específico (ξ=0.5)** en integración 1D.
   El módulo muestra el error teórico genérico; idealmente evaluar f''(0.5) o
   f⁽⁴⁾(0.5) y dar el valor numérico de la cota.

9. **Comparación 1ª/2ª iteración con "datos de salida"** en EDO (Parcial A-5a).
   > "Los cálculos pueden hacerlos en código u otra herramienta informática,
   > pero compruebe que la primera y segunda iteración coincidan con los datos
   > de salida."

   Es una verificación manual del estudiante. Ya alcanza con que la tabla
   muestre y₁, y₂ claros. OK.

10. **Presets de ejercicios** del profe cargables de un click:
    Ya existen en varios módulos ("Ej 1", "Ej 2", ...). Podríamos agregar
    **Presets por parcial** que carguen automáticamente el setup exacto:
    - "Parcial A-1" → f=e^x−3x², x₀=0.5, tol=1e-8 (punto fijo)
    - "Parcial D-2" → nodos sin(πx), ξ=0.45 (Lagrange)
    - etc.

---

## 4. Roadmap sugerido (en orden de impacto)

1. Agregar bloque **10 repeticiones + tabla estadística + IC 95%** a
   `_integracion_multidimensional` y `_integracion_1d` (ambos lo piden).
2. Agregar bloque pedagógico **error ∝ 1/√n** (toggle + comparación n vs 2n).
3. Verificar que `modules/edo.py` tabula k₁..k₄ por iteración en RK4.
   Si falta, agregar tabla detallada.
4. Agregar expander **"📝 Estructura de respuesta para examen"** en:
   - `modules/edo.py` (Euler, Heun, RK4)
   - `modules/integracion.py` (rectángulo, trapecio, Simpson)
5. Agregar preset **"Área entre curvas"** a Montecarlo rechazo 2D.
6. Crear **selector de preset por parcial** (`Parcial A/B/C/D` → ítem 1/2/3/4/5)
   que setee todos los campos del módulo correspondiente.

---

## 5. Notas de contexto

- Los parciales son del profesor Omar J. Cáceres (UADE, Modelado y Simulación).
- El PDF oficial del libro está en `docs/Modelado y Simulacion por Omar J. Cáceres...pdf`.
- El estudiante usa la app para practicar para el parcial del 2026.
- Convención de cátedra: precisión con `round(·, 5)` o 6 cifras según el ítem,
  4 criterios de detención oficiales para iterativos, notación `f^(I)`, `f^(II)`
  para derivadas sucesivas en Lagrange.
