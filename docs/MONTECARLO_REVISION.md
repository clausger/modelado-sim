# Revisión Montecarlo — nuestro script vs repo del amigo

> Comparativa entre `/Users/germanmieth/Desktop/modelado-sim/modules/montecarlo.py` (Python/Streamlit,
> sigue convención Cáceres) y `/Users/germanmieth/Desktop/Modelado/src/methods/integration/*.ts`
> (TypeScript/web, no sigue cátedra pero tiene buenas ideas pedagógicas).
>
> **Objetivo**: dejar nuestro Montecarlo con calidad de parcial sin prisa.

---

## 1. Features comparados

| Feature | Nosotros | Amigo | Diagnóstico |
|---|---|---|---|
| Teoría cátedra (Cáceres pag. 36-42) | ✅ expander completo | ✗ | Mantenemos ventaja |
| Convención SE = σ/√N (sin V(D)) | ✅ sigue cátedra | ✗ usa (b-a)σ/√N textbook | **No portar**, nuestra convención es la oficial |
| Input simbólico (π, √2, e) en límites | ✅ | ✗ parseFloat | Mantenemos ventaja |
| Semilla Python stdlib (reproducible con profe) | ✅ `random.seed` | ✗ Mulberry32 TS | Mantenemos, los ejemplos del profe dan igual |
| Valores intermedios del cálculo | ✅ tabla detallada | ✗ | Mantenemos ventaja |
| Selector de confianza flexible | ✅ 90/95/99/99.7 + custom | ✗ fijo 95% | Mantenemos ventaja |
| Repeticiones K + IC empírico (1D) | ✅ recién agregado | ✗ | Empate — nuestro es más flexible |
| Repeticiones K + IC empírico (multidim) | ✅ recién agregado | ✅ de serie | Empate |
| **Valor exacto opcional como input en multidim** | ✗ (sólo intenta sympy) | ✅ campo manual | 🟡 agregar fallback manual |
| **Gráfico convergencia con banda IC 95%** | ~ tenemos convergencia | ✅ banda sombreada | 🟡 mejorar |
| **Gráfico \|error real\| vs SE vs 1/√N teórico (log)** | ✗ | ✅ en π y base 1D | 🔴 esto *es* el bloque error∝1/√n del roadmap |
| **Preset Monte Carlo π dedicado** | ~ en rechazo 2D hay preset círculo | ✅ módulo aparte con todo cableado | 🟡 ya cubierto, pero nos falta tabla Bernoulli explícita |
| **Preset Área entre curvas** | ✅ en rechazo 2D | ✅ módulo aparte | Empate (nuestro está dentro de rechazo 2D, funcional) |
| Tabla de lotes incrementales (N/20 batches) | ~ tenemos paso-a-paso | ✅ nativo | 🟡 agregar vista de lotes |
| Referencias explícitas a fecha de parcial | ✗ | ✅ menciona "parcial 30/04/2025", etc | 🟡 agregar en pasos |
| Manejo \|f-g\| cuando curvas se cruzan | ✅ | ✅ | Empate |

---

## 2. Qué tiene el amigo que vale la pena portar

### 🔴 Alta prioridad (directo a cátedra)

**A. Bloque error ∝ 1/√N (gráfico log)**
- Ya está en nuestro roadmap. El amigo lo resuelve con un gráfico que pone en escala log:
  - `|error real|` (si hay valor exacto),
  - `SE` del estimador,
  - curva teórica `c/√N` de referencia.
- Ubicación natural: dentro del tab Visualizaciones de 1D/multidim y del preset π.
- Extra pedagógico: comparar N vs 2N y verificar `SE(2N) ≈ SE(N)/√2`.

**B. Tabla Bernoulli explícita en preset π**
- El amigo lista por lote: `N, dentro, p̂, π̂, Var=p̂(1-p̂), σ=√Var, SE=4·σ/√N, |π̂-π|, IC low/up`.
- Nosotros tenemos el cálculo pero no la tabla con *todas* las columnas. Es literalmente la tabla que pide la Prueba Evaluativa.

### 🟡 Media prioridad (calidad de vida)

**C. Campo "valor exacto" opcional en multidim y rechazo 2D**
- Cuando `sp.integrate` no puede o es muy lento, el alumno suele tenerlo a mano desde la consigna. Agregar un `st.text_input` que acepte expresión simbólica (ya tenemos `parse_expr_to_float`).
- Importante además en rechazo 2D para que el profe ponga "A = 1/12" y la app lo use directo.

**D. Referencias explícitas a parciales en los pasos**
- El amigo pone "parcial 30/04/2025: f=x², g=x³, [0,1], A=1/12". Eso ayuda al alumno a identificar qué preset usar. Podemos meterlo en los captions bajo los headers de cada tab.

**E. Vista de lotes incrementales (N/20)**
- Nuestro "paso a paso" muestra valores intermedios a distintos N, pero no como una tabla de 20 filas. El amigo sí → la banda IC sombreada en convergencia queda perfecta.

### 🟢 Baja prioridad

- Mulberry32: no lo queremos (rompe reproducibilidad con `random.uniform` del profe).
- Mensajes de `toPrecision(8)` concatenados: nuestros metrics con columnas son mejores.

---

## 3. Qué hacemos mejor y no tocamos

1. **Convención Cáceres SE = σ/√N** (sin V(D)). El amigo usa `(b-a)·σ/√N`. Si mezclamos, rompemos la coincidencia con los resultados del profe. ← dejar como está y *explicar* en teoría la diferencia.
2. **Input simbólico π, √2** en límites, nodos, valor exacto.
3. **Teoría del libro** con referencias a página (pags. 36-42).
4. **Selector de confianza** flexible (90/95/99/99.7/custom) vs fijo 95%.
5. **Valores intermedios** detallados: N, V(D), Σf, f̄, σ, min, max, z, EE, margen — el amigo no muestra esto.
6. **Repeticiones K reutilizable** (`_render_repeticiones_examen`) que enchufamos en 1D y multidim.

---

## 4. Plan propuesto (en orden, sin apurar)

1. **Bloque `error ∝ 1/√N`** (🔴) → nuevo helper `_render_convergencia_sqrt_n(...)` invocado desde 1D, multidim, π y área entre curvas.
   - Dos modalidades: (a) una corrida creciente con snapshots en N, 2N, 4N, 8N, 16N y comparar SE y \|error real\|; (b) curva teórica `SE(N)/√(n/N)` superpuesta.
2. **Tabla Bernoulli en preset π** (🔴) → expander dentro del bloque círculo en `_muestreo_rechazo_2d`.
   - Columnas exactas: `lote | N | dentro | p̂ | π̂ | p̂(1-p̂) | σ | SE | |π̂-π| | IC_low | IC_up`.
3. **Campo "valor exacto" manual** (🟡) → en multidim y en rechazo 2D (ambos modos). Usar `parse_expr_to_float`. Si el usuario lo deja vacío, sigue intentando sympy.
4. **Lotes incrementales en convergencia 1D/multidim** (🟡) → helper `_render_convergencia_lotes(...)` con tabla y banda IC 95% sombreada en el gráfico.
5. **Referencias a parciales** (🟡) → captions bajo los tabs de cada submódulo citando la consigna real de cátedra (con fecha/tema).
6. **Repeticiones K también en rechazo 2D** (🟡) → el amigo tiene K nativo en área entre curvas y aporta IC empírico que es lo que pide el parcial D-4 en 2D. Adaptar `_render_repeticiones_examen` a pasar `volumen/area_rect` + `sampler_hits`.

---

## 5. Decisiones (German confirmó: "cuanto más personalizable mejor")

1. **Bloque error ∝ 1/√N** → expander dentro de tab Visualizaciones con switches: mostrar curva teórica, elegir serie de N (N, 2N, 4N, 8N…), y un botón "reducir error a la mitad" que corre con 4N (exigido por parcial B-4b).
2. **Valor exacto manual** → aceptar **expresiones simbólicas** (`pi**2/6`, `(exp(2)-1)**2/exp(2)`, `1/3`). Usamos `parse_expr_to_float`.
3. **Preset π** → mantenerlo dentro de rechazo 2D + agregar **preset-button** "Prueba Evaluativa 3a" que precargue `lado=2, r=1, K=10, N=10000, seed=0` y active la **tabla Bernoulli completa** (columnas `N | dentro | p̂ | π̂ | p̂(1-p̂) | σ | SE | \|π̂-π\| | IC_low | IC_up`).
4. **Citas de parcial** → ver sección 6.

---

## 6. Consignas reales de parcial (provistas por German)

### Parcial A-4 (integral doble)
> Dada la Integral **I = ∫₀² ∫₀² e^(x−y) dy dx**. Modele una solución Montecarlo para
> aproximar el volumen sobre el cuadrado [0,2]×[0,2] considere: fijar semilla con
> **`np.random.seed(0)`** para reproducibilidad, población de muestra **n = 10000**,
> resolver analíticamente la integral (mostrar pasos) y presentar tabla resumen con
> **desviación estándar, varianza, error estándar, media muestral, IC 95%**.

- Valor exacto: `(e² − 1) · (1 − e⁻²) = (e² − 1)² / e² ≈ 4.0384`
- Preset sugerido: `f(x,y) = exp(x-y)`, x∈[0,2], y∈[0,2], N=10000, seed=0, **backend numpy**.

### Parcial (variante rectangular)
> Dada **I = ∫₀¹ ∫₁³ x·e^y dy dx** sobre el rectángulo [0,1]×[1,3]. Mismas condiciones
> (`np.random.seed(0)`, n=10000, resolver analíticamente, tabla resumen con σ, s², EE,
> media, IC 95%).

- Valor exacto: `(1/2) · (e³ − e) ≈ 8.6836`
- Preset sugerido: `f(x,y) = x·exp(y)`, x∈[0,1], y∈[1,3], N=10000, seed=0.

### Parcial B-4b (demostración 1/√n)
> Demuestre que el error se reduce como **1/√n**. Haga la demostración matemática para
> el caso que se desee reducir el error **a la mitad** con una nueva muestra. Simule la
> estimación con la nueva muestra demostrando que efectivamente el error se redujo a la
> mitad.

- Teoría: SE(n) = σ/√n  ⇒  SE(n')/SE(n) = √(n/n') = 1/2 requiere **n' = 4n**.
- Bloque UI: tomar el N actual, correr con 4N, mostrar tabla comparativa SE(N), SE(4N),
  ratio SE(4N)/SE(N) (debe dar ≈ 0.5) y gráfico log con curva teórica `c/√N`.

### Prueba Evaluativa — 3
> Use Montecarlo con generador uniforme (al menos **10000**), el experimento se repite
> **10 veces**, promedie los resultados y preséntelos en una tabla.
>
> **a)** Aproxime π con círculo de radio 1 centrado en el origen, inscrito en el cuadrado
> [−1,1]×[−1,1].
>
> **b)** Área que contiene la intersección de las curvas **y = x²** con **y = √x**.

- Preset 3a: rechazo 2D, lado=2, r=1, N=10000, K=10, seed=0, tabla Bernoulli.
- Preset 3b: rechazo 2D curvas, f(x)=√x (superior en [0,1]), g(x)=x² (inferior), [a,b]=[0,1],
  N=10000, K=10, seed=0. **Exacto: A = ∫₀¹(√x − x²)dx = 2/3 − 1/3 = 1/3 ≈ 0.3333.**

---

## 7. ⚠ Hallazgo crítico: `np.random.seed(0)` vs `random.seed(42)`

**Nuestro código usa Python stdlib `random.uniform`. El profe en los parciales pide
`np.random.seed(0)` + `np.random.uniform`.** Los streams **no coinciden**.

Si un alumno reproduce un parcial con nuestra app usando `seed=0`, los resultados
numéricos no van a dar igual que los que calcula el profe con numpy.

**Propuesta**: agregar **selector de backend RNG** en cada submódulo:
- `stdlib random` (default actual, lo usamos para coincidir con ejemplos del libro de
  Cáceres que también usan stdlib).
- `numpy random` (para coincidir con los parciales `np.random.seed(...)`).

Internamente, `_sample_uniform`, `_sample_uniform_2d` y `_seed_rng` aceptan un parámetro
`backend: Literal["stdlib","numpy"]` y routean a `np.random.default_rng(seed)` o
`np.random.seed/uniform` según corresponda.

Default recomendado: **numpy**, porque los parciales lo piden explícitamente. Dejamos
stdlib como opcional para los alumnos que sigan el libro al pie de la letra.

---

## 8. Plan de implementación afinado (orden propuesto)

1. 🔴 **Backend RNG configurable** (`stdlib` / `numpy`) + default a `numpy` para coincidir
   con parciales. Expuesto como selectbox en cada submódulo.
2. 🔴 **Bloque `error ∝ 1/√N`** en Visualizaciones (1D, multidim, rechazo 2D):
   - Tabla N, 2N, 4N, 8N con σ, SE, ratio SE(k·N)/SE(N), esperado 1/√k.
   - Botón "Reducir error a la mitad" → corre 4N y compara.
   - Gráfico log-log con curva teórica `c/√N`.
3. 🔴 **Tabla Bernoulli completa** en preset π (columnas exactas del amigo).
4. 🔴 **Preset-buttons de parcial** con enunciado + autollenado:
   - "A-4: ∫∫ e^(x-y) en [0,2]²"
   - "Variante: ∫∫ x·e^y en [0,1]×[1,3]"
   - "Prueba Evaluativa 3a: π círculo r=1"
   - "Prueba Evaluativa 3b: área y=x² vs y=√x"
5. 🟡 **Valor exacto simbólico** en multidim y rechazo 2D.
6. 🟡 **Lotes incrementales** (N/20) con banda IC 95% sombreada.
7. 🟡 **Referencias a parciales** en captions.

Todo es *activable/desactivable* desde la UI (cumpliendo "cuanto más personalizable mejor").
