# Plan módulo EDO — inventario y hoja de ruta

Documento de planning para el módulo de Ecuaciones Diferenciales Ordinarias
(`modules/edo.py`), actualmente stub. Este archivo sobrevive refresh de sesión.

Última actualización: 2026-04-21.

---

## 0. Contexto

- `modules/edo.py` hoy es un stub de 15 líneas (`st.info("Proximamente")`).
- Hay que construirlo desde cero siguiendo el nivel de pulido de los módulos
  ya implementados (Lagrange, Integración, Newton-Raphson, Steffensen).
- Los parciales del profe Cáceres **siempre** piden EDO en el ítem 5.

---

## 1. Inventario de helpers reutilizables

### 1.1 Helpers matemáticos genéricos (`utils/`)

| Helper | Archivo | Uso en EDO |
|---|---|---|
| `error_absoluto(aprox, exacto)` | `utils/errores.py` | error local/global vs analítica |
| `error_relativo(aprox, exacto)` | `utils/errores.py` | con guarda div/0 |
| `math_input(label, default_latex, key)` | `utils/math_keyboard/` | teclado MathQuill para f(t,y) |
| `parse_latex(latex, [vars])` | `utils/math_keyboard/` | **multi-variable** (t, y) — CRÍTICO |
| `parse_expr_to_float(txt)` | `utils/math_keyboard/` | h, t₀, y₀, t_final (acepta `pi/8` etc) |

### 1.2 UI compartida (`utils/ui/`)

| Helper | Archivo | Uso |
|---|---|---|
| `fmt_decimal(v, precision)` | `tablas.py` | respeta `modo_libro` (5 dec) |
| `render_tabla_iteraciones(df, resaltar, ...)` | `tablas.py` | DataFrame + export CSV/LaTeX |
| `resaltar_tolerancia(col, tol)` | `tablas.py` | pinta verde filas con error ≤ tol |
| `Paso(...)` + `render_pasos([...])` | `pasos.py` | paso-a-paso técnico/coloquial |
| `render_teoria("edo_teoria")` | `teoria.py` | lee `docs/teoria/edo_teoria.md` — **hay que crear** |
| `get_config()` | `config.py` | `modo_libro`, `modo_examen`, `tema_pasos`, `precision_efectiva` |
| `GLOSARIO`, `var_tooltip("h")` | `glosario.py` | tooltips centrales |

### 1.3 Gráficos (`utils/graficos.py`)

Ya existen (reutilizables tal cual):
- `plot_funcion(f, a, b)` — solución analítica
- `plot_comparacion_barras(metodos, res, err)` — comparar Euler vs Heun vs RK4
- `apply_geogebra_style(fig)` — estilo claro para reportes

**Faltan** (crear nuevos):
- `plot_trayectoria_edo(ts, ys, y_analitica=None)` — (t, y) con analítica
- `plot_campo_pendientes(f, t_range, y_range, trayectoria=None)` — slope field
- `plot_orden_convergencia(hs, errs)` — log-log con pendiente teórica

### 1.4 Helpers de integración reutilizables (`modules/integracion.py`)

El refactor reciente dejó helpers que aplican 1:1 a EDO:

| Helper | Uso en EDO |
|---|---|
| `_fmt_tex(v, n_dec)` | formato LaTeX entero/decimal limpio |
| `_tex(expr)` | `sp.latex(..., ln_notation=True)` |
| `_FuncionSegura(expr, x_sym)` | wrapper con L'Hôpital automático |
| `_explicar_lhopital(expr, x_sym, x0)` | info num/den + derivadas |
| `_mostrar_avisos_lhopital(f_segura)` | expander L'Hôpital |

**Decisión pendiente**: si EDO también los usa, extraer a `utils/simbolico.py`
para no importar entre módulos. Por ahora podemos copiar y consolidar luego.

### 1.5 Patrón "Respuesta lista para examen (formato alumno)"

Ya existe en: Lagrange, Newton-Raphson, Steffensen, Integración (los 4 métodos).

**Patrón estándar a replicar:**
```python
with st.expander("📝 Respuesta lista para examen (formato alumno)"):
    bloque = []
    bloque.append("**Planteo.** ...")
    bloque.append(rf"$$...$$")
    # ... tabla, fórmulas, sustituciones, resultado, error
    texto = "\n".join(bloque)
    st.markdown(texto)                      # render
    st.code(texto, language="markdown")     # copy-paste
```

---

## 2. Convenciones de cátedra que aplican

- **Precisión**: `round(·, 5)` si `modo_libro` activo, sino 6 decimales.
- **4 criterios de detención** para iterativos (aplica a Euler implícito si
  hay resolución interna por punto fijo / Newton).
- **Notación profe**: `f⁽ᴵ⁾`, `f⁽ᴵᴵ⁾` (romano) para derivadas sucesivas.
- **Tabla del alumno**: `i | tᵢ | yᵢ | f(tᵢ,yᵢ) | y_{i+1}` con TODAS las iteraciones.

---

## 3. Lo que piden los parciales (de `docs/PARCIALES_MAPEO.md`)

Los 4 parciales (A, B, C, D) piden EDO en el ítem 5:

| Parcial | Ejercicio |
|---|---|
| **A-5** | `dy/dx = y·sin(t)`, `y(0)=1`, `[0, π]`. a) Euler o RK con h=π/10, tol 10⁻¹, comparar 1ª y 2ª iter. b) RK4 con **k₁..k₄ explícitos por paso**, tol 10⁻⁶ |
| **B-5** | a) `dy/dx = (x³−1)/y²`, `y(1)=3`, h=0.1, **Heun**, hallar y(1.5). b) `dy/dt = 2t√y`, `y(1)=3`, h=0.1, **RK4**, hallar y(1.2) |
| **C-5** | Idéntico a B-5 con variantes menores (8 decimales en algunos items) |
| **D-5** | `dy/dx = cos(x)+x`, `y(0)=1`, `[0, π/2]`. a) Euler h=π/8, tol 10⁻¹. b) RK4 con k₁..k₄, tol 10⁻⁶ |

**Patrón común del profe:**
- Input: `f(t,y)`, `y(t₀)=y₀`, intervalo `[t₀, t_final]`, `h` (o `n`)
- Output esperado: tabla iteración-por-iteración
- RK4: **tabla adicional** con k₁, k₂, k₃, k₄ por paso
- Comparar con solución analítica cuando la haya
- Tolerancias: 10⁻¹ (Euler), 10⁻⁶ (RK4)

---

## 4. Métodos a implementar (por prioridad)

| # | Método | Orden | Costo/paso | Fórmula |
|---|---|---|---|---|
| 1 | **Euler explícito** | O(h) | 1 eval f | `y_{i+1} = y_i + h·f(tᵢ, yᵢ)` |
| 2 | **Euler implícito** (backward) | O(h) | 1 eval + solver | `y_{i+1} = y_i + h·f(t_{i+1}, y_{i+1})` |
| 3 | **Heun / Euler mejorado** | O(h²) | 2 evals f | predictor (Euler) + corrector (trapecio) |
| 4 | **Punto medio RK2** | O(h²) | 2 evals | `k₁=f(tᵢ,yᵢ); k₂=f(tᵢ+h/2, yᵢ+h·k₁/2); y_{i+1}=yᵢ+h·k₂` |
| 5 | **RK4** | O(h⁴) | 4 evals | clásico k₁..k₄ |
| 6 | **Adams-Bashforth 2/3/4** | multipaso | 1 eval | requiere arranque RK |
| 7 | **Sistemas de EDOs** | — | — | vector y⃗, vector f⃗ |

**Must-have (Fase 1)**: #1 Euler, #3 Heun, #5 RK4.
**Nice-to-have (Fase 2)**: #2 Euler implícito, #4 RK2.
**Avanzado (Fase 3)**: #6 multipaso, #7 sistemas.

---

## 5. Features UI específicas para EDO

| Feature | Obligatorio / Opcional |
|---|---|
| Tabla detallada `i | tᵢ | yᵢ | f(tᵢ,yᵢ) | y_{i+1}` | ✅ Obligatorio (estilo profe) |
| RK4 con tabla adicional `k₁, k₂, k₃, k₄, y_new` por paso | ✅ Obligatorio (lo pide literal) |
| Input solución analítica opcional (sympy) | ✅ Obligatorio |
| Auto-resolución simbólica `y(t)` via `sp.dsolve` | ✅ si es posible |
| Gráfico trayectoria (t, y) con analítica superpuesta | ✅ Obligatorio |
| Campo de pendientes (slope field) | 🟡 Nice-to-have |
| Tabla de convergencia h → h/2 → h/4 con orden empírico | 🟡 Nice-to-have |
| Comparación Euler vs Heun vs RK4 (tab) | ✅ Obligatorio |
| Bloque "📝 Respuesta lista para examen" por método | ✅ Obligatorio |
| Presets de los 4 parciales | 🟡 Nice-to-have |

---

## 6. Lo que hay que crear desde cero

1. **`modules/edo.py`** (~800–1000 líneas, estructura similar a integración).
2. **`docs/teoria/edo_teoria.md`** — fórmulas, errores locales/globales,
   estabilidad absoluta, orden de convergencia empírico.
3. **Nuevas funciones en `utils/graficos.py`**:
   - `plot_trayectoria_edo(...)`
   - `plot_campo_pendientes(...)`
   - `plot_orden_convergencia(...)`

---

## 7. Hoja de ruta por fases

### Fase 1 — Entregable mínimo para el parcial (OBJETIVO PRINCIPAL)

Euler explícito + Heun + RK4, cada uno con:
- Inputs: `f(t,y)`, `t₀`, `y₀`, `t_final`, `h` (o `n`), analítica opcional
- Auto-`dsolve` cuando sea posible
- Tabla iteración-por-iteración (estilo profe)
- RK4: tabla adicional `k₁..k₄`
- Gráfico trayectoria (numérica + analítica)
- Bloque "📝 Respuesta lista para examen"
- Métricas de error (si hay analítica)

### Fase 2 — Profundidad

- Euler implícito (con solver interno + 4 criterios de detención)
- RK2 punto medio
- Tab "Comparación de métodos" (como integración)
- Tabla de convergencia con orden empírico `p ≈ log₂(err(h)/err(h/2))`
- Campo de pendientes interactivo

### Fase 3 — Avanzado

- Adams-Bashforth 2/3/4 (multipaso) con arranque RK
- Sistemas de EDOs 2D (Lotka-Volterra, oscilador, etc.)
- Detección de rigidez (stiffness)
- Presets por parcial (1-click)

---

## 8. Orden de implementación sugerido

1. Crear `docs/teoria/edo_teoria.md` con fórmulas y teoría.
2. Algoritmos puros (sin Streamlit) en `modules/edo.py`:
   `_euler`, `_heun`, `_rk4` que devuelvan `(ts, ys, tabla_iter)`.
3. Validar numéricamente contra un caso con solución analítica conocida
   (ej. `dy/dx = y`, `y(0)=1` → `y(x)=eˣ`).
4. Validar contra los 4 parciales del profe.
5. Agregar UI: inputs, tablas, gráficos, bloque examen.
6. Smoke test con streamlit.

---

## 9. Preguntas abiertas

- [ ] ¿Usamos variable `t` o `x` como independiente? El profe mezcla ambas
      (B-5 usa `x`, A-5 usa `t`). Propuesta: input configurable, default `t`.
- [ ] ¿Presets por parcial en Fase 1 o Fase 3? Depende del tiempo.
- [ ] ¿Extraer helpers simbólicos (`_FuncionSegura`, `_tex`, etc) a
      `utils/simbolico.py` antes de EDO para evitar import cruzado?

---

## 10. Notas de sesión

- **2026-04-21**: Plan creado tras refactor exitoso de `modules/integracion.py`
  (L'Hôpital automático + error de truncamiento con ξ + respuesta examen).
  Validado contra ejercicio 3 del profe (sin(x)/(x+ln(1+x))) — todos los
  valores matchean con el manuscrito del alumno.

- **2026-04-21 (tarde)**: El alumno mandó 4 fotos de ejercicios de examen
  adicionales. Se incorporaron al plan como requisitos (sección 11).

---

## 11. Requisitos extra de ejercicios de examen (2026-04-21)

Se recibieron 4 ejercicios que refinan/refuerzan los requisitos del módulo:

### 11.1 Ejercicio "cuadro resumen con error absoluto" (repetido 2 veces)

> "Modele y Simule bien sea en Excel o Python la solución de las siguientes
> ecuaciones diferenciales ordinarias. Calcule la solución real de ser posible
> y presente un cuadro resumen con las iteraciones y el error Absoluto para
> cada una. **Analice el comportamiento del error a medida que itera.**"

- (a) `dy/dx = (x³−1)/y²`, `y(1)=3`, `h=0.1`, **Euler mejorado (Heun)**, `y(1.5)`
- (b) `dy/dt = 2t·√y`, `y(1)=3`, `h=0.1`, **RK4**, `y(1.2)` (cohete)

**Exige:**
- ✅ Columna "error absoluto" por iteración (cuando hay analítica).
- ✅ Análisis textual/gráfico del **comportamiento del error** (crece, oscila, estable).
- ✅ Solución real vía `sp.dsolve` obligatoria.

### 11.2 Ejercicio "Euler + RK4 comparados, mismo problema" (2 variantes)

Variante A — `dy/dx = cos(x)+x`, `y(0)=1`, `[0, π/2]`:
- (a) Analítica + **Euler** con `h=π/8`, precisión `10⁻¹`. "Compruebe que la
  **primera y segunda iteración** coincidan con los datos de salida."
- (b) **RK4** precisión `10⁻⁶`, tabla con `k₁, k₂, k₃, k₄` y nueva `y`.
  "Compara de modo gráfico y numérico los resultados obtenidos en este ítem
  con el anterior."

Variante B — `dy/dx = y·sin(t)`, `y(0)=1`, `[0, π]`:
- Idéntico patrón con `h=π/10`.
- **Detalle**: el enunciado mezcla `x` y `t` → el módulo debe aceptar cualquiera.

**Exige:**
- ✅ **Resaltar 1ª y 2ª iteración** en la tabla (el profe lo pide explícito).
- ✅ **Tolerancia objetivo** configurable (10⁻¹ Euler, 10⁻⁶ RK4).
- ✅ **Comparación (a) Euler vs (b) RK4** en mismo problema → subir esto a
  Fase 1 (antes era Fase 2).
- ✅ Variable independiente configurable (`t` o `x`).

### 11.3 Refinamientos al diseño

| Cambio | Antes | Ahora |
|---|---|---|
| Columna error absoluto en tabla iter | Opcional | **Obligatorio si hay analítica** |
| Análisis comportamiento del error | No listado | **Obligatorio** (texto + mini-plot) |
| Resaltar y₁, y₂ en tabla | No listado | **Obligatorio** |
| Comparación Euler vs RK4 | Fase 2 | **Fase 1** |
| Tolerancia objetivo (10⁻¹, 10⁻⁶) | No listado | **Obligatorio** (input) |
| Variable independiente `t`/`x` | Pregunta abierta | **Decidido: selector con default `t`** |

### 11.4 Checklist Fase 1 actualizada

Cada método (Euler, Heun, RK4) debe incluir:
- [ ] Inputs: f(variable, y), t₀/x₀, y₀, t_final, h (o n), analítica opcional
- [ ] Selector de variable independiente (`t` o `x`)
- [ ] Auto-`dsolve` con fallback si falla
- [ ] Input "tolerancia objetivo" (sugerido 10⁻¹ para Euler, 10⁻⁶ para RK4)
- [ ] Tabla iter con columnas: `i | varᵢ | yᵢ | f(varᵢ,yᵢ) | y_{i+1} | y_real | |error|`
- [ ] **Resaltar filas i=1 e i=2** (amarillo/verde suave)
- [ ] **Resaltar filas con |error| > tolerancia** (rojo suave)
- [ ] RK4: tabla adicional `k₁, k₂, k₃, k₄, y_new` por paso
- [ ] Gráfico trayectoria (numérica + analítica)
- [ ] **Mini-gráfico "evolución del error"** (|error| vs iteración)
- [ ] **Texto automático** analizando comportamiento del error
  (ej. "el error crece linealmente con `i`", "el error se estabiliza en X")
- [ ] Bloque "📝 Respuesta lista para examen"

Y a nivel módulo:
- [ ] **Tab "Comparación Euler vs Heun vs RK4"** — mismo problema, 3 métodos,
      tabla lado-a-lado de errores, gráfico superpuesto.
