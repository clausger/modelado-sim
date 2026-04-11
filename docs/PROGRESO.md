# Progreso del Proyecto

Registro de avance sesion por sesion.

---

## Sesion 2 — 2026-04-11

### Objetivos
- Revisar la implementacion de Monte Carlo contra el material de catedra
- Alinear formulas e interfaz con la convencion del profesor Caceres
- Agregar modo "paso a paso" pedagogico para resolver ejercicios de la guia
- Cubrir los ejercicios 1 y 10 (muestreo por rechazo) con el script

### Trabajo realizado

1. **Revision del PDF del profesor (slides 10-14)**
   - Confirmado: IC = Î ± z · σ/√N, **sin** factor (b-a) ni V(D)
   - Confirmado: σ usa f̄ (media de las evaluaciones), NO x̄
   - Convencion: `random.seed(42)` + `random.uniform(a,b)` de stdlib

2. **montecarlo_teoria.md (nuevo)**
   - Referencia con las 6 secciones del material de clase
   - Formulas alineadas a la convencion de catedra
   - Nota critica sobre σ con f̄ en lugar de x̄

3. **modules/montecarlo.py — refactor del modulo**
   - **Alineacion con catedra**: IC sin (b-a) ni V(D) en los 3 submodulos existentes
   - **Selector de confianza customizable**: presets 90/95/99/99.7% + modo "Personalizado" via `scipy.stats.norm.ppf`
   - **Formato decimal legible**: helper `_fmt_decimal()` nunca usa notacion cientifica, ajusta decimales segun magnitud
   - **Tabs Symbolab**: layout `[Resumen | Paso a paso | Visualizaciones]` en 1D, multidim y comparacion
   - **Paso a paso pedagogico**: 9 pasos con LaTeX intermedios, primeras 5 muestras/evaluaciones, f̄, σ, IC, comparacion con exacto
   - **Valores intermedios**: panel con N, (b-a), Σf(xᵢ), f̄, σ, min, max, z, EE, margen
   - **Convencion stdlib**: `_seed_rng()` + `_sample_uniform()` en un unico stream reproducible
   - **Fixes**: eliminar llamada muerta a `intervalo_confianza_95`, reemplazar `np.trapz` deprecated

4. **Nuevo submodulo "Muestreo por rechazo 2D"**
   - Menu principal ahora tiene 4 submodulos
   - **Modo circulo inscripto (π)**: L, r customizables; auto-deteccion del caso π cuando L=2, r=1
   - **Modo area entre curvas f(x), g(x)**: bounding box via grilla densa (2001 pts), valor teorico con `sympy.integrate(Abs(f-g))` + fallback `scipy.integrate.quad`
   - Estimador Bernoulli: Â = A_rect · k/N, σ_p = √(p̂(1-p̂)), IC = Â ± z · A_rect · σ_p/√N
   - Sampler `_sample_uniform_2d()` interleaved (x₁,y₁,x₂,y₂,...) para matchear el stream de un loop manual
   - 3 tabs: Resumen, Paso a paso (9 pasos), Visualizacion (scatter con aciertos/rechazos + contorno)
   - **Verificado**: ejercicio 1 (n=10000, seed=42) → k=7848, π̂=3.1392, IC 95% [3.1070, 3.1714] ✅
   - **Verificado**: ejercicio 10 (f=√x, g=x² en [0,1], n=10000) → Â=0.3250, IC 95% [0.3156, 0.3344] cubre 1/3 ✅

5. **Logic check contra el PDF del profesor (pags. 36-42)**
   - Leido el PDF oficial del profesor Caceres, secciones de Monte Carlo (pags. 36-42)
   - **Pag. 36**: 4 pasos del metodo (definir dominio → generar → evaluar → calcular) + caracteristicas
   - **Pag. 37**: Metodo grafico de rechazo: Area(R) ≈ A_rect × (aciertos/total)
   - **Pag. 38**: Codigo python oficial para π con `random.uniform(-1,1)` stdlib
   - **Pag. 39**: Formulas σ, EE=σ/√n, IC = [Î - z·σ/√n, Î + z·σ/√n] sin (b-a); Tabla 4 con z criticos
   - **Pag. 40**: LGN, x̄ = (1/n)·Σf(xᵢ), verificacion seed=42 → primeros 5 valores
   - **Pag. 41**: Fundamento I = V(Ω)·E[f(x)], casos 1D y 2D
   - **Pag. 42**: Codigo oficial `monte_carlo_with_ci()` — inconsistencia menor: usa `np.random.uniform` en vez del stdlib de pag. 40
   - **Decision**: seguir con `random.seed(42)` + `random.uniform()` stdlib porque pag. 40 lo fija explicitamente y es la unica manera de verificar la semilla con los 5 valores del libro

6. **Tanda 1 — Fixes de texto (resultado del logic check)**
   - **`montecarlo_teoria.md` §6.2 reescrito**: la version anterior decia que el libro tenia un "error de formula" cuando en realidad el libro define `x̄ = (1/n)·Σ f(xᵢ)` explicitamente en pag. 40 (o sea, x̄ ≡ f̄). Corregido para flagear solo el erratum tipografico menor (falta `Σ` en la formula de pag. 39)
   - **Caption Bernoulli en rechazo**: el libro NO define IC para muestreo por rechazo (seccion 4 del libro cubre solo el estimador de area). Agregado caption explicando que usamos el IC Bernoulli estandar sobre p̂ = k/N porque es el formalismo correcto para un conteo binomial

7. **Tanda 2 — Didactico en UI (helper `_render_teoria_libro()`)**
   - Nueva funcion helper que renderiza un **expander colapsable** con toda la teoria del libro al tope de cada submodulo
   - 6 secciones autocontenidas:
     1. Los 4 pasos del metodo (pag. 36)
     2. Fundamento: I = V(Ω)·E[f(x)] + LGN (pag. 40-41)
     3. Formulas operativas: Î, σ, EE, IC (pag. 39) con caption sobre la convencion sin (b-a)
     4. Nota `x̄` libro ≡ `f̄` app
     5. Tabla 4 de z criticos (90/95/99/99.7)
     6. Verificacion de semilla: primeros 5 valores `random.uniform(0,1)` con seed=42 para que el alumno chequee que su stdlib matchea el del libro
   - Wired en los 4 submodulos: Integracion 1D, Multidim, Muestreo por rechazo 2D, Comparacion de metodos

### Cobertura de la guia de ejercicios

| # | Tipo | Submodulo |
|---|------|-----------|
| 1 | π por rechazo | **Muestreo por rechazo 2D** (nuevo) ✅ |
| 2-9 | Integrales 1D / 2D | Integracion 1D / Multidim ✅ |
| 10 | Area entre curvas por rechazo | **Muestreo por rechazo 2D** (nuevo) ✅ |
| 11 | Simulador orbital | Pendiente |
| 12 | Black-Scholes + VaR | Pendiente |

### Commits de la sesion

| Hash | Mensaje |
|------|---------|
| `1c5ef39` | docs(montecarlo): add class theory reference |
| `3140389` | feat(montecarlo): catedra alignment, step-by-step tabs, rejection sampling 2D |
| `801af79` | docs(progreso): log session 2 Monte Carlo improvements |
| `53e13e6` | feat(montecarlo): in-app theory reference + clarify rejection CI |

### Proximos pasos
- **Ejercicio 2**: I = ∫[0,1] e^(-x²) dx con IC 99.7%, error maximo 1/100 (pendiente para proxima sesion)
- Resolver ejercicios 3-9 uno por uno con el script
- Decidir si los ejercicios 11 y 12 van como submodulos dedicados o inline
- Pushear los 4 commits locales a `origin/main` cuando estemos listos

---

## Sesion 1 (continuacion) — 2026-04-08

### Trabajo realizado

1. **modules/integracion.py — Implementacion completa (Newton-Cotes)**
   - 4 metodos: Rectangulo (Punto Medio), Trapecio Compuesto, Simpson 1/3, Simpson 3/8
   - Submenú con radio horizontal para navegar entre metodos
   - Cada metodo incluye:
     - Expander con teoria: idea geometrica, formula LaTeX, error de truncamiento, restriccion de n
     - Inputs comunes: funcion f(x), limites a/b, subintervalos n, slider de tolerancia
     - Validaciones: Simpson 1/3 fuerza n par, Simpson 3/8 fuerza multiplo de 3
     - Metricas: integral aproximada, valor exacto (SymPy), error absoluto/relativo
     - Tabla de puntos de evaluacion: i, x_i, f(x_i), peso, contribucion (resalta maximo)
     - Tabla de convergencia: n, h, resultado, error vs exacto, convergencia
     - Grafico Plotly: curva f(x) + figuras geometricas (rectangulos/trapecios) + puntos
   - Comparacion de Metodos:
     - Tabla con los 4 metodos, n ajustado, resultado, error, orden
     - Grafico de barras de error absoluto (escala log)
     - Grafico de convergencia superpuesto (4 lineas + valor exacto)
   - Tabla comparativa fija siempre visible al inicio del modulo

2. **app.py** — Integracion Numerica marcado como activo (ya no WIP)

### Estado actual

| Modulo | Estado |
|--------|--------|
| Raices y Punto Fijo | WIP placeholder |
| Newton-Raphson / Aitken | WIP placeholder |
| Interpolacion y Derivacion | WIP placeholder |
| **Integracion Numerica** | **Completo** |
| **Monte Carlo** | **Completo** |
| EDOs | WIP placeholder |

---

## Sesion 1 — 2026-04-08

### Objetivos
- Crear la estructura inicial del proyecto
- Implementar el modulo de Monte Carlo completo
- Crear el repositorio en GitHub

### Trabajo realizado

1. **Estructura del proyecto**
   - Creacion de carpetas `modules/` y `utils/`
   - Configuracion de `requirements.txt` con dependencias versionadas
   - Archivo `.gitignore` para Python/Streamlit

2. **app.py — Entry point**
   - Menu lateral (sidebar) con navegacion entre modulos
   - Iconos por modulo, indicador WIP para los no implementados
   - Montecarlo seleccionado por defecto

3. **Modulos WIP (placeholders)**
   - `biseccion.py`, `punto_fijo.py`, `newton_raphson.py`, `aitken.py`
   - `lagrange.py`, `derivacion.py`, `integracion.py`, `edo.py`
   - Cada uno muestra mensaje "Proximamente" con temario del modulo

4. **utils/errores.py**
   - `error_absoluto()`, `error_relativo()`, `intervalo_confianza_95()`

5. **utils/graficos.py**
   - `plot_funcion()` — grafico de f(x)
   - `plot_convergencia()` — estimacion vs N con banda de confianza
   - `plot_scatter_montecarlo()` — puntos dentro/fuera del area
   - `plot_scatter_3d()` — visualizacion 3D para integracion 2D
   - `plot_comparacion_barras()` — barras de errores entre metodos

6. **modules/montecarlo.py — Implementacion completa**
   - Integracion 1D: funcion parseada con SymPy, tabla de iteraciones con resaltado de tolerancia, scatter de puntos, grafico de convergencia con IC 95%, calculo de valor exacto con SymPy
   - Integracion multidimensional: soporte 2D y 3D, visualizacion scatter 3D, valor exacto cuando es posible
   - Comparacion de metodos: Monte Carlo vs Trapecios vs Simpson, tabla con resultados/errores/tiempos, graficos de barras

7. **Repositorio GitHub**
   - Repo creado: https://github.com/clausger/modelado-sim
   - Commit inicial con los 17 archivos
   - Push a `main`

8. **Ejecucion local**
   - Instalacion de dependencias con pip3
   - Configuracion de Streamlit (headless, sin email prompt)
   - App corriendo en http://localhost:8501

9. **Documentacion**
   - `docs/DESCRIPCION.md` — descripcion completa del proyecto
   - `docs/PROGRESO.md` — este documento

### Estado actual

| Modulo | Estado |
|--------|--------|
| Raices y Punto Fijo | WIP placeholder |
| Newton-Raphson / Aitken | WIP placeholder |
| Interpolacion y Derivacion | WIP placeholder |
| Integracion Numerica | WIP placeholder |
| **Monte Carlo** | **Completo** |
| EDOs | WIP placeholder |

### Proximos pasos
- Verificar funcionamiento de Monte Carlo en browser
- Implementar siguiente modulo segun prioridad de la cursada

---

<!-- Agregar nuevas sesiones arriba de este comentario -->
