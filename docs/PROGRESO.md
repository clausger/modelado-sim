# Progreso del Proyecto

Registro de avance sesion por sesion.

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
