import random as _pyrandom
import time
from typing import Literal, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sympy as sp
from scipy import integrate, optimize
from scipy.stats import norm

from utils.errores import error_absoluto, error_relativo
from utils.graficos import (
    plot_comparacion_barras,
    plot_convergencia,
    plot_funcion,
    plot_scatter_3d,
    plot_scatter_montecarlo,
)
from utils.math_keyboard import math_input, parse_expr_to_float, parse_latex

# Dos backends de RNG soportados:
#   - "numpy": np.random.seed(...) + np.random.uniform(...). Coincide con el
#     formato pedido por el profesor en los parciales (ej. A-4, prueba
#     evaluativa, etc. piden literalmente np.random.seed(0), n=10000).
#   - "stdlib": random.seed(...) + random.uniform(...). Coincide con los
#     ejemplos del libro de Caceres.
# Default: numpy (es lo que exigen los parciales).
RngBackend = Literal["numpy", "stdlib"]
_DEFAULT_BACKEND: RngBackend = "numpy"

_Z_PRESETS = {
    "90%": 90.0,
    "95%": 95.0,
    "99%": 99.0,
    "99.7% (3σ)": 99.7,
}


def _fmt_decimal(val: float, min_decimals: int = 6) -> str:
    """Formato decimal legible (sin notacion cientifica).

    Ajusta dinamicamente la cantidad de decimales para que siempre se vean
    al menos 3 cifras significativas del valor, con un piso de `min_decimals`.

    Ejemplos:
        0.002        -> "0.002000"
        0.00000760   -> "0.000007600"
        123.456789   -> "123.456789"
    """
    if val is None:
        return "-"
    try:
        fval = float(val)
    except (TypeError, ValueError):
        return str(val)
    if not np.isfinite(fval):
        return str(fval)
    if fval == 0:
        return "0." + "0" * min_decimals
    abs_val = abs(fval)
    if abs_val >= 1:
        decimals = min_decimals
    else:
        exp = int(np.floor(np.log10(abs_val)))
        decimals = max(min_decimals, -exp + 3)
    decimals = min(decimals, 12)
    return f"{fval:.{decimals}f}"


def _z_from_confianza(confianza_pct: float) -> float:
    """Calcula z_(alpha/2) para un nivel de confianza dado en porcentaje.

    Ejemplos:
        95.0  -> 1.9600
        99.0  -> 2.5758
        99.7  -> 2.9677  (regla 3-sigma)
    """
    alpha = 1.0 - (confianza_pct / 100.0)
    return float(norm.ppf(1.0 - alpha / 2.0))


def _seed_rng(seed: Optional[int], backend: RngBackend = _DEFAULT_BACKEND) -> None:
    """Fija la semilla del RNG elegido. seed=None => no hace nada."""
    if seed is None:
        return
    if backend == "numpy":
        np.random.seed(int(seed))
    else:
        _pyrandom.seed(int(seed))


def _sample_uniform(
    a: float, b: float, n: int, backend: RngBackend = _DEFAULT_BACKEND
) -> np.ndarray:
    """Genera n muestras U(a,b) con el backend elegido.

    - numpy: np.random.uniform(a, b, n) (vectorizado, idiomatico en parciales).
    - stdlib: loop de random.uniform (coincide con los ejemplos del libro).

    No resetea la semilla: el caller debe llamar _seed_rng(...) una sola vez
    antes del primer sampleo para mantener un unico stream reproducible.
    """
    N = int(n)
    if backend == "numpy":
        return np.random.uniform(a, b, N).astype(float)
    return np.fromiter(
        (_pyrandom.uniform(a, b) for _ in range(N)),
        dtype=float,
        count=N,
    )


def _sample_uniform_2d(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    n: int,
    backend: RngBackend = _DEFAULT_BACKEND,
) -> tuple[np.ndarray, np.ndarray]:
    """Muestrea n puntos 2D.

    - numpy: draws vectorizados (todos los x, luego todos los y). Coincide con
      `x = np.random.uniform(...); y = np.random.uniform(...)` escrito en un
      parcial.
    - stdlib: interleaved (x1, y1, x2, y2, ...) para coincidir con un loop
      manual `for _ in range(n): x = uniform(...); y = uniform(...)` del libro.
    """
    N = int(n)
    if backend == "numpy":
        xs = np.random.uniform(x_min, x_max, N).astype(float)
        ys = np.random.uniform(y_min, y_max, N).astype(float)
        return xs, ys
    xs = np.empty(N, dtype=float)
    ys = np.empty(N, dtype=float)
    for i in range(N):
        xs[i] = _pyrandom.uniform(x_min, x_max)
        ys[i] = _pyrandom.uniform(y_min, y_max)
    return xs, ys


def _rng_selector(
    key: str, seed_default: int = 0
) -> tuple[RngBackend, Optional[int]]:
    """Renderiza un selector de backend RNG + semilla reproducible.

    Devuelve `(backend, seed)` donde `seed=None` significa "sin semilla"
    (cada corrida da un resultado distinto).

    Los parciales del profesor piden literalmente `np.random.seed(0)` con
    `n=10000`, por eso el default es numpy + semilla 0.
    """
    col_b, col_s1, col_s2 = st.columns([1.2, 1, 1])
    with col_b:
        backend_label = st.selectbox(
            "Backend RNG",
            ["numpy (parcial)", "stdlib (libro)"],
            index=0 if _DEFAULT_BACKEND == "numpy" else 1,
            key=f"{key}_backend",
            help=(
                "numpy: coincide con np.random.seed(...) + np.random.uniform(...), "
                "que es lo que piden los parciales (A-4, Prueba Evaluativa, etc.). "
                "stdlib: random.seed(...) + random.uniform(...), convencion del "
                "libro de Caceres."
            ),
        )
    backend: RngBackend = "numpy" if backend_label.startswith("numpy") else "stdlib"

    with col_s1:
        fijar = st.checkbox(
            "Fijar semilla",
            value=True,
            key=f"{key}_use_seed",
            help="Desactivar = resultados no reproducibles.",
        )
    with col_s2:
        semilla_val = st.number_input(
            "Semilla",
            value=int(seed_default),
            min_value=0,
            step=1,
            key=f"{key}_seed_val",
            disabled=not fijar,
            help="Los parciales del profesor usan semilla 0.",
        )

    seed = int(semilla_val) if fijar else None
    return backend, seed


def _fmt_seed_call(semilla: Optional[int], backend: RngBackend) -> str:
    """Formatea el fijado de semilla como codigo Python legible.

    Ejemplos:
        _fmt_seed_call(0, "numpy")   -> "np.random.seed(0)"
        _fmt_seed_call(42, "stdlib") -> "random.seed(42)"
        _fmt_seed_call(None, ...)    -> "(sin semilla fija)"
    """
    if semilla is None:
        return "(sin semilla fija)"
    prefix = "np.random.seed" if backend == "numpy" else "random.seed"
    return f"{prefix}({int(semilla)})"


def _fmt_seed_short(semilla: Optional[int]) -> str:
    """Representacion corta de la semilla para mensajes (e.g. `0` o `aleatoria`)."""
    return "aleatoria" if semilla is None else str(int(semilla))


def _confianza_selector(key: str) -> tuple[str, float]:
    """Renderiza un selector de nivel de confianza totalmente personalizable.

    Expone un preset (90/95/99/99.7/personalizado) y un number_input para
    afinar el porcentaje exacto. Devuelve (label, z_score) donde label es
    el string a mostrar en las metricas (ej: "IC 99.7%").
    """
    preset = st.selectbox(
        "Nivel de confianza",
        list(_Z_PRESETS.keys()) + ["Personalizado"],
        index=1,  # 95% por defecto
        key=f"{key}_preset",
        help="Elegi un preset o 'Personalizado' para ingresar un valor exacto.",
    )
    if preset == "Personalizado":
        confianza_pct = st.number_input(
            "Confianza (%)",
            min_value=50.0,
            max_value=99.999,
            value=95.0,
            step=0.1,
            format="%.3f",
            key=f"{key}_custom",
            help="Porcentaje de confianza entre 50 y 99.999.",
        )
    else:
        confianza_pct = _Z_PRESETS[preset]

    z = _z_from_confianza(float(confianza_pct))
    label = f"{confianza_pct:g}%"
    st.caption(f"z_(α/2) = {z:.4f}  |  confianza = {label}")
    return label, z


def _fmt_latex_num(val: float) -> str:
    """Formato compacto para insertar un numero dentro de LaTeX (sin ceros colgantes)."""
    if val == int(val):
        return str(int(val))
    return f"{val:.6f}".rstrip("0").rstrip(".")


def _render_teoria_libro() -> None:
    """Renderiza un expander con la teoria del libro del profesor Caceres (pags. 36-42).

    Contenido:
      - Los 4 pasos del metodo (pag. 36)
      - Fundamento teorico: I = V(Omega) * E[f(x)] + LGN (pag. 40-41)
      - Formulas operativas: estimador, sigma, EE, IC (pag. 39)
      - Nota sobre x_bar (libro) = f_bar (nuestra notacion)
      - Tabla 4 de z criticos (pag. 39)
      - Verificacion de semilla: primeros 5 valores U(0,1) con seed=42 (pag. 40)

    Se muestra como expander colapsable al tope de cada submodulo para tener
    siempre a mano lo necesario para responder preguntas teoricas.
    """
    with st.expander("📖 Teoria del libro (Caceres, pags. 36-42)", expanded=False):
        st.markdown("##### 1. Los 4 pasos del metodo (pag. 36)")
        st.markdown(
            "1. **Definir el dominio** de muestreo (intervalo, rectangulo, volumen).\n"
            "2. **Generar valores aleatorios** uniformes sobre el dominio.\n"
            "3. **Evaluar la funcion** en cada muestra.\n"
            "4. **Calcular la estimacion** como promedio ponderado por el volumen del dominio."
        )

        st.markdown("##### 2. Fundamento: valor esperado + LGN (pag. 40-41)")
        st.latex(r"I \;=\; \int_\Omega f(x)\,dx \;=\; V(\Omega)\cdot E[f(x)]")
        st.latex(
            r"\lim_{n \to \infty}\; \frac{1}{n}\sum_{i=1}^{n} f(x_i) \;=\; E[f(x)]"
            r"\quad\text{(Ley de los Grandes Numeros)}"
        )
        st.caption(
            "El estimador Monte Carlo se apoya en la LGN: el promedio de "
            "f(x_i) converge al valor esperado cuando n crece."
        )

        st.markdown("##### 3. Formulas operativas (pag. 39)")
        st.latex(
            r"\hat{I} \;=\; V(\Omega)\cdot \bar{f}, \qquad "
            r"\bar{f} \;=\; \frac{1}{n}\sum_{i=1}^{n} f(x_i)"
        )
        st.latex(
            r"\sigma \;=\; \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}\bigl(f(x_i)-\bar{f}\bigr)^{2}}"
        )
        st.latex(
            r"\text{EE} \;=\; \frac{\sigma}{\sqrt{n}}, \qquad "
            r"\text{IC} \;=\; \hat{I} \;\pm\; z_{\alpha/2}\cdot \frac{\sigma}{\sqrt{n}}"
        )
        st.caption(
            "Convencion de catedra: la formula del IC NO lleva (b-a) ni V(Omega). "
            "El factor de volumen ya esta en Î y la dispersion se mide sobre los "
            "valores f(x_i) directamente."
        )

        st.markdown("##### 4. Nota: `x̄` en el libro ≡ `f̄` en la app")
        st.markdown(
            "El libro escribe la formula de σ con `x̄`, pero en pag. 40 define "
            "explicitamente `x̄ = (1/n)·Σ f(xᵢ)` — o sea, el **promedio de las "
            "evaluaciones de la funcion**, no de las coordenadas. En esta app "
            "lo llamamos `f̄` para dejarlo explicito. Son matematicamente lo mismo."
        )

        st.markdown("##### 5. Valores criticos z (Tabla 4, pag. 39)")
        df_z = pd.DataFrame(
            {
                "Nivel de confianza": ["90%", "95%", "99%", "99.7% (3σ)"],
                "z_(α/2)": [1.645, 1.960, 2.576, 2.968],
            }
        )
        st.dataframe(df_z, use_container_width=True, hide_index=True)

        st.markdown("##### 6. Verificacion de semilla (pag. 40)")
        st.caption(
            "Con `random.seed(42)` + `random.uniform(0, 1)`, los primeros 5 valores "
            "deben coincidir exactamente con los del libro:"
        )
        df_seed = pd.DataFrame(
            {
                "i": [1, 2, 3, 4, 5],
                "random.uniform(0, 1)": [
                    0.6394267984578837,
                    0.025010755222666936,
                    0.27502931836911926,
                    0.22321073814882275,
                    0.7364712141640124,
                ],
            }
        )
        st.dataframe(df_seed, use_container_width=True, hide_index=True)
        st.caption(
            "Si estos valores coinciden con los que obtiene el alumno, la "
            "convencion de semilla + RNG es la correcta (stdlib, no numpy)."
        )


def _render_valores_intermedios_1d(
    f_vals: np.ndarray,
    a: float,
    b: float,
    n_muestras: int,
    z_score: float,
    conf_label: str,
) -> None:
    """Muestra los valores intermedios del calculo de Monte Carlo 1D.

    Expone todas las estadisticas crudas (Sigma, f_bar, sigma, min, max, z, margen)
    para que el alumno pueda responder preguntas teoricas sin estar a ciegas.
    """
    N = int(n_muestras)
    suma = float(np.sum(f_vals))
    f_mean = float(np.mean(f_vals))
    f_std = float(np.std(f_vals, ddof=1)) if N > 1 else 0.0
    f_var = f_std ** 2
    f_min = float(np.min(f_vals))
    f_max = float(np.max(f_vals))
    # Convencion de catedra (slides 11-12 de Caceres): EE = sigma / sqrt(N),
    # sin factor (b - a). Ver montecarlo_teoria.md seccion 6.
    err_std_val = f_std / np.sqrt(N)
    margen = z_score * err_std_val
    estimacion = (b - a) * f_mean
    ic_low = estimacion - (b - a) * margen
    ic_up = estimacion + (b - a) * margen

    with st.container(border=True):
        st.markdown("##### 🔬 Valores intermedios del cálculo")
        st.caption(
            "Cada métrica está etiquetada con la terminología de la consigna "
            "(media muestral, varianza muestral s², desvío estándar muestral σ, "
            "error estándar EE, estimación Î, intervalo de confianza IC)."
        )

        # Fila 1: parametros del dominio y sumas
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        r1c1.metric("N (muestras)", f"{N:,}")
        r1c2.metric("b − a", _fmt_decimal(b - a))
        r1c3.metric("Σ f(xᵢ)", _fmt_decimal(suma))
        r1c4.metric("f̄ (media muestral)", _fmt_decimal(f_mean))

        # Fila 2: dispersion (varianza + desvio) y extremos
        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        r2c1.metric("s² (varianza muestral)", _fmt_decimal(f_var))
        r2c2.metric("σ (desvío estándar muestral)", _fmt_decimal(f_std))
        r2c3.metric("min f(xᵢ)", _fmt_decimal(f_min))
        r2c4.metric("max f(xᵢ)", _fmt_decimal(f_max))

        # Fila 3: componentes del IC (convencion de catedra, sin (b-a))
        r3c1, r3c2, r3c3 = st.columns(3)
        r3c1.metric(f"z ({conf_label})", f"{z_score:.4f}")
        r3c2.metric("EE (error estándar) = σ/√N", _fmt_decimal(err_std_val))
        r3c3.metric("Margen = z · EE", _fmt_decimal(margen))

        # Fila 4: estimacion e intervalo de confianza finales
        r4c1, r4c2 = st.columns(2)
        r4c1.metric("Î (estimación) = (b−a)·f̄", _fmt_decimal(estimacion))
        r4c2.metric(
            f"IC {conf_label}",
            f"[{_fmt_decimal(ic_low)}, {_fmt_decimal(ic_up)}]",
        )


def _render_valores_intermedios_multidim(
    f_vals: np.ndarray,
    volumen: float,
    n_muestras: int,
    z_score: float,
    conf_label: str,
) -> None:
    """Muestra los valores intermedios del calculo de Monte Carlo multidimensional."""
    N = int(n_muestras)
    suma = float(np.sum(f_vals))
    f_mean = float(np.mean(f_vals))
    f_std = float(np.std(f_vals, ddof=1)) if N > 1 else 0.0
    f_var = f_std ** 2
    f_min = float(np.min(f_vals))
    f_max = float(np.max(f_vals))
    # Convencion de catedra (slides 11-12 de Caceres): EE = sigma / sqrt(N),
    # sin factor V(D). Ver montecarlo_teoria.md seccion 6.
    err_std_val = f_std / np.sqrt(N)
    margen = z_score * err_std_val
    estimacion = volumen * f_mean
    ic_low = estimacion - volumen * margen
    ic_up = estimacion + volumen * margen

    with st.container(border=True):
        st.markdown("##### 🔬 Valores intermedios del cálculo")
        st.caption(
            "Cada métrica está etiquetada con la terminología de la consigna "
            "(media muestral, varianza muestral s², desvío estándar muestral σ, "
            "error estándar EE, estimación Î, intervalo de confianza IC)."
        )

        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        r1c1.metric("N (muestras)", f"{N:,}")
        r1c2.metric("V(D) (volumen)", _fmt_decimal(volumen))
        r1c3.metric("Σ f(xᵢ)", _fmt_decimal(suma))
        r1c4.metric("f̄ (media muestral)", _fmt_decimal(f_mean))

        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        r2c1.metric("s² (varianza muestral)", _fmt_decimal(f_var))
        r2c2.metric("σ (desvío estándar muestral)", _fmt_decimal(f_std))
        r2c3.metric("min f(xᵢ)", _fmt_decimal(f_min))
        r2c4.metric("max f(xᵢ)", _fmt_decimal(f_max))

        # Componentes del IC (convencion de catedra, sin V(D))
        r3c1, r3c2, r3c3 = st.columns(3)
        r3c1.metric(f"z ({conf_label})", f"{z_score:.4f}")
        r3c2.metric("EE (error estándar) = σ/√N", _fmt_decimal(err_std_val))
        r3c3.metric("Margen = z · EE", _fmt_decimal(margen))

        # Estimacion e intervalo de confianza finales
        r4c1, r4c2 = st.columns(2)
        r4c1.metric("Î (estimación) = V(D)·f̄", _fmt_decimal(estimacion))
        r4c2.metric(
            f"IC {conf_label}",
            f"[{_fmt_decimal(ic_low)}, {_fmt_decimal(ic_up)}]",
        )


def _render_repeticiones_examen(
    sampler_fn,
    n_repeticiones: int,
    semilla_base: Optional[int],
    z_score: float,
    conf_label: str,
    n_decimales: int = 6,
) -> None:
    """Renderiza el bloque 'repetir experimento k veces' requerido por la cátedra.

    La consigna oficial (parciales A-4a, D-4 y Prueba Evaluativa 3) dice:
      «repita el experimento K veces, promedie los resultados y presente una
       tabla resumen con desviación estándar, varianza, error estándar,
       media muestral, intervalos de confianza para 95%.»

    `sampler_fn(seed_or_None) -> estimacion_float` devuelve una estimación
    fresca para cada semilla. Si `semilla_base is None`, cada corrida va sin
    semilla (aleatoria pura). Si hay semilla, cada corrida k usa
    `semilla_base + k` para mantener independencia entre corridas.
    """
    K = int(n_repeticiones)
    if K < 2:
        return

    estimaciones = np.empty(K, dtype=float)
    semillas_usadas: list = []
    for i in range(K):
        if semilla_base is None:
            seed_i: Optional[int] = None
            semillas_usadas.append("—")
        else:
            seed_i = int(semilla_base) + i
            semillas_usadas.append(seed_i)
        estimaciones[i] = float(sampler_fn(seed_i))

    media = float(np.mean(estimaciones))
    # Desvio y varianza MUESTRALES (ddof=1), como pide la consigna.
    sigma = float(np.std(estimaciones, ddof=1)) if K > 1 else 0.0
    varianza = float(np.var(estimaciones, ddof=1)) if K > 1 else 0.0
    ee = sigma / np.sqrt(K)
    margen = z_score * ee
    ic_low = media - margen
    ic_up = media + margen

    with st.container(border=True):
        st.markdown(f"##### 🔁 Repeticiones del experimento (K = {K})")
        st.caption(
            "Convención de cátedra (parciales A y D): repetir la simulación K veces, "
            "promediar las estimaciones y reportar estadísticos sobre los K resultados."
        )

        df = pd.DataFrame(
            {
                "i": np.arange(1, K + 1),
                "semilla": semillas_usadas,
                "estimación Î_i": [round(v, n_decimales) for v in estimaciones],
            }
        )
        st.dataframe(df, use_container_width=True, hide_index=True)

        r1c1, r1c2, r1c3 = st.columns(3)
        r1c1.metric("Media muestral Î̄", _fmt_decimal(media))
        r1c2.metric("Varianza muestral s²", _fmt_decimal(varianza))
        r1c3.metric("Desvío muestral s", _fmt_decimal(sigma))

        r2c1, r2c2, r2c3 = st.columns(3)
        r2c1.metric("Error estándar EE = s/√K", _fmt_decimal(ee))
        r2c2.metric(f"z ({conf_label})", f"{z_score:.4f}")
        r2c3.metric(
            f"IC {conf_label}",
            f"[{ic_low:.{n_decimales}f}, {ic_up:.{n_decimales}f}]",
        )

        st.latex(
            rf"\bar{{I}} = \frac{{1}}{{K}}\sum_{{i=1}}^{{K}} \hat{{I}}_i = "
            rf"{_fmt_latex_num(media)}"
        )
        st.latex(
            rf"s = \sqrt{{\frac{{1}}{{K-1}}\sum_{{i=1}}^{{K}} (\hat{{I}}_i - \bar{{I}})^2}} "
            rf"= {_fmt_latex_num(sigma)}"
        )
        st.latex(
            rf"\mathrm{{IC}}_{{{conf_label}}} = \bar{{I}} \pm z \cdot \frac{{s}}{{\sqrt{{K}}}} "
            rf"= [{_fmt_latex_num(ic_low)},\ {_fmt_latex_num(ic_up)}]"
        )


_ERROR_SQRT_N_MAX = 2_000_000


def _render_error_sqrt_n(
    sampler_fn,
    n_base: int,
    semilla_base: Optional[int],
    backend: RngBackend,
    n_decimales: int = 6,
    key: str = "err_sqrt_n",
) -> None:
    """Bloque 'Error ∝ 1/√N' — tabla N, 2N, 4N, 8N con ratios esperados vs observados.

    `sampler_fn(N, seed)` debe devolver (estimacion, sigma) donde sigma es
    el desvío muestral (ddof=1) de f_i evaluadas. El SE se computa acá como
    sigma/√N para aplicar la convención de cátedra.

    Útil para la demo del parcial B-4b: cuántas muestras hacen falta para
    bajar el error a la mitad (respuesta: 4N, no 2N).
    """
    N = int(n_base)
    if N < 10:
        st.info("Para este bloque necesitás N ≥ 10 muestras base.")
        return

    factores = [1, 2, 4, 8, 16, 32]
    factores_validos = [k for k in factores if k * N <= _ERROR_SQRT_N_MAX]
    if len(factores_validos) < 2:
        st.warning(
            f"N base = {N:,} es demasiado grande para ampliar a 2N, 4N, ... "
            f"sin superar el límite de {_ERROR_SQRT_N_MAX:,} muestras. "
            "Reducí N base para ver el decaimiento 1/√N."
        )
        return

    with st.container(border=True):
        st.markdown("##### 📉 Convergencia teórica — error ∝ 1/√N")
        st.markdown(
            r"El error estándar de Monte Carlo decrece como "
            r"$\mathrm{SE} \propto 1/\sqrt{N}$: para reducirlo a la mitad "
            r"**no alcanza con duplicar las muestras, hacen falta 4×**. "
            r"Esta tasa es independiente de la dimensión del dominio, por eso "
            r"Monte Carlo es preferido cuando $d \geq 4$."
        )
        st.caption(
            "El ratio SE(k·N) / SE(N) debe acercarse a 1/√k: "
            "k=2 → ≈0.707, k=4 → ≈0.500, k=8 → ≈0.354."
        )

        if semilla_base is None:
            st.caption(
                "⚠️ Semilla aleatoria: los ratios fluctúan entre corridas. "
                "Fijá semilla para una demostración reproducible."
            )

        filas = []
        se_base: Optional[float] = None
        for k in factores_validos:
            Nk = k * N
            seed_k = None if semilla_base is None else int(semilla_base) + k
            estimacion_k, sigma_k = sampler_fn(Nk, seed_k)
            se_k = float(sigma_k) / float(np.sqrt(Nk))
            if k == 1:
                se_base = se_k
            ratio_obs = (se_k / se_base) if (se_base is not None and se_base > 1e-15) else float("nan")
            ratio_esp = 1.0 / float(np.sqrt(k))
            filas.append({
                "k": k,
                "N_k = k·N": Nk,
                "estimación": round(float(estimacion_k), n_decimales),
                "σ (ddof=1)": round(float(sigma_k), n_decimales),
                "SE = σ/√N_k": round(se_k, n_decimales),
                "ratio SE(kN)/SE(N)": round(ratio_obs, 4) if not np.isnan(ratio_obs) else "—",
                "esperado 1/√k": round(ratio_esp, 4),
            })

        if se_base is None or se_base < 1e-15:
            st.info(
                "La función es (casi) constante: σ ≈ 0, no hay varianza para "
                "estudiar convergencia."
            )
            return

        df = pd.DataFrame(filas)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # --- Gráfico log-log: SE observado vs curva teórica c/√N ---
        Ns = np.array([row["N_k = k·N"] for row in filas], dtype=float)
        SEs = np.array([row["SE = σ/√N_k"] for row in filas], dtype=float)
        c = se_base * float(np.sqrt(N))
        N_theor = np.geomspace(Ns.min(), Ns.max(), 100)
        SE_theor = c / np.sqrt(N_theor)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=Ns, y=SEs, mode="markers+lines",
            name="SE observado", marker=dict(size=10, color="#4FC3F7"),
            line=dict(color="#4FC3F7", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=N_theor, y=SE_theor, mode="lines",
            name=f"Teórico c/√N (c = {c:.4g})",
            line=dict(color="#FFB74D", width=2, dash="dash"),
        ))
        fig.update_layout(
            template="plotly_dark",
            xaxis=dict(type="log", title="N (escala log)"),
            yaxis=dict(type="log", title="Error estándar (escala log)"),
            title="Decaimiento 1/√N: SE observado vs teórico",
            height=380,
            margin=dict(l=50, r=20, t=50, b=40),
        )
        st.plotly_chart(fig, use_container_width=True, key=f"{key}_plot")

        # --- Botón "Reducir error a la mitad" ---
        st.markdown("###### 🎯 Simulación del parcial B-4b")
        st.caption(
            "El parcial pregunta: si duplico N ¿cuánto baja el error? "
            "Respuesta: SE(2N)/SE(N) ≈ 0.707 (no la mitad). "
            "Para **reducirlo a la mitad** hay que usar 4N."
        )
        if semilla_base is None:
            st.caption(
                "💡 Con semilla aleatoria el ratio puede fluctuar por encima o "
                "por debajo de 0.5 — es sólo ruido muestral. Fijá semilla para "
                "una demostración limpia."
            )
        if st.button(
            "Demostrar: SE con N vs 4N (error a la mitad)",
            key=f"{key}_half",
            use_container_width=True,
        ):
            N4 = 4 * N
            if N4 > _ERROR_SQRT_N_MAX:
                st.warning(f"4N = {N4:,} excede el límite ({_ERROR_SQRT_N_MAX:,}).")
            else:
                seed_a = None if semilla_base is None else int(semilla_base)
                seed_b = None if semilla_base is None else int(semilla_base) + 99
                est_n, sig_n = sampler_fn(N, seed_a)
                est_4n, sig_4n = sampler_fn(N4, seed_b)
                se_n = float(sig_n) / float(np.sqrt(N))
                se_4n = float(sig_4n) / float(np.sqrt(N4))
                ratio_real = se_4n / se_n if se_n > 1e-15 else float("nan")
                col_a, col_b, col_c = st.columns(3)
                col_a.metric(f"SE con N = {N:,}", _fmt_decimal(se_n))
                col_b.metric(f"SE con 4N = {N4:,}", _fmt_decimal(se_4n))
                col_c.metric(
                    "Ratio SE(4N)/SE(N)",
                    f"{ratio_real:.4f}",
                    delta=f"esperado 0.5000 (1/√4)",
                    delta_color="off",
                )


def _render_pasos_1d(
    expr,
    a: float,
    b: float,
    x_vals: np.ndarray,
    f_vals: np.ndarray,
    estimacion: float,
    valor_exacto,
    conf_label: str,
    z_score: float,
    n_muestras: int,
    semilla: Optional[int],
    n_decimales: int,
    backend: RngBackend = _DEFAULT_BACKEND,
) -> None:
    """Renderiza la solucion paso a paso estilo Symbolab para Monte Carlo 1D."""
    N = int(n_muestras)
    f_mean = float(np.mean(f_vals))
    f_std = float(np.std(f_vals, ddof=1)) if N > 1 else 0.0
    # Convencion de catedra (slides 11-12): margen = z * sigma / sqrt(N), sin (b-a).
    margen = z_score * f_std / np.sqrt(N)
    ic_low = estimacion - margen
    ic_up = estimacion + margen

    # --- Paso 1: Planteo ---
    st.markdown("#### Paso 1 — Planteo del problema")
    st.markdown(
        f"Aproximar la integral definida de $f(x)$ usando Monte Carlo con "
        f"$N = {N:,}$ muestras y semilla `{_fmt_seed_short(semilla)}`."
    )
    st.latex(
        rf"I = \int_{{{_fmt_latex_num(a)}}}^{{{_fmt_latex_num(b)}}} "
        rf"{sp.latex(expr)} \, dx"
    )
    st.divider()

    # --- Paso 2: Formula del estimador ---
    st.markdown("#### Paso 2 — Fórmula del estimador Monte Carlo")
    st.latex(
        r"\hat{I} \;=\; (b - a) \cdot \frac{1}{N} \sum_{i=1}^{N} f(x_i), "
        r"\qquad x_i \sim \mathcal{U}(a, b)"
    )
    st.divider()

    # --- Paso 3: Generacion de muestras ---
    st.markdown("#### Paso 3 — Generación de muestras")
    st.markdown(
        f"Se generan $N = {N:,}$ puntos aleatorios uniformes en "
        f"$[{_fmt_latex_num(a)},\\, {_fmt_latex_num(b)}]$ con "
        f"`{_fmt_seed_call(semilla, backend)}`."
    )
    st.markdown("**Primeras 5 muestras:**")
    muestras_latex = r" \quad ".join(
        rf"x_{{{i+1}}} = {x_vals[i]:.6f}" for i in range(min(5, N))
    )
    st.latex(muestras_latex)
    st.divider()

    # --- Paso 4: Evaluacion ---
    st.markdown("#### Paso 4 — Evaluación de la función")
    st.markdown("Se evalúa $f(x)$ en cada muestra. **Primeras 5 evaluaciones:**")
    for i in range(min(5, N)):
        st.latex(
            rf"f(x_{{{i+1}}}) \;=\; f({x_vals[i]:.4f}) \;=\; {f_vals[i]:.6f}"
        )
    st.divider()

    # --- Paso 5: Promedio muestral ---
    st.markdown("#### Paso 5 — Promedio muestral de las evaluaciones")
    st.latex(
        rf"\bar{{f}} \;=\; \frac{{1}}{{N}} \sum_{{i=1}}^{{N}} f(x_i) "
        rf"\;=\; {f_mean:.6f}"
    )
    st.divider()

    # --- Paso 6: Estimacion ---
    st.markdown("#### Paso 6 — Cálculo del estimador")
    st.latex(
        rf"\hat{{I}} \;=\; (b - a) \cdot \bar{{f}} "
        rf"\;=\; ({_fmt_latex_num(b)} - {_fmt_latex_num(a)}) \cdot {f_mean:.6f} "
        rf"\;=\; {_fmt_latex_num(b - a)} \cdot {f_mean:.6f}"
    )
    st.latex(rf"\boxed{{\;\hat{{I}} \;=\; {estimacion:.{n_decimales}f}\;}}")
    st.divider()

    # --- Paso 7: Desvio estandar ---
    st.markdown("#### Paso 7 — Desvío estándar muestral")
    st.caption(
        "Nota: se usa la media de los $f(x_i)$ (denotada $\\bar{f}$), "
        "NO la de los $x_i$ — corrección de la sección 6.2 de la teoría."
    )
    st.latex(
        r"\sigma \;=\; \sqrt{\frac{1}{N - 1} "
        r"\sum_{i=1}^{N} \bigl( f(x_i) - \bar{f} \bigr)^2}"
    )
    st.latex(rf"\sigma \;=\; {f_std:.6f}")
    st.divider()

    # --- Paso 8: IC ---
    st.markdown(f"#### Paso 8 — Intervalo de confianza ({conf_label})")
    st.caption(
        "Convención de cátedra (slides 11-12): $IC = \\hat{I} \\pm z_{\\alpha/2} "
        "\\cdot \\sigma/\\sqrt{N}$, sin factor $(b - a)$."
    )
    st.latex(
        r"IC \;=\; \hat{I} \;\pm\; z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{N}}"
    )
    st.latex(
        rf"IC \;=\; {estimacion:.{n_decimales}f} \;\pm\; {z_score:.4f} "
        rf"\cdot \frac{{{f_std:.6f}}}{{\sqrt{{{N}}}}}"
    )
    st.latex(
        rf"IC \;=\; {estimacion:.{n_decimales}f} \;\pm\; {margen:.6f} "
        rf"\;=\; [\,{ic_low:.{n_decimales}f},\;\; {ic_up:.{n_decimales}f}\,]"
    )

    # --- Paso 9: Comparacion con exacto ---
    if valor_exacto is not None:
        st.divider()
        st.markdown("#### Paso 9 — Comparación con el valor exacto (SymPy)")
        err_abs = abs(estimacion - valor_exacto)
        err_rel = err_abs / abs(valor_exacto) if valor_exacto != 0 else float("inf")
        st.latex(rf"I_{{\text{{exacto}}}} \;=\; {valor_exacto:.{n_decimales}f}")
        st.latex(
            rf"\text{{Error absoluto}} \;=\; \bigl|\,\hat{{I}} - I\,\bigr| "
            rf"\;=\; \bigl|\,{estimacion:.{n_decimales}f} - {valor_exacto:.{n_decimales}f}\,\bigr| "
            rf"\;=\; {_fmt_decimal(err_abs)}"
        )
        st.latex(
            rf"\text{{Error relativo}} \;=\; \frac{{\bigl|\,\hat{{I}} - I\,\bigr|}}{{|I|}} "
            rf"\;=\; {_fmt_decimal(err_rel)}"
        )


def _render_pasos_multidim(
    expr,
    simbolos: list,
    rangos: list,
    puntos: list,
    f_vals: np.ndarray,
    volumen: float,
    estimacion: float,
    exacto,
    conf_label: str,
    z_score: float,
    n_muestras: int,
    semilla: Optional[int],
    backend: RngBackend = _DEFAULT_BACKEND,
) -> None:
    """Paso a paso estilo Symbolab para Monte Carlo multidimensional."""
    N = int(n_muestras)
    n_dims = len(simbolos)
    f_mean = float(np.mean(f_vals))
    f_std = float(np.std(f_vals, ddof=1)) if N > 1 else 0.0
    # Convencion de catedra (slides 11-12): margen = z * sigma / sqrt(N), sin V(D).
    margen = z_score * f_std / np.sqrt(N)
    ic_low = estimacion - margen
    ic_up = estimacion + margen
    nombres = [str(s) for s in simbolos]

    # --- Paso 1: Planteo ---
    st.markdown("#### Paso 1 — Planteo del problema")
    st.markdown(
        f"Aproximar la integral {n_dims}-dimensional de "
        f"$f({', '.join(nombres)})$ usando Monte Carlo con "
        f"$N = {N:,}$ muestras y semilla `{_fmt_seed_short(semilla)}`."
    )
    integral_latex = "I = "
    for i, (ai, bi) in enumerate(rangos):
        integral_latex += rf"\int_{{{_fmt_latex_num(ai)}}}^{{{_fmt_latex_num(bi)}}} "
    integral_latex += rf"{sp.latex(expr)} \, " + r"\,".join(f"d{n}" for n in nombres)
    st.latex(integral_latex)
    st.divider()

    # --- Paso 2: Formula ---
    st.markdown("#### Paso 2 — Fórmula del estimador")
    st.latex(
        r"\hat{I} \;=\; V(D) \cdot \frac{1}{N} \sum_{i=1}^{N} f(\mathbf{x}_i), "
        r"\qquad \mathbf{x}_i \sim \mathcal{U}(D)"
    )
    st.divider()

    # --- Paso 3: Volumen ---
    st.markdown("#### Paso 3 — Volumen del dominio")
    vol_factores = r" \cdot ".join(
        rf"({_fmt_latex_num(bi)} - {_fmt_latex_num(ai)})" for ai, bi in rangos
    )
    st.latex(rf"V(D) \;=\; {vol_factores} \;=\; {_fmt_latex_num(volumen)}")
    st.divider()

    # --- Paso 4: Generacion ---
    st.markdown("#### Paso 4 — Generación de muestras")
    st.markdown(
        f"Se generan $N = {N:,}$ puntos aleatorios uniformes en el dominio "
        f"con `{_fmt_seed_call(semilla, backend)}`."
    )
    st.markdown("**Primeras 5 muestras:**")
    for i in range(min(5, N)):
        coords = ", ".join(f"{puntos[d][i]:.4f}" for d in range(n_dims))
        st.latex(rf"\mathbf{{x}}_{{{i+1}}} \;=\; ({coords})")
    st.divider()

    # --- Paso 5: Evaluacion ---
    st.markdown("#### Paso 5 — Evaluación de la función")
    st.markdown("**Primeras 5 evaluaciones:**")
    for i in range(min(5, N)):
        st.latex(rf"f(\mathbf{{x}}_{{{i+1}}}) \;=\; {f_vals[i]:.6f}")
    st.divider()

    # --- Paso 6: Promedio + estimacion ---
    st.markdown("#### Paso 6 — Promedio muestral y estimación")
    st.latex(
        rf"\bar{{f}} \;=\; \frac{{1}}{{N}} \sum_{{i=1}}^{{N}} f(\mathbf{{x}}_i) "
        rf"\;=\; {f_mean:.6f}"
    )
    st.latex(
        rf"\hat{{I}} \;=\; V(D) \cdot \bar{{f}} "
        rf"\;=\; {_fmt_latex_num(volumen)} \cdot {f_mean:.6f}"
    )
    st.latex(rf"\boxed{{\;\hat{{I}} \;=\; {estimacion:.6f}\;}}")
    st.divider()

    # --- Paso 7: Sigma + IC ---
    st.markdown(f"#### Paso 7 — Intervalo de confianza ({conf_label})")
    st.caption(
        "Se usa $\\bar{f}$ (media de las evaluaciones), NO $\\bar{x}$. "
        "Convención de cátedra: $IC = \\hat{I} \\pm z_{\\alpha/2} \\cdot \\sigma/\\sqrt{N}$, sin $V(D)$."
    )
    st.latex(
        r"\sigma \;=\; \sqrt{\frac{1}{N - 1} "
        r"\sum_{i=1}^{N} \bigl( f(\mathbf{x}_i) - \bar{f} \bigr)^2} "
        rf"\;=\; {f_std:.6f}"
    )
    st.latex(
        r"IC \;=\; \hat{I} \;\pm\; z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{N}}"
    )
    st.latex(
        rf"IC \;=\; {estimacion:.6f} \;\pm\; {z_score:.4f} "
        rf"\cdot \frac{{{f_std:.6f}}}{{\sqrt{{{N}}}}} "
        rf"\;=\; [\,{ic_low:.6f},\;\; {ic_up:.6f}\,]"
    )

    # --- Paso 8: Comparacion ---
    if exacto is not None:
        st.divider()
        st.markdown("#### Paso 8 — Comparación con el valor exacto (SymPy)")
        err_abs = abs(estimacion - exacto)
        err_rel = err_abs / abs(exacto) if exacto != 0 else float("inf")
        st.latex(rf"I_{{\text{{exacto}}}} \;=\; {exacto:.6f}")
        st.latex(rf"\text{{Error absoluto}} \;=\; {_fmt_decimal(err_abs)}")
        st.latex(rf"\text{{Error relativo}} \;=\; {_fmt_decimal(err_rel)}")


def _render_pasos_comparacion(
    expr,
    a: float,
    b: float,
    x_mc: np.ndarray,
    f_mc: np.ndarray,
    mc_est: float,
    trap_est: float,
    simp_est: float,
    n_muestras: int,
    n_trapecios: int,
    semilla: Optional[int],
    valor_exacto,
    backend: RngBackend = _DEFAULT_BACKEND,
) -> None:
    """Paso a paso para los 3 metodos comparados."""
    N = int(n_muestras)
    M = int(n_trapecios)
    h = (b - a) / M

    st.markdown("#### Integral a calcular")
    st.latex(
        rf"I = \int_{{{_fmt_latex_num(a)}}}^{{{_fmt_latex_num(b)}}} "
        rf"{sp.latex(expr)} \, dx"
    )
    st.divider()

    # --- Monte Carlo ---
    st.markdown("### Método 1 — Monte Carlo")
    st.latex(
        r"\hat{I}_{MC} \;=\; (b - a) \cdot \frac{1}{N} \sum_{i=1}^{N} f(x_i)"
    )
    st.markdown(
        f"Con $N = {N:,}$ muestras ({_fmt_seed_call(semilla, backend)}). **Primeras 5 muestras:**"
    )
    for i in range(min(5, N)):
        st.latex(
            rf"x_{{{i+1}}} = {x_mc[i]:.4f}, \quad f(x_{{{i+1}}}) = {f_mc[i]:.6f}"
        )
    f_mean_mc = float(np.mean(f_mc))
    st.latex(rf"\bar{{f}} \;=\; {f_mean_mc:.6f}")
    st.latex(
        rf"\hat{{I}}_{{MC}} \;=\; {_fmt_latex_num(b - a)} \cdot {f_mean_mc:.6f} "
        rf"\;=\; \boxed{{{mc_est:.10f}}}"
    )
    st.divider()

    # --- Trapecios ---
    st.markdown("### Método 2 — Trapecios compuesto")
    st.latex(
        r"I_T \;=\; \frac{h}{2} \left[ f(x_0) + 2\sum_{i=1}^{M-1} f(x_i) + f(x_M) \right]"
    )
    st.markdown(
        f"Con $M = {M}$ subintervalos, "
        f"$h = (b - a)/M = {_fmt_decimal(h)}$."
    )
    st.latex(rf"I_T \;=\; \boxed{{{trap_est:.10f}}}")
    st.divider()

    # --- Simpson ---
    st.markdown("### Método 3 — Simpson 1/3 compuesto")
    st.latex(
        r"I_S \;=\; \frac{h}{3} \left[ f(x_0) + 4\!\!\sum_{i\,impar}\!\!f(x_i) "
        r"+ 2\!\!\sum_{i\,par}\!\!f(x_i) + f(x_M) \right]"
    )
    st.markdown(
        "Requiere $M$ par. Si ingresaste un $M$ impar se usa $M + 1$ automáticamente."
    )
    st.latex(rf"I_S \;=\; \boxed{{{simp_est:.10f}}}")

    if valor_exacto is not None:
        st.divider()
        st.markdown("#### Comparación final con el valor exacto")
        st.latex(rf"I_{{\text{{exacto}}}} \;=\; {valor_exacto:.10f}")
        for nombre, val in [("MC", mc_est), ("Trapecios", trap_est), ("Simpson", simp_est)]:
            err = abs(val - valor_exacto)
            st.latex(
                rf"|\, I_{{\text{{{nombre}}}}} - I\,| \;=\; {_fmt_decimal(err)}"
            )


def _calcular_valor_exacto_1d(expr, x_sym, a, b):
    try:
        resultado = sp.integrate(expr, (x_sym, a, b))
        valor = float(resultado.evalf())
        if not np.isfinite(valor):
            return None
        return valor
    except Exception:
        return None


def _integracion_1d():
    st.subheader("Integracion Monte Carlo 1D")

    _render_teoria_libro()

    st.latex(r"I = \int_a^b f(x)\,dx \approx \frac{b-a}{N}\sum_{i=1}^{N} f(x_i)")
    st.latex(r"x_i \sim \mathcal{U}(a, b)")

    latex = math_input(label="f(x) =", default_latex="x^{2}+\\sin(x)", key="mc1d_func")
    col1, col2 = st.columns(2)
    with col1:
        a_str = st.text_input("Limite inferior (a)", value="0", key="mc1d_a",
                              help="Acepta expresiones: pi/2, sqrt(2), -pi, e, etc.")
        b_str = st.text_input("Limite superior (b)", value="2", key="mc1d_b",
                              help="Acepta expresiones: pi/2, sqrt(2), -pi, e, etc.")
    with col2:
        n_muestras = st.number_input("Numero de muestras (N)", value=10000, min_value=10,
                                     max_value=10_000_000, step=1000, key="mc1d_n")
        n_decimales = st.number_input("Precision (decimales)", value=4, min_value=1,
                                      max_value=15, key="mc1d_tol")
        conf_label, z_score = _confianza_selector("mc1d_conf")

    backend, semilla = _rng_selector("mc1d_rng")

    with st.expander("🔁 Modo examen: repetir el experimento K veces", expanded=False):
        st.caption(
            "Parciales A-4a y D-4 piden «repita el experimento K veces, promedie los "
            "resultados y presente una tabla con desviación estándar, varianza, error "
            "estándar, media muestral e IC 95%»."
        )
        rep_on_1d = st.checkbox(
            "Activar repeticiones", value=False, key="mc1d_rep_on",
            help="Ejecuta K corridas independientes cambiando la semilla de cada una.",
        )
        k_rep_1d = st.number_input(
            "Número de repeticiones (K)", value=10, min_value=2, max_value=200,
            step=1, key="mc1d_rep_k",
        )

    with st.expander("🎯 Valor exacto (opcional, simbólico)", expanded=False):
        st.caption(
            "Si conocés el valor exacto, ingresalo acá para usarlo como "
            "referencia. Acepta `pi`, `pi**2/6`, `(exp(2)-1)**2/exp(2)`, "
            "`1/3`, `sqrt(2)`, etc. Vacío = se intenta con SymPy automático."
        )
        exacto_1d_str = st.text_input(
            "Valor exacto I*", value="", key="mc1d_exacto",
            placeholder="dejar vacío para usar SymPy automático",
        )

    tolerancia = 10 ** (-n_decimales)
    st.latex(rf"\text{{Precision: }} 10^{{-{n_decimales}}} = {tolerancia}")

    if st.button("Calcular", key="mc1d_calc"):
        x_sym = sp.Symbol("x")
        expr, f_np = parse_latex(latex, [x_sym])
        if expr is None:
            return
        a = parse_expr_to_float(a_str, "a")
        b = parse_expr_to_float(b_str, "b")
        if a is None or b is None:
            return

        _seed_rng(semilla, backend)
        x_vals = _sample_uniform(a, b, int(n_muestras), backend)
        f_vals = f_np(x_vals)

        estimacion = (b - a) * np.mean(f_vals)
        # Convencion de catedra (slides 11-12 de Caceres): EE = sigma / sqrt(n),
        # sin factor (b - a). Ver docs/DESCRIPCION.md y montecarlo_teoria.md seccion 6.
        error_std = np.std(f_vals, ddof=1) / np.sqrt(len(f_vals))

        ic_low_final = estimacion - z_score * error_std
        ic_up_final = estimacion + z_score * error_std

        exacto_manual_1d = (exacto_1d_str or "").strip()
        valor_exacto = None
        if exacto_manual_1d:
            valor_exacto = parse_expr_to_float(exacto_manual_1d, "Valor exacto I*")
        if valor_exacto is None:
            valor_exacto = _calcular_valor_exacto_1d(expr, x_sym, a, b)

        tab_resumen, tab_pasos, tab_viz, tab_conv = st.tabs(
            ["📊 Resumen", "🧮 Paso a paso", "📈 Visualizaciones", "📉 Convergencia 1/√N"]
        )

        # =================== TAB 1: RESUMEN ===================
        with tab_resumen:
            with st.container(border=True):
                st.markdown("##### Resultados")
                col_r1, col_r2, col_r3 = st.columns(3)
                col_r1.metric("Estimacion", f"{estimacion:.{n_decimales}f}")
                col_r2.metric("Error estandar", _fmt_decimal(error_std))
                col_r3.metric(
                    f"IC {conf_label}",
                    f"[{ic_low_final:.{n_decimales}f}, {ic_up_final:.{n_decimales}f}]",
                )

                if valor_exacto is not None:
                    st.divider()
                    col_e1, col_e2, col_e3 = st.columns(3)
                    col_e1.metric("Valor exacto (SymPy)", f"{valor_exacto:.{n_decimales}f}")
                    err_abs = error_absoluto(estimacion, valor_exacto)
                    err_rel = error_relativo(estimacion, valor_exacto)
                    col_e2.metric("Error absoluto", _fmt_decimal(err_abs))
                    col_e3.metric("Error relativo", _fmt_decimal(err_rel))

            _render_valores_intermedios_1d(
                f_vals=f_vals,
                a=a,
                b=b,
                n_muestras=n_muestras,
                z_score=z_score,
                conf_label=conf_label,
            )

            if rep_on_1d:
                def _sampler_1d(seed_i: Optional[int]) -> float:
                    _seed_rng(seed_i, backend)
                    xs = _sample_uniform(a, b, int(n_muestras), backend)
                    fs = f_np(xs)
                    return float((b - a) * np.mean(fs))

                _render_repeticiones_examen(
                    sampler_fn=_sampler_1d,
                    n_repeticiones=int(k_rep_1d),
                    semilla_base=semilla,
                    z_score=z_score,
                    conf_label=conf_label,
                    n_decimales=int(n_decimales),
                )

        # =================== TAB 2: PASO A PASO ===================
        with tab_pasos:
            _render_pasos_1d(
                expr=expr,
                a=a,
                b=b,
                x_vals=x_vals,
                f_vals=f_vals,
                estimacion=estimacion,
                valor_exacto=valor_exacto,
                conf_label=conf_label,
                z_score=z_score,
                n_muestras=n_muestras,
                semilla=semilla,
                n_decimales=n_decimales,
                backend=backend,
            )

        # =================== TAB 3: VISUALIZACIONES ===================
        with tab_viz:
            _render_visualizaciones_1d(
                x_vals=x_vals,
                f_vals=f_vals,
                f_np=f_np,
                a=a,
                b=b,
                n_muestras=n_muestras,
                n_decimales=n_decimales,
                tolerancia=tolerancia,
                valor_exacto=valor_exacto,
                z_score=z_score,
            )

        # =================== TAB 4: CONVERGENCIA 1/sqrt(N) ===================
        with tab_conv:
            def _sampler_conv_1d(Nk: int, seed_k: Optional[int]) -> tuple:
                _seed_rng(seed_k, backend)
                xs = _sample_uniform(a, b, int(Nk), backend)
                fs = f_np(xs)
                est = float((b - a) * np.mean(fs))
                sigma = float(np.std(fs, ddof=1)) if Nk > 1 else 0.0
                # Convención cátedra: sigma del integrando, SE = sigma/sqrt(N).
                # Para que el ratio SE(kN)/SE(N) refleje la reducción del error
                # en la estimación (b-a)*mean, escalamos sigma por (b-a).
                return est, sigma * abs(b - a)

            _render_error_sqrt_n(
                sampler_fn=_sampler_conv_1d,
                n_base=int(n_muestras),
                semilla_base=semilla,
                backend=backend,
                n_decimales=int(n_decimales),
                key="mc1d_conv",
            )

        st.session_state["mc1d_resultado"] = estimacion


def _render_visualizaciones_1d(
    x_vals, f_vals, f_np, a, b, n_muestras, n_decimales, tolerancia, valor_exacto, z_score
):
    """Renderiza tabla de iteraciones + scatter + convergencia para 1D."""
    # Tabla de iteraciones
    st.markdown("#### Tabla de iteraciones")
    paso = max(1, int(n_muestras) // 200)
    indices = list(range(paso, int(n_muestras) + 1, paso))
    if indices[-1] != int(n_muestras):
        indices.append(int(n_muestras))

    cumsum = np.cumsum(f_vals)
    filas = []
    tolerancia_alcanzada_idx = None
    prev_est = None

    for idx in indices:
        est = (b - a) * cumsum[idx - 1] / idx
        f_val_actual = f_vals[idx - 1]
        err_abs = error_absoluto(est, valor_exacto) if valor_exacto is not None else abs(est - prev_est) if prev_est is not None else 0.0
        err_rel = error_relativo(est, valor_exacto) if valor_exacto is not None else (abs(est - prev_est) / abs(est) if prev_est is not None and est != 0 else 0.0)
        filas.append({
            "n": idx,
            "valor_actual": est,
            "f(valor)": f_val_actual,
            "error_absoluto": err_abs,
            "error_relativo": err_rel,
        })
        if tolerancia_alcanzada_idx is None and err_abs < tolerancia:
            tolerancia_alcanzada_idx = len(filas) - 1
        prev_est = est

    df_iter = pd.DataFrame(filas)

    def _resaltar_tolerancia(row):
        if tolerancia_alcanzada_idx is not None and row.name == tolerancia_alcanzada_idx:
            return ["background-color: #1a472a; color: #00ff88"] * len(row)
        return [""] * len(row)

    st.dataframe(
        df_iter.style.apply(_resaltar_tolerancia, axis=1).format({
            "valor_actual": f"{{:.{n_decimales}f}}",
            "f(valor)": f"{{:.{n_decimales}f}}",
            "error_absoluto": _fmt_decimal,
            "error_relativo": _fmt_decimal,
        }),
        use_container_width=True,
        height=400,
    )

    if tolerancia_alcanzada_idx is not None:
        st.success(f"Tolerancia alcanzada en n = {filas[tolerancia_alcanzada_idx]['n']}")

    # Grafico scatter
    st.markdown("#### Puntos Monte Carlo")
    y_min_f = min(0, float(np.min(f_np(np.linspace(a, b, 200)))))
    y_max_f = float(np.max(f_np(np.linspace(a, b, 200)))) * 1.1

    max_plot = min(int(n_muestras), 5000)
    plot_x = x_vals[:max_plot]
    plot_y = _sample_uniform(y_min_f, y_max_f, max_plot)
    f_at_plot_x = f_np(plot_x)
    dentro = (plot_y >= 0) & (plot_y <= f_at_plot_x) | (plot_y <= 0) & (plot_y >= f_at_plot_x)

    fig_scatter = plot_scatter_montecarlo(plot_x, plot_y, dentro, f_np, a, b)
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Grafico de convergencia
    st.markdown("#### Convergencia")
    potencias = []
    p = 10
    while p <= int(n_muestras):
        potencias.append(p)
        p *= 10
    if potencias[-1] != int(n_muestras):
        potencias.append(int(n_muestras))

    est_conv = []
    ic_lows = []
    ic_ups = []
    for n_p in potencias:
        sub_vals = f_vals[:n_p]
        est_p = (b - a) * np.mean(sub_vals)
        # Convencion de catedra: EE = sigma / sqrt(n), sin (b - a).
        std_p = np.std(sub_vals, ddof=1) / np.sqrt(n_p)
        est_conv.append(est_p)
        ic_lows.append(est_p - z_score * std_p)
        ic_ups.append(est_p + z_score * std_p)

    fig_conv = plot_convergencia(potencias, est_conv, ic_lows, ic_ups, valor_exacto)
    st.plotly_chart(fig_conv, use_container_width=True)


_PRESETS_MULTIDIM = {
    "A-4: ∫∫ e^(x−y) en [0,2]²": {
        "descripcion": (
            "Parcial A-4. Aproximar I = ∫₀² ∫₀² e^(x−y) dy dx con "
            "`np.random.seed(0)`, n=10000. Valor exacto: (e²−1)²/e² ≈ 4.0384."
        ),
        "state": {
            "mc_nd_dims": 2,
            "mc_nd_func": "e^{x-y}",
            "mc_nd_x_min": "0", "mc_nd_x_max": "2",
            "mc_nd_y_min": "0", "mc_nd_y_max": "2",
            "mc_nd_n": 10000,
            "mc_nd_rng_backend": "numpy (parcial)",
            "mc_nd_rng_use_seed": True,
            "mc_nd_rng_seed_val": 0,
            "mc_nd_conf_preset": "95%",
            "mc_nd_rep_on": True,
            "mc_nd_rep_k": 10,
            "mc_nd_exacto": "(exp(2)-1)**2/exp(2)",
        },
    },
    "Variante: ∫∫ x·e^y en [0,1]×[1,3]": {
        "descripcion": (
            "Variante rectangular. Aproximar I = ∫₀¹ ∫₁³ x·e^y dy dx con "
            "`np.random.seed(0)`, n=10000. Valor exacto: (e³−e)/2 ≈ 8.6836."
        ),
        "state": {
            "mc_nd_dims": 2,
            "mc_nd_func": "x\\cdot e^{y}",
            "mc_nd_x_min": "0", "mc_nd_x_max": "1",
            "mc_nd_y_min": "1", "mc_nd_y_max": "3",
            "mc_nd_n": 10000,
            "mc_nd_rng_backend": "numpy (parcial)",
            "mc_nd_rng_use_seed": True,
            "mc_nd_rng_seed_val": 0,
            "mc_nd_conf_preset": "95%",
            "mc_nd_rep_on": True,
            "mc_nd_rep_k": 10,
            "mc_nd_exacto": "(exp(3)-exp(1))/2",
        },
    },
}


def _apply_preset(state_dict: dict) -> None:
    """Aplica un preset sobreescribiendo st.session_state y fuerza rerun."""
    for k, v in state_dict.items():
        st.session_state[k] = v
    st.rerun()


def _render_presets_parcial(presets: dict, key_prefix: str) -> None:
    """Renderiza un expander con botones que aplican presets de parcial."""
    with st.expander("🎓 Presets de parcial", expanded=False):
        st.caption(
            "Clickeá un preset para autollenar todos los campos con los valores "
            "exactos de la consigna. Después ajustá si querés y calculá."
        )
        for label, cfg in presets.items():
            col_btn, col_desc = st.columns([1, 3])
            with col_btn:
                if st.button(
                    label,
                    key=f"{key_prefix}_preset_{abs(hash(label))}",
                    use_container_width=True,
                ):
                    _apply_preset(cfg["state"])
            with col_desc:
                st.caption(cfg["descripcion"])


def _detectar_intersecciones_curvas(
    f_expr,
    g_expr,
    x_sym,
    key_a: str,
    key_b: str,
) -> None:
    """Resuelve f(x)=g(x), muestra procedimiento y verifica cada raíz.

    Primero intenta `sympy.solve` (simbólico). Si no encuentra nada o falla,
    cae a `scipy.optimize.brentq` sobre una grilla [-10, 10]. Cada raíz se
    verifica numéricamente mostrando f(xᵢ) y g(xᵢ) para que el alumno pueda
    corroborarlo a mano.
    """
    diff_expr = sp.simplify(f_expr - g_expr)

    with st.container(border=True):
        st.markdown("##### 🔍 Intersecciones de f(x) = g(x)")

        st.markdown("**Paso 1.** Planteamos la ecuación:")
        st.latex(
            f"f(x) = g(x) \\implies {sp.latex(f_expr)} = {sp.latex(g_expr)}"
        )

        st.markdown("**Paso 2.** Reordenamos igualando a cero:")
        st.latex(
            f"f(x) - g(x) = 0 \\implies {sp.latex(diff_expr)} = 0"
        )

        try:
            factored = sp.factor(diff_expr)
            if factored != diff_expr:
                st.markdown("**Paso 3.** Factorizamos:")
                st.latex(f"{sp.latex(factored)} = 0")
        except Exception:
            pass

        raices_reales: list = []
        fuente = "sympy.solve (simbólico)"
        try:
            soluciones = sp.solve(diff_expr, x_sym, dict=False)
        except Exception as exc:
            st.warning(
                f"sympy.solve falló ({exc}). Uso búsqueda numérica de fallback."
            )
            soluciones = []

        for sol in soluciones:
            try:
                sol_num = complex(sol)
                if abs(sol_num.imag) < 1e-9:
                    raices_reales.append((sol, float(sol_num.real)))
            except (TypeError, ValueError):
                continue

        if not raices_reales:
            fuente = "scipy.optimize.brentq (numérico, grilla [-10, 10])"
            try:
                f_diff_num = sp.lambdify(x_sym, diff_expr, "numpy")
                xs_grid = np.linspace(-10.0, 10.0, 2001)
                ys_grid = np.asarray(f_diff_num(xs_grid), dtype=float)
                for i in range(len(xs_grid) - 1):
                    y0, y1 = ys_grid[i], ys_grid[i + 1]
                    if not (np.isfinite(y0) and np.isfinite(y1)):
                        continue
                    if y0 == 0.0:
                        raices_reales.append((sp.Float(xs_grid[i]), float(xs_grid[i])))
                    elif y0 * y1 < 0:
                        try:
                            r = optimize.brentq(f_diff_num, xs_grid[i], xs_grid[i + 1])
                            raices_reales.append((sp.Float(r), float(r)))
                        except Exception:
                            pass
            except Exception as exc:
                st.error(f"Búsqueda numérica también falló: {exc}")
                return

        if not raices_reales:
            st.error(
                "No encontré intersecciones reales. Revisá las expresiones de "
                "f y g, o cargá los límites a mano."
            )
            return

        raices_reales.sort(key=lambda par: par[1])

        st.markdown(f"**Paso 4.** Raíces reales encontradas con {fuente}:")

        f_num_ev = sp.lambdify(x_sym, f_expr, "numpy")
        g_num_ev = sp.lambdify(x_sym, g_expr, "numpy")
        filas = []
        for i, (sim, num) in enumerate(raices_reales, start=1):
            try:
                f_at = float(f_num_ev(num))
                g_at = float(g_num_ev(num))
                dif = abs(f_at - g_at)
            except Exception:
                f_at, g_at, dif = float("nan"), float("nan"), float("nan")
            filas.append({
                "i": i,
                "xᵢ (simbólico)": str(sim),
                "xᵢ (decimal)": round(num, 6),
                "f(xᵢ)": round(f_at, 6),
                "g(xᵢ)": round(g_at, 6),
                "|f − g|": f"{dif:.2e}",
            })
        df_raices = pd.DataFrame(filas)
        st.dataframe(df_raices, use_container_width=True, hide_index=True)

        st.caption(
            "**Verificación a mano**: en cada raíz xᵢ debe cumplirse "
            "f(xᵢ) = g(xᵢ), o sea |f − g| ≈ 0. Los valores no son exactamente "
            "cero por redondeo de punto flotante (10⁻¹⁰ o menos es correcto)."
        )

        n_raices = len(raices_reales)
        if n_raices == 2:
            a_sim = str(raices_reales[0][0])
            b_sim = str(raices_reales[1][0])
            st.success(
                f"Se encontraron **2 intersecciones**: a = `{a_sim}`, b = `{b_sim}`. "
                "Clickeá el botón para autocompletar los límites."
            )
            if st.button("✅ Usar como [a, b]", key="mrc_apply_intersec"):
                st.session_state[key_a] = a_sim
                st.session_state[key_b] = b_sim
                st.rerun()
        elif n_raices == 1:
            st.warning(
                f"Solo encontré 1 intersección (x = `{raices_reales[0][0]}`). "
                "Necesitás 2 puntos para definir [a, b] — cargá el otro límite "
                "a mano según tu problema (p.ej. una asíntota o un extremo de dominio)."
            )
        else:
            pares_legibles = ", ".join(f"`{r[0]}`" for r in raices_reales)
            st.warning(
                f"Encontré {n_raices} intersecciones: {pares_legibles}. "
                "Elegí a mano el par que corresponde al área que querés calcular "
                "(habitualmente dos raíces consecutivas)."
            )


def _integracion_multidimensional():
    st.subheader("Integracion Monte Carlo Multidimensional")

    _render_teoria_libro()

    st.latex(r"I = \int_D f(\mathbf{x})\,d\mathbf{x} \approx \frac{V(D)}{N}\sum_{i=1}^{N} f(\mathbf{x}_i)")

    _render_presets_parcial(_PRESETS_MULTIDIM, "mc_nd")

    n_dims = st.radio("Dimensiones", [2, 3], horizontal=True, key="mc_nd_dims")

    with st.expander("📖 ¿Cómo leer `dy dx` vs `dx dy`? (ayudamemoria)", expanded=False):
        st.markdown(
            "**Regla:** los diferenciales se aparean con los límites **en orden inverso**. "
            "El diferencial *más interno* (pegado a la función) se corresponde con el "
            "par de límites *más cercano al integrando* (el último escrito)."
        )
        st.latex(
            r"\int_{x=a_x}^{b_x}\,\underbrace{\int_{y=a_y}^{b_y} f(x,y)\,dy}_"
            r"{\text{interior}}\,dx"
        )
        st.markdown(
            "- **`dy` (interno)** ↔ segundo par de límites ↔ cargalo como **y min / y max**\n"
            "- **`dx` (externo)** ↔ primer par de límites ↔ cargalo como **x min / x max**\n\n"
            "Para Monte Carlo el orden `dy dx` vs `dx dy` **no importa** "
            "(teorema de Fubini sobre dominios rectangulares). Lo único que importa "
            "es qué rango va con qué variable."
        )
        st.markdown("**Ejemplos concretos:**")
        st.markdown(
            """
| Consigna | Variable x | Variable y | V(D) |
|---|---|---|---|
| `∫₀³ ∫₀¹ f(x,y) dy dx` | [0, 3] | [0, 1] | 3 |
| `∫₀³ ∫₀¹ f(x,y) dx dy` | [0, 1] | [0, 3] | 3 |
| `∫₀² ∫₀² eˣ⁻ʸ dy dx` (A-4) | [0, 2] | [0, 2] | 4 |
| `∫₁² ∫₀^π sin(xy) dx dy` | [0, π] | [1, 2] | π |
            """
        )
        st.caption(
            "💡 **Truco visual**: si la consigna tiene subíndices `∫_{x=a}^{b}`, "
            "no hace falta regla — leé directamente. Si no los tiene, usá la regla "
            "del orden inverso."
        )

    if n_dims == 2:
        latex = math_input(label="f(x,y) =", default_latex="x^{2}+y^{2}", key="mc_nd_func")
    else:
        latex = math_input(label="f(x,y,z) =", default_latex="x^{2}+y^{2}+z^{2}", key="mc_nd_func")

    cols = st.columns(n_dims)
    rangos_str = []
    nombres = ["x", "y", "z"]
    for i in range(n_dims):
        with cols[i]:
            ai_str = st.text_input(f"{nombres[i]} min", value="0", key=f"mc_nd_{nombres[i]}_min",
                                   help="Acepta: pi/2, sqrt(2), etc.")
            bi_str = st.text_input(f"{nombres[i]} max", value="1", key=f"mc_nd_{nombres[i]}_max",
                                   help="Acepta: pi/2, sqrt(2), etc.")
            rangos_str.append((ai_str, bi_str))

    n_muestras = st.number_input("Numero de muestras (N)", value=10000, min_value=100,
                                 max_value=10_000_000, step=1000, key="mc_nd_n",
                                 help="Parciales piden literalmente n = 10000.")
    conf_label, z_score = _confianza_selector("mc_nd_conf")
    backend, semilla = _rng_selector("mc_nd_rng")

    with st.expander("🔁 Modo examen: repetir el experimento K veces", expanded=False):
        st.caption(
            "Parciales A-4a y D-4 piden «repita el experimento K veces, promedie los "
            "resultados y presente una tabla con desviación estándar, varianza, error "
            "estándar, media muestral e IC 95%»."
        )
        rep_on_nd = st.checkbox(
            "Activar repeticiones", value=False, key="mc_nd_rep_on",
            help="Ejecuta K corridas independientes cambiando la semilla de cada una.",
        )
        k_rep_nd = st.number_input(
            "Número de repeticiones (K)", value=10, min_value=2, max_value=200,
            step=1, key="mc_nd_rep_k",
        )

    with st.expander("🎯 Valor exacto (opcional, simbólico)", expanded=False):
        st.caption(
            "Si conocés el valor exacto analítico, ingresalo acá y se usará "
            "como referencia (en lugar del cálculo automático con SymPy). "
            "Acepta expresiones simbólicas: `pi**2/6`, `(exp(2)-1)**2/exp(2)`, "
            "`1/3`, `sqrt(2)`, etc."
        )
        exacto_nd_str = st.text_input(
            "Valor exacto I*", value="", key="mc_nd_exacto",
            placeholder="dejar vacío para usar SymPy automático",
        )

    if st.button("Calcular", key="mc_nd_calc"):
        simbolos = [sp.Symbol(nombres[i]) for i in range(n_dims)]
        expr, f_np = parse_latex(latex, simbolos)
        if expr is None:
            return

        rangos = []
        for i, (ai_str, bi_str) in enumerate(rangos_str):
            ai = parse_expr_to_float(ai_str, f"{nombres[i]} min")
            bi = parse_expr_to_float(bi_str, f"{nombres[i]} max")
            if ai is None or bi is None:
                return
            rangos.append((ai, bi))

        _seed_rng(semilla, backend)
        puntos = [
            _sample_uniform(rangos[i][0], rangos[i][1], int(n_muestras), backend)
            for i in range(n_dims)
        ]

        f_vals = f_np(*puntos)
        volumen = 1.0
        for ai, bi in rangos:
            volumen *= (bi - ai)

        estimacion = volumen * np.mean(f_vals)
        # Convencion de catedra (slides 11-12 de Caceres): EE = sigma / sqrt(n),
        # sin factor V(D). Ver montecarlo_teoria.md seccion 6.
        error_std = np.std(f_vals, ddof=1) / np.sqrt(int(n_muestras))
        ic_low = estimacion - z_score * error_std
        ic_up = estimacion + z_score * error_std

        # Valor exacto: override manual si el usuario escribió algo, sino SymPy automático.
        exacto = None
        exacto_manual_str = (exacto_nd_str or "").strip()
        if exacto_manual_str:
            exacto = parse_expr_to_float(exacto_manual_str, "Valor exacto I*")
        if exacto is None:
            try:
                limites = [(simbolos[i], rangos[i][0], rangos[i][1]) for i in range(n_dims)]
                exacto_sym = sp.integrate(expr, *limites)
                exacto_val = float(exacto_sym.evalf())
                if np.isfinite(exacto_val):
                    exacto = exacto_val
            except Exception:
                pass

        tab_resumen, tab_pasos, tab_viz, tab_conv = st.tabs(
            ["📊 Resumen", "🧮 Paso a paso", "📈 Visualizaciones", "📉 Convergencia 1/√N"]
        )

        # =================== TAB 1: RESUMEN ===================
        with tab_resumen:
            with st.container(border=True):
                st.markdown("##### Resultados")
                col_r1, col_r2, col_r3 = st.columns(3)
                col_r1.metric("Estimacion", f"{estimacion:.6f}")
                col_r2.metric("Error estandar", _fmt_decimal(error_std))
                col_r3.metric(f"IC {conf_label}", f"[{ic_low:.6f}, {ic_up:.6f}]")

                if exacto is not None:
                    st.divider()
                    col_e1, col_e2, col_e3 = st.columns(3)
                    col_e1.metric("Valor exacto (SymPy)", f"{exacto:.6f}")
                    col_e2.metric("Error absoluto", _fmt_decimal(error_absoluto(estimacion, exacto)))
                    col_e3.metric("Error relativo", _fmt_decimal(error_relativo(estimacion, exacto)))

            _render_valores_intermedios_multidim(
                f_vals=f_vals,
                volumen=volumen,
                n_muestras=n_muestras,
                z_score=z_score,
                conf_label=conf_label,
            )

            if rep_on_nd:
                def _sampler_nd(seed_i: Optional[int]) -> float:
                    _seed_rng(seed_i, backend)
                    pts = [
                        _sample_uniform(rangos[i][0], rangos[i][1], int(n_muestras), backend)
                        for i in range(n_dims)
                    ]
                    fs = f_np(*pts)
                    return float(volumen * np.mean(fs))

                _render_repeticiones_examen(
                    sampler_fn=_sampler_nd,
                    n_repeticiones=int(k_rep_nd),
                    semilla_base=semilla,
                    z_score=z_score,
                    conf_label=conf_label,
                    n_decimales=6,
                )

        # =================== TAB 2: PASO A PASO ===================
        with tab_pasos:
            _render_pasos_multidim(
                expr=expr,
                simbolos=simbolos,
                rangos=rangos,
                puntos=puntos,
                f_vals=f_vals,
                volumen=volumen,
                estimacion=estimacion,
                exacto=exacto,
                conf_label=conf_label,
                z_score=z_score,
                n_muestras=n_muestras,
                semilla=semilla,
                backend=backend,
            )

        # =================== TAB 3: VISUALIZACIONES ===================
        with tab_viz:
            if n_dims == 2:
                st.markdown("#### Visualizacion 3D")
                max_plot = min(int(n_muestras), 5000)
                x_plot = puntos[0][:max_plot]
                y_plot = puntos[1][:max_plot]
                z_plot = f_vals[:max_plot]

                z_random = _sample_uniform(0, float(np.max(z_plot)) * 1.1, max_plot)
                dentro = z_random <= z_plot

                fig_3d = plot_scatter_3d(x_plot, y_plot, z_random, dentro)
                fig_3d.update_layout(scene=dict(
                    xaxis_title="x", yaxis_title="y", zaxis_title="z"
                ))
                st.plotly_chart(fig_3d, use_container_width=True)
            else:
                st.info(
                    "La visualizacion 3D solo aplica para dominios de 2 dimensiones. "
                    "Para 3D, usa el tab 'Paso a paso' para ver las muestras generadas."
                )

        # =================== TAB 4: CONVERGENCIA 1/sqrt(N) ===================
        with tab_conv:
            def _sampler_conv_nd(Nk: int, seed_k: Optional[int]) -> tuple:
                _seed_rng(seed_k, backend)
                pts = [
                    _sample_uniform(rangos[i][0], rangos[i][1], int(Nk), backend)
                    for i in range(n_dims)
                ]
                fs = f_np(*pts)
                est = float(volumen * np.mean(fs))
                sigma = float(np.std(fs, ddof=1)) if Nk > 1 else 0.0
                return est, sigma * abs(volumen)

            _render_error_sqrt_n(
                sampler_fn=_sampler_conv_nd,
                n_base=int(n_muestras),
                semilla_base=semilla,
                backend=backend,
                n_decimales=6,
                key="mcnd_conv",
            )


def _comparacion_metodos():
    st.subheader("Comparacion de Metodos")

    _render_teoria_libro()

    st.latex(r"\text{Monte Carlo vs Trapecios vs Simpson}")

    latex = math_input(label="f(x) =", default_latex="x^{2}+\\sin(x)", key="mc_comp_func")
    col1, col2 = st.columns(2)
    with col1:
        a_str = st.text_input("Limite inferior (a)", value="0", key="mc_comp_a",
                              help="Acepta expresiones: pi/2, sqrt(2), -pi, e, etc.")
        b_str = st.text_input("Limite superior (b)", value="2", key="mc_comp_b",
                              help="Acepta expresiones: pi/2, sqrt(2), -pi, e, etc.")
    with col2:
        n_muestras = st.number_input("N (Monte Carlo)", value=100000, min_value=100,
                                     max_value=10_000_000, key="mc_comp_n")
        n_trapecios = st.number_input("N (subdivisiones Trapecios/Simpson)", value=1000,
                                      min_value=2, max_value=1_000_000, key="mc_comp_nt")

    backend, semilla = _rng_selector("mc_comp_rng")

    if st.button("Comparar", key="mc_comp_calc"):
        x_sym = sp.Symbol("x")
        expr, f_np = parse_latex(latex, [x_sym])
        if expr is None:
            return
        a = parse_expr_to_float(a_str, "a")
        b = parse_expr_to_float(b_str, "b")
        if a is None or b is None:
            return

        valor_exacto = _calcular_valor_exacto_1d(expr, x_sym, a, b)

        resultados = {}

        # Monte Carlo con backend elegido (numpy o stdlib).
        t0 = time.perf_counter()
        _seed_rng(semilla, backend)
        x_mc = _sample_uniform(a, b, int(n_muestras), backend)
        f_mc = f_np(x_mc)
        mc_est = (b - a) * np.mean(f_mc)
        t_mc = time.perf_counter() - t0
        resultados["Monte Carlo"] = (mc_est, t_mc)

        # Trapecios
        t0 = time.perf_counter()
        x_trap = np.linspace(a, b, int(n_trapecios) + 1)
        trap_est = float(np.trapezoid(f_np(x_trap), x_trap))
        t_trap = time.perf_counter() - t0
        resultados["Trapecios"] = (trap_est, t_trap)

        # Simpson
        t0 = time.perf_counter()
        n_simp = int(n_trapecios) if int(n_trapecios) % 2 == 0 else int(n_trapecios) + 1
        x_simp = np.linspace(a, b, n_simp + 1)
        simp_est = float(integrate.simpson(f_np(x_simp), x=x_simp))
        t_simp = time.perf_counter() - t0
        resultados["Simpson"] = (simp_est, t_simp)

        # Tabla
        filas = []
        for metodo, (res, t) in resultados.items():
            fila = {
                "Metodo": metodo,
                "Resultado": res,
                "Tiempo (s)": t,
            }
            if valor_exacto is not None:
                fila["Error absoluto"] = error_absoluto(res, valor_exacto)
                fila["Error relativo"] = error_relativo(res, valor_exacto)
            filas.append(fila)

        df = pd.DataFrame(filas)

        tab_resumen, tab_pasos, tab_viz, tab_conv = st.tabs(
            ["📊 Resumen", "🧮 Paso a paso", "📈 Visualizaciones", "📉 Convergencia 1/√N"]
        )

        # =================== TAB 1: RESUMEN ===================
        with tab_resumen:
            with st.container(border=True):
                st.markdown("##### Resultados")
                if valor_exacto is not None:
                    st.metric("Valor exacto (SymPy)", f"{valor_exacto:.10f}")

                formato = {"Resultado": "{:.10f}", "Tiempo (s)": "{:.6f}"}
                if valor_exacto is not None:
                    formato["Error absoluto"] = _fmt_decimal
                    formato["Error relativo"] = _fmt_decimal

                st.dataframe(df.style.format(formato), use_container_width=True)

        # =================== TAB 2: PASO A PASO ===================
        with tab_pasos:
            _render_pasos_comparacion(
                expr=expr,
                a=a,
                b=b,
                x_mc=x_mc,
                f_mc=f_mc,
                mc_est=mc_est,
                trap_est=trap_est,
                simp_est=simp_est,
                n_muestras=n_muestras,
                n_trapecios=n_trapecios,
                semilla=semilla,
                valor_exacto=valor_exacto,
                backend=backend,
            )

        # =================== TAB 3: VISUALIZACIONES ===================
        with tab_viz:
            # Grafico de barras
            if valor_exacto is not None:
                metodos_list = list(resultados.keys())
                errores_list = [error_absoluto(resultados[m][0], valor_exacto) for m in metodos_list]
                fig_bar = plot_comparacion_barras(metodos_list, None, errores_list)
                st.plotly_chart(fig_bar, use_container_width=True)

            # Grafico de tiempos
            st.markdown("#### Tiempo de computo")
            fig_t = go.Figure()
            fig_t.add_trace(go.Bar(
                x=list(resultados.keys()),
                y=[resultados[m][1] for m in resultados],
                marker_color=["#00d4ff", "#ffd700", "#ff6b6b"],
                text=[f"{resultados[m][1]:.4f}s" for m in resultados],
                textposition="auto",
            ))
            fig_t.update_layout(
                template="plotly_dark",
                yaxis_title="Tiempo (segundos)",
                margin=dict(l=40, r=20, t=30, b=40),
            )
            st.plotly_chart(fig_t, use_container_width=True)

        # =================== TAB 4: CONVERGENCIA 1/sqrt(N) ===================
        with tab_conv:
            st.caption(
                "Esta demostración solo aplica al método Monte Carlo: los métodos "
                "deterministas (Trapecios, Simpson) tienen otra tasa de convergencia."
            )

            def _sampler_conv_comp(Nk: int, seed_k: Optional[int]) -> tuple:
                _seed_rng(seed_k, backend)
                xs = _sample_uniform(a, b, int(Nk), backend)
                fs = f_np(xs)
                est = float((b - a) * np.mean(fs))
                sigma = float(np.std(fs, ddof=1)) if Nk > 1 else 0.0
                return est, sigma * abs(b - a)

            _render_error_sqrt_n(
                sampler_fn=_sampler_conv_comp,
                n_base=int(n_muestras),
                semilla_base=semilla,
                backend=backend,
                n_decimales=6,
                key="mccomp_conv",
            )


def _muestreo_rechazo_2d():
    """Submodulo de muestreo por rechazo 2D.

    Sirve para estimar pi (circulo inscripto en cuadrado) y areas entre curvas
    f(x), g(x) en un intervalo [a, b]. La estimacion es:

        A_hat = A_rect * (k / N)           (proporcion de aciertos * area envolvente)
        sigma_p = sqrt(p_hat * (1 - p_hat))
        IC = A_hat +/- z * A_rect * sigma_p / sqrt(N)
    """
    st.subheader("Muestreo por rechazo 2D")

    _render_teoria_libro()

    st.caption(
        "Estima un area (o probabilidad) generando puntos uniformes en un "
        "rectangulo envolvente y contando cuantos caen dentro de la region de "
        "interes. Sirve para estimar pi (circulo inscripto) o areas entre curvas."
    )

    st.latex(
        r"\hat{A} \;=\; A_{\text{rect}} \cdot \frac{k}{N}, \qquad "
        r"IC \;=\; \hat{A} \;\pm\; z_{\alpha/2} \cdot A_{\text{rect}} \cdot \frac{\sigma_p}{\sqrt{N}}"
    )
    st.latex(r"\sigma_p \;=\; \sqrt{\hat{p}\,(1 - \hat{p})}, \qquad \hat{p} = k/N")
    st.caption(
        "⚠ Nota: el libro de catedra define la formula del IC para integracion "
        "(seccion 5, pag. 39), pero **no** define explicitamente un IC para el "
        "metodo de rechazo (seccion 4, pag. 37). Aca usamos el IC Bernoulli "
        "estandar sobre la proporcion p̂ = k/N, que es el formalismo correcto "
        "para un conteo de aciertos/rechazos (k ~ Binomial(N, p))."
    )

    _PRESETS_RECHAZO = {
        "PE-3a: estimar π (círculo r=1, cuadrado [−1,1]²)": {
            "descripcion": (
                "Prueba Evaluativa 3a. Aproximar π con el círculo inscripto en "
                "[−1,1]². `np.random.seed(0)`, n=10000, K=10 repeticiones."
            ),
            "state": {
                "mrc_tipo": "Circulo inscripto (estimar pi)",
                "mrc_lado": "2",
                "mrc_radio": "1",
                "mrc_n": 10000,
                "mrc_rng_backend": "numpy (parcial)",
                "mrc_rng_use_seed": True,
                "mrc_rng_seed_val": 0,
                "mrc_conf_preset": "95%",
                "mrc_tol": 4,
            },
        },
        "PE-3b: área entre y=√x y y=x² en [0,1]": {
            "descripcion": (
                "Prueba Evaluativa 3b. Área encerrada por las curvas y=√x "
                "(superior) e y=x² (inferior) en [0,1]. Valor exacto: 1/3."
            ),
            "state": {
                "mrc_tipo": "Area entre curvas f(x), g(x)",
                "mrc_f": "\\sqrt{x}",
                "mrc_g": "x^{2}",
                "mrc_a_in": "0",
                "mrc_b_in": "1",
                "mrc_n": 10000,
                "mrc_rng_backend": "numpy (parcial)",
                "mrc_rng_use_seed": True,
                "mrc_rng_seed_val": 0,
                "mrc_conf_preset": "95%",
                "mrc_tol": 4,
                "mrc_exacto": "1/3",
            },
        },
    }
    _render_presets_parcial(_PRESETS_RECHAZO, "mrc")

    tipo = st.radio(
        "Tipo de problema",
        ["Circulo inscripto (estimar pi)", "Area entre curvas f(x), g(x)"],
        key="mrc_tipo",
    )

    col1, col2 = st.columns(2)
    with col1:
        n_muestras = st.number_input(
            "Muestras (N)", value=10000, min_value=100, max_value=5_000_000,
            step=1000, key="mrc_n",
            help="La Prueba Evaluativa exige al menos 10000.",
        )
    with col2:
        conf_label, z_score = _confianza_selector("mrc_conf")
        n_decimales = st.number_input(
            "Precision (decimales)", value=4, min_value=1, max_value=15,
            key="mrc_tol",
        )

    backend, semilla = _rng_selector("mrc_rng")

    with st.expander("🔁 Modo examen: repetir el experimento K veces", expanded=False):
        st.caption(
            "Parciales A-4a, D-4 y PE-3 piden «repita el experimento K veces, "
            "promedie los resultados y presente una tabla con desviación estándar, "
            "varianza, error estándar, media muestral e IC 95%». Cada repetición "
            "usa una semilla distinta (independiente de la tabla por lotes)."
        )
        rep_on_rc = st.checkbox(
            "Activar repeticiones", value=False, key="mrc_rep_on",
            help="Ejecuta K corridas independientes cambiando la semilla de cada una.",
        )
        k_rep_rc = st.number_input(
            "Número de repeticiones (K)", value=10, min_value=2, max_value=200,
            step=1, key="mrc_rep_k",
        )

    if tipo.startswith("Circulo"):
        st.markdown("#### Geometria")
        g1, g2 = st.columns(2)
        with g1:
            lado_str = st.text_input(
                "Lado del cuadrado (L)", value="2", key="mrc_lado",
                help="Acepta expresiones: 2*pi, sqrt(2), etc.",
            )
        with g2:
            radio_str = st.text_input(
                "Radio del circulo (r)", value="1", key="mrc_radio",
                help="Debe cumplir r <= L/2.",
            )

        if st.button("Calcular", key="mrc_calc_pi"):
            lado = parse_expr_to_float(lado_str, "L")
            radio = parse_expr_to_float(radio_str, "r")
            if lado is None or radio is None:
                return
            if radio > lado / 2 + 1e-12:
                st.error("El circulo no entra en el cuadrado: se requiere r <= L/2.")
                return

            x_min, x_max = -lado / 2, lado / 2
            y_min, y_max = -lado / 2, lado / 2
            area_rect = (x_max - x_min) * (y_max - y_min)

            _seed_rng(semilla, backend)
            x_pts, y_pts = _sample_uniform_2d(x_min, x_max, y_min, y_max, int(n_muestras), backend)
            hits = (x_pts * x_pts + y_pts * y_pts) <= (radio * radio)
            area_teorica = float(np.pi * radio * radio)
            mostrar_pi = abs(lado - 2.0) < 1e-9 and abs(radio - 1.0) < 1e-9

            def _resampler_pi(Nk: int, seed_k: Optional[int]) -> tuple:
                _seed_rng(seed_k, backend)
                xs, ys = _sample_uniform_2d(
                    x_min, x_max, y_min, y_max, int(Nk), backend
                )
                hk = (xs * xs + ys * ys) <= (radio * radio)
                p_hat_k = float(hk.sum()) / float(Nk)
                sigma_p_k = float(np.sqrt(max(p_hat_k * (1.0 - p_hat_k), 0.0)))
                area_est_k = area_rect * p_hat_k
                # El helper computa SE = sigma/sqrt(N); devolvemos sigma escalado
                # por A_rect para que el SE coincida con A_rect * sigma_p / sqrt(N).
                return area_est_k, area_rect * sigma_p_k

            _run_rechazo_2d(
                tipo="pi",
                x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                x_pts=x_pts, y_pts=y_pts, hits=hits,
                area_rect=area_rect, area_teorica=area_teorica,
                mostrar_pi=mostrar_pi, lado=lado, radio=radio,
                f_expr=None, g_expr=None, a=None, b=None,
                n=int(n_muestras), z_score=z_score, conf_label=conf_label,
                n_decimales=int(n_decimales), semilla=semilla, backend=backend,
                resampler_fn=_resampler_pi,
                rep_on=bool(rep_on_rc), k_rep=int(k_rep_rc),
            )
    else:
        st.markdown("#### Curvas")
        f_latex = math_input(label="f(x) =", default_latex="\\sqrt{x}", key="mrc_f")
        g_latex = math_input(label="g(x) =", default_latex="x^{2}", key="mrc_g")
        c1, c2 = st.columns(2)
        with c1:
            a_str = st.text_input(
                "Limite inferior (a)", value="0", key="mrc_a_in",
                help="Acepta expresiones: pi/2, sqrt(2), etc.",
            )
        with c2:
            b_str = st.text_input(
                "Limite superior (b)", value="1", key="mrc_b_in",
                help="Acepta expresiones: pi/2, sqrt(2), etc.",
            )

        with st.expander("🔍 Detectar intersecciones automáticamente", expanded=False):
            st.caption(
                "Resuelve f(x) = g(x) simbólicamente para encontrar los puntos "
                "donde las curvas se cortan. En muchos parciales esos puntos son "
                "justamente los límites de integración [a, b]. Muestra el "
                "procedimiento paso a paso y la verificación numérica."
            )
            if st.button(
                "Calcular puntos de intersección",
                key="mrc_find_intersec_btn",
            ):
                x_sym_tmp = sp.Symbol("x")
                f_expr_tmp, _ = parse_latex(f_latex, [x_sym_tmp])
                g_expr_tmp, _ = parse_latex(g_latex, [x_sym_tmp])
                if f_expr_tmp is None or g_expr_tmp is None:
                    st.error("No pude parsear f o g. Revisá las expresiones.")
                else:
                    _detectar_intersecciones_curvas(
                        f_expr=f_expr_tmp,
                        g_expr=g_expr_tmp,
                        x_sym=x_sym_tmp,
                        key_a="mrc_a_in",
                        key_b="mrc_b_in",
                    )

        with st.expander("🎯 Valor exacto (opcional, simbólico)", expanded=False):
            st.caption(
                "Si conocés el área exacta, escribila acá: "
                "`1/3` (y=√x vs y=x²), `1/12` (y=x² vs y=x³), "
                "`pi` (círculo unitario), etc. Vacío = SymPy automático."
            )
            exacto_curvas_str = st.text_input(
                "Área exacta A*", value="", key="mrc_exacto",
                placeholder="dejar vacío para usar SymPy automático",
            )

        if st.button("Calcular", key="mrc_calc_curvas"):
            x_sym = sp.Symbol("x")
            f_expr, f_np = parse_latex(f_latex, [x_sym])
            g_expr, g_np = parse_latex(g_latex, [x_sym])
            if f_expr is None or g_expr is None:
                return
            a_val = parse_expr_to_float(a_str, "a")
            b_val = parse_expr_to_float(b_str, "b")
            if a_val is None or b_val is None:
                return
            if b_val <= a_val:
                st.error("Se requiere b > a.")
                return

            # Rectangulo envolvente por muestreo denso de f y g
            xs_grid = np.linspace(a_val, b_val, 2001)
            try:
                fs_grid = np.asarray(f_np(xs_grid), dtype=float)
                gs_grid = np.asarray(g_np(xs_grid), dtype=float)
            except Exception as exc:
                st.error(f"No pude evaluar las funciones en la grilla: {exc}")
                return
            y_lo = float(np.nanmin([np.nanmin(fs_grid), np.nanmin(gs_grid)]))
            y_hi = float(np.nanmax([np.nanmax(fs_grid), np.nanmax(gs_grid)]))
            if not (np.isfinite(y_lo) and np.isfinite(y_hi)):
                st.error(
                    "f o g producen valores no finitos en [a, b]. "
                    "Elegi un intervalo donde ambas sean finitas."
                )
                return
            # Rectángulo contenedor ajustado exactamente a los extremos de f, g.
            # Sin padding: A_rect = (b-a)(y_max-y_min) con los máximos reales,
            # como pide la consigna. Un pad inflaba A_rect (y desperdiciaba
            # muestras en franjas sin hits posibles).
            y_min = y_lo
            y_max = y_hi
            x_min, x_max = float(a_val), float(b_val)
            area_rect = (x_max - x_min) * (y_max - y_min)

            _seed_rng(semilla, backend)
            x_pts, y_pts = _sample_uniform_2d(x_min, x_max, y_min, y_max, int(n_muestras), backend)
            try:
                fxs = np.asarray(f_np(x_pts), dtype=float)
                gxs = np.asarray(g_np(x_pts), dtype=float)
            except Exception as exc:
                st.error(f"No pude evaluar f, g en los puntos muestreados: {exc}")
                return
            lo = np.minimum(fxs, gxs)
            hi = np.maximum(fxs, gxs)
            hits = (y_pts >= lo) & (y_pts <= hi)

            # Valor teorico: override manual > analitico via sympy > fallback numerico
            area_teorica = None
            exacto_manual_curvas = (exacto_curvas_str or "").strip()
            if exacto_manual_curvas:
                area_teorica = parse_expr_to_float(exacto_manual_curvas, "Área exacta A*")
            if area_teorica is None:
                try:
                    integral_sym = sp.integrate(
                        sp.Abs(f_expr - g_expr), (x_sym, a_val, b_val)
                    )
                    area_teorica = float(integral_sym)
                except Exception:
                    try:
                        def _abs_diff(x: float) -> float:
                            return float(abs(float(f_np(x)) - float(g_np(x))))
                        val, _ = integrate.quad(_abs_diff, a_val, b_val)
                        area_teorica = float(val)
                    except Exception:
                        area_teorica = None

            def _resampler_curvas(Nk: int, seed_k: Optional[int]) -> tuple:
                _seed_rng(seed_k, backend)
                xs, ys = _sample_uniform_2d(
                    x_min, x_max, y_min, y_max, int(Nk), backend
                )
                try:
                    fk = np.asarray(f_np(xs), dtype=float)
                    gk = np.asarray(g_np(xs), dtype=float)
                except Exception:
                    return 0.0, 0.0
                lo_k = np.minimum(fk, gk)
                hi_k = np.maximum(fk, gk)
                hk = (ys >= lo_k) & (ys <= hi_k)
                p_hat_k = float(hk.sum()) / float(Nk)
                sigma_p_k = float(np.sqrt(max(p_hat_k * (1.0 - p_hat_k), 0.0)))
                area_est_k = area_rect * p_hat_k
                return area_est_k, area_rect * sigma_p_k

            _run_rechazo_2d(
                tipo="curvas",
                x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                x_pts=x_pts, y_pts=y_pts, hits=hits,
                area_rect=area_rect, area_teorica=area_teorica,
                mostrar_pi=False, lado=None, radio=None,
                f_expr=f_expr, g_expr=g_expr, a=a_val, b=b_val,
                n=int(n_muestras), z_score=z_score, conf_label=conf_label,
                n_decimales=int(n_decimales), semilla=semilla, backend=backend,
                resampler_fn=_resampler_curvas,
                rep_on=bool(rep_on_rc), k_rep=int(k_rep_rc),
            )


def _run_rechazo_2d(
    *,
    tipo: str,
    x_min: float, x_max: float, y_min: float, y_max: float,
    x_pts: np.ndarray, y_pts: np.ndarray, hits: np.ndarray,
    area_rect: float, area_teorica,
    mostrar_pi: bool, lado, radio,
    f_expr, g_expr, a, b,
    n: int, z_score: float, conf_label: str,
    n_decimales: int, semilla: Optional[int],
    backend: RngBackend = _DEFAULT_BACKEND,
    resampler_fn=None,
    rep_on: bool = False,
    k_rep: int = 10,
) -> None:
    """Renderiza las 3 tabs (resumen / pasos / viz) para el muestreo por rechazo 2D."""
    k = int(hits.sum())
    p_hat = k / n
    sigma_p = float(np.sqrt(p_hat * (1.0 - p_hat)))
    area_est = area_rect * p_hat
    ee = area_rect * sigma_p / np.sqrt(n)
    margen = z_score * ee
    ic_low = area_est - margen
    ic_up = area_est + margen

    tab_resumen, tab_pasos, tab_viz, tab_conv = st.tabs(
        ["📊 Resumen", "🧮 Paso a paso", "📈 Visualizacion", "📉 Convergencia 1/√N"]
    )

    # =================== RESUMEN ===================
    with tab_resumen:
        with st.container(border=True):
            st.markdown("##### Resultados")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("N", f"{n:,}")
            c2.metric("k (aciertos)", f"{k:,}")
            c3.metric("p̂ = k/N", _fmt_decimal(p_hat))
            c4.metric("A_rect", _fmt_decimal(area_rect))

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("A estimada", f"{area_est:.{n_decimales}f}")
            c6.metric("σ_p", _fmt_decimal(sigma_p))
            c7.metric("EE = A_rect·σ_p/√N", _fmt_decimal(float(ee)))
            c8.metric(f"Margen ({conf_label})", _fmt_decimal(float(margen)))

            st.markdown(
                f"**IC {conf_label} del área:** "
                f"[{ic_low:.{n_decimales}f}, {ic_up:.{n_decimales}f}]"
            )

            if area_teorica is not None:
                err_abs = abs(area_est - area_teorica)
                cubre = ic_low <= area_teorica <= ic_up
                st.markdown(
                    f"**Valor teórico:** {area_teorica:.{n_decimales}f}  •  "
                    f"**Error absoluto:** {_fmt_decimal(err_abs)}  •  "
                    f"**¿IC lo cubre?** {'✅ sí' if cubre else '❌ no'}"
                )

        if mostrar_pi:
            with st.container(border=True):
                st.markdown("##### 🎯 Estimación de π")
                pi_hat = 4.0 * p_hat
                pi_margen = z_score * 4.0 * sigma_p / np.sqrt(n)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("π̂ = 4·p̂", f"{pi_hat:.{n_decimales}f}")
                c2.metric("π real", f"{float(np.pi):.{n_decimales}f}")
                c3.metric("Error abs", _fmt_decimal(abs(pi_hat - float(np.pi))))
                c4.metric(f"Margen ({conf_label})", _fmt_decimal(float(pi_margen)))

                ic_pi_low = pi_hat - pi_margen
                ic_pi_up = pi_hat + pi_margen
                cubre_pi = ic_pi_low <= float(np.pi) <= ic_pi_up
                st.markdown(
                    f"**IC {conf_label} para π:** "
                    f"[{ic_pi_low:.{n_decimales}f}, {ic_pi_up:.{n_decimales}f}]  •  "
                    f"**¿Cubre π?** {'✅ sí' if cubre_pi else '❌ no'}"
                )

            # --- Tabla Bernoulli completa por lotes (estilo parcial PE-3) ---
            with st.container(border=True):
                st.markdown("##### 📋 Tabla Bernoulli por lotes (parcial PE-3)")
                st.caption(
                    "La consigna del parcial pide reportar, en 10 lotes "
                    "incrementales, cómo evoluciona la estimación de π junto "
                    "con los estadísticos Bernoulli. Cada fila acumula los "
                    "puntos hasta N_i = i·(N/10)."
                )
                st.latex(
                    r"\hat{p}_i = \frac{k_i}{N_i}, \quad "
                    r"\hat{\pi}_i = 4\hat{p}_i, \quad "
                    r"\sigma_{p,i} = \sqrt{\hat{p}_i(1-\hat{p}_i)}, \quad "
                    r"\mathrm{SE}_{\pi,i} = \frac{4\,\sigma_{p,i}}{\sqrt{N_i}}"
                )

                n_lotes = 10
                paso = max(1, int(n) // n_lotes)
                filas_bern = []
                pi_real = float(np.pi)
                for i in range(1, n_lotes + 1):
                    Ni = min(i * paso, int(n))
                    hits_i = hits[:Ni]
                    ki = int(hits_i.sum())
                    pi_hat_i = 4.0 * ki / Ni
                    p_hat_i = ki / Ni
                    var_bern_i = p_hat_i * (1.0 - p_hat_i)
                    sigma_p_i = float(np.sqrt(max(var_bern_i, 0.0)))
                    se_pi_i = 4.0 * sigma_p_i / np.sqrt(Ni)
                    margen_i = z_score * se_pi_i
                    err_i = abs(pi_hat_i - pi_real)
                    filas_bern.append({
                        "i": i,
                        "N_i": Ni,
                        "k_i (dentro)": ki,
                        "p̂_i": round(p_hat_i, n_decimales),
                        "π̂_i = 4·p̂_i": round(pi_hat_i, n_decimales),
                        "p̂(1−p̂)": round(var_bern_i, n_decimales),
                        "σ_p = √(p̂(1−p̂))": round(sigma_p_i, n_decimales),
                        "SE_π = 4σ_p/√N": round(se_pi_i, n_decimales),
                        "|π̂ − π|": round(err_i, n_decimales),
                        f"IC low ({conf_label})": round(pi_hat_i - margen_i, n_decimales),
                        f"IC up ({conf_label})": round(pi_hat_i + margen_i, n_decimales),
                    })
                df_bern = pd.DataFrame(filas_bern)
                st.dataframe(df_bern, use_container_width=True, hide_index=True)

                # Gráfico de convergencia de π vs índice de lote
                fig_bern = go.Figure()
                Ns_b = [row["N_i"] for row in filas_bern]
                pis_b = [row["π̂_i = 4·p̂_i"] for row in filas_bern]
                ic_lo_b = [row[f"IC low ({conf_label})"] for row in filas_bern]
                ic_up_b = [row[f"IC up ({conf_label})"] for row in filas_bern]

                fig_bern.add_trace(go.Scatter(
                    x=Ns_b, y=pis_b, mode="markers+lines",
                    name="π̂ por lote", marker=dict(size=8, color="#4FC3F7"),
                    line=dict(color="#4FC3F7"),
                ))
                fig_bern.add_trace(go.Scatter(
                    x=Ns_b + Ns_b[::-1],
                    y=ic_up_b + ic_lo_b[::-1],
                    fill="toself", fillcolor="rgba(79,195,247,0.15)",
                    line=dict(width=0), name=f"IC {conf_label}",
                    hoverinfo="skip",
                ))
                fig_bern.add_hline(
                    y=pi_real, line_dash="dash", line_color="#FFB74D",
                    annotation_text="π real", annotation_position="right",
                )
                fig_bern.update_layout(
                    template="plotly_dark",
                    xaxis_title="N acumulado",
                    yaxis_title="π̂",
                    title="Convergencia de π̂ por lotes con banda IC",
                    height=380,
                    margin=dict(l=50, r=20, t=50, b=40),
                )
                st.plotly_chart(fig_bern, use_container_width=True, key="mrc_bern_chart")

        # --- Repeticiones independientes del experimento (consigna K veces) ---
        if rep_on and resampler_fn is not None:
            def _sampler_rechazo(seed_i: Optional[int]) -> float:
                area_k, _ = resampler_fn(int(n), seed_i)
                return float(area_k)

            _render_repeticiones_examen(
                sampler_fn=_sampler_rechazo,
                n_repeticiones=int(k_rep),
                semilla_base=semilla,
                z_score=z_score,
                conf_label=conf_label,
                n_decimales=int(n_decimales),
            )

    # =================== PASO A PASO ===================
    with tab_pasos:
        _render_pasos_rechazo(
            tipo=tipo,
            x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
            x_pts=x_pts, y_pts=y_pts, hits=hits,
            area_rect=area_rect, area_est=area_est, area_teorica=area_teorica,
            k=k, n=n, p_hat=p_hat, sigma_p=sigma_p,
            margen=float(margen), ic_low=ic_low, ic_up=ic_up,
            z_score=z_score, conf_label=conf_label, n_decimales=n_decimales,
            lado=lado, radio=radio, mostrar_pi=mostrar_pi,
            f_expr=f_expr, g_expr=g_expr, a=a, b=b, semilla=semilla,
            backend=backend,
        )

    # =================== VISUALIZACION ===================
    with tab_viz:
        _render_viz_rechazo(
            tipo=tipo,
            x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
            x_pts=x_pts, y_pts=y_pts, hits=hits,
            radio=radio, f_expr=f_expr, g_expr=g_expr, a=a, b=b,
        )

    # =================== CONVERGENCIA 1/sqrt(N) ===================
    with tab_conv:
        if resampler_fn is None:
            st.info(
                "Bloque de convergencia no disponible para esta configuración "
                "(falta el re-sampler)."
            )
        else:
            _render_error_sqrt_n(
                sampler_fn=resampler_fn,
                n_base=int(n),
                semilla_base=semilla,
                backend=backend,
                n_decimales=int(n_decimales),
                key=f"mrc_conv_{tipo}",
            )


def _render_pasos_rechazo(
    *,
    tipo: str,
    x_min: float, x_max: float, y_min: float, y_max: float,
    x_pts: np.ndarray, y_pts: np.ndarray, hits: np.ndarray,
    area_rect: float, area_est: float, area_teorica,
    k: int, n: int, p_hat: float, sigma_p: float,
    margen: float, ic_low: float, ic_up: float,
    z_score: float, conf_label: str, n_decimales: int,
    lado, radio, mostrar_pi: bool,
    f_expr, g_expr, a, b,
    semilla: Optional[int],
    backend: RngBackend = _DEFAULT_BACKEND,
) -> None:
    """Paso a paso estilo Symbolab para el muestreo por rechazo 2D."""
    N = int(n)

    # --- Paso 1: Planteo ---
    st.markdown("#### Paso 1 — Planteo del problema")
    if tipo == "pi":
        st.markdown(
            f"Estimar el área del círculo de radio $r = {_fmt_latex_num(radio)}$ "
            f"inscripto en un cuadrado de lado $L = {_fmt_latex_num(lado)}$ "
            f"usando muestreo por rechazo con $N = {N:,}$ puntos."
        )
        if mostrar_pi:
            st.markdown(
                "Como el cuadrado tiene lado $2$ y el círculo radio $1$, "
                r"$A_{\text{círculo}} = \pi r^{2} = \pi$, "
                "así que estimar el área equivale a **estimar π**."
            )
    else:
        st.markdown(
            f"Estimar el área encerrada entre $f(x)$ y $g(x)$ en "
            f"$[{_fmt_latex_num(a)}, {_fmt_latex_num(b)}]$ por muestreo por rechazo, "
            f"con $N = {N:,}$ puntos."
        )
        st.latex(rf"f(x) = {sp.latex(f_expr)}, \qquad g(x) = {sp.latex(g_expr)}")
        st.latex(
            rf"A = \int_{{{_fmt_latex_num(a)}}}^{{{_fmt_latex_num(b)}}} "
            rf"|f(x) - g(x)|\, dx"
        )
    st.divider()

    # --- Paso 2: Rectangulo envolvente ---
    st.markdown("#### Paso 2 — Rectángulo envolvente")
    st.latex(
        rf"\text{{Rect}} \;=\; [{_fmt_latex_num(x_min)},\, {_fmt_latex_num(x_max)}] "
        rf"\times [{_fmt_latex_num(y_min)},\, {_fmt_latex_num(y_max)}]"
    )
    st.latex(
        r"A_{\text{rect}} \;=\; "
        rf"({_fmt_latex_num(x_max)} - {_fmt_latex_num(x_min)}) \cdot "
        rf"({_fmt_latex_num(y_max)} - {_fmt_latex_num(y_min)}) "
        rf"\;=\; {_fmt_latex_num(area_rect)}"
    )
    st.divider()

    # --- Paso 3: Generacion de puntos ---
    st.markdown("#### Paso 3 — Generación de puntos uniformes")
    seed_call = _fmt_seed_call(semilla, backend)
    if backend == "numpy":
        orden_txt = (
            "primero se genera el vector de $x_i$ y luego el vector de $y_i$ "
            "(muestreo vectorizado de NumPy)"
        )
        fn_txt = "`np.random.uniform`"
    else:
        orden_txt = (
            "en orden interleaved $x_1, y_1, x_2, y_2, \\ldots$ "
            "(convención de cátedra con `random` de stdlib)"
        )
        fn_txt = "`random.uniform`"
    st.markdown(
        f"Se generan $N = {N:,}$ puntos $(x_i, y_i)$ con "
        f"`{seed_call}` y {fn_txt}, {orden_txt}."
    )
    st.latex(
        r"x_i \sim \mathcal{U}(" + _fmt_latex_num(x_min) + r",\, " + _fmt_latex_num(x_max) + r"), "
        r"\quad "
        r"y_i \sim \mathcal{U}(" + _fmt_latex_num(y_min) + r",\, " + _fmt_latex_num(y_max) + r")"
    )
    st.markdown("**Primeras 5 muestras (✔ = acierto, ✗ = rechazo):**")
    for i in range(min(5, N)):
        marca = r"\checkmark" if bool(hits[i]) else r"\times"
        st.latex(
            rf"({x_pts[i]:.4f},\;\; {y_pts[i]:.4f}) "
            rf"\quad \Rightarrow \quad {marca}"
        )
    st.divider()

    # --- Paso 4: Criterio de acierto ---
    st.markdown("#### Paso 4 — Criterio de acierto")
    if tipo == "pi":
        st.latex(
            r"\text{Acierto}_i \iff x_i^{2} + y_i^{2} \;\leq\; "
            + _fmt_latex_num(float(radio) * float(radio))
        )
    else:
        st.latex(
            r"\text{Acierto}_i \iff "
            r"\min\bigl(f(x_i),\, g(x_i)\bigr) \;\leq\; y_i "
            r"\;\leq\; \max\bigl(f(x_i),\, g(x_i)\bigr)"
        )
    st.divider()

    # --- Paso 5: Conteo ---
    st.markdown("#### Paso 5 — Conteo de aciertos")
    st.latex(
        rf"k \;=\; \sum_{{i=1}}^{{N}} \mathbb{{1}}[\text{{acierto}}_i] \;=\; {k}"
    )
    st.latex(
        rf"\hat{{p}} \;=\; \frac{{k}}{{N}} \;=\; \frac{{{k}}}{{{N}}} "
        rf"\;=\; {p_hat:.6f}"
    )
    st.divider()

    # --- Paso 6: Estimador del area ---
    st.markdown("#### Paso 6 — Estimador del área")
    st.latex(
        r"\hat{A} \;=\; A_{\text{rect}} \cdot \hat{p} "
        rf"\;=\; {_fmt_latex_num(area_rect)} \cdot {p_hat:.6f} "
        rf"\;=\; {area_est:.{n_decimales}f}"
    )
    st.latex(rf"\boxed{{\;\hat{{A}} \;=\; {area_est:.{n_decimales}f}\;}}")
    st.divider()

    # --- Paso 7: Desvio de la proporcion ---
    st.markdown("#### Paso 7 — Desvío estándar de la proporción")
    st.caption(
        "Cada punto es un ensayo Bernoulli (acierto / rechazo), "
        "así que $\\sigma_p = \\sqrt{\\hat{p}\\,(1 - \\hat{p})}$."
    )
    st.latex(
        rf"\sigma_p \;=\; \sqrt{{{p_hat:.6f} \cdot (1 - {p_hat:.6f})}} "
        rf"\;=\; {sigma_p:.6f}"
    )
    st.divider()

    # --- Paso 8: IC ---
    st.markdown(f"#### Paso 8 — Intervalo de confianza ({conf_label})")
    st.caption(
        r"Como $\hat{A} = A_{\text{rect}} \cdot \hat{p}$, "
        r"$\text{Var}(\hat{A}) = A_{\text{rect}}^{2} \cdot \hat{p}\,(1-\hat{p}) / N$, "
        r"de donde $\text{EE}(\hat{A}) = A_{\text{rect}} \cdot \sigma_p / \sqrt{N}$."
    )
    st.latex(
        r"IC \;=\; \hat{A} \;\pm\; z_{\alpha/2} \cdot A_{\text{rect}} \cdot "
        r"\frac{\sigma_p}{\sqrt{N}}"
    )
    st.latex(
        rf"IC \;=\; {area_est:.{n_decimales}f} \;\pm\; {z_score:.4f} \cdot "
        rf"{_fmt_latex_num(area_rect)} \cdot \frac{{{sigma_p:.6f}}}{{\sqrt{{{N}}}}}"
    )
    st.latex(
        rf"IC \;=\; {area_est:.{n_decimales}f} \;\pm\; {margen:.6f} "
        rf"\;=\; [\,{ic_low:.{n_decimales}f},\;\; {ic_up:.{n_decimales}f}\,]"
    )

    # --- Paso 9: Comparacion / pi ---
    if mostrar_pi:
        st.divider()
        st.markdown("#### Paso 9 — Estimación de π")
        pi_hat = 4.0 * p_hat
        pi_mg = z_score * 4.0 * sigma_p / np.sqrt(N)
        pi_lo = pi_hat - pi_mg
        pi_hi = pi_hat + pi_mg
        st.latex(
            rf"\hat{{\pi}} \;=\; 4 \cdot \hat{{p}} "
            rf"\;=\; 4 \cdot {p_hat:.6f} \;=\; {pi_hat:.{n_decimales}f}"
        )
        st.latex(
            r"IC_{\pi} \;=\; \hat{\pi} \;\pm\; z_{\alpha/2} \cdot "
            r"4 \cdot \frac{\sigma_p}{\sqrt{N}} "
            rf"\;=\; [\,{pi_lo:.{n_decimales}f},\;\; {pi_hi:.{n_decimales}f}\,]"
        )
        err_abs_pi = abs(pi_hat - float(np.pi))
        st.latex(
            rf"|\hat{{\pi}} - \pi| \;=\; {_fmt_decimal(err_abs_pi)}"
        )
    elif area_teorica is not None:
        st.divider()
        st.markdown("#### Paso 9 — Comparación con el valor exacto (SymPy)")
        err_abs = abs(area_est - area_teorica)
        err_rel = err_abs / abs(area_teorica) if area_teorica != 0 else float("inf")
        st.latex(
            rf"A_{{\text{{exacto}}}} \;=\; {area_teorica:.{n_decimales}f}"
        )
        st.latex(
            rf"\text{{Error absoluto}} \;=\; \bigl|\hat{{A}} - A\bigr| "
            rf"\;=\; {_fmt_decimal(err_abs)}"
        )
        st.latex(
            rf"\text{{Error relativo}} \;=\; {_fmt_decimal(err_rel)}"
        )


def _render_viz_rechazo(
    *,
    tipo: str,
    x_min: float, x_max: float, y_min: float, y_max: float,
    x_pts: np.ndarray, y_pts: np.ndarray, hits: np.ndarray,
    radio, f_expr, g_expr, a, b,
) -> None:
    """Scatter 2D con aciertos / rechazos y el contorno de la region."""
    # Submuestreo para no saturar plotly
    max_plot = 5000
    n_total = len(x_pts)
    if n_total > max_plot:
        idx = np.linspace(0, n_total - 1, max_plot).astype(int)
        x_plot = x_pts[idx]
        y_plot = y_pts[idx]
        hits_plot = hits[idx]
        st.caption(
            f"Mostrando {max_plot:,} puntos de {n_total:,} (submuestra uniforme)."
        )
    else:
        x_plot = x_pts
        y_plot = y_pts
        hits_plot = hits

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_plot[hits_plot], y=y_plot[hits_plot],
        mode="markers",
        marker=dict(size=3, color="#00d4ff", opacity=0.65),
        name="Aciertos",
    ))
    fig.add_trace(go.Scatter(
        x=x_plot[~hits_plot], y=y_plot[~hits_plot],
        mode="markers",
        marker=dict(size=3, color="#ff6b6b", opacity=0.45),
        name="Rechazos",
    ))

    if tipo == "pi":
        theta = np.linspace(0, 2 * np.pi, 400)
        fig.add_trace(go.Scatter(
            x=float(radio) * np.cos(theta),
            y=float(radio) * np.sin(theta),
            mode="lines",
            line=dict(color="#ffd700", width=3),
            name=f"Círculo r={_fmt_latex_num(float(radio))}",
        ))
    else:
        try:
            x_sym = sp.Symbol("x")
            f_np = sp.lambdify(x_sym, f_expr, "numpy")
            g_np = sp.lambdify(x_sym, g_expr, "numpy")
            xs_line = np.linspace(float(a), float(b), 400)
            fs_line = np.asarray(f_np(xs_line), dtype=float)
            gs_line = np.asarray(g_np(xs_line), dtype=float)
            fig.add_trace(go.Scatter(
                x=xs_line, y=fs_line, mode="lines",
                line=dict(color="#ffd700", width=3),
                name=f"f(x) = {sp.pretty(f_expr)}",
            ))
            fig.add_trace(go.Scatter(
                x=xs_line, y=gs_line, mode="lines",
                line=dict(color="#00ff80", width=3),
                name=f"g(x) = {sp.pretty(g_expr)}",
            ))
        except Exception:
            pass

    fig.update_layout(
        template="plotly_dark",
        title="Muestreo por rechazo 2D",
        xaxis_title="x",
        yaxis_title="y",
        xaxis=dict(range=[x_min, x_max], scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[y_min, y_max]),
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=20, r=20, t=40, b=20),
        height=600,
    )
    st.plotly_chart(fig, use_container_width=True)


def render():
    st.header("Metodo de Monte Carlo")

    submenu = st.radio(
        "Selecciona el submenu:",
        [
            "Integracion 1D",
            "Integracion Multidimensional",
            "Muestreo por rechazo 2D",
            "Comparacion de Metodos",
            "Vista Catedra (comparativa)",
        ],
        horizontal=True,
        key="mc_submenu",
    )

    if submenu == "Integracion 1D":
        _integracion_1d()
    elif submenu == "Integracion Multidimensional":
        _integracion_multidimensional()
    elif submenu == "Muestreo por rechazo 2D":
        _muestreo_rechazo_2d()
    elif submenu == "Comparacion de Metodos":
        _comparacion_metodos()
    else:
        # Import local para evitar circular: montecarlo_catedra importa helpers
        # desde este modulo.
        from modules.montecarlo_catedra import render_catedra
        render_catedra()
