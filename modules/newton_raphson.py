"""Modulo Newton-Raphson — iteracion x_{n+1} = x_n − f(x_n)/f'(x_n).

Implementa el metodo tal como lo define el prof. Caceres (pg 15-16):
  Usa la recta tangente a f en x_n para estimar la raiz.
  Convergencia cuadratica si f'(x*) ≠ 0 y x_0 esta cerca de x*.

Features:
- Banco de ejercicios oficiales (pg 19) como presets.
- Derivada f'(x) calculada simbolicamente via SymPy.
- Pre-checks: f'(x_0) ≠ 0, monitoreo de f'(x_n) en cada iteracion.
- 4 criterios de detencion combinables.
- Plot de la tangente por iteracion (slider — visualiza el metodo grafico).
- Verifica orden de convergencia empirico (≈ 2 para raiz simple, ≈ 1 para multiple).
- 3 tabs: Resumen, Paso a paso, Visualizaciones.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sympy as sp

from utils.math_keyboard import math_input, parse_latex
from utils.ui.config import get_config
from utils.ui.glosario import render_glosario_expander
from utils.ui.pasos import Paso, render_pasos
from utils.ui.tablas import fmt_decimal, render_tabla_iteraciones, resaltar_tolerancia
from utils.ui.teoria import render_teoria

# --- Banco de ejercicios (Caceres pg 19) ---

EJERCICIOS: dict[str, dict] = {
    "Personalizado": {},
    "Ej 1 — f(x) = (x−1)^2, x₀ = 0 (raiz doble)": {
        "latex": r"(x-1)^{2}", "x0": 0.0, "tol": 1e-6,
    },
    "Ej 2 — f(x) = x^3 − 2x − 5, x₀ = 1.5": {
        "latex": r"x^{3} - 2x - 5", "x0": 1.5, "tol": 1e-6,
    },
    "Ej 3 — f(x) = x^5 − x − 1, x₀ = 1": {
        "latex": r"x^{5} - x - 1", "x0": 1.0, "tol": 1e-6,
    },
    "Ej 4 — √[6]{2} via x^6 − 2 = 0, x₀ = 1.2 (tol 10^-8)": {
        "latex": r"x^{6} - 2", "x0": 1.2, "tol": 1e-8,
    },
    "Ej 5 — f(x) = e^x + x^2 − 4, x₀ = 0.5": {
        "latex": r"e^{x} + x^{2} - 4", "x0": 0.5, "tol": 1e-6,
    },
    "Ej 6 — f(x) = x^2 − 3x − 4, x₀ = 8": {
        "latex": r"x^{2} - 3x - 4", "x0": 8.0, "tol": 1e-6,
    },
    "Ej 7 — f(x) = ln(x) − 1, x₀ = 2": {
        "latex": r"\ln(x) - 1", "x0": 2.0, "tol": 1e-6,
    },
    "Ej 8 — f(x) = x^4 − 16, x₀ = 2": {
        "latex": r"x^{4} - 16", "x0": 2.0, "tol": 1e-6,
    },
    "Ej 9 — f(x) = x^3 − 2x + 1, x₀ = −1.5": {
        "latex": r"x^{3} - 2x + 1", "x0": -1.5, "tol": 1e-6,
    },
    "Ej 10 — f(x) = e^(3x) − 4, x₀ = 0": {
        "latex": r"e^{3x} - 4", "x0": 0.0, "tol": 1e-6,
    },
    "Ej 11 — f(x) = x^2 − 2x + 1, x₀ = 0 (raiz doble)": {
        "latex": r"x^{2} - 2x + 1", "x0": 0.0, "tol": 1e-6,
    },
    "Ej 12 — f(x) = x·e^(−x), x₀ = −1": {
        "latex": r"x e^{-x}", "x0": -1.0, "tol": 1e-6,
    },
}


# --- Estructuras ---

@dataclass(frozen=True)
class CriteriosDetencion:
    usar_abs: bool = True
    tol_abs: float = 1e-6
    usar_rel: bool = False
    tol_rel: float = 1e-6
    usar_residuo: bool = True
    tol_residuo: float = 1e-6
    usar_max_iter: bool = True
    max_iter: int = 100


@dataclass(frozen=True)
class Iteracion:
    n: int
    x_n: float
    f_xn: float
    fp_xn: float
    x_next: float
    error_abs: float
    error_rel: float


@dataclass
class ResultadoNR:
    iteraciones: list[Iteracion] = field(default_factory=list)
    raiz: float | None = None
    motivo_corte: str = ""
    convergio: bool = False
    x0: float = 0.0


# --- Algoritmo ---

def newton_raphson(
    f_np, fp_np,
    x0: float,
    criterios: CriteriosDetencion,
    precision: int | None = None,
    tol_derivada: float = 1e-14,
) -> ResultadoNR:
    """Ejecuta Newton-Raphson siguiendo el algoritmo del libro.

    Args:
        f_np: f(x) numerica.
        fp_np: f'(x) numerica (derivada simbolica).
        x0: punto inicial.
        criterios: condiciones de parada.
        precision: si no es None, aplica round(·, precision).
        tol_derivada: umbral para considerar f' ≈ 0 y abortar.
    """
    def r(v: float) -> float:
        return round(v, precision) if precision is not None else v

    res = ResultadoNR(x0=x0)
    n_max = criterios.max_iter if criterios.usar_max_iter else 10_000
    x_n = r(x0)

    for n in range(1, n_max + 1):
        try:
            f_xn = r(float(f_np(x_n)))
            fp_xn = r(float(fp_np(x_n)))
        except Exception as e:
            res.motivo_corte = f"Error al evaluar f o f' en iter {n} (x={x_n}): {e}"
            return res

        if not (np.isfinite(f_xn) and np.isfinite(fp_xn)):
            res.motivo_corte = f"f o f' no finitos en iter {n}."
            return res

        if abs(fp_xn) < tol_derivada:
            res.motivo_corte = (
                f"f'(x) ≈ 0 en iter {n} (f'={fp_xn:.2e}). "
                f"La tangente es horizontal — el metodo falla."
            )
            return res

        x_next = r(x_n - f_xn / fp_xn)

        if not np.isfinite(x_next):
            res.motivo_corte = f"x_{{n+1}} no finito en iter {n}. Divergencia."
            return res

        err_abs = abs(x_next - x_n)
        err_rel = err_abs / abs(x_next) if x_next != 0 else float("inf")

        res.iteraciones.append(Iteracion(
            n=n, x_n=x_n, f_xn=f_xn, fp_xn=fp_xn,
            x_next=x_next, error_abs=err_abs, error_rel=err_rel,
        ))

        if criterios.usar_residuo and abs(f_xn) <= criterios.tol_residuo:
            res.raiz = x_next
            res.motivo_corte = f"|f(x)| ≤ {fmt_decimal(criterios.tol_residuo)} (residuo)"
            res.convergio = True
            return res
        if criterios.usar_abs and err_abs <= criterios.tol_abs:
            res.raiz = x_next
            res.motivo_corte = f"|x_n+1 − x_n| ≤ {fmt_decimal(criterios.tol_abs)} (error absoluto)"
            res.convergio = True
            return res
        if criterios.usar_rel and err_rel <= criterios.tol_rel:
            res.raiz = x_next
            res.motivo_corte = f"error relativo ≤ {fmt_decimal(criterios.tol_rel)}"
            res.convergio = True
            return res

        if abs(x_next) > 1e12:
            res.motivo_corte = f"Divergencia: |x_{n}| = {abs(x_next):.2e}"
            return res

        x_n = x_next

    res.raiz = x_n
    res.motivo_corte = f"Se alcanzo el maximo de iteraciones (N_max = {criterios.max_iter})"
    return res


# --- Plots ---

def _plot_curva_con_raiz(f_np, res: ResultadoNR, a: float, b: float) -> go.Figure:
    """Grafico estatico para examen: curva f(x) + ejes x e y visibles + raiz marcada.

    Garantiza que el eje y (x=0) y el eje x (y=0) se vean en la figura,
    extendiendo el rango si es necesario, para responder "grafique la curva
    y ubique la raiz calculada".
    """
    from utils.graficos import apply_geogebra_style

    # Extender rango en x para que incluya x=0 (eje y visible)
    x_lo = min(a, 0.0)
    x_hi = max(b, 0.0)
    if x_lo == x_hi:
        x_lo, x_hi = x_lo - 1.0, x_hi + 1.0

    xs = np.linspace(x_lo, x_hi, 700)
    try:
        ys = np.asarray(f_np(xs), dtype=float)
    except Exception:
        ys = np.array([float(f_np(xi)) for xi in xs])

    # Rango en y que incluya y=0 y la curva, con margen
    y_finite = ys[np.isfinite(ys)]
    if len(y_finite) == 0:
        y_lo, y_hi = -1.0, 1.0
    else:
        y_min, y_max = float(np.min(y_finite)), float(np.max(y_finite))
        y_lo = min(y_min, 0.0)
        y_hi = max(y_max, 0.0)
        span = y_hi - y_lo if y_hi > y_lo else 1.0
        y_lo -= span * 0.1
        y_hi += span * 0.1

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines", name="f(x)",
        line=dict(color="#1f77b4", width=2.5),
    ))
    # Ejes x e y remarcados (cruzan en 0,0)
    fig.add_hline(y=0, line=dict(color="#222", width=1.5))
    fig.add_vline(x=0, line=dict(color="#222", width=1.5))

    if res.raiz is not None:
        try:
            f_raiz = float(f_np(res.raiz))
        except Exception:
            f_raiz = 0.0
        # Linea vertical punteada hasta la raiz
        fig.add_trace(go.Scatter(
            x=[res.raiz, res.raiz], y=[0, f_raiz],
            mode="lines", line=dict(color="#d62728", dash="dot", width=1),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=[res.raiz], y=[0],
            mode="markers+text",
            marker=dict(size=14, color="#d62728", symbol="circle",
                         line=dict(color="#7a0b0b", width=1.5)),
            text=[f"Raiz ≈ {fmt_decimal(res.raiz)}"],
            textposition="top center",
            textfont=dict(size=13, color="#7a0b0b"),
            name="Raiz",
        ))

    fig.update_layout(
        title=dict(text="Curva f(x) con raiz ubicada", x=0.5),
        xaxis_title="x", yaxis_title="f(x)",
        height=480, showlegend=True,
        xaxis=dict(range=[x_lo, x_hi]),
        yaxis=dict(range=[y_lo, y_hi]),
    )
    return apply_geogebra_style(fig)


def _plot_funcion_tangente(f_np, res: ResultadoNR, iter_focus: int,
                            x_min: float, x_max: float) -> go.Figure:
    margen = (x_max - x_min) * 0.2
    xs = np.linspace(x_min - margen, x_max + margen, 600)
    try:
        ys = f_np(xs)
    except Exception:
        ys = np.array([f_np(xi) for xi in xs])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="f(x)",
                              line=dict(color="#1f77b4", width=2)))
    fig.add_hline(y=0, line=dict(color="black", width=0.7))

    if 1 <= iter_focus <= len(res.iteraciones):
        it = res.iteraciones[iter_focus - 1]
        # Tangente: y = f(xn) + f'(xn)·(x − xn)
        tan_y = it.f_xn + it.fp_xn * (xs - it.x_n)
        fig.add_trace(go.Scatter(
            x=xs, y=tan_y, mode="lines", name=f"tangente en x_{it.n}",
            line=dict(color="#ff7f0e", dash="dash", width=1.5),
        ))
        fig.add_trace(go.Scatter(
            x=[it.x_n, it.x_next], y=[it.f_xn, 0],
            mode="markers+text",
            marker=dict(size=[12, 14], color=["#d62728", "#2ca02c"]),
            text=[f"x_{it.n}={fmt_decimal(it.x_n)}", f"x_{it.n+1}={fmt_decimal(it.x_next)}"],
            textposition="top center", name=f"iter {it.n}",
        ))

    if res.raiz is not None:
        fig.add_vline(x=res.raiz, line=dict(color="green", dash="dash"),
                       annotation_text=f"raiz ≈ {fmt_decimal(res.raiz)}",
                       annotation_position="bottom right")

    fig.update_layout(
        title="f(x) y tangente en x_n (animada por iteracion)",
        xaxis_title="x", yaxis_title="f(x)",
        height=460, hovermode="x unified",
    )
    return fig


def _plot_convergencia(res: ResultadoNR) -> go.Figure:
    ns = [0] + [it.n for it in res.iteraciones]
    xs = [res.x0] + [it.x_next for it in res.iteraciones]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=xs, mode="lines+markers", name="x_n",
                              line=dict(color="#ff7f0e")))
    if res.raiz is not None:
        fig.add_hline(y=res.raiz, line=dict(color="green", dash="dash"),
                       annotation_text=f"raiz ≈ {fmt_decimal(res.raiz)}")
    fig.update_layout(title="Convergencia — x_n vs n",
                      xaxis_title="n", yaxis_title="x_n", height=380)
    return fig


def _plot_error_cuadratico(res: ResultadoNR) -> go.Figure:
    """Visualiza |e_n| en log. La convergencia cuadratica se ve como pendiente creciente."""
    ns = [it.n for it in res.iteraciones]
    errs = [it.error_abs for it in res.iteraciones]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=errs, mode="lines+markers",
                              name="E_abs real", line=dict(color="#d62728")))
    fig.update_layout(title="Error absoluto (escala log)",
                      xaxis_title="n", yaxis_title="E_abs",
                      yaxis_type="log", height=380)
    return fig


# --- Pasos ---

def _construir_pasos(res: ResultadoNR, fp_latex: str, n_pasos: int = 3) -> list[Paso]:
    pasos: list[Paso] = []
    if res.iteraciones:
        pasos.append(Paso(
            titulo="Pre-check: derivada no nula en x₀",
            formula=r"f'(x_0) \neq 0",
            sustitucion=rf"f'(x_0) = {fmt_decimal(res.iteraciones[0].fp_xn)}",
            explicacion_tecnica=(
                "Newton-Raphson usa la recta tangente: si f'(x_n) ≈ 0, la tangente es "
                "horizontal y el metodo se rompe (division por cero)."
            ),
            explicacion_coloquial=(
                "La tangente tiene que tener pendiente. Si f' da 0, la recta no corta "
                "el eje x y no sabemos para donde saltar."
            ),
            notas=(rf"$f'(x) = {fp_latex}$",),
        ))

    for it in res.iteraciones[:n_pasos]:
        pasos.append(Paso(
            titulo=f"Iteracion {it.n} — aplicar la formula de Newton",
            formula=r"x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}",
            sustitucion=(
                rf"x_{{{it.n}}} = {fmt_decimal(it.x_n)} - "
                rf"\frac{{{fmt_decimal(it.f_xn)}}}{{{fmt_decimal(it.fp_xn)}}} "
                rf"= {fmt_decimal(it.x_next)}"
            ),
            resultado=rf"|x_{{{it.n}}} - x_{{{it.n-1}}}| = {fmt_decimal(it.error_abs)}",
            explicacion_tecnica=(
                f"f({fmt_decimal(it.x_n)}) = {fmt_decimal(it.f_xn)}, "
                f"f'({fmt_decimal(it.x_n)}) = {fmt_decimal(it.fp_xn)}. "
                f"La tangente corta el eje x en x_{it.n} = {fmt_decimal(it.x_next)}."
            ),
            explicacion_coloquial=(
                f"Desde {fmt_decimal(it.x_n)} disparamos la tangente y vemos donde cruza "
                f"el eje x: {fmt_decimal(it.x_next)}. Nos movimos {fmt_decimal(it.error_abs)}."
            ),
        ))

    if res.convergio and res.raiz is not None:
        pasos.append(Paso(
            titulo="Corte del metodo",
            resultado=rf"\hat{{x}} \approx {fmt_decimal(res.raiz)}",
            explicacion_tecnica=f"Criterio aplicado: {res.motivo_corte}.",
            explicacion_coloquial=f"Cortamos: {res.motivo_corte}.",
        ))

    return pasos


def _tabla_dataframe(res: ResultadoNR) -> pd.DataFrame:
    filas = []
    for it in res.iteraciones:
        filas.append({
            "n": it.n,
            "x_n": it.x_n,
            "f(x_n)": it.f_xn,
            "f'(x_n)": it.fp_xn,
            "x_n+1": it.x_next,
            "E_abs": it.error_abs,
            "E_rel": it.error_rel,
        })
    return pd.DataFrame(filas)


# --- Render principal ---

def render_newton_raphson() -> None:
    st.header("Metodo de Newton-Raphson")
    st.caption("x_{n+1} = x_n − f(x_n)/f'(x_n) — prof. Caceres, pg 15–16")

    cfg = get_config()
    render_teoria("raices_teoria", titulo="Teoria del libro (Caceres pg 15–16)",
                   expanded=False)

    st.subheader("Entrada")

    col_preset, col_info = st.columns([2, 3])
    with col_preset:
        preset_key = st.selectbox("Preset de la guia",
                                    options=list(EJERCICIOS.keys()),
                                    index=0,
                                    help="Ejercicios oficiales (pg 19).")
    preset = EJERCICIOS[preset_key]
    with col_info:
        if preset:
            st.info(f"**{preset_key}**")

    default_latex = preset.get("latex", r"x^{3} - 2x - 5")
    latex = math_input("f(x) =", default_latex=default_latex,
                        key=f"nr_fx_{preset_key}")

    x_sym = sp.Symbol("x")
    f_expr, f_np = parse_latex(latex, [x_sym])

    if f_expr is None or f_np is None:
        st.warning("Ingresa una f(x) valida.")
        return

    st.latex(rf"f(x) = {sp.latex(f_expr)}")

    # Derivada simbolica
    try:
        fp_expr = sp.diff(f_expr, x_sym)
        fp_np = sp.lambdify(x_sym, fp_expr, modules=["numpy"])
    except Exception as e:
        st.error(f"No se pudo derivar f(x) simbolicamente: {e}")
        return

    st.latex(rf"f'(x) = {sp.latex(sp.simplify(fp_expr))}")

    col_x0, col_info2 = st.columns([1, 2])
    with col_x0:
        x0 = st.number_input("x₀ (punto inicial)",
                              value=float(preset.get("x0", 1.0)),
                              format="%.6f",
                              help="Aproximacion inicial — cuanto mas cerca de la raiz, mejor.")

    try:
        fp_x0 = float(fp_np(x0))
        f_x0 = float(f_np(x0))
    except Exception as e:
        st.error(f"Error al evaluar f o f' en x₀={x0}: {e}")
        return

    with col_info2:
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("f(x₀)", fmt_decimal(f_x0))
        col_b.metric("f'(x₀)", fmt_decimal(fp_x0),
                      delta="OK" if abs(fp_x0) > 1e-10 else "≈ 0 ⚠",
                      delta_color="normal" if abs(fp_x0) > 1e-10 else "inverse")
        col_c.metric("x₁ (primer paso)",
                      fmt_decimal(x0 - f_x0 / fp_x0) if abs(fp_x0) > 1e-14 else "—")

    if abs(fp_x0) < 1e-14:
        st.error(
            "**f'(x₀) ≈ 0**: la tangente es horizontal. "
            "Newton-Raphson no puede arrancar — elegi otro x₀."
        )
        return

    # Criterios
    with st.expander("Criterios de detencion (4 oficiales)", expanded=not cfg.modo_examen):
        tol_default = float(preset.get("tol", 1e-6))
        col1, col2 = st.columns(2)
        with col1:
            usar_abs = st.checkbox("Error absoluto |x_n+1 − x_n| ≤ ε",
                                    value=True, key="nr_ua")
            tol_abs = st.number_input("ε absoluto", value=tol_default,
                                       format="%.1e", min_value=0.0,
                                       disabled=not usar_abs, key="nr_ta")
            usar_rel = st.checkbox("Error relativo", value=False, key="nr_ur")
            tol_rel = st.number_input("ε relativo", value=tol_default,
                                       format="%.1e", min_value=0.0,
                                       disabled=not usar_rel, key="nr_tr")
        with col2:
            usar_res = st.checkbox("Residuo |f(x)| ≤ ε",
                                    value=True, key="nr_ures")
            tol_res = st.number_input("ε residuo", value=tol_default,
                                       format="%.1e", min_value=0.0,
                                       disabled=not usar_res, key="nr_tres")
            usar_max = st.checkbox("Max iteraciones", value=True, key="nr_um")
            max_iter = st.number_input("N_max", value=100, min_value=1, max_value=10_000,
                                        disabled=not usar_max, key="nr_nmax")

    criterios = CriteriosDetencion(
        usar_abs=usar_abs, tol_abs=tol_abs,
        usar_rel=usar_rel, tol_rel=tol_rel,
        usar_residuo=usar_res, tol_residuo=tol_res,
        usar_max_iter=usar_max, max_iter=int(max_iter),
    )

    precision_usada = 5 if cfg.modo_libro else None
    res = newton_raphson(f_np, fp_np, x0, criterios, precision=precision_usada)

    if not res.iteraciones:
        st.error(res.motivo_corte or "No se pudieron ejecutar iteraciones.")
        return

    # --- Resumen ---
    st.subheader("Resultado")
    if res.convergio:
        st.success(f"Convergio — {res.motivo_corte}")
    else:
        st.warning(f"NO convergio — {res.motivo_corte}")

    col_r, col_n, col_fr, col_e = st.columns(4)
    col_r.metric("Raiz aproximada", fmt_decimal(res.raiz))
    col_n.metric("Iteraciones", len(res.iteraciones))
    col_fr.metric("f(raiz)",
                   fmt_decimal(float(f_np(res.raiz))) if res.raiz is not None else "—")
    col_e.metric("E_abs final", fmt_decimal(res.iteraciones[-1].error_abs))

    # Diagnostico de orden empirico
    errs = [it.error_abs for it in res.iteraciones
             if np.isfinite(it.error_abs) and it.error_abs > 1e-15]
    if len(errs) >= 4:
        log_e = np.log(errs)
        try:
            slope, _ = np.polyfit(log_e[:-1], log_e[1:], 1)
            tipo = ("cuadratica (raiz simple)" if abs(slope - 2) < 0.3
                     else "lineal (raiz multiple)" if abs(slope - 1) < 0.3
                     else f"orden {slope:.2f}")
            st.caption(
                f"**Orden empirico de convergencia**: p ≈ {slope:.3f} — {tipo}. "
                f"(teorico: 2 para raiz simple, 1 para multiple)"
            )
        except Exception:
            pass

    tab_resumen, tab_pasos, tab_viz = st.tabs(["Resumen", "Paso a paso", "Visualizaciones"])

    # Intervalo para el plot estatico: cubre x0, todas las iteraciones y la raiz,
    # mas un margen del 25%.
    xs_all = [res.x0] + [it.x_n for it in res.iteraciones] + [it.x_next for it in res.iteraciones]
    if res.raiz is not None:
        xs_all.append(res.raiz)
    a_plot, b_plot = min(xs_all), max(xs_all)
    if a_plot == b_plot:
        a_plot, b_plot = a_plot - 1.0, b_plot + 1.0
    else:
        span = b_plot - a_plot
        a_plot -= span * 0.25
        b_plot += span * 0.25

    with tab_resumen:
        st.markdown("**Grafico para examen** — curva f(x) con la raiz ubicada:")
        st.plotly_chart(_plot_curva_con_raiz(f_np, res, a_plot, b_plot),
                         use_container_width=True)
        df = _tabla_dataframe(res)
        resaltar = resaltar_tolerancia("E_abs", criterios.tol_abs) if criterios.usar_abs else None
        render_tabla_iteraciones(df, resaltar=resaltar,
                                  titulo="Tabla de iteraciones",
                                  key_export="newton_raphson")

    with tab_pasos:
        n_pasos = st.slider("Iteraciones a explicar", 1,
                             min(10, len(res.iteraciones)),
                             value=min(3, len(res.iteraciones)), key="nr_np")
        render_pasos(_construir_pasos(res,
                                        fp_latex=sp.latex(sp.simplify(fp_expr)),
                                        n_pasos=n_pasos), titulo="")

    with tab_viz:
        xs_iter = [res.x0] + [it.x_n for it in res.iteraciones] + [it.x_next for it in res.iteraciones]
        if res.raiz is not None:
            xs_iter.append(res.raiz)
        x_min, x_max = min(xs_iter), max(xs_iter)
        if x_min == x_max:
            x_min, x_max = x_min - 1, x_max + 1

        iter_focus = st.slider("Iteracion a visualizar (tangente)", 1,
                                len(res.iteraciones), value=1, key="nr_focus")
        st.plotly_chart(_plot_funcion_tangente(f_np, res, iter_focus, x_min, x_max),
                         use_container_width=True)
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.plotly_chart(_plot_convergencia(res), use_container_width=True)
        with col_g2:
            st.plotly_chart(_plot_error_cuadratico(res), use_container_width=True)

    if cfg.mostrar_glosario:
        render_glosario_expander(
            vars=["x_0", "x_n", "f", "f_prima", "x_star", "E_abs", "E_r",
                   "epsilon", "N_max", "n"],
            titulo="Glosario — Newton-Raphson",
        )


def render() -> None:
    render_newton_raphson()
