"""Modulo Derivacion Numerica — diferencias finitas (Caceres pg 24-26).

Tres esquemas: progresiva, regresiva, central, para 1ra y 2da derivada.
Convencion de catedra (pg 26): centrales en interiores, progresivas/regresivas en extremos.

Features:
- Modo "desde f(x)": deriva una funcion en una grilla de puntos con paso h
  y compara contra la derivada exacta.
- Modo "desde tabla": ingresar t/x y calcular v = dx/dt y a = d²x/dt² (ej 6-7 pg 26).
- Banco de ejercicios oficiales (pg 26).
- Plot: f y sus derivadas por metodo + error absoluto vs h (estudio de convergencia).
- 3 tabs: Resumen, Paso a paso, Visualizaciones.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sympy as sp

from utils.math_keyboard import math_input, parse_latex
from utils.ui.config import get_config
from utils.ui.glosario import render_glosario_expander
from utils.ui.pasos import Paso, render_pasos
from utils.ui.tablas import fmt_decimal, render_tabla_iteraciones
from utils.ui.teoria import render_teoria

# --- Banco de ejercicios (Caceres pg 26) ---

EJERCICIOS: dict[str, dict] = {
    "Personalizado": {},
    "Ej 1 — f(x)=sin(x), x=[0..0.5], h=0.1 (central)": {
        "modo": "funcion", "latex": r"\sin(x)",
        "x_grid": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], "h": 0.1,
    },
    "Ej 2 — f(x)=e^x, x=[0..0.5], h=0.1": {
        "modo": "funcion", "latex": r"e^{x}",
        "x_grid": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], "h": 0.1,
    },
    "Ej 3 — f(x)=x³−x en x=1, h=0.1": {
        "modo": "funcion", "latex": r"x^{3} - x",
        "x_grid": [1.0], "h": 0.1,
    },
    "Ej 4 — f(x)=e^x·sin(x) en x=1, h=0.01": {
        "modo": "funcion", "latex": r"e^{x}\sin(x)",
        "x_grid": [1.0], "h": 0.01,
    },
    "Ej 5 — f(x)=e^(−2x)−x en x=2 (comparar 3 esquemas)": {
        "modo": "funcion", "latex": r"e^{-2x} - x",
        "x_grid": [2.0], "h": 0.1,
    },
    "Ej 6 — tabla cinematica t=[0..8]": {
        "modo": "tabla",
        "t_pts": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "x_pts": [0.0, 1.9, 4.2, 7.8, 12.0, 17.0, 25.0, 32.0, 42.0],
    },
    "Ej 7 — tabla cinematica t=[0..16] paso 2": {
        "modo": "tabla",
        "t_pts": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
        "x_pts": [0.0, 0.7, 2.0, 3.5, 5.4, 6.5, 7.5, 8.5, 8.5],
    },
}


# --- Formulas ---

def df_progresiva(f, x: float, h: float) -> float:
    return (f(x + h) - f(x)) / h


def df_regresiva(f, x: float, h: float) -> float:
    return (f(x) - f(x - h)) / h


def df_central(f, x: float, h: float) -> float:
    return (f(x + h) - f(x - h)) / (2 * h)


def d2f_progresiva(f, x: float, h: float) -> float:
    return (f(x + 2 * h) - 2 * f(x + h) + f(x)) / (h ** 2)


def d2f_regresiva(f, x: float, h: float) -> float:
    return (f(x) - 2 * f(x - h) + f(x - 2 * h)) / (h ** 2)


def d2f_central(f, x: float, h: float) -> float:
    return (f(x + h) - 2 * f(x) + f(x - h)) / (h ** 2)


# --- Diferencias desde tabla (nodos equiespaciados) ---

def derivar_tabla(t_pts: list[float], x_pts: list[float]) -> pd.DataFrame:
    """Para una tabla t/x con paso uniforme, devuelve velocidad y aceleracion.
    Usa centrales en interiores; progresivas/regresivas en los extremos.
    """
    t = np.asarray(t_pts, dtype=float)
    x = np.asarray(x_pts, dtype=float)
    n = len(t)
    if n < 3:
        raise ValueError("Se requieren al menos 3 puntos para derivar.")
    h = t[1] - t[0]
    # Validacion paso uniforme
    if not np.allclose(np.diff(t), h, rtol=1e-6):
        raise ValueError("El paso entre tiempos debe ser uniforme.")

    v = np.zeros(n)
    a = np.zeros(n)
    metodo_v: list[str] = []
    metodo_a: list[str] = []

    for i in range(n):
        if i == 0:
            v[i] = (x[i + 1] - x[i]) / h
            a[i] = (x[i + 2] - 2 * x[i + 1] + x[i]) / (h ** 2)
            metodo_v.append("progresiva")
            metodo_a.append("progresiva")
        elif i == n - 1:
            v[i] = (x[i] - x[i - 1]) / h
            a[i] = (x[i] - 2 * x[i - 1] + x[i - 2]) / (h ** 2)
            metodo_v.append("regresiva")
            metodo_a.append("regresiva")
        else:
            v[i] = (x[i + 1] - x[i - 1]) / (2 * h)
            a[i] = (x[i + 1] - 2 * x[i] + x[i - 1]) / (h ** 2)
            metodo_v.append("central")
            metodo_a.append("central")

    return pd.DataFrame({
        "t(seg)": t,
        "x(m)": x,
        "v(m/s)": v,
        "metodo v": metodo_v,
        "a(m/s²)": a,
        "metodo a": metodo_a,
    })


# --- Derivacion desde f(x) con comparacion contra exacta ---

@dataclass
class FilaDerivacion:
    x: float
    f_x: float
    d_exact: float | None
    d_prog: float
    d_regr: float
    d_cent: float
    d2_cent: float
    err_prog: float | None
    err_regr: float | None
    err_cent: float | None


def derivar_funcion(f_np, f_expr: sp.Expr, x_sym: sp.Symbol,
                     xs: list[float], h: float) -> pd.DataFrame:
    """Para cada x en la grilla, calcula f'(x) por los 3 metodos + 2da derivada central.
    Si se puede derivar simbolicamente, agrega error absoluto.
    """
    try:
        fp_expr = sp.diff(f_expr, x_sym)
        fp_np = sp.lambdify(x_sym, fp_expr, modules=["numpy"])
    except Exception:
        fp_np = None

    filas = []
    for x in xs:
        f_x = float(f_np(x))
        d_prog = df_progresiva(f_np, x, h)
        d_regr = df_regresiva(f_np, x, h)
        d_cent = df_central(f_np, x, h)
        d2_cent = d2f_central(f_np, x, h)
        d_exact = None
        err_prog = err_regr = err_cent = None
        if fp_np is not None:
            try:
                d_exact = float(fp_np(x))
                err_prog = abs(d_prog - d_exact)
                err_regr = abs(d_regr - d_exact)
                err_cent = abs(d_cent - d_exact)
            except Exception:
                pass

        fila = {
            "x": x,
            "f(x)": f_x,
            "f'(x) progresiva": d_prog,
            "f'(x) regresiva": d_regr,
            "f'(x) central": d_cent,
            "f''(x) central": d2_cent,
        }
        if d_exact is not None:
            fila["f'(x) exacta"] = d_exact
            fila["err prog"] = err_prog
            fila["err regr"] = err_regr
            fila["err cent"] = err_cent
        filas.append(fila)

    return pd.DataFrame(filas)


# --- Plots ---

def _plot_derivadas(f_np, f_expr: sp.Expr, x_sym: sp.Symbol,
                     x_min: float, x_max: float, h: float) -> go.Figure:
    margen = (x_max - x_min) * 0.1
    xs = np.linspace(x_min - margen, x_max + margen, 200)
    ys_f = np.asarray(f_np(xs), dtype=float)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys_f, mode="lines", name="f(x)",
                              line=dict(color="#1f77b4", width=2)))

    try:
        fp_np = sp.lambdify(x_sym, sp.diff(f_expr, x_sym), modules=["numpy"])
        ys_exact = np.asarray(fp_np(xs), dtype=float)
        fig.add_trace(go.Scatter(x=xs, y=ys_exact, mode="lines",
                                  name="f'(x) exacta",
                                  line=dict(color="#2ca02c", width=2, dash="dash")))
    except Exception:
        pass

    # Aproximaciones en grilla (centrales)
    xs_grid = np.linspace(x_min, x_max, 15)
    d_cent = np.array([df_central(f_np, float(x), h) for x in xs_grid])
    fig.add_trace(go.Scatter(x=xs_grid, y=d_cent, mode="markers",
                              marker=dict(size=10, color="#d62728"),
                              name=f"f'(x) central (h={h})"))

    fig.update_layout(title="f(x) y su derivada (exacta vs central)",
                      xaxis_title="x", yaxis_title="y",
                      height=400, hovermode="x unified")
    return fig


def _plot_error_vs_h(f_np, f_expr: sp.Expr, x_sym: sp.Symbol, x0: float) -> go.Figure:
    """Estudio de convergencia: error absoluto vs h, en log-log."""
    try:
        fp_np = sp.lambdify(x_sym, sp.diff(f_expr, x_sym), modules=["numpy"])
        d_exact = float(fp_np(x0))
    except Exception:
        return go.Figure().update_layout(title="No se pudo derivar simbolicamente.")

    hs = np.logspace(-10, -1, 40)
    err_p = [abs(df_progresiva(f_np, x0, h) - d_exact) for h in hs]
    err_r = [abs(df_regresiva(f_np, x0, h) - d_exact) for h in hs]
    err_c = [abs(df_central(f_np, x0, h) - d_exact) for h in hs]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hs, y=err_p, mode="lines+markers",
                              name="progresiva O(h)", line=dict(color="#ff7f0e")))
    fig.add_trace(go.Scatter(x=hs, y=err_r, mode="lines+markers",
                              name="regresiva O(h)", line=dict(color="#9467bd")))
    fig.add_trace(go.Scatter(x=hs, y=err_c, mode="lines+markers",
                              name="central O(h²)", line=dict(color="#2ca02c")))
    fig.update_layout(
        title=f"Error absoluto vs h en x={fmt_decimal(x0)} (log-log) — "
               f"se ve truncamiento (h grande) y redondeo (h chico)",
        xaxis_title="h", yaxis_title="|error|",
        xaxis_type="log", yaxis_type="log",
        height=400,
    )
    return fig


# --- Pasos ---

def _pasos_formulas(tipo: str = "central") -> list[Paso]:
    pasos: list[Paso] = []
    if tipo == "central":
        pasos.append(Paso(
            titulo="Diferencias centrales — mas preciso O(h²)",
            formula=r"f'(x_i) \approx \frac{f(x_{i+1}) - f(x_{i-1})}{2h}",
            explicacion_tecnica=(
                "Usa un vecino a cada lado del punto. Error de orden O(h²) — mas chico "
                "que progresiva/regresiva O(h) para el mismo h."
            ),
            explicacion_coloquial=(
                "Miramos a los dos vecinos (izquierda y derecha) y promediamos. Mas preciso."
            ),
        ))
        pasos.append(Paso(
            titulo="Segunda derivada central",
            formula=r"f''(x_i) \approx \frac{f(x_{i+1}) - 2 f(x_i) + f(x_{i-1})}{h^2}",
            explicacion_tecnica="Tres puntos consecutivos. Error O(h²).",
            explicacion_coloquial=(
                "Como la aceleracion: cuanto cambia la pendiente al pasar de un vecino al otro."
            ),
        ))
    elif tipo == "progresiva":
        pasos.append(Paso(
            titulo="Diferencias progresivas",
            formula=r"f'(x_i) \approx \frac{f(x_{i+1}) - f(x_i)}{h}",
            explicacion_tecnica="Error O(h). Uso cuando no hay vecino a la izquierda (extremo izquierdo).",
            explicacion_coloquial="Miramos solo al vecino de la derecha.",
        ))
    elif tipo == "regresiva":
        pasos.append(Paso(
            titulo="Diferencias regresivas",
            formula=r"f'(x_i) \approx \frac{f(x_i) - f(x_{i-1})}{h}",
            explicacion_tecnica="Error O(h). Uso cuando no hay vecino a la derecha (extremo derecho).",
            explicacion_coloquial="Miramos solo al vecino de la izquierda.",
        ))
    return pasos


# --- Render principal ---

def render_derivacion() -> None:
    st.header("Derivacion Numerica — Diferencias Finitas")
    st.caption("Esquemas progresivo / regresivo / central — prof. Caceres, pg 24–26")

    cfg = get_config()
    render_teoria("interpolacion_teoria",
                   titulo="Teoria del libro (pg 24–26)",
                   seccion="4. Diferencias finitas",
                   expanded=False)

    preset_key = st.selectbox("Preset de la guia",
                                options=list(EJERCICIOS.keys()),
                                index=0,
                                help="Ejercicios oficiales pg 26.")
    preset = EJERCICIOS[preset_key]

    modo_default = preset.get("modo", "funcion")
    modo = st.radio("Modo de entrada",
                     options=["Desde f(x)", "Desde tabla (cinematica)"],
                     index=0 if modo_default == "funcion" else 1,
                     horizontal=True)

    if modo == "Desde f(x)":
        _render_modo_funcion(preset, preset_key, cfg)
    else:
        _render_modo_tabla(preset, preset_key, cfg)

    if cfg.mostrar_glosario:
        render_glosario_expander(
            vars=["f", "f_prima", "n"],
            titulo="Glosario — Diferencias Finitas",
        )


def _render_modo_funcion(preset: dict, preset_key: str, cfg) -> None:
    default_latex = preset.get("latex", r"\sin(x)")

    # Integracion con Lagrange: si ya se construyo P(x), ofrecer importarlo.
    P_latex = st.session_state.get("shared_lagrange_P_latex")
    nodos_lag = st.session_state.get("shared_lagrange_nodes", [])
    if P_latex:
        col_imp, col_info = st.columns([1, 2])
        with col_imp:
            if st.button("📥 Usar P(x) de Lagrange",
                          key=f"deriv_usar_P_{preset_key}",
                          help="Carga el polinomio interpolante construido en la "
                                "pestana Lagrange. Responde 'derivar la funcion reconstruida'."):
                st.session_state[f"deriv_fx_{preset_key}"] = P_latex
                st.rerun()
        with col_info:
            rango = (f"[{fmt_decimal(min(nodos_lag))}, {fmt_decimal(max(nodos_lag))}]"
                      if nodos_lag else "—")
            st.caption(
                f"P(x) disponible de Lagrange (nodos en {rango}). "
                "Al derivar P(x) estas aplicando diferencias finitas sobre la "
                "**funcion reconstruida** (no sobre f original). Ojo: los puntos de "
                "derivacion ± h deberian caer dentro del rango de interpolacion."
            )

    latex = math_input("f(x) =", default_latex=default_latex,
                        key=f"deriv_fx_{preset_key}")
    x_sym = sp.Symbol("x")
    f_expr, f_np = parse_latex(latex, [x_sym])
    if f_expr is None or f_np is None:
        st.warning("Ingresa una f(x) valida.")
        return

    st.latex(rf"f(x) = {sp.latex(f_expr)}")

    # Intentar mostrar derivada simbolica
    try:
        fp_expr = sp.simplify(sp.diff(f_expr, x_sym))
        st.caption(f"Derivada simbolica: $f'(x) = {sp.latex(fp_expr)}$")
    except Exception:
        fp_expr = None

    x_grid_default = preset.get("x_grid", [1.0])
    col_grid, col_h = st.columns(2)
    with col_grid:
        grid_str = st.text_input(
            "Puntos donde derivar (separados por coma)",
            value=", ".join(fmt_decimal(v) for v in x_grid_default),
            key=f"deriv_grid_{preset_key}",
        )
    with col_h:
        h = st.number_input("Paso h", value=float(preset.get("h", 0.1)),
                              format="%.6f", min_value=1e-12, step=0.01,
                              key=f"deriv_h_{preset_key}")

    try:
        xs = [float(s.strip()) for s in grid_str.split(",") if s.strip()]
    except Exception as e:
        st.error(f"No se pudieron parsear los puntos: {e}")
        return

    df = derivar_funcion(f_np, f_expr, x_sym, xs, h)

    tab_resumen, tab_pasos, tab_viz = st.tabs(["Resumen", "Paso a paso", "Visualizaciones"])

    with tab_resumen:
        render_tabla_iteraciones(df, titulo="Tabla de derivadas por metodo",
                                  key_export="derivacion_funcion")
        # Comentario analitico
        if "err cent" in df.columns:
            err_c = float(df["err cent"].mean())
            err_p = float(df["err prog"].mean())
            st.info(
                f"**Error promedio** — central: {fmt_decimal(err_c)}, "
                f"progresiva: {fmt_decimal(err_p)}. "
                f"La central es O(h²), las otras O(h): para h={h} se ve la diferencia."
            )

    with tab_pasos:
        pasos = _pasos_formulas("central") + _pasos_formulas("progresiva") + _pasos_formulas("regresiva")
        render_pasos(pasos, titulo="")

    with tab_viz:
        if len(xs) > 1:
            st.plotly_chart(
                _plot_derivadas(f_np, f_expr, x_sym, min(xs), max(xs), h),
                use_container_width=True,
            )
        if len(xs) >= 1:
            st.markdown("#### Estudio de convergencia — error vs h")
            x_studio = st.selectbox("Punto para estudiar convergencia", xs, index=0,
                                     key=f"deriv_studio_{preset_key}")
            st.plotly_chart(
                _plot_error_vs_h(f_np, f_expr, x_sym, float(x_studio)),
                use_container_width=True,
            )


def _render_modo_tabla(preset: dict, preset_key: str, cfg) -> None:
    t_default = preset.get("t_pts", [0.0, 1.0, 2.0, 3.0, 4.0])
    x_default = preset.get("x_pts", [0.0, 1.0, 4.0, 9.0, 16.0])

    df_input = pd.DataFrame({"t(seg)": t_default, "x(m)": x_default})
    df_edit = st.data_editor(
        df_input, num_rows="dynamic", use_container_width=True,
        key=f"deriv_tabla_{preset_key}",
        column_config={
            "t(seg)": st.column_config.NumberColumn("t(seg)", format="%.4f"),
            "x(m)": st.column_config.NumberColumn("x(m)", format="%.4f"),
        },
    )

    try:
        t_pts = df_edit["t(seg)"].dropna().astype(float).tolist()
        x_pts = df_edit["x(m)"].dropna().astype(float).tolist()
    except Exception as e:
        st.error(f"Error al parsear la tabla: {e}")
        return

    if len(t_pts) != len(x_pts) or len(t_pts) < 3:
        st.warning("Se requieren al menos 3 filas con t y x.")
        return

    try:
        df_res = derivar_tabla(t_pts, x_pts)
    except ValueError as e:
        st.error(str(e))
        return

    st.subheader("Velocidad y aceleracion")
    render_tabla_iteraciones(df_res, titulo=None, key_export="derivacion_tabla")

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_res["t(seg)"], y=df_res["x(m)"],
                              mode="lines+markers", name="x(t)",
                              line=dict(color="#1f77b4", width=2)))
    fig.add_trace(go.Scatter(x=df_res["t(seg)"], y=df_res["v(m/s)"],
                              mode="lines+markers", name="v(t)",
                              line=dict(color="#ff7f0e", width=2)))
    fig.add_trace(go.Scatter(x=df_res["t(seg)"], y=df_res["a(m/s²)"],
                              mode="lines+markers", name="a(t)",
                              line=dict(color="#d62728", width=2)))
    fig.update_layout(title="Cinematica — posicion, velocidad, aceleracion",
                      xaxis_title="t (seg)", yaxis_title="valor",
                      height=440, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # Analisis
    v = df_res["v(m/s)"].values
    a = df_res["a(m/s²)"].values
    comentario = []
    if np.all(np.diff(v) > 0):
        comentario.append("La **velocidad aumenta** monotonamente.")
    elif np.all(np.diff(v) < 0):
        comentario.append("La **velocidad disminuye** monotonamente.")
    else:
        comentario.append("La velocidad tiene cambios de signo — hay aceleracion y frenado.")

    if np.mean(a) > 0.01:
        comentario.append(f"Aceleracion promedio positiva: {fmt_decimal(float(np.mean(a)))} m/s².")
    elif np.mean(a) < -0.01:
        comentario.append(f"Aceleracion promedio negativa (frenando): {fmt_decimal(float(np.mean(a)))} m/s².")
    else:
        comentario.append("Aceleracion promedio ≈ 0 (movimiento cuasi-uniforme).")

    st.info("  \n".join(comentario))


def render() -> None:
    render_derivacion()
