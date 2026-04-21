"""Modulo Lagrange — polinomio interpolante.

Implementa la interpolacion tal como la define el prof. Caceres (pg 20-23):
  P(x) = Σ y_i · L_i(x), con L_i(x) = Π (x - x_j)/(x_i - x_j) para j ≠ i.

Features:
- Banco de ejercicios oficiales (pg 25-26) como presets.
- Tabla editable de puntos (x_i, y_i).
- f(x) opcional para comparar vs polinomio y calcular cota de error.
- Muestra P(x) simbolicamente expandido + cada base L_i(x).
- Evaluacion del polinomio en un x consultado (interp vs extrapolacion).
- Plot: puntos + curva + (si hay f) original + region extrapolacion sombreada.
- Error local y cota teorica cuando se dispone de f(x).
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

# --- Banco de ejercicios (Caceres pg 25-26) ---

EJERCICIOS: dict[str, dict] = {
    "Personalizado": {},
    "Ej 1 — puntos (1,1),(2,4),(3,9) [reconstruye x²]": {
        "x_pts": [1.0, 2.0, 3.0], "y_pts": [1.0, 4.0, 9.0],
        "f_latex": r"x^{2}", "x_eval": 2.5,
    },
    "Ej 2 — (0,1),(1,3),(2,2),(3,5)": {
        "x_pts": [0.0, 1.0, 2.0, 3.0], "y_pts": [1.0, 3.0, 2.0, 5.0],
        "x_eval": 1.5,
    },
    "Ej 4 — x=[0,1,2,3,4], f=[1,2,0,2,3]": {
        "x_pts": [0.0, 1.0, 2.0, 3.0, 4.0],
        "y_pts": [1.0, 2.0, 0.0, 2.0, 3.0], "x_eval": 2.5,
    },
    "Ej 5 — x=[0,1,2], y=[1,3,0]": {
        "x_pts": [0.0, 1.0, 2.0], "y_pts": [1.0, 3.0, 0.0], "x_eval": 0.5,
    },
    "Ej 6 — x=[1,2,3], y=[10,15,80]": {
        "x_pts": [1.0, 2.0, 3.0], "y_pts": [10.0, 15.0, 80.0], "x_eval": 2.5,
    },
    "Ej 7 — x=[2,4,5], f=[5,6,3]": {
        "x_pts": [2.0, 4.0, 5.0], "y_pts": [5.0, 6.0, 3.0], "x_eval": 3.0,
    },
    "Ej 8 — x=[-2,0,2], f=[0,1,0]": {
        "x_pts": [-2.0, 0.0, 2.0], "y_pts": [0.0, 1.0, 0.0], "x_eval": 1.0,
    },
    "Ej 9 — sin(x) en [0,π] grado 2": {
        "x_pts": [0.0, float(np.pi) / 2, float(np.pi)],
        "y_pts": [0.0, 1.0, 0.0],
        "f_latex": r"\sin(x)", "x_eval": float(np.pi) / 4,
    },
    "Ej 10 — x=[0,1,2], f=[1,2,7]": {
        "x_pts": [0.0, 1.0, 2.0], "y_pts": [1.0, 2.0, 7.0], "x_eval": 1.5,
    },
    "Ej 11 — f=1/x, nodos 2, 2.5, 4.5": {
        "x_pts": [2.0, 2.5, 4.5], "y_pts": [0.5, 0.4, 1.0 / 4.5],
        "f_latex": r"1/x", "x_eval": 3.0,
    },
    "Ej 12 — f=2sin(πx/6), nodos 1,2,3 — aprox f(4)": {
        "x_pts": [1.0, 2.0, 3.0],
        "y_pts": [2 * float(np.sin(np.pi / 6)),
                   2 * float(np.sin(2 * np.pi / 6)),
                   2 * float(np.sin(3 * np.pi / 6))],
        "f_latex": r"2\sin\left(\pi x/6\right)", "x_eval": 4.0,
    },
    "Ej 12b — f=2sin(πx/6), nodos 1,2,3 — aprox f(1.5)": {
        "x_pts": [1.0, 2.0, 3.0],
        "y_pts": [2 * float(np.sin(np.pi / 6)),
                   2 * float(np.sin(2 * np.pi / 6)),
                   2 * float(np.sin(3 * np.pi / 6))],
        "f_latex": r"2\sin\left(\pi x/6\right)", "x_eval": 1.5,
    },
    "Ej 13a — cos(x), nodos 0,0.6,0.9 — aprox f(0.45)": {
        "x_pts": [0.0, 0.6, 0.9],
        "y_pts": [1.0, float(np.cos(0.6)), float(np.cos(0.9))],
        "f_latex": r"\cos(x)", "x_eval": 0.45,
    },
    "Ej 13b — sqrt(x+1), nodos 0,0.6,0.9 — aprox f(0.45)": {
        "x_pts": [0.0, 0.6, 0.9],
        "y_pts": [1.0, float(np.sqrt(1.6)), float(np.sqrt(1.9))],
        "f_latex": r"\sqrt{x+1}", "x_eval": 0.45,
    },
    "Ej 13c — ln(x+1), nodos 0,0.6,0.9 — aprox f(0.45)": {
        "x_pts": [0.0, 0.6, 0.9],
        "y_pts": [0.0, float(np.log(1.6)), float(np.log(1.9))],
        "f_latex": r"\ln(x+1)", "x_eval": 0.45,
    },
}


@dataclass
class ResultadoLagrange:
    x_pts: list[float]
    y_pts: list[float]
    P_expr: sp.Expr
    P_expandido: sp.Expr
    bases: list[sp.Expr]
    x_sym: sp.Symbol


def construir_lagrange(x_pts: list[float], y_pts: list[float]) -> ResultadoLagrange:
    """Construye simbolicamente el polinomio P(x) y cada base L_i(x)."""
    x = sp.Symbol("x")
    n = len(x_pts)
    bases: list[sp.Expr] = []
    P = sp.Integer(0)
    for i in range(n):
        li = sp.Integer(1)
        for j in range(n):
            if i != j:
                li *= (x - sp.Rational(str(x_pts[j]))) / (
                    sp.Rational(str(x_pts[i])) - sp.Rational(str(x_pts[j]))
                )
        li_simpl = sp.simplify(li)
        bases.append(li_simpl)
        P += sp.Rational(str(y_pts[i])) * li_simpl if _is_rational(y_pts[i]) else y_pts[i] * li_simpl
    P_exp = sp.expand(P)
    return ResultadoLagrange(
        x_pts=list(x_pts), y_pts=list(y_pts),
        P_expr=P, P_expandido=P_exp, bases=bases, x_sym=x,
    )


def _is_rational(v: float) -> bool:
    try:
        sp.Rational(str(v))
        return True
    except Exception:
        return False


def evaluar(res: ResultadoLagrange, x0: float) -> float:
    try:
        return float(res.P_expandido.subs(res.x_sym, x0).evalf())
    except Exception:
        return float("nan")


def cota_error(f_expr: sp.Expr, x_sym: sp.Symbol,
                x_pts: list[float], x_eval: float,
                n_muestra: int = 201) -> dict:
    """Cota de Lagrange: |E(x)| ≤ M_{n+1}/(n+1)! · |Π (x - x_i)|."""
    n = len(x_pts) - 1
    try:
        fn1 = sp.diff(f_expr, x_sym, n + 1)
        fn1_np = sp.lambdify(x_sym, fn1, modules=["numpy"])
        x_min, x_max = min(min(x_pts), x_eval), max(max(x_pts), x_eval)
        xs = np.linspace(x_min, x_max, n_muestra)
        vals = np.abs(np.asarray(fn1_np(xs), dtype=float))
        vals = vals[np.isfinite(vals)]
        M = float(np.max(vals)) if len(vals) else float("nan")

        prod = 1.0
        for xi in x_pts:
            prod *= x_eval - xi

        # Cota GLOBAL: max del |producto nodal| sobre [x_0, x_n].
        xs_nodal = np.linspace(min(x_pts), max(x_pts), 1001)
        prod_nodal = np.ones_like(xs_nodal)
        for xi in x_pts:
            prod_nodal *= (xs_nodal - xi)
        prod_max = float(np.max(np.abs(prod_nodal)))
        x_peor = float(xs_nodal[int(np.argmax(np.abs(prod_nodal)))])

        from math import factorial
        cota_val = M / factorial(n + 1) * abs(prod)
        cota_global = M / factorial(n + 1) * prod_max

        return {
            "n": n, "M": M, "producto": abs(prod),
            "factorial": factorial(n + 1),
            "cota": cota_val,
            "producto_max": prod_max,
            "x_peor": x_peor,
            "cota_global": cota_global,
            "fn1_expr": sp.simplify(fn1),
        }
    except Exception as e:
        return {"error": str(e)}


# --- Plots ---

def _plot_lagrange(res: ResultadoLagrange, f_np=None, x_eval: float | None = None) -> go.Figure:
    x_min, x_max = min(res.x_pts), max(res.x_pts)
    margen = (x_max - x_min) * 0.4 if x_max > x_min else 1.0
    x_plot_min = min(x_min - margen, x_eval - 0.5) if x_eval is not None else x_min - margen
    x_plot_max = max(x_max + margen, x_eval + 0.5) if x_eval is not None else x_max + margen
    xs = np.linspace(x_plot_min, x_plot_max, 500)

    P_np = sp.lambdify(res.x_sym, res.P_expandido, modules=["numpy"])
    try:
        ys_P = np.asarray(P_np(xs), dtype=float)
    except Exception:
        ys_P = np.array([float(P_np(xi)) for xi in xs])

    fig = go.Figure()

    # Region de extrapolacion
    fig.add_vrect(x0=x_plot_min, x1=x_min, fillcolor="#f0f0f0", opacity=0.5, line_width=0,
                   annotation_text="extrapolacion", annotation_position="top left")
    fig.add_vrect(x0=x_max, x1=x_plot_max, fillcolor="#f0f0f0", opacity=0.5, line_width=0,
                   annotation_text="extrapolacion", annotation_position="top right")

    if f_np is not None:
        try:
            ys_f = np.asarray(f_np(xs), dtype=float)
            fig.add_trace(go.Scatter(x=xs, y=ys_f, mode="lines", name="f(x) original",
                                      line=dict(color="#2ca02c", width=2, dash="dot")))
        except Exception:
            pass

    fig.add_trace(go.Scatter(x=xs, y=ys_P, mode="lines", name="P(x) Lagrange",
                              line=dict(color="#1f77b4", width=2)))
    fig.add_trace(go.Scatter(x=res.x_pts, y=res.y_pts, mode="markers",
                              marker=dict(size=12, color="#d62728", symbol="circle"),
                              name="puntos dados"))

    if x_eval is not None:
        y_eval = evaluar(res, x_eval)
        es_extrap = x_eval < x_min or x_eval > x_max
        fig.add_trace(go.Scatter(
            x=[x_eval], y=[y_eval], mode="markers+text",
            marker=dict(size=14, color="orange", symbol="star"),
            text=[f"P({fmt_decimal(x_eval)})={fmt_decimal(y_eval)}"],
            textposition="top center",
            name=f"eval ({'extrap' if es_extrap else 'interp'})",
        ))

    fig.update_layout(
        title="Polinomio interpolante de Lagrange",
        xaxis_title="x", yaxis_title="y",
        height=460, hovermode="x unified",
    )
    return fig


def _plot_bases(res: ResultadoLagrange) -> go.Figure:
    x_min, x_max = min(res.x_pts), max(res.x_pts)
    margen = (x_max - x_min) * 0.2 if x_max > x_min else 1.0
    xs = np.linspace(x_min - margen, x_max + margen, 300)

    fig = go.Figure()
    colores = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    for i, base in enumerate(res.bases):
        L_np = sp.lambdify(res.x_sym, base, modules=["numpy"])
        try:
            ys = np.asarray(L_np(xs), dtype=float)
        except Exception:
            ys = np.array([float(L_np(xi)) for xi in xs])
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                                  name=f"L_{i}(x)",
                                  line=dict(color=colores[i % len(colores)], width=2)))

    fig.add_trace(go.Scatter(x=res.x_pts, y=[0] * len(res.x_pts),
                              mode="markers",
                              marker=dict(size=10, color="black", symbol="x"),
                              name="nodos x_i"))

    fig.update_layout(title="Bases de Lagrange L_i(x) — cada una vale 1 en su nodo y 0 en los demas",
                      xaxis_title="x", yaxis_title="L_i(x)",
                      height=380, hovermode="x unified")
    return fig


# --- Pasos ---

def _construir_pasos(res: ResultadoLagrange, x_eval: float | None) -> list[Paso]:
    pasos: list[Paso] = []
    n = len(res.x_pts) - 1

    pasos.append(Paso(
        titulo=f"Paso 1 — construir las {n+1} bases de Lagrange L_i(x)",
        formula=r"L_i(x) = \prod_{\substack{j=0 \\ j \neq i}}^{n} \frac{x - x_j}{x_i - x_j}",
        explicacion_tecnica=(
            f"Con {n+1} puntos, hay {n+1} bases de grado {n} cada una. "
            f"Por construccion L_i(x_i) = 1 y L_i(x_j) = 0 si j ≠ i."
        ),
        explicacion_coloquial=(
            f"Armamos {n+1} polinomios 'elementales'. Cada uno vale 1 en su nodo "
            f"y 0 en los demas — asi, al pesarlos por y_i y sumar, el polinomio "
            f"pasa justo por cada punto."
        ),
    ))

    for i, base in enumerate(res.bases):
        pasos.append(Paso(
            titulo=f"Base L_{i}(x) — asociada al nodo x_{i} = {fmt_decimal(res.x_pts[i])}",
            formula=rf"L_{{{i}}}(x) = \prod_{{j \neq {i}}} \frac{{x - x_j}}{{x_{{{i}}} - x_j}}",
            resultado=rf"L_{{{i}}}(x) = {sp.latex(base)}",
            explicacion_tecnica=(
                f"Excluyendo j = {i}, multiplicamos (x − x_j)/(x_{i} − x_j) para j = "
                + ", ".join(str(j) for j in range(len(res.x_pts)) if j != i)
                + "."
            ),
            explicacion_coloquial=(
                f"Esta base 'activa' solo el nodo {i}. Fijate que L_{i}({fmt_decimal(res.x_pts[i])}) "
                f"= {fmt_decimal(float(base.subs(res.x_sym, res.x_pts[i])))}."
            ),
        ))

    pasos.append(Paso(
        titulo="Paso 2 — armar P(x) = Σ y_i · L_i(x)",
        formula=r"P(x) = \sum_{i=0}^{n} y_i \, L_i(x)",
        resultado=rf"P(x) = {sp.latex(res.P_expandido)}",
        explicacion_tecnica=(
            "Combinacion lineal de las bases con pesos y_i. "
            "El resultado es un polinomio de grado ≤ n que pasa por todos los puntos."
        ),
        explicacion_coloquial=(
            "Sumamos cada base multiplicada por su y. Queda un polinomio que encaja "
            "con todos los puntos."
        ),
    ))

    if x_eval is not None:
        y_eval = evaluar(res, x_eval)
        x_min, x_max = min(res.x_pts), max(res.x_pts)
        es_extrap = x_eval < x_min or x_eval > x_max
        pasos.append(Paso(
            titulo=f"Paso 3 — evaluar en x = {fmt_decimal(x_eval)}",
            formula=r"P(x_{\text{eval}})",
            resultado=rf"P({fmt_decimal(x_eval)}) = {fmt_decimal(y_eval)}",
            explicacion_tecnica=(
                f"Sustituimos x = {fmt_decimal(x_eval)} en el polinomio expandido. "
                + ("⚠ Es **extrapolacion** (fuera del rango de los nodos) — el error crece rapidamente."
                   if es_extrap else
                   "Es **interpolacion** (dentro del rango) — error controlado por la cota.")
            ),
            explicacion_coloquial=(
                f"Metemos {fmt_decimal(x_eval)} en P y sale {fmt_decimal(y_eval)}. "
                + ("Ojo: estamos afuera del intervalo de los puntos, el resultado puede ser impreciso."
                   if es_extrap else
                   "Estamos dentro del intervalo, el polinomio es confiable aca.")
            ),
        ))

    return pasos


# --- Render principal ---

def render_lagrange() -> None:
    st.header("Interpolacion de Lagrange")
    st.caption("P(x) = Σ y_i · L_i(x) — prof. Caceres, pg 20–23")

    cfg = get_config()
    render_teoria("interpolacion_teoria",
                   titulo="Teoria del libro (Caceres pg 20–23)",
                   expanded=False)

    st.subheader("Entrada — puntos (x_i, y_i)")

    preset_key = st.selectbox("Preset de la guia",
                                options=list(EJERCICIOS.keys()),
                                index=0,
                                help="Ejercicios oficiales pg 25-26.")
    preset = EJERCICIOS[preset_key]

    x_default = preset.get("x_pts", [0.0, 1.0, 2.0])
    y_default = preset.get("y_pts", [1.0, 2.0, 5.0])

    df_default = pd.DataFrame({"x_i": x_default, "y_i": y_default})
    df_edit = st.data_editor(
        df_default, num_rows="dynamic", use_container_width=True,
        key=f"lagrange_puntos_{preset_key}",
        column_config={
            "x_i": st.column_config.NumberColumn("x_i", format="%.6f"),
            "y_i": st.column_config.NumberColumn("y_i", format="%.6f"),
        },
    )

    try:
        x_pts = df_edit["x_i"].dropna().astype(float).tolist()
        y_pts = df_edit["y_i"].dropna().astype(float).tolist()
    except Exception as e:
        st.error(f"Error al parsear los puntos: {e}")
        return

    if len(x_pts) != len(y_pts):
        st.error("Las columnas x_i e y_i deben tener la misma cantidad de filas.")
        return
    if len(x_pts) < 2:
        st.warning("Se necesitan al menos 2 puntos.")
        return
    if len(set(x_pts)) != len(x_pts):
        st.error("Las abscisas x_i deben ser distintas entre si.")
        return

    idx_sort = np.argsort(x_pts)
    x_pts = [x_pts[i] for i in idx_sort]
    y_pts = [y_pts[i] for i in idx_sort]

    n = len(x_pts) - 1

    # f(x) opcional
    usar_f = st.checkbox(
        "Tengo una f(x) original (para comparar y calcular cota de error)",
        value=bool(preset.get("f_latex")), key=f"lag_usef_{preset_key}",
    )
    f_expr = None
    f_np = None
    if usar_f:
        default_f = preset.get("f_latex", r"\sin(x)")
        latex_f = math_input("f(x) =", default_latex=default_f,
                              key=f"lagrange_fx_{preset_key}")
        x_sym_f = sp.Symbol("x")
        f_expr, f_np = parse_latex(latex_f, [x_sym_f])
        if f_expr is not None:
            st.latex(rf"f(x) = {sp.latex(f_expr)}")

    # Autocalcular y_i desde f(x_i): evita errores de tipeo manual.
    # Recordatorio: argumentos de sin/cos siempre en radianes (numpy/sympy).
    if f_np is not None:
        autocalc_y = st.checkbox(
            "Calcular y_i automaticamente desde f(x_i) [radianes]",
            value=True,
            key=f"lag_autocalc_{preset_key}",
            help=("Si esta activado, se ignora la columna y_i de la tabla y se usa "
                   "y_i = f(x_i). Recorda: funciones trigonometricas trabajan en radianes."),
        )
        if autocalc_y:
            try:
                y_pts = [float(f_np(xi)) for xi in x_pts]
                # Snap a cero el ruido de punto flotante (ej: sin(pi) = 1.22e-16).
                y_pts = [0.0 if abs(y) < 1e-12 else y for y in y_pts]
                st.caption(
                    "y_i = f(x_i): "
                    + ",  ".join(f"f({fmt_decimal(xi)})={fmt_decimal(yi)}"
                                   for xi, yi in zip(x_pts, y_pts))
                )
            except Exception as e:
                st.warning(
                    f"No se pudo evaluar f en los nodos: {e}. "
                    "Se usan los y_i de la tabla."
                )

    # Construir polinomio
    try:
        res = construir_lagrange(x_pts, y_pts)
    except Exception as e:
        st.error(f"No se pudo construir el polinomio: {e}")
        return

    x_eval_default = float(preset.get("x_eval", (x_pts[0] + x_pts[-1]) / 2))
    col_eval, col_info = st.columns([1, 2])
    with col_eval:
        x_eval = st.number_input("Evaluar P(x) en x =",
                                   value=x_eval_default, format="%.6f",
                                   key=f"lagrange_xeval_{preset_key}")
    y_eval = evaluar(res, x_eval)
    with col_info:
        es_extrap = x_eval < min(x_pts) or x_eval > max(x_pts)
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Grado del polinomio", n)
        col_b.metric(f"P({fmt_decimal(x_eval)})", fmt_decimal(y_eval),
                      delta=("extrapolacion ⚠" if es_extrap else "interpolacion"),
                      delta_color=("inverse" if es_extrap else "normal"))
        if f_np is not None:
            try:
                f_eval_val = float(f_np(x_eval))
                err_local = abs(f_eval_val - y_eval)
                col_c.metric(f"f({fmt_decimal(x_eval)}) real",
                              fmt_decimal(f_eval_val),
                              delta=f"|error| = {fmt_decimal(err_local)}",
                              delta_color="off")
            except Exception:
                pass

    # Cota de error si hay f
    if f_expr is not None:
        cota_info = cota_error(f_expr, sp.Symbol("x"), x_pts, x_eval)
        if "error" not in cota_info:
            with st.expander(f"Cota teorica de error (Caceres pg 22) — grado {n}",
                              expanded=True):
                st.markdown("**Teorema del error de Lagrange.** "
                             "Si $f \\in C^{n+1}[a,b]$, para cada $x \\in [a,b]$ existe "
                             "$\\xi(x) \\in (a,b)$ tal que:")
                st.latex(
                    r"E(x) = f(x) - P(x) = \frac{f^{(n+1)}(\xi)}{(n+1)!} "
                    r"\prod_{i=0}^{n} (x - x_i)"
                )
                st.markdown("Como no conocemos $\\xi$, se **acota** tomando el peor caso:")
                st.latex(
                    r"|E(x)| \le \frac{M_{n+1}}{(n+1)!} \left| \prod_{i=0}^{n} (x - x_i) \right|"
                )

                st.markdown("**Definiciones de cada termino:**")
                st.markdown(
                    f"- $n = {n}$ **grado** del polinomio (nro de nodos − 1).\n"
                    f"- $f^{{({n+1})}}(x)$ **derivada de orden n+1** de la funcion original. "
                    "Mide que tanto se 'curva' f mas alla de lo que P puede capturar.\n"
                    f"- $M_{{{n+1}}} = \\max_{{x \\in [a,b]}} |f^{{({n+1})}}(x)|$ **maximo absoluto** "
                    "de esa derivada en el intervalo. Es la 'constante de suavidad': si f es muy "
                    "oscilante, $M$ grande => cota grande.\n"
                    f"- $({n+1})! = {cota_info['factorial']}$ **factorial**: atenua el error a medida "
                    "que agregamos nodos (divisor crece rapido).\n"
                    f"- $\\prod_{{i=0}}^{{{n}}}(x - x_i)$ **polinomio nodal**: vale 0 en cada nodo "
                    "(por eso el error es 0 ahi) y crece al alejarse de ellos. Controla la **forma** del error."
                )

                st.markdown(f"**Para esta f(x):**")
                st.latex(rf"f^{{({n+1})}}(x) = {sp.latex(cota_info['fn1_expr'])}")

                st.markdown(f"#### Cota **local** en $x = {fmt_decimal(x_eval)}$")
                col_m, col_f, col_p, col_cota = st.columns(4)
                col_m.metric(f"M_{n+1} = max|f^({n+1})|", fmt_decimal(cota_info["M"]))
                col_f.metric(f"({n+1})!", cota_info["factorial"])
                col_p.metric(f"|Π(x−x_i)| en x={fmt_decimal(x_eval)}",
                              fmt_decimal(cota_info["producto"]))
                col_cota.metric("Cota |E(x)|", fmt_decimal(cota_info["cota"]))

                st.markdown(f"#### Cota **global** en $[{fmt_decimal(min(x_pts))},\\, {fmt_decimal(max(x_pts))}]$")
                col_gp, col_gx, col_gc = st.columns(3)
                col_gp.metric("max |Π(x−x_i)| en el intervalo",
                               fmt_decimal(cota_info["producto_max"]))
                col_gx.metric("se alcanza en x ≈",
                               fmt_decimal(cota_info["x_peor"]))
                col_gc.metric("Cota global max|E(x)|",
                               fmt_decimal(cota_info["cota_global"]))

                # Analisis teorico automatico
                try:
                    err_real = abs(float(f_np(x_eval)) - y_eval)
                except Exception:
                    err_real = None

                st.markdown("#### Analisis teorico")
                lineas = []
                lineas.append(
                    f"**M_{{{n+1}}} = {fmt_decimal(cota_info['M'])}** viene de que "
                    f"$f^{{({n+1})}}(x)={sp.latex(cota_info['fn1_expr'])}$ alcanza su maximo "
                    "absoluto en el intervalo. Este es el termino que **domina** la magnitud "
                    "de la cota: cuanto mas oscilante es f, peor aproxima Lagrange."
                )
                lineas.append(
                    f"El **polinomio nodal** $|\\prod(x-x_i)|$ vale {fmt_decimal(cota_info['producto'])} "
                    f"en $x={fmt_decimal(x_eval)}$ y llega a un maximo de "
                    f"{fmt_decimal(cota_info['producto_max'])} cerca de "
                    f"$x \\approx {fmt_decimal(cota_info['x_peor'])}$. Por eso la cota global "
                    f"({fmt_decimal(cota_info['cota_global'])}) es mayor que la cota local "
                    f"({fmt_decimal(cota_info['cota'])})."
                )
                if err_real is not None:
                    ratio = err_real / cota_info['cota'] if cota_info['cota'] > 0 else 0.0
                    lineas.append(
                        f"**Error real** en $x={fmt_decimal(x_eval)}$: "
                        f"$|f(x)-P(x)| = {fmt_decimal(err_real)}$. "
                        f"La cota local predice $\\le {fmt_decimal(cota_info['cota'])}$ "
                        f"=> el error real es el **{fmt_decimal(ratio*100)}%** de la cota. "
                        + ("La cota es **ajustada** (tight) — f casi satura el peor caso."
                           if ratio > 0.5 else
                           "La cota es **conservadora** — el error real es bastante menor "
                           "porque $\\xi$ no cae en el maximo de $f^{{(n+1)}}$.")
                    )
                lineas.append(
                    f"**Para mejorar**: agregar nodos => $(n+1)!$ crece factorialmente, "
                    "pero ojo con el **fenomeno de Runge** (nodos equiespaciados con $n$ grande "
                    "hacen que $M_{n+1}$ explote mas rapido que $(n+1)!$ en funciones no analiticas). "
                    "Alternativa: nodos de **Chebyshev** minimizan $\\max|\\prod(x-x_i)|$."
                )
                for linea in lineas:
                    st.markdown(f"- {linea}")

                # Bloque unificado listo para copiar al examen
                with st.expander("📝 Texto listo para copiar al examen", expanded=False):
                    fn1_latex = sp.latex(cota_info['fn1_expr'])
                    try:
                        f_latex_txt = sp.latex(f_expr)
                    except Exception:
                        f_latex_txt = "f(x)"
                    a_val, b_val = min(x_pts), max(x_pts)
                    M_val = cota_info["M"]
                    fact_val = cota_info["factorial"]
                    prod_loc = cota_info["producto"]
                    cota_loc = cota_info["cota"]
                    prod_max = cota_info["producto_max"]
                    x_peor = cota_info["x_peor"]
                    cota_glob = cota_info["cota_global"]

                    txt_err_real = ""
                    if err_real is not None:
                        ratio = err_real / cota_loc if cota_loc > 0 else 0.0
                        ajuste = ("ajustada (tight) — f casi satura el peor caso"
                                   if ratio > 0.5 else
                                   "conservadora — el error real es menor porque "
                                   "ξ no cae en el máximo de f^(n+1)")
                        txt_err_real = (
                            f"\n\n**Error real en x = {fmt_decimal(x_eval)}:**  \n"
                            f"|f(x) − P(x)| = |{fmt_decimal(float(f_np(x_eval)))} − "
                            f"{fmt_decimal(y_eval)}| = {fmt_decimal(err_real)}  \n"
                            f"La cota local predice ≤ {fmt_decimal(cota_loc)}, "
                            f"el error real es el {fmt_decimal(ratio*100)}% de la cota "
                            f"→ cota {ajuste}."
                        )

                    texto = f"""
**Análisis del error de interpolación de Lagrange** (grado n = {n})

**Fórmula del error** (Cáceres pg 22):
$$E(x) = f(x) - P(x) = \\frac{{f^{{({n+1})}}(\\xi)}}{{({n+1})!}} \\prod_{{i=0}}^{{{n}}}(x - x_i), \\quad \\xi \\in (a,b)$$

**Cota (peor caso):**
$$|E(x)| \\le \\frac{{M_{{{n+1}}}}}{{({n+1})!}} \\left|\\prod_{{i=0}}^{{{n}}}(x - x_i)\\right|$$

**Datos de esta corrida:**
- Función: $f(x) = {f_latex_txt}$ en $[{fmt_decimal(a_val)},\\, {fmt_decimal(b_val)}]$
- Nodos: {", ".join(fmt_decimal(xi) for xi in x_pts)}
- Derivada (n+1): $f^{{({n+1})}}(x) = {fn1_latex}$
- $M_{{{n+1}}} = \\max_{{[a,b]}}|f^{{({n+1})}}(x)| = {fmt_decimal(M_val)}$
- $({n+1})! = {fact_val}$

**Error local en x = {fmt_decimal(x_eval)}:**
- $|\\prod(x - x_i)| = {fmt_decimal(prod_loc)}$
- Cota local: $|E({fmt_decimal(x_eval)})| \\le \\dfrac{{{fmt_decimal(M_val)}}}{{{fact_val}}} \\cdot {fmt_decimal(prod_loc)} = {fmt_decimal(cota_loc)}$

**Error global (peor caso en todo el intervalo):**
- $\\max_{{[a,b]}}|\\prod(x - x_i)| = {fmt_decimal(prod_max)}$ (se alcanza en $x \\approx {fmt_decimal(x_peor)}$)
- Cota global: $\\max|E(x)| \\le \\dfrac{{{fmt_decimal(M_val)}}}{{{fact_val}}} \\cdot {fmt_decimal(prod_max)} = {fmt_decimal(cota_glob)}${txt_err_real}

**Interpretación:**
- $M_{{{n+1}}}$ mide la **curvatura** de f que P no captura; cuanto mayor, peor aproxima Lagrange.
- El **polinomio nodal** $\\prod(x - x_i)$ se anula en cada nodo (error exactamente 0 allí) y crece al alejarse. Por eso la cota global ({fmt_decimal(cota_glob)}) es mayor que la local ({fmt_decimal(cota_loc)}).
- El error máximo potencial cae en $x \\approx {fmt_decimal(x_peor)}$, el punto más alejado de todos los nodos simultáneamente.

**Cómo mejorar:**
- Agregar nodos → $(n+1)!$ crece factorialmente en el denominador.
- Cuidado con el **fenómeno de Runge**: con nodos equiespaciados y $n$ grande, $M_{{n+1}}$ puede crecer más rápido que $(n+1)!$ (f no analítica).
- Alternativa óptima: **nodos de Chebyshev**, que minimizan $\\max|\\prod(x-x_i)|$ concentrando más nodos en los bordes.
"""
                    st.markdown(texto)
                    st.code(texto, language="markdown")

    # Mostrar polinomio
    st.subheader("Polinomio interpolante")
    st.latex(rf"P(x) = {sp.latex(res.P_expandido)}")

    tab_resumen, tab_pasos, tab_viz = st.tabs(["Resumen", "Paso a paso", "Visualizaciones"])

    with tab_resumen:
        st.markdown("#### Bases de Lagrange y su contribucion en x_eval")
        filas = []
        for i, base in enumerate(res.bases):
            li_en_eval = float(base.subs(res.x_sym, x_eval))
            filas.append({
                "i": i,
                "x_i": res.x_pts[i],
                "y_i": res.y_pts[i],
                "L_i(x)": f"${sp.latex(base)}$",
                f"L_i({fmt_decimal(x_eval)})": li_en_eval,
                f"y_i · L_i({fmt_decimal(x_eval)})": res.y_pts[i] * li_en_eval,
            })
        df_bases = pd.DataFrame(filas)
        # Formatear numericas con fmt_decimal
        for col in df_bases.columns:
            if col != "L_i(x)" and pd.api.types.is_numeric_dtype(df_bases[col]):
                df_bases[col] = df_bases[col].map(lambda v: fmt_decimal(v))
        st.dataframe(df_bases, use_container_width=True, hide_index=True)

        if f_np is not None:
            st.markdown("#### Comparacion P(x) vs f(x) en grilla")
            x_grid = np.linspace(min(x_pts), max(x_pts), 11)
            P_np = sp.lambdify(res.x_sym, res.P_expandido, modules=["numpy"])
            try:
                y_P = np.asarray(P_np(x_grid), dtype=float)
                y_f = np.asarray(f_np(x_grid), dtype=float)
                df_grid = pd.DataFrame({
                    "x": x_grid,
                    "P(x)": y_P,
                    "f(x)": y_f,
                    "|error|": np.abs(y_P - y_f),
                })
                render_tabla_iteraciones(df_grid, titulo=None,
                                          key_export="lagrange_grid")
            except Exception as e:
                st.warning(f"No se pudo evaluar en grilla: {e}")

    with tab_pasos:
        render_pasos(_construir_pasos(res, x_eval), titulo="")

    with tab_viz:
        st.plotly_chart(_plot_lagrange(res, f_np, x_eval), use_container_width=True)
        st.plotly_chart(_plot_bases(res), use_container_width=True)

    if cfg.mostrar_glosario:
        render_glosario_expander(
            vars=["x_n", "f", "n"],
            titulo="Glosario — Lagrange",
        )


def render() -> None:
    render_lagrange()
