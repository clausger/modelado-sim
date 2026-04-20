"""Modulo Biseccion — metodo de busqueda binaria de raices.

Implementa el metodo tal como lo define el prof. Caceres (pg 8-10):
  1. Escoger [a,b] con f(a)·f(b) < 0
  2. c = (a+b)/2
  3. Evaluar f(c)
  4. Si f(a)·f(c) < 0 → b=c
  5. Si f(b)·f(c) < 0 → a=c

Features:
- Banco de ejercicios oficiales de la guia (pg 17) como presets
- 4 criterios de deteccion combinables (abs / rel / residuo / max iter)
- Pre-checks: Bolzano, multi-raices en el intervalo, continuidad
- Modo libro: round(·, 5) como en el codigo oficial
- Cota teorica (b-a)/2^(n+1) vs error real
- 3 tabs: Resumen, Paso a paso (tecnico/coloquial), Visualizaciones
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

# --- Banco de ejercicios (Caceres pg 17) ---

EJERCICIOS: dict[str, dict] = {
    "Personalizado": {},
    "Ej 3a — sqrt(x) − cos(x) = 0 en [0,1], tol=1e-3": {
        "latex": r"\sqrt{x} - \cos(x)", "a": 0.0, "b": 1.0, "tol": 1e-3,
    },
    "Ej 3b — x − 2^(−x) = 0 en [0,1], tol=1e-3": {
        "latex": r"x - 2^{-x}", "a": 0.0, "b": 1.0, "tol": 1e-3,
    },
    "Ej 3c — e^x − x^2 + 3x − 2 = 0 en [0,1], tol=1e-3": {
        "latex": r"e^{x} - x^{2} + 3x - 2", "a": 0.0, "b": 1.0, "tol": 1e-3,
    },
    "Ej 3d — 2x·cos(x) − (x+1)^2 = 0 en [−3,−2], tol=1e-3": {
        "latex": r"2x\cos(x) - (x+1)^{2}", "a": -3.0, "b": -2.0, "tol": 1e-3,
    },
    "Ej 3d — 2x·cos(x) − (x+1)^2 = 0 en [−1,0], tol=1e-3": {
        "latex": r"2x\cos(x) - (x+1)^{2}", "a": -1.0, "b": 0.0, "tol": 1e-3,
    },
    "Ej 3e — x·cos(x) − 2x^2 + 3x − 1 en [0.2,0.3], tol=1e-3": {
        "latex": r"x\cos(x) - 2x^{2} + 3x - 1", "a": 0.2, "b": 0.3, "tol": 1e-3,
    },
    "Ej 3e — x·cos(x) − 2x^2 + 3x − 1 en [1.2,1.3], tol=1e-3": {
        "latex": r"x\cos(x) - 2x^{2} + 3x - 1", "a": 1.2, "b": 1.3, "tol": 1e-3,
    },
    "Ej 4a — x^4 − 2x^3 − 4x^2 + 4x + 4 en [−2,−1], tol=1e-2": {
        "latex": r"x^{4} - 2x^{3} - 4x^{2} + 4x + 4", "a": -2.0, "b": -1.0, "tol": 1e-2,
    },
    "Ej 4b — x^4 − 2x^3 − 4x^2 + 4x + 4 en [0,2], tol=1e-2": {
        "latex": r"x^{4} - 2x^{3} - 4x^{2} + 4x + 4", "a": 0.0, "b": 2.0, "tol": 1e-2,
    },
    "Ej 4c — x^4 − 2x^3 − 4x^2 + 4x + 4 en [2,3], tol=1e-2": {
        "latex": r"x^{4} - 2x^{3} - 4x^{2} + 4x + 4", "a": 2.0, "b": 3.0, "tol": 1e-2,
    },
}


# --- Estructuras de datos ---

@dataclass(frozen=True)
class CriteriosDetencion:
    """Los 4 criterios oficiales (Caceres pg 6-7). Se activan con bools; el metodo
    corta con el primero que se cumple."""

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
    a: float
    b: float
    c: float
    fa: float
    fb: float
    fc: float
    signo_fa_fc: float  # f(a)·f(c)
    sub_intervalo: str  # "[a, c]" o "[c, b]"
    error_abs: float  # |c_n − c_{n-1}| (0 en la primera)
    error_rel: float  # |c_n − c_{n-1}| / |c_n|
    cota_teorica: float  # (b0 − a0) / 2^(n+1)


@dataclass
class ResultadoBiseccion:
    iteraciones: list[Iteracion] = field(default_factory=list)
    raiz: float | None = None
    motivo_corte: str = ""
    convergio: bool = False
    a_inicial: float = 0.0
    b_inicial: float = 0.0


# --- Algoritmo ---

def biseccion(
    f_np,
    a: float,
    b: float,
    criterios: CriteriosDetencion,
    precision: int | None = None,
) -> ResultadoBiseccion:
    """Ejecuta biseccion siguiendo literalmente el algoritmo del profe.

    Args:
        f_np: funcion numerica (callable) que acepta un float.
        a, b: extremos del intervalo.
        criterios: condiciones de parada.
        precision: si no es None, aplica round(·, precision) como en el codigo del libro.

    Returns:
        ResultadoBiseccion con todas las iteraciones y metadatos.
    """
    def r(x: float) -> float:
        return round(x, precision) if precision is not None else x

    res = ResultadoBiseccion(a_inicial=a, b_inicial=b)
    a_cur, b_cur = a, b
    c_prev: float | None = None
    ancho_inicial = b - a

    fa = r(float(f_np(a_cur)))
    fb = r(float(f_np(b_cur)))

    if fa * fb >= 0:
        res.motivo_corte = (
            "Pre-check Bolzano fallo: f(a)·f(b) ≥ 0. "
            "El metodo necesita signos opuestos en los extremos para garantizar una raiz."
        )
        return res

    n_max = criterios.max_iter if criterios.usar_max_iter else 10_000

    for n in range(1, n_max + 1):
        c = r((a_cur + b_cur) / 2.0)
        fa = r(float(f_np(a_cur)))
        fb = r(float(f_np(b_cur)))
        fc = r(float(f_np(c)))

        signo = fa * fc
        if signo < 0:
            nuevo_sub = f"[{fmt_decimal(a_cur)}, {fmt_decimal(c)}]"
        else:
            nuevo_sub = f"[{fmt_decimal(c)}, {fmt_decimal(b_cur)}]"

        if c_prev is None:
            err_abs = float("nan")
            err_rel = float("nan")
        else:
            err_abs = abs(c - c_prev)
            err_rel = err_abs / abs(c) if c != 0 else float("inf")

        cota = ancho_inicial / (2 ** (n + 1))

        res.iteraciones.append(Iteracion(
            n=n, a=a_cur, b=b_cur, c=c,
            fa=fa, fb=fb, fc=fc,
            signo_fa_fc=signo,
            sub_intervalo=nuevo_sub,
            error_abs=err_abs, error_rel=err_rel,
            cota_teorica=cota,
        ))

        # Chequear criterios (el primero que se cumple corta)
        if criterios.usar_residuo and abs(fc) <= criterios.tol_residuo:
            res.raiz = c
            res.motivo_corte = f"|f(c)| ≤ {fmt_decimal(criterios.tol_residuo)} (residuo)"
            res.convergio = True
            break
        if criterios.usar_abs and c_prev is not None and err_abs <= criterios.tol_abs:
            res.raiz = c
            res.motivo_corte = f"|c_n − c_(n−1)| ≤ {fmt_decimal(criterios.tol_abs)} (error absoluto)"
            res.convergio = True
            break
        if criterios.usar_rel and c_prev is not None and err_rel <= criterios.tol_rel:
            res.raiz = c
            res.motivo_corte = f"error relativo ≤ {fmt_decimal(criterios.tol_rel)}"
            res.convergio = True
            break

        # Actualizar intervalo (algoritmo del profe)
        if signo < 0:
            b_cur = c
        else:
            a_cur = c
        c_prev = c
    else:
        res.raiz = c_prev
        res.motivo_corte = f"Se alcanzo el maximo de iteraciones (N_max = {criterios.max_iter})"
        res.convergio = False

    return res


# --- Pre-checks ---

def contar_cambios_de_signo(f_np, a: float, b: float, n_puntos: int = 501) -> int:
    """Cuenta cambios de signo en una grilla densa — heuristica para detectar multi-raices."""
    try:
        xs = np.linspace(a, b, n_puntos)
        ys = f_np(xs)
        signos = np.sign(ys)
        cambios = int(np.sum(np.abs(np.diff(signos)) > 0))
        return cambios
    except Exception:
        return -1


# --- Helpers de renderizado ---

def _tabla_dataframe(res: ResultadoBiseccion) -> pd.DataFrame:
    filas = []
    for it in res.iteraciones:
        filas.append({
            "n": it.n,
            "a": it.a,
            "b": it.b,
            "c": it.c,
            "f(a)": it.fa,
            "f(b)": it.fb,
            "f(c)": it.fc,
            "f(a)·f(c)": it.signo_fa_fc,
            "nuevo intervalo": it.sub_intervalo,
            "E_abs": it.error_abs,
            "E_rel": it.error_rel,
            "cota (b−a)/2^(n+1)": it.cota_teorica,
        })
    return pd.DataFrame(filas)


def _plot_funcion_e_intervalos(f_np, a0: float, b0: float, res: ResultadoBiseccion, iter_focus: int) -> go.Figure:
    margen = (b0 - a0) * 0.2
    xs = np.linspace(a0 - margen, b0 + margen, 600)
    try:
        ys = f_np(xs)
    except Exception:
        ys = np.array([f_np(xi) for xi in xs])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="f(x)", line=dict(color="#1f77b4", width=2)))
    fig.add_hline(y=0, line=dict(color="black", width=0.7))

    if 1 <= iter_focus <= len(res.iteraciones):
        it = res.iteraciones[iter_focus - 1]
        fig.add_vrect(x0=it.a, x1=it.b, fillcolor="#ffd580", opacity=0.35,
                       line_width=0, annotation_text=f"[a_{it.n}, b_{it.n}]", annotation_position="top left")
        fig.add_trace(go.Scatter(
            x=[it.a, it.b, it.c], y=[it.fa, it.fb, it.fc],
            mode="markers+text",
            marker=dict(size=[10, 10, 14], color=["#2ca02c", "#d62728", "#ff7f0e"]),
            text=[f"a={fmt_decimal(it.a)}", f"b={fmt_decimal(it.b)}", f"c={fmt_decimal(it.c)}"],
            textposition="top center", name=f"Iter {it.n}",
        ))

    if res.raiz is not None:
        fig.add_vline(x=res.raiz, line=dict(color="green", dash="dash"),
                       annotation_text=f"raiz ≈ {fmt_decimal(res.raiz)}", annotation_position="bottom right")

    fig.update_layout(
        title="f(x) e intervalo de busqueda",
        xaxis_title="x", yaxis_title="f(x)",
        height=420, hovermode="x unified",
    )
    return fig


def _plot_convergencia(res: ResultadoBiseccion) -> go.Figure:
    ns = [it.n for it in res.iteraciones]
    cs = [it.c for it in res.iteraciones]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=cs, mode="lines+markers", name="c_n", line=dict(color="#ff7f0e")))
    if res.raiz is not None:
        fig.add_hline(y=res.raiz, line=dict(color="green", dash="dash"),
                       annotation_text=f"raiz final ≈ {fmt_decimal(res.raiz)}")
    fig.update_layout(
        title="Convergencia — c_n vs n",
        xaxis_title="n (iteracion)", yaxis_title="c_n",
        height=380,
    )
    return fig


def _plot_error(res: ResultadoBiseccion) -> go.Figure:
    ns = [it.n for it in res.iteraciones if not np.isnan(it.error_abs)]
    errs = [it.error_abs for it in res.iteraciones if not np.isnan(it.error_abs)]
    cotas = [it.cota_teorica for it in res.iteraciones]
    ns_todos = [it.n for it in res.iteraciones]

    fig = go.Figure()
    if errs:
        fig.add_trace(go.Scatter(x=ns, y=errs, mode="lines+markers",
                                  name="E_abs real", line=dict(color="#d62728")))
    fig.add_trace(go.Scatter(x=ns_todos, y=cotas, mode="lines+markers",
                              name="Cota teorica (b−a)/2^(n+1)", line=dict(color="#2ca02c", dash="dash")))
    fig.update_layout(
        title="Error absoluto vs cota teorica (escala log)",
        xaxis_title="n", yaxis_title="error",
        yaxis_type="log",
        height=380, hovermode="x unified",
    )
    return fig


def _construir_pasos(res: ResultadoBiseccion, n_pasos: int = 3) -> list[Paso]:
    pasos: list[Paso] = []
    pasos.append(Paso(
        titulo="Pre-check de Bolzano",
        formula=r"f(a) \cdot f(b) < 0",
        sustitucion=(
            f"f({fmt_decimal(res.a_inicial)}) \\cdot f({fmt_decimal(res.b_inicial)}) = "
            f"{fmt_decimal(res.iteraciones[0].fa if res.iteraciones else 0)} \\cdot "
            f"{fmt_decimal(res.iteraciones[0].fb if res.iteraciones else 0)}"
        ),
        explicacion_tecnica=(
            "El Teorema de Bolzano garantiza que si f es continua en [a,b] y f(a)·f(b)<0, "
            "existe al menos un c ∈ (a,b) tal que f(c)=0. Este es el fundamento de Biseccion."
        ),
        explicacion_coloquial=(
            "Si en un extremo del intervalo f vale positivo y en el otro negativo, "
            "obligatoriamente la curva cruza el eje x en algun lugar entre medio. "
            "Biseccion usa ese hecho."
        ),
    ))

    for it in res.iteraciones[:n_pasos]:
        pasos.append(Paso(
            titulo=f"Iteracion {it.n} — calcular punto medio y decidir subintervalo",
            formula=r"c_n = \frac{a_n + b_n}{2}, \quad \text{signo}(f(a_n) \cdot f(c_n))",
            sustitucion=(
                rf"c_{{{it.n}}} = \frac{{{fmt_decimal(it.a)} + {fmt_decimal(it.b)}}}{{2}} = {fmt_decimal(it.c)}"
            ),
            resultado=(
                rf"f(c_{{{it.n}}}) = {fmt_decimal(it.fc)}, \quad f(a) \cdot f(c) = {fmt_decimal(it.signo_fa_fc)}"
                rf" \Rightarrow \text{{nuevo intervalo}} = {it.sub_intervalo}"
            ),
            explicacion_tecnica=(
                f"Se toma el punto medio c={fmt_decimal(it.c)}. Se evalua f(c)={fmt_decimal(it.fc)}. "
                f"Como f(a)·f(c) = {fmt_decimal(it.signo_fa_fc)} "
                + ("< 0, la raiz esta en [a,c] y actualizamos b=c." if it.signo_fa_fc < 0
                   else "≥ 0, la raiz esta en [c,b] y actualizamos a=c.")
            ),
            explicacion_coloquial=(
                f"Cortamos el intervalo a la mitad en c={fmt_decimal(it.c)}. "
                + ("Como el signo de f cambia entre a y c, la raiz esta en esa mitad izquierda." if it.signo_fa_fc < 0
                   else "Como el signo de f NO cambia entre a y c, la raiz esta en la mitad derecha.")
            ),
        ))

    if res.convergio and res.raiz is not None:
        ult = res.iteraciones[-1]
        pasos.append(Paso(
            titulo="Corte del metodo",
            resultado=rf"\hat{{x}} \approx {fmt_decimal(res.raiz)}, \quad |f(\hat{{x}})| = {fmt_decimal(abs(ult.fc))}",
            explicacion_tecnica=f"Criterio de corte aplicado: {res.motivo_corte}.",
            explicacion_coloquial=f"Cortamos porque: {res.motivo_corte}. Ya estamos suficientemente cerca de la raiz.",
        ))

    return pasos


# --- Renderizado principal ---

def render_biseccion() -> None:
    st.header("Metodo de Biseccion")
    st.caption("Busqueda binaria de raices — prof. Omar J. Caceres, pg 8–10")

    cfg = get_config()

    # Teoria del libro al tope
    render_teoria("raices_teoria", titulo="Teoria del libro (Caceres pg 5–10)", expanded=False)

    # --- Inputs ---
    st.subheader("Entrada")

    col_preset, col_preset_info = st.columns([2, 3])
    with col_preset:
        preset_key = st.selectbox(
            "Preset de la guia",
            options=list(EJERCICIOS.keys()),
            index=0,
            help="Ejercicios oficiales de la guia (pg 17). Autocompletan f(x), intervalo y tolerancia.",
        )
    preset = EJERCICIOS[preset_key]
    with col_preset_info:
        if preset:
            st.info(f"**{preset_key}** — verificar que el script reproduce el resultado esperado.")

    default_latex = preset.get("latex", r"x^{3} - x - 1")
    latex = math_input("f(x) =", default_latex=default_latex, key=f"biseccion_fx_{preset_key}")
    expr, f_np = parse_latex(latex, [sp.Symbol("x")])

    if expr is None or f_np is None:
        st.warning("Ingresa una funcion valida para continuar.")
        return

    st.latex(f"f(x) = {sp.latex(expr)}")

    col_a, col_b = st.columns(2)
    with col_a:
        a = st.number_input("a (extremo izquierdo)",
                             value=float(preset.get("a", 1.0)),
                             format="%.6f",
                             help="Limite inferior del intervalo. Debe cumplir f(a)·f(b) < 0.")
    with col_b:
        b = st.number_input("b (extremo derecho)",
                             value=float(preset.get("b", 2.0)),
                             format="%.6f",
                             help="Limite superior del intervalo.")

    if a >= b:
        st.error("Se requiere a < b.")
        return

    # Criterios de detencion
    with st.expander("Criterios de detencion (4 oficiales del libro)", expanded=not cfg.modo_examen):
        tol_default = float(preset.get("tol", 1e-6))
        col1, col2 = st.columns(2)
        with col1:
            usar_abs = st.checkbox("Error absoluto |c_n − c_(n−1)| ≤ ε",
                                    value=True, key="bis_usar_abs")
            tol_abs = st.number_input("ε absoluto", value=tol_default,
                                       format="%.1e", min_value=0.0, step=1e-7,
                                       disabled=not usar_abs, key="bis_tol_abs")
            usar_rel = st.checkbox("Error relativo |c_n − c_(n−1)|/|c_n| ≤ ε",
                                    value=False, key="bis_usar_rel")
            tol_rel = st.number_input("ε relativo", value=tol_default,
                                       format="%.1e", min_value=0.0, step=1e-7,
                                       disabled=not usar_rel, key="bis_tol_rel")
        with col2:
            usar_residuo = st.checkbox("Residuo |f(c)| ≤ ε",
                                        value=True, key="bis_usar_res")
            tol_residuo = st.number_input("ε residuo", value=tol_default,
                                           format="%.1e", min_value=0.0, step=1e-7,
                                           disabled=not usar_residuo, key="bis_tol_res")
            usar_max = st.checkbox("Maximo de iteraciones",
                                    value=True, key="bis_usar_max")
            max_iter = st.number_input("N_max", value=100, min_value=1, max_value=10_000, step=10,
                                        disabled=not usar_max, key="bis_nmax")

    criterios = CriteriosDetencion(
        usar_abs=usar_abs, tol_abs=tol_abs,
        usar_rel=usar_rel, tol_rel=tol_rel,
        usar_residuo=usar_residuo, tol_residuo=tol_residuo,
        usar_max_iter=usar_max, max_iter=int(max_iter),
    )

    # --- Pre-checks ---
    try:
        fa = float(f_np(a))
        fb = float(f_np(b))
    except Exception as e:
        st.error(f"Error al evaluar f en los extremos: {e}")
        return

    col_fa, col_fb, col_prod = st.columns(3)
    col_fa.metric("f(a)", fmt_decimal(fa))
    col_fb.metric("f(b)", fmt_decimal(fb))
    col_prod.metric("f(a)·f(b)", fmt_decimal(fa * fb),
                     delta="OK (signos opuestos)" if fa * fb < 0 else "⚠ mismo signo",
                     delta_color="normal" if fa * fb < 0 else "inverse")

    if fa * fb >= 0:
        st.error(
            "**Bolzano no se cumple**: f(a) y f(b) tienen el mismo signo (o uno es cero). "
            "El metodo de Biseccion requiere signos opuestos para garantizar una raiz. "
            "Ajustar el intervalo."
        )
        return

    cambios = contar_cambios_de_signo(f_np, a, b)
    if cambios > 1:
        st.warning(
            f"Se detectaron **{cambios} cambios de signo** en [a, b]. "
            f"Biseccion va a converger a UNA de las raices (no necesariamente a la mas cercana a un extremo). "
            f"Si esperas una raiz especifica, acota mas el intervalo."
        )

    # --- Ejecutar ---
    precision_usada = 5 if cfg.modo_libro else None
    res = biseccion(f_np, a, b, criterios, precision=precision_usada)

    if not res.iteraciones:
        st.error(res.motivo_corte or "No se pudieron ejecutar iteraciones.")
        return

    # --- Resultado resumen ---
    st.subheader("Resultado")
    if res.convergio:
        st.success(f"Convergio — {res.motivo_corte}")
    else:
        st.warning(f"NO convergio — {res.motivo_corte}")

    col_r, col_n, col_fr, col_cota = st.columns(4)
    col_r.metric("Raiz aproximada", fmt_decimal(res.raiz))
    col_n.metric("Iteraciones", len(res.iteraciones))
    col_fr.metric("f(raiz)", fmt_decimal(float(f_np(res.raiz)) if res.raiz is not None else None))
    col_cota.metric("Cota teorica final", fmt_decimal(res.iteraciones[-1].cota_teorica))

    # --- Tabs principales ---
    tab_resumen, tab_pasos, tab_viz = st.tabs(["Resumen", "Paso a paso", "Visualizaciones"])

    with tab_resumen:
        df = _tabla_dataframe(res)
        resaltar = None
        if criterios.usar_abs:
            resaltar = resaltar_tolerancia("E_abs", criterios.tol_abs)
        render_tabla_iteraciones(df, resaltar=resaltar,
                                  titulo="Tabla completa de iteraciones",
                                  key_export="biseccion")

    with tab_pasos:
        n_pasos = st.slider("Cantidad de iteraciones a explicar paso a paso",
                              min_value=1, max_value=min(10, len(res.iteraciones)),
                              value=min(3, len(res.iteraciones)))
        pasos = _construir_pasos(res, n_pasos=n_pasos)
        render_pasos(pasos, titulo="")

    with tab_viz:
        iter_focus = st.slider("Iteracion a visualizar en el grafico de f(x)",
                                min_value=1, max_value=len(res.iteraciones),
                                value=min(3, len(res.iteraciones)))
        st.plotly_chart(_plot_funcion_e_intervalos(f_np, a, b, res, iter_focus),
                         use_container_width=True)
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.plotly_chart(_plot_convergencia(res), use_container_width=True)
        with col_g2:
            st.plotly_chart(_plot_error(res), use_container_width=True)

    # Glosario al final si esta activo
    if cfg.mostrar_glosario:
        render_glosario_expander(
            vars=["a", "b", "c", "x_star", "x_n", "f", "e_n", "E_abs", "E_r", "epsilon", "N_max", "n", "p"],
            titulo="Glosario de variables — Biseccion",
        )


# Backwards compat: el app.py antiguo llamaba a render()
def render() -> None:
    render_biseccion()
