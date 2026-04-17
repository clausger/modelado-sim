"""Modulo Punto Fijo — iteracion x_{n+1} = g(x_n).

Implementa el metodo tal como lo define el prof. Caceres (pg 11-12):
  Resolver f(x) = 0 se reformula como x = g(x).
  La iteracion converge cuando |g'(x)| < 1 en el compacto (Lipschitz con L < 1).

Features:
- Asistente de reformulacion: propone 3-4 g(x) candidatos dado f(x)=0, y los rankea
  por si cumplen |g'| < 1 en el intervalo.
- Pre-checks: contraccion (max |g'|), g(X) subset X, Lipschitz L.
- Ejercicios oficiales (pg 18) como presets.
- 4 criterios de detencion combinables.
- Cobweb diagram (telarana) — visualizacion clave de la materia.
- Cota teorica de Banach: d(x_n, x*) ≤ k^n · d(x_0, x*).
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

# --- Banco de ejercicios (Caceres pg 18) ---

EJERCICIOS: dict[str, dict] = {
    "Personalizado": {},
    "Ej 1 — f(x) = 2·e^(x^2) − 5x, x* ∈ [0,1], x₀ = 0": {
        "tipo": "f", "latex_f": r"2e^{x^{2}} - 5x",
        "a": 0.0, "b": 1.0, "x0": 0.0,
    },
    "Ej 2 — f(x) = cos(x), x* ∈ [1,2], x₀ = 1": {
        "tipo": "f", "latex_f": r"\cos(x)",
        "a": 1.0, "b": 2.0, "x0": 1.0,
    },
    "Ej 3 — f(x) = e^(−x) − x, x* ∈ [0,1], x₀ = 0": {
        "tipo": "f", "latex_f": r"e^{-x} - x",
        "a": 0.0, "b": 1.0, "x0": 0.0,
    },
    "Ej 4 — f(x) = x^3 − x − 1, x* ∈ [1,2], x₀ = 1": {
        "tipo": "f", "latex_f": r"x^{3} - x - 1",
        "a": 1.0, "b": 2.0, "x0": 1.0,
    },
    "Ej 6a — g(x) = (3 + x − 2x^2)^(1/4) (directo)": {
        "tipo": "g", "latex_g": r"(3 + x - 2x^{2})^{1/4}",
        "a": 0.0, "b": 2.0, "x0": 1.0,
    },
    "Ej 7 — g(x) = 2^(−x), x* ∈ [1/3, 1]": {
        "tipo": "g", "latex_g": r"2^{-x}",
        "a": 1/3, "b": 1.0, "x0": 0.5,
    },
    "Ej 10 — g(x) = sqrt(e^x / 3)": {
        "tipo": "g", "latex_g": r"\sqrt{e^{x}/3}",
        "a": 0.0, "b": 2.0, "x0": 1.0,
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
    g_xn: float
    error_abs: float
    error_rel: float
    cota_banach: float


@dataclass
class ResultadoPuntoFijo:
    iteraciones: list[Iteracion] = field(default_factory=list)
    raiz: float | None = None
    motivo_corte: str = ""
    convergio: bool = False
    x0: float = 0.0
    L: float | None = None
    k_contraccion: float | None = None


# --- Asistente de reformulacion ---

def generar_reformulaciones(f_expr: sp.Expr, x: sp.Symbol) -> list[tuple[str, sp.Expr]]:
    """Genera hasta 4 g(x) candidatos para x = g(x) a partir de f(x)=0."""
    candidatos: list[tuple[str, sp.Expr]] = []
    try:
        g1 = sp.simplify(x + f_expr)
        candidatos.append((r"g_1(x) = x + f(x)", g1))
    except Exception:
        pass
    try:
        g2 = sp.simplify(x - f_expr)
        candidatos.append((r"g_2(x) = x - f(x)", g2))
    except Exception:
        pass
    try:
        df = sp.diff(f_expr, x)
        g_newton = sp.simplify(x - f_expr / df)
        candidatos.append((r"g_3(x) = x - f(x)/f'(x) \; (\text{tipo Newton})", g_newton))
    except Exception:
        pass
    return candidatos


def analizar_convergencia(g_expr: sp.Expr, x: sp.Symbol, a: float, b: float,
                          n_puntos: int = 201) -> dict:
    """Analiza contraccion de g en [a,b]: L, rango, g(X)⊆X."""
    try:
        gp = sp.diff(g_expr, x)
        g_np = sp.lambdify(x, g_expr, modules=["numpy"])
        gp_np = sp.lambdify(x, gp, modules=["numpy"])
    except Exception:
        return {"error": "No se pudo derivar g(x)."}

    try:
        xs = np.linspace(a, b, n_puntos)
        g_vals = np.asarray(g_np(xs), dtype=float)
        gp_vals = np.abs(np.asarray(gp_np(xs), dtype=float))

        # Filtrar NaN/Inf de g' (ej: sqrt en borde)
        gp_finite = gp_vals[np.isfinite(gp_vals)]
        if len(gp_finite) == 0:
            return {"error": "g'(x) no es finita en el intervalo."}
        L = float(np.max(gp_finite))

        g_finite = g_vals[np.isfinite(g_vals)]
        if len(g_finite) == 0:
            return {"error": "g(x) no es finita en el intervalo."}
        g_min = float(np.min(g_finite))
        g_max = float(np.max(g_finite))
        g_en_X = (g_min >= a - 1e-9) and (g_max <= b + 1e-9)

        return {
            "L": L, "contractiva": L < 1.0,
            "g_min": g_min, "g_max": g_max,
            "g_de_X_en_X": g_en_X,
            "gp_expr": sp.simplify(gp),
        }
    except Exception as e:
        return {"error": f"Error al evaluar g o g' en [{a},{b}]: {e}"}


# --- Algoritmo ---

def punto_fijo(
    g_np,
    x0: float,
    criterios: CriteriosDetencion,
    a: float, b: float,
    L_estimado: float | None = None,
    precision: int | None = None,
) -> ResultadoPuntoFijo:
    def r(v: float) -> float:
        return round(v, precision) if precision is not None else v

    res = ResultadoPuntoFijo(x0=x0, L=L_estimado)
    if L_estimado is not None and L_estimado < 1:
        res.k_contraccion = L_estimado

    n_max = criterios.max_iter if criterios.usar_max_iter else 10_000
    x_n = r(x0)

    for n in range(1, n_max + 1):
        try:
            g_xn = r(float(g_np(x_n)))
        except Exception as e:
            res.motivo_corte = f"Error al evaluar g en iter {n}: {e}"
            return res

        if not np.isfinite(g_xn):
            res.motivo_corte = f"g(x) no finito en iter {n}. Divergencia."
            return res

        err_abs = abs(g_xn - x_n)
        err_rel = err_abs / abs(g_xn) if g_xn != 0 else float("inf")
        cota = (res.k_contraccion ** n * abs(x0 - x_n)) if res.k_contraccion else float("nan")

        res.iteraciones.append(Iteracion(
            n=n, x_n=x_n, g_xn=g_xn,
            error_abs=err_abs, error_rel=err_rel,
            cota_banach=cota,
        ))

        if criterios.usar_abs and err_abs <= criterios.tol_abs:
            res.raiz = g_xn
            res.motivo_corte = f"|x_n+1 − x_n| ≤ {criterios.tol_abs:g} (error absoluto)"
            res.convergio = True
            return res
        if criterios.usar_rel and err_rel <= criterios.tol_rel:
            res.raiz = g_xn
            res.motivo_corte = f"error relativo ≤ {criterios.tol_rel:g}"
            res.convergio = True
            return res
        if criterios.usar_residuo and err_abs <= criterios.tol_residuo:
            res.raiz = g_xn
            res.motivo_corte = f"|g(x) − x| ≤ {criterios.tol_residuo:g} (residuo)"
            res.convergio = True
            return res

        if abs(g_xn) > 1e12:
            res.motivo_corte = f"Divergencia: |x_{n}| = {abs(g_xn):.2e}"
            return res

        x_n = g_xn

    res.raiz = x_n
    res.motivo_corte = f"Se alcanzo el maximo de iteraciones (N_max = {criterios.max_iter})"
    return res


# --- Plots ---

def _plot_cobweb(g_np, res: ResultadoPuntoFijo, a: float, b: float) -> go.Figure:
    margen = (b - a) * 0.15
    xs = np.linspace(a - margen, b + margen, 400)
    try:
        ys_g = np.asarray(g_np(xs), dtype=float)
    except Exception:
        ys_g = np.array([float(g_np(xi)) for xi in xs])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=xs, mode="lines", name="y = x",
                              line=dict(color="#2ca02c", dash="dash")))
    fig.add_trace(go.Scatter(x=xs, y=ys_g, mode="lines", name="y = g(x)",
                              line=dict(color="#1f77b4", width=2)))

    if res.iteraciones:
        traza_x, traza_y = [res.x0], [res.x0]
        for it in res.iteraciones:
            traza_x.extend([it.x_n, it.g_xn])
            traza_y.extend([it.g_xn, it.g_xn])
        fig.add_trace(go.Scatter(x=traza_x, y=traza_y, mode="lines+markers",
                                  name="iteracion",
                                  line=dict(color="#ff7f0e", width=1),
                                  marker=dict(size=6)))

    if res.raiz is not None:
        fig.add_trace(go.Scatter(x=[res.raiz], y=[res.raiz], mode="markers",
                                  marker=dict(size=14, color="green", symbol="star"),
                                  name=f"punto fijo ≈ {fmt_decimal(res.raiz)}"))

    fig.update_layout(
        title="Diagrama de telarana (cobweb)",
        xaxis_title="x", yaxis_title="y",
        height=500, hovermode="closest",
        xaxis=dict(range=[a - margen, b + margen]),
        yaxis=dict(range=[a - margen, b + margen], scaleanchor="x", scaleratio=1),
    )
    return fig


def _plot_convergencia(res: ResultadoPuntoFijo) -> go.Figure:
    ns = [0] + [it.n for it in res.iteraciones]
    xs = [res.x0] + [it.g_xn for it in res.iteraciones]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=xs, mode="lines+markers", name="x_n",
                              line=dict(color="#ff7f0e")))
    if res.raiz is not None:
        fig.add_hline(y=res.raiz, line=dict(color="green", dash="dash"),
                       annotation_text=f"punto fijo ≈ {fmt_decimal(res.raiz)}")
    fig.update_layout(title="Convergencia — x_n vs n",
                      xaxis_title="n", yaxis_title="x_n", height=380)
    return fig


def _plot_error(res: ResultadoPuntoFijo) -> go.Figure:
    ns = [it.n for it in res.iteraciones]
    errs = [it.error_abs for it in res.iteraciones]
    cotas = [it.cota_banach for it in res.iteraciones] if res.k_contraccion else []

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=errs, mode="lines+markers",
                              name="E_abs real", line=dict(color="#d62728")))
    if cotas and all(np.isfinite(c) for c in cotas):
        fig.add_trace(go.Scatter(x=ns, y=cotas, mode="lines+markers",
                                  name=f"Cota Banach (k={fmt_decimal(res.k_contraccion)})",
                                  line=dict(color="#2ca02c", dash="dash")))
    fig.update_layout(title="Error absoluto vs cota de Banach (log)",
                      xaxis_title="n", yaxis_title="error",
                      yaxis_type="log", height=380)
    return fig


def _construir_pasos(res: ResultadoPuntoFijo, n_pasos: int = 3) -> list[Paso]:
    pasos: list[Paso] = []
    if res.L is not None:
        pasos.append(Paso(
            titulo="Pre-check de contraccion (Banach + Lipschitz)",
            formula=r"L = \max_{x \in [a,b]} |g'(x)| < 1 \; \Rightarrow \; \text{converge}",
            sustitucion=rf"L = {fmt_decimal(res.L)}",
            explicacion_tecnica=(
                "Por el teorema del punto fijo de Banach, si g es contractiva en un compacto "
                "(|g'(x)| ≤ L < 1 ∀ x ∈ [a,b]), existe un unico punto fijo y la iteracion converge. "
                + ("**L < 1, converge garantizado.**" if res.L < 1 else
                   "**L ≥ 1: la iteracion puede NO converger.**")
            ),
            explicacion_coloquial=(
                "Si g 'acerca' los puntos (pendiente siempre menor que 1 en modulo), "
                "al iterar te vas acercando al punto fijo. "
                + ("L < 1 → vamos bien." if res.L < 1 else "L ≥ 1 → probar otra g(x).")
            ),
        ))

    for it in res.iteraciones[:n_pasos]:
        pasos.append(Paso(
            titulo=f"Iteracion {it.n} — aplicar g al valor actual",
            formula=r"x_{n+1} = g(x_n)",
            sustitucion=rf"x_{{{it.n}}} = g({fmt_decimal(it.x_n)}) = {fmt_decimal(it.g_xn)}",
            resultado=rf"|x_{{{it.n}}} - x_{{{it.n-1}}}| = {fmt_decimal(it.error_abs)}",
            explicacion_tecnica=(
                f"Se evalua g en x_(n-1) = {fmt_decimal(it.x_n)} y se obtiene "
                f"x_{it.n} = {fmt_decimal(it.g_xn)}. La diferencia es {fmt_decimal(it.error_abs)}."
            ),
            explicacion_coloquial=(
                f"Metemos {fmt_decimal(it.x_n)} en g y sale {fmt_decimal(it.g_xn)}. "
                f"Nos movimos {fmt_decimal(it.error_abs)}."
            ),
        ))

    if res.convergio and res.raiz is not None:
        pasos.append(Paso(
            titulo="Corte del metodo",
            resultado=rf"x^* \approx {fmt_decimal(res.raiz)}",
            explicacion_tecnica=f"Criterio: {res.motivo_corte}.",
            explicacion_coloquial=f"Cortamos: {res.motivo_corte}.",
        ))

    return pasos


def _tabla_dataframe(res: ResultadoPuntoFijo) -> pd.DataFrame:
    filas = []
    for it in res.iteraciones:
        filas.append({
            "n": it.n,
            "x_n": it.x_n,
            "x_n+1 = g(x_n)": it.g_xn,
            "E_abs": it.error_abs,
            "E_rel": it.error_rel,
            "cota Banach": it.cota_banach,
        })
    return pd.DataFrame(filas)


# --- Render principal ---

def render_punto_fijo() -> None:
    st.header("Metodo del Punto Fijo")
    st.caption("Iteracion x_{n+1} = g(x_n) — prof. Caceres, pg 11–12")

    cfg = get_config()
    render_teoria("raices_teoria", titulo="Teoria del libro (Caceres pg 11–12)",
                   expanded=False)

    st.subheader("Entrada")

    preset_key = st.selectbox("Preset de la guia", options=list(EJERCICIOS.keys()),
                                index=0, help="Ejercicios oficiales (pg 18).")
    preset = EJERCICIOS[preset_key]

    modo_entrada = st.radio(
        "Modo de entrada",
        options=["Desde f(x)=0 con asistente", "g(x) directo"],
        index=1 if preset.get("tipo") == "g" else 0,
        horizontal=True,
    )

    x_sym = sp.Symbol("x")
    g_expr: sp.Expr | None = None

    if modo_entrada == "Desde f(x)=0 con asistente":
        default_f = preset.get("latex_f", r"x^{3} - x - 1") if preset else r"x^{3} - x - 1"
        latex_f = math_input("f(x) = (queremos f(x)=0)", default_latex=default_f,
                              key=f"pf_fx_{preset_key}")
        f_expr, _ = parse_latex(latex_f, [x_sym])
        if f_expr is None:
            st.warning("Ingresa una f(x) valida.")
            return
        st.latex(rf"f(x) = {sp.latex(f_expr)} = 0")

        col_a, col_b, col_x0 = st.columns(3)
        with col_a:
            a = st.number_input("a", value=float(preset.get("a", 0.0)), format="%.6f", key="pf_a")
        with col_b:
            b = st.number_input("b", value=float(preset.get("b", 1.0)), format="%.6f", key="pf_b")
        with col_x0:
            x0 = st.number_input("x₀", value=float(preset.get("x0", 0.5)), format="%.6f", key="pf_x0")

        if a >= b:
            st.error("Se requiere a < b.")
            return

        st.markdown("#### Asistente de reformulacion")
        st.caption(
            "Dado f(x)=0 te propongo varios g(x) para iterar x = g(x). "
            "Cada candidato se evalua contra el criterio de Banach (L = max|g'| < 1 en [a,b])."
        )

        candidatos = generar_reformulaciones(f_expr, x_sym)
        analisis_cands = [(nombre, g_e, analizar_convergencia(g_e, x_sym, a, b))
                           for nombre, g_e in candidatos]

        filas = []
        for nombre, g_e, an in analisis_cands:
            if "error" in an:
                filas.append({
                    "candidato": nombre, "g(x)": str(g_e),
                    "L = max|g'|": "—", "L<1": "—", "g(X)⊆X": "—",
                    "veredicto": f"✗ {an['error']}",
                })
            else:
                veredicto = []
                veredicto.append("✓ L<1" if an["contractiva"] else "✗ L≥1")
                veredicto.append("✓ g(X)⊆X" if an["g_de_X_en_X"] else "✗ g(X)⊄X")
                filas.append({
                    "candidato": nombre, "g(x)": str(g_e),
                    "L = max|g'|": fmt_decimal(an["L"]),
                    "L<1": "Si" if an["contractiva"] else "No",
                    "g(X)⊆X": "Si" if an["g_de_X_en_X"] else "No",
                    "veredicto": " | ".join(veredicto),
                })
        st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)

        opciones = [f"{nombre}" for nombre, _, _ in analisis_cands] + ["Escribir g(x) a mano"]
        idx_default = 0
        for i, (_, _, an) in enumerate(analisis_cands):
            if "error" not in an and an.get("contractiva"):
                idx_default = i
                break
        seleccion = st.radio("Elegi la g(x) a iterar", opciones, index=idx_default,
                              key=f"pf_elec_{preset_key}")

        if seleccion == "Escribir g(x) a mano":
            latex_g = math_input("g(x) =", default_latex=r"x - \left(" + sp.latex(f_expr) + r"\right)",
                                  key=f"pf_gx_manual_{preset_key}")
            g_expr, _ = parse_latex(latex_g, [x_sym])
            if g_expr is None:
                st.warning("Ingresa una g(x) valida.")
                return
        else:
            idx_sel = opciones.index(seleccion)
            _, g_expr, _ = analisis_cands[idx_sel]

    else:
        default_g = preset.get("latex_g", r"\cos(x)") if preset else r"\cos(x)"
        latex_g = math_input("g(x) = (iteramos x = g(x))", default_latex=default_g,
                              key=f"pf_gd_{preset_key}")
        g_expr, _ = parse_latex(latex_g, [x_sym])
        if g_expr is None:
            st.warning("Ingresa una g(x) valida.")
            return

        col_a, col_b, col_x0 = st.columns(3)
        with col_a:
            a = st.number_input("a", value=float(preset.get("a", 0.0)), format="%.6f", key="pf_ad")
        with col_b:
            b = st.number_input("b", value=float(preset.get("b", 1.0)), format="%.6f", key="pf_bd")
        with col_x0:
            x0 = st.number_input("x₀", value=float(preset.get("x0", 0.5)), format="%.6f", key="pf_x0d")

    st.latex(rf"g(x) = {sp.latex(g_expr)}")

    # Analisis de convergencia de la g elegida
    analisis = analizar_convergencia(g_expr, x_sym, a, b)
    if "error" in analisis:
        st.error(analisis["error"])
        return

    L = analisis["L"]
    col_L, col_gx, col_ver = st.columns(3)
    col_L.metric("L = max|g'(x)|", fmt_decimal(L),
                  delta="contractiva" if L < 1 else "NO contractiva",
                  delta_color="normal" if L < 1 else "inverse")
    col_gx.metric("g([a,b]) ⊆ [a,b]",
                   "Si" if analisis["g_de_X_en_X"] else "No",
                   delta=f"rango=[{fmt_decimal(analisis['g_min'])}, {fmt_decimal(analisis['g_max'])}]")
    col_ver.metric("Pronostico Banach",
                    "Converge" if (L < 1 and analisis["g_de_X_en_X"]) else "Puede divergir",
                    delta="✓" if L < 1 else "⚠",
                    delta_color="normal" if L < 1 else "inverse")
    st.caption(f"g'(x) = {sp.latex(analisis['gp_expr'])}")

    if L >= 1:
        st.warning(
            f"**|g'| ≥ 1 en [a,b]** (L = {fmt_decimal(L)}). La iteracion puede no converger."
        )

    # Criterios
    with st.expander("Criterios de detencion", expanded=not cfg.modo_examen):
        col1, col2 = st.columns(2)
        with col1:
            usar_abs = st.checkbox("Error absoluto", value=True, key="pf_ua")
            tol_abs = st.number_input("ε absoluto", value=1e-6, format="%.1e",
                                       min_value=0.0, disabled=not usar_abs, key="pf_ta")
            usar_rel = st.checkbox("Error relativo", value=False, key="pf_ur")
            tol_rel = st.number_input("ε relativo", value=1e-6, format="%.1e",
                                       min_value=0.0, disabled=not usar_rel, key="pf_tr")
        with col2:
            usar_res = st.checkbox("Residuo |g(x)−x|", value=True, key="pf_ures")
            tol_res = st.number_input("ε residuo", value=1e-6, format="%.1e",
                                       min_value=0.0, disabled=not usar_res, key="pf_tres")
            usar_max = st.checkbox("Max iteraciones", value=True, key="pf_um")
            max_iter = st.number_input("N_max", value=100, min_value=1, max_value=10_000,
                                        disabled=not usar_max, key="pf_nmax")

    criterios = CriteriosDetencion(
        usar_abs=usar_abs, tol_abs=tol_abs,
        usar_rel=usar_rel, tol_rel=tol_rel,
        usar_residuo=usar_res, tol_residuo=tol_res,
        usar_max_iter=usar_max, max_iter=int(max_iter),
    )

    g_np = sp.lambdify(x_sym, g_expr, modules=["numpy"])
    precision_usada = 5 if cfg.modo_libro else None
    res = punto_fijo(g_np, x0, criterios, a, b, L_estimado=L, precision=precision_usada)

    if not res.iteraciones:
        st.error(res.motivo_corte or "No se pudieron ejecutar iteraciones.")
        return

    st.subheader("Resultado")
    if res.convergio:
        st.success(f"Convergio — {res.motivo_corte}")
    else:
        st.warning(f"NO convergio — {res.motivo_corte}")

    col_r, col_n, col_e = st.columns(3)
    col_r.metric("Punto fijo aproximado", fmt_decimal(res.raiz))
    col_n.metric("Iteraciones", len(res.iteraciones))
    col_e.metric("|g(x*)−x*|", fmt_decimal(res.iteraciones[-1].error_abs))

    tab_resumen, tab_pasos, tab_viz = st.tabs(["Resumen", "Paso a paso", "Visualizaciones"])

    with tab_resumen:
        df = _tabla_dataframe(res)
        resaltar = resaltar_tolerancia("E_abs", criterios.tol_abs) if criterios.usar_abs else None
        render_tabla_iteraciones(df, resaltar=resaltar,
                                  titulo="Tabla de iteraciones", key_export="punto_fijo")

    with tab_pasos:
        n_pasos = st.slider("Iteraciones a explicar", 1, min(10, len(res.iteraciones)),
                             value=min(3, len(res.iteraciones)), key="pf_npasos")
        render_pasos(_construir_pasos(res, n_pasos=n_pasos), titulo="")

    with tab_viz:
        st.plotly_chart(_plot_cobweb(g_np, res, a, b), use_container_width=True)
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.plotly_chart(_plot_convergencia(res), use_container_width=True)
        with col_g2:
            st.plotly_chart(_plot_error(res), use_container_width=True)

    if cfg.mostrar_glosario:
        render_glosario_expander(
            vars=["x_0", "x_n", "g", "g_prima", "x_star", "L", "E_abs", "E_r", "epsilon", "N_max", "n"],
            titulo="Glosario — Punto Fijo",
        )


def render() -> None:
    render_punto_fijo()
