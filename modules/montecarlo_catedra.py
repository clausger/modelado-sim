"""Vista Catedra: replica el enfoque del simulador del profesor.

Diferencias con el modulo principal:
  - Usa Gauss-Legendre como valor de referencia numerico (ademas de SymPy).
  - Intervalos de confianza con t-student (no z-normal) y convencion
    stderr = sigma * (b-a) / sqrt(n), tal como el script del profesor.
  - Muestra hit-or-miss + metodo promedio + Gauss lado a lado en 1D.
  - Histograma de f(x)*(b-a) con ajuste normal superpuesto.

Se mantiene la estructura de tabs (Resumen / Paso a paso / Visualizaciones)
del modulo principal para uniformidad.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sympy as sp
from numpy.polynomial.legendre import leggauss
from scipy import integrate
from scipy.stats import norm, t as student_t

from utils.errores import error_absoluto, error_relativo
from utils.math_keyboard import math_input, parse_expr_to_float, parse_latex

from modules.montecarlo import (
    _fmt_decimal,
    _sample_uniform,
    _sample_uniform_2d,
    _seed_rng,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _gauss_legendre_1d(f_np, a: float, b: float, n_pts: int) -> float:
    """Cuadratura Gauss-Legendre en [a, b] con n_pts nodos."""
    nodes, weights = leggauss(int(n_pts))
    x_trans = 0.5 * (nodes + 1.0) * (b - a) + a
    return float(0.5 * (b - a) * np.sum(weights * f_np(x_trans)))


def _t_ci(media: float, std: float, n: int, conf_pct: float) -> tuple[float, float, float]:
    """Intervalo de confianza con t-student (convencion del profesor).

    stderr = std * (b-a) / sqrt(n) debe venir ya aplicado en `std` si el caller
    decidio incorporar el factor de volumen. Aca solo divide por sqrt(n) y
    multiplica por el cuantil t.
    """
    alpha = 1.0 - conf_pct / 100.0
    t_val = float(student_t.ppf(1.0 - alpha / 2.0, df=n - 1))
    stderr = std / np.sqrt(n)
    return media - t_val * stderr, media + t_val * stderr, t_val


def _render_teoria_catedra() -> None:
    with st.expander("📖 Enfoque de la vista cátedra (script del profesor)", expanded=False):
        st.markdown(
            "Esta vista reproduce el simulador compartido por el profesor. "
            "Las diferencias principales con el módulo principal son:"
        )
        st.markdown(
            "- **Valor de referencia numérico**: cuadratura de Gauss-Legendre "
            "con *n* nodos (configurable), además del valor simbólico de SymPy.\n"
            "- **Convención del error estándar**: se multiplica por el factor "
            "de volumen *(b−a)* antes de dividir por √n.\n"
            "- **Intervalo de confianza**: t-Student con *n−1* grados de libertad.\n"
            "- **Hit-or-miss + promedio + Gauss**: los tres métodos se muestran "
            "lado a lado para comparar convergencia."
        )
        st.latex(r"\text{Gauss-Legendre: } \int_a^b f(x)\,dx \approx \frac{b-a}{2} \sum_{k=1}^{m} w_k\, f\!\left(\tfrac{b-a}{2}\xi_k + \tfrac{a+b}{2}\right)")
        st.latex(r"\text{IC (cátedra): } \hat{I} \pm t_{\alpha/2,\,n-1} \cdot \frac{\sigma \cdot (b-a)}{\sqrt{n}}")


# --------------------------------------------------------------------------- #
# 1D
# --------------------------------------------------------------------------- #
def _vista_catedra_1d() -> None:
    st.subheader("Vista Cátedra — Integración 1D")
    _render_teoria_catedra()

    latex = math_input(label="f(x) =", default_latex="\\sin(x)+1", key="mc_cat1d_func")
    col1, col2 = st.columns(2)
    with col1:
        a_str = st.text_input("Límite inferior (a)", value="0", key="mc_cat1d_a")
        b_str = st.text_input("Límite superior (b)", value="pi", key="mc_cat1d_b")
    with col2:
        n_muestras = st.number_input(
            "Número de muestras (N)", value=1000, min_value=10,
            max_value=10_000_000, step=500, key="mc_cat1d_n",
        )
        n_gauss = st.number_input(
            "Nodos Gauss-Legendre", value=5, min_value=2, max_value=64,
            step=1, key="mc_cat1d_gauss",
        )
        semilla = st.number_input(
            "Semilla (0 = sin semilla)", value=0, min_value=0, key="mc_cat1d_seed",
        )

    conf_pct = st.selectbox(
        "Nivel de confianza (t-Student)",
        [90.0, 95.0, 99.0], index=1, key="mc_cat1d_conf",
    )

    if not st.button("Calcular", key="mc_cat1d_calc"):
        return

    x_sym = sp.Symbol("x")
    expr, f_np = parse_latex(latex, [x_sym])
    if expr is None:
        return
    a = parse_expr_to_float(a_str, "a")
    b = parse_expr_to_float(b_str, "b")
    if a is None or b is None:
        return
    if b <= a:
        st.error("El límite superior debe ser mayor que el inferior.")
        return

    _seed_rng(int(semilla))

    # Determinar rectangulo envolvente
    xs_dense = np.linspace(a, b, 1000)
    ys_dense = np.nan_to_num(f_np(xs_dense))
    y_min = float(min(0.0, np.min(ys_dense)))
    y_max = float(max(0.0, np.max(ys_dense)))

    # Hit-or-miss: el profesor genera primero todas las x, luego todas las y
    xs = _sample_uniform(a, b, int(n_muestras))
    ys = _sample_uniform(y_min, y_max, int(n_muestras))
    fx_vals = np.nan_to_num(f_np(xs))

    # Mascara de exito maneja areas negativas (curvas debajo del eje)
    success_mask = ((ys >= 0) & (ys <= fx_vals)) | ((ys <= 0) & (ys >= fx_vals))
    count = int(np.sum(success_mask))

    rect_area = (b - a) * (y_max - y_min)
    # En hit-or-miss la convencion del profe conserva el signo mediante el
    # balance de puntos arriba/abajo del eje — usamos la formula equivalente:
    # suma algebraica de areas firmadas.
    signo = np.where(fx_vals >= 0, 1.0, -1.0)
    hit_or_miss = float(rect_area * np.sum(success_mask * signo) / int(n_muestras))
    mc_promedio = float((b - a) * np.mean(fx_vals))
    gauss_val = _gauss_legendre_1d(f_np, a, b, int(n_gauss))

    # IC con convencion del profesor: stderr = std * (b-a) / sqrt(n), t-Student
    std_fx = float(np.std(fx_vals, ddof=1))
    std_vol = std_fx * (b - a)
    ic_low, ic_up, t_val = _t_ci(mc_promedio, std_vol, int(n_muestras), float(conf_pct))
    stderr_vol = std_vol / np.sqrt(int(n_muestras))

    try:
        exacto = float(sp.integrate(expr, (x_sym, a, b)).evalf())
        if not np.isfinite(exacto):
            exacto = None
    except Exception:
        exacto = None

    tab_resumen, tab_pasos, tab_viz = st.tabs(
        ["📊 Resumen", "🧮 Paso a paso", "📈 Visualizaciones"]
    )

    # ---------- RESUMEN ----------
    with tab_resumen:
        with st.container(border=True):
            st.markdown("##### Tres métodos")
            col1, col2, col3 = st.columns(3)
            col1.metric("Hit-or-miss", f"{hit_or_miss:.6f}")
            col2.metric("MC promedio", f"{mc_promedio:.6f}")
            col3.metric(f"Gauss-Legendre ({n_gauss} pts)", f"{gauss_val:.6f}")

        with st.container(border=True):
            st.markdown("##### Error estándar e IC (t-Student)")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("σ · (b−a)", _fmt_decimal(std_vol))
            col_b.metric("EE = σ·(b−a)/√n", _fmt_decimal(stderr_vol))
            col_c.metric(f"IC {conf_pct:g}%", f"[{ic_low:.6f}, {ic_up:.6f}]")
            st.caption(
                f"t_(α/2, n−1) = {t_val:.4f}  |  gl = {int(n_muestras) - 1}  |  "
                "referencia de convergencia: Gauss-Legendre."
            )

        if exacto is not None:
            with st.container(border=True):
                st.markdown("##### Comparación con valor exacto (SymPy)")
                col_e1, col_e2, col_e3 = st.columns(3)
                col_e1.metric("Valor exacto", f"{exacto:.6f}")
                err_prom = error_absoluto(mc_promedio, exacto)
                err_gauss = error_absoluto(gauss_val, exacto)
                col_e2.metric("Error abs. promedio", _fmt_decimal(err_prom))
                col_e3.metric("Error abs. Gauss", _fmt_decimal(err_gauss))

    # ---------- PASO A PASO ----------
    with tab_pasos:
        st.markdown("#### 1. Dominio y rectángulo envolvente")
        st.latex(
            rf"[a, b] = [{a}, {b}] \quad | \quad y_{{min}} = {y_min:.6f},\ "
            rf"y_{{max}} = {y_max:.6f}"
        )
        st.latex(rf"\text{{Área del rectángulo}} = (b-a)(y_{{max}}-y_{{min}}) = {rect_area:.6f}")

        st.markdown("#### 2. Hit-or-miss")
        st.latex(
            rf"\text{{Éxitos}} = {count} \;/\; N = {int(n_muestras)} "
            rf"\Rightarrow \hat{{I}}_{{HM}} \approx A_{{rect}} \cdot "
            rf"\tfrac{{\#\text{{éxitos firmados}}}}{{N}} = {hit_or_miss:.6f}"
        )

        st.markdown("#### 3. Método del promedio")
        st.latex(
            rf"\hat{{I}}_{{prom}} = (b-a)\cdot \bar{{f}} = "
            rf"({b-a:.6f})\cdot({np.mean(fx_vals):.6f}) = {mc_promedio:.6f}"
        )

        st.markdown("#### 4. Gauss-Legendre")
        nodes, weights = leggauss(int(n_gauss))
        x_trans = 0.5 * (nodes + 1.0) * (b - a) + a
        df_gl = pd.DataFrame({
            "k": np.arange(1, int(n_gauss) + 1),
            "ξ_k (nodo)": nodes,
            "w_k (peso)": weights,
            "x_k = (b-a)/2·ξ_k + (a+b)/2": x_trans,
            "f(x_k)": f_np(x_trans),
        })
        st.dataframe(df_gl, use_container_width=True, hide_index=True)
        st.latex(rf"\hat{{I}}_{{GL}} = \tfrac{{b-a}}{{2}}\sum w_k f(x_k) = {gauss_val:.6f}")

        st.markdown("#### 5. Intervalo de confianza (t-Student)")
        st.latex(
            rf"\sigma = {std_fx:.6f}, \quad \sigma\cdot(b-a) = {std_vol:.6f}, "
            rf"\quad \text{{EE}} = {stderr_vol:.6f}"
        )
        st.latex(
            rf"t_{{\alpha/2,\,n-1}} = {t_val:.4f} \Rightarrow "
            rf"\text{{IC}}_{{{conf_pct:g}\%}} = [{ic_low:.6f},\ {ic_up:.6f}]"
        )

    # ---------- VISUALIZACIONES ----------
    with tab_viz:
        # Scatter hit-or-miss estilo profesor
        st.markdown("#### Scatter hit-or-miss")
        max_plot = min(int(n_muestras), 5000)
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=xs_dense, y=ys_dense, mode="lines", name="f(x)",
            line=dict(color="royalblue", width=2),
        ))
        fig_sc.add_trace(go.Scatter(
            x=xs[:max_plot][success_mask[:max_plot]],
            y=ys[:max_plot][success_mask[:max_plot]],
            mode="markers", name="Éxitos",
            marker=dict(color="green", size=4, opacity=0.6),
        ))
        fig_sc.add_trace(go.Scatter(
            x=xs[:max_plot][~success_mask[:max_plot]],
            y=ys[:max_plot][~success_mask[:max_plot]],
            mode="markers", name="Fallidos",
            marker=dict(color="red", size=4, opacity=0.6),
        ))
        fig_sc.add_hline(y=0, line_color="black", line_width=1)
        fig_sc.update_layout(
            xaxis_title="x", yaxis_title="y",
            title=f"Hit-or-miss: {hit_or_miss:.6f} | Promedio: {mc_promedio:.6f} | Gauss: {gauss_val:.6f}",
            height=450,
        )
        st.plotly_chart(fig_sc, use_container_width=True)

        # Convergencia vs Gauss
        st.markdown("#### Convergencia del método promedio vs Gauss-Legendre")
        cum_avg = np.cumsum(fx_vals) / np.arange(1, int(n_muestras) + 1)
        cum_est = cum_avg * (b - a)
        # desvío acumulado
        std_run = np.empty(int(n_muestras))
        std_run[0] = 0.0
        for i in range(1, int(n_muestras)):
            std_run[i] = np.std(fx_vals[: i + 1], ddof=1)
        banda = std_run * (b - a) / np.sqrt(np.arange(1, int(n_muestras) + 1))
        t_banda = float(student_t.ppf(1.0 - (1.0 - conf_pct / 100.0) / 2.0, df=int(n_muestras) - 1))
        ic_cum_low = cum_est - t_banda * banda
        ic_cum_up = cum_est + t_banda * banda

        ns = np.arange(1, int(n_muestras) + 1)
        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(
            x=ns, y=cum_est, mode="lines", name="MC promedio acumulado",
            line=dict(color="steelblue"),
        ))
        fig_conv.add_trace(go.Scatter(
            x=np.concatenate([ns, ns[::-1]]),
            y=np.concatenate([ic_cum_up, ic_cum_low[::-1]]),
            fill="toself", fillcolor="rgba(100,100,100,0.2)",
            line=dict(color="rgba(0,0,0,0)"),
            name=f"IC {conf_pct:g}%", showlegend=True,
        ))
        fig_conv.add_hline(
            y=gauss_val, line_color="red", line_dash="dash",
            annotation_text=f"Gauss = {gauss_val:.4f}",
        )
        if exacto is not None:
            fig_conv.add_hline(
                y=exacto, line_color="green", line_dash="dot",
                annotation_text=f"Exacto = {exacto:.4f}",
            )
        fig_conv.update_layout(
            xaxis_title="n", yaxis_title="Estimación",
            xaxis_type="log", height=420,
        )
        st.plotly_chart(fig_conv, use_container_width=True)

        # Histograma de f(x)*(b-a) con normal ajustada (estilo profe)
        st.markdown("#### Distribución muestral de f(x)·(b−a)")
        datos = fx_vals * (b - a)
        bins = min(30, max(5, int(n_muestras) // 5))
        x_norm = np.linspace(float(datos.min()), float(datos.max()), 200)
        y_norm = norm.pdf(x_norm, float(np.mean(datos)), float(np.std(datos, ddof=1)))

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=datos, nbinsx=bins, histnorm="probability density",
            marker=dict(color="lightblue", line=dict(color="black", width=1)),
            name="Muestras",
        ))
        fig_hist.add_trace(go.Scatter(
            x=x_norm, y=y_norm, mode="lines", name="Normal ajustada",
            line=dict(color="orange", width=2),
        ))
        fig_hist.add_vline(x=mc_promedio, line_color="blue", annotation_text="Media")
        fig_hist.add_vline(x=ic_low, line_color="red", line_dash="dash", annotation_text="IC-")
        fig_hist.add_vline(x=ic_up, line_color="red", line_dash="dash", annotation_text="IC+")
        fig_hist.update_layout(
            xaxis_title="f(x)·(b−a)", yaxis_title="Densidad", height=400,
        )
        st.plotly_chart(fig_hist, use_container_width=True)


# --------------------------------------------------------------------------- #
# 2D / 3D
# --------------------------------------------------------------------------- #
def _vista_catedra_multidim() -> None:
    st.subheader("Vista Cátedra — Integrales Múltiples")
    _render_teoria_catedra()

    n_dims = st.radio("Dimensiones", [2, 3], horizontal=True, key="mc_catnd_dims")

    if n_dims == 2:
        latex = math_input(label="f(x,y) =", default_latex="x\\cdot y", key="mc_catnd_func")
    else:
        latex = math_input(label="f(x,y,z) =", default_latex="x\\cdot y\\cdot z", key="mc_catnd_func")

    cols = st.columns(n_dims)
    nombres = ["x", "y", "z"]
    rangos_str: list[tuple[str, str]] = []
    for i in range(n_dims):
        with cols[i]:
            ai = st.text_input(f"{nombres[i]} min", value="0", key=f"mc_catnd_{nombres[i]}_min")
            bi = st.text_input(f"{nombres[i]} max", value="1", key=f"mc_catnd_{nombres[i]}_max")
            rangos_str.append((ai, bi))

    col_a, col_b = st.columns(2)
    with col_a:
        n_muestras = st.number_input(
            "Número de muestras (N)",
            value=500 if n_dims == 2 else 2000,
            min_value=100, max_value=10_000_000, step=500,
            key="mc_catnd_n",
        )
    with col_b:
        semilla = st.number_input("Semilla (0 = sin semilla)", value=0, min_value=0, key="mc_catnd_seed")

    conf_pct = st.selectbox(
        "Nivel de confianza (t-Student)",
        [90.0, 95.0, 99.0], index=1, key="mc_catnd_conf",
    )

    if not st.button("Calcular", key="mc_catnd_calc"):
        return

    simbolos = [sp.Symbol(nombres[i]) for i in range(n_dims)]
    expr, f_np = parse_latex(latex, simbolos)
    if expr is None:
        return

    rangos: list[tuple[float, float]] = []
    for i, (ai_str, bi_str) in enumerate(rangos_str):
        ai = parse_expr_to_float(ai_str, f"{nombres[i]} min")
        bi = parse_expr_to_float(bi_str, f"{nombres[i]} max")
        if ai is None or bi is None:
            return
        if bi <= ai:
            st.error(f"El límite superior de {nombres[i]} debe ser mayor que el inferior.")
            return
        rangos.append((ai, bi))

    _seed_rng(int(semilla))
    puntos = [_sample_uniform(rangos[i][0], rangos[i][1], int(n_muestras)) for i in range(n_dims)]
    fx_vals = np.nan_to_num(f_np(*puntos))

    volumen = 1.0
    for ai, bi in rangos:
        volumen *= (bi - ai)

    estimacion = float(volumen * np.mean(fx_vals))
    std_fx = float(np.std(fx_vals, ddof=1))
    std_vol = std_fx * volumen
    stderr_vol = std_vol / np.sqrt(int(n_muestras))
    ic_low, ic_up, t_val = _t_ci(estimacion, std_vol, int(n_muestras), float(conf_pct))

    # Referencia numerica (scipy) ademas de SymPy
    ref_numerica = None
    try:
        if n_dims == 2:
            ref_numerica, _ = integrate.dblquad(
                lambda y, x: float(f_np(x, y)),
                rangos[0][0], rangos[0][1],
                lambda _x: rangos[1][0], lambda _x: rangos[1][1],
            )
        else:
            ref_numerica, _ = integrate.tplquad(
                lambda z, y, x: float(f_np(x, y, z)),
                rangos[0][0], rangos[0][1],
                lambda _x: rangos[1][0], lambda _x: rangos[1][1],
                lambda _x, _y: rangos[2][0], lambda _x, _y: rangos[2][1],
            )
    except Exception:
        ref_numerica = None

    exacto = None
    try:
        limites = [(simbolos[i], rangos[i][0], rangos[i][1]) for i in range(n_dims)]
        valor_sym = float(sp.integrate(expr, *limites).evalf())
        if np.isfinite(valor_sym):
            exacto = valor_sym
    except Exception:
        exacto = None

    tab_resumen, tab_pasos, tab_viz = st.tabs(
        ["📊 Resumen", "🧮 Paso a paso", "📈 Visualizaciones"]
    )

    with tab_resumen:
        with st.container(border=True):
            st.markdown("##### Estimación Monte Carlo")
            col1, col2, col3 = st.columns(3)
            col1.metric("Estimación", f"{estimacion:.6f}")
            col2.metric("EE = σ·V/√n", _fmt_decimal(stderr_vol))
            col3.metric(f"IC {conf_pct:g}%", f"[{ic_low:.6f}, {ic_up:.6f}]")
            st.caption(f"t_(α/2, n−1) = {t_val:.4f}  |  V(D) = {volumen:.6f}")

        if exacto is not None or ref_numerica is not None:
            with st.container(border=True):
                st.markdown("##### Valores de referencia")
                col_e1, col_e2, col_e3 = st.columns(3)
                if exacto is not None:
                    col_e1.metric("Exacto (SymPy)", f"{exacto:.6f}")
                    col_e2.metric("Error abs.", _fmt_decimal(error_absoluto(estimacion, exacto)))
                    col_e3.metric("Error rel.", _fmt_decimal(error_relativo(estimacion, exacto)))
                elif ref_numerica is not None:
                    col_e1.metric("Referencia (scipy)", f"{ref_numerica:.6f}")
                    col_e2.metric("Error abs.", _fmt_decimal(error_absoluto(estimacion, ref_numerica)))
                    col_e3.metric("Error rel.", _fmt_decimal(error_relativo(estimacion, ref_numerica)))

    with tab_pasos:
        st.markdown("#### 1. Dominio de integración")
        for i, (ai, bi) in enumerate(rangos):
            st.latex(rf"{nombres[i]} \in [{ai},\ {bi}]")
        st.latex(rf"V(D) = \prod_i (b_i - a_i) = {volumen:.6f}")

        st.markdown("#### 2. Estimador del promedio")
        st.latex(
            rf"\hat{{I}} = V(D)\cdot \bar{{f}} = ({volumen:.6f})\cdot({np.mean(fx_vals):.6f}) "
            rf"= {estimacion:.6f}"
        )

        st.markdown("#### 3. IC con t-Student")
        st.latex(
            rf"\sigma = {std_fx:.6f}, \quad \sigma\cdot V(D) = {std_vol:.6f}, "
            rf"\quad \text{{EE}} = {stderr_vol:.6f}"
        )
        st.latex(
            rf"t_{{\alpha/2,\,n-1}} = {t_val:.4f} \Rightarrow "
            rf"\text{{IC}}_{{{conf_pct:g}\%}} = [{ic_low:.6f},\ {ic_up:.6f}]"
        )

        st.markdown("#### 4. Muestra de puntos")
        max_show = min(int(n_muestras), 500)
        df_data = {nombres[i]: puntos[i][:max_show] for i in range(n_dims)}
        df_data["f"] = fx_vals[:max_show]
        st.dataframe(pd.DataFrame(df_data), use_container_width=True, height=300)

    with tab_viz:
        max_plot = min(int(n_muestras), 5000)
        if n_dims == 2:
            fig = go.Figure(data=go.Scatter(
                x=puntos[0][:max_plot], y=puntos[1][:max_plot],
                mode="markers",
                marker=dict(
                    color=fx_vals[:max_plot], colorscale="Viridis",
                    size=6, showscale=True, colorbar=dict(title="f(x,y)"),
                ),
            ))
            fig.update_layout(
                xaxis_title="x", yaxis_title="y",
                title=f"Integral doble ≈ {estimacion:.6f}",
                height=500,
            )
        else:
            fig = go.Figure(data=go.Scatter3d(
                x=puntos[0][:max_plot], y=puntos[1][:max_plot], z=puntos[2][:max_plot],
                mode="markers",
                marker=dict(
                    color=fx_vals[:max_plot], colorscale="Viridis",
                    size=3, showscale=True, colorbar=dict(title="f(x,y,z)"),
                ),
            ))
            fig.update_layout(
                scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
                title=f"Integral triple ≈ {estimacion:.6f}",
                height=550,
            )
        st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #
def render_catedra() -> None:
    """Submenu de vista cátedra (1D / multidimensional)."""
    tipo = st.radio(
        "Tipo de integral:",
        ["1D", "2D / 3D"],
        horizontal=True,
        key="mc_cat_tipo",
    )
    if tipo == "1D":
        _vista_catedra_1d()
    else:
        _vista_catedra_multidim()
