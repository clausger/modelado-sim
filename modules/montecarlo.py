import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sympy as sp
from scipy import integrate

from utils.errores import error_absoluto, error_relativo, intervalo_confianza_95
from utils.graficos import (
    plot_comparacion_barras,
    plot_convergencia,
    plot_funcion,
    plot_scatter_3d,
    plot_scatter_montecarlo,
)
from utils.math_keyboard import math_input, parse_latex


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

    st.latex(r"I = \int_a^b f(x)\,dx \approx \frac{b-a}{N}\sum_{i=1}^{N} f(x_i)")
    st.latex(r"x_i \sim \mathcal{U}(a, b)")

    latex = math_input(label="f(x) =", default_latex="x^{2}+\\sin(x)", key="mc1d_func")
    col1, col2 = st.columns(2)
    with col1:
        a = st.number_input("Limite inferior (a)", value=0.0, key="mc1d_a")
        b = st.number_input("Limite superior (b)", value=2.0, key="mc1d_b")
    with col2:
        n_muestras = st.number_input("Numero de muestras (N)", value=10000, min_value=10,
                                     max_value=10_000_000, step=1000, key="mc1d_n")
        semilla = st.number_input("Semilla aleatoria (0 = sin semilla)", value=42,
                                  min_value=0, key="mc1d_seed")
        n_decimales = st.number_input("Precision (decimales)", value=4, min_value=1,
                                      max_value=15, key="mc1d_tol")

    tolerancia = 10 ** (-n_decimales)
    st.latex(rf"\text{{Precision: }} 10^{{-{n_decimales}}} = {tolerancia}")

    if st.button("Calcular", key="mc1d_calc"):
        x_sym = sp.Symbol("x")
        expr, f_np = parse_latex(latex, [x_sym])
        if expr is None:
            return

        rng = np.random.default_rng(semilla if semilla > 0 else None)
        x_vals = rng.uniform(a, b, size=int(n_muestras))
        f_vals = f_np(x_vals)

        estimacion = (b - a) * np.mean(f_vals)
        error_std = (b - a) * np.std(f_vals, ddof=1) / np.sqrt(len(f_vals))
        media, ic_low, ic_up = intervalo_confianza_95((b - a) * f_vals / len(f_vals))

        ic_low_final = estimacion - 1.96 * error_std
        ic_up_final = estimacion + 1.96 * error_std

        valor_exacto = _calcular_valor_exacto_1d(expr, x_sym, a, b)

        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Estimacion", f"{estimacion:.{n_decimales}f}")
        col_r2.metric("Error estandar", f"{error_std:.2e}")
        col_r3.metric("IC 95%", f"[{ic_low_final:.{n_decimales}f}, {ic_up_final:.{n_decimales}f}]")

        if valor_exacto is not None:
            col_e1, col_e2 = st.columns(2)
            col_e1.metric("Valor exacto (SymPy)", f"{valor_exacto:.{n_decimales}f}")
            col_e2.metric("Error absoluto", f"{error_absoluto(estimacion, valor_exacto):.2e}")

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
                "error_absoluto": "{:.2e}",
                "error_relativo": "{:.2e}",
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
        plot_y = rng.uniform(y_min_f, y_max_f, size=max_plot)
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
            std_p = (b - a) * np.std(sub_vals, ddof=1) / np.sqrt(n_p)
            est_conv.append(est_p)
            ic_lows.append(est_p - 1.96 * std_p)
            ic_ups.append(est_p + 1.96 * std_p)

        fig_conv = plot_convergencia(potencias, est_conv, ic_lows, ic_ups, valor_exacto)
        st.plotly_chart(fig_conv, use_container_width=True)

        st.session_state["mc1d_resultado"] = estimacion


def _integracion_multidimensional():
    st.subheader("Integracion Monte Carlo Multidimensional")

    st.latex(r"I = \int_D f(\mathbf{x})\,d\mathbf{x} \approx \frac{V(D)}{N}\sum_{i=1}^{N} f(\mathbf{x}_i)")

    n_dims = st.radio("Dimensiones", [2, 3], horizontal=True, key="mc_nd_dims")

    if n_dims == 2:
        latex = math_input(label="f(x,y) =", default_latex="x^{2}+y^{2}", key="mc_nd_func")
    else:
        latex = math_input(label="f(x,y,z) =", default_latex="x^{2}+y^{2}+z^{2}", key="mc_nd_func")

    cols = st.columns(n_dims)
    rangos = []
    nombres = ["x", "y", "z"]
    for i in range(n_dims):
        with cols[i]:
            ai = st.number_input(f"{nombres[i]} min", value=0.0, key=f"mc_nd_{nombres[i]}_min")
            bi = st.number_input(f"{nombres[i]} max", value=1.0, key=f"mc_nd_{nombres[i]}_max")
            rangos.append((ai, bi))

    n_muestras = st.number_input("Numero de muestras (N)", value=50000, min_value=100,
                                 max_value=10_000_000, step=5000, key="mc_nd_n")
    semilla = st.number_input("Semilla aleatoria (0 = sin semilla)", value=42,
                              min_value=0, key="mc_nd_seed")

    if st.button("Calcular", key="mc_nd_calc"):
        simbolos = [sp.Symbol(nombres[i]) for i in range(n_dims)]
        expr, f_np = parse_latex(latex, simbolos)
        if expr is None:
            return

        rng = np.random.default_rng(semilla if semilla > 0 else None)
        puntos = [rng.uniform(rangos[i][0], rangos[i][1], size=int(n_muestras)) for i in range(n_dims)]

        f_vals = f_np(*puntos)
        volumen = 1.0
        for ai, bi in rangos:
            volumen *= (bi - ai)

        estimacion = volumen * np.mean(f_vals)
        error_std = volumen * np.std(f_vals, ddof=1) / np.sqrt(int(n_muestras))
        ic_low = estimacion - 1.96 * error_std
        ic_up = estimacion + 1.96 * error_std

        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Estimacion", f"{estimacion:.6f}")
        col_r2.metric("Error estandar", f"{error_std:.2e}")
        col_r3.metric("IC 95%", f"[{ic_low:.6f}, {ic_up:.6f}]")

        # Intentar valor exacto
        try:
            limites = [(simbolos[i], rangos[i][0], rangos[i][1]) for i in range(n_dims)]
            exacto_sym = sp.integrate(expr, *limites)
            exacto = float(exacto_sym.evalf())
            if np.isfinite(exacto):
                col_e1, col_e2 = st.columns(2)
                col_e1.metric("Valor exacto (SymPy)", f"{exacto:.6f}")
                col_e2.metric("Error absoluto", f"{error_absoluto(estimacion, exacto):.2e}")
        except Exception:
            pass

        # Grafico 3D para 2D
        if n_dims == 2:
            st.markdown("#### Visualizacion 3D")
            max_plot = min(int(n_muestras), 5000)
            x_plot = puntos[0][:max_plot]
            y_plot = puntos[1][:max_plot]
            z_plot = f_vals[:max_plot]

            z_random = rng.uniform(0, float(np.max(z_plot)) * 1.1, size=max_plot)
            dentro = z_random <= z_plot

            fig_3d = plot_scatter_3d(x_plot, y_plot, z_random, dentro)
            fig_3d.update_layout(scene=dict(
                xaxis_title="x", yaxis_title="y", zaxis_title="z"
            ))
            st.plotly_chart(fig_3d, use_container_width=True)


def _comparacion_metodos():
    st.subheader("Comparacion de Metodos")

    st.latex(r"\text{Monte Carlo vs Trapecios vs Simpson}")

    latex = math_input(label="f(x) =", default_latex="x^{2}+\\sin(x)", key="mc_comp_func")
    col1, col2 = st.columns(2)
    with col1:
        a = st.number_input("Limite inferior (a)", value=0.0, key="mc_comp_a")
        b = st.number_input("Limite superior (b)", value=2.0, key="mc_comp_b")
    with col2:
        n_muestras = st.number_input("N (Monte Carlo)", value=100000, min_value=100,
                                     max_value=10_000_000, key="mc_comp_n")
        n_trapecios = st.number_input("N (subdivisiones Trapecios/Simpson)", value=1000,
                                      min_value=2, max_value=1_000_000, key="mc_comp_nt")

    if st.button("Comparar", key="mc_comp_calc"):
        x_sym = sp.Symbol("x")
        expr, f_np = parse_latex(latex, [x_sym])
        if expr is None:
            return

        valor_exacto = _calcular_valor_exacto_1d(expr, x_sym, a, b)

        resultados = {}

        # Monte Carlo
        t0 = time.perf_counter()
        rng = np.random.default_rng(42)
        x_mc = rng.uniform(a, b, size=int(n_muestras))
        mc_est = (b - a) * np.mean(f_np(x_mc))
        t_mc = time.perf_counter() - t0
        resultados["Monte Carlo"] = (mc_est, t_mc)

        # Trapecios
        t0 = time.perf_counter()
        x_trap = np.linspace(a, b, int(n_trapecios) + 1)
        trap_est = float(np.trapz(f_np(x_trap), x_trap))
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

        if valor_exacto is not None:
            st.metric("Valor exacto (SymPy)", f"{valor_exacto:.10f}")

        formato = {"Resultado": "{:.10f}", "Tiempo (s)": "{:.6f}"}
        if valor_exacto is not None:
            formato["Error absoluto"] = "{:.2e}"
            formato["Error relativo"] = "{:.2e}"

        st.dataframe(df.style.format(formato), use_container_width=True)

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


def render():
    st.header("Metodo de Monte Carlo")

    submenu = st.radio(
        "Selecciona el submenu:",
        ["Integracion 1D", "Integracion Multidimensional", "Comparacion de Metodos"],
        horizontal=True,
        key="mc_submenu",
    )

    if submenu == "Integracion 1D":
        _integracion_1d()
    elif submenu == "Integracion Multidimensional":
        _integracion_multidimensional()
    else:
        _comparacion_metodos()
