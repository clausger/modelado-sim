"""Modulo de Ecuaciones Diferenciales Ordinarias (EDO).

Problema de Cauchy de primer orden: dy/dt = f(t, y), y(t0) = y0.

Metodos implementados (Fase 1):
- Euler explicito (orden 1)
- Heun / Euler mejorado (orden 2)
- Runge-Kutta 4 (orden 4)

Cada metodo produce tabla iteracion-por-iteracion con estilo Caceres:
    i | t_i | y_i | f(t_i, y_i) | y_{i+1} | y_real | |error|

RK4 ademas muestra tabla de k1..k4 por paso.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sympy as sp

from utils.math_keyboard import math_input, parse_expr_to_float, parse_latex
from utils.ui import get_config, render_teoria
from utils.ui.tablas import fmt_decimal


# ---------------------------------------------------------------------------
# Helpers simbolicos / formato (replicados de integracion.py por ahora —
# si crece, se consolidan en utils/simbolico.py)
# ---------------------------------------------------------------------------

def _fmt_tex(v: float, n_dec: int = 6) -> str:
    """Formato limpio para LaTeX: entero si es entero, sino n_dec decimales."""
    if not np.isfinite(v):
        return str(v)
    if abs(v - round(v)) < 1e-12:
        return str(int(round(v)))
    return f"{v:.{n_dec}f}"


def _tex(expr: sp.Expr) -> str:
    """sp.latex con notacion 'ln' (como escribe el profe)."""
    return sp.latex(expr, ln_notation=True)


# ---------------------------------------------------------------------------
# Algoritmos puros (sin Streamlit)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResultadoEDO:
    """Resultado completo de una simulacion EDO."""
    ts: np.ndarray            # vector de tiempos t_0 .. t_N
    ys: np.ndarray            # vector de aproximaciones y_0 .. y_N
    fs: np.ndarray            # f(t_i, y_i) evaluado (N+1 valores)
    ks: np.ndarray | None = None      # (N, 4) solo para RK4
    y_pred: np.ndarray | None = None  # predictor Heun, index i+1 = y*_{i+1}
    metodo: str = ""          # "euler" | "heun" | "rk4"
    h: float = 0.0
    n: int = 0


def _euler(f: Callable[[float, float], float], t0: float, y0: float,
           h: float, n: int) -> ResultadoEDO:
    ts = np.zeros(n + 1)
    ys = np.zeros(n + 1)
    fs = np.zeros(n + 1)
    ts[0], ys[0] = t0, y0
    for i in range(n):
        fi = float(f(ts[i], ys[i]))
        fs[i] = fi
        ts[i + 1] = ts[i] + h
        ys[i + 1] = ys[i] + h * fi
    fs[n] = float(f(ts[n], ys[n]))
    return ResultadoEDO(ts=ts, ys=ys, fs=fs, metodo="euler", h=h, n=n)


def _heun(f: Callable[[float, float], float], t0: float, y0: float,
          h: float, n: int) -> ResultadoEDO:
    ts = np.zeros(n + 1)
    ys = np.zeros(n + 1)
    fs = np.zeros(n + 1)        # aqui guardamos k1 = f(t_i, y_i)
    y_pred = np.zeros(n + 1)    # predictor y*_{i+1} (index i+1)
    ts[0], ys[0] = t0, y0
    for i in range(n):
        k1 = float(f(ts[i], ys[i]))
        y_star = ys[i] + h * k1          # predictor (Euler)
        y_pred[i + 1] = y_star
        t_next = ts[i] + h
        k2 = float(f(t_next, y_star))
        fs[i] = k1
        ts[i + 1] = t_next
        ys[i + 1] = ys[i] + (h / 2.0) * (k1 + k2)   # corrector (trapecio)
    fs[n] = float(f(ts[n], ys[n]))
    return ResultadoEDO(ts=ts, ys=ys, fs=fs, y_pred=y_pred,
                        metodo="heun", h=h, n=n)


def _rk4(f: Callable[[float, float], float], t0: float, y0: float,
         h: float, n: int) -> ResultadoEDO:
    ts = np.zeros(n + 1)
    ys = np.zeros(n + 1)
    fs = np.zeros(n + 1)
    ks = np.zeros((n, 4))
    ts[0], ys[0] = t0, y0
    for i in range(n):
        ti, yi = ts[i], ys[i]
        k1 = float(f(ti, yi))
        k2 = float(f(ti + h / 2.0, yi + (h / 2.0) * k1))
        k3 = float(f(ti + h / 2.0, yi + (h / 2.0) * k2))
        k4 = float(f(ti + h, yi + h * k3))
        ks[i] = [k1, k2, k3, k4]
        fs[i] = k1
        ts[i + 1] = ti + h
        ys[i + 1] = yi + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    fs[n] = float(f(ts[n], ys[n]))
    return ResultadoEDO(ts=ts, ys=ys, fs=fs, ks=ks, metodo="rk4", h=h, n=n)


# ---------------------------------------------------------------------------
# Solucion analitica via sp.dsolve
# ---------------------------------------------------------------------------

def _resolver_analitica(
    expr_f: sp.Expr,
    t_sym: sp.Symbol,
    y_sym: sp.Symbol,
    t0: float,
    y0: float,
) -> tuple[sp.Expr | None, Callable | None]:
    """Intenta resolver dy/dt = f(t,y) con y(t0)=y0 simbolicamente.

    Estrategia:
    1. dsolve con ics (ruta rapida, funciona para EDOs lineales).
    2. Si falla (ej: ambiguedad de constante al cuadrado), dsolve sin ics +
       resolver la constante manualmente eligiendo la raiz que matchea y0.

    Devuelve (expr_y_de_t, callable) o (None, None) si no hay solucion cerrada.
    """
    y_func = sp.Function("y")
    ode = sp.Eq(sp.Derivative(y_func(t_sym), t_sym),
                expr_f.subs(y_sym, y_func(t_sym)))

    # Pendiente esperada en (t0, y0) segun la EDO original
    try:
        slope_esperada = float(expr_f.subs({t_sym: t0, y_sym: y0}).evalf())
    except Exception:
        slope_esperada = None

    # --- Intento 1: dsolve con ics ---
    try:
        sol = sp.dsolve(ode, y_func(t_sym), ics={y_func(t0): y0})
        if isinstance(sol, list):
            sol = sol[0]
        y_expr = sp.simplify(sol.rhs)
        candidato = _validar_analitica(y_expr, t_sym, t0, y0, slope_esperada)
        if candidato is not None:
            return candidato
    except Exception:
        pass

    # --- Intento 2: dsolve sin ics + resolver constante manualmente ---
    try:
        sol = sp.dsolve(ode, y_func(t_sym))
        ramas = sol if isinstance(sol, list) else [sol]
        mejores: list[tuple[sp.Expr, Callable]] = []
        for rama in ramas:
            expr_gen = sp.simplify(rama.rhs)
            libres = expr_gen.free_symbols - {t_sym}
            if not libres:
                # No tiene constante — posiblemente ya esta particularizada
                candidato = _validar_analitica(expr_gen, t_sym, t0, y0, slope_esperada)
                if candidato is not None:
                    mejores.append(candidato)
                continue
            if len(libres) > 1:
                continue
            const = libres.pop()
            # Resolver expr_gen(t0) = y0 para 'const'
            try:
                soluciones = sp.solve(expr_gen.subs(t_sym, t0) - y0, const)
            except Exception:
                continue
            for val_const in soluciones:
                if not val_const.is_real and not val_const.is_number:
                    continue
                y_particular = sp.simplify(expr_gen.subs(const, val_const))
                candidato = _validar_analitica(
                    y_particular, t_sym, t0, y0, slope_esperada
                )
                if candidato is not None:
                    mejores.append(candidato)
        if mejores:
            return mejores[0]
    except Exception:
        pass

    return None, None


def _validar_analitica(
    y_expr: sp.Expr,
    t_sym: sp.Symbol,
    t0: float,
    y0: float,
    slope_esperada: float | None = None,
) -> tuple[sp.Expr, Callable] | None:
    """Verifica que y_expr pase por (t0, y0) Y que dy/dt(t0) matchee la EDO.

    El segundo chequeo es CRITICO para EDOs separables con formas cuadradas
    (ej. 2t·sqrt(y) → y = ((t² + C)/2)² tiene dos ramas, ambas cumplen
    y(t0)=y0 pero solo una satisface la EDO original).
    """
    libres = y_expr.free_symbols - {t_sym}
    if libres:
        return None
    try:
        y_np = sp.lambdify(t_sym, y_expr, modules=["numpy"])
        chk = float(y_np(t0))
    except Exception:
        return None
    if not np.isfinite(chk):
        return None
    if abs(chk - y0) > 1e-6 * max(1.0, abs(y0)):
        return None
    # Verificar que dy/dt(t0) coincida con f(t0, y0) — descarta rama espuria
    if slope_esperada is not None and np.isfinite(slope_esperada):
        try:
            dy_expr = sp.diff(y_expr, t_sym)
            slope_real = float(dy_expr.subs(t_sym, t0).evalf())
            if np.isfinite(slope_real):
                tol_slope = 1e-6 * max(1.0, abs(slope_esperada))
                if abs(slope_real - slope_esperada) > tol_slope:
                    return None
        except Exception:
            pass
    return y_expr, y_np


# ---------------------------------------------------------------------------
# Armado de tabla (con error absoluto y resaltado)
# ---------------------------------------------------------------------------

def _tabla_iteraciones(res: ResultadoEDO, var_indep: str,
                       y_analitica: Callable | None) -> pd.DataFrame:
    """Construye la tabla i | t_i | y_i | f(t_i,y_i) | y_{i+1} | y_real | |E|.

    El renglon i describe la transicion de y_i a y_{i+1}, por lo que hay N filas
    (una por paso).
    """
    filas = []
    for i in range(res.n):
        fila = {
            "i": i,
            var_indep + "_i": res.ts[i],
            "y_i": res.ys[i],
            f"f({var_indep}_i, y_i)": res.fs[i],
            "y_{i+1}": res.ys[i + 1],
        }
        if y_analitica is not None:
            y_real_next = float(y_analitica(res.ts[i + 1]))
            fila["y_real"] = y_real_next
            fila["|error|"] = abs(res.ys[i + 1] - y_real_next)
        filas.append(fila)
    return pd.DataFrame(filas)


def _tabla_ks_rk4(res: ResultadoEDO, var_indep: str) -> pd.DataFrame:
    """Tabla adicional para RK4: k1, k2, k3, k4 por paso."""
    assert res.ks is not None
    filas = []
    for i in range(res.n):
        filas.append({
            "i": i,
            var_indep + "_i": res.ts[i],
            "y_i": res.ys[i],
            "k_1": res.ks[i, 0],
            "k_2": res.ks[i, 1],
            "k_3": res.ks[i, 2],
            "k_4": res.ks[i, 3],
            "y_{i+1}": res.ys[i + 1],
        })
    return pd.DataFrame(filas)


def _tabla_predictor_corrector_heun(res: ResultadoEDO,
                                    var_indep: str) -> pd.DataFrame:
    """Tabla adicional para Heun: f(t_i,y_i), y*_{i+1} (predictor), y_{i+1}."""
    assert res.y_pred is not None
    filas = []
    for i in range(res.n):
        filas.append({
            "i": i,
            var_indep + "_i": res.ts[i],
            "y_i": res.ys[i],
            f"f({var_indep}_i, y_i)": res.fs[i],
            "y*_{i+1} (predictor)": res.y_pred[i + 1],
            "y_{i+1} (corrector)": res.ys[i + 1],
        })
    return pd.DataFrame(filas)


def _tabla_maestra(
    euler_res: ResultadoEDO,
    heun_res: ResultadoEDO,
    rk4_res: ResultadoEDO,
    var_indep: str,
    y_analitica: Callable | None,
) -> pd.DataFrame:
    """Tabla estilo profe: n | t | Euler | Heun | RK4 | Exacta.

    Una fila por nodo (n = 0..N). La columna Exacta aparece solo si hay
    solucion analitica.
    """
    filas = []
    for i in range(euler_res.n + 1):
        fila = {
            "n": i,
            var_indep: euler_res.ts[i],
            "Euler": euler_res.ys[i],
            "Heun": heun_res.ys[i],
            "RK4": rk4_res.ys[i],
        }
        if y_analitica is not None:
            try:
                fila["Exacta"] = float(y_analitica(euler_res.ts[i]))
            except Exception:
                fila["Exacta"] = float("nan")
        filas.append(fila)
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Paso a paso detallado (estilo del profe) — sustituye valores explicitos
# ---------------------------------------------------------------------------

def _paso_heun_detallado(res: ResultadoEDO, var_indep: str,
                         h: float, n_dec: int,
                         max_pasos: int = 10) -> None:
    """Renderiza cada paso de Heun mostrando predictor y corrector explicitos.

    Estilo del profe:
        Paso n=0: t=0, y=2
        Predictor:  y*_1 = y_0 + h·f(t_0, y_0) = 2 + (pi/10)·(2·sen(0)) = 2
        Corrector:  y_1 = y_0 + h/2·(f(t_0,y_0) + f(t_1, y*_1))
                        = 2 + (pi/20)·(0 + 2·sen(pi/10)) = 2.097081
    """
    assert res.y_pred is not None
    v = var_indep
    total = min(res.n, max_pasos)
    if res.n > max_pasos:
        st.caption(f"Mostrando los primeros {max_pasos} pasos (de {res.n} totales).")
    for i in range(total):
        ti = res.ts[i]
        yi = res.ys[i]
        k1 = res.fs[i]
        y_star = res.y_pred[i + 1]
        t_next = res.ts[i + 1]
        y_next = res.ys[i + 1]
        # Recupero k2 a partir del corrector:  y_{i+1} = y_i + h/2 (k1+k2)
        k2 = 2.0 * (y_next - yi) / h - k1

        st.markdown(
            f"**Paso n = {i}:** "
            f"${v} = {_fmt_tex(ti, n_dec)}$, "
            f"$y_{i} = {_fmt_tex(yi, n_dec)}$"
        )
        # Predictor
        st.latex(
            rf"y^{{*}}_{{{i+1}}} = y_{{{i}}} + h \cdot f({v}_{{{i}}}, y_{{{i}}}) "
            rf"= {_fmt_tex(yi, n_dec)} + ({_fmt_tex(h, n_dec)}) \cdot "
            rf"({_fmt_tex(k1, n_dec)}) = {_fmt_tex(y_star, n_dec)}"
        )
        # Corrector
        st.latex(
            rf"y_{{{i+1}}} = y_{{{i}}} + \tfrac{{h}}{{2}}\!\left[f({v}_{{{i}}}, y_{{{i}}}) "
            rf"+ f({v}_{{{i+1}}}, y^{{*}}_{{{i+1}}})\right]"
        )
        st.latex(
            rf"y_{{{i+1}}} = {_fmt_tex(yi, n_dec)} + "
            rf"\tfrac{{{_fmt_tex(h, n_dec)}}}{{2}}\!\left[{_fmt_tex(k1, n_dec)} + "
            rf"{_fmt_tex(k2, n_dec)}\right] = {_fmt_tex(y_next, n_dec)}"
        )
        if i < total - 1:
            st.divider()


def _paso_rk4_detallado(res: ResultadoEDO, var_indep: str,
                        h: float, n_dec: int,
                        max_pasos: int = 10) -> None:
    """Renderiza cada paso de RK4 mostrando k1..k4 con sustitucion numerica.

    Estilo del profe:
        Paso n=0: t=0, y=2
        k1 = f(0, 2) = 0
        k2 = f(0 + h/2, 2 + (h/2)·k1) = f(pi/20, 2) = ...
        k3 = f(0 + h/2, 2 + (h/2)·k2) = f(pi/20, ...) = ...
        k4 = f(0 + h, 2 + h·k3) = f(pi/10, ...) = ...
        y_1 = 2 + (h/6)·(k1 + 2k2 + 2k3 + k4) = ...
    """
    assert res.ks is not None
    v = var_indep
    total = min(res.n, max_pasos)
    if res.n > max_pasos:
        st.caption(f"Mostrando los primeros {max_pasos} pasos (de {res.n} totales).")
    for i in range(total):
        ti = res.ts[i]
        yi = res.ys[i]
        k1, k2, k3, k4 = (float(res.ks[i, j]) for j in range(4))
        y_next = res.ys[i + 1]
        ti_medio = ti + h / 2.0
        ti_final = ti + h
        y_k1 = yi + (h / 2.0) * k1
        y_k2 = yi + (h / 2.0) * k2
        y_k3 = yi + h * k3
        suma = k1 + 2 * k2 + 2 * k3 + k4

        st.markdown(
            f"**Paso n = {i}:** "
            f"${v} = {_fmt_tex(ti, n_dec)}$, "
            f"$y_{i} = {_fmt_tex(yi, n_dec)}$"
        )
        st.latex(
            rf"k_1 = f({v}_{{{i}}},\, y_{{{i}}}) "
            rf"= f({_fmt_tex(ti, n_dec)},\, {_fmt_tex(yi, n_dec)}) "
            rf"= {_fmt_tex(k1, n_dec)}"
        )
        st.latex(
            rf"k_2 = f\!\left({v}_{{{i}}} + \tfrac{{h}}{{2}},\, "
            rf"y_{{{i}}} + \tfrac{{h}}{{2}} k_1\right) "
            rf"= f({_fmt_tex(ti_medio, n_dec)},\, {_fmt_tex(y_k1, n_dec)}) "
            rf"= {_fmt_tex(k2, n_dec)}"
        )
        st.latex(
            rf"k_3 = f\!\left({v}_{{{i}}} + \tfrac{{h}}{{2}},\, "
            rf"y_{{{i}}} + \tfrac{{h}}{{2}} k_2\right) "
            rf"= f({_fmt_tex(ti_medio, n_dec)},\, {_fmt_tex(y_k2, n_dec)}) "
            rf"= {_fmt_tex(k3, n_dec)}"
        )
        st.latex(
            rf"k_4 = f({v}_{{{i}}} + h,\, y_{{{i}}} + h k_3) "
            rf"= f({_fmt_tex(ti_final, n_dec)},\, {_fmt_tex(y_k3, n_dec)}) "
            rf"= {_fmt_tex(k4, n_dec)}"
        )
        st.latex(
            rf"y_{{{i+1}}} = y_{{{i}}} + \tfrac{{h}}{{6}}(k_1 + 2k_2 + 2k_3 + k_4) "
            rf"= {_fmt_tex(yi, n_dec)} + "
            rf"\tfrac{{{_fmt_tex(h, n_dec)}}}{{6}}\,({_fmt_tex(suma, n_dec)}) "
            rf"= {_fmt_tex(y_next, n_dec)}"
        )
        if i < total - 1:
            st.divider()


def _analizar_comportamiento_error(errores: np.ndarray) -> str:
    """Devuelve un breve texto diagnosticando como evoluciona |error|."""
    if len(errores) < 2:
        return "Pocos pasos para analizar tendencia."
    err = np.asarray(errores, dtype=float)
    err = err[np.isfinite(err)]
    if len(err) < 2:
        return "Error no analizable (NaN/Inf)."
    e_ini, e_fin = err[0], err[-1]
    e_max = float(np.max(err))
    # Monotonia
    dif = np.diff(err)
    crece = np.all(dif >= -1e-14)
    decrece = np.all(dif <= 1e-14)
    oscila = not crece and not decrece

    if e_max == 0:
        return ("El error se mantiene en 0 en todos los pasos — la aproximacion "
                "coincide exactamente con la solucion analitica.")
    ratio = e_fin / e_ini if e_ini > 0 else float("inf")
    if crece:
        tendencia = (f"crece de |E_1| = {e_ini:.3e} a |E_N| = {e_fin:.3e} "
                     f"(x{ratio:.1f})")
        return ("El error **crece monotonamente** a medida que iteramos: "
                + tendencia + ". Este es el comportamiento esperado por "
                "acumulacion del error de truncamiento.")
    if decrece:
        return ("El error **decrece** con las iteraciones, probablemente porque "
                "la trayectoria numerica se aproxima a la analitica en la "
                "region de evaluacion (cancelacion de errores por curvatura).")
    if oscila:
        return (f"El error **oscila** entre {float(np.min(err)):.3e} y "
                f"{e_max:.3e}. Esto sugiere que el paso h no es optimo para "
                "este problema o que la ecuacion tiene componentes con signos "
                "alternados.")
    return f"El error varia entre {float(np.min(err)):.3e} y {e_max:.3e}."


# ---------------------------------------------------------------------------
# Graficos
# ---------------------------------------------------------------------------

def _plot_trayectoria(res: ResultadoEDO, var_indep: str,
                      y_analitica: Callable | None,
                      titulo: str) -> go.Figure:
    fig = go.Figure()
    if y_analitica is not None:
        t_dense = np.linspace(res.ts[0], res.ts[-1], 400)
        try:
            y_dense = np.array([float(y_analitica(t)) for t in t_dense])
            fig.add_trace(go.Scatter(
                x=t_dense, y=y_dense, mode="lines", name="Solucion analitica",
                line=dict(color="#00d4ff", width=2, dash="dash"),
            ))
        except Exception:
            pass
    fig.add_trace(go.Scatter(
        x=res.ts, y=res.ys, mode="lines+markers", name="Aproximacion",
        line=dict(color="#ffd700", width=2),
        marker=dict(size=7),
        hovertemplate=f"{var_indep}=%{{x:.6f}}<br>y=%{{y:.6f}}<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark",
        title=titulo,
        xaxis_title=var_indep, yaxis_title="y",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def _plot_error(errores: np.ndarray, tolerancia: float | None,
                titulo: str = "Evolucion del error") -> go.Figure:
    indices = np.arange(1, len(errores) + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=indices, y=errores, mode="lines+markers", name="|y_i - y_real|",
        line=dict(color="#ff6b6b", width=2),
        marker=dict(size=6),
    ))
    if tolerancia is not None and tolerancia > 0:
        fig.add_hline(
            y=tolerancia, line_dash="dash", line_color="#ffd700",
            annotation_text=f"tolerancia = {tolerancia:.0e}",
            annotation_position="top right",
        )
    fig.update_layout(
        template="plotly_dark",
        title=titulo,
        xaxis_title="iteracion i",
        yaxis_title="|error|",
        yaxis_type="log" if np.all(errores > 0) else "linear",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Respuesta formato examen (markdown + copy-paste)
# ---------------------------------------------------------------------------

_METODO_NOMBRE = {
    "euler": "Euler explicito",
    "heun": "Euler mejorado (Heun)",
    "rk4": "Runge-Kutta 4 (RK4)",
}

_METODO_FORMULA = {
    "euler": r"y_{i+1} = y_i + h \cdot f(t_i, y_i)",
    "heun": r"y_{i+1} = y_i + \tfrac{h}{2}\,(k_1 + k_2),\quad k_1=f(t_i,y_i),\ k_2=f(t_{i+1},\,y_i+h k_1)",
    "rk4": (
        r"y_{i+1} = y_i + \tfrac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)"
    ),
}


def _respuesta_examen(
    metodo: str,
    expr_f: sp.Expr,
    t_sym: sp.Symbol,
    y_sym: sp.Symbol,
    var_indep: str,
    t0: float,
    y0: float,
    h: float,
    res: ResultadoEDO,
    y_expr: sp.Expr | None,
    y_analitica: Callable | None,
    tolerancia: float | None,
    n_dec: int,
) -> str:
    """Genera el bloque markdown con formato manuscrito del alumno."""
    bloque: list[str] = []
    nombre = _METODO_NOMBRE[metodo]
    v = var_indep

    bloque.append(f"## Ejercicio EDO — {nombre}")
    bloque.append("")

    # 1. Planteo
    bloque.append("**Planteo.**")
    bloque.append(rf"$$\frac{{dy}}{{d{v}}} = f({v},y) = {_tex(expr_f)}, "
                  rf"\quad y({_fmt_tex(t0)}) = {_fmt_tex(y0)}, "
                  rf"\quad h = {_fmt_tex(h)}$$")
    bloque.append("")

    # 2. Solucion analitica
    if y_expr is not None:
        bloque.append("**Solucion analitica** (via `dsolve`):")
        bloque.append(rf"$$y({v}) = {_tex(y_expr)}$$")
    else:
        bloque.append("**Solucion analitica:** no se obtuvo forma cerrada "
                      "(la EDO no admite resolucion simbolica inmediata).")
    bloque.append("")

    # 3. Formula del metodo
    bloque.append(f"**Formula — {nombre}:**")
    bloque.append(rf"$$ {_METODO_FORMULA[metodo]} $$")
    if metodo == "rk4":
        bloque.append(
            r"$$ k_1 = f(t_i,y_i),\quad "
            r"k_2 = f(t_i+\tfrac{h}{2},\,y_i+\tfrac{h}{2}k_1) $$"
        )
        bloque.append(
            r"$$ k_3 = f(t_i+\tfrac{h}{2},\,y_i+\tfrac{h}{2}k_2),\quad "
            r"k_4 = f(t_i+h,\,y_i+h k_3) $$"
        )
    bloque.append("")

    # 4. Tabla de iteraciones (markdown)
    bloque.append("**Tabla de iteraciones:**")
    bloque.append("")
    header = f"| i | {v}_i | y_i | f({v}_i,y_i) | y_{{i+1}}"
    sep = "|---|---|---|---|---"
    if y_analitica is not None:
        header += " | y_real | \\|error\\| "
        sep += "|---|---"
    bloque.append(header + " |")
    bloque.append(sep + "|")
    for i in range(res.n):
        t_i = _fmt_tex(res.ts[i], n_dec)
        y_i = _fmt_tex(res.ys[i], n_dec)
        f_i = _fmt_tex(res.fs[i], n_dec)
        y_next = _fmt_tex(res.ys[i + 1], n_dec)
        row = f"| {i} | {t_i} | {y_i} | {f_i} | {y_next}"
        if y_analitica is not None:
            yr = float(y_analitica(res.ts[i + 1]))
            err = abs(res.ys[i + 1] - yr)
            row += f" | {_fmt_tex(yr, n_dec)} | {_fmt_tex(err, n_dec)}"
        bloque.append(row + " |")
    bloque.append("")

    # 5. Tabla k1..k4 para RK4
    if metodo == "rk4" and res.ks is not None:
        bloque.append("**Pendientes $k_1, k_2, k_3, k_4$ por paso:**")
        bloque.append("")
        bloque.append(f"| i | {v}_i | y_i | k_1 | k_2 | k_3 | k_4 | y_{{i+1}} |")
        bloque.append("|---|---|---|---|---|---|---|---|")
        for i in range(res.n):
            bloque.append(
                f"| {i} | {_fmt_tex(res.ts[i], n_dec)} "
                f"| {_fmt_tex(res.ys[i], n_dec)} "
                f"| {_fmt_tex(res.ks[i, 0], n_dec)} "
                f"| {_fmt_tex(res.ks[i, 1], n_dec)} "
                f"| {_fmt_tex(res.ks[i, 2], n_dec)} "
                f"| {_fmt_tex(res.ks[i, 3], n_dec)} "
                f"| {_fmt_tex(res.ys[i + 1], n_dec)} |"
            )
        bloque.append("")

    # 6. Resultado final
    bloque.append(f"**Resultado final:** "
                  rf"$y({_fmt_tex(res.ts[-1])}) \approx "
                  rf"{_fmt_tex(res.ys[-1], n_dec)}$")

    if y_analitica is not None:
        y_real_fin = float(y_analitica(res.ts[-1]))
        err_fin = abs(res.ys[-1] - y_real_fin)
        bloque.append("")
        bloque.append(f"**Valor real:** $y({_fmt_tex(res.ts[-1])}) = "
                      f"{_fmt_tex(y_real_fin, n_dec)}$")
        bloque.append(f"**Error absoluto final:** ${_fmt_tex(err_fin, n_dec)}$")
        if tolerancia is not None:
            ok = "✅" if err_fin <= tolerancia else "❌"
            bloque.append(f"**Tolerancia objetivo:** "
                          f"${tolerancia:.0e}$ — {ok}")

    # 7. Comportamiento del error
    if y_analitica is not None:
        errs = np.array([
            abs(res.ys[i + 1] - float(y_analitica(res.ts[i + 1])))
            for i in range(res.n)
        ])
        bloque.append("")
        bloque.append("**Comportamiento del error:** "
                      + _analizar_comportamiento_error(errs))

    return "\n".join(bloque)


# ---------------------------------------------------------------------------
# Inputs compartidos
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InputsEDO:
    """Paquete de inputs comunes a los tres metodos."""
    latex_f: str
    var_indep: str     # "t" o "x"
    t0: float
    y0: float
    t_final: float
    h: float
    n: int
    n_dec: int
    tolerancia: float
    latex_analitica: str   # opcional — si vacio, usar dsolve


def _inputs_comunes(key_prefix: str, tol_default: float,
                    default_f_latex: str = r"y\sin(t)",
                    default_t0: str = "0",
                    default_y0: str = "1",
                    default_tfinal: str = r"\pi",
                    default_h: str = r"\pi/10") -> InputsEDO | None:
    """Renderiza inputs y devuelve InputsEDO o None si hubo error.

    Parseo inmediato de todos los campos — devolvemos None si algo no parsea.
    """
    col_var, _ = st.columns([1, 3])
    with col_var:
        var_indep = st.selectbox(
            "Variable independiente",
            options=["t", "x"],
            index=0,
            key=f"{key_prefix}_var",
            help="El profe mezcla 't' y 'x'. Elegi la que usa tu enunciado.",
        )

    latex_f = math_input(
        label=f"f({var_indep}, y) =",
        default_latex=default_f_latex,
        key=f"{key_prefix}_f",
    )

    col1, col2 = st.columns(2)
    with col1:
        t0_str = st.text_input(
            f"{var_indep}_0 (valor inicial de {var_indep})",
            value=default_t0, key=f"{key_prefix}_t0",
            help="Acepta expresiones: pi/2, sqrt(2), 0, etc.",
        )
        y0_str = st.text_input(
            "y_0 (condicion inicial y(t_0))",
            value=default_y0, key=f"{key_prefix}_y0",
        )
    with col2:
        tfinal_str = st.text_input(
            f"{var_indep}_final",
            value=default_tfinal, key=f"{key_prefix}_tfinal",
            help="Acepta expresiones: pi, pi/2, 1.5, etc.",
        )
        h_str = st.text_input(
            "Paso h",
            value=default_h, key=f"{key_prefix}_h",
            help="Acepta expresiones: 0.1, pi/10, pi/8, etc.",
        )

    col3, col4 = st.columns(2)
    with col3:
        n_dec = st.slider(
            "Decimales (display)", min_value=1, max_value=10, value=6,
            key=f"{key_prefix}_ndec",
        )
    with col4:
        tol_exp = st.slider(
            "Tolerancia objetivo 10^(-k)", min_value=1, max_value=10,
            value=int(-np.log10(tol_default)),
            key=f"{key_prefix}_tol",
            help="Precision esperada para este metodo. Euler: 10^-1. RK4: 10^-6.",
        )
        tolerancia = 10 ** (-tol_exp)
        st.latex(rf"\text{{Tolerancia}} = 10^{{-{tol_exp}}}")

    with st.expander("Solucion analitica manual (opcional)", expanded=False):
        st.caption("Dejar vacio para resolver automaticamente con SymPy dsolve.")
        latex_analitica = math_input(
            label=f"y({var_indep}) = (opcional)",
            default_latex="",
            key=f"{key_prefix}_yan",
        )

    # Parse
    t0 = parse_expr_to_float(t0_str, "t_0")
    y0 = parse_expr_to_float(y0_str, "y_0")
    t_final = parse_expr_to_float(tfinal_str, "t_final")
    h = parse_expr_to_float(h_str, "h")
    if t0 is None or y0 is None or t_final is None or h is None:
        return None
    if h <= 0:
        st.error("El paso h debe ser positivo.")
        return None
    if t_final <= t0:
        st.error("t_final debe ser mayor que t_0.")
        return None
    n = int(round((t_final - t0) / h))
    if n < 1:
        st.error("n debe ser >= 1. Revisa h y el intervalo.")
        return None
    # Ajustar h si no divide exacto
    h_real = (t_final - t0) / n
    if abs(h_real - h) > 1e-9:
        st.caption(f"_h ajustado exacto: {h_real:.10f} (n = {n} pasos)._")
        h = h_real

    return InputsEDO(
        latex_f=latex_f, var_indep=var_indep, t0=t0, y0=y0,
        t_final=t_final, h=h, n=n, n_dec=n_dec, tolerancia=tolerancia,
        latex_analitica=latex_analitica,
    )


# ---------------------------------------------------------------------------
# Corrida compartida: parsea, resuelve, renderiza tabla/grafico/examen
# ---------------------------------------------------------------------------

_METODO_FUNC = {
    "euler": _euler,
    "heun": _heun,
    "rk4": _rk4,
}


def _resaltar_filas_tabla(tolerancia: float | None,
                          col_error: str | None):
    """Factory: resalta i=0 e i=1 (1ra y 2da iter) y filas con error > tol."""

    def _resaltar(fila: pd.Series) -> list[str]:
        estilos = [""] * len(fila)
        # 1ra y 2da iteracion (i=0 y i=1)
        try:
            i_val = int(fila["i"])
            if i_val in (0, 1):
                estilos = ["background-color: #0b3b5c; color: #ffffff"] * len(fila)
        except (KeyError, TypeError, ValueError):
            pass
        # Error excede tolerancia
        if col_error and tolerancia is not None and col_error in fila.index:
            try:
                e = float(fila[col_error])
                if np.isfinite(e) and e > tolerancia:
                    estilos = ["background-color: #5c1a1a; color: #ffffff"] * len(fila)
            except (TypeError, ValueError):
                pass
        return estilos

    return _resaltar


def _correr_metodo(metodo: str, inputs: InputsEDO) -> None:
    """Parsea, simula, renderiza todo para un metodo dado."""
    t_sym = sp.Symbol(inputs.var_indep)
    y_sym = sp.Symbol("y")
    expr_f, f_np = parse_latex(inputs.latex_f, [t_sym, y_sym])
    if expr_f is None or f_np is None:
        return

    # Analitica: primero la que dio el alumno, si no dsolve automatico
    y_expr: sp.Expr | None = None
    y_analitica: Callable | None = None
    if inputs.latex_analitica.strip():
        y_expr, y_callable = parse_latex(inputs.latex_analitica, [t_sym])
        if y_callable is not None:
            y_analitica = y_callable
    if y_analitica is None:
        y_expr, y_analitica = _resolver_analitica(
            expr_f, t_sym, y_sym, inputs.t0, inputs.y0
        )

    # Corrida numerica
    def _f(t: float, y: float) -> float:
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            val = float(f_np(t, y))
        if not np.isfinite(val):
            raise ValueError(
                f"f({inputs.var_indep}={t:.6f}, y={y:.6f}) no es finito "
                "(division por 0, raiz de negativo, etc.)."
            )
        return val

    try:
        res = _METODO_FUNC[metodo](_f, inputs.t0, inputs.y0, inputs.h, inputs.n)
    except Exception as e:
        st.error(f"Error al simular: {e}")
        return

    # Header con resultado
    st.markdown(f"#### Resultado: $y({_fmt_tex(res.ts[-1])}) "
                rf"\approx {_fmt_tex(res.ys[-1], inputs.n_dec)}$")

    if y_expr is not None:
        st.info(f"Solucion analitica: $y({inputs.var_indep}) = {_tex(y_expr)}$")
    elif not inputs.latex_analitica.strip():
        st.warning("No se pudo obtener solucion analitica cerrada con `dsolve`. "
                   "Se omiten las columnas de error vs analitica.")

    # Metricas
    if y_analitica is not None:
        try:
            y_real_fin = float(y_analitica(res.ts[-1]))
            err_fin = abs(res.ys[-1] - y_real_fin)
            c1, c2, c3 = st.columns(3)
            c1.metric(f"y({_fmt_tex(res.ts[-1])}) aprox",
                      fmt_decimal(res.ys[-1], inputs.n_dec))
            c2.metric(f"y({_fmt_tex(res.ts[-1])}) real",
                      fmt_decimal(y_real_fin, inputs.n_dec))
            ok = "✓" if err_fin <= inputs.tolerancia else "✗"
            c3.metric(
                "Error absoluto final",
                fmt_decimal(err_fin, inputs.n_dec),
                delta=f"tol {inputs.tolerancia:.0e} {ok}",
                delta_color="off",
            )
        except Exception:
            pass

    # Tabla de iteraciones
    df_iter = _tabla_iteraciones(res, inputs.var_indep, y_analitica)
    st.markdown("#### Tabla de iteraciones")
    col_error = "|error|" if "|error|" in df_iter.columns else None
    resaltar = _resaltar_filas_tabla(inputs.tolerancia, col_error)
    df_display = df_iter.copy()
    for c in df_display.columns:
        if c == "i":
            continue
        df_display[c] = df_display[c].map(
            lambda v: fmt_decimal(v, inputs.n_dec)
        )
    st.dataframe(
        df_display.style.apply(resaltar, axis=1),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        "Filas azules: 1ra y 2da iteracion (para comprobar con datos de salida). "
        + ("Filas rojas: |error| > tolerancia." if col_error else "")
    )

    # Tabla predictor/corrector para Heun
    if metodo == "heun" and res.y_pred is not None:
        st.markdown("#### Predictor y corrector por paso")
        df_pc = _tabla_predictor_corrector_heun(res, inputs.var_indep)
        df_pc_d = df_pc.copy()
        for c in df_pc_d.columns:
            if c == "i":
                continue
            df_pc_d[c] = df_pc_d[c].map(
                lambda v: fmt_decimal(v, inputs.n_dec)
            )
        st.dataframe(df_pc_d, use_container_width=True, hide_index=True)
        with st.expander("📘 Detalle paso a paso (predictor + corrector)",
                         expanded=False):
            _paso_heun_detallado(res, inputs.var_indep, inputs.h, inputs.n_dec)

    # Tabla k1..k4 para RK4
    if metodo == "rk4" and res.ks is not None:
        st.markdown("#### Pendientes RK4 por paso")
        df_ks = _tabla_ks_rk4(res, inputs.var_indep)
        df_ks_d = df_ks.copy()
        for c in df_ks_d.columns:
            if c == "i":
                continue
            df_ks_d[c] = df_ks_d[c].map(
                lambda v: fmt_decimal(v, inputs.n_dec)
            )
        st.dataframe(df_ks_d, use_container_width=True, hide_index=True)
        with st.expander("📘 Detalle paso a paso (k₁, k₂, k₃, k₄)",
                         expanded=False):
            _paso_rk4_detallado(res, inputs.var_indep, inputs.h, inputs.n_dec)

    # Grafico de trayectoria
    st.plotly_chart(
        _plot_trayectoria(
            res, inputs.var_indep, y_analitica,
            titulo=f"{_METODO_NOMBRE[metodo]} — y({_fmt_tex(res.ts[-1])}) "
                   f"≈ {res.ys[-1]:.{inputs.n_dec}f}",
        ),
        use_container_width=True,
    )

    # Grafico + analisis del error
    if y_analitica is not None:
        errs = np.array([
            abs(res.ys[i + 1] - float(y_analitica(res.ts[i + 1])))
            for i in range(res.n)
        ])
        st.plotly_chart(
            _plot_error(errs, inputs.tolerancia), use_container_width=True,
        )
        st.markdown(f"**Comportamiento del error:** "
                    f"{_analizar_comportamiento_error(errs)}")

    # Respuesta lista para examen
    with st.expander("📝 Respuesta lista para examen (formato alumno)",
                     expanded=False):
        texto = _respuesta_examen(
            metodo, expr_f, t_sym, y_sym, inputs.var_indep,
            inputs.t0, inputs.y0, inputs.h, res, y_expr, y_analitica,
            inputs.tolerancia, inputs.n_dec,
        )
        st.markdown(texto)
        st.code(texto, language="markdown")


# ---------------------------------------------------------------------------
# Submodulos por metodo
# ---------------------------------------------------------------------------

_PRESETS_EULER = {
    "Parcial D-5a — dy/dx = cos(x)+x": {
        "f": r"\cos(x)+x", "var": "x",
        "t0": "0", "y0": "1", "tf": r"\pi/2", "h": r"\pi/8",
    },
    "Parcial A-5a — dy/dx = y·sin(t)": {
        "f": r"y\sin(t)", "var": "t",
        "t0": "0", "y0": "1", "tf": r"\pi", "h": r"\pi/10",
    },
}

_PRESETS_HEUN = {
    "Parcial B-5a — dy/dx = (x^3-1)/y^2": {
        "f": r"\frac{x^{3}-1}{y^{2}}", "var": "x",
        "t0": "1", "y0": "3", "tf": "1.5", "h": "0.1",
    },
}

_PRESETS_RK4 = {
    "Parcial B-5b — dy/dt = 2t√y (cohete)": {
        "f": r"2t\sqrt{y}", "var": "t",
        "t0": "1", "y0": "3", "tf": "1.2", "h": "0.1",
    },
    "Parcial D-5b — dy/dx = cos(x)+x": {
        "f": r"\cos(x)+x", "var": "x",
        "t0": "0", "y0": "1", "tf": r"\pi/2", "h": r"\pi/8",
    },
    "Parcial A-5b — dy/dx = y·sin(t)": {
        "f": r"y\sin(t)", "var": "t",
        "t0": "0", "y0": "1", "tf": r"\pi", "h": r"\pi/10",
    },
}


def _selector_preset(presets: dict[str, dict], key: str) -> dict | None:
    with st.expander("Presets de parciales Caceres"):
        opciones = ["(manual)"] + list(presets.keys())
        sel = st.selectbox("Cargar preset", opciones, key=f"{key}_preset")
        if sel != "(manual)":
            st.caption("Preset cargado. Ajusta manualmente si hace falta.")
            return presets[sel]
    return None


def _metodo_euler() -> None:
    st.subheader("Euler explicito")
    with st.expander("Teoria del metodo"):
        st.latex(r"y_{i+1} = y_i + h \cdot f(t_i, y_i)")
        st.markdown(
            "- **Orden global:** $O(h)$ (el mas simple de los metodos).\n"
            "- **Costo:** 1 evaluacion de $f$ por paso.\n"
            "- **Interpretacion:** extrapolar linealmente con la pendiente en "
            "$(t_i, y_i)$ durante un paso $h$.\n"
            "- **Tolerancia tipica:** $10^{-1}$ (es impreciso salvo $h$ muy chico)."
        )

    preset = _selector_preset(_PRESETS_EULER, "euler")
    if preset:
        inputs = _inputs_comunes(
            "euler", tol_default=1e-1,
            default_f_latex=preset["f"],
            default_t0=preset["t0"], default_y0=preset["y0"],
            default_tfinal=preset["tf"], default_h=preset["h"],
        )
    else:
        inputs = _inputs_comunes("euler", tol_default=1e-1)
    if inputs is None:
        return
    if st.button("Calcular", key="euler_calc"):
        _correr_metodo("euler", inputs)


def _metodo_heun() -> None:
    st.subheader("Euler mejorado (Heun)")
    with st.expander("Teoria del metodo"):
        st.latex(
            r"k_1 = f(t_i, y_i),\quad "
            r"y^{*}_{i+1} = y_i + h\,k_1"
        )
        st.latex(
            r"k_2 = f(t_{i+1}, y^{*}_{i+1}),\quad "
            r"y_{i+1} = y_i + \tfrac{h}{2}(k_1 + k_2)"
        )
        st.markdown(
            "- **Orden global:** $O(h^2)$.\n"
            "- **Costo:** 2 evaluaciones de $f$ por paso.\n"
            "- **Interpretacion:** predictor-corrector — primero estima con "
            "Euler, despues corrige con la regla del trapecio.\n"
            "- **Tolerancia tipica:** $10^{-2}$ a $10^{-4}$."
        )

    preset = _selector_preset(_PRESETS_HEUN, "heun")
    if preset:
        inputs = _inputs_comunes(
            "heun", tol_default=1e-4,
            default_f_latex=preset["f"],
            default_t0=preset["t0"], default_y0=preset["y0"],
            default_tfinal=preset["tf"], default_h=preset["h"],
        )
    else:
        inputs = _inputs_comunes(
            "heun", tol_default=1e-4,
            default_f_latex=r"\frac{x^{3}-1}{y^{2}}",
            default_t0="1", default_y0="3",
            default_tfinal="1.5", default_h="0.1",
        )
    if inputs is None:
        return
    if st.button("Calcular", key="heun_calc"):
        _correr_metodo("heun", inputs)


def _metodo_rk4() -> None:
    st.subheader("Runge-Kutta 4 (RK4)")
    with st.expander("Teoria del metodo"):
        st.latex(r"k_1 = f(t_i, y_i)")
        st.latex(r"k_2 = f\!\left(t_i + \tfrac{h}{2},\, y_i + \tfrac{h}{2}k_1\right)")
        st.latex(r"k_3 = f\!\left(t_i + \tfrac{h}{2},\, y_i + \tfrac{h}{2}k_2\right)")
        st.latex(r"k_4 = f(t_i + h,\, y_i + h\,k_3)")
        st.latex(r"y_{i+1} = y_i + \tfrac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)")
        st.markdown(
            "- **Orden global:** $O(h^4)$ — el metodo por defecto para EDOs.\n"
            "- **Costo:** 4 evaluaciones de $f$ por paso.\n"
            "- **Interpretacion:** promedio ponderado de 4 pendientes "
            "(2 en los extremos, 2 en el medio).\n"
            "- **Tolerancia tipica:** $10^{-6}$ a $10^{-8}$."
        )

    preset = _selector_preset(_PRESETS_RK4, "rk4")
    if preset:
        inputs = _inputs_comunes(
            "rk4", tol_default=1e-6,
            default_f_latex=preset["f"],
            default_t0=preset["t0"], default_y0=preset["y0"],
            default_tfinal=preset["tf"], default_h=preset["h"],
        )
    else:
        inputs = _inputs_comunes(
            "rk4", tol_default=1e-6,
            default_f_latex=r"2t\sqrt{y}",
            default_t0="1", default_y0="3",
            default_tfinal="1.2", default_h="0.1",
        )
    if inputs is None:
        return
    if st.button("Calcular", key="rk4_calc"):
        _correr_metodo("rk4", inputs)


_PRESETS_COMPLETOS = {
    "Apunte del profe — dy/dt = y·sin(t), y(0)=2, [0,π]": {
        "f": r"y\sin(t)", "var": "t", "t0": "0", "y0": "2",
        "tf": r"\pi", "h": r"\pi/10",
        "yan": r"2e^{1-\cos(t)}",
    },
    "Parcial A-5 — dy/dx = y·sin(t), y(0)=1, [0,π]": {
        "f": r"y\sin(t)", "var": "t", "t0": "0", "y0": "1",
        "tf": r"\pi", "h": r"\pi/10", "yan": r"e^{1-\cos(t)}",
    },
    "Parcial D-5 — dy/dx = cos(x)+x, y(0)=1, [0,π/2]": {
        "f": r"\cos(x)+x", "var": "x", "t0": "0", "y0": "1",
        "tf": r"\pi/2", "h": r"\pi/8",
        "yan": r"\sin(x)+\frac{x^{2}}{2}+1",
    },
    "Parcial B-5a — dy/dx = (x³-1)/y², y(1)=3, y(1.5)": {
        "f": r"\frac{x^{3}-1}{y^{2}}", "var": "x", "t0": "1", "y0": "3",
        "tf": "1.5", "h": "0.1", "yan": "",
    },
    "Parcial B-5b — dy/dt = 2t·√y (cohete), y(1)=3, y(1.2)": {
        "f": r"2t\sqrt{y}", "var": "t", "t0": "1", "y0": "3",
        "tf": "1.2", "h": "0.1", "yan": "",
    },
}


def _metodo_comparacion() -> None:
    st.subheader("Simulacion completa — estilo profe Caceres")
    st.markdown(
        "**Flujo del profe:** "
        "(1) resolver analiticamente, "
        "(2) aplicar los 3 metodos numericos, "
        "(3) comparar numerica y graficamente."
    )

    preset = _selector_preset(_PRESETS_COMPLETOS, "cmp")
    if preset:
        latex_f_def = preset["f"]
        t0_def, y0_def = preset["t0"], preset["y0"]
        tf_def, h_def = preset["tf"], preset["h"]
    else:
        latex_f_def = r"y\sin(t)"
        t0_def, y0_def = "0", "2"
        tf_def, h_def = r"\pi", r"\pi/10"

    inputs = _inputs_comunes(
        "cmp", tol_default=1e-6,
        default_f_latex=latex_f_def,
        default_t0=t0_def, default_y0=y0_def,
        default_tfinal=tf_def, default_h=h_def,
    )
    if inputs is None:
        return

    # Si el preset trae analitica sugerida, la mostramos como hint
    if preset and preset.get("yan"):
        st.caption(
            f"_Solucion analitica del preset: `{preset['yan']}` — "
            "copiala al campo 'Solucion analitica manual' si el auto-`dsolve` falla._"
        )

    if not st.button("Calcular los 3", key="cmp_calc"):
        return

    t_sym = sp.Symbol(inputs.var_indep)
    y_sym = sp.Symbol("y")
    expr_f, f_np = parse_latex(inputs.latex_f, [t_sym, y_sym])
    if expr_f is None or f_np is None:
        return

    # ------------------------------------------------------------------ 1. Analitica
    st.markdown("### 1. Solucion analitica")
    y_expr: sp.Expr | None = None
    y_analitica: Callable | None = None
    if inputs.latex_analitica.strip():
        y_expr, y_callable = parse_latex(inputs.latex_analitica, [t_sym])
        if y_callable is not None:
            y_analitica = y_callable
            st.success("Solucion analitica ingresada manualmente:")
    if y_analitica is None:
        y_expr, y_analitica = _resolver_analitica(
            expr_f, t_sym, y_sym, inputs.t0, inputs.y0
        )
        if y_expr is not None:
            st.success("Solucion analitica obtenida con `dsolve`:")

    if y_expr is not None:
        st.latex(rf"y({inputs.var_indep}) = {_tex(y_expr)}")
    else:
        st.warning(
            "No se encontro solucion analitica cerrada. "
            "Los metodos numericos corren igual, pero sin columna Exacta ni error."
        )

    # ------------------------------------------------------------------ 2. Numerico
    st.markdown("### 2. Metodos numericos")

    def _f(t: float, y: float) -> float:
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            return float(f_np(t, y))

    try:
        euler_res = _euler(_f, inputs.t0, inputs.y0, inputs.h, inputs.n)
        heun_res = _heun(_f, inputs.t0, inputs.y0, inputs.h, inputs.n)
        rk4_res = _rk4(_f, inputs.t0, inputs.y0, inputs.h, inputs.n)
    except Exception as e:
        st.error(f"Error al simular: {e}")
        return

    # Tabla maestra estilo profe: n | t | Euler | Heun | RK4 | Exacta
    st.markdown("#### Tabla comparativa (estilo manuscrito del profe)")
    df_master = _tabla_maestra(
        euler_res, heun_res, rk4_res, inputs.var_indep, y_analitica,
    )
    df_master_d = df_master.copy()
    for c in df_master_d.columns:
        if c == "n":
            continue
        df_master_d[c] = df_master_d[c].map(
            lambda v: fmt_decimal(v, inputs.n_dec)
        )
    # Resalta 1ra y 2da iteracion (n=1 y n=2) como pide el profe
    def _resaltar_primeras(fila: pd.Series) -> list[str]:
        try:
            n = int(fila["n"])
            if n in (1, 2):
                return ["background-color: #0b3b5c; color: #ffffff"] * len(fila)
        except (KeyError, TypeError, ValueError):
            pass
        return [""] * len(fila)

    st.dataframe(
        df_master_d.style.apply(_resaltar_primeras, axis=1),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        "Filas azules: 1ra y 2da iteracion (n=1, n=2) — "
        "el profe pide explicitamente comprobarlas con los datos de salida."
    )

    # Metricas finales
    if y_analitica is not None:
        try:
            y_real_fin = float(y_analitica(euler_res.ts[-1]))
            err_eul = abs(euler_res.ys[-1] - y_real_fin)
            err_heu = abs(heun_res.ys[-1] - y_real_fin)
            err_rk4 = abs(rk4_res.ys[-1] - y_real_fin)
            st.markdown(f"#### Resultado final en {inputs.var_indep} = "
                        f"{_fmt_tex(inputs.t_final)}")
            c0, c1, c2, c3 = st.columns(4)
            c0.metric("Exacta", fmt_decimal(y_real_fin, inputs.n_dec))
            c1.metric("Euler", fmt_decimal(euler_res.ys[-1], inputs.n_dec),
                      delta=f"err {err_eul:.2e}", delta_color="off")
            c2.metric("Heun", fmt_decimal(heun_res.ys[-1], inputs.n_dec),
                      delta=f"err {err_heu:.2e}", delta_color="off")
            c3.metric("RK4", fmt_decimal(rk4_res.ys[-1], inputs.n_dec),
                      delta=f"err {err_rk4:.2e}", delta_color="off")
        except Exception:
            pass

    # ------------------------------------------------------------------ 3. Detalle
    st.markdown("### 3. Detalle paso a paso")
    col_e, col_h, col_r = st.columns(3)
    with col_e:
        st.caption("Euler")
        st.latex(r"y_{i+1} = y_i + h\,f(t_i, y_i)")
    with col_h:
        st.caption("Heun (predictor + corrector)")
        st.latex(
            r"y^{*}_{i+1} = y_i + h\,f(t_i, y_i)"
        )
        st.latex(
            r"y_{i+1} = y_i + \tfrac{h}{2}\!\left[f(t_i,y_i) + f(t_{i+1}, y^{*}_{i+1})\right]"
        )
    with col_r:
        st.caption("RK4")
        st.latex(r"y_{i+1} = y_i + \tfrac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)")

    with st.expander("📘 Heun — predictor y corrector por paso",
                     expanded=False):
        _paso_heun_detallado(heun_res, inputs.var_indep,
                             inputs.h, inputs.n_dec, max_pasos=10)
    with st.expander("📘 RK4 — k₁, k₂, k₃, k₄ por paso",
                     expanded=False):
        _paso_rk4_detallado(rk4_res, inputs.var_indep,
                            inputs.h, inputs.n_dec, max_pasos=10)

    # ------------------------------------------------------------------ 4. Grafico
    st.markdown("### 4. Comparacion grafica")
    fig = go.Figure()
    if y_analitica is not None:
        t_dense = np.linspace(inputs.t0, inputs.t_final, 400)
        try:
            y_dense = np.array([float(y_analitica(t)) for t in t_dense])
            fig.add_trace(go.Scatter(
                x=t_dense, y=y_dense, mode="lines", name="Solucion exacta",
                line=dict(color="#00d4ff", width=3),
            ))
        except Exception:
            pass
    fig.add_trace(go.Scatter(
        x=euler_res.ts, y=euler_res.ys, mode="lines+markers", name="Euler",
        line=dict(color="#ff6b6b", width=2, dash="dot"),
        marker=dict(size=7, symbol="circle"),
    ))
    fig.add_trace(go.Scatter(
        x=heun_res.ts, y=heun_res.ys, mode="lines+markers",
        name="Heun (Euler Mejorado)",
        line=dict(color="#ffd700", width=2, dash="dash"),
        marker=dict(size=7, symbol="square"),
    ))
    fig.add_trace(go.Scatter(
        x=rk4_res.ts, y=rk4_res.ys, mode="lines+markers", name="RK4",
        line=dict(color="#2ecc71", width=2),
        marker=dict(size=7, symbol="diamond"),
    ))
    fig.update_layout(
        template="plotly_dark",
        title="Trayectorias — exacta (solida) · Euler (puntos) · Heun (guiones) · RK4 (solido)",
        xaxis_title=inputs.var_indep, yaxis_title="y",
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0.4)"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Grafico de errores
    if y_analitica is not None:
        fig_err = go.Figure()
        colores = {"Euler": "#ff6b6b", "Heun": "#ffd700", "RK4": "#2ecc71"}
        for nombre, res in (("Euler", euler_res), ("Heun", heun_res),
                            ("RK4", rk4_res)):
            errs = np.array([
                abs(res.ys[i + 1] - float(y_analitica(res.ts[i + 1])))
                for i in range(res.n)
            ])
            # Reemplazar 0 exactos por un piso visible en log
            pisos = np.where(errs > 0, errs, 1e-16)
            fig_err.add_trace(go.Scatter(
                x=np.arange(1, len(errs) + 1), y=pisos,
                mode="lines+markers", name=nombre,
                line=dict(color=colores[nombre], width=2),
                marker=dict(size=6),
            ))
        fig_err.add_hline(
            y=inputs.tolerancia, line_dash="dash", line_color="#ffffff",
            annotation_text=f"tolerancia = {inputs.tolerancia:.0e}",
            annotation_position="top right",
        )
        fig_err.update_layout(
            template="plotly_dark",
            title="|error| por iteracion (log)",
            xaxis_title="iteracion i", yaxis_title="|error|",
            yaxis_type="log",
            margin=dict(l=40, r=20, t=50, b=40),
        )
        st.plotly_chart(fig_err, use_container_width=True)

    # ------------------------------------------------------------------ 5. Examen
    with st.expander("📝 Respuesta lista para examen (formato alumno — los 3 metodos)",
                     expanded=False):
        texto = _respuesta_examen_combinada(
            expr_f, t_sym, y_sym, inputs.var_indep,
            inputs.t0, inputs.y0, inputs.h,
            euler_res, heun_res, rk4_res,
            y_expr, y_analitica, inputs.n_dec,
        )
        st.markdown(texto)
        st.code(texto, language="markdown")


def _respuesta_examen_combinada(
    expr_f: sp.Expr,
    t_sym: sp.Symbol,
    y_sym: sp.Symbol,
    var_indep: str,
    t0: float,
    y0: float,
    h: float,
    euler_res: ResultadoEDO,
    heun_res: ResultadoEDO,
    rk4_res: ResultadoEDO,
    y_expr: sp.Expr | None,
    y_analitica: Callable | None,
    n_dec: int,
) -> str:
    """Bloque unificado con los 3 metodos + analitica (estilo profe)."""
    bloque: list[str] = []
    v = var_indep
    bloque.append("## Ejercicio EDO — Euler / Heun / RK4")
    bloque.append("")
    bloque.append("**Planteo.**")
    bloque.append(
        rf"$$\frac{{dy}}{{d{v}}} = f({v},y) = {_tex(expr_f)}, "
        rf"\quad y({_fmt_tex(t0)}) = {_fmt_tex(y0)}, "
        rf"\quad h = {_fmt_tex(h)}$$"
    )
    bloque.append("")

    # 1. Analitica
    bloque.append("### 1. Solucion analitica")
    if y_expr is not None:
        bloque.append(rf"$$y({v}) = {_tex(y_expr)}$$")
    else:
        bloque.append("_No se obtuvo solucion analitica cerrada._")
    bloque.append("")

    # 2. Formulas
    bloque.append("### 2. Metodos aplicados")
    bloque.append("")
    bloque.append("**Euler:**")
    bloque.append(r"$$y_{i+1} = y_i + h \cdot f(t_i, y_i)$$")
    bloque.append("")
    bloque.append("**Heun (Euler Mejorado):**")
    bloque.append(r"$$y^{*}_{i+1} = y_i + h\,f(t_i, y_i)"
                  r"\quad\text{(predictor)}$$")
    bloque.append(
        r"$$y_{i+1} = y_i + \tfrac{h}{2}\left[f(t_i,y_i) "
        r"+ f(t_{i+1}, y^{*}_{i+1})\right]\quad\text{(corrector)}$$"
    )
    bloque.append("")
    bloque.append("**Runge-Kutta 4 (RK4):**")
    bloque.append(r"$$k_1 = f(t_i,y_i)$$")
    bloque.append(r"$$k_2 = f(t_i+\tfrac{h}{2},\,y_i+\tfrac{h}{2}k_1)$$")
    bloque.append(r"$$k_3 = f(t_i+\tfrac{h}{2},\,y_i+\tfrac{h}{2}k_2)$$")
    bloque.append(r"$$k_4 = f(t_i+h,\,y_i+h\,k_3)$$")
    bloque.append(r"$$y_{i+1} = y_i + \tfrac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$")
    bloque.append("")

    # 3. Tabla maestra
    bloque.append("### 3. Tabla comparativa")
    bloque.append("")
    header = f"| n | {v} | Euler | Heun | RK4"
    sep = "|---|---|---|---|---"
    if y_analitica is not None:
        header += " | Exacta | err Euler | err Heun | err RK4 "
        sep += "|---|---|---|---"
    bloque.append(header + " |")
    bloque.append(sep + "|")
    for i in range(euler_res.n + 1):
        t = euler_res.ts[i]
        row = (
            f"| {i} | {_fmt_tex(t, n_dec)} "
            f"| {_fmt_tex(euler_res.ys[i], n_dec)} "
            f"| {_fmt_tex(heun_res.ys[i], n_dec)} "
            f"| {_fmt_tex(rk4_res.ys[i], n_dec)}"
        )
        if y_analitica is not None:
            try:
                yr = float(y_analitica(t))
                row += (
                    f" | {_fmt_tex(yr, n_dec)} "
                    f"| {_fmt_tex(abs(euler_res.ys[i]-yr), n_dec)} "
                    f"| {_fmt_tex(abs(heun_res.ys[i]-yr), n_dec)} "
                    f"| {_fmt_tex(abs(rk4_res.ys[i]-yr), n_dec)}"
                )
            except Exception:
                row += " | — | — | — | —"
        bloque.append(row + " |")
    bloque.append("")

    # 4. Resultado final
    bloque.append("### 4. Resultado final")
    bloque.append("")
    t_fin = euler_res.ts[-1]
    bloque.append(f"- **Euler:** "
                  rf"$y({_fmt_tex(t_fin)}) \approx "
                  rf"{_fmt_tex(euler_res.ys[-1], n_dec)}$")
    bloque.append(f"- **Heun:** "
                  rf"$y({_fmt_tex(t_fin)}) \approx "
                  rf"{_fmt_tex(heun_res.ys[-1], n_dec)}$")
    bloque.append(f"- **RK4:** "
                  rf"$y({_fmt_tex(t_fin)}) \approx "
                  rf"{_fmt_tex(rk4_res.ys[-1], n_dec)}$")
    if y_analitica is not None:
        y_real_fin = float(y_analitica(t_fin))
        bloque.append(f"- **Exacta:** "
                      rf"$y({_fmt_tex(t_fin)}) = "
                      rf"{_fmt_tex(y_real_fin, n_dec)}$")
    bloque.append("")

    # 5. Conclusion
    bloque.append("### 5. Conclusion")
    if y_analitica is not None:
        y_real_fin = float(y_analitica(t_fin))
        errs = {
            "Euler": abs(euler_res.ys[-1] - y_real_fin),
            "Heun": abs(heun_res.ys[-1] - y_real_fin),
            "RK4": abs(rk4_res.ys[-1] - y_real_fin),
        }
        mejor = min(errs, key=errs.get)
        bloque.append(
            f"De los 3 metodos, **{mejor}** ofrece la mejor aproximacion "
            f"(|error| = {errs[mejor]:.2e}), consistente con su orden "
            "de convergencia teorico mas alto. Euler acumula error lineal "
            "con $i$, Heun cuadratico, y RK4 cuartico — por eso RK4 "
            "suele dominar salvo $h$ muy grande."
        )
    else:
        bloque.append(
            "Sin solucion analitica no se puede cuantificar el error. "
            "Como convergencia empirica, comparar los valores finales: "
            "si Euler, Heun y RK4 convergen al mismo valor, el metodo "
            "numerico esta bien calibrado."
        )

    return "\n".join(bloque)


# ---------------------------------------------------------------------------
# Render principal
# ---------------------------------------------------------------------------

def render() -> None:
    st.header("Ecuaciones Diferenciales Ordinarias (EDO)")
    st.markdown(
        "Problema de Cauchy: "
        r"$\dfrac{dy}{dt} = f(t, y), \quad y(t_0) = y_0$."
    )

    render_teoria("edo_teoria", expanded=False)

    tabs = st.tabs([
        "🎯 Comparacion (estilo profe)",
        "Euler explicito",
        "Euler mejorado (Heun)",
        "Runge-Kutta 4",
    ])
    with tabs[0]:
        _metodo_comparacion()
    with tabs[1]:
        _metodo_euler()
    with tabs[2]:
        _metodo_heun()
    with tabs[3]:
        _metodo_rk4()
