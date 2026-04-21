"""Utilidad compartida del Teorema de Bolzano.

Extraido de modules/biseccion.py para que cualquier modulo (Biseccion,
Newton-Raphson, Punto Fijo, Steffensen, Comparacion) pueda mostrarlo
como justificacion de existencia de raiz antes de aplicar el metodo.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import sympy as sp

from utils.ui.tablas import fmt_decimal


def plot_bolzano(f_np, a: float, b: float, raiz: float | None) -> go.Figure:
    """Visual didactico: f continua en [a,b] + cambio de signo => raiz."""
    from utils.graficos import apply_geogebra_style

    margen = (b - a) * 0.15 if b > a else 1.0
    xs = np.linspace(a - margen, b + margen, 600)
    try:
        ys = np.asarray(f_np(xs), dtype=float)
    except Exception:
        ys = np.array([float(f_np(xi)) for xi in xs])

    fa = float(f_np(a))
    fb = float(f_np(b))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="f(x)",
                              line=dict(color="#1f77b4", width=2.5)))
    fig.add_hline(y=0, line=dict(color="#444", dash="dot", width=1))
    fig.add_vrect(x0=a, x1=b, fillcolor="#ffe08a", opacity=0.25,
                   line_width=0, annotation_text="[a, b]",
                   annotation_position="top left")
    fig.add_trace(go.Scatter(
        x=[a, b], y=[fa, fb], mode="markers+text",
        marker=dict(size=12, color=["#d62728", "#2ca02c"]),
        text=[f"f(a)={fmt_decimal(fa)}", f"f(b)={fmt_decimal(fb)}"],
        textposition=["bottom center" if fa < 0 else "top center",
                       "bottom center" if fb < 0 else "top center"],
        name="Extremos",
    ))
    fig.add_vline(x=a, line=dict(color="#d62728", dash="dash", width=1))
    fig.add_vline(x=b, line=dict(color="#2ca02c", dash="dash", width=1))
    if raiz is not None:
        fig.add_trace(go.Scatter(
            x=[raiz], y=[0], mode="markers+text",
            marker=dict(size=14, color="#7a0b0b", symbol="star"),
            text=[f"c ≈ {fmt_decimal(raiz)}"], textposition="top center",
            name="Raiz garantizada",
        ))
    fig.update_layout(
        title=dict(text="Bolzano: f continua en [a,b] y f(a)·f(b) < 0 ⇒ ∃ c ∈ (a,b) : f(c) = 0",
                    x=0.5),
        xaxis_title="x", yaxis_title="f(x)", height=480,
    )
    return apply_geogebra_style(fig)


def render_bolzano(f_np, f_expr, a: float, b: float,
                    raiz: float | None = None,
                    mostrar_demostracion: bool = True,
                    titulo: str = "### Teorema de Bolzano (Valor Intermedio — caso f(c)=0)") -> dict:
    """Bloque completo de Bolzano: enunciado + demostracion + verificacion + visual.

    Args:
        f_np: f(x) numerica.
        f_expr: f(x) simbolica (para mostrar LaTeX).
        a, b: extremos del intervalo.
        raiz: raiz aproximada para marcar en el visual (opcional).
        mostrar_demostracion: si False, omite la demostracion formal.
        titulo: encabezado del bloque.

    Returns:
        dict con fa, fb, fa_por_fb, aplica (bool).
    """
    try:
        fa, fb = float(f_np(a)), float(f_np(b))
    except Exception as e:
        st.error(f"No se pudo evaluar f en los extremos: {e}")
        return {"fa": None, "fb": None, "fa_por_fb": None, "aplica": False}

    aplica = fa * fb < 0
    st.markdown(titulo)
    st.markdown("**Enunciado.**")
    st.latex(
        r"\text{Si } f:[a,b]\to\mathbb{R} \text{ es continua en } [a,b] "
        r"\text{ y } f(a)\cdot f(b) < 0, "
        r"\text{ entonces } \exists\, c \in (a,b) \text{ tal que } f(c) = 0."
    )

    st.markdown("**Hipotesis a verificar.**")
    st.markdown(
        "1. **Continuidad** de $f$ en $[a, b]$.\n"
        "2. **Cambio de signo**: $f(a) \\cdot f(b) < 0$."
    )

    if mostrar_demostracion:
        with st.expander("Demostracion formal (para copiar al examen)", expanded=False):
            st.markdown(
                "Sea $f$ continua en $[a, b]$ con $f(a) \\cdot f(b) < 0$. "
                "Sin perdida de generalidad supongamos $f(a) < 0 < f(b)$."
            )
            st.markdown("**Paso 1 — Definir el conjunto.**")
            st.latex(r"S = \{\, x \in [a,b] \;:\; f(x) < 0 \,\}")
            st.markdown(
                "$S$ es **no vacio** ($a \\in S$) y esta **acotado superiormente** por $b$."
            )
            st.markdown("**Paso 2 — Tomar el supremo.**")
            st.markdown("Por el **axioma del supremo** en $\\mathbb{R}$:")
            st.latex(r"c = \sup S, \qquad c \in [a, b].")
            st.markdown("**Paso 3 — Probar $f(c) = 0$** (por el absurdo).")
            st.markdown(
                "- Si $f(c) < 0$: por continuidad existe $\\delta > 0$ con $f(x) < 0$ "
                "en $(c-\\delta, c+\\delta)$, luego $c + \\delta/2 \\in S$ contradice $c = \\sup S$."
            )
            st.markdown(
                "- Si $f(c) > 0$: por continuidad existe $\\delta > 0$ con $f(x) > 0$ "
                "en $(c-\\delta, c+\\delta)$, luego $c - \\delta/2$ es cota superior menor que $c$, contradice la minimalidad."
            )
            st.markdown("**Paso 4 — Conclusion.**")
            st.markdown(
                "Como $f(c)$ no es $<0$ ni $>0$, entonces $f(c) = 0$. "
                "Ademas $c \\neq a, b$, luego $c \\in (a,b)$. $\\blacksquare$"
            )

    st.markdown("**Verificacion numerica.**")
    col1, col2, col3 = st.columns(3)
    col1.metric("f(a)", fmt_decimal(fa),
                 delta="negativo" if fa < 0 else "positivo")
    col2.metric("f(b)", fmt_decimal(fb),
                 delta="positivo" if fb > 0 else "negativo")
    col3.metric("f(a)·f(b)", fmt_decimal(fa * fb),
                 delta="< 0 ✓ Bolzano aplica" if aplica else "≥ 0 ✗",
                 delta_color="normal" if aplica else "inverse")

    try:
        f_latex = sp.latex(f_expr)
    except Exception:
        f_latex = "f(x)"
    st.latex(rf"f(x) = {f_latex} \text{{ continua en }} [{fmt_decimal(a)},\, {fmt_decimal(b)}]")
    st.latex(
        rf"f({fmt_decimal(a)}) \cdot f({fmt_decimal(b)}) = {fmt_decimal(fa*fb)} "
        + (r"< 0" if aplica else r"\geq 0")
    )
    if aplica:
        msg = (f"Por el **Teorema de Bolzano** existe "
               f"$c \\in ({fmt_decimal(a)}, {fmt_decimal(b)})$ con $f(c) = 0$.")
        if raiz is not None:
            msg += f" El metodo numerico aproximo $c \\approx {fmt_decimal(raiz)}$."
        st.success(msg)
    else:
        st.error(
            "Bolzano NO se cumple con este intervalo: los signos no son opuestos. "
            "No se puede garantizar raiz sin elegir otro $[a,b]$."
        )

    st.plotly_chart(plot_bolzano(f_np, a, b, raiz), use_container_width=True)

    return {"fa": fa, "fb": fb, "fa_por_fb": fa * fb, "aplica": aplica}
