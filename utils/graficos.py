import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_funcion(f_np, a: float, b: float, nombre_f: str = "f(x)") -> go.Figure:
    x = np.linspace(a, b, 500)
    y = f_np(x)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=nombre_f,
                             line=dict(color="#00d4ff", width=2)))
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="x",
        yaxis_title="f(x)",
        margin=dict(l=40, r=20, t=30, b=40),
    )
    return fig


def plot_convergencia(ns, estimaciones, ic_lower, ic_upper, valor_exacto=None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ns, y=ic_upper, mode="lines", name="IC superior",
        line=dict(width=0), showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=ns, y=ic_lower, mode="lines", name="IC 95%",
        fill="tonexty", fillcolor="rgba(0,212,255,0.15)",
        line=dict(width=0),
    ))
    fig.add_trace(go.Scatter(
        x=ns, y=estimaciones, mode="lines+markers", name="Estimacion",
        line=dict(color="#00d4ff", width=2),
        marker=dict(size=5),
    ))
    if valor_exacto is not None:
        fig.add_hline(y=valor_exacto, line_dash="dash",
                      line_color="#ff6b6b", annotation_text="Valor exacto")
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="N (muestras)",
        yaxis_title="Estimacion",
        xaxis_type="log",
        margin=dict(l=40, r=20, t=30, b=40),
    )
    return fig


def plot_scatter_montecarlo(x_vals, y_vals, dentro, f_np, a, b) -> go.Figure:
    x_curva = np.linspace(a, b, 500)
    y_curva = f_np(x_curva)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals[dentro], y=y_vals[dentro],
        mode="markers", name="Dentro",
        marker=dict(color="#00d4ff", size=3, opacity=0.5),
    ))
    fig.add_trace(go.Scatter(
        x=x_vals[~dentro], y=y_vals[~dentro],
        mode="markers", name="Fuera",
        marker=dict(color="#ff6b6b", size=3, opacity=0.5),
    ))
    fig.add_trace(go.Scatter(
        x=x_curva, y=y_curva, mode="lines", name="f(x)",
        line=dict(color="#ffd700", width=2),
    ))
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="x",
        yaxis_title="y",
        margin=dict(l=40, r=20, t=30, b=40),
    )
    return fig


def plot_scatter_3d(x_vals, y_vals, z_vals, dentro) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=x_vals[dentro], y=y_vals[dentro], z=z_vals[dentro],
        mode="markers", name="Dentro",
        marker=dict(color="#00d4ff", size=2, opacity=0.4),
    ))
    fig.add_trace(go.Scatter3d(
        x=x_vals[~dentro], y=y_vals[~dentro], z=z_vals[~dentro],
        mode="markers", name="Fuera",
        marker=dict(color="#ff6b6b", size=2, opacity=0.4),
    ))
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, t=30, b=0),
    )
    return fig


def plot_comparacion_barras(metodos, resultados, errores) -> go.Figure:
    colores = ["#00d4ff", "#ffd700", "#ff6b6b"]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=metodos, y=errores,
        marker_color=colores[:len(metodos)],
        text=[f"{e:.2e}" for e in errores],
        textposition="auto",
    ))
    fig.update_layout(
        template="plotly_dark",
        yaxis_title="Error absoluto",
        yaxis_type="log",
        margin=dict(l=40, r=20, t=30, b=40),
    )
    return fig
