"""Formato de tablas con resaltado + export.

Wrapper sobre DataFrame con:
- Formato decimal legible (nunca notacion cientifica si se puede evitar)
- Resaltado condicional (tolerancia alcanzada, cambio de signo, divergencia)
- Botones de export a CSV y LaTeX
"""

from __future__ import annotations

from typing import Callable

import pandas as pd
import streamlit as st

from utils.ui.config import get_config


def fmt_decimal(valor: float | None, precision: int | None = None) -> str:
    """Formato decimal legible que evita notacion cientifica si es razonable.

    Reglas:
    - None o NaN -> "—"
    - |x| >= 1e-4 y |x| < 1e6 -> decimal plano con `precision` cifras significativas
    - Fuera de ese rango -> notacion cientifica
    """
    if valor is None:
        return "—"
    try:
        x = float(valor)
    except (TypeError, ValueError):
        return str(valor)
    if x != x:  # NaN
        return "—"

    if precision is None:
        precision = get_config().precision_efectiva

    abs_x = abs(x)
    if abs_x == 0:
        return "0"
    if abs_x < 1e-4 or abs_x >= 1e6:
        return f"{x:.{precision}e}"
    return f"{x:.{precision}f}".rstrip("0").rstrip(".") or "0"


def render_tabla_iteraciones(
    df: pd.DataFrame,
    resaltar: Callable[[pd.Series], list[str]] | None = None,
    titulo: str = "Tabla de iteraciones",
    formato_columnas: dict[str, int] | None = None,
    key_export: str | None = None,
) -> None:
    """Renderiza un DataFrame con formato decimal, resaltado opcional, y export.

    Args:
        df: DataFrame a mostrar.
        resaltar: funcion que recibe una fila (Series) y devuelve lista de estilos CSS
                  uno por columna. Si None, sin resaltado.
        titulo: titulo opcional (markdown level 4).
        formato_columnas: dict {columna: precision}. Si None, usa precision global para todas.
        key_export: sufijo para las keys de los botones de download, para evitar colisiones.
    """
    if titulo:
        st.markdown(f"#### {titulo}")

    cfg = get_config()
    prec_default = cfg.precision_efectiva

    df_display = df.copy()
    for col in df_display.columns:
        if pd.api.types.is_numeric_dtype(df_display[col]) and col.lower() not in ("n", "iter", "iteracion", "iteración"):
            prec = (formato_columnas or {}).get(col, prec_default)
            df_display[col] = df_display[col].map(lambda v, p=prec: fmt_decimal(v, p))

    if resaltar is not None:
        styled = df_display.style.apply(resaltar, axis=1)
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.dataframe(df_display, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    suffix = f"_{key_export}" if key_export else ""
    with col1:
        st.download_button(
            label="Descargar CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"{key_export or 'tabla'}.csv",
            mime="text/csv",
            key=f"csv{suffix}",
        )
    with col2:
        st.download_button(
            label="Descargar LaTeX",
            data=_df_to_latex(df, prec_default).encode("utf-8"),
            file_name=f"{key_export or 'tabla'}.tex",
            mime="text/plain",
            key=f"latex{suffix}",
        )


def _df_to_latex(df: pd.DataFrame, precision: int) -> str:
    """Convierte DataFrame a tabular LaTeX listo para pegar en un documento."""
    formatters: dict[str, Callable[[object], str]] = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and col.lower() not in ("n", "iter", "iteracion", "iteración"):
            formatters[col] = lambda v, p=precision: fmt_decimal(v, p)
    try:
        return df.to_latex(index=False, formatters=formatters, escape=True)
    except Exception:
        return df.to_string(index=False)


def resaltar_tolerancia(columna_error: str, tolerancia: float) -> Callable[[pd.Series], list[str]]:
    """Factory: devuelve una funcion de resaltado que pinta verde las filas donde columna_error <= tolerancia."""

    def _resaltar(fila: pd.Series) -> list[str]:
        try:
            err = float(fila[columna_error])
            if err <= tolerancia:
                return ["background-color: #d4edda"] * len(fila)
        except (KeyError, TypeError, ValueError):
            pass
        return [""] * len(fila)

    return _resaltar
