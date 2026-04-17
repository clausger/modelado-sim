"""Renderer de paso-a-paso con dos registros: tecnico y coloquial.

Cada Paso tiene: titulo, formula (LaTeX opcional), sustitucion, resultado, explicacion_tecnica, explicacion_coloquial.
El tema global (get_config().tema_pasos) decide cual mostrar.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import streamlit as st

from utils.ui.config import get_config


@dataclass(frozen=True)
class Paso:
    """Un paso pedagogico del proceso numerico.

    titulo: enunciado corto (ej: "Paso 3: calcular punto medio").
    formula: LaTeX con la formula general (sin numeros). Ej: r"c = \\frac{a+b}{2}"
    sustitucion: LaTeX con los numeros sustituidos. Ej: r"c = \\frac{1 + 2}{2}"
    resultado: LaTeX con el resultado. Ej: r"c = 1.5"
    explicacion_tecnica: parrafo con notacion formal.
    explicacion_coloquial: parrafo en palabras simples.
    notas: lista de bullets opcionales (warnings, observaciones).
    """

    titulo: str
    formula: str = ""
    sustitucion: str = ""
    resultado: str = ""
    explicacion_tecnica: str = ""
    explicacion_coloquial: str = ""
    notas: tuple[str, ...] = field(default_factory=tuple)


def render_pasos(pasos: list[Paso], titulo: str = "Paso a paso") -> None:
    """Renderiza una lista de pasos respetando el tema global (tecnico/coloquial/ambos)."""
    cfg = get_config()
    tema = cfg.tema_pasos

    if titulo:
        st.markdown(f"### {titulo}")

    for i, paso in enumerate(pasos, start=1):
        _render_un_paso(paso, numero=i, tema=tema)


def _render_un_paso(paso: Paso, numero: int, tema: str) -> None:
    st.markdown(f"**Paso {numero} — {paso.titulo}**")

    if paso.formula:
        st.latex(paso.formula)
    if paso.sustitucion and paso.sustitucion != paso.formula:
        st.latex(paso.sustitucion)
    if paso.resultado:
        st.latex(paso.resultado)

    if tema == "tecnico":
        if paso.explicacion_tecnica:
            st.markdown(paso.explicacion_tecnica)
    elif tema == "coloquial":
        if paso.explicacion_coloquial:
            st.markdown(paso.explicacion_coloquial)
    else:  # ambos
        col_t, col_c = st.columns(2)
        with col_t:
            st.caption("Tecnico")
            if paso.explicacion_tecnica:
                st.markdown(paso.explicacion_tecnica)
        with col_c:
            st.caption("Coloquial")
            if paso.explicacion_coloquial:
                st.markdown(paso.explicacion_coloquial)

    for nota in paso.notas:
        st.info(nota)

    st.divider()
