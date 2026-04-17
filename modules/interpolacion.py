"""Host del modulo de Interpolacion/Derivacion — submenu Lagrange y Diferencias Finitas."""

from __future__ import annotations

import streamlit as st

from modules.lagrange import render_lagrange
from modules.derivacion import render_derivacion

SUBMODULOS = {
    "Lagrange": {"icon": "📈", "wip": False},
    "Diferencias Finitas": {"icon": "📉", "wip": False},
}


def render() -> None:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Submodulo de Interpolacion/Derivacion**")
    opciones = [
        f"{v['icon']} {k}{' (WIP)' if v['wip'] else ''}"
        for k, v in SUBMODULOS.items()
    ]
    seleccion = st.sidebar.radio("Metodo", opciones, key="interp_submodulo")
    nombre = seleccion.split(" ", 1)[1].replace(" (WIP)", "").strip()

    if nombre == "Lagrange":
        render_lagrange()
    elif nombre == "Diferencias Finitas":
        render_derivacion()
