"""Host del modulo de Raices — submenu con Biseccion, Punto Fijo, Comparacion.

Cada submodulo se implementa en su propio archivo y se importa aca.
"""

from __future__ import annotations

import streamlit as st

from modules.biseccion import render_biseccion
from modules.punto_fijo import render_punto_fijo
from modules.newton_raphson import render_newton_raphson
from modules.comparacion_raices import render_comparacion

SUBMODULOS = {
    "Biseccion": {"icon": "✂️", "wip": False},
    "Punto Fijo": {"icon": "🔁", "wip": False},
    "Newton-Raphson": {"icon": "📐", "wip": False},
    "Comparacion": {"icon": "⚖️", "wip": False},
}


def render() -> None:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Submodulo de Raices**")
    opciones = [
        f"{v['icon']} {k}{' (WIP)' if v['wip'] else ''}"
        for k, v in SUBMODULOS.items()
    ]
    seleccion = st.sidebar.radio("Metodo", opciones, key="raices_submodulo")
    # Extraer el nombre limpio
    nombre = seleccion.split(" ", 1)[1].replace(" (WIP)", "").strip()

    if nombre == "Biseccion":
        render_biseccion()
    elif nombre == "Punto Fijo":
        render_punto_fijo()
    elif nombre == "Newton-Raphson":
        render_newton_raphson()
    elif nombre == "Comparacion":
        render_comparacion()
