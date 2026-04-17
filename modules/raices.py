"""Host del modulo de Raices — submenu con Biseccion, Punto Fijo, Comparacion.

Cada submodulo se implementa en su propio archivo y se importa aca.
"""

from __future__ import annotations

import streamlit as st

from modules.biseccion import render_biseccion

SUBMODULOS = {
    "Biseccion": {"icon": "✂️", "wip": False},
    "Punto Fijo": {"icon": "🔁", "wip": True},
    "Comparacion": {"icon": "⚖️", "wip": True},
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
        st.header("Metodo del Punto Fijo")
        st.info("Proximamente — Capa 1 en construccion")
        st.markdown(
            "- Reformulacion de f(x)=0 como x=g(x)\n"
            "- Pre-check Lipschitz: |g'(x)| < 1 en todo el intervalo\n"
            "- Cobweb diagram (telarana)\n"
            "- Asistente de reformulacion con 3-4 candidatos de g(x)"
        )
    elif nombre == "Comparacion":
        st.header("Comparacion de metodos")
        st.info("Proximamente — requiere Biseccion + Punto Fijo + NR")
