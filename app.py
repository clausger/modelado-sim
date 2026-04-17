import streamlit as st

from utils.ui.config import render_sidebar_config

st.set_page_config(
    page_title="Modelado y Simulacion - UADE",
    page_icon="📐",
    layout="wide",
)

MODULOS = {
    "Raices y Punto Fijo": {"icon": "🔍", "wip": False},
    "Newton-Raphson / Aitken": {"icon": "📐", "wip": True},
    "Interpolacion y Derivacion": {"icon": "📊", "wip": True},
    "Integracion Numerica": {"icon": "∫", "wip": False},
    "Montecarlo": {"icon": "🎲", "wip": False},
    "EDOs": {"icon": "📈", "wip": True},
}

st.sidebar.title("Modelado y Simulacion")
st.sidebar.markdown("**UADE** - Metodos Numericos")
st.sidebar.divider()

opciones = [f"{v['icon']}  {k}{'  (WIP)' if v['wip'] else ''}" for k, v in MODULOS.items()]
seleccion = st.sidebar.radio("Navegacion", opciones, index=4)

nombre_modulo = list(MODULOS.keys())[[o.split("  ")[1].replace("  (WIP)", "").strip()
                                       for o in opciones].index(
    seleccion.split("  ")[1].replace("  (WIP)", "").strip()
)]

# Panel de configuracion global (siempre al final del sidebar)
render_sidebar_config()

if nombre_modulo == "Raices y Punto Fijo":
    from modules.raices import render
    render()
elif nombre_modulo == "Newton-Raphson / Aitken":
    from modules.newton_raphson import render
    render()
elif nombre_modulo == "Interpolacion y Derivacion":
    from modules.lagrange import render
    render()
elif nombre_modulo == "Integracion Numerica":
    from modules.integracion import render
    render()
elif nombre_modulo == "Montecarlo":
    from modules.montecarlo import render
    render()
elif nombre_modulo == "EDOs":
    from modules.edo import render
    render()
