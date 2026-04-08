import streamlit as st


def render():
    st.header("Newton-Raphson / Aitken")
    st.info("Proximamente")
    st.markdown("""
    **Temario del modulo:**
    - Metodo de Newton-Raphson
    - Derivacion de la formula iterativa
    - Convergencia cuadratica
    - Aceleracion de Aitken (Delta^2)
    - Metodo de Steffensen
    - Casos problematicos y raices multiples
    """)
