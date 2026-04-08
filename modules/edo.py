import streamlit as st


def render():
    st.header("Ecuaciones Diferenciales Ordinarias")
    st.info("Proximamente")
    st.markdown("""
    **Temario del modulo:**
    - Metodo de Euler (explicito e implicito)
    - Metodo de Heun (Euler mejorado)
    - Runge-Kutta de orden 4 (RK4)
    - Metodos multipaso (Adams-Bashforth)
    - Sistemas de EDOs
    - Estabilidad y rigidez (stiffness)
    """)
