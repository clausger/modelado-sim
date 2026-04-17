"""Glosario central de variables matematicas.

Fuente unica de verdad para tooltips y explicaciones de simbolos.
Usar `var_tooltip("a")` para obtener el texto del tooltip de una variable.
Usar `render_glosario_expander(vars=["a","b","c"])` para renderizar un expander
con las definiciones seleccionadas.
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st


@dataclass(frozen=True)
class Variable:
    simbolo: str
    nombre: str
    definicion_tecnica: str
    definicion_coloquial: str
    unidades: str = ""


GLOSARIO: dict[str, Variable] = {
    # --- Intervalos y puntos ---
    "a": Variable(
        simbolo="a",
        nombre="Extremo izquierdo del intervalo",
        definicion_tecnica="Limite inferior del intervalo cerrado [a, b] donde se busca la raiz.",
        definicion_coloquial="El numero mas chico del intervalo donde tenemos la raiz atrapada.",
    ),
    "b": Variable(
        simbolo="b",
        nombre="Extremo derecho del intervalo",
        definicion_tecnica="Limite superior del intervalo cerrado [a, b] donde se busca la raiz.",
        definicion_coloquial="El numero mas grande del intervalo donde tenemos la raiz atrapada.",
    ),
    "c": Variable(
        simbolo="c",
        nombre="Punto medio (Biseccion)",
        definicion_tecnica="Punto medio del intervalo actual: c = (a + b) / 2. Candidato a raiz en cada iteracion.",
        definicion_coloquial="El medio exacto entre a y b. Bisectar = cortar al medio.",
    ),
    "x_n": Variable(
        simbolo="x_n",
        nombre="Iterado n-esimo",
        definicion_tecnica="Aproximacion n-esima de la raiz, obtenida por la regla iterativa del metodo.",
        definicion_coloquial="El valor que vas obteniendo en cada paso. Idealmente se acerca cada vez mas a la raiz.",
    ),
    "x_star": Variable(
        simbolo="x*",
        nombre="Raiz exacta",
        definicion_tecnica="Valor exacto tal que f(x*) = 0. En general desconocido; se estima con x_n.",
        definicion_coloquial="La raiz de verdad — la que nunca conocemos del todo y vamos aproximando.",
    ),
    "x_0": Variable(
        simbolo="x_0",
        nombre="Aproximacion inicial",
        definicion_tecnica="Valor inicial desde el cual arranca la iteracion.",
        definicion_coloquial="Por donde arrancas a iterar. En NR y Punto Fijo importa mucho; en Biseccion directamente elegis [a,b].",
    ),
    # --- Funciones ---
    "f": Variable(
        simbolo="f(x)",
        nombre="Funcion original",
        definicion_tecnica="Funcion cuya raiz se busca: f(x) = 0.",
        definicion_coloquial="La funcion del problema. Queremos encontrar donde cruza el eje x.",
    ),
    "g": Variable(
        simbolo="g(x)",
        nombre="Funcion iteradora (Punto Fijo)",
        definicion_tecnica="Reformulacion de f(x)=0 como x = g(x). La iteracion es x_{n+1} = g(x_n).",
        definicion_coloquial="Es f(x)=0 reescrito como x=g(x). Despues iteras metiendo el resultado de vuelta en g.",
    ),
    "f_prima": Variable(
        simbolo="f'(x)",
        nombre="Derivada de f",
        definicion_tecnica="Derivada de f(x), necesaria para Newton-Raphson: x_{n+1} = x_n - f(x_n)/f'(x_n).",
        definicion_coloquial="La pendiente de f en cada punto. NR la usa para pegar el salto hacia la raiz.",
    ),
    "g_prima": Variable(
        simbolo="g'(x)",
        nombre="Derivada de g",
        definicion_tecnica="Derivada de g. La iteracion de Punto Fijo converge si |g'(x)| < 1 en el entorno del punto fijo (condicion de Lipschitz con L < 1).",
        definicion_coloquial="Si |g'| < 1 en todo el intervalo, Punto Fijo te garantiza que converge. Si no, se puede dispersar.",
    ),
    # --- Errores y tolerancias ---
    "e_n": Variable(
        simbolo="e_n",
        nombre="Error verdadero",
        definicion_tecnica="Distancia real entre iterado y raiz exacta: e_n = |x_n - x*|. Requiere conocer x* (teorico).",
        definicion_coloquial="Que tan lejos estas de la raiz de verdad. En la practica no la conocemos, por eso aproximamos.",
    ),
    "E_abs": Variable(
        simbolo="E_abs",
        nombre="Error absoluto (aproximado)",
        definicion_tecnica="Aproximacion del error verdadero via diferencia entre iteraciones: E_abs = |x_{n+1} - x_n|.",
        definicion_coloquial="La diferencia entre el valor nuevo y el anterior. Si es chica, ya casi no te movés.",
    ),
    "E_r": Variable(
        simbolo="E_r",
        nombre="Error relativo (aproximado)",
        definicion_tecnica="Error absoluto normalizado por la magnitud del ultimo iterado: E_r = |x_{n+1} - x_n| / |x_{n+1}|.",
        definicion_coloquial="Lo mismo que E_abs pero en proporcion — util cuando los valores son muy grandes o muy chicos.",
    ),
    "epsilon": Variable(
        simbolo="ε",
        nombre="Tolerancia",
        definicion_tecnica="Umbral fijado por el usuario para detener la iteracion (ej. 1e-3, 1e-6).",
        definicion_coloquial="Cuan chico queremos el error antes de cortar. Cuanto mas chico ε, mas iteraciones.",
    ),
    "N_max": Variable(
        simbolo="N_max",
        nombre="Iteraciones maximas",
        definicion_tecnica="Tope de seguridad: si n > N_max se interrumpe el metodo aunque no haya alcanzado la tolerancia.",
        definicion_coloquial="Para que no se cuelgue si no converge. Un freno de mano.",
    ),
    # --- Orden de convergencia ---
    "p": Variable(
        simbolo="p",
        nombre="Orden de convergencia",
        definicion_tecnica="Exponente en |e_{n+1}| = C·|e_n|^p. Lineal: p=1. Cuadratico: p=2. Mayor p = mas rapido.",
        definicion_coloquial="Que tan rapido te acercas. Biseccion y Punto Fijo son lineales (p=1). Newton-Raphson es cuadratico (p=2), duplica decimales correctos por paso.",
    ),
    "L": Variable(
        simbolo="L",
        nombre="Constante de Lipschitz",
        definicion_tecnica="Constante que acota la razon entre distancia de imagenes y distancia de preimagenes: |g(x)-g(y)| ≤ L·|x-y|. Si g ∈ C¹, L = max|g'|.",
        definicion_coloquial="Que tanto estira o comprime g al mapear puntos. Si L<1 (contractiva), Punto Fijo converge seguro.",
    ),
    # --- Varios ---
    "n": Variable(
        simbolo="n",
        nombre="Contador de iteracion",
        definicion_tecnica="Indice de la iteracion actual, comenzando en n=0 o n=1 segun convencion.",
        definicion_coloquial="El numero del paso en el que estas.",
    ),
}


def var_tooltip(key: str, tecnico: bool = True) -> str:
    """Devuelve el texto de tooltip de una variable.

    Args:
        key: clave en GLOSARIO.
        tecnico: True para definicion tecnica, False para coloquial.
    """
    v = GLOSARIO.get(key)
    if v is None:
        return f"[variable desconocida: {key}]"
    return f"{v.simbolo} — {v.nombre}: " + (v.definicion_tecnica if tecnico else v.definicion_coloquial)


def render_glosario_expander(vars: list[str] | None = None, titulo: str = "Glosario de variables") -> None:
    """Renderiza un expander colapsable con definiciones de variables.

    Args:
        vars: lista de claves a mostrar. Si None, muestra todas.
        titulo: titulo del expander.
    """
    claves = vars if vars else list(GLOSARIO.keys())
    with st.expander(titulo, expanded=False):
        st.caption("Todas las variables que aparecen en este modulo, con explicacion tecnica y coloquial.")
        for key in claves:
            v = GLOSARIO.get(key)
            if v is None:
                continue
            st.markdown(f"**{v.simbolo}** — {v.nombre}")
            col_tec, col_col = st.columns(2)
            with col_tec:
                st.markdown(f"*Tecnico*: {v.definicion_tecnica}")
            with col_col:
                st.markdown(f"*Coloquial*: {v.definicion_coloquial}")
            st.divider()
