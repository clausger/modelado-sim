"""Renderer de expander de teoria.

Levanta markdown desde docs/teoria/*.md y lo renderiza en un expander.
Una sola fuente de verdad para la teoria del libro.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

_TEORIA_DIR = Path(__file__).resolve().parents[2] / "docs" / "teoria"


def cargar_teoria(nombre: str) -> str:
    """Lee un archivo de teoria por nombre (sin extension).

    Ej: cargar_teoria("raices_teoria") -> contenido de docs/teoria/raices_teoria.md
    """
    path = _TEORIA_DIR / f"{nombre}.md"
    if not path.exists():
        return f"_Archivo de teoria no encontrado: `{path}`_"
    return path.read_text(encoding="utf-8")


def render_teoria(
    nombre: str,
    titulo: str = "Teoria del libro (Caceres, 2a ed 2026)",
    expanded: bool = False,
    seccion: str | None = None,
) -> None:
    """Renderiza un expander con la teoria del libro.

    Args:
        nombre: nombre del archivo en docs/teoria/ (sin .md).
        titulo: titulo del expander.
        expanded: si abrir el expander por defecto.
        seccion: si se pasa, extrae solo esa seccion (primer heading `## {seccion}` o similar).
                 Si None, muestra todo el archivo.
    """
    contenido = cargar_teoria(nombre)

    if seccion:
        contenido = _extraer_seccion(contenido, seccion)

    with st.expander(titulo, expanded=expanded):
        st.markdown(contenido)


def _extraer_seccion(md: str, titulo_seccion: str) -> str:
    """Extrae una seccion del markdown por el texto de su heading.

    Busca el primer heading (cualquier nivel ##, ###) que contenga titulo_seccion
    y devuelve hasta el siguiente heading del mismo nivel o menor.
    """
    lineas = md.split("\n")
    inicio = None
    nivel_inicio = 0

    for i, linea in enumerate(lineas):
        if linea.startswith("#") and titulo_seccion.lower() in linea.lower():
            inicio = i
            nivel_inicio = len(linea) - len(linea.lstrip("#"))
            break

    if inicio is None:
        return f"_Seccion no encontrada: `{titulo_seccion}`_"

    fin = len(lineas)
    for i in range(inicio + 1, len(lineas)):
        linea = lineas[i]
        if linea.startswith("#"):
            nivel = len(linea) - len(linea.lstrip("#"))
            if nivel <= nivel_inicio:
                fin = i
                break

    return "\n".join(lineas[inicio:fin])
