"""Configuracion global del harness — panel en sidebar.

Un unico stream de preferencias que todos los modulos consumen via get_config().
Se persiste en st.session_state para sobrevivir reruns.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import streamlit as st

TemaPaso = Literal["tecnico", "coloquial", "ambos"]
ModoError = Literal["absoluto", "relativo", "ambos"]

_SESSION_KEY = "modelado_sim_config"


@dataclass(frozen=True)
class ConfigGlobal:
    """Preferencias globales accesibles desde cualquier modulo."""

    precision_decimal: int = 6
    modo_libro: bool = False
    tema_pasos: TemaPaso = "tecnico"
    modo_error: ModoError = "ambos"
    modo_examen: bool = False
    seed: int = 42
    mostrar_cotas_teoricas: bool = True
    mostrar_glosario: bool = True

    @property
    def precision_efectiva(self) -> int:
        """Precision a usar en round(): 5 si Modo libro, custom si no."""
        return 5 if self.modo_libro else self.precision_decimal


def get_config() -> ConfigGlobal:
    """Devuelve la configuracion actual (o los defaults si aun no se inicializo)."""
    if _SESSION_KEY not in st.session_state:
        st.session_state[_SESSION_KEY] = ConfigGlobal()
    return st.session_state[_SESSION_KEY]


def _set_config(cfg: ConfigGlobal) -> None:
    st.session_state[_SESSION_KEY] = cfg


def render_sidebar_config() -> ConfigGlobal:
    """Renderiza el panel de configuracion en el sidebar y devuelve la config vigente.

    Debe llamarse una sola vez al inicio de app.py.
    """
    cfg = get_config()

    with st.sidebar:
        with st.expander("Configuracion global", expanded=False):
            modo_examen = st.toggle(
                "Modo Examen",
                value=cfg.modo_examen,
                help="Oculta controles avanzados y maximiza salida entregable.",
            )
            modo_libro = st.toggle(
                "Modo libro (round a 5 decimales)",
                value=cfg.modo_libro,
                help="Replica exactamente la precision del codigo del profe Caceres. "
                "Usalo para verificar que el script matchea los valores del libro.",
            )

            precision_decimal = st.slider(
                "Precision decimal (display)",
                min_value=2,
                max_value=12,
                value=cfg.precision_decimal,
                disabled=modo_libro,
                help="Cantidad de decimales a mostrar en tablas y metricas. "
                "Ignorado si Modo libro esta activo (fuerza 5).",
            )

            tema_pasos = st.radio(
                "Paso a paso",
                options=["tecnico", "coloquial", "ambos"],
                index=["tecnico", "coloquial", "ambos"].index(cfg.tema_pasos),
                horizontal=True,
                help="Tecnico: notacion formal + LaTeX. "
                "Coloquial: explicacion en palabras. "
                "Ambos: ambos en paralelo.",
            )

            modo_error = st.radio(
                "Modo error",
                options=["absoluto", "relativo", "ambos"],
                index=["absoluto", "relativo", "ambos"].index(cfg.modo_error),
                horizontal=True,
            )

            seed = st.number_input(
                "Seed (Monte Carlo, etc.)",
                min_value=0,
                max_value=10_000,
                value=cfg.seed,
                step=1,
            )

            mostrar_cotas = st.checkbox(
                "Mostrar cotas teoricas",
                value=cfg.mostrar_cotas_teoricas,
                help="Ej: cota de Bolzano en Biseccion, Banach en Punto Fijo.",
            )
            mostrar_glosario = st.checkbox(
                "Mostrar glosario de variables",
                value=cfg.mostrar_glosario,
            )

    nueva = replace(
        cfg,
        precision_decimal=precision_decimal,
        modo_libro=modo_libro,
        tema_pasos=tema_pasos,  # type: ignore[arg-type]
        modo_error=modo_error,  # type: ignore[arg-type]
        modo_examen=modo_examen,
        seed=seed,
        mostrar_cotas_teoricas=mostrar_cotas,
        mostrar_glosario=mostrar_glosario,
    )
    _set_config(nueva)
    return nueva
