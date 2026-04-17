"""Infraestructura UI compartida para todos los modulos numericos.

Capa 0 del plan de arquitectura (ver docs/PROGRESO.md Sesion 3):
todos los modulos consumen estas piezas para heredar el mismo nivel de pulido.
"""

from utils.ui.config import ConfigGlobal, get_config, render_sidebar_config
from utils.ui.glosario import GLOSARIO, var_tooltip
from utils.ui.pasos import Paso, render_pasos
from utils.ui.tablas import fmt_decimal, render_tabla_iteraciones
from utils.ui.teoria import render_teoria

__all__ = [
    "ConfigGlobal",
    "GLOSARIO",
    "Paso",
    "fmt_decimal",
    "get_config",
    "render_pasos",
    "render_sidebar_config",
    "render_tabla_iteraciones",
    "render_teoria",
    "var_tooltip",
]
