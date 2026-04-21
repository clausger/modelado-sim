"""Tests del contador de evaluaciones de g en Steffensen.

Verifica la invariante: evaluaciones = 2 * ciclos, siempre recalculada
desde la corrida actual, nunca cacheada.
"""
from __future__ import annotations

import sympy as sp

from modules.punto_fijo import (
    CriteriosDetencion,
    g_evaluaciones_steffensen,
    steffensen,
)


def _g_cubica(x_sym: sp.Symbol):
    """g(x) = (sin(x) + 5)^(1/3) — viene del ejercicio x^3 - sin(x) - 5 = 0."""
    g_expr = (sp.sin(x_sym) + 5) ** sp.Rational(1, 3)
    return sp.lambdify(x_sym, g_expr, modules=["numpy"])


def test_cuenta_siempre_2_por_ciclo():
    x = sp.Symbol("x")
    g_np = _g_cubica(x)
    crit = CriteriosDetencion(
        usar_abs=True, tol_abs=1e-6,
        usar_rel=False, tol_rel=0.0,
        usar_residuo=False, tol_residuo=0.0,
        usar_max_iter=True, max_iter=50,
    )
    res = steffensen(g_np, x0=1.0, criterios=crit)
    assert res.convergio
    assert g_evaluaciones_steffensen(res) == 2 * len(res.ciclos)


def test_cambia_con_tolerancia():
    """Distinta tolerancia => distinto numero de ciclos => distinto conteo.

    Esto es lo que motivo el bug report: el resumen tiene que reflejar la
    corrida actual, no una anterior con otra tolerancia.
    """
    x = sp.Symbol("x")
    g_np = _g_cubica(x)
    corridas = []
    for tol in (1e-2, 1e-6, 1e-10):
        crit = CriteriosDetencion(
            usar_abs=True, tol_abs=tol,
            usar_rel=False, tol_rel=0.0,
            usar_residuo=False, tol_residuo=0.0,
            usar_max_iter=True, max_iter=50,
        )
        res = steffensen(g_np, x0=1.0, criterios=crit)
        corridas.append((tol, len(res.ciclos), g_evaluaciones_steffensen(res)))

    # Tolerancias mas estrictas requieren >= ciclos (monotono no decreciente)
    for i in range(1, len(corridas)):
        _, n_prev, _ = corridas[i - 1]
        _, n_curr, _ = corridas[i]
        assert n_curr >= n_prev, (
            f"Tolerancia mas estricta deberia requerir >= ciclos: {corridas}"
        )

    # La invariante evals = 2*ciclos se cumple en todas las corridas
    for tol, n, evals in corridas:
        assert evals == 2 * n, f"tol={tol}: ciclos={n}, evals={evals}"


def test_ejemplo_bug_report_3_ciclos_6_evaluaciones():
    """Caso de prueba directo del bug report del usuario:
    3 ciclos => 6 evaluaciones de g (no 4).
    """
    x = sp.Symbol("x")
    g_np = _g_cubica(x)
    crit = CriteriosDetencion(
        usar_abs=True, tol_abs=1e-8,
        usar_rel=False, tol_rel=0.0,
        usar_residuo=False, tol_residuo=0.0,
        usar_max_iter=True, max_iter=50,
    )
    res = steffensen(g_np, x0=1.0, criterios=crit)
    n = len(res.ciclos)
    evals = g_evaluaciones_steffensen(res)
    assert evals == 2 * n, (
        f"Steffensen con {n} ciclos deberia reportar {2 * n} evals, reporto {evals}"
    )


def test_sin_cache_entre_corridas():
    """Dos corridas consecutivas con distinta tolerancia deben dar distinto conteo
    — si el helper cachea, este test falla.
    """
    x = sp.Symbol("x")
    g_np = _g_cubica(x)

    crit_flojo = CriteriosDetencion(
        usar_abs=True, tol_abs=1e-2,
        usar_rel=False, tol_rel=0.0,
        usar_residuo=False, tol_residuo=0.0,
        usar_max_iter=True, max_iter=50,
    )
    res_flojo = steffensen(g_np, x0=1.0, criterios=crit_flojo)
    evals_flojo = g_evaluaciones_steffensen(res_flojo)

    crit_estricto = CriteriosDetencion(
        usar_abs=True, tol_abs=1e-12,
        usar_rel=False, tol_rel=0.0,
        usar_residuo=False, tol_residuo=0.0,
        usar_max_iter=True, max_iter=50,
    )
    res_estricto = steffensen(g_np, x0=1.0, criterios=crit_estricto)
    evals_estricto = g_evaluaciones_steffensen(res_estricto)

    # Las corridas son independientes: cada una reporta sus propios ciclos
    assert evals_flojo == 2 * len(res_flojo.ciclos)
    assert evals_estricto == 2 * len(res_estricto.ciclos)
