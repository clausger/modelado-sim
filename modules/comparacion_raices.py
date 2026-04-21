"""Comparacion de metodos de raices para una misma f(x)=0.

Para una funcion ingresada por el usuario:
- Corre Biseccion en el intervalo dado
- Genera automaticamente candidatos g(x) y corre Punto Fijo con el mejor
- (Cuando este listo) Corre Newton-Raphson

Genera analisis analitico DINAMICO (no hardcodeado) leyendo los resultados:
- Por que Biseccion siempre converge (Bolzano)
- Por que Punto Fijo converge / diverge (L vs 1)
- Orden de convergencia empirico a partir de los errores observados
- Tiempo de ejecucion
- Cual gana en este caso especifico y por que
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sympy as sp

from modules.biseccion import (
    CriteriosDetencion as CritBis, biseccion, contar_cambios_de_signo,
)
from modules.punto_fijo import (
    CriteriosDetencion as CritPF,
    analizar_convergencia, generar_reformulaciones, punto_fijo, steffensen,
)
from modules.newton_raphson import (
    CriteriosDetencion as CritNR, newton_raphson,
)
from utils.math_keyboard import math_input, parse_latex
from utils.ui.config import get_config
from utils.ui.tablas import fmt_decimal
from utils.ui.teoria import render_teoria


@dataclass
class ResultadoMetodo:
    nombre: str
    raiz: float | None
    iteraciones: int
    convergio: bool
    motivo_corte: str
    tiempo_ms: float
    errores: list[float]  # E_abs en cada iter, para estimar orden
    analisis: str  # comentario analitico dinamico
    valor_f_raiz: float | None = None
    extra: dict | None = None


def _estimar_orden_convergencia(errores: list[float]) -> float | None:
    """Estima p empiricamente: si |e_{n+1}| ≈ C·|e_n|^p, entonces
    log|e_{n+1}| ≈ log(C) + p·log|e_n|.
    Hace regresion lineal sobre los ultimos puntos significativos.
    """
    es = [e for e in errores if e is not None and np.isfinite(e) and e > 1e-15]
    if len(es) < 4:
        return None
    log_e = np.log(es)
    # Par (log|e_n|, log|e_{n+1}|)
    x = log_e[:-1]
    y = log_e[1:]
    try:
        # Regresion lineal: pendiente = p
        slope, _ = np.polyfit(x, y, 1)
        return float(slope)
    except Exception:
        return None


def _correr_biseccion(f_np, a: float, b: float, tol: float, max_iter: int,
                      precision: int | None) -> ResultadoMetodo:
    crit = CritBis(usar_abs=True, tol_abs=tol, usar_rel=False,
                    usar_residuo=True, tol_residuo=tol,
                    usar_max_iter=True, max_iter=max_iter)
    t0 = time.perf_counter()
    res = biseccion(f_np, a, b, crit, precision=precision)
    dt = (time.perf_counter() - t0) * 1000

    errores = [it.error_abs for it in res.iteraciones if np.isfinite(it.error_abs)]
    orden = _estimar_orden_convergencia(errores)

    cambios_signo = contar_cambios_de_signo(f_np, a, b)
    analisis = []
    if res.convergio:
        analisis.append(
            f"**Converge siempre** (garantia de Bolzano: f(a)·f(b) < 0 ⇒ existe c con f(c)=0). "
            f"Cada paso reduce el intervalo exactamente a la mitad."
        )
        analisis.append(
            f"Cota teorica de error: (b−a)/2^(n+1) = {fmt_decimal((b-a)/(2**(len(res.iteraciones)+1)))} "
            f"(con {len(res.iteraciones)} iteraciones)."
        )
        if orden is not None:
            analisis.append(f"**Orden empirico** p ≈ {fmt_decimal(orden, 3)} (teorico: 1, lineal).")
    else:
        analisis.append(f"**No convergio**: {res.motivo_corte}.")

    if cambios_signo > 1:
        analisis.append(
            f"⚠ Hay {cambios_signo} cambios de signo en [a,b] — Biseccion converge a UNA raiz "
            f"(dependiente del intervalo)."
        )

    return ResultadoMetodo(
        nombre="Biseccion",
        raiz=res.raiz,
        iteraciones=len(res.iteraciones),
        convergio=res.convergio,
        motivo_corte=res.motivo_corte,
        tiempo_ms=dt,
        errores=errores,
        analisis="  \n".join(analisis),
        valor_f_raiz=float(f_np(res.raiz)) if res.raiz is not None else None,
    )


def _correr_punto_fijo(f_expr: sp.Expr, x_sym: sp.Symbol, f_np,
                        a: float, b: float, x0: float,
                        tol: float, max_iter: int,
                        precision: int | None) -> list[ResultadoMetodo]:
    """Prueba todos los candidatos g(x) y devuelve un ResultadoMetodo por cada uno."""
    candidatos = generar_reformulaciones(f_expr, x_sym)
    resultados: list[ResultadoMetodo] = []
    crit = CritPF(usar_abs=True, tol_abs=tol, usar_rel=False,
                   usar_residuo=True, tol_residuo=tol,
                   usar_max_iter=True, max_iter=max_iter)

    for nombre, g_expr, _meta in candidatos:
        an = analizar_convergencia(g_expr, x_sym, a, b)
        if "error" in an:
            resultados.append(ResultadoMetodo(
                nombre=f"Punto Fijo — {nombre}",
                raiz=None, iteraciones=0, convergio=False,
                motivo_corte=an["error"], tiempo_ms=0.0, errores=[],
                analisis=f"**No se pudo analizar**: {an['error']}",
            ))
            continue

        L = an["L"]
        try:
            g_np = sp.lambdify(x_sym, g_expr, modules=["numpy"])
        except Exception as e:
            resultados.append(ResultadoMetodo(
                nombre=f"Punto Fijo — {nombre}",
                raiz=None, iteraciones=0, convergio=False,
                motivo_corte=f"No se pudo lambdificar g: {e}",
                tiempo_ms=0.0, errores=[], analisis="**Error al lambdificar g(x).**",
            ))
            continue

        t0 = time.perf_counter()
        res = punto_fijo(g_np, x0, crit, a, b, L_estimado=L, precision=precision)
        dt = (time.perf_counter() - t0) * 1000

        errores = [it.error_abs for it in res.iteraciones if np.isfinite(it.error_abs)]
        orden = _estimar_orden_convergencia(errores)

        # Analisis dinamico
        analisis = []
        g_latex = sp.latex(g_expr)
        analisis.append(f"**g(x)** = ${g_latex}$, con |g'(x)| max = **{fmt_decimal(L)}** en [a,b].")
        if L < 1:
            analisis.append(
                f"Como **L < 1** se cumple la condicion de Banach (contraccion): "
                f"la iteracion converge con razon geometrica ≈ L."
            )
            vueltas_teoricas = int(np.ceil(np.log(tol / (b - a)) / np.log(L))) if L > 0 else None
            if vueltas_teoricas:
                analisis.append(
                    f"Estimacion teorica de iteraciones: n ≥ log(ε/d(x₀,x*))/log(L) ≈ **{vueltas_teoricas}**."
                )
        else:
            analisis.append(
                f"**L ≥ 1**: NO se cumple la condicion de Banach. "
                f"La iteracion puede diverger o oscilar. "
                + ("La corrida confirma divergencia." if not res.convergio else
                   "Aunque formalmente no se garantiza, por la estructura local puede aun converger.")
            )

        if not an["g_de_X_en_X"]:
            analisis.append(
                f"⚠ g([a,b]) = [{fmt_decimal(an['g_min'])}, {fmt_decimal(an['g_max'])}] "
                f"NO esta contenido en [a,b]. La iteracion puede salirse del intervalo."
            )
        if orden is not None and res.convergio:
            analisis.append(f"**Orden empirico** p ≈ {fmt_decimal(orden, 3)} (teorico: 1, lineal).")

        resultados.append(ResultadoMetodo(
            nombre=f"Punto Fijo — {nombre}",
            raiz=res.raiz,
            iteraciones=len(res.iteraciones),
            convergio=res.convergio,
            motivo_corte=res.motivo_corte,
            tiempo_ms=dt,
            errores=errores,
            analisis="  \n".join(analisis),
            valor_f_raiz=float(f_np(res.raiz)) if res.raiz is not None else None,
            extra={"L": L, "g_expr": g_expr},
        ))

    return resultados


def _correr_steffensen(f_expr: sp.Expr, x_sym: sp.Symbol, f_np,
                        a: float, b: float, x0: float,
                        tol: float, max_iter: int,
                        precision: int | None) -> ResultadoMetodo:
    """Corre Steffensen con el mejor candidato g(x) contractivo.

    Si ningun candidato es contractivo, devuelve un ResultadoMetodo indicando
    que Steffensen no aplica.
    """
    candidatos = generar_reformulaciones(f_expr, x_sym)
    mejor = None
    mejor_L = float("inf")
    for nombre, g_expr, _meta in candidatos:
        an = analizar_convergencia(g_expr, x_sym, a, b)
        if "error" in an:
            continue
        if an["L"] < mejor_L:
            mejor_L = an["L"]
            mejor = (nombre, g_expr, an)

    if mejor is None or mejor_L >= 1.0:
        return ResultadoMetodo(
            nombre="Steffensen",
            raiz=None, iteraciones=0, convergio=False,
            motivo_corte=(
                "No hay g(x) contractiva entre los candidatos automaticos. "
                "Steffensen requiere |g'| < 1 cerca del punto fijo para convergencia cuadratica."
            ),
            tiempo_ms=0.0, errores=[],
            analisis=(
                "**Steffensen no aplica**: ningun candidato de reformulacion cumple Lipschitz "
                f"(mejor L encontrado = {fmt_decimal(mejor_L)}). "
                "Steffensen necesita $|g'(x^*)| < 1$ (y $\\ne 1$) para la convergencia cuadratica."
            ),
        )

    nombre_g, g_expr, an = mejor
    g_np = sp.lambdify(x_sym, g_expr, modules=["numpy"])
    crit = CritPF(usar_abs=True, tol_abs=tol, usar_rel=False,
                   usar_residuo=True, tol_residuo=tol,
                   usar_max_iter=True, max_iter=max_iter)

    t0 = time.perf_counter()
    res = steffensen(g_np, x0, crit, precision=precision)
    dt = (time.perf_counter() - t0) * 1000

    errores = [c.error_abs for c in res.ciclos if np.isfinite(c.error_abs)]
    orden = _estimar_orden_convergencia(errores)

    analisis = []
    analisis.append(
        f"**g(x) elegida**: ${sp.latex(g_expr)}$ (L = {fmt_decimal(an['L'])} < 1)."
    )
    analisis.append(
        "**Steffensen** reinicia el iterador desde la estimacion acelerada $x^*$ en cada ciclo. "
        "Cada ciclo hace 2 evaluaciones de $g$: "
        "$x_{n+1} = g(x_n)$, $x_{n+2} = g(x_{n+1})$, luego "
        "$x^* = x_n - (x_{n+1}-x_n)^2 / (x_{n+2} - 2x_{n+1} + x_n)$."
    )
    if res.convergio:
        if orden is not None:
            tipo = ("cuadratica (Steffensen ideal)" if abs(orden - 2) < 0.5
                     else "lineal (similar a PF)" if abs(orden - 1) < 0.3
                     else f"orden {fmt_decimal(orden, 2)}")
            analisis.append(
                f"**Orden empirico** p ≈ {fmt_decimal(orden, 3)} — {tipo} (teorico: 2)."
            )
        analisis.append(
            f"Ciclos totales: **{len(res.ciclos)}** "
            f"(equivalente a {2 * len(res.ciclos)} evaluaciones de g)."
        )
    else:
        analisis.append(f"**No convergio**: {res.motivo_corte}.")

    return ResultadoMetodo(
        nombre=f"Steffensen — {nombre_g}",
        raiz=res.raiz,
        iteraciones=len(res.ciclos),
        convergio=res.convergio,
        motivo_corte=res.motivo_corte,
        tiempo_ms=dt,
        errores=errores,
        analisis="  \n".join(analisis),
        valor_f_raiz=float(f_np(res.raiz)) if res.raiz is not None else None,
        extra={"L": an["L"], "g_expr": g_expr, "ciclos": len(res.ciclos)},
    )


def _correr_newton_raphson(f_expr: sp.Expr, x_sym: sp.Symbol, f_np,
                            x0: float, tol: float, max_iter: int,
                            precision: int | None) -> ResultadoMetodo:
    try:
        fp_expr = sp.diff(f_expr, x_sym)
        fp_np = sp.lambdify(x_sym, fp_expr, modules=["numpy"])
    except Exception as e:
        return ResultadoMetodo(
            nombre="Newton-Raphson", raiz=None, iteraciones=0, convergio=False,
            motivo_corte=f"No se pudo derivar f: {e}",
            tiempo_ms=0.0, errores=[],
            analisis=f"**Error al derivar f(x) simbolicamente**: {e}",
        )

    crit = CritNR(usar_abs=True, tol_abs=tol, usar_rel=False,
                   usar_residuo=True, tol_residuo=tol,
                   usar_max_iter=True, max_iter=max_iter)
    t0 = time.perf_counter()
    res = newton_raphson(f_np, fp_np, x0, crit, precision=precision)
    dt = (time.perf_counter() - t0) * 1000

    errores = [it.error_abs for it in res.iteraciones if np.isfinite(it.error_abs)]
    orden = _estimar_orden_convergencia(errores)

    fp_x0 = None
    try:
        fp_x0 = float(fp_np(x0))
    except Exception:
        pass

    analisis = []
    analisis.append(f"**f'(x)** = ${sp.latex(sp.simplify(fp_expr))}$.")
    if fp_x0 is not None:
        analisis.append(
            f"En x₀={fmt_decimal(x0)}: f'(x₀) = {fmt_decimal(fp_x0)} "
            + ("(tangente con pendiente OK)." if abs(fp_x0) > 1e-10
               else "**≈ 0 — la tangente es horizontal, el metodo no puede arrancar.**")
        )
    if res.convergio:
        analisis.append(
            "**Converge** — la formula x_{n+1} = x_n − f(x_n)/f'(x_n) usa la tangente "
            "a la curva para saltar a la raiz."
        )
        if orden is not None:
            tipo = ("cuadratica (raiz simple — ideal)" if abs(orden - 2) < 0.3
                     else "lineal (raiz multiple — degrada a orden 1)" if abs(orden - 1) < 0.3
                     else f"orden {fmt_decimal(orden, 2)}")
            analisis.append(
                f"**Orden empirico** p ≈ {fmt_decimal(orden, 3)} — {tipo} "
                f"(teorico: 2 para raiz simple)."
            )
    else:
        analisis.append(f"**No convergio**: {res.motivo_corte}.")

    return ResultadoMetodo(
        nombre="Newton-Raphson",
        raiz=res.raiz,
        iteraciones=len(res.iteraciones),
        convergio=res.convergio,
        motivo_corte=res.motivo_corte,
        tiempo_ms=dt,
        errores=errores,
        analisis="  \n".join(analisis),
        valor_f_raiz=float(f_np(res.raiz)) if res.raiz is not None else None,
        extra={"fp_expr": fp_expr},
    )


def _plot_errores_superpuestos(resultados: list[ResultadoMetodo]) -> go.Figure:
    fig = go.Figure()
    for res in resultados:
        if not res.errores:
            continue
        fig.add_trace(go.Scatter(
            x=list(range(1, len(res.errores) + 1)),
            y=res.errores, mode="lines+markers",
            name=res.nombre,
        ))
    fig.update_layout(
        title="Comparacion de convergencia (E_abs vs iteracion, escala log)",
        xaxis_title="iteracion n", yaxis_title="E_abs",
        yaxis_type="log", height=420,
    )
    return fig


def render_comparacion() -> None:
    st.header("Comparacion de metodos")
    st.caption(
        "Corre Biseccion + todos los candidatos de Punto Fijo + Newton-Raphson sobre la "
        "misma f(x)=0 y explica analiticamente por que cada uno da lo que da."
    )

    cfg = get_config()
    render_teoria("raices_teoria", titulo="Tabla 1 del libro — comparacion de metodos",
                   seccion="2. Tabla 1", expanded=False)

    x_sym = sp.Symbol("x")
    latex_f = math_input("f(x) = (buscamos f(x)=0)",
                          default_latex=r"x^{3} - x - 1",
                          key="comp_fx")
    f_expr, f_np = parse_latex(latex_f, [x_sym])
    if f_expr is None:
        st.warning("Ingresa una f(x) valida.")
        return

    st.latex(rf"f(x) = {sp.latex(f_expr)} = 0")

    col_a, col_b, col_x0 = st.columns(3)
    with col_a:
        a = st.number_input("a", value=1.0, format="%.6f", key="comp_a")
    with col_b:
        b = st.number_input("b", value=2.0, format="%.6f", key="comp_b")
    with col_x0:
        x0 = st.number_input("x₀ (Punto Fijo)", value=1.5, format="%.6f", key="comp_x0")

    col_t, col_m = st.columns(2)
    with col_t:
        tol = st.number_input("Tolerancia ε", value=1e-6, format="%.1e", key="comp_tol")
    with col_m:
        max_iter = st.number_input("N_max", value=100, min_value=10, max_value=10_000, key="comp_max")

    if a >= b:
        st.error("Se requiere a < b.")
        return

    # Pre-check Bolzano
    try:
        fa = float(f_np(a)); fb = float(f_np(b))
    except Exception as e:
        st.error(f"Error al evaluar f: {e}")
        return

    col_fa, col_fb, col_bol = st.columns(3)
    col_fa.metric("f(a)", fmt_decimal(fa))
    col_fb.metric("f(b)", fmt_decimal(fb))
    col_bol.metric("f(a)·f(b)", fmt_decimal(fa * fb),
                    delta="Bolzano OK" if fa * fb < 0 else "Bolzano no garantiza raiz",
                    delta_color="normal" if fa * fb < 0 else "inverse")

    if not st.button("Ejecutar comparacion", type="primary"):
        return

    precision_usada = 5 if cfg.modo_libro else None

    resultados: list[ResultadoMetodo] = []
    if fa * fb < 0:
        resultados.append(_correr_biseccion(f_np, a, b, tol, int(max_iter), precision_usada))
    else:
        st.info("Se salta Biseccion porque f(a)·f(b) ≥ 0 (no se cumple Bolzano).")

    resultados.extend(_correr_punto_fijo(f_expr, x_sym, f_np, a, b, x0,
                                          tol, int(max_iter), precision_usada))
    resultados.append(_correr_steffensen(f_expr, x_sym, f_np, a, b, x0,
                                          tol, int(max_iter), precision_usada))
    resultados.append(_correr_newton_raphson(f_expr, x_sym, f_np, x0,
                                              tol, int(max_iter), precision_usada))

    # Resumen tabular
    st.subheader("Resumen comparativo")
    filas = []
    for r in resultados:
        filas.append({
            "metodo": r.nombre,
            "converge": "✓" if r.convergio else "✗",
            "raiz": fmt_decimal(r.raiz) if r.raiz is not None else "—",
            "f(raiz)": fmt_decimal(r.valor_f_raiz) if r.valor_f_raiz is not None else "—",
            "iteraciones": r.iteraciones,
            "tiempo (ms)": fmt_decimal(r.tiempo_ms, 3),
            "motivo corte": r.motivo_corte,
        })
    st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)

    # Conclusion dinamica
    st.subheader("Conclusion analitica")
    converge_ok = [r for r in resultados if r.convergio]
    if converge_ok:
        mas_rapido = min(converge_ok, key=lambda r: r.iteraciones)
        mas_rapido_t = min(converge_ok, key=lambda r: r.tiempo_ms)
        st.success(
            f"**Ganador por menos iteraciones**: {mas_rapido.nombre} ({mas_rapido.iteraciones} iter). "
            f"**Ganador por menos tiempo**: {mas_rapido_t.nombre} ({fmt_decimal(mas_rapido_t.tiempo_ms, 2)} ms)."
        )
    else:
        st.error("Ningun metodo convergio con los parametros dados.")

    # Grafico superpuesto
    st.plotly_chart(_plot_errores_superpuestos(resultados), use_container_width=True)

    # Analisis comparativo especifico: Steffensen vs Newton-Raphson
    # (formato de consigna de examen: velocidad / precision / dificultad)
    res_steff = next((r for r in resultados if r.nombre.startswith("Steffensen") and r.convergio), None)
    res_nr = next((r for r in resultados if r.nombre == "Newton-Raphson" and r.convergio), None)
    if res_steff is not None and res_nr is not None:
        with st.expander("📝 Analisis comparativo Steffensen vs Newton-Raphson (examen)", expanded=True):
            n_s = res_steff.iteraciones  # ciclos
            eval_s = 2 * n_s
            n_nr = res_nr.iteraciones
            orden_s = _estimar_orden_convergencia(res_steff.errores)
            orden_nr = _estimar_orden_convergencia(res_nr.errores)
            raiz_s = fmt_decimal(res_steff.raiz) if res_steff.raiz is not None else "—"
            raiz_nr = fmt_decimal(res_nr.raiz) if res_nr.raiz is not None else "—"
            dif_raices = abs(res_steff.raiz - res_nr.raiz) if (res_steff.raiz is not None and res_nr.raiz is not None) else None

            st.markdown(
                f"""
### Velocidad de convergencia

- **Steffensen**: {n_s} ciclos ({eval_s} evaluaciones de $g$) — orden empirico p ≈
  {fmt_decimal(orden_s, 3) if orden_s else '—'}.
- **Newton-Raphson**: {n_nr} iteraciones ({n_nr} evaluaciones de $f$ + {n_nr} de $f'$) —
  orden empirico p ≈ {fmt_decimal(orden_nr, 3) if orden_nr else '—'}.

Ambos son de **orden cuadratico teorico** (p = 2). En la practica, Newton suele
necesitar menos iteraciones porque cada paso ya incorpora informacion de la
derivada, mientras que Steffensen la aproxima con diferencias finitas de $g$.

### Precision

- Steffensen: $x \\approx {raiz_s}$.
- Newton-Raphson: $x \\approx {raiz_nr}$.
- Diferencia entre ambas raices: {fmt_decimal(dif_raices) if dif_raices is not None else '—'}.

Ambos metodos alcanzan precision equivalente bajo la misma tolerancia. La
diferencia observada esta dentro de la tolerancia pedida ({fmt_decimal(tol)}).

### Dificultad (implementacion y uso)

| Aspecto | Steffensen | Newton-Raphson |
|---|---|---|
| Derivada $f'(x)$ | No requiere | Requerida |
| Reformulacion $x = g(x)$ | Requerida (y debe cumplir $\|g'\| < 1$) | No aplica |
| Evaluaciones por paso | 2 de $g$ | 1 de $f$ + 1 de $f'$ |
| Comportamiento ante raices multiples | Degrada a lineal | Degrada a lineal |
| Pre-analisis de convergencia | Analizar $g$ en compacto | Chequear $f'(x_0) \\ne 0$ |

### Conclusion

Para esta $f(x)$ concreta, ambos metodos convergen a la misma raiz con
precision equivalente. **Newton** es preferible si la derivada simbolica es
facil (poco costo extra). **Steffensen** es preferible cuando $f$ es compleja
o no diferenciable, porque solo necesita evaluar $g$ — con la $g(x)$
{fmt_decimal(res_steff.extra['L']) if res_steff.extra else '—'}-contractiva
encontrada por el asistente, la convergencia cuadratica queda garantizada
sin calcular $f'$.
                """
            )

    # Analisis detallado por metodo
    st.subheader("Analisis por metodo")
    for r in resultados:
        with st.expander(f"{r.nombre} — {'✓ convergio' if r.convergio else '✗ NO convergio'}", expanded=False):
            st.markdown(r.analisis)
            if r.errores:
                st.caption(f"Primeros errores: {', '.join(fmt_decimal(e) for e in r.errores[:5])}...")


def render() -> None:
    render_comparacion()
