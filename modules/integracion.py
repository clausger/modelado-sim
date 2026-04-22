from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sympy as sp

from utils.errores import error_absoluto, error_relativo
from utils.graficos import plot_comparacion_barras, plot_funcion
from utils.math_keyboard import math_input, parse_expr_to_float, parse_latex
from utils.ui.tablas import fmt_decimal


def _valor_exacto(expr, x_sym, a: float, b: float):
    try:
        resultado = sp.integrate(expr, (x_sym, a, b))
        valor = float(resultado.evalf())
        if np.isfinite(valor):
            return valor
    except Exception:
        pass
    return None


TABLA_COMPARATIVA = pd.DataFrame({
    "Regla": ["Rectangulo", "Trapecio", "Simpson 1/3", "Simpson 3/8"],
    "Grado": [0, 1, 2, 3],
    "Puntos/Seg": [1, 2, 3, 4],
    "Error Compuesto": ["O(h²)", "O(h²)", "O(h⁴)", "O(h⁴)"],
    "Restriccion n": ["Ninguna", "Ninguna", "Par", "Multiplo de 3"],
})


# ---------------------------------------------------------------------------
# Algoritmos puros (sin Streamlit)
# ---------------------------------------------------------------------------

def _rectangulo(f_np, a: float, b: float, n: int):
    h = (b - a) / n
    x_mid = np.array([a + (i + 0.5) * h for i in range(n)])
    f_mid = f_np(x_mid)
    pesos = np.ones(n)
    contribuciones = h * f_mid
    resultado = float(np.sum(contribuciones))
    return resultado, x_mid, f_mid, pesos, contribuciones, h


def _trapecio(f_np, a: float, b: float, n: int):
    h = (b - a) / n
    x_vals = np.linspace(a, b, n + 1)
    f_vals = f_np(x_vals)
    pesos = np.ones(n + 1) * 2.0
    pesos[0] = 1.0
    pesos[-1] = 1.0
    contribuciones = (h / 2.0) * pesos * f_vals
    resultado = float(np.sum(contribuciones))
    return resultado, x_vals, f_vals, pesos, contribuciones, h


def _simpson13(f_np, a: float, b: float, n: int):
    h = (b - a) / n
    x_vals = np.linspace(a, b, n + 1)
    f_vals = f_np(x_vals)
    pesos = np.ones(n + 1)
    for i in range(1, n):
        pesos[i] = 4.0 if i % 2 == 1 else 2.0
    contribuciones = (h / 3.0) * pesos * f_vals
    resultado = float(np.sum(contribuciones))
    return resultado, x_vals, f_vals, pesos, contribuciones, h


def _simpson38(f_np, a: float, b: float, n: int):
    h = (b - a) / n
    x_vals = np.linspace(a, b, n + 1)
    f_vals = f_np(x_vals)
    pesos = np.ones(n + 1)
    for i in range(1, n):
        pesos[i] = 2.0 if i % 3 == 0 else 3.0
    contribuciones = (3.0 * h / 8.0) * pesos * f_vals
    resultado = float(np.sum(contribuciones))
    return resultado, x_vals, f_vals, pesos, contribuciones, h


# ---------------------------------------------------------------------------
# Funcion segura con L'Hopital automatico en extremos/puntos problematicos
# ---------------------------------------------------------------------------

def _fmt_tex(v: float, n_dec: int = 6) -> str:
    """Formato limpio para LaTeX: entero si es entero, sino n_dec decimales."""
    if not np.isfinite(v):
        return str(v)
    if abs(v - round(v)) < 1e-12:
        return str(int(round(v)))
    return f"{v:.{n_dec}f}"


def _tex(expr: sp.Expr) -> str:
    """sp.latex con notacion 'ln' (como escribe el profe) en vez de 'log'."""
    return sp.latex(expr, ln_notation=True)


def _explicar_lhopital(expr: sp.Expr, x_sym: sp.Symbol, x0: float) -> dict | None:
    """Si expr = num/den con num(x0)=den(x0)=0, devuelve info del paso L'Hopital."""
    try:
        frac = sp.together(expr)
        num, den = sp.fraction(frac)
        if den == 1:
            return None
        num_val = sp.limit(num, x_sym, x0)
        den_val = sp.limit(den, x_sym, x0)
        if num_val == 0 and den_val == 0:
            num_p = sp.diff(num, x_sym)
            den_p = sp.diff(den, x_sym)
            ratio = num_p / den_p
            limite = sp.limit(ratio, x_sym, x0)
            val = float(limite.evalf())
            if np.isfinite(val):
                return {
                    "num": num,
                    "den": den,
                    "num_p": num_p,
                    "den_p": den_p,
                    "ratio": sp.simplify(ratio),
                    "limite": val,
                }
    except Exception:
        pass
    return None


class _FuncionSegura:
    """Wrapper sobre sp.lambdify que detecta NaN/Inf y aplica limite simbolico.

    Uso:
        f = _FuncionSegura(expr, x_sym)
        f.registrar_extremos(a, b)   # pre-evalua para detectar 0/0 en a, b
        # luego f(x_array) funciona como una funcion numpy normal
    """

    def __init__(self, expr: sp.Expr, x_sym: sp.Symbol) -> None:
        self.expr = expr
        self.x_sym = x_sym
        self._raw = sp.lambdify(x_sym, expr, modules=["numpy"])
        self._cache: dict[float, float] = {}
        self.avisos: list[tuple[float, float, dict | None]] = []

    def _intentar_limite(self, x_val: float) -> float | None:
        for xc, vc in self._cache.items():
            if abs(x_val - xc) < 1e-12:
                return vc
        try:
            val = float(sp.limit(self.expr, self.x_sym, x_val).evalf())
            if np.isfinite(val):
                self._cache[x_val] = val
                return val
        except Exception:
            pass
        return None

    def registrar_extremos(self, a: float, b: float) -> None:
        for x_val in (a, b):
            try:
                with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                    y_raw = self._raw(np.float64(x_val))
                y_float = float(np.asarray(y_raw, dtype=float))
                if np.isfinite(y_float):
                    continue
            except Exception:
                pass
            val = self._intentar_limite(x_val)
            if val is not None:
                info = _explicar_lhopital(self.expr, self.x_sym, x_val)
                if not any(abs(x_val - x0) < 1e-12 for (x0, _, _) in self.avisos):
                    self.avisos.append((x_val, val, info))

    def __call__(self, x):
        escalar = np.isscalar(x) or np.ndim(x) == 0
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            raw = self._raw(x_arr)
            y = np.asarray(raw, dtype=float)
            if y.shape != x_arr.shape:
                y = np.broadcast_to(y, x_arr.shape).astype(float).copy()
            else:
                y = y.astype(float, copy=True)

        mask_bad = ~np.isfinite(y)
        if mask_bad.any():
            for i in np.where(mask_bad)[0]:
                val = self._intentar_limite(float(x_arr[i]))
                if val is not None:
                    y[i] = val

        if escalar:
            return float(y[0])
        return y


def _mostrar_avisos_lhopital(f_segura: _FuncionSegura) -> None:
    if not f_segura.avisos:
        return
    for (x0, limite, info) in f_segura.avisos:
        x0_tex = _fmt_tex(x0)
        with st.expander(
            f"⚠️ Indeterminacion 0/0 detectada en x = {x0_tex} — L'Hopital aplicado",
            expanded=True,
        ):
            if info is not None:
                st.markdown(
                    "El integrando presenta una **indeterminacion 0/0** en "
                    f"x = {x0_tex}. Se aplica regla de **L'Hopital** para obtener "
                    "el valor limite usado en la cuadratura:"
                )
                st.latex(
                    rf"\lim_{{x\to {x0_tex}}} "
                    rf"\frac{{{_tex(info['num'])}}}{{{_tex(info['den'])}}}"
                    rf" = \frac{{0}}{{0}} \;\Rightarrow\; "
                    rf"\lim_{{x\to {x0_tex}}} "
                    rf"\frac{{{_tex(info['num_p'])}}}{{{_tex(info['den_p'])}}}"
                )
                st.latex(
                    rf"f({x0_tex}) = {_tex(info['ratio'])}\Bigg|_{{x={x0_tex}}}"
                    rf" = {_fmt_tex(info['limite'])}"
                )
            else:
                st.info(
                    f"Valor limite aplicado en x = {x0_tex}: "
                    f"{_fmt_tex(limite)}"
                )


# ---------------------------------------------------------------------------
# Errores de truncamiento (simbolico, con xi configurable)
# ---------------------------------------------------------------------------

_ERROR_META = {
    "rectangulo": {
        "orden_deriv": 2,
        "label_deriv": "f''",
        "signo": +1,
        "const": lambda a, b, n: (b - a) ** 3 / (24.0 * n ** 2),
        "formula": r"E_R = \frac{(b-a)^3}{24\,n^2}\,f''(\xi)",
        "nombre": "E_R",
        "denom_tex": lambda n: rf"24 \cdot {n}^2",
        "exp_ba": 3,
        "signo_tex": "",
    },
    "trapecio": {
        "orden_deriv": 2,
        "label_deriv": "f''",
        "signo": -1,
        "const": lambda a, b, n: -((b - a) ** 3) / (12.0 * n ** 2),
        "formula": r"E_T = -\frac{(b-a)^3}{12\,n^2}\,f''(\xi)",
        "nombre": "E_T",
        "denom_tex": lambda n: rf"12 \cdot {n}^2",
        "exp_ba": 3,
        "signo_tex": "-",
    },
    "simpson13": {
        "orden_deriv": 4,
        "label_deriv": "f^{(4)}",
        "signo": -1,
        "const": lambda a, b, n: -((b - a) ** 5) / (180.0 * n ** 4),
        "formula": r"E_S = -\frac{(b-a)^5}{180\,n^4}\,f^{(4)}(\xi)",
        "nombre": "E_S",
        "denom_tex": lambda n: rf"180 \cdot {n}^4",
        "exp_ba": 5,
        "signo_tex": "-",
    },
    "simpson38": {
        "orden_deriv": 4,
        "label_deriv": "f^{(4)}",
        "signo": -1,
        "const": lambda a, b, n: -((b - a) ** 5) / (6480.0 * n ** 4),
        "formula": r"E_{3/8} = -\frac{(b-a)^5}{6480\,n^4}\,f^{(4)}(\xi)",
        "nombre": "E_{3/8}",
        "denom_tex": lambda n: rf"6480 \cdot {n}^4",
        "exp_ba": 5,
        "signo_tex": "-",
    },
}


def _calcular_error_trunc(metodo: str, expr: sp.Expr, x_sym: sp.Symbol,
                          a: float, b: float, n: int, xi: float) -> dict:
    meta = _ERROR_META[metodo]
    deriv = sp.diff(expr, x_sym, meta["orden_deriv"])
    deriv_xi_sym = deriv.subs(x_sym, xi)
    deriv_xi = float(sp.N(deriv_xi_sym))
    E = meta["const"](a, b, n) * deriv_xi
    return {
        "meta": meta,
        "deriv_expr": sp.simplify(deriv),
        "deriv_xi_sym": deriv_xi_sym,
        "deriv_xi": deriv_xi,
        "E": E,
    }


def _mostrar_error_trunc(metodo: str, expr: sp.Expr, x_sym: sp.Symbol,
                         a: float, b: float, n: int, xi: float,
                         n_dec: int) -> dict:
    info = _calcular_error_trunc(metodo, expr, x_sym, a, b, n, xi)
    meta = info["meta"]
    etiq = meta["label_deriv"]
    a_tex, b_tex, xi_tex = _fmt_tex(a), _fmt_tex(b), _fmt_tex(xi)

    st.markdown("**Formula del error de truncamiento:**")
    st.latex(meta["formula"])

    st.markdown(f"**Derivada simbolica $ {etiq}(x) $:**")
    st.latex(rf"{etiq}(x) = {_tex(info['deriv_expr'])}")

    st.markdown(rf"**Evaluada en $ \xi = {xi_tex} $:**")
    st.latex(rf"{etiq}({xi_tex}) = {_fmt_tex(info['deriv_xi'], n_dec)}")

    st.markdown("**Sustitucion numerica:**")
    deriv_val_tex = _fmt_tex(info["deriv_xi"], n_dec)
    st.latex(
        rf"{meta['nombre']} = {meta['signo_tex']}"
        rf"\frac{{({b_tex}-{a_tex})^{meta['exp_ba']}}}{{{meta['denom_tex'](n)}}}"
        rf" \cdot ({deriv_val_tex})"
    )
    # Resultado: tambien formato cientifico si es muy chico
    E = info["E"]
    if abs(E) != 0 and abs(E) < 1e-4:
        E_tex = f"{E:.6e}"
    else:
        E_tex = _fmt_tex(E, n_dec)
    st.latex(rf"{meta['nombre']} = {E_tex}")
    return info


# ---------------------------------------------------------------------------
# Paso a paso estilo examen (sustitucion numerica en formulas oficiales)
# ---------------------------------------------------------------------------

def _paso_trapecio_examen(a: float, b: float, n: int, h: float,
                          f_vals: np.ndarray, resultado: float, n_dec: int) -> None:
    st.markdown("#### Paso a paso (formato examen)")
    st.latex(
        r"I \approx \frac{h}{2}\Bigl[f(x_0) + "
        r"2\sum_{i=1}^{n-1}f(x_i) + f(x_n)\Bigr]"
    )
    f_str = [_fmt_tex(v, n_dec) for v in f_vals]
    h_tex = _fmt_tex(h)
    if n > 1:
        internos = " + ".join(f_str[1:-1])
        st.latex(
            rf"I \approx \frac{{{h_tex}}}{{2}}\bigl["
            rf"{f_str[0]} + 2({internos}) + {f_str[-1]}\bigr]"
        )
    else:
        st.latex(
            rf"I \approx \frac{{{h_tex}}}{{2}}\bigl["
            rf"{f_str[0]} + {f_str[-1]}\bigr]"
        )
    suma_int = 2.0 * float(np.sum(f_vals[1:-1])) if n > 1 else 0.0
    bracket = float(f_vals[0]) + suma_int + float(f_vals[-1])
    st.latex(
        rf"I \approx \frac{{{h_tex}}}{{2}} \cdot {_fmt_tex(bracket, n_dec)}"
        rf" = {_fmt_tex(resultado, n_dec)}"
    )


def _paso_simpson13_examen(a: float, b: float, n: int, h: float,
                           f_vals: np.ndarray, resultado: float, n_dec: int) -> None:
    st.markdown("#### Paso a paso (formato examen)")
    st.latex(
        r"I \approx \frac{h}{3}\Bigl[f(x_0) + "
        r"4\sum_{i\,\text{impar}}f(x_i) + "
        r"2\sum_{i\,\text{par}}f(x_i) + f(x_n)\Bigr]"
    )
    f_str = [_fmt_tex(v, n_dec) for v in f_vals]
    h_tex = _fmt_tex(h)
    impares = [f_str[i] for i in range(1, n, 2)]
    pares = [f_str[i] for i in range(2, n, 2)]
    imp_tex = " + ".join(impares) if impares else "0"
    if pares:
        par_tex = " + ".join(pares)
        st.latex(
            rf"I \approx \frac{{{h_tex}}}{{3}}\bigl["
            rf"{f_str[0]} + 4({imp_tex}) + 2({par_tex}) + {f_str[-1]}\bigr]"
        )
    else:
        st.latex(
            rf"I \approx \frac{{{h_tex}}}{{3}}\bigl["
            rf"{f_str[0]} + 4({imp_tex}) + {f_str[-1]}\bigr]"
        )
    s_imp = 4.0 * float(sum(f_vals[i] for i in range(1, n, 2)))
    s_par = 2.0 * float(sum(f_vals[i] for i in range(2, n, 2))) if pares else 0.0
    bracket = float(f_vals[0]) + s_imp + s_par + float(f_vals[-1])
    st.latex(
        rf"I \approx \frac{{{h_tex}}}{{3}} \cdot {_fmt_tex(bracket, n_dec)}"
        rf" = {_fmt_tex(resultado, n_dec)}"
    )


def _paso_simpson38_examen(a: float, b: float, n: int, h: float,
                           f_vals: np.ndarray, resultado: float, n_dec: int) -> None:
    st.markdown("#### Paso a paso (formato examen)")
    st.latex(
        r"I \approx \frac{3h}{8}\Bigl[f(x_0) + "
        r"3\sum_{i\,\text{no mult. 3}}f(x_i) + "
        r"2\sum_{i\,\text{mult. 3}}f(x_i) + f(x_n)\Bigr]"
    )
    f_str = [_fmt_tex(v, n_dec) for v in f_vals]
    h_tex = _fmt_tex(h)
    tres = [f_str[i] for i in range(1, n) if i % 3 != 0]
    dos = [f_str[i] for i in range(3, n, 3)]
    tres_tex = " + ".join(tres) if tres else "0"
    if dos:
        dos_tex = " + ".join(dos)
        st.latex(
            rf"I \approx \frac{{3 \cdot {h_tex}}}{{8}}\bigl["
            rf"{f_str[0]} + 3({tres_tex}) + 2({dos_tex}) + {f_str[-1]}\bigr]"
        )
    else:
        st.latex(
            rf"I \approx \frac{{3 \cdot {h_tex}}}{{8}}\bigl["
            rf"{f_str[0]} + 3({tres_tex}) + {f_str[-1]}\bigr]"
        )
    s_tres = 3.0 * float(sum(f_vals[i] for i in range(1, n) if i % 3 != 0))
    s_dos = 2.0 * float(sum(f_vals[i] for i in range(3, n, 3))) if dos else 0.0
    bracket = float(f_vals[0]) + s_tres + s_dos + float(f_vals[-1])
    st.latex(
        rf"I \approx \frac{{3 \cdot {h_tex}}}{{8}} \cdot {_fmt_tex(bracket, n_dec)}"
        rf" = {_fmt_tex(resultado, n_dec)}"
    )


def _respuesta_examen_integracion(
    metodo: str,
    expr: sp.Expr,
    x_sym: sp.Symbol,
    a: float,
    b: float,
    n: int,
    h: float,
    x_vals: np.ndarray,
    f_vals: np.ndarray,
    resultado: float,
    xi: float,
    n_dec: int,
    f_segura: _FuncionSegura,
    valor_exacto: float | None,
) -> str:
    """Genera el bloque markdown completo estilo examen/alumno para un metodo."""
    f_latex = _tex(expr)
    a_tex, b_tex = _fmt_tex(a), _fmt_tex(b)
    n_dec_tabla = min(n_dec, 6)

    bloque: list[str] = []

    # 1. Planteo
    bloque.append("**Planteo.**")
    bloque.append(rf"$$I = \int_{{{a_tex}}}^{{{b_tex}}} {f_latex}\, dx$$")
    bloque.append(rf"$f(x) = {f_latex}$")
    if valor_exacto is not None:
        bloque.append(f"**Valor exacto (SymPy):** $A = {_fmt_tex(valor_exacto, 10)}$")
    bloque.append("")

    # 2. L'Hopital si hubo
    if f_segura.avisos:
        bloque.append("**Evaluación en los extremos — indeterminación 0/0.**")
        for (x0, limite, info) in f_segura.avisos:
            x0_tex = _fmt_tex(x0)
            if info is not None:
                bloque.append(
                    rf"$$f({x0_tex}) = "
                    rf"\frac{{{_tex(info['num'])}}}{{{_tex(info['den'])}}}"
                    rf"\Bigg|_{{x={x0_tex}}} = \frac{{0}}{{0}} "
                    rf"\;\Rightarrow\; \text{{L'Hôpital}}$$"
                )
                bloque.append(
                    rf"$$\lim_{{x\to {x0_tex}}} "
                    rf"\frac{{{_tex(info['num_p'])}}}{{{_tex(info['den_p'])}}}"
                    rf" = {_tex(info['ratio'])}\Bigg|_{{x={x0_tex}}}"
                    rf" = {_fmt_tex(limite, n_dec_tabla)}$$"
                )
            else:
                bloque.append(
                    f"$f({x0_tex}) = {_fmt_tex(limite, n_dec_tabla)}$ "
                    "_(límite aplicado)_"
                )
        bloque.append("")

    # 3. Tabla i | x_i | f(x_i)
    bloque.append(
        f"**Tabla de puntos** ($n = {n}$, $h = {_fmt_tex(h, n_dec_tabla)}$):"
    )
    bloque.append("")
    bloque.append("| i | $x_i$ | $f(x_i)$ |")
    bloque.append("|---|---|---|")
    for i in range(len(x_vals)):
        bloque.append(
            f"| {i} | ${_fmt_tex(float(x_vals[i]), n_dec_tabla)}$ | "
            f"${_fmt_tex(float(f_vals[i]), n_dec_tabla)}$ |"
        )
    bloque.append("")

    # 4. Aplicacion del metodo
    f_str = [_fmt_tex(float(v), n_dec_tabla) for v in f_vals]
    h_tex = _fmt_tex(h, n_dec_tabla)

    if metodo == "trapecio":
        bloque.append(f"**Aplicación — Trapecio compuesto ($n = {n}$):**")
        bloque.append(
            r"$$I \approx \frac{h}{2}\Bigl[f(x_0) + "
            r"2\sum_{i=1}^{n-1} f(x_i) + f(x_n)\Bigr]$$"
        )
        if n > 1:
            internos = " + ".join(f_str[1:-1])
            bloque.append(
                rf"$$I \approx \frac{{{h_tex}}}{{2}}\bigl["
                rf"{f_str[0]} + 2({internos}) + {f_str[-1]}\bigr]$$"
            )
        else:
            bloque.append(
                rf"$$I \approx \frac{{{h_tex}}}{{2}}\bigl["
                rf"{f_str[0]} + {f_str[-1]}\bigr]$$"
            )
        bloque.append(rf"$$I \approx {_fmt_tex(resultado, n_dec)}$$")
    elif metodo == "simpson13":
        bloque.append(f"**Aplicación — Simpson 1/3 compuesto ($n = {n}$):**")
        bloque.append(
            r"$$I \approx \frac{h}{3}\Bigl[f(x_0) + "
            r"4\!\!\sum_{i\,\text{impar}}\!\! f(x_i) + "
            r"2\!\!\sum_{i\,\text{par}}\!\! f(x_i) + f(x_n)\Bigr]$$"
        )
        impares = [f_str[i] for i in range(1, n, 2)]
        pares = [f_str[i] for i in range(2, n, 2)]
        imp_tex = " + ".join(impares) if impares else "0"
        if pares:
            par_tex = " + ".join(pares)
            bloque.append(
                rf"$$I \approx \frac{{{h_tex}}}{{3}}\bigl["
                rf"{f_str[0]} + 4({imp_tex}) + 2({par_tex}) + {f_str[-1]}\bigr]$$"
            )
        else:
            bloque.append(
                rf"$$I \approx \frac{{{h_tex}}}{{3}}\bigl["
                rf"{f_str[0]} + 4({imp_tex}) + {f_str[-1]}\bigr]$$"
            )
        bloque.append(rf"$$I \approx {_fmt_tex(resultado, n_dec)}$$")
    elif metodo == "simpson38":
        bloque.append(f"**Aplicación — Simpson 3/8 compuesto ($n = {n}$):**")
        bloque.append(
            r"$$I \approx \frac{3h}{8}\Bigl[f(x_0) + "
            r"3\!\!\sum_{i\,\text{no mult.3}}\!\! f(x_i) + "
            r"2\!\!\sum_{i\,\text{mult.3}}\!\! f(x_i) + f(x_n)\Bigr]$$"
        )
        tres = [f_str[i] for i in range(1, n) if i % 3 != 0]
        dos = [f_str[i] for i in range(3, n, 3)]
        tres_tex = " + ".join(tres) if tres else "0"
        if dos:
            dos_tex = " + ".join(dos)
            bloque.append(
                rf"$$I \approx \frac{{3 \cdot {h_tex}}}{{8}}\bigl["
                rf"{f_str[0]} + 3({tres_tex}) + 2({dos_tex}) + {f_str[-1]}\bigr]$$"
            )
        else:
            bloque.append(
                rf"$$I \approx \frac{{3 \cdot {h_tex}}}{{8}}\bigl["
                rf"{f_str[0]} + 3({tres_tex}) + {f_str[-1]}\bigr]$$"
            )
        bloque.append(rf"$$I \approx {_fmt_tex(resultado, n_dec)}$$")
    elif metodo == "rectangulo":
        bloque.append(f"**Aplicación — Rectángulo (punto medio, $n = {n}$):**")
        bloque.append(
            r"$$I \approx h\sum_{i=0}^{n-1} "
            r"f\!\left(a + \left(i+\tfrac{1}{2}\right)h\right)$$"
        )
        suma_tex = " + ".join(f_str)
        bloque.append(rf"$$I \approx {h_tex} \cdot ({suma_tex})$$")
        bloque.append(rf"$$I \approx {_fmt_tex(resultado, n_dec)}$$")

    bloque.append("")

    # 5. Error de truncamiento
    info_err = _calcular_error_trunc(metodo, expr, x_sym, a, b, n, xi)
    meta = info_err["meta"]
    etiq = meta["label_deriv"]
    xi_tex = _fmt_tex(xi)
    deriv_val_tex = _fmt_tex(info_err["deriv_xi"], n_dec)
    E = info_err["E"]
    if abs(E) != 0 and abs(E) < 1e-4:
        E_tex = f"{E:.6e}"
    else:
        E_tex = _fmt_tex(E, n_dec)

    bloque.append(rf"**Error de truncamiento ($\xi = {xi_tex}$):**")
    bloque.append(rf"$$ {meta['formula']} $$")
    bloque.append(rf"$$ {etiq}(x) = {_tex(info_err['deriv_expr'])} $$")
    bloque.append(rf"$$ {etiq}({xi_tex}) = {deriv_val_tex} $$")
    bloque.append(
        rf"$$ {meta['nombre']} = {meta['signo_tex']}"
        rf"\frac{{({b_tex}-{a_tex})^{meta['exp_ba']}}}"
        rf"{{{meta['denom_tex'](n)}}}"
        rf" \cdot ({deriv_val_tex}) = {E_tex} $$"
    )

    # 6. Error real (si hay valor exacto)
    if valor_exacto is not None:
        err_real = abs(valor_exacto - resultado)
        bloque.append("")
        bloque.append(
            "**Error real** (contra valor exacto de SymPy): "
            rf"$|A - I| = {_fmt_tex(err_real, n_dec)}$"
        )

    return "\n".join(bloque)


def _paso_rectangulo_examen(a: float, b: float, n: int, h: float,
                            f_mid: np.ndarray, resultado: float, n_dec: int) -> None:
    st.markdown("#### Paso a paso (formato examen)")
    st.latex(
        r"I \approx h\sum_{i=0}^{n-1}f\!\left(a+\left(i+\tfrac{1}{2}\right)h\right)"
    )
    f_str = [_fmt_tex(v, n_dec) for v in f_mid]
    h_tex = _fmt_tex(h)
    suma_tex = " + ".join(f_str)
    st.latex(rf"I \approx {h_tex} \cdot ({suma_tex})")
    suma = float(np.sum(f_mid))
    st.latex(
        rf"I \approx {h_tex} \cdot {_fmt_tex(suma, n_dec)}"
        rf" = {_fmt_tex(resultado, n_dec)}"
    )


# ---------------------------------------------------------------------------
# Graficos de figuras geometricas
# ---------------------------------------------------------------------------

def _plot_rectangulos(f_np, a: float, b: float, n: int, resultado: float) -> go.Figure:
    h = (b - a) / n
    x_curva = np.linspace(a, b, 500)
    y_curva = f_np(x_curva)

    fig = go.Figure()
    for i in range(n):
        xi = a + i * h
        xm = xi + h / 2
        fm = float(f_np(np.array([xm]))[0]) if hasattr(f_np(np.array([xm])), '__len__') else float(f_np(xm))
        fig.add_shape(
            type="rect", x0=xi, x1=xi + h, y0=0, y1=fm,
            fillcolor="rgba(255,165,0,0.25)", line=dict(color="orange", width=1),
        )

    fig.add_trace(go.Scatter(x=x_curva, y=y_curva, mode="lines", name="f(x)",
                             line=dict(color="#00d4ff", width=2)))
    x_mid = np.array([a + (i + 0.5) * h for i in range(n)])
    f_mid = f_np(x_mid)
    fig.add_trace(go.Scatter(x=x_mid, y=f_mid, mode="markers", name="Puntos de evaluacion",
                             marker=dict(color="red", size=7),
                             hovertemplate="x=%{x:.6f}<br>f(x)=%{y:.6f}<extra></extra>"))
    fig.update_layout(
        template="plotly_dark",
        title=f"Rectangulo (Punto Medio) — I ≈ {resultado:.8f}",
        xaxis_title="x", yaxis_title="f(x)",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def _plot_trapecios(f_np, a: float, b: float, n: int, resultado: float) -> go.Figure:
    h = (b - a) / n
    x_curva = np.linspace(a, b, 500)
    y_curva = f_np(x_curva)
    x_vals = np.linspace(a, b, n + 1)
    f_vals = f_np(x_vals)

    fig = go.Figure()
    for i in range(n):
        xi, xi1 = x_vals[i], x_vals[i + 1]
        fi, fi1 = f_vals[i], f_vals[i + 1]
        fig.add_trace(go.Scatter(
            x=[xi, xi1, xi1, xi, xi], y=[0, 0, fi1, fi, 0],
            fill="toself", fillcolor="rgba(255,165,0,0.25)",
            line=dict(color="orange", width=1),
            showlegend=False, hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter(x=x_curva, y=y_curva, mode="lines", name="f(x)",
                             line=dict(color="#00d4ff", width=2)))
    fig.add_trace(go.Scatter(x=x_vals, y=f_vals, mode="markers", name="Puntos de evaluacion",
                             marker=dict(color="red", size=7),
                             hovertemplate="x=%{x:.6f}<br>f(x)=%{y:.6f}<extra></extra>"))
    fig.update_layout(
        template="plotly_dark",
        title=f"Trapecio Compuesto — I ≈ {resultado:.8f}",
        xaxis_title="x", yaxis_title="f(x)",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def _plot_simpson(f_np, a: float, b: float, n: int, resultado: float, nombre: str) -> go.Figure:
    x_curva = np.linspace(a, b, 500)
    y_curva = f_np(x_curva)
    x_vals = np.linspace(a, b, n + 1)
    f_vals = f_np(x_vals)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals, y=f_vals, fill="tozeroy",
        fillcolor="rgba(255,165,0,0.25)",
        line=dict(color="orange", width=1),
        name="Area aproximada",
    ))
    fig.add_trace(go.Scatter(x=x_curva, y=y_curva, mode="lines", name="f(x)",
                             line=dict(color="#00d4ff", width=2)))
    fig.add_trace(go.Scatter(x=x_vals, y=f_vals, mode="markers", name="Puntos de evaluacion",
                             marker=dict(color="red", size=7),
                             hovertemplate="x=%{x:.6f}<br>f(x)=%{y:.6f}<extra></extra>"))
    fig.update_layout(
        template="plotly_dark",
        title=f"{nombre} — I ≈ {resultado:.8f}",
        xaxis_title="x", yaxis_title="f(x)",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Inputs / outputs compartidos
# ---------------------------------------------------------------------------

def _inputs_comunes(key_prefix: str, con_xi: bool = False):
    latex = math_input(
        label="f(x) =",
        default_latex="x^{2}+\\sin(x)",
        key=f"{key_prefix}_func",
    )
    col1, col2 = st.columns(2)
    with col1:
        a_str = st.text_input("Limite inferior (a)", value="0", key=f"{key_prefix}_a",
                              help="Acepta expresiones: pi/2, sqrt(2), -pi, e, etc.")
        b_str = st.text_input("Limite superior (b)", value="2", key=f"{key_prefix}_b",
                              help="Acepta expresiones: pi/2, sqrt(2), -pi, e, etc.")
    with col2:
        n = st.number_input("Numero de subintervalos (n)", value=10, min_value=1,
                            max_value=10_000, key=f"{key_prefix}_n")
        n_dec = st.slider("Tolerancia (decimales)", min_value=1, max_value=10, value=6,
                          key=f"{key_prefix}_tol")
    tolerancia = 10 ** (-n_dec)
    st.latex(rf"\text{{Tolerancia}} = 10^{{-{n_dec}}}")

    xi_str = ""
    if con_xi:
        xi_str = st.text_input(
            "ξ para error de truncamiento (opcional)",
            value="",
            placeholder="vacio = punto medio (a+b)/2",
            key=f"{key_prefix}_xi",
            help="Punto donde se evalua la derivada en la formula del error. "
                 "Acepta expresiones: 0.5, pi/4, (a+b)/2, etc.",
        )
    return latex, a_str, b_str, int(n), n_dec, tolerancia, xi_str


def _mostrar_resultados(resultado: float, valor_exacto, n_dec: int,
                        x_vals, f_vals, pesos, contribuciones):
    # Metricas
    cols = st.columns(3) if valor_exacto is not None else st.columns(1)
    cols[0].metric("Integral aproximada", f"{resultado:.{n_dec}f}")
    if valor_exacto is not None:
        cols[1].metric("Valor exacto (SymPy)", f"{valor_exacto:.{n_dec}f}")
        ea = error_absoluto(resultado, valor_exacto)
        er = error_relativo(resultado, valor_exacto)
        cols[2].metric("Error absoluto", fmt_decimal(ea), delta=f"relativo: {fmt_decimal(er)}", delta_color="off")

    # Tabla de iteraciones
    st.markdown("#### Tabla de puntos de evaluacion")
    idx_max = int(np.argmax(np.abs(f_vals)))
    filas = []
    for i in range(len(x_vals)):
        filas.append({
            "i": i,
            "x_i": x_vals[i],
            "f(x_i)": f_vals[i],
            "peso": pesos[i],
            "contribucion": contribuciones[i],
        })
    df = pd.DataFrame(filas)

    def _resaltar_max(row):
        if row.name == idx_max:
            return ["background-color: #3a2a00; color: #ffd700"] * len(row)
        return [""] * len(row)

    st.dataframe(
        df.style.apply(_resaltar_max, axis=1).format({
            "x_i": f"{{:.{n_dec}f}}",
            "f(x_i)": f"{{:.{n_dec}f}}",
            "peso": "{:.4f}",
            "contribucion": f"{{:.{n_dec}f}}",
        }),
        use_container_width=True,
        height=min(400, 35 * len(filas) + 50),
    )


def _tabla_convergencia(metodo_fn, f_np, a: float, b: float, n_user: int,
                        valor_exacto, n_dec: int):
    st.markdown("#### Tabla de convergencia")
    ns = []
    n_val = 2
    while n_val <= n_user:
        ns.append(n_val)
        n_val *= 2
    if not ns or ns[-1] != n_user:
        ns.append(n_user)

    filas = []
    prev_res = None
    for n_val in ns:
        res, *_ = metodo_fn(f_np, a, b, n_val)
        h = (b - a) / n_val
        err = error_absoluto(res, valor_exacto) if valor_exacto is not None else None
        converge = ""
        if prev_res is not None:
            converge = "Si" if abs(res - prev_res) < 10 ** (-n_dec) else "No"
        filas.append({
            "n": n_val,
            "h": h,
            "resultado": res,
            "error_vs_exacto": err if err is not None else "N/D",
            "converge?": converge,
        })
        prev_res = res

    df = pd.DataFrame(filas)
    formato = {"h": "{:.6e}", "resultado": f"{{:.{n_dec}f}}"}
    if valor_exacto is not None:
        formato["error_vs_exacto"] = "{:.2e}"
    st.dataframe(df.style.format(formato, na_rep="N/D"), use_container_width=True)


# ---------------------------------------------------------------------------
# Submodulos individuales
# ---------------------------------------------------------------------------

def _metodo_rectangulo():
    st.subheader("Rectangulo (Punto Medio)")

    with st.expander("Teoria del metodo"):
        st.markdown("""
        La regla del **rectangulo** (punto medio) aproxima la integral usando
        rectangulos cuya altura es el valor de la funcion en el punto medio
        de cada subintervalo.
        """)
        st.latex(r"\int_a^b f(x)\,dx \approx h\sum_{i=0}^{n-1} f\!\left(a + \left(i+\tfrac{1}{2}\right)h\right)")
        st.markdown("**Error de truncamiento:**")
        st.latex(r"E_R = \frac{(b-a)^3}{24\,n^2}\,f''(\xi)")
        st.markdown("**Restriccion de n:** Ninguna.")

    latex, a_str, b_str, n, n_dec, tol, xi_str = _inputs_comunes("rect", con_xi=True)

    if st.button("Calcular", key="rect_calc"):
        x_sym = sp.Symbol("x")
        expr, _ = parse_latex(latex, [x_sym])
        if expr is None:
            return
        a = parse_expr_to_float(a_str, "a")
        b = parse_expr_to_float(b_str, "b")
        if a is None or b is None:
            return

        f_segura = _FuncionSegura(expr, x_sym)
        f_segura.registrar_extremos(a, b)
        _mostrar_avisos_lhopital(f_segura)

        xi = parse_expr_to_float(xi_str, "xi") if xi_str.strip() else (a + b) / 2.0
        if xi is None:
            return

        valor_exacto = _valor_exacto(expr, x_sym, a, b)
        resultado, x_vals, f_vals, pesos, contrib, h = _rectangulo(f_segura, a, b, n)
        _mostrar_resultados(resultado, valor_exacto, n_dec, x_vals, f_vals, pesos, contrib)

        _paso_rectangulo_examen(a, b, n, h, f_vals, resultado, n_dec)

        with st.expander(f"📐 Error de truncamiento (ξ = {_fmt_tex(xi)})", expanded=True):
            _mostrar_error_trunc("rectangulo", expr, x_sym, a, b, n, xi, n_dec)

        with st.expander("📝 Respuesta lista para examen (formato alumno)", expanded=False):
            texto = _respuesta_examen_integracion(
                "rectangulo", expr, x_sym, a, b, n, h,
                x_vals, f_vals, resultado, xi, n_dec, f_segura, valor_exacto,
            )
            st.markdown(texto)
            st.code(texto, language="markdown")

        st.plotly_chart(_plot_rectangulos(f_segura, a, b, n, resultado), use_container_width=True)
        _tabla_convergencia(_rectangulo, f_segura, a, b, n, valor_exacto, n_dec)
        st.session_state["int_rect_res"] = resultado


def _metodo_trapecio():
    st.subheader("Trapecio Compuesto")

    with st.expander("Teoria del metodo"):
        st.markdown("""
        La regla del **trapecio** aproxima la integral conectando los valores
        de la funcion con segmentos de recta, formando trapecios en cada
        subintervalo.
        """)
        st.latex(r"\int_a^b f(x)\,dx \approx \frac{h}{2}\left[f(a) + 2\sum_{i=1}^{n-1}f(a+ih) + f(b)\right]")
        st.markdown("**Error de truncamiento:**")
        st.latex(r"E_T = -\frac{(b-a)^3}{12\,n^2}\,f''(\xi)")
        st.markdown("**Restriccion de n:** Ninguna.")

    latex, a_str, b_str, n, n_dec, tol, xi_str = _inputs_comunes("trap", con_xi=True)

    if st.button("Calcular", key="trap_calc"):
        x_sym = sp.Symbol("x")
        expr, _ = parse_latex(latex, [x_sym])
        if expr is None:
            return
        a = parse_expr_to_float(a_str, "a")
        b = parse_expr_to_float(b_str, "b")
        if a is None or b is None:
            return

        f_segura = _FuncionSegura(expr, x_sym)
        f_segura.registrar_extremos(a, b)
        _mostrar_avisos_lhopital(f_segura)

        xi = parse_expr_to_float(xi_str, "xi") if xi_str.strip() else (a + b) / 2.0
        if xi is None:
            return

        valor_exacto = _valor_exacto(expr, x_sym, a, b)
        resultado, x_vals, f_vals, pesos, contrib, h = _trapecio(f_segura, a, b, n)
        _mostrar_resultados(resultado, valor_exacto, n_dec, x_vals, f_vals, pesos, contrib)

        _paso_trapecio_examen(a, b, n, h, f_vals, resultado, n_dec)

        with st.expander(f"📐 Error de truncamiento (ξ = {_fmt_tex(xi)})", expanded=True):
            _mostrar_error_trunc("trapecio", expr, x_sym, a, b, n, xi, n_dec)

        with st.expander("📝 Respuesta lista para examen (formato alumno)", expanded=False):
            texto = _respuesta_examen_integracion(
                "trapecio", expr, x_sym, a, b, n, h,
                x_vals, f_vals, resultado, xi, n_dec, f_segura, valor_exacto,
            )
            st.markdown(texto)
            st.code(texto, language="markdown")

        st.plotly_chart(_plot_trapecios(f_segura, a, b, n, resultado), use_container_width=True)
        _tabla_convergencia(_trapecio, f_segura, a, b, n, valor_exacto, n_dec)
        st.session_state["int_trap_res"] = resultado


def _metodo_simpson13():
    st.subheader("Simpson 1/3 Compuesto")

    with st.expander("Teoria del metodo"):
        st.markdown("""
        La regla de **Simpson 1/3** aproxima la funcion con parabolas (polinomios
        de grado 2) en pares de subintervalos consecutivos.
        """)
        st.latex(r"\int_a^b f(x)\,dx \approx \frac{h}{3}\left[f(a) + 4\sum_{\text{imp}}f(a+ih) + 2\sum_{\text{par}}f(a+ih) + f(b)\right]")
        st.markdown("**Error de truncamiento:**")
        st.latex(r"E = -\frac{(b-a)^5}{180\,n^4}\,f^{(4)}(\xi)")
        st.markdown("**Restriccion de n:** n debe ser **par**.")

    latex, a_str, b_str, n, n_dec, tol, xi_str = _inputs_comunes("s13", con_xi=True)

    if n % 2 != 0:
        st.warning(f"Simpson 1/3 requiere n par. Se ajusta n = {n} → {n + 1}")
        n = n + 1

    if st.button("Calcular", key="s13_calc"):
        x_sym = sp.Symbol("x")
        expr, _ = parse_latex(latex, [x_sym])
        if expr is None:
            return
        a = parse_expr_to_float(a_str, "a")
        b = parse_expr_to_float(b_str, "b")
        if a is None or b is None:
            return

        f_segura = _FuncionSegura(expr, x_sym)
        f_segura.registrar_extremos(a, b)
        _mostrar_avisos_lhopital(f_segura)

        xi = parse_expr_to_float(xi_str, "xi") if xi_str.strip() else (a + b) / 2.0
        if xi is None:
            return

        valor_exacto = _valor_exacto(expr, x_sym, a, b)
        resultado, x_vals, f_vals, pesos, contrib, h = _simpson13(f_segura, a, b, n)
        _mostrar_resultados(resultado, valor_exacto, n_dec, x_vals, f_vals, pesos, contrib)

        _paso_simpson13_examen(a, b, n, h, f_vals, resultado, n_dec)

        with st.expander(f"📐 Error de truncamiento (ξ = {_fmt_tex(xi)})", expanded=True):
            _mostrar_error_trunc("simpson13", expr, x_sym, a, b, n, xi, n_dec)

        with st.expander("📝 Respuesta lista para examen (formato alumno)", expanded=False):
            texto = _respuesta_examen_integracion(
                "simpson13", expr, x_sym, a, b, n, h,
                x_vals, f_vals, resultado, xi, n_dec, f_segura, valor_exacto,
            )
            st.markdown(texto)
            st.code(texto, language="markdown")

        st.plotly_chart(_plot_simpson(f_segura, a, b, n, resultado, "Simpson 1/3"), use_container_width=True)

        def _s13_safe(f_np_, a_, b_, n_):
            n_adj = n_ if n_ % 2 == 0 else n_ + 1
            return _simpson13(f_np_, a_, b_, n_adj)

        _tabla_convergencia(_s13_safe, f_segura, a, b, n, valor_exacto, n_dec)
        st.session_state["int_s13_res"] = resultado


def _metodo_simpson38():
    st.subheader("Simpson 3/8 Compuesto")

    with st.expander("Teoria del metodo"):
        st.markdown("""
        La regla de **Simpson 3/8** aproxima la funcion con polinomios cubicos
        (grado 3) en grupos de 3 subintervalos.
        """)
        st.latex(r"\int_a^b f(x)\,dx \approx \frac{3h}{8}\left[f(x_0) + 3f(x_1) + 3f(x_2) + 2f(x_3) + \cdots + f(x_n)\right]")
        st.markdown("**Error de truncamiento:**")
        st.latex(r"E = -\frac{(b-a)^5}{6480\,n^4}\,f^{(4)}(\xi)")
        st.markdown("**Restriccion de n:** n debe ser **multiplo de 3**.")

    latex, a_str, b_str, n, n_dec, tol, xi_str = _inputs_comunes("s38", con_xi=True)

    if n % 3 != 0:
        n_adj = round(n / 3) * 3
        if n_adj < 3:
            n_adj = 3
        st.warning(f"Simpson 3/8 requiere n multiplo de 3. Se ajusta n = {n} → {n_adj}")
        n = n_adj

    if st.button("Calcular", key="s38_calc"):
        x_sym = sp.Symbol("x")
        expr, _ = parse_latex(latex, [x_sym])
        if expr is None:
            return
        a = parse_expr_to_float(a_str, "a")
        b = parse_expr_to_float(b_str, "b")
        if a is None or b is None:
            return

        f_segura = _FuncionSegura(expr, x_sym)
        f_segura.registrar_extremos(a, b)
        _mostrar_avisos_lhopital(f_segura)

        xi = parse_expr_to_float(xi_str, "xi") if xi_str.strip() else (a + b) / 2.0
        if xi is None:
            return

        valor_exacto = _valor_exacto(expr, x_sym, a, b)
        resultado, x_vals, f_vals, pesos, contrib, h = _simpson38(f_segura, a, b, n)
        _mostrar_resultados(resultado, valor_exacto, n_dec, x_vals, f_vals, pesos, contrib)

        _paso_simpson38_examen(a, b, n, h, f_vals, resultado, n_dec)

        with st.expander(f"📐 Error de truncamiento (ξ = {_fmt_tex(xi)})", expanded=True):
            _mostrar_error_trunc("simpson38", expr, x_sym, a, b, n, xi, n_dec)

        with st.expander("📝 Respuesta lista para examen (formato alumno)", expanded=False):
            texto = _respuesta_examen_integracion(
                "simpson38", expr, x_sym, a, b, n, h,
                x_vals, f_vals, resultado, xi, n_dec, f_segura, valor_exacto,
            )
            st.markdown(texto)
            st.code(texto, language="markdown")

        st.plotly_chart(_plot_simpson(f_segura, a, b, n, resultado, "Simpson 3/8"), use_container_width=True)

        def _s38_safe(f_np_, a_, b_, n_):
            n_adj = round(n_ / 3) * 3
            if n_adj < 3:
                n_adj = 3
            return _simpson38(f_np_, a_, b_, n_adj)

        _tabla_convergencia(_s38_safe, f_segura, a, b, n, valor_exacto, n_dec)
        st.session_state["int_s38_res"] = resultado


# ---------------------------------------------------------------------------
# Comparacion de metodos
# ---------------------------------------------------------------------------

def _comparacion():
    st.subheader("Comparacion de Metodos")

    latex, a_str, b_str, n, n_dec, tol, _ = _inputs_comunes("int_comp")

    if st.button("Comparar", key="int_comp_calc"):
        x_sym = sp.Symbol("x")
        expr, _f_np = parse_latex(latex, [x_sym])
        if expr is None:
            return
        a = parse_expr_to_float(a_str, "a")
        b = parse_expr_to_float(b_str, "b")
        if a is None or b is None:
            return

        f_np = _FuncionSegura(expr, x_sym)
        f_np.registrar_extremos(a, b)
        _mostrar_avisos_lhopital(f_np)

        valor_exacto = _valor_exacto(expr, x_sym, a, b)

        # Ajustar n para cada metodo
        n_rect = n
        n_trap = n
        n_s13 = n if n % 2 == 0 else n + 1
        n_s38_raw = round(n / 3) * 3
        n_s38 = n_s38_raw if n_s38_raw >= 3 else 3

        metodos = {
            "Rectangulo": (_rectangulo, n_rect),
            "Trapecio": (_trapecio, n_trap),
            "Simpson 1/3": (_simpson13, n_s13),
            "Simpson 3/8": (_simpson38, n_s38),
        }
        ordenes = {"Rectangulo": "O(h²)", "Trapecio": "O(h²)",
                    "Simpson 1/3": "O(h⁴)", "Simpson 3/8": "O(h⁴)"}

        filas = []
        resultados = {}
        for nombre, (fn, n_usado) in metodos.items():
            res, *_ = fn(f_np, a, b, n_usado)
            resultados[nombre] = res
            fila = {
                "Metodo": nombre,
                "n usado": n_usado,
                "Resultado": res,
                "Orden": ordenes[nombre],
            }
            if valor_exacto is not None:
                fila["Error absoluto"] = error_absoluto(res, valor_exacto)
                fila["Error relativo"] = error_relativo(res, valor_exacto)
            filas.append(fila)

        if valor_exacto is not None:
            st.metric("Valor exacto (SymPy)", f"{valor_exacto:.{n_dec}f}")

        df = pd.DataFrame(filas)
        formato = {"Resultado": f"{{:.{n_dec}f}}"}
        if valor_exacto is not None:
            formato["Error absoluto"] = lambda v: fmt_decimal(v)
            formato["Error relativo"] = lambda v: fmt_decimal(v)
        st.dataframe(df.style.format(formato), use_container_width=True)

        # Grafico de barras de errores
        if valor_exacto is not None:
            st.markdown("#### Error absoluto por metodo")
            nombres_list = list(resultados.keys())
            errores_list = [error_absoluto(resultados[m], valor_exacto) for m in nombres_list]
            colores = ["#00d4ff", "#ffd700", "#ff6b6b", "#77dd77"]
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=nombres_list, y=errores_list,
                marker_color=colores[:len(nombres_list)],
                text=[fmt_decimal(e) for e in errores_list],
                textposition="auto",
            ))
            fig_bar.update_layout(
                template="plotly_dark",
                yaxis_title="Error absoluto (escala log)", yaxis_type="log",
                margin=dict(l=40, r=20, t=30, b=40),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Grafico de convergencia superpuesto
        st.markdown("#### Convergencia comparada")
        ns_conv = []
        nv = 2
        while nv <= n:
            ns_conv.append(nv)
            nv *= 2
        if not ns_conv or ns_conv[-1] != n:
            ns_conv.append(n)

        colores_linea = {"Rectangulo": "#00d4ff", "Trapecio": "#ffd700",
                         "Simpson 1/3": "#ff6b6b", "Simpson 3/8": "#77dd77"}
        fig_conv = go.Figure()
        for nombre, (fn, _) in metodos.items():
            ests = []
            ns_plot = []
            for nv in ns_conv:
                if nombre == "Simpson 1/3":
                    nv_adj = nv if nv % 2 == 0 else nv + 1
                elif nombre == "Simpson 3/8":
                    nv_adj = round(nv / 3) * 3
                    if nv_adj < 3:
                        nv_adj = 3
                else:
                    nv_adj = nv
                res, *_ = fn(f_np, a, b, nv_adj)
                ests.append(res)
                ns_plot.append(nv_adj)
            fig_conv.add_trace(go.Scatter(
                x=ns_plot, y=ests, mode="lines+markers", name=nombre,
                line=dict(color=colores_linea[nombre], width=2),
                marker=dict(size=5),
            ))

        if valor_exacto is not None:
            fig_conv.add_hline(y=valor_exacto, line_dash="dash",
                               line_color="white", annotation_text="Valor exacto")

        fig_conv.update_layout(
            template="plotly_dark",
            xaxis_title="n (subintervalos)", yaxis_title="Resultado",
            xaxis_type="log",
            margin=dict(l=40, r=20, t=30, b=40),
        )
        st.plotly_chart(fig_conv, use_container_width=True)

        # Grafico log-log de error vs n — muestra pendientes del orden teorico
        if valor_exacto is not None:
            st.markdown("#### Error absoluto vs n (log-log)")
            st.caption(
                "La pendiente de cada recta coincide con el orden del metodo: "
                "-2 para Rectangulo/Trapecio (O(h²)), -4 para Simpson (O(h⁴)). "
                "Duplicar n reduce el error ×4 en Trapecio pero ×16 en Simpson."
            )
            ns_log = []
            nv = 4
            while nv <= max(n, 64):
                ns_log.append(nv)
                nv *= 2
            fig_loglog = go.Figure()
            for nombre, (fn, _) in metodos.items():
                errs = []
                ns_plot = []
                for nv in ns_log:
                    if nombre == "Simpson 1/3":
                        nv_adj = nv if nv % 2 == 0 else nv + 1
                    elif nombre == "Simpson 3/8":
                        nv_adj = round(nv / 3) * 3
                        if nv_adj < 3:
                            nv_adj = 3
                    else:
                        nv_adj = nv
                    res, *_ = fn(f_np, a, b, nv_adj)
                    err = error_absoluto(res, valor_exacto)
                    if err > 0:
                        errs.append(err)
                        ns_plot.append(nv_adj)
                fig_loglog.add_trace(go.Scatter(
                    x=ns_plot, y=errs, mode="lines+markers", name=nombre,
                    line=dict(color=colores_linea[nombre], width=2),
                    marker=dict(size=6),
                ))
            fig_loglog.update_layout(
                template="plotly_dark",
                xaxis_title="n (subintervalos)", yaxis_title="|error|",
                xaxis_type="log", yaxis_type="log",
                margin=dict(l=40, r=20, t=30, b=40),
            )
            st.plotly_chart(fig_loglog, use_container_width=True)

        # Analisis dinamico lista para examen
        if valor_exacto is not None:
            errores_map = {m: error_absoluto(resultados[m], valor_exacto)
                             for m in resultados}
            ganador = min(errores_map, key=errores_map.get)
            peor = max(errores_map, key=errores_map.get)
            ratio = errores_map[peor] / errores_map[ganador] if errores_map[ganador] > 0 else float("inf")
            cumple_tol = {m: (error_relativo(resultados[m], valor_exacto) <= tol)
                            for m in resultados}
            cumplen = [m for m, ok in cumple_tol.items() if ok]
            no_cumplen = [m for m, ok in cumple_tol.items() if not ok]

            with st.expander("📝 Respuesta lista para examen (analisis)"):
                tol_pct = tol * 100
                linea_cumplen = (
                    f"**Con tolerancia del {tol_pct:.2f}%** cumplieron: "
                    f"{', '.join(cumplen) if cumplen else 'ninguno'}."
                )
                linea_no_cumplen = (
                    f"No cumplieron: {', '.join(no_cumplen)}." if no_cumplen else ""
                )
                st.markdown(
                    f"""
**Valor exacto**: $I = {fmt_decimal(valor_exacto)}$ (via SymPy).

**Resultados numericos**:
"""
                )
                for nombre, res in resultados.items():
                    err_abs = errores_map[nombre]
                    err_rel_pct = error_relativo(res, valor_exacto) * 100
                    st.markdown(
                        f"- **{nombre}** (n={metodos[nombre][1]}, {ordenes[nombre]}): "
                        f"$I \\approx {fmt_decimal(res)}$ — "
                        f"error abs = {fmt_decimal(err_abs)}, "
                        f"error rel = {err_rel_pct:.4f}%"
                    )
                st.markdown(
                    f"""
{linea_cumplen} {linea_no_cumplen}

**Comparacion de velocidad de convergencia**:
- **Ganador en precision**: {ganador} con error {fmt_decimal(errores_map[ganador])}.
- **Peor**: {peor} con error {fmt_decimal(errores_map[peor])}.
- Ratio de precision: {fmt_decimal(ratio)} veces (Simpson reduce el error
  con pendiente $-4$ en log-log vs $-2$ de Rectangulo/Trapecio, asi que
  cada vez que duplicas n, Simpson mejora $\\times 16$ mientras que
  Rectangulo/Trapecio mejoran solo $\\times 4$).

**Conclusion**: para la misma cantidad de evaluaciones, Simpson es mas
preciso porque integra exactamente polinomios de grado 3, mientras que
Rectangulo solo es exacto para constantes y Trapecio para lineales.
La pendiente del log-log (ver grafico) confirma empiricamente el orden
teorico del metodo.
                    """
                )


# ---------------------------------------------------------------------------
# Render principal
# ---------------------------------------------------------------------------

def render():
    st.header("Integracion Numerica (Newton-Cotes)")

    st.markdown("#### Tabla comparativa de reglas")
    st.dataframe(TABLA_COMPARATIVA, use_container_width=True, hide_index=True)
    st.divider()

    submenu = st.radio(
        "Metodo:",
        ["Rectangulo (Punto Medio)", "Trapecio Compuesto", "Simpson 1/3",
         "Simpson 3/8", "Comparacion de Metodos"],
        horizontal=True,
        key="int_submenu",
    )

    if submenu == "Rectangulo (Punto Medio)":
        _metodo_rectangulo()
    elif submenu == "Trapecio Compuesto":
        _metodo_trapecio()
    elif submenu == "Simpson 1/3":
        _metodo_simpson13()
    elif submenu == "Simpson 3/8":
        _metodo_simpson38()
    else:
        _comparacion()
