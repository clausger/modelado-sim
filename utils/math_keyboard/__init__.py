from __future__ import annotations

import os
import re
from typing import Optional

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import sympy as sp

_FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
_component_func = components.declare_component("math_keyboard", path=_FRONTEND_DIR)


def math_input(
    label: str = "f(x) =",
    default_latex: str = "",
    key: Optional[str] = None,
    height: int = 360,
) -> str:
    """Render a MathQuill keyboard and return the LaTeX string."""
    restore = default_latex
    if key and key in st.session_state:
        restore = st.session_state[key]

    value = _component_func(
        default_latex=restore,
        label=label,
        key=key,
        default=default_latex,
        height=height,
    )
    return value if value is not None else default_latex


def _preprocess_latex(latex: str) -> str:
    """Normalize MathQuill LaTeX for latex2sympy2."""
    s = latex.strip()
    s = s.replace("\\left(", "(").replace("\\right)", ")")
    s = s.replace("\\left|", "|").replace("\\right|", "|")
    s = s.replace("\\left[", "[").replace("\\right]", "]")
    s = s.replace("\\cdot", " \\cdot ")
    s = s.replace("\\div", " / ")
    s = s.replace("\\times", " \\cdot ")
    s = s.replace("\\operatorname{arcsin}", "\\arcsin")
    s = s.replace("\\operatorname{arccos}", "\\arccos")
    s = s.replace("\\operatorname{arctan}", "\\arctan")
    return s


def _latex_to_text(latex: str) -> str:
    """Rough LaTeX → plain text conversion as fallback."""
    s = _preprocess_latex(latex)
    s = s.replace("\\sin", "sin").replace("\\cos", "cos").replace("\\tan", "tan")
    s = s.replace("\\csc", "csc").replace("\\sec", "sec").replace("\\cot", "cot")
    s = s.replace("\\arcsin", "asin").replace("\\arccos", "acos").replace("\\arctan", "atan")
    s = s.replace("\\ln", "log").replace("\\log", "log")
    s = s.replace("\\exp", "exp")
    s = s.replace("\\pi", "pi").replace("\\infty", "oo")
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = s.replace("\\left", "").replace("\\right", "")
    # \frac{a}{b} → ((a)/(b))
    while "\\frac" in s:
        s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", s)
    # \sqrt{a} → sqrt(a)
    s = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", s)
    # \sqrt[n]{a} → root(a, n) — but just approximate
    s = re.sub(r"\\sqrt\[([^\]]*)\]\{([^{}]*)\}", r"((\2)**(1/(\1)))", s)
    # x^{n} → x**(n)
    s = re.sub(r"\^{([^{}]*)}", r"**(\1)", s)
    s = re.sub(r"\^(\w)", r"**\1", s)
    # Clean remaining braces
    s = s.replace("{", "(").replace("}", ")")
    return s


def parse_latex(
    latex: str,
    variables: list[sp.Symbol],
) -> tuple[sp.Expr | None, object | None]:
    """Convert LaTeX string to (sympy_expr, numpy_callable) or (None, None)."""
    if not latex or not latex.strip():
        return None, None

    expr = None

    # Attempt 1: latex2sympy2
    try:
        from latex2sympy2 import latex2sympy
        processed = _preprocess_latex(latex)
        expr = latex2sympy(processed)
    except Exception:
        pass

    # Attempt 2: manual conversion → sympify
    if expr is None:
        try:
            text = _latex_to_text(latex)
            expr = sp.sympify(text, locals={
                "x": sp.Symbol("x"), "y": sp.Symbol("y"), "z": sp.Symbol("z"),
                "t": sp.Symbol("t"), "n": sp.Symbol("n"),
                "e": sp.E, "pi": sp.pi, "oo": sp.oo,
                "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
                "csc": sp.csc, "sec": sp.sec, "cot": sp.cot,
                "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
                "log": sp.log, "sqrt": sp.sqrt, "exp": sp.exp,
                "Abs": sp.Abs,
            })
        except Exception as e:
            st.error(f"Error al parsear la expresion: {e}")
            return None, None

    try:
        f_np = sp.lambdify(variables, expr, modules=["numpy"])
        return expr, f_np
    except Exception as e:
        st.error(f"Error al crear funcion numerica: {e}")
        return None, None
