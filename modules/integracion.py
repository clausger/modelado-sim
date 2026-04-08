import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sympy as sp

from utils.errores import error_absoluto, error_relativo
from utils.graficos import plot_comparacion_barras, plot_funcion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parsear_funcion(expr_str: str):
    try:
        x_sym = sp.Symbol("x")
        expr = sp.sympify(expr_str)
        f_np = sp.lambdify(x_sym, expr, modules=["numpy"])
        return expr, f_np, x_sym
    except (sp.SympifyError, SyntaxError, TypeError) as e:
        st.error(f"Error al parsear la funcion: {e}")
        return None, None, None


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

def _inputs_comunes(key_prefix: str):
    col1, col2 = st.columns(2)
    with col1:
        func_str = st.text_input(
            "Funcion f(x)", value="x**2 + sin(x)", key=f"{key_prefix}_func",
            help="Ejemplos: x**2, sin(x), exp(x), log(x), x**3 - 2*x",
        )
        a = st.number_input("Limite inferior (a)", value=0.0, key=f"{key_prefix}_a")
        b = st.number_input("Limite superior (b)", value=2.0, key=f"{key_prefix}_b")
    with col2:
        n = st.number_input("Numero de subintervalos (n)", value=10, min_value=1,
                            max_value=10_000, key=f"{key_prefix}_n")
        n_dec = st.slider("Tolerancia (decimales)", min_value=1, max_value=10, value=6,
                          key=f"{key_prefix}_tol")
    tolerancia = 10 ** (-n_dec)
    st.latex(rf"\text{{Tolerancia}} = 10^{{-{n_dec}}}")
    return func_str, a, b, int(n), n_dec, tolerancia


def _mostrar_resultados(resultado: float, valor_exacto, n_dec: int,
                        x_vals, f_vals, pesos, contribuciones):
    # Metricas
    cols = st.columns(3) if valor_exacto is not None else st.columns(1)
    cols[0].metric("Integral aproximada", f"{resultado:.{n_dec}f}")
    if valor_exacto is not None:
        cols[1].metric("Valor exacto (SymPy)", f"{valor_exacto:.{n_dec}f}")
        ea = error_absoluto(resultado, valor_exacto)
        er = error_relativo(resultado, valor_exacto)
        cols[2].metric("Error absoluto", f"{ea:.2e}", delta=f"relativo: {er:.2e}", delta_color="off")

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
        st.latex(r"E = \frac{(b-a)^3}{24\,n^2}\,f''(\xi)")
        st.markdown("**Restriccion de n:** Ninguna.")

    func_str, a, b, n, n_dec, tol = _inputs_comunes("rect")

    if st.button("Calcular", key="rect_calc"):
        expr, f_np, x_sym = _parsear_funcion(func_str)
        if expr is None:
            return
        valor_exacto = _valor_exacto(expr, x_sym, a, b)
        resultado, x_vals, f_vals, pesos, contrib, h = _rectangulo(f_np, a, b, n)
        _mostrar_resultados(resultado, valor_exacto, n_dec, x_vals, f_vals, pesos, contrib)
        st.plotly_chart(_plot_rectangulos(f_np, a, b, n, resultado), use_container_width=True)
        _tabla_convergencia(_rectangulo, f_np, a, b, n, valor_exacto, n_dec)
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

    func_str, a, b, n, n_dec, tol = _inputs_comunes("trap")

    if st.button("Calcular", key="trap_calc"):
        expr, f_np, x_sym = _parsear_funcion(func_str)
        if expr is None:
            return
        valor_exacto = _valor_exacto(expr, x_sym, a, b)
        resultado, x_vals, f_vals, pesos, contrib, h = _trapecio(f_np, a, b, n)
        _mostrar_resultados(resultado, valor_exacto, n_dec, x_vals, f_vals, pesos, contrib)
        st.plotly_chart(_plot_trapecios(f_np, a, b, n, resultado), use_container_width=True)
        _tabla_convergencia(_trapecio, f_np, a, b, n, valor_exacto, n_dec)
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

    func_str, a, b, n, n_dec, tol = _inputs_comunes("s13")

    if n % 2 != 0:
        st.warning(f"Simpson 1/3 requiere n par. Se ajusta n = {n} → {n + 1}")
        n = n + 1

    if st.button("Calcular", key="s13_calc"):
        expr, f_np, x_sym = _parsear_funcion(func_str)
        if expr is None:
            return
        valor_exacto = _valor_exacto(expr, x_sym, a, b)
        resultado, x_vals, f_vals, pesos, contrib, h = _simpson13(f_np, a, b, n)
        _mostrar_resultados(resultado, valor_exacto, n_dec, x_vals, f_vals, pesos, contrib)
        st.plotly_chart(_plot_simpson(f_np, a, b, n, resultado, "Simpson 1/3"), use_container_width=True)

        def _s13_safe(f_np_, a_, b_, n_):
            n_adj = n_ if n_ % 2 == 0 else n_ + 1
            return _simpson13(f_np_, a_, b_, n_adj)

        _tabla_convergencia(_s13_safe, f_np, a, b, n, valor_exacto, n_dec)
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

    func_str, a, b, n, n_dec, tol = _inputs_comunes("s38")

    if n % 3 != 0:
        n_adj = round(n / 3) * 3
        if n_adj < 3:
            n_adj = 3
        st.warning(f"Simpson 3/8 requiere n multiplo de 3. Se ajusta n = {n} → {n_adj}")
        n = n_adj

    if st.button("Calcular", key="s38_calc"):
        expr, f_np, x_sym = _parsear_funcion(func_str)
        if expr is None:
            return
        valor_exacto = _valor_exacto(expr, x_sym, a, b)
        resultado, x_vals, f_vals, pesos, contrib, h = _simpson38(f_np, a, b, n)
        _mostrar_resultados(resultado, valor_exacto, n_dec, x_vals, f_vals, pesos, contrib)
        st.plotly_chart(_plot_simpson(f_np, a, b, n, resultado, "Simpson 3/8"), use_container_width=True)

        def _s38_safe(f_np_, a_, b_, n_):
            n_adj = round(n_ / 3) * 3
            if n_adj < 3:
                n_adj = 3
            return _simpson38(f_np_, a_, b_, n_adj)

        _tabla_convergencia(_s38_safe, f_np, a, b, n, valor_exacto, n_dec)
        st.session_state["int_s38_res"] = resultado


# ---------------------------------------------------------------------------
# Comparacion de metodos
# ---------------------------------------------------------------------------

def _comparacion():
    st.subheader("Comparacion de Metodos")

    func_str, a, b, n, n_dec, tol = _inputs_comunes("int_comp")

    if st.button("Comparar", key="int_comp_calc"):
        expr, f_np, x_sym = _parsear_funcion(func_str)
        if expr is None:
            return
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
            formato["Error absoluto"] = "{:.2e}"
            formato["Error relativo"] = "{:.2e}"
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
                text=[f"{e:.2e}" for e in errores_list],
                textposition="auto",
            ))
            fig_bar.update_layout(
                template="plotly_dark",
                yaxis_title="Error absoluto", yaxis_type="log",
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
