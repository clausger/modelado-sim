# Modelado y Simulacion - Descripcion del Proyecto

## Contexto Academico

Aplicacion desarrollada para la materia **Modelado y Simulacion** de la **Universidad Argentina de la Empresa (UADE)**. El objetivo es construir una herramienta interactiva que permita visualizar, experimentar y comprender los metodos numericos que se estudian en la cursada.

## Objetivo

Crear una aplicacion web con Streamlit que cubra los principales metodos numericos del programa de la materia, permitiendo al usuario:

- Ingresar funciones matematicas de forma simbolica
- Configurar parametros de cada metodo (tolerancia, iteraciones, semilla, etc.)
- Visualizar resultados con graficos interactivos (Plotly)
- Comparar metodos entre si
- Ver tablas de iteraciones con analisis de error

## Stack Tecnologico

| Tecnologia | Rol |
|------------|-----|
| **Python** | Lenguaje base |
| **Streamlit** | Interfaz web interactiva |
| **Plotly** | Graficos interactivos (scatter, barras, 3D, convergencia) |
| **NumPy** | Calculo numerico y generacion de muestras aleatorias |
| **SciPy** | Metodos de integracion numerica (trapecios, Simpson) |
| **SymPy** | Parseo de funciones simbolicas, calculo de integrales exactas |
| **Pandas** | Tablas de datos y formateo |

## Estructura del Proyecto

```
modelado-sim/
├── app.py                  ← Entry point, menu principal con sidebar
├── modules/
│   ├── biseccion.py        ← Metodo de biseccion (WIP)
│   ├── punto_fijo.py       ← Iteracion de punto fijo (WIP)
│   ├── newton_raphson.py   ← Newton-Raphson (WIP)
│   ├── aitken.py           ← Aceleracion de Aitken (WIP)
│   ├── lagrange.py         ← Interpolacion de Lagrange (WIP)
│   ├── derivacion.py       ← Derivacion numerica (WIP)
│   ├── integracion.py      ← Integracion numerica (WIP)
│   ├── montecarlo.py       ← Metodo de Monte Carlo (COMPLETO)
│   └── edo.py              ← Ecuaciones diferenciales ordinarias (WIP)
├── utils/
│   ├── graficos.py         ← Helpers de Plotly compartidos
│   └── errores.py          ← Calculo de errores absoluto/relativo
├── docs/
│   ├── DESCRIPCION.md      ← Este documento
│   └── PROGRESO.md         ← Registro de avance por sesion
├── requirements.txt
├── .gitignore
└── README.md
```

## Modulos Planificados

### 1. Raices y Punto Fijo
- Metodo de biseccion (Teorema de Bolzano)
- Iteracion de punto fijo g(x) = x
- Condiciones de convergencia
- Analisis de error y criterios de parada

### 2. Newton-Raphson / Aitken
- Metodo de Newton-Raphson y su derivacion
- Convergencia cuadratica
- Aceleracion Delta^2 de Aitken
- Metodo de Steffensen
- Raices multiples y casos problematicos

### 3. Interpolacion y Derivacion
- Polinomio interpolante de Lagrange
- Diferencias divididas de Newton
- Fenomeno de Runge y nodos de Chebyshev
- Diferencias finitas (adelante, atras, central)
- Extrapolacion de Richardson

### 4. Integracion Numerica
- Regla del trapecio (simple y compuesta)
- Regla de Simpson 1/3 y 3/8
- Cuadratura de Gauss
- Integracion de Romberg

### 5. Monte Carlo (IMPLEMENTADO)
- **Integracion 1D**: estimacion de integrales con muestras aleatorias, tabla de iteraciones con resaltado de tolerancia, graficos de scatter y convergencia
- **Integracion multidimensional**: funciones de 2 y 3 variables sobre dominios compactos, visualizacion 3D
- **Comparacion de metodos**: Monte Carlo vs Trapecios vs Simpson con tabla comparativa, errores y tiempos de computo

### 6. EDOs
- Metodo de Euler (explicito e implicito)
- Metodo de Heun
- Runge-Kutta de orden 4
- Metodos multipaso (Adams-Bashforth)
- Sistemas de EDOs y analisis de estabilidad

## Funcionalidades Transversales

- **Parseo simbolico**: todas las funciones se ingresan como texto y se parsean con `sympy.sympify()` con manejo de errores
- **Formulas LaTeX**: cada metodo muestra su formula antes de los inputs usando `st.latex()`
- **Graficos Plotly**: tema oscuro, interactivos, reutilizables via `utils/graficos.py`
- **Analisis de error**: error absoluto, relativo e intervalo de confianza al 95% via `utils/errores.py`
- **Session state**: uso de `st.session_state` para mantener resultados entre rerenders

## Repositorio

- **GitHub**: https://github.com/clausger/modelado-sim
- **Ejecutar localmente**: `pip install -r requirements.txt && streamlit run app.py`
