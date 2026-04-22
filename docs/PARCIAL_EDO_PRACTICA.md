# Práctica EDO — Repaso para el parcial

Compilación de los 4 ejercicios de EDO del banco del profe Cáceres que
trabajamos en sesión. Cada uno con: consigna → solución analítica →
Euler/Heun/RK4 → validación contra el código de la app.

Última actualización: 2026-04-21.

---

## Ejercicio 1 — Heun + RK4 (cuadro resumen con error absoluto)

### Consigna
> Modele y simule la solución de las siguientes EDOs. Calcule la solución real.
> Presente cuadro resumen con iteraciones + error absoluto. Analice el
> comportamiento del error a medida que itera.
>
> **a)** `dy/dx = (x³-1)/y²`, `y(1)=3`, `h=0.1`, **Heun**, aproximar `y(1.5)`.
>
> **b)** `dy/dt = 2t·√y` (cohete), `y(1)=3`, `h=0.1`, **RK4**, aproximar `y(1.2)`.

### Solución analítica a)
EDO separable:
```
y²·dy = (x³-1)·dx
y³/3 = x⁴/4 - x + C
y(1)=3  →  C = 9.75
y(x) = ((6x⁴ - 24x + 234)/8)^(1/3)
y(1.5) = 3.056231
```

### Solución analítica b)
EDO separable:
```
dy/√y = 2t·dt
2√y = t² + C
y(1)=3  →  C = 2√3 - 1
y(t) = ((t² + 2√3 - 1)/2)² = t⁴/4 + (√3-½)t² + (13-4√3)/4
y(1.2) = 3.810502
```

### Resultados

**a) Heun:** `y(1.5) ≈ 3.056577` — error final 3.5e-04 (5 iter)

**b) RK4:** `y(1.2) ≈ 3.810502` — error final 5.2e-07 (2 iter)

### Comportamiento del error
Ambos métodos: error crece monótonamente con las iteraciones (acumulación
de error local). Heun O(h²), RK4 O(h⁴) — órdenes consistentes.

---

## Ejercicio 2 — Euler + RK4 en `dy/dx = cos(x) + x`

### Consigna
> **Parcial D-5** — EDO: `dy/dx = cos(x) + x`, `y(0)=1`, intervalo `[0, π/2]`.
>
> **a)** Solución analítica + **Euler** (RK orden 1), precisión 10⁻¹, `h = π/8`.
> Comprobar que la 1ra y 2da iteración coincidan con los datos de salida.
>
> **b)** **RK4** precisión cercana a 10⁻⁶, tabla con `k₁..k₄`. Comparar.

### Solución analítica
EDO directa (sólo depende de x):
```
y(x) = ∫(cos(x) + x) dx = sin(x) + x²/2 + C
y(0)=1  →  C = 1
y(x) = sin(x) + x²/2 + 1
y(π/2) = 2 + π²/8 = 3.23370055
```

### Resultados

**a) Euler:** `y(π/2) ≈ 3.10874075` — error 0.1250 — **NO cumple** tol 10⁻¹

**b) RK4:** `y(π/2) ≈ 3.23370885` — error 8.3e-06 — cumple "cercano a 10⁻⁶"

### Ratio y observación
RK4 es **~15 000× más preciso** que Euler con el mismo h.

Aprendizaje para el parcial: cuando `f(x, y)` no depende de y, `k₂ = k₃`
en RK4 (ambos evalúan en `x_i + h/2`).

---

## Ejercicio 3 — Heun + RK4 con `x²` en denominador

### Consigna
> **a)** `dy/dx = (x³-1)/x²`, `y(1)=3`, `h=0.1`, **Heun**, aproximar `y(1.5)`.
>
> **b)** `dy/dt = 2t·√y` (cohete), `y(1)=3`, `h=0.1`, **RK4**, aproximar `y(1.2)`.

### Trampa
Este ejercicio es casi idéntico al **Ejercicio 1**, pero el denominador de
(a) es `x²`, no `y²`. Por eso la EDO es autónoma en x (no depende de y) y
Heun se reduce a la **regla del trapecio** sobre `∫f(x) dx`.

### Solución analítica a)
```
f(x) = (x³ - 1)/x² = x - 1/x²
y(x) = ∫(x - 1/x²) dx = x²/2 + 1/x + C
y(1)=3  →  C = 3/2
y(x) = x²/2 + 1/x + 3/2
y(1.5) = 3.291667
```

### Resultados

**a) Heun:** `y(1.5) ≈ 3.290497` — error 1.17e-03

**b) RK4 cohete:** `y(1.2) ≈ 3.810502` — error 5.2e-07 (idéntico a Ej. 1)

---

## Ejercicio 4 — Euler + RK4 en EDO lineal

### Consigna
> EDO: `dy/dx = x·e^(-sin(x)) - y·cos(x)`, `y(0)=1`, intervalo `[0, π]`.
>
> **a)** Solución analítica + **Euler** precisión 10⁻¹, `h = π/4`.
> Verificar 1ra y 2da iteración.
>
> **b)** **RK4** precisión 10⁻⁶, tabla con `k₁..k₄`. Comparar.

### Solución analítica — EDO lineal con factor integrante
Forma estándar: `y' + cos(x)·y = x·e^(-sin(x))` (es lineal)
```
P(x) = cos(x),  Q(x) = x·e^(-sin(x))
μ(x) = e^(∫cos(x)dx) = e^(sin(x))
d/dx[y·e^(sin(x))] = x·e^(-sin(x))·e^(sin(x)) = x
y·e^(sin(x)) = x²/2 + C
y(x) = (x²/2 + C)·e^(-sin(x))
y(0)=1  →  C = 1
y(x) = (x²/2 + 1)·e^(-sin(x))
y(π) = π²/2 + 1 = 5.9348022
```

### Resultados

**a) Euler:** `y(π) ≈ 2.240` — error **3.695** — **NO cumple** (ni cerca)

**b) RK4:** `y(π) ≈ 5.920` — error 0.015 — **NO cumple** 10⁻⁶

### Aprendizaje clave
Con `h = π/4` **ningún método cumple la tolerancia**. La EDO tiene fuerte
curvatura cerca de `x = π` (el término `x·e^(-sin(x))` crece rápido).

**Qué hacer en el parcial:** reportar explícitamente que el `h` sugerido
es insuficiente y estimar cuál se necesitaría (`h ≤ π/16` o menor para
Euler, `h ≤ π/32` para RK4).

RK4 es 246× más preciso que Euler — sigue siendo dramática la diferencia.

---

## Bugs encontrados y arreglados en la app durante la práctica

### Bug 1 — `_resolver_analitica` elegía rama espuria
**Síntoma:** para `dy/dt = 2t√y`, la app devolvía `y(t) = t⁴/4 - 2.232·t² + 4.982`
(rama **negativa** que cumple `y(t₀)=y₀` pero no satisface la EDO).

**Fix:** `_validar_analitica()` ahora verifica que `dy/dt(t₀)` coincida
con `f(t₀, y₀)`, descartando la rama equivocada. Archivo:
`modules/edo.py:196-231`.

### Bug 2 — Diferenciación comparaba contra `P'(x)` en vez de `f'(x)`
**Síntoma:** en el ítem 2 del parcial 2025-I (Lagrange + derivación), la
app reportaba error 0.167 (contra `P'(x)`) en vez de 0.221 (contra la
derivada real `f'(x) = π·cos(πx)`).

**Fix:** Lagrange expone `f_expr` al session state, y el módulo de
derivación prioriza `f'(x)` cuando está disponible, usando `P'(x)` sólo
como fallback. Archivos: `modules/lagrange.py:598-604`, `modules/derivacion.py:662-681`.
