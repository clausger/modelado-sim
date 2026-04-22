# EDO — Teoría de Cátedra (Cáceres, 2ª ed. 2026)

> Referencia canónica para el módulo de Ecuaciones Diferenciales Ordinarias.
> Notación y convenciones alineadas con los parciales del profe Cáceres.

---

## 1. Problema de Cauchy (valor inicial)

Dada una EDO de primer orden:

$$\frac{dy}{dt} = f(t, y), \qquad y(t_0) = y_0$$

buscamos aproximar $y(t)$ en el intervalo $[t_0, t_f]$ usando pasos discretos
$t_0 < t_1 < \cdots < t_N = t_f$, donde $t_{i+1} = t_i + h$ y $h = (t_f - t_0)/N$.

> El profe a veces usa $x$ en vez de $t$. Ambas son la **variable independiente**
> — la notación es intercambiable. Este módulo te deja elegir.

---

## 2. Método de Euler explícito (orden 1)

**Fórmula iterativa:**

$$y_{i+1} = y_i + h \cdot f(t_i, y_i)$$

**Interpretación geométrica:** tomar la pendiente en $(t_i, y_i)$ y extrapolar
en línea recta un paso $h$.

**Error local:** $O(h^2)$. **Error global:** $O(h)$.

**Ventajas:** simple, 1 evaluación de $f$ por paso.
**Desventajas:** impreciso salvo $h$ muy chico; puede ser inestable.

---

## 3. Método de Euler mejorado — Heun (orden 2)

Predictor-corrector: Euler como predictor + regla del trapecio como corrector.

**Fórmulas por paso:**

$$
\begin{aligned}
k_1 &= f(t_i, y_i) \\
y^{*}_{i+1} &= y_i + h \cdot k_1 \quad \text{(predictor: Euler)} \\
k_2 &= f(t_{i+1}, y^{*}_{i+1}) \\
y_{i+1} &= y_i + \frac{h}{2}(k_1 + k_2) \quad \text{(corrector)}
\end{aligned}
$$

**Error local:** $O(h^3)$. **Error global:** $O(h^2)$.

**Ventajas:** mucho más preciso que Euler con solo 2 evaluaciones por paso.
**Desventajas:** el doble de costo que Euler.

---

## 4. Runge-Kutta clásico de orden 4 — RK4

**Fórmulas por paso (4 pendientes):**

$$
\begin{aligned}
k_1 &= f(t_i, y_i) \\
k_2 &= f\!\left(t_i + \tfrac{h}{2},\; y_i + \tfrac{h}{2}k_1\right) \\
k_3 &= f\!\left(t_i + \tfrac{h}{2},\; y_i + \tfrac{h}{2}k_2\right) \\
k_4 &= f(t_i + h,\; y_i + h\,k_3) \\
y_{i+1} &= y_i + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)
\end{aligned}
$$

**Interpretación:** promedio ponderado de 4 pendientes (extremos + 2 en el medio).

**Error local:** $O(h^5)$. **Error global:** $O(h^4)$.

**Ventajas:** altísima precisión con paso moderado. Es el método por defecto.
**Desventajas:** 4 evaluaciones de $f$ por paso.

---

## 5. Análisis del error

### Error absoluto por iteración

Si conocemos la solución analítica $y(t)$:

$$E_i = |y_i - y(t_i)|$$

### Comportamiento típico del error

- **Euler explícito**: el error **crece** aproximadamente lineal en $i$
  (acumulación O(h)).
- **Heun**: el error crece más lento, como $i \cdot h^2$.
- **RK4**: el error global es $O(h^4)$, en general casi imperceptible en pasos
  razonables.

### Orden de convergencia empírico

Si duplicamos los pasos ($h \to h/2$), el error debe decaer como:

$$p \approx \log_2\!\left(\frac{E(h)}{E(h/2)}\right)$$

Esperado: $p \approx 1$ (Euler), $p \approx 2$ (Heun), $p \approx 4$ (RK4).

---

## 6. Tabla comparativa

| Método | Orden global | Evals/paso | Cuándo usarlo |
|---|---|---|---|
| Euler explícito | $O(h)$ | 1 | Didáctico, paso muy chico |
| Heun | $O(h^2)$ | 2 | Compromiso costo/precisión |
| RK4 | $O(h^4)$ | 4 | **Default** — precisión alta |

---

## 7. Criterios de parada / tolerancia

Como el método **no es iterativo** (corre un número fijo de pasos $N$), la
"tolerancia" funciona distinto que en raíces:

1. **Precisión objetivo** vs solución analítica: se verifica que
   $\max_i |y_i - y(t_i)| \leq \varepsilon$.
2. **Convergencia por refinamiento**: comparar $y_N$ con paso $h$ vs paso $h/2$.

**Convenciones del profe Cáceres:**
- Euler: tolerancia $10^{-1}$.
- RK4: tolerancia $10^{-6}$.

Si el método **no alcanza** la tolerancia con el $h$ dado, se **advierte** al
alumno de que tiene que reducir $h$.

---

## 8. Solución analítica (cuando es posible)

Cuando la EDO es **separable, lineal, exacta, Bernoulli**, etc., SymPy puede
resolverla vía `sp.dsolve`. Se usa para:

- Mostrar la solución real $y(t)$ al alumno.
- Calcular el error absoluto por iteración.
- Graficar la trayectoria exacta junto a la aproximada.

Si `sp.dsolve` falla (EDO no-elemental), el módulo reporta "sin solución
analítica cerrada" y procede solo con los valores numéricos.

---

## 9. Convenciones para los parciales

1. **Tabla iteración-por-iteración** obligatoria, con columnas:
   $i \mid t_i \mid y_i \mid f(t_i, y_i) \mid y_{i+1} \mid y_{\text{real}} \mid |E_i|$
2. **RK4**: tabla adicional con $k_1, k_2, k_3, k_4$ por paso.
3. **Resaltar** filas $i=1$ y $i=2$ (el profe pide "comprobar que la primera
   y segunda iteración coincidan con los datos de salida").
4. **Graficar** aproximación vs solución analítica cuando existe.
5. **Comparar gráfica y numéricamente** métodos Euler vs RK4 en el mismo
   problema (D-5 y A-5 lo piden explícito).
6. **Análisis del comportamiento del error** a medida que itera: crece,
   se estabiliza, oscila.

---

## 10. Banco de ejercicios del profe

### Parcial A-5
- `dy/dx = y·sin(t)`, `y(0)=1`, `[0, π]`
  - (a) Euler/RK con `h=π/10`, tol `10⁻¹`
  - (b) RK4 con `k₁..k₄`, tol `10⁻⁶`

### Parcial B-5 / C-5
- (a) `dy/dx = (x³−1)/y²`, `y(1)=3`, `h=0.1`, **Heun**, aproximar `y(1.5)`
- (b) `dy/dt = 2t·√y` (cohete), `y(1)=3`, `h=0.1`, **RK4**, aproximar `y(1.2)`

### Parcial D-5
- `dy/dx = cos(x)+x`, `y(0)=1`, `[0, π/2]`
  - (a) Euler `h=π/8`, tol `10⁻¹`
  - (b) RK4 `k₁..k₄`, tol `10⁻⁶`
