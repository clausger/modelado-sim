# Raíces — Teoría de Cátedra (Cáceres, 2ª ed. 2026)

> Fuente: `docs/Modelado y Simulacion por Omar J. Cáceres segunda edición 2026-1-20.pdf`, páginas 5–19.
> Esta es la referencia canónica para los módulos de Bisección, Punto Fijo, Aitken y Newton-Raphson.
> Toda fórmula, notación y convención acá listada debe matchear literalmente la del profe.

---

## 1. Conceptos básicos (pg. 5–6)

### a) Modelo matemático
Formulación o ecuación que expresa las características esenciales de un sistema físico o proceso en términos matemáticos. Forma general:

> Variable dependiente = f(variables independientes, parámetros, funciones de fuerza)

(Cita literal del libro, referencia Chapra 5ª ed, pg 11.)

### b) Simular
Ejecutar un modelo para estudiar el comportamiento del sistema a través del tiempo o bajo diferentes condiciones.

### c) Iterar
Repetir un proceso con el objetivo de acercarse a un resultado deseado.

$$x_{n+1} = x_n$$

### d) Convergencia
Cómo las aproximaciones sucesivas se acercan a la solución $x^*$ de $f(x) = 0$.

**Velocidad de convergencia** — rapidez con la que se reduce el error:
$$e_n = |x_n - x^*|$$

### e) Orden de convergencia
$$\lim_{n \to \infty} \frac{|X_{n+1} - X^*|}{|X_n - X^*|^p} = C$$

Donde $C$ es constante, $p$ el orden, $X^*$ la solución. Equivalente:

$$|e_{n+1}| = C \cdot |e_n|^p, \quad C > 0, \; p \geq 1$$

---

## 2. Tabla 1 — Comparación de métodos iterativos (pg. 6)

| Método | Orden de convergencia | Ventajas | Desventajas |
|--------|----------------------|----------|-------------|
| **Newton Raphson** | Cuadrática | Muy rápido cerca del valor real | Necesita derivadas |
| **Bisección** | Lineal | Siempre converge | Lento |
| **Punto Fijo** | Lineal | Fácil de implementar | Requiere la condición $\lvert g'(x_0) \rvert < 1$ |

---

## 3. Cota de error (pg. 6)

Estimación del error máximo en el proceso iterativo. Dos tipos principales:

### Error Absoluto
$$E_{abs} = |x^* - x_n|$$

Como $x^*$ no se conoce, se aproxima por la diferencia entre iteraciones:

$$E_{abs} \approx |x_{n+1} - x_n|$$

### Error Relativo
$$E_r = \frac{|x^* - x_n|}{|x^*|}$$

Aproximación:

$$E_r \approx \frac{|x_{n+1} - x_n|}{|x_{n+1}|}$$

---

## 4. Criterios de detención (pg. 6–7)

Los 4 criterios oficiales del libro:

1. **Tolerancia en el error absoluto**: $|x_{n+1} - x_n| \leq \varepsilon$
2. **Tolerancia en el error relativo**: $\dfrac{|x_{n+1} - x_n|}{|x_{n+1}|} \leq \varepsilon$
3. **Condición sobre el residuo**: $|f(x)| \leq \varepsilon$
4. **Número máximo de iteraciones**: $n > N_{max}$

> **Convención UI**: los 4 deben poder activarse/desactivarse independientemente, y el método corta con el primero que se cumple.

---

## 5. Conjuntos compactos y Lipschitz (pg. 7)

### Conjunto compacto
En un espacio métrico, $K$ es compacto si es cerrado (contiene puntos límite) y acotado (distancia finita desde algún punto fijo).

Ejemplos:
1. Disco cerrado $\{(x,y) \in \mathbb{R}^2 : x^2 + y^2 \leq 1\}$
2. Segmento parametrizado $\{x = ta + (1-t)b, \; 0 \leq t \leq 1\}$
3. Rectángulo $[a,b] \times [c,d]$ en $\mathbb{R}^2$
4. Funciones continuas acotadas (Arzelá-Ascolí)

### Condición de Lipschitz
Una función $g: K \to \mathbb{R}$ en un compacto $K$ satisface Lipschitz con constante $L$ si:

$$\lvert g(x) - g(y) \rvert \leq L \cdot |x - y|, \quad \forall x, y \in K$$

Si $g \in C^1(K)$ entonces $\lvert g'(x) \rvert < M$ para todo $x \in K$, y podemos tomar $L = M$.

**Garantía de convergencia**: si $L < 1$ en $x_{n+1} = g(x_n)$, la iteración converge.

### Teorema del punto fijo de Banach
Si $X$ es compacto completo en un espacio métrico y $f$ es contractiva, existe un único punto fijo $x^* \in X$ tal que $f(x^*) = x^*$.

**Función contractiva**: existe $k \in [0,1)$ tal que:

$$d(f(x), f(y)) \leq k \cdot d(x, y), \quad \forall x, y \in X$$

Cota de error teórica:

$$d(x_n, x^*) \leq k^n \cdot d(x_0, x^*)$$

---

## 6. Búsqueda binaria de raíces — Bisección (pg. 8–10)

### Fundamento: Teorema de Bolzano
Sea $f$ continua en $[a,b]$. Si $f(a) \cdot f(b) < 0$ (signos opuestos), existe al menos un $c \in (a,b)$ tal que $f(c) = 0$.

### Algoritmo (5 pasos del profe)

1. Escoger $[a, b]$ con $f(a) \cdot f(b) < 0$ (garantiza al menos una raíz).
2. Calcular el punto medio: $c = \dfrac{a + b}{2}$
3. Evaluar $f(c)$: si $f(c) = 0$, parar.
4. Si $f(a) \cdot f(c) < 0$, la raíz está en $[a, c]$ → actualizar $b = c$.
5. Si $f(b) \cdot f(c) < 0$, la raíz está en $[c, b]$ → actualizar $a = c$.

### Cota teórica del error
Después de $n$ iteraciones:

$$|e_n| \leq \frac{b - a}{2^{n+1}}$$

### Código oficial (pg. 9–10)

```python
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

def biseccion(f, a, b, iteraciones=100, tolerancia=1e-6, precision=5):
    if f(a) * f(b) >= 0:
        raise ValueError("La función debe tener signos opuestos en los extremos del intervalo [a, b].")
    results = []
    for i in range(iteraciones):
        c = round((a + b) / 2.0, precision)
        fc = round(f(c), precision)
        results.append([i+1, a, b, c, fc])
        print(tabulate(results, headers=["Iteración", "a", "b", "c", "f(c)"], tablefmt="grid"))
        if abs(fc) < tolerancia or (b - a) / 2.0 < tolerancia:
            return c
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    raise ValueError("El método no convergió o faltan iteraciones.")
```

> **Convención crítica**: `precision=5` con `round()` — replicar en el "Modo libro" para que los valores matcheen exacto.

---

## 7. Método del punto fijo (pg. 11–12)

### Definición
Un punto fijo de $g$ es un $x^*$ tal que $g(x^*) = x^*$. Resolver $f(x) = 0$ se reformula como $x = g(x)$.

### Convergencia iterativa
Partiendo de $x_0 \in X$, la sucesión $x_{n+1} = g(x_n)$ converge al punto fijo $x^*$:

$$\lim_{n \to \infty} x_n = x^*$$

### Para aplicar Banach
- $g: X \to X$ debe ser contractiva
- $X$ debe ser completo

El punto fijo de $g$ es la solución buscada.

### Casos de convergencia (Figura 5)
- **Monótona** (staircase): cuando $0 < g'(x^*) < 1$, la iteración se acerca en escalones en el mismo sentido.
- **Oscilante** (telaraña/cobweb): cuando $-1 < g'(x^*) < 0$, la iteración alterna de lado al acercarse.

### Código oficial (pg. 12)

```python
import math

def f(x):
    return math.cos(x)

def g(x):
    return math.cos(x) + x  # reformulación: 0 = cos(x)  →  x = cos(x) + x

def fixed_point_iteration(x0, tol=1e-5, max_iter=100):
    x = x0
    iter_values = [x0]
    for i in range(max_iter):
        x_new = g(x)
        iter_values.append(x_new)
        if abs(x_new - x) < tol:
            print("Tolerance exceeded...")
            break
        x = x_new
    return x_new, iter_values

x0 = 1.0
root, iter_values = fixed_point_iteration(x0)
```

---

## 8. Método de aceleración Aitken (pg. 13–14)

### Fórmula central
$$x^*_n = x_{(n)} - \frac{(x_{n+1} - x_n)^2}{x_{n+2} - 2 x_{n+1} + x_n}$$

Donde:
- $x_{(n)}$: término de sucesión original
- $x^*_n$: término de sucesión acelerado

Requiere 3 términos consecutivos para iniciar la extrapolación.

### Condiciones de convergencia
- Secuencia convergente (no diverge ni oscila significativamente)
- Disponibilidad de 3 términos consecutivos
- Denominador no nulo
- Más útil cuando la secuencia converge lentamente
- Estabilidad numérica (cuidado con errores de cancelación en el denominador)

---

## 9. Método de Newton-Raphson (pg. 15–16)

### Fórmula iterativa
$$x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}$$

### Ventajas
- Convergencia rápida (cuadrática) si el punto inicial está cerca de la raíz
- Simple de implementar

### Limitaciones
- Puede fallar o divergir si $f'(x) \approx 0$ o si el punto inicial está lejos
- Requiere conocer (o aproximar) la derivada de $f$

### Código oficial (pg. 16)

```python
from scipy.misc import derivative

def newton_raphson(f, valor_inicial, iteraciones=100, tolerancia=1e-6, precision=5):
    x = valor_inicial
    results = []
    for i in range(iteraciones):
        fx = round(f(x), precision)
        dfx = round(derivative(f, x, dx=tolerancia), precision)
        if dfx == 0:
            raise ValueError("La derivada es cero. El método no puede continuar.")
        x_new = round(x - fx / dfx, precision)
        results.append([i+1, x, fx, dfx, x_new])
        if abs(x_new - x) < tolerancia:
            return x_new
        x = x_new
    raise ValueError("El método no convergió o faltan iteraciones.")
```

---

## 10. Banco de ejercicios (pg. 17–19)

### Bisección
1. Hallar intervalos $[a,b]$ con $f(a) \cdot f(b) < 0$:
   - a) $f(x) = e^x - 2 - x$
   - b) $f(x) = \cos(x) + x$
   - c) $f(x) = \ln(x) - 5 - x$
   - d) $f(x) = x^2 - 10x + 23$
2. $f(x) = 3(x+1)(x - \tfrac{1}{2})(x-1)$ en:
   - a) $[-1, 1.5]$  b) $[-1.25, 2.5]$
3. Bisección con tolerancia $10^{-3}$:
   - a) $\sqrt{x} - \cos(x) = 0$, $[0, 1]$
   - b) $x - 2^{-x} = 0$, $[0, 1]$
   - c) $e^x - x^2 + 3x - 2 = 0$, $[0, 1]$
   - d) $2x \cos(x) - (x+1)^2 = 0$, $[-3, -2]$ y $[-1, 0]$
   - e) $x \cos(x) - 2x^2 + 3x - 1 = 0$, $[0.2, 0.3]$ y $[1.2, 1.3]$
4. Bisección tolerancia $10^{-2}$, $p(x) = x^4 - 2x^3 - 4x^2 + 4x + 4$:
   - a) $[-2, -1]$  b) $[0, 2]$  c) $[2, 3]$  d) $[-1, 0]$
5. ¿A qué cero converge? $f(x) = (x+2)(x+1)(x-1)^3(x-2)$:
   - a) $[-3, 2.5]$  b) $[-2.5, 3]$  c) $[-1.75, 1.5]$  d) $[-1.5, 1.75]$

### Punto Fijo
1. $f(x) = 2e^{x^2} - 5x$, $x^* \in [0,1]$, $x_0 = 0$
2. $f(x) = \cos(x)$, $x^* \in [1,2]$, $x_0 = 1$
3. $f(x) = e^{-x} - x$, $x^* \in [0,1]$, $x_0 = 0$
4. $f(x) = x^3 - x - 1$, $x^* \in [1,2]$, $x_0 = 1$
5. $f(x) = \pi + 0.5 \sin(x/2) - x$, $x^* \in [0, 2\pi]$, $x_0 = 0$
6. Demostrar que tienen punto fijo en $p$ cuando $f(p)=0$ con $f(x) = x^4 + 2x^2 - x - 3$:
   - a) $g(x) = (3 + x - 2x^2)^{1/4}$
   - b) $g(x) = \left(\tfrac{x + 3 - x^4}{2}\right)^{1/2}$
7. Demostrar que $g(x) = 2^{-x}$ tiene punto fijo en $[\tfrac{1}{3}, 1]$
8. Convertir $\sqrt{3}$ con exactitud $10^{-4}$ usando punto fijo
9. ¿En qué intervalo $[a,b]$ converge con $10^{-3}$ para $x = \tfrac{5}{x^2} + 2$?
10. Hallar intervalo donde $g(x) = \sqrt{e^x / 3}$ tiene punto fijo

### Aitken
1. $f(x) = \tfrac{\pi}{2}x^2 - x - 2$, $x_0 = 1.4$ (hallar $g(x)$)
2. $f(x) = \cos(x) - x$, $x_0 = 0.5$ (hallar $g(x)$)
3. $g(x) = \sqrt[3]{3x^2 - 4x + 1}$, $x_0 = 0.3$
4. $g(x) = e^{-x}$, $x_0 = 1$
5. $g(x) = \sqrt{3x - 2}$, $x_0 = 2$
6. $g(x) = \ln(x+1)$, $x_0 = 0.5$
7. $g(x) = 1 - x^3$, $x_0 = 0.5$
8. $g(x) = \tfrac{1}{2}(x^2 - 3)$, $x_0 = 0.5$
9. $g(x) = \tfrac{\sin(x) + 5}{x^2}$, $x_0 = 2$
10. $g(x) = x^2$, $x_0 = 0.4, 0.9, 1.5$
11. $g(x) = \tfrac{3}{2}x + \tfrac{1}{x^2}$, $x_0 = 0.25$

### Newton-Raphson
1. $f(x) = (x-1)^2$, $x_0 = 0$
2. $f(x) = x^3 - 2x - 5$, $x_0 = 1.5$
3. $f(x) = x^5 - x - 1$, $x_0 = 1$
4. Aproximar $\sqrt[6]{2}$ con precisión de 8 cifras
5. $f(x) = e^x + x^2 - 4$, $x_0 = 0.5$
6. $f(x) = x^2 - 3x - 4$, $x_0 = 8$
7. $f(x) = \ln(x) - 1$, $x_0 = 2$
8. $f(x) = x^4 - 16$, $x_0 = 2$
9. $f(x) = x^3 - 2x + 1$, $x_0 = -1.5$
10. $f(x) = e^{3x} - 4$, $x_0 = 0$
11. $f(x) = x^2 - 2x + 1$, $x_0 = 0$
12. $f(x) = xe^{-x}$, $x_0 = -1$

---

## 11. Convenciones críticas para la UI

1. **Precisión**: `round(valor, 5)` en el "Modo libro" para matchear exactamente los valores del profe.
2. **Los 4 criterios de detención** deben estar como toggles combinables.
3. **Cotas teóricas** (Bolzano/Bisección, Banach/Punto Fijo) mostradas junto al error real.
4. **Reformulación de g(x)**: el asistente debe chequear $\lvert g'(x) \rvert < 1$ en **todo el intervalo**, no solo en $x_0$ (Lipschitz uniforme en conjuntos compactos).
5. **Condiciones de pre-cheque**:
   - Bisección: continuidad + $f(a) \cdot f(b) < 0$
   - Punto Fijo: $g(X) \subseteq X$ + $\max_{X} \lvert g' \rvert < 1$
   - NR: $f'(x_0) \neq 0$, monitoreo de $f'$ en cada iteración
6. **Banco de ejercicios del profe** como presets en cada submódulo.
