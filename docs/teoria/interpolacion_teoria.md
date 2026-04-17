# Interpolación y Derivación Numérica — Cátedra Cáceres

> Referencia canónica destilada del PDF oficial (pg. 20–26). Base para los módulos
> de Lagrange (interpolación) y Diferencias Finitas (derivación numérica).

---

## 1. Polinomio interpolante de Lagrange (pg. 20)

### Fórmula central

$$P(x) = \sum_{i=0}^{n} y_i \, L_i(x), \qquad L_i(x) = \prod_{\substack{j=0 \\ j \neq i}}^{n} \frac{x - x_j}{x_i - x_j}$$

- $P(x)$: polinomio interpolante de grado $\le n$.
- $L_i(x)$: base de Lagrange asociada al nodo $x_i$. Cumple $L_i(x_i) = 1$ y $L_i(x_j) = 0$ si $j \ne i$.
- Pasa exactamente por todos los puntos $(x_i, y_i)$.

### Existencia y unicidad

Para $(n+1)$ puntos con abscisas distintas, existe un **único** polinomio de grado $\le n$
que pasa por todos ellos (teorema fundamental de la interpolación).

### Interpolación vs extrapolación

- **Interpolación**: evaluar $P(x)$ dentro del rango $[x_0, x_n]$ — confiable.
- **Extrapolación**: evaluar fuera del rango — peligroso, el error crece rápidamente
  porque $|\prod (x - x_i)|$ explota fuera del intervalo (Figura 9).

---

## 2. Error de interpolación (pg. 21–22)

### Error exacto

Si $f$ es $(n+1)$ veces derivable en $[x_0, x_n]$, existe $\xi \in [x_0, x_n]$ tal que:

$$f(x) - P(x) = \frac{f^{(n+1)}(\xi)}{(n+1)!} \prod_{i=0}^{n} (x - x_i)$$

### Cota de error (uso práctico)

$$|E(x)| \le \frac{M_{n+1}}{(n+1)!} \left| \prod_{i=0}^{n} (x - x_i) \right|$$

con $M_{n+1} = \max_{\xi \in [x_0, x_n]} \left| f^{(n+1)}(\xi) \right|$.

### Error local (verificación puntual)

$$|E(x)| = |f(x) - P(x)|$$

### Pasos del libro para calcular errores (pg. 22)

1. Construir el polinomio interpolante $P(x)$.
2. Calcular las bases $L_i(x)$.
3. Determinar $M_{n+1} = \max |f^{(n+1)}(\xi)|$ en $[x_0, x_n]$.
4. Aplicar la cota.
5. Verificar con el valor real (error local).

---

## 3. Código oficial de Lagrange (pg. 23)

```python
import numpy as np
import matplotlib.pyplot as plt

def polinomio_lagrange(x, x_puntos, y_puntos):
    n = len(x_puntos)
    L = 0
    for i in range(n):
        li = 1
        for j in range(n):
            if i != j:
                li *= (x - x_puntos[j]) / (x_puntos[i] - x_puntos[j])
        L += y_puntos[i] * li
    return L

def reconstruccion_lagrange(x_puntos, y_puntos):
    n = len(x_puntos)
    coeficientes = np.zeros(n)
    for i in range(n):
        li = np.poly1d([1])
        for j in range(n):
            if i != j:
                li *= np.poly1d([1, -x_puntos[j]]) / (x_puntos[i] - x_puntos[j])
        coeficientes += y_puntos[i] * li.coefficients
    return coeficientes
```

---

## 4. Diferencias finitas (pg. 24)

Aproximan derivadas usando evaluaciones de $f$ en nodos equiespaciados, con paso $h$.

### Progresivas (forward)

$$f'(x_i) \approx \frac{f(x_{i+1}) - f(x_i)}{h}, \qquad
f''(x_i) \approx \frac{f(x_{i+2}) - 2 f(x_{i+1}) + f(x_i)}{h^2}$$

### Regresivas (backward)

$$f'(x_i) \approx \frac{f(x_i) - f(x_{i-1})}{h}, \qquad
f''(x_i) \approx \frac{f(x_i) - 2 f(x_{i-1}) + f(x_{i-2})}{h^2}$$

### Centrales (central) — **más precisas**

$$f'(x_i) \approx \frac{f(x_{i+1}) - f(x_{i-1})}{2h}, \qquad
f''(x_i) \approx \frac{f(x_{i+1}) - 2 f(x_i) + f(x_{i-1})}{h^2}$$

### Orden de error

- Progresiva/regresiva de 1er orden: error $O(h)$.
- Central de 1er orden: error $O(h^2)$.
- Central de 2do orden (2da derivada): error $O(h^2)$.

**Regla práctica (pg. 26)**: usar **centrales** en todos los puntos interiores; en los
extremos, usar progresivas (izquierdo) o regresivas (derecho) porque no hay vecino
del lado faltante.

### Tradeoff truncamiento vs redondeo

- $h$ grande: error de truncamiento alto.
- $h$ chico: error de redondeo alto (restás números casi iguales).
- Óptimo práctico: $h \approx 10^{-4}$ a $10^{-6}$ para `float64`.

---

## 5. Código oficial de diferencias finitas (pg. 25)

```python
def primera_derivada(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

def segunda_derivada(f, x, h):
    return (f(x + h) - 2*f(x) + f(x - h)) / (h**2)
```

---

## 6. Banco de ejercicios Lagrange (pg. 25–26)

1. Puntos $(1,1)$, $(2,4)$, $(3,9)$ (polinomio cuadrático clásico — reconstruye $x^2$).
2. $(0,1)$, $(1,3)$, $(2,2)$, $(3,5)$.
3. Dado $x = [0,1,2,3,4]$, $y = [1,2,b,2,3]$ — hallar $b$ (parámetro).
4. $x = [0,1,2,3,4]$, $f(x) = [1,2,0,2,3]$.
5. $x = [0,1,2]$, $y = [1,3,0]$.
6. Polinomio grado 2 por $x = [1,2,3]$, $y = [10, 15, 80]$.
7. $x = [2,4,5]$, $f(x) = [5,6,3]$.
8. $x = [-2, 0, 2]$, $f(x) = [0, 1, 0]$.
9. Aproximar $f(x) = \sin(x)$ en $[0, \pi]$ con polinomio grado 2.
10. $x = [0,1,2]$, $f(x) = [1,2,7]$.
11. Nodos $x_0=2, x_1=2.5, x_2=4.5$ para aproximar $f(x) = 1/x$.
12. $f(x) = 2\sin(\pi x/6)$, nodos $1,2,3$, grado 2 — aproximar $f(4)$ y $f(1.5)$.
13. Nodos $x_0=0, x_1=0.6, x_2=0.9$ — aproximar $f(0.45)$ para:
    - a) $f(x) = \cos(x)$
    - b) $f(x) = \sqrt{x+1}$
    - c) $f(x) = \ln(x+1)$

---

## 7. Banco de ejercicios Diferencias Finitas (pg. 26)

1. Central para $f(x) = \sin(x)$ en $x = [0, 0.1, ..., 0.5]$, $h = 0.1$.
2. Central para $f(x) = e^x$ en $x = [0, 0.1, ..., 0.5]$, $h = 0.1$.
3. $f(x) = x^3 - x$, derivada 1ra y 2da en $x=1$, $h=0.1$.
4. $f(x) = e^x \sin(x)$, en $x=1$ con $h = 0.01$:
   - a) $f'(1)$ con central
   - b) Error absoluto
   - c) $f''(1)$ con central
5. Comparar progresiva, regresiva y central de 2do orden para $f(x) = e^{-2x} - x$ en $x = 2$.
6. Tabla $t$ vs $x(m)$ — calcular $v = dx/dt$ y $a = d^2x/dt^2$ (centrales en interiores, progresiva/regresiva en extremos).
7. Ídem con otra tabla — analizar comportamiento de velocidad y aceleración.

---

## 8. Convenciones críticas para la UI

1. **Precisión**: `round(·, 5)` en "Modo libro" para matchear tablas del profe.
2. **Plot siempre** con puntos originales + polinomio + región de interpolación vs extrapolación.
3. **Bases de Lagrange** deben poder mostrarse individualmente — es clave pedagógicamente.
4. **Cota de error**: solo tiene sentido si hay una $f$ original (no solo puntos).
5. **Diferencias finitas**: default centrales en interiores, adelante/atrás en extremos.
