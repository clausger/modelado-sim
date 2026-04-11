# Método de Monte Carlo — Teoría

> Referencia teórica oficial de la materia **Modelado y Simulación** (UADE).
> Docente: Ing. Omar Cáceres.
> Este archivo es la base de conocimiento para todos los ejercicios de Montecarlo.

---

## 1. Definición

El **método de Monte Carlo** es una técnica numérica que utiliza **muestreo aleatorio** y **probabilidad** para resolver problemas complejos que serían difíciles o imposibles de abordar de forma analítica.

La idea central es:

> Reemplazar un cálculo determinístico por un **experimento estadístico**: generar muchas muestras aleatorias, observar qué fracción cumple cierta condición, y usar esa fracción para estimar la cantidad buscada.

A mayor cantidad de muestras, mejor es la aproximación (ver *Ley de los Grandes Números*, sección 4).

---

## 2. Ejemplo clásico: estimación de π

Considerar un cuadrado de lado 2 centrado en el origen, con un círculo de radio 1 inscripto.

- Área del cuadrado: `A_cuadrado = 2 · 2 = 4`
- Área del círculo:  `A_circulo = π · 1² = π`

La razón entre ambas áreas es:

```
A_circulo / A_cuadrado = π / 4
```

Si generamos `n` puntos aleatorios uniformemente distribuidos dentro del cuadrado y contamos cuántos caen dentro del círculo (`n_dentro`), por la Ley de los Grandes Números:

```
n_dentro / n  ≈  π / 4
```

Despejando:

```
π ≈ 4 · (n_dentro / n)
```

Cuanto más grande sea `n`, más precisa es la estimación.

---

## 3. Los 4 pasos del método de Monte Carlo

Todo problema de Monte Carlo sigue el mismo esquema:

### Paso 1 — Definir el dominio
Identificar el espacio sobre el cual se va a muestrear (un intervalo, un rectángulo, un volumen, etc.).

### Paso 2 — Generar muestras aleatorias
Generar `n` puntos aleatorios dentro del dominio usando una distribución uniforme (salvo que el problema indique lo contrario).

### Paso 3 — Evaluar y contar
Evaluar cada muestra con la función o condición del problema. Contar (o acumular) los resultados que cumplen la condición.

### Paso 4 — Calcular la estimación
Usar la proporción o el promedio obtenido para estimar la cantidad buscada (área, integral, probabilidad, etc.).

---

## 4. Ley de los Grandes Números (LGN)

El método de Monte Carlo se apoya formalmente en la **Ley de los Grandes Números**, que establece que el promedio muestral de una variable aleatoria converge al valor esperado cuando el número de muestras tiende a infinito:

```
lim (n → ∞)   (1/n) · Σᵢ f(xᵢ)   =   E[f(x)]
```

Donde:
- `xᵢ` son muestras aleatorias del dominio.
- `f(xᵢ)` es la evaluación de la función en esa muestra.
- `E[f(x)]` es el valor esperado (promedio teórico).

**Consecuencia práctica:** a más muestras, menor el error de estimación (y el error decrece proporcional a `1/√n`).

---

## 5. Monte Carlo para integración

### 5.1 Integración en una dimensión (1D)

Para estimar la integral definida de una función `f(x)` en el intervalo `[a, b]`:

```
Î = (b - a) · (1/n) · Σᵢ f(xᵢ)
```

Donde `xᵢ` son `n` puntos aleatorios uniformemente distribuidos en `[a, b]`.

**Interpretación:** el promedio de `f(xᵢ)` estima el valor medio de la función en el intervalo, y se multiplica por el ancho del dominio `(b - a)` para obtener el área.

### 5.2 Integración en dos dimensiones (2D)

Para estimar una integral doble de `f(x, y)` sobre el rectángulo `[a, b] × [c, d]`:

```
Î = (b - a)(d - c) · (1/n) · Σᵢ f(xᵢ, yᵢ)
```

Donde `(xᵢ, yᵢ)` son `n` puntos aleatorios uniformemente distribuidos en el rectángulo.

**Generalización:** para dimensiones superiores, se multiplica por el volumen del dominio y se promedian las evaluaciones.

---

## 6. Intervalo de confianza

La estimación `Î` es una variable aleatoria; por lo tanto, además del valor puntual debemos reportar un **intervalo de confianza** que cuantifique la incertidumbre:

```
IC = Î ± z_(α/2) · (σ / √n)
```

Donde:
- `Î` es la estimación (el promedio obtenido).
- `σ` es el **desvío estándar muestral** de las evaluaciones de la función (ver sección 6.2).
- `n` es el número de muestras.
- `z_(α/2)` es el valor crítico de la distribución normal estándar para el nivel de confianza deseado.

### 6.1 Valores de z más usados

| Nivel de confianza | z_(α/2) |
|---|---|
| 90 % | 1.645 |
| 95 % | 1.960 |
| 99 % | 2.576 |

### 6.2 CORRECCIÓN CRÍTICA — Cálculo del desvío estándar

> ⚠️ **Importante:** en las diapositivas del curso aparece una fórmula con un error de notación. La fórmula correcta es la siguiente.

El desvío estándar debe calcularse sobre las **evaluaciones de la función** `f(xᵢ)`, **no** sobre los puntos del dominio `xᵢ`. Es decir, se usa la **media de los `f(xᵢ)`**, denotada `f̄`:

```
σ = √(  (1 / (n - 1)) · Σᵢ ( f(xᵢ) - f̄ )²  )
```

Donde:

```
f̄ = (1/n) · Σᵢ f(xᵢ)
```

**Por qué:** lo que estamos promediando en Monte Carlo son los valores `f(xᵢ)`, no los `xᵢ`. La dispersión relevante es cuánto varían esas evaluaciones alrededor de su propio promedio `f̄`. Usar `x̄` (la media de las posiciones) no tiene sentido estadístico en este contexto.

---

## 7. Herramientas en Python

### 7.1 Librerías estándar

```python
import random
import math
```

### 7.2 Generación de números aleatorios

- `random.uniform(a, b)` → genera un número real aleatorio uniformemente distribuido en `[a, b]`.
- `random.seed(42)` → fija la semilla del generador para obtener resultados **reproducibles**.

> 🎯 **Convención de cátedra:** siempre usar `random.seed(42)` al inicio de los ejercicios, salvo indicación contraria. Esto permite que todos los alumnos obtengan los mismos resultados y facilita la corrección.

### 7.3 Implementación completa — estimación de π

```python
import random
import math

random.seed(42)  # convención de cátedra

n = 10_000
n_dentro = 0

for _ in range(n):
    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)
    if x**2 + y**2 <= 1:
        n_dentro += 1

pi_estimado = 4 * n_dentro / n
error = abs(pi_estimado - math.pi)

print(f"π estimado: {pi_estimado}")
print(f"π real:     {math.pi}")
print(f"error abs:  {error}")
```

**Salida esperada (con seed 42 y n=10 000):** un valor cercano a `3.14xx` con un error del orden de `10⁻²` a `10⁻³`.

---

## 8. Comparación 1D vs 2D

| Aspecto | Monte Carlo 1D | Monte Carlo 2D |
|---|---|---|
| Dominio | Intervalo `[a, b]` | Rectángulo `[a, b] × [c, d]` |
| Muestra | `xᵢ ∈ [a, b]` | `(xᵢ, yᵢ) ∈ [a,b] × [c,d]` |
| Factor geométrico | `(b - a)` | `(b - a)(d - c)` |
| Estimador | `Î = (b-a) · (1/n) · Σ f(xᵢ)` | `Î = (b-a)(d-c) · (1/n) · Σ f(xᵢ, yᵢ)` |
| Ventaja frente a métodos clásicos | Similar a Newton-Cotes | **Mucho mejor** en dimensiones altas (la complejidad no explota con la dimensión) |
| Convergencia | `O(1/√n)` | `O(1/√n)` (¡independiente de la dimensión!) |

> 💡 **Clave:** la gran ventaja del método de Monte Carlo es que su **tasa de convergencia no depende de la dimensión del problema**. Por eso es el método de elección en integrales multidimensionales, simulaciones físicas, finanzas cuantitativas, etc.

---

## 9. Resumen operativo

Para cualquier ejercicio de Monte Carlo de la materia, seguir siempre este esquema:

1. **Definir el dominio** de muestreo.
2. **Fijar la semilla** con `random.seed(42)`.
3. **Generar `n` muestras uniformes** con `random.uniform(...)`.
4. **Evaluar la función** en cada muestra y acumular en un array/lista.
5. **Calcular el estimador** `Î` con la fórmula correspondiente (1D, 2D, etc.).
6. **Calcular `f̄`** (media de las evaluaciones).
7. **Calcular `σ`** usando la fórmula corregida de la sección 6.2.
8. **Construir el intervalo de confianza** con el `z` del nivel pedido (por defecto 95 % → 1.960).
9. **Reportar:** valor estimado, error absoluto, error relativo (si se conoce el valor real) y el IC.
