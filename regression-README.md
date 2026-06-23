# Regression Formulas: Linear, Polynomial, and Multiple Regression

These notes summarize three important regression formulas:

1. Simple Linear Regression
2. Polynomial Regression
3. Multiple Regression

The key idea is that regression tries to predict an output value based on one or more input values.

---

## 1. Simple Linear Regression

Simple linear regression uses **one input feature** and fits a **straight line**.

### Formula

```text
ŷ = β₀ + β₁x
```

### Meaning

| Symbol | Meaning |
|---|---|
| `ŷ` | predicted value |
| `x` | input feature |
| `β₀` | intercept |
| `β₁` | slope |

### Intuition

This says:

> The prediction changes in a straight-line way as `x` changes.

### Example

```text
predicted CPU usage = 10 + 0.8 × request_rate
```

If request rate increases, CPU usage increases in a linear pattern.

### Shape

In 2D, simple linear regression looks like a **straight line**.

```text
y
│
│        •
│     •     •
│   •   ───────  best-fit line
│ •
└──────────────── x
```

---

## 2. Polynomial Regression

Polynomial regression still may use **one input feature**, but it allows the relationship to **curve**.

The curve comes from powers such as:

```text
x², x³, x⁴, ...
```

### Degree 2 Formula

```text
ŷ = β₀ + β₁x + β₂x²
```

This is also called **quadratic regression**.

### Degree 3 Formula

```text
ŷ = β₀ + β₁x + β₂x² + β₃x³
```

This is also called **cubic regression**.

### General Polynomial Formula

```text
ŷ = β₀ + β₁x + β₂x² + β₃x³ + ... + βₙxⁿ
```

### Meaning of Degree

The **degree** is the highest power of `x`.

| Formula | Degree | Shape |
|---|---:|---|
| `ŷ = β₀ + β₁x` | 1 | straight line |
| `ŷ = β₀ + β₁x + β₂x²` | 2 | curve |
| `ŷ = β₀ + β₁x + β₂x² + β₃x³` | 3 | more flexible curve |

### Intuition

Polynomial regression says:

> The prediction may not grow at the same speed forever.  
> The relationship can bend.

### Example

```text
predicted latency = 20 + 1.5 × request_rate + 0.03 × request_rate²
```

This means latency may increase slowly at first, then faster when traffic becomes heavier.

### Shape

In 2D, polynomial regression can look like a **curve**.

```text
y
│              •
│          •
│      •
│   •
│ •      curved best-fit line
└──────────────── x
```

---

## 3. Multiple Regression

Multiple regression uses **more than one input feature**.

It does not necessarily mean squared terms.  
It means several variables are used together.

### Formula

```text
ŷ = β₀ + β₁x₁ + β₂x₂ + β₃x₃ + ... + βₚxₚ
```

### Meaning

| Symbol | Meaning |
|---|---|
| `ŷ` | predicted value |
| `x₁, x₂, x₃, ..., xₚ` | input features |
| `β₀` | intercept |
| `β₁, β₂, β₃, ..., βₚ` | coefficients/slopes for each input |

### Intuition

This says:

> The prediction depends on several things at the same time.

### Example

```text
predicted response time =
    β₀
  + β₁ × request_rate
  + β₂ × payload_size
  + β₃ × cache_miss_rate
```

So response time may depend on:

- how many requests arrive per second
- how large each request is
- how often the cache misses

### Shape

With two input features, multiple regression creates a **plane** in 3D.

```text
z = predicted value

        z
        │
        │      / plane
        │    /
        │  /
        └────────── x₁
       /
      x₂
```

---

## Important Difference: Degree vs Number of Variables

This is the most important distinction.

| Regression Type | Main Idea | Example Formula | Shape |
|---|---|---|---|
| Simple linear regression | one input, degree 1 | `ŷ = β₀ + β₁x` | line |
| Polynomial regression | one input, higher degree | `ŷ = β₀ + β₁x + β₂x²` | curve |
| Multiple regression | many inputs, usually degree 1 | `ŷ = β₀ + β₁x₁ + β₂x₂` | plane |
| Multiple polynomial regression | many inputs and higher degree | `ŷ = β₀ + β₁x₁ + β₂x₂ + β₃x₁² + β₄x₂²` | curved surface |

---

## A Simple Cloud Infrastructure Analogy

### Simple Linear Regression

```text
CPU usage depends on request rate.
```

Formula:

```text
CPU = β₀ + β₁ × request_rate
```

One input. Straight-line relationship.

---

### Polynomial Regression

```text
Latency depends on request rate, but it increases faster when traffic becomes heavy.
```

Formula:

```text
latency = β₀ + β₁ × request_rate + β₂ × request_rate²
```

One input. Curved relationship.

---

### Multiple Regression

```text
Response time depends on request rate, payload size, and cache miss rate.
```

Formula:

```text
response_time =
    β₀
  + β₁ × request_rate
  + β₂ × payload_size
  + β₃ × cache_miss_rate
```

Several inputs. Linear relationship in multiple dimensions.

---

## Clean Mental Model

Simple linear regression:

```text
One input → straight line
```

Polynomial regression:

```text
One input → curved line
```

Multiple regression:

```text
Many inputs → plane or higher-dimensional flat surface
```

Multiple polynomial regression:

```text
Many inputs + powers like x² → curved surface
```

---

## Final Intuition

The square term `x²` belongs to the idea of **degree**.

The multiple inputs `x₁, x₂, x₃` belong to the idea of **many variables**.

So:

```text
Polynomial regression is about degree.
Multiple regression is about number of input features.
```

They can also be combined.
