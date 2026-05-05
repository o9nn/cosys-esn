# Reservoir Computing: Theory and Practice

## 1. Echo State Networks (ESNs)

Echo State Networks are a class of **recurrent neural networks** introduced by Jaeger (2001) that exploit a randomly-connected, fixed recurrent layer (the *reservoir*) to project input signals into a high-dimensional space. Only the output (readout) weights are trained, making them computationally cheap compared to fully-trained RNNs.

### Core Architecture

```
u(t) ──[W_in]──► ┌──────────────────┐ ──[W_out]──► y(t)
                  │    Reservoir     │
                  │  (fixed weights) │
                  └────────W─────────┘
                        ↑        │
                        └────────┘
                        recurrent
```

The reservoir state update equation:

```
x(t+1) = (1 - α) · x(t) + α · f(W_in · u(t) + W · x(t))
```

where:
- `x(t)` — reservoir state vector of dimension `N`
- `u(t)` — input vector of dimension `K`
- `α` — leak rate (0 < α ≤ 1)
- `f(·)` — element-wise nonlinearity (typically `tanh`)
- `W_in` — input weight matrix `(N × K)`, randomly initialised and **fixed**
- `W` — recurrent weight matrix `(N × N)`, sparse and **fixed**
- `W_out` — output weight matrix `(n_outputs × (N+1))`, **trained**

The output is a linear combination of reservoir states with an optional bias:

```
y(t) = W_out · [x(t); 1]
```

---

## 2. Spectral Radius and the Edge of Chaos

The **spectral radius** ρ(W) is the largest absolute eigenvalue of the recurrent weight matrix W:

```
ρ(W) = max |λ_i(W)|
```

### Why it matters

The spectral radius controls the *fading memory* of the reservoir:

- **ρ < 1**: Reservoir activity decays over time. Short memory, stable but limited.
- **ρ ≈ 1 (edge of chaos)**: Long memory, rich dynamics. Optimal for most tasks.
- **ρ > 1**: Reservoir may explode (grow without bound). Unstable without leak rate.

### Scaling the spectral radius

In COSYS-ESN, all reservoir weight matrices are post-scaled to exactly hit the target spectral radius:

```python
W_scaled = W * (target_sr / current_sr)
```

### Rule of thumb

For tasks requiring long-range temporal dependencies (e.g., Mackey-Glass prediction):
- Use `spectral_radius ≈ 0.9–0.99`
- Combine with `leak_rate ≈ 0.1–0.4`

---

## 3. Ridge Regression Training

Only the output weights `W_out` are trained. The training procedure:

1. **Warm up** the reservoir on the first `washout` inputs (discard states).
2. **Collect** reservoir states `X = [x(washout+1), ..., x(T)]` as rows.
3. **Augment** with bias: `X̃ = [X | 1]` — shape `(T-washout, N+1)`.
4. **Solve** the ridge regression:

```
W_out = (X̃ᵀ X̃ + λI)⁻¹ X̃ᵀ Y
```

where λ is the regularisation parameter (`ridge_lambda`).

### Choosing λ

| λ | Effect |
|---|--------|
| `1e-8` to `1e-6` | Very little regularisation; risk of overfitting |
| `1e-4` to `1e-2` | Moderate regularisation; good default |
| `1.0+` | Strong regularisation; underfitting |

In COSYS-ESN:
```python
config = ESNConfig(ridge_lambda=1e-6)
```

---

## 4. Memory Capacity

The **memory capacity (MC)** measures how well the reservoir can recall past inputs:

```
MC = Σ_{k=1}^{∞} r²(u(t-k), y_k(t))
```

where `r²` is the squared correlation between delayed input `u(t-k)` and the reservoir's best linear reconstruction `y_k(t)`. For a linear reservoir with `N` neurons, `MC ≤ N`.

Practical estimation in COSYS-ESN via `ReservoirAutognosis`:

```python
from reservoir_core.autognosis.autognosis import ReservoirAutognosis
ag = ReservoirAutognosis(esn)
cap = ag.capacity_estimation()
```

---

## 5. The Echo State Property

The **echo state property (ESP)** guarantees that any two reservoir trajectories with identical inputs eventually converge, regardless of initial conditions. Formally:

> For any bounded input sequence, the reservoir state becomes uniquely determined by the input history (not the initial state).

The ESP holds when ρ(W) < 1 (a sufficient but not necessary condition). With leak rate α:

```
Effective spectral radius ≈ (1 - α) + α · ρ(W)
```

For the ESP to hold in practice:
- `ρ(W) < 1 / (1 - α)` roughly

### Verifying ESP experimentally

```python
# Two runs with same inputs but different initial states → states converge
esn.state = np.zeros(N)
for u in input_seq:
    esn.update(u)
state1 = esn.state.copy()

esn.state = np.random.randn(N) * 0.01
for u in input_seq:
    esn.update(u)
state2 = esn.state.copy()

np.testing.assert_allclose(state1, state2, atol=0.1)  # Should pass
```

---

## 6. Mackey-Glass Benchmark

The **Mackey-Glass equation** is the standard chaotic time series benchmark for reservoir computing:

```
dx/dt = β · x(t-τ) / (1 + x(t-τ)^n) - γ · x(t)
```

Standard parameters: β=0.2, γ=0.1, n=10, τ=17. With τ≥17, the system is chaotic.

### Evaluation metric: NRMSE

```
NRMSE = RMSE / std(target)
```

Target: NRMSE < 0.1 is considered *excellent*. A well-configured ESN achieves NRMSE ≈ 0.01–0.05.

---

## 7. Usage Examples

### Basic Training and Prediction

```python
import numpy as np
from models.echo_state_network import EchoStateNetwork, ESNConfig

config = ESNConfig(
    n_inputs=5,
    n_reservoir=200,
    n_outputs=1,
    spectral_radius=0.95,
    leak_rate=0.3,
    sparsity=0.1,
    ridge_lambda=1e-6,
    washout=100,
    random_seed=42,
)
esn = EchoStateNetwork(config)

# Train
X_train = np.random.randn(500, 5)   # (n_samples, n_inputs)
y_train = np.random.randn(500, 1)   # (n_samples, n_outputs)
esn.train(X_train, y_train)

# Predict on new data
X_test = np.random.randn(100, 5)
predictions = esn.run(X_test)       # (100, 1)
```

### Streaming (step-by-step)

```python
for u in input_stream:
    prediction = esn.predict(u)  # updates state and returns output
```

### Save and Load

```python
esn.save("model.npz")
esn2 = EchoStateNetwork.load("model.npz", config)
```

### Using Different Topologies

```python
# Small-world reservoir (higher clustering than random)
config = ESNConfig(topology="small_world", ...)

# Scale-free reservoir (power-law degree distribution)
config = ESNConfig(topology="scale_free", ...)
```

### Mackey-Glass Full Example

```python
import numpy as np
from models.echo_state_network import EchoStateNetwork, ESNConfig

def generate_mackey_glass(n=2000, tau=17):
    x = np.zeros(n)
    x[0] = 1.2
    for t in range(tau, n - 1):
        x[t+1] = x[t] + 0.2*x[t-tau]/(1+x[t-tau]**10) - 0.1*x[t]
    return x

mg = generate_mackey_glass()
X = np.array([mg[i:i+5] for i in range(len(mg)-6)])
y = mg[5:].reshape(-1, 1)

config = ESNConfig(n_inputs=5, n_reservoir=200, n_outputs=1,
                   spectral_radius=0.95, leak_rate=0.3, washout=100,
                   ridge_lambda=1e-6, random_seed=42)
esn = EchoStateNetwork(config)
esn.train(X[:1000], y[:1000])
preds = esn.run(X[1000:])

nrmse = np.sqrt(np.mean((preds - y[1000:])**2)) / np.std(y[1000:])
print(f"NRMSE: {nrmse:.4f}")  # Typically < 0.05
```

---

## References

- Jaeger, H. (2001). *The "echo state" approach to analysing and training recurrent neural networks*. GMD Technical Report 148.
- Lukoševičius, M., & Jaeger, H. (2009). Reservoir computing approaches to recurrent neural network training. *Computer Science Review*, 3(3), 127–149.
- Maass, W., Natschläger, T., & Markram, H. (2002). Real-time computing without stable states. *Neural Computation*, 14(11), 2531–2560.
