# Membrane Computing and COSYS-ESN

## 1. P-Systems and the Membrane Computing Paradigm

**Membrane computing** (P-systems) is a computational model introduced by Gheorghe Păun (2000), inspired by the structure and functioning of biological cells. The key abstraction is a **hierarchically nested membrane structure** where:

- Each **membrane** encloses a **region** containing objects (multisets of symbols or numerical values).
- **Evolution rules** transform objects within a region.
- Objects can be **transported** through membranes (in/out) or dissolved.
- Membranes form a **tree structure** — inner compartments nested inside outer ones.

### Formal Structure

```
Membrane Hierarchy:
┌─────────────── Skin Membrane (outermost) ──────────────┐
│  ┌─── Membrane 1 ───┐   ┌─── Membrane 2 ───┐           │
│  │  Objects: {a,b}  │   │  Objects: {c,d}  │           │
│  │  Rules: a→aa     │   │  Rules: c→out d  │           │
│  └──────────────────┘   └──────────────────┘           │
│  Objects in outer region: {e, f}                        │
└─────────────────────────────────────────────────────────┘
```

### Relevance to Neural Computing

In COSYS-ESN, the membrane metaphor maps onto the **triadic service architecture**:
- Each *triad* acts as a membrane layer with distinct computational responsibilities.
- Information flows through service boundaries (membranes) with transformation at each crossing.
- The reservoir is the innermost compartment where complex dynamics emerge.

---

## 2. Membrane Layers in COSYS-ESN

COSYS-ESN organises 18 services across three concentric membrane layers, each corresponding to a biological triad in Stafford Beer's Viable System Model (VSM):

```
┌────────────────────────────────────────────────────────────┐
│           CEREBRAL TRIAD (Outermost Membrane)              │
│  Learning / Readout / Output delivery to environment        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         SOMATIC TRIAD (Middle Membrane)              │  │
│  │    Reservoir dynamics / State management             │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │      AUTONOMIC TRIAD (Innermost Membrane)      │  │  │
│  │  │  Input validation / Preprocessing / Encoding  │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

### Information flow

```
Environment
    │
    ▼ RAW_INPUT
[Autonomic Membrane] ──→ VALIDATED_INPUT → PREPROCESSED_INPUT → ENCODED_INPUT
    │
    ▼ MEMBRANE_INPUT
[Somatic Membrane] ──→ RESERVOIR_STATE
    │
    ▼ RESERVOIR_STATE
[Cerebral Membrane] ──→ READOUT_OUTPUT → FINAL_OUTPUT
    │
    ▼ Predictions / Actions
```

---

## 3. Boundary Conditions and Input Scaling (M-1 Service)

Each membrane boundary is implemented by the **M-1 (Membrane Interface)** service. In biological systems, membrane interfaces regulate what passes through via:
- **Selective permeability** — only certain signal types pass
- **Signal amplification/attenuation** — scaling up or down
- **Active transport** — adaptive, state-dependent scaling

### Autonomic M-1: `InputMonitoringService`

```python
class InputMonitoringService(BaseCosmosService):
    def __init__(self, config, input_dim, input_scaling=1.0)
```

Guards the outer boundary:
- Validates input dimensionality
- Replaces NaN/Inf with safe values
- Applies static input scaling

### Somatic M-1: `MembraneInterfaceService`

```python
class MembraneInterfaceService(BaseCosmosService):
    def __init__(self, config, input_scaling=1.0, adaptive=False)
```

The somatic boundary is the true membrane interface — it can **adaptively** scale input based on current reservoir state norms:

```python
# Adaptive scaling: reduce scaling if reservoir is saturating
svc = MembraneInterfaceService(config, input_scaling=1.0, adaptive=True)

# Feedback loop: high reservoir norms → reduce input scaling
svc.update_state_feedback(state_norm=8.0)
# After 10+ feedback calls: svc.input_scaling < 1.0
```

### Cerebral M-1: `TargetInterfaceService`

The cerebral boundary receives target signals from the environment (for training) and computes loss:

```python
loss = svc.compute_loss(prediction, target)  # MSE or MAE
```

---

## 4. Three Triads as Membrane Hierarchy

### Autonomic Triad — Sensory/Afferent Membrane

| Position | Service | Membrane Role |
|----------|---------|---------------|
| M-1 | `InputMonitoringService` | Outer boundary — validates and scales raw signals |
| S-8 | `SignalStateService` | Buffer membrane — temporal windowing |
| PD-2 | `PreprocessingDirectorService` | Transform membrane — normalisation, PCA |
| P-5 | `TransformProcessingService` | Spectral membrane — FFT/DCT transforms |
| O-4 | `InputOrganizationService` | Organisation membrane — batching and windows |
| T-7 | `EncodingTriggersService` | Sparse coding gate — k-WTA, thresholding |

The autonomic triad implements **afferent signal conditioning**: filtering, normalising, and structuring sensory input before it reaches the reservoir core.

### Somatic Triad — Reservoir/Process Membrane

| Position | Service | Membrane Role |
|----------|---------|---------------|
| M-1 | `MembraneInterfaceService` | Inner boundary — final input scaling |
| S-8 | `StateManagementService` | State membrane — reservoir state persistence |
| P-5 | `RecurrentProcessingService` | Core membrane — nonlinear reservoir dynamics |
| O-4 | `DynamicsOrganizationService` | Monitor membrane — spectral radius enforcement |
| PD-2 | `DynamicsDevelopmentService` | Topology membrane — reservoir structure tuning |
| T-7 | `StateTreasuryService` | Memory membrane — echo state storage |

The somatic triad is the **commitment layer** — it commits to temporal processing, maintaining and evolving the internal state of the reservoir.

### Cerebral Triad — Readout/Efferent Membrane

| Position | Service | Membrane Role |
|----------|---------|---------------|
| M-1 | `TargetInterfaceService` | Target boundary — loss computation |
| S-8 | `OutputDeliveryService` | Output membrane — prediction generation |
| P-5 | `ReadoutProcessingService` | Readout membrane — linear decoding |
| O-4 | `OutputOrganizationService` | Format membrane — output structuring |
| PD-2 | `LearningDirectorService` | Learning membrane — ridge/RLS/FORCE |
| T-7 | `WeightTreasuryService` | Weight membrane — model persistence |

The cerebral triad extracts **potential** — linear combinations of high-dimensional reservoir states that represent the system's learned output.

---

## 5. ReservoirSystem5 Compartment Model

`ReservoirSystem5` (in `src/models/system5_esn.py`) implements the full VSM System 5 hierarchy as a three-compartment membrane model:

```
System 5 (Identity/Policy)
    │
    ├── Compartment A: Autonomic  ←── sensory input
    │       Input normalisation
    │       Temporal buffering
    │       Feature extraction
    │
    ├── Compartment B: Somatic    ←── internal dynamics
    │       Reservoir state x(t)
    │       Spectral radius control
    │       Echo state property
    │
    └── Compartment C: Cerebral   ←── output generation
            Readout W_out
            Learning (ridge regression)
            Output delivery
```

### Compartment boundaries as membrane filters

Each compartment boundary transforms the signal:

1. **A→B boundary** (Autonomic→Somatic): The encoded input vector is scaled and injected into the reservoir via `W_in`. This is the *selective import* step — only dimensions that W_in projects into the reservoir matter.

2. **B→C boundary** (Somatic→Cerebral): The reservoir state `x(t)` is read out via `W_out`. This is *selective export* — W_out selects the relevant patterns from the high-dimensional state space.

3. **C→Environment**: The prediction is formatted and delivered with confidence scores.

### Membrane metaphor in the update equation

```
x(t+1) = (1 - α) · x(t) + α · tanh(W_in · u(t) + W · x(t))
               ↑                        ↑                ↑
         [retention]            [import across        [internal
          (leak)                 membrane via W_in]    circulation]
```

The leak rate `α` controls the membrane's **permeability to new information** vs. its **retention of past state**.

---

## References

- Păun, G. (2000). Computing with membranes. *Journal of Computer and System Sciences*, 61(1), 108–143.
- Beer, S. (1972). *Brain of the Firm*. Allen Lane, London.
- Jaeger, H. (2001). *The "echo state" approach to analysing and training recurrent neural networks*. GMD Technical Report 148.
- Lukoševičius, M. (2012). A practical guide to applying echo state networks. *Neural Networks: Tricks of the Trade*, 659–686.
