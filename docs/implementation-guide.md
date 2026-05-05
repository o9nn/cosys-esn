# COSYS-ESN Implementation Guide

## 1. Package Structure

```
cosys-esn/
├── src/
│   ├── cosmos_core/
│   │   └── __init__.py          # Shared primitives (enums, dataclasses, base classes)
│   ├── autonomic_triad/         # Input processing layer (6 services)
│   │   ├── input_monitoring.py       (M-1)
│   │   ├── signal_state.py           (S-8)
│   │   ├── preprocessing_director.py (PD-2)
│   │   ├── transform_processing.py   (P-5)
│   │   ├── input_organization.py     (O-4)
│   │   └── encoding_triggers.py      (T-7)
│   ├── somatic_triad/           # Reservoir dynamics layer (6 services)
│   │   ├── membrane_interface.py     (M-1)
│   │   ├── state_management.py       (S-8)
│   │   ├── recurrent_processing.py   (P-5)
│   │   ├── dynamics_organization.py  (O-4)
│   │   ├── dynamics_development.py   (PD-2)
│   │   └── state_treasury.py         (T-7)
│   ├── cerebral_triad/          # Output/learning layer (6 services)
│   │   ├── target_interface.py       (M-1)
│   │   ├── output_delivery.py        (S-8)
│   │   ├── readout_processing.py     (P-5)
│   │   ├── output_organization.py    (O-4)
│   │   ├── learning_director.py      (PD-2)
│   │   └── weight_treasury.py        (T-7)
│   ├── models/
│   │   ├── echo_state_network.py    # EchoStateNetwork + ESNConfig
│   │   └── system5_esn.py           # ReservoirSystem5
│   ├── reservoir_core/
│   │   ├── echo_beats/
│   │   │   └── echo_beats.py        # EchoBeatsLoop + CognitiveStream
│   │   ├── autognosis/
│   │   │   └── autognosis.py        # ReservoirAutognosis
│   │   └── ontogenesis/
│   │       └── ontogenesis.py       # ReservoirKernelGenome + Population
│   └── integration_hub/
│       ├── event_bus.py
│       ├── stream_manager.py
│       └── cosmos_reservoir_system.py  # CosmosReservoirSystem
├── tests/                       # pytest test suite (54 tests)
├── examples/                    # Runnable demo scripts
└── docs/                        # Documentation
```

---

## 2. `cosmos_core` Primitives

All shared infrastructure lives in `src/cosmos_core/__init__.py`.

### Enums

```python
from cosmos_core import Triad, Polarity, ServicePosition, Dimension

Triad.AUTONOMIC    # "autonomic"
Triad.SOMATIC      # "somatic"
Triad.CEREBRAL     # "cerebral"

Polarity.SYMPATHETIC       # input-oriented
Polarity.PARASYMPATHETIC   # rest/integration-oriented
Polarity.SOMATIC           # action-oriented

ServicePosition.M1   # "M-1" — Membrane Interface
ServicePosition.S8   # "S-8" — State/Signal
ServicePosition.P5   # "P-5" — Processing
ServicePosition.O4   # "O-4" — Organisation
ServicePosition.PD2  # "PD-2" — Process Director
ServicePosition.T7   # "T-7" — Treasury/Trigger

Dimension.PERFORMANCE   # autonomic efficiency
Dimension.COMMITMENT    # somatic persistence
Dimension.POTENTIAL     # cerebral learning
```

### `ServiceConfig`

```python
@dataclass
class ServiceConfig:
    service_name: str
    triad: Triad
    position: ServicePosition
    polarity: Polarity
    dimension: Dimension
```

### `ServiceMessage`

```python
@dataclass
class ServiceMessage:
    type: str          # Message type string (e.g. "RAW_INPUT", "RESERVOIR_STATE")
    payload: Any       # Typically np.ndarray or dict
    source: str        # Originating service name
    target: Optional[str] = None
    timestamp: float   # Auto-set to time.time()
    metadata: Dict[str, Any]  # Arbitrary metadata
```

### `create_message`

```python
from cosmos_core import create_message
msg = create_message("RAW_INPUT", np.array([1.0, 2.0, 3.0]), source="sensor")
```

### `BaseCosmosService`

Abstract base for all services. Override `initialize`, `process`, `shutdown`:

```python
class MyService(BaseCosmosService):
    async def initialize(self) -> None: ...
    async def process(self, message: ServiceMessage) -> Optional[ServiceMessage]: ...
    async def shutdown(self) -> None: ...
```

### `TriadicCoordinator`

```python
from cosmos_core import TriadicCoordinator
coord = TriadicCoordinator()
coord.register_service(my_service)
services = coord.get_services()            # All services
auto_services = coord.get_by_triad(Triad.AUTONOMIC)
```

---

## 3. All 18 Services

### Autonomic Triad

#### `InputMonitoringService` (M-1)

```python
from autonomic_triad.input_monitoring import InputMonitoringService
svc = InputMonitoringService(config, input_dim=5, input_scaling=1.0)
```

- **Input message type**: `"RAW_INPUT"` with `np.ndarray` payload
- **Output message type**: `"VALIDATED_INPUT"`
- **input_dim**: Expected input dimensionality; returns `None` if mismatch
- **input_scaling**: Scalar multiplier applied to validated input
- Replaces NaN/Inf with `nan_to_num(nan=0.0, posinf=1.0, neginf=-1.0)`

#### `SignalStateService` (S-8)

```python
from autonomic_triad.signal_state import SignalStateService
svc = SignalStateService(config, window_size=10)
```

- **Input message types**: `"VALIDATED_INPUT"`, `"PREPROCESSED_INPUT"`
- **Output message type**: `"BUFFERED_INPUT"` with windowed `np.ndarray`
- **window_size**: Circular buffer size (uses `deque(maxlen=window_size)`)
- `svc.buffer` — the deque; `svc.get_window()` — returns `np.ndarray`

#### `PreprocessingDirectorService` (PD-2)

```python
from autonomic_triad.preprocessing_director import PreprocessingDirectorService
svc = PreprocessingDirectorService(config, method="zscore", n_components=None)
```

- **methods**: `"zscore"`, `"minmax"`, `"none"`
- **n_components**: If set, applies PCA to `n_components` dimensions after normalisation
- `svc.fit(data)` — fit on `(n_samples, n_features)` array
- `svc.transform(x)` — transform a single vector
- **Input message types**: `"VALIDATED_INPUT"`, `"RAW_INPUT"`
- **Output message type**: `"PREPROCESSED_INPUT"`

#### `TransformProcessingService` (P-5)

```python
from autonomic_triad.transform_processing import TransformProcessingService
svc = TransformProcessingService(config, transform="none", embed_dim=None)
```

- **transform**: `"none"`, `"fft"` (real FFT magnitude), `"dct"`
- **embed_dim**: Optional random projection to `embed_dim` dimensions via tanh
- **Input message types**: `"PREPROCESSED_INPUT"`, `"VALIDATED_INPUT"`, `"RAW_INPUT"`
- **Output message type**: `"TRANSFORMED_INPUT"`

#### `InputOrganizationService` (O-4)

```python
from autonomic_triad.input_organization import InputOrganizationService
svc = InputOrganizationService(config, window_size=1, stride=1, pad_mode="zero")
```

- Maintains a rolling history of inputs, returns windowed batches
- `svc.create_windows(sequence)` — create sliding windows from a 2D array
- **Output message type**: `"ORGANIZED_INPUT"` with shape `(<=window_size, n_features)`

#### `EncodingTriggersService` (T-7)

```python
from autonomic_triad.encoding_triggers import EncodingTriggersService
svc = EncodingTriggersService(config, k=5, threshold=None, output_dim=None)
```

- **k**: k-WTA: keep top-k activations by magnitude, zero the rest
- **threshold**: Hard threshold (overrides k-WTA if set); keeps values with `|x| >= threshold`
- **output_dim**: Optional random projection before encoding
- **Input message types**: `"PREPROCESSED_INPUT"`, `"VALIDATED_INPUT"`, `"ORGANIZED_INPUT"`, `"TRANSFORMED_INPUT"`
- **Output message type**: `"ENCODED_INPUT"`

---

### Somatic Triad

#### `MembraneInterfaceService` (M-1)

```python
from somatic_triad.membrane_interface import MembraneInterfaceService
svc = MembraneInterfaceService(config, input_scaling=1.0, adaptive=False)
```

- Scales input by `input_scaling`
- **adaptive=True**: Calls `update_state_feedback(state_norm)` to auto-adjust scaling. If mean norm > 2.0 → multiply scaling by 0.95. If mean norm < 0.5 → multiply by 1.05.
- `svc.update_state_feedback(norm)` — call after each reservoir update
- **Input message types**: `"PREPROCESSED_INPUT"`, `"ENCODED_INPUT"`, `"VALIDATED_INPUT"`, `"ORGANIZED_INPUT"`, `"RAW_INPUT"`
- **Output message type**: `"MEMBRANE_INPUT"`

#### `StateManagementService` (S-8)

```python
from somatic_triad.state_management import StateManagementService
svc = StateManagementService(config, reservoir_dim=500, max_history=1000)
```

- `svc.state` — current `(reservoir_dim,)` state vector
- `svc.state_history` — list of past states (capped at `max_history`)
- `svc.update_state(new_state)` — updates state and appends to history
- `svc.reset()` — zeros the state

#### `RecurrentProcessingService` (P-5)

```python
from somatic_triad.recurrent_processing import RecurrentProcessingService
svc = RecurrentProcessingService(
    config,
    reservoir_dim=500, input_dim=5,
    spectral_radius=0.95, sparsity=0.1,
    leak_rate=0.3, activation="tanh",
    random_seed=42,
)
await svc.initialize()  # Builds W_in and W
```

- `svc.update(u)` — update reservoir state synchronously; returns state
- `svc.state` — current reservoir state
- **Input message type**: `"MEMBRANE_INPUT"`
- **Output message type**: `"RESERVOIR_STATE"`

#### `DynamicsOrganizationService` (O-4)

```python
from somatic_triad.dynamics_organization import DynamicsOrganizationService
svc = DynamicsOrganizationService(config, target_spectral_radius=0.95, target_leak_rate=0.3)
```

- `svc.monitor_state(state)` — returns `{"state_norm", "activation_sparsity", "edge_of_chaos"}`
- **Input message type**: `"RESERVOIR_STATE"`
- **Output message type**: `"DYNAMICS_METRICS"`

#### `DynamicsDevelopmentService` (PD-2)

```python
from somatic_triad.dynamics_development import DynamicsDevelopmentService
svc = DynamicsDevelopmentService(config)
```

- `svc.build_topology(N, topology, sparsity, seed)` — returns sparse `(N, N)` matrix
  - topology: `"random"`, `"small_world"`, `"scale_free"`
- `svc.attach_reservoir(reservoir_service)` — link to P-5 for live tuning

#### `StateTreasuryService` (T-7)

```python
from somatic_triad.state_treasury import StateTreasuryService
svc = StateTreasuryService(config, capacity=500, compress=False, compress_dim=50)
```

- `svc.store(state, label="")` — append state to memory store
- `svc.retrieve(query, top_k=1)` — cosine-similarity nearest-neighbour lookup
- `svc._store` — list of stored states

---

### Cerebral Triad

#### `TargetInterfaceService` (M-1)

```python
from cerebral_triad.target_interface import TargetInterfaceService
svc = TargetInterfaceService(config, loss="mse")
```

- `svc.compute_loss(prediction, target)` — returns scalar float
- **loss**: `"mse"` or `"mae"`
- **Input message type**: `"PREDICTION_WITH_TARGET"` with payload `{"prediction": ..., "target": ...}`
- **Output message type**: `"LOSS_SIGNAL"` with `{"loss": ..., "error": ...}`

#### `OutputDeliveryService` (S-8)

```python
from cerebral_triad.output_delivery import OutputDeliveryService
svc = OutputDeliveryService(config, task="regression")
```

- **task**: `"regression"` or `"classification"`
- For regression: `prediction` is the raw output array
- For classification: `prediction` is `argmax(output)`; `confidence` is `max(softmax(output))`
- **Input message types**: `"READOUT_OUTPUT"`, `"FINAL_OUTPUT"`
- **Output message type**: `"DELIVERED_OUTPUT"` with `{"prediction": ..., "confidence": ...}`

#### `ReadoutProcessingService` (P-5)

```python
from cerebral_triad.readout_processing import ReadoutProcessingService
svc = ReadoutProcessingService(config, reservoir_dim=500, output_dim=1)
await svc.initialize()
```

- `svc.train(states, targets, ridge_lambda=1e-6)` — fits W_out via ridge regression
- `svc.W_out` — `(output_dim, reservoir_dim+1)` weight matrix
- **Input message type**: `"RESERVOIR_STATE"`
- **Output message type**: `"READOUT_OUTPUT"`

#### `OutputOrganizationService` (O-4)

```python
from cerebral_triad.output_organization import OutputOrganizationService
svc = OutputOrganizationService(config, output_scaling=1.0)
```

- **Input message types**: `"READOUT_OUTPUT"`, `"DELIVERED_OUTPUT"`
- **Output message type**: `"FINAL_OUTPUT"` with `{"output": ..., "timestamp": ..., "source_triad": "cerebral"}`

#### `LearningDirectorService` (PD-2)

```python
from cerebral_triad.learning_director import LearningDirectorService
svc = LearningDirectorService(config, method="ridge", ridge_lambda=1e-6, rls_forgetting=0.99)
```

- `svc.batch_train(states, targets)` — batch ridge regression; returns `W_out`
- `svc.rls_init(feature_dim, output_dim)` — initialise RLS matrices
- `svc.rls_update(state, target)` — one-step RLS update; returns `W_out`
- `svc.force_update(state, error)` — one-step FORCE update

#### `WeightTreasuryService` (T-7)

```python
from cerebral_triad.weight_treasury import WeightTreasuryService
svc = WeightTreasuryService(config)
```

- `svc.store(name, weights, metadata=None)` — store named weight matrix
- `svc.retrieve(name)` — returns the weight array (or None)
- `svc.save(path)` — save all weights to `.npz`
- `svc.load(path)` — load all weights from `.npz`

---

## 4. `EchoStateNetwork` API

```python
from models.echo_state_network import EchoStateNetwork, ESNConfig
```

### `ESNConfig`

```python
@dataclass
class ESNConfig:
    n_inputs: int = 10
    n_reservoir: int = 500
    n_outputs: int = 1
    spectral_radius: float = 0.9
    input_scaling: float = 0.5
    leak_rate: float = 0.3
    sparsity: float = 0.1          # connection density
    activation: str = "tanh"       # "tanh", "relu", "sigmoid"
    topology: str = "random"       # "random", "small_world", "scale_free"
    ridge_lambda: float = 1e-6
    random_seed: Optional[int] = 42
    washout: int = 100
```

### `EchoStateNetwork`

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(config)` | Builds W_in, W, W_out |
| `reset_state` | `()` | Zeros reservoir state x |
| `update` | `(u: ndarray) → ndarray` | One step forward; returns state |
| `predict` | `(u: ndarray) → ndarray` | Update + readout; returns `(n_outputs,)` |
| `train` | `(inputs, targets, ridge_param=None) → W_out` | Batch training |
| `run` | `(inputs: ndarray) → ndarray` | Predict over sequence; returns `(n_samples, n_outputs)` |
| `save` | `(path: str)` | Save to `.npz` |
| `load` | `(path, config) → ESN` | Class method; load from `.npz` |

**Attributes**: `config`, `W_in`, `W` (sparse), `W_out`, `x` (current state)

---

## 5. `ReservoirSystem5` API

```python
from models.system5_esn import ReservoirSystem5
```

Implements Beer's Viable System Model with 5 systems mapping to reservoir computing:

| VSM System | Role in ESN |
|------------|-------------|
| System 1 | Reservoir units (operational) |
| System 2 | Anti-oscillation (spectral radius normalisation) |
| System 3 | Internal regulation (leak rate) |
| System 4 | Environment scanning (readout W_out) |
| System 5 | Identity/Policy (ESNConfig) |

---

## 6. `EchoBeatsLoop` API

```python
from reservoir_core.echo_beats.echo_beats import EchoBeatsLoop, CognitiveStream, Phase, PHASE_SEQUENCE
```

### `PHASE_SEQUENCE`

12 phases cycling through: `PERCEIVE, ATTEND, FRAME, REASON, PERCEIVE, INTEND, PERCEIVE, EXECUTE, PERCEIVE, EVALUATE, PERCEIVE, INTEGRATE`

### `CognitiveStream`

```python
stream = CognitiveStream(esn, phase_offset=0, stream_id=0)
```

- `stream.get_phase(step_idx)` → `Phase`
- `stream.perceive(u)` — update reservoir state; append to perception buffer
- `stream.attend()` — attention-weighted combination of buffer states
- `stream.frame()` — construct frame from attended state
- `stream.reason()` — apply readout to current state
- `stream.intend()` — generate intention from reasoning output
- `stream.execute()` — execute intention via W_out
- `stream.evaluate()` — compute prediction error
- `stream.integrate()` — blend all stream states

### `EchoBeatsLoop`

```python
loop = EchoBeatsLoop(esn)
```

Three streams with phase offsets 0, 4, 8 (120° apart):

```python
# Single step
result = loop.step(step_idx=0, input_data=u)
# result = {
#   "stream_0": {"phase": "perceive", "output": ndarray},
#   "stream_1": {"phase": "perceive", "output": ndarray},
#   "stream_2": {"phase": "perceive", "output": ndarray},
#   "integrated_state": ndarray,
# }

# Full 12-step cycle
outputs = loop.run_cycle(input_sequence)  # input_sequence: (n_steps, n_inputs)
# returns list of 12 step result dicts
```

---

## 7. `ReservoirAutognosis` API

```python
from reservoir_core.autognosis.autognosis import ReservoirAutognosis
ag = ReservoirAutognosis(esn, window_size=100)
```

### Layer 1: Self-Monitoring

| Method | Returns | Description |
|--------|---------|-------------|
| `state_norm_tracking()` | `float` | L2 norm of current reservoir state |
| `spectral_analysis()` | `float` | Current effective spectral radius |
| `echo_index()` | `float` | Proxy for memory capacity = 1/leak_rate |
| `activation_statistics()` | `dict` | mean, std, sparsity, max_abs |
| `monitor()` | `dict` | Full metrics pass; appends to `metrics_history` |

### Layer 2: Self-Modeling

| Method | Returns | Description |
|--------|---------|-------------|
| `capacity_estimation()` | `float` | MC + nonlinear capacity estimate |
| `kernel_quality()` | `dict` | separation, approximation scores |
| `memory_depth()` | `int` | Effective echo horizon in timesteps |
| `nonlinearity_profile()` | `dict` | distortion, saturation_fraction |

### Layer 3: Meta-Cognitive

| Method | Returns | Description |
|--------|---------|-------------|
| `performance_prediction()` | `float [0,1]` | Stability-based performance proxy |
| `confidence_scoring()` | `float [0,1]` | Rolling confidence estimate |
| `anomaly_detection(threshold_sigma=3.0)` | `bool` | OOD state detection |
| `should_adapt()` | `bool` | Whether adaptation is needed |

### Layer 4: Self-Optimization

| Method | Returns | Description |
|--------|---------|-------------|
| `tune_spectral_radius(target=None)` | `float` | Scales W toward target SR |
| `adapt_leak_rate(delta=0.01)` | `float` | Adjusts leak rate from trend |
| `optimize_input_scaling()` | `float` | Adjusts W_in scaling |
| `refine_topology()` | `str` | Returns topology suggestion string |

---

## 8. `ReservoirKernelGenome` / `Population` / Ontogenesis API

```python
from reservoir_core.ontogenesis.ontogenesis import ReservoirKernelGenome, Population
```

### `ReservoirKernelGenome`

Encodes an ESN configuration as an evolvable genome:

```python
genome = ReservoirKernelGenome(
    n_reservoir=200, spectral_radius=0.9, leak_rate=0.3,
    sparsity=0.1, topology="random", input_scaling=0.5,
    ridge_lambda=1e-6, random_seed=42,
)
config = genome.to_esn_config(n_inputs=5, n_outputs=1)
child = genome.mutate(mutation_rate=0.1)
offspring = genome.crossover(other_genome)
fitness = genome.fitness  # assigned externally
```

### `Population`

```python
pop = Population(size=50, n_inputs=5, n_outputs=1, random_seed=0)
pop.initialize()
# Evolution loop
for generation in range(100):
    for genome in pop.genomes:
        genome.fitness = evaluate(genome)
    pop.evolve(tournament_size=3, mutation_rate=0.1, elite_fraction=0.1)
best = pop.best_genome()
```

---

## 9. `CosmosReservoirSystem` API

```python
from integration_hub.cosmos_reservoir_system import CosmosReservoirSystem
from models.echo_state_network import ESNConfig
```

Full 18-service wired async pipeline:

```python
config = ESNConfig(
    n_inputs=5, n_reservoir=200, n_outputs=1,
    spectral_radius=0.95, leak_rate=0.3, sparsity=0.1,
    ridge_lambda=1e-6, washout=100, random_seed=42,
)
system = CosmosReservoirSystem(config)
await system.initialize()          # Boot all 18 services

await system.train(X_train, y_train)   # Batch training
output = await system.process_input(u)  # Single-step prediction
```

The `process_input` method routes the input through the full pipeline:
1. Autonomic triad: validate → preprocess → transform → organise → encode
2. Somatic triad: membrane scale → reservoir update → state management
3. Cerebral triad: readout → deliver → organise

---

## 10. Full End-to-End Example

```python
import numpy as np
import asyncio
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from integration_hub.cosmos_reservoir_system import CosmosReservoirSystem
from models.echo_state_network import ESNConfig
from reservoir_core.autognosis.autognosis import ReservoirAutognosis
from reservoir_core.echo_beats.echo_beats import EchoBeatsLoop

# ── 1. Generate Mackey-Glass time series ──────────────────────────────────────
def mackey_glass(n=2000, tau=17):
    x = np.zeros(n); x[0] = 1.2
    for t in range(tau, n-1):
        x[t+1] = x[t] + 0.2*x[t-tau]/(1+x[t-tau]**10) - 0.1*x[t]
    return x

data = mackey_glass()
X = np.array([data[i:i+5] for i in range(len(data)-6)])
y = data[5:].reshape(-1, 1)

# ── 2. Configure and initialise the system ────────────────────────────────────
config = ESNConfig(
    n_inputs=5, n_reservoir=200, n_outputs=1,
    spectral_radius=0.95, leak_rate=0.3, sparsity=0.1,
    ridge_lambda=1e-6, washout=100, random_seed=42,
)

system = CosmosReservoirSystem(config)
asyncio.run(system.initialize())

# ── 3. Train ──────────────────────────────────────────────────────────────────
asyncio.run(system.train(X[:1000], y[:1000]))

# ── 4. Evaluate ───────────────────────────────────────────────────────────────
predictions = []
for u in X[1000:1100]:
    pred = asyncio.run(system.process_input(u))
    predictions.append(pred[0])

predictions = np.array(predictions)
targets = y[1000:1100].flatten()
nrmse = np.sqrt(np.mean((predictions - targets)**2)) / np.std(targets)
print(f"NRMSE: {nrmse:.4f}")  # Expect < 0.1

# ── 5. Autognosis monitoring ───────────────────────────────────────────────────
ag = ReservoirAutognosis(system.esn, window_size=50)
for t in range(50):
    system.esn.update(X[1100+t])
    ag.monitor()

if ag.should_adapt():
    new_sr = ag.tune_spectral_radius()
    print(f"Adapted spectral radius to {new_sr:.3f}")

# ── 6. EchoBeats cognitive cycle ──────────────────────────────────────────────
loop = EchoBeatsLoop(system.esn)
cycle_result = loop.run_cycle(X[1150:1162])
for i, step in enumerate(cycle_result):
    phases = [step[f"stream_{j}"]["phase"] for j in range(3)]
    print(f"Step {i+1:2d}: {phases}")
```

---

## Appendix: Message Type Reference

| Message Type | Direction | Payload | Produced By |
|---|---|---|---|
| `RAW_INPUT` | External → Autonomic | `ndarray` | User/Environment |
| `VALIDATED_INPUT` | Autonomic M-1 → S-8 | `ndarray` | InputMonitoringService |
| `BUFFERED_INPUT` | Autonomic S-8 → PD-2 | `ndarray (window, feat)` | SignalStateService |
| `PREPROCESSED_INPUT` | Autonomic PD-2 → P-5 | `ndarray` | PreprocessingDirectorService |
| `TRANSFORMED_INPUT` | Autonomic P-5 → O-4 | `ndarray` | TransformProcessingService |
| `ORGANIZED_INPUT` | Autonomic O-4 → T-7 | `ndarray (<=w, feat)` | InputOrganizationService |
| `ENCODED_INPUT` | Autonomic T-7 → Somatic | `ndarray` | EncodingTriggersService |
| `MEMBRANE_INPUT` | Somatic M-1 → P-5 | `ndarray` | MembraneInterfaceService |
| `RESERVOIR_STATE` | Somatic P-5 → Cerebral | `ndarray (reservoir_dim,)` | RecurrentProcessingService |
| `READOUT_OUTPUT` | Cerebral P-5 → S-8 | `ndarray (output_dim,)` | ReadoutProcessingService |
| `DELIVERED_OUTPUT` | Cerebral S-8 → O-4 | `dict {prediction, confidence}` | OutputDeliveryService |
| `FINAL_OUTPUT` | Cerebral O-4 → User | `dict {output, timestamp, ...}` | OutputOrganizationService |
| `LOSS_SIGNAL` | Cerebral M-1 → PD-2 | `dict {loss, error}` | TargetInterfaceService |
| `DYNAMICS_METRICS` | Somatic O-4 → PD-2 | `dict {state_norm, ...}` | DynamicsOrganizationService |
