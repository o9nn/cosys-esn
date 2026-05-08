# COSYS-ESN Implementation

## Status: ✅ All 10 Phases Complete

This repository contains the complete implementation of the **Cosmos System 5** model applied to **Echo State Networks** and **Reservoir Computing**.

---

## Architecture

### Triadic Structure (18 Services)

```
┌─────────────────────────────────────────────────────────────┐
│                    COSYS-ESN IMPLEMENTATION                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   AUTONOMIC TRIAD (Input/Sensory Processing)               │
│   ├── M-1: InputMonitoringService   — validate, scale       │
│   ├── S-8: SignalStateService       — circular buffer       │
│   ├── PD-2: PreprocessingDirectorService — zscore/minmax/PCA│
│   ├── P-5: TransformProcessingService — FFT/DCT/embed       │
│   ├── O-4: InputOrganizationService — sliding windows       │
│   └── T-7: EncodingTriggersService  — k-WTA, threshold      │
│                                                             │
│   SOMATIC TRIAD (Reservoir Dynamics)                       │
│   ├── M-1: MembraneInterfaceService — adaptive scaling      │
│   ├── S-8: StateManagementService  — state persistence      │
│   ├── P-5: RecurrentProcessingService — ESN dynamics        │
│   ├── O-4: DynamicsOrganizationService — spectral monitor   │
│   ├── PD-2: DynamicsDevelopmentService — topology builder   │
│   └── T-7: StateTreasuryService    — echo memory store      │
│                                                             │
│   CEREBRAL TRIAD (Readout & Learning)                      │
│   ├── M-1: TargetInterfaceService  — loss computation       │
│   ├── S-8: OutputDeliveryService   — classification/regress │
│   ├── P-5: ReadoutProcessingService — ridge regression W_out│
│   ├── O-4: OutputOrganizationService — format output        │
│   ├── PD-2: LearningDirectorService — ridge/RLS/FORCE       │
│   └── T-7: WeightTreasuryService   — model save/load        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase Completion

### Phase 1: Project Foundation ✅
- `pyproject.toml` with dependencies (numpy, scipy)
- `src/cosmos_core/__init__.py`: Triad, Polarity, ServicePosition, Dimension, ServiceConfig, ServiceMessage, create_message, BaseCosmosService, TriadicCoordinator

### Phase 2: Autonomic Triad ✅
All 6 services implemented in `src/autonomic_triad/`:
- `input_monitoring.py` — M-1: validation, NaN handling, scaling
- `signal_state.py` — S-8: circular buffer (deque)
- `preprocessing_director.py` — PD-2: z-score, min-max, PCA
- `transform_processing.py` — P-5: FFT, DCT, random projection
- `input_organization.py` — O-4: sliding windows, stride, padding
- `encoding_triggers.py` — T-7: k-WTA, threshold, projection

### Phase 3: Somatic Triad ✅
All 6 services implemented in `src/somatic_triad/`:
- `membrane_interface.py` — M-1: input scaling with adaptive feedback
- `state_management.py` — S-8: reservoir state + history
- `recurrent_processing.py` — P-5: sparse ESN reservoir (scipy.sparse)
- `dynamics_organization.py` — O-4: spectral radius monitoring
- `dynamics_development.py` — PD-2: random/small-world/scale-free topologies
- `state_treasury.py` — T-7: cosine-similarity memory store

### Phase 4: Cerebral Triad ✅
All 6 services implemented in `src/cerebral_triad/`:
- `target_interface.py` — M-1: MSE/MAE loss
- `output_delivery.py` — S-8: regression/classification output
- `readout_processing.py` — P-5: ridge regression readout
- `output_organization.py` — O-4: output formatting
- `learning_director.py` — PD-2: batch ridge, RLS, FORCE
- `weight_treasury.py` — T-7: model save/load (.npz)

### Phase 5: EchoStateNetwork Model ✅
`src/models/echo_state_network.py`:
- `ESNConfig` dataclass with full configuration
- `EchoStateNetwork` with `update`, `train`, `predict`, `run`, `save`, `load`
- Topologies: random, small-world, scale-free
- Spectral radius scaling via scipy.sparse.linalg.eigs

### Phase 6: EchoBeats Cognitive Loop ✅
`src/reservoir_core/echo_beats/echo_beats.py`:
- `Phase` enum: PERCEIVE, ATTEND, FRAME, REASON, INTEND, EXECUTE, EVALUATE, INTEGRATE
- `PHASE_SEQUENCE`: 12-step sequence
- `CognitiveStream`: Single stream with phase offset and perception buffer
- `EchoBeatsLoop`: 3 interleaved streams (0°, 120°, 240° offsets)

### Phase 7: ReservoirAutognosis ✅
`src/reservoir_core/autognosis/autognosis.py`:
- Layer 1 (Self-Monitoring): state_norm, spectral_analysis, echo_index, activation_statistics
- Layer 2 (Self-Modeling): capacity_estimation, kernel_quality, memory_depth
- Layer 3 (Meta-Cognitive): performance_prediction, anomaly_detection, should_adapt
- Layer 4 (Self-Optimization): tune_spectral_radius, adapt_leak_rate, optimize_input_scaling

### Phase 8: Integration Hub ✅
`src/integration_hub/cosmos_reservoir_system.py`:
- `CosmosReservoirSystem`: Full 18-service wired async pipeline
- Autonomic → Somatic → Cerebral routing with message passing
- `initialize`, `train`, `process_input` async methods

Also implemented:
- `src/reservoir_core/ontogenesis/ontogenesis.py`: `ReservoirKernelGenome`, `Population` for evolutionary reservoir optimisation
- `src/models/system5_esn.py`: `ReservoirSystem5` mapping VSM to ESN

### Phase 9: Tests ✅
54 tests in `tests/` directory (all passing):
- `test_cosmos_core.py` — primitives: enums, configs, messages, coordinator
- `test_autonomic.py` — all 6 autonomic services
- `test_somatic.py` — all 6 somatic services
- `test_cerebral.py` — all 6 cerebral services
- `test_esn.py` — ESN integration: Mackey-Glass NRMSE < 0.1, topologies, save/load
- `test_echo_beats.py` — 12-step cognitive loop
- `test_autognosis.py` — 4-layer self-awareness system

Run with: `python -m pytest tests/ -q`

### Phase 10: Documentation & Examples ✅

**Docs** (`docs/`):
- `reservoir-computing.md` — ESNs, spectral radius, ridge regression, memory capacity, ESP, Mackey-Glass, usage examples
- `membrane-computing.md` — P-systems, membrane layers, boundary conditions, triadic hierarchy, compartment model
- `implementation-guide.md` — Full API reference for all 18 services, ESNConfig, EchoStateNetwork, EchoBeatsLoop, ReservoirAutognosis, ReservoirKernelGenome, CosmosReservoirSystem

**Examples** (`examples/`):
- `time_series_prediction.py` — Mackey-Glass prediction via CosmosReservoirSystem
- `echo_beats_demo.py` — 12-step cognitive loop demonstration
- `autognosis_demo.py` — 4-layer self-awareness monitoring and adaptation

---

## Quick Start

```python
import sys, os
sys.path.insert(0, 'src')

from models.echo_state_network import EchoStateNetwork, ESNConfig
import numpy as np

config = ESNConfig(n_inputs=5, n_reservoir=200, n_outputs=1,
                   spectral_radius=0.95, leak_rate=0.3, random_seed=42)
esn = EchoStateNetwork(config)

X = np.random.randn(500, 5)
y = np.sum(X, axis=1, keepdims=True)
esn.train(X[:400], y[:400])
preds = esn.run(X[400:])
```

---

## Key Design Principles

1. **Triadic layering**: Input (Autonomic) → Dynamics (Somatic) → Output (Cerebral)
2. **Message passing**: All services communicate via typed `ServiceMessage` objects
3. **Fixed reservoir**: Only `W_out` is trained; `W_in` and `W` are fixed after init
4. **Echo state property**: ρ(W) < 1 guarantees fading memory and convergence
5. **Spectral radius at edge-of-chaos**: ρ ≈ 0.9–0.99 maximises temporal capacity
6. **Self-awareness**: `ReservoirAutognosis` monitors and adapts reservoir parameters at runtime

---

## Dependencies

```
numpy >= 1.24
scipy >= 1.10
pytest >= 7.0  (tests only)
matplotlib     (time_series_prediction.py example only)
```
