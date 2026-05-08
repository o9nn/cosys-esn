"""
COSYS-ESN Example: Reservoir Autognosis (Self-Awareness) Demo
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from models.echo_state_network import EchoStateNetwork, ESNConfig
from reservoir_core.autognosis.autognosis import ReservoirAutognosis


def main():
    print("=" * 60)
    print("COSYS-ESN: Reservoir Autognosis Demo")
    print("=" * 60)

    config = ESNConfig(
        n_inputs=3, n_reservoir=100, n_outputs=1,
        spectral_radius=0.9, leak_rate=0.3, random_seed=42,
    )
    esn = EchoStateNetwork(config)
    ag = ReservoirAutognosis(esn, window_size=50)

    # Simulate streaming inputs
    print("\n[Phase 1: Monitoring reservoir health over 100 steps]")
    for t in range(100):
        u = np.array([np.sin(t * 0.1), np.cos(t * 0.1), np.sin(t * 0.2)])
        esn.update(u)
        if t % 20 == 19:
            metrics = ag.monitor()
            print(f"  t={t+1:3d} | norm={metrics['state_norm']:.3f} | "
                  f"SR={metrics['spectral_radius']:.3f} | "
                  f"echo_idx={metrics['echo_index']:.1f}")

    print("\n[Phase 2: Capacity and quality estimation]")
    cap = ag.capacity_estimation()
    kq = ag.kernel_quality()
    md = ag.memory_depth()
    print(f"  Capacity estimate:    {cap:.2f}")
    print(f"  Kernel quality:       sep={kq['separation']:.3f}, approx={kq['approximation']:.3f}")
    print(f"  Effective memory:     {md} timesteps")

    print("\n[Phase 3: Meta-cognitive assessment]")
    perf = ag.performance_prediction()
    conf = ag.confidence_scoring()
    anomaly = ag.anomaly_detection()
    adapt = ag.should_adapt()
    print(f"  Performance score:    {perf:.3f}")
    print(f"  Confidence score:     {conf:.3f}")
    print(f"  Anomaly detected:     {anomaly}")
    print(f"  Needs adaptation:     {adapt}")

    if adapt:
        print("\n[Phase 4: Self-optimization]")
        new_sr = ag.tune_spectral_radius()
        new_lr = ag.adapt_leak_rate()
        new_is = ag.optimize_input_scaling()
        topo_rec = ag.refine_topology()
        print(f"  New spectral radius: {new_sr:.3f}")
        print(f"  New leak rate:       {new_lr:.3f}")
        print(f"  New input scaling:   {new_is:.3f}")
        print(f"  Topology suggestion: {topo_rec}")

    print("\n✓ Autognosis demo complete")


if __name__ == "__main__":
    main()
