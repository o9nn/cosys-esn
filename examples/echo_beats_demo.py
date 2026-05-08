"""
COSYS-ESN Example: EchoBeats 12-Step Cognitive Loop Demo
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from models.echo_state_network import EchoStateNetwork, ESNConfig
from reservoir_core.echo_beats.echo_beats import EchoBeatsLoop, Phase


def main():
    print("=" * 60)
    print("COSYS-ESN: EchoBeats 12-Step Cognitive Loop Demo")
    print("=" * 60)

    config = ESNConfig(
        n_inputs=5, n_reservoir=100, n_outputs=1,
        spectral_radius=0.95, leak_rate=0.3, random_seed=42,
    )
    esn = EchoStateNetwork(config)

    # Train on simple sine wave
    t = np.linspace(0, 20, 500)
    inputs = np.column_stack([np.sin(t + i * 0.5) for i in range(5)])
    targets = np.sin(t + 1.0).reshape(-1, 1)
    esn.train(inputs, targets)
    print("✓ ESN trained on sine wave data")

    # Run one full 12-step cognitive cycle
    loop = EchoBeatsLoop(esn)
    input_seq = inputs[:12]
    print("\n[Running 12-step cognitive cycle]")
    cycle_outputs = loop.run_cycle(input_seq)

    for step_idx, step_out in enumerate(cycle_outputs):
        phases = [step_out[f"stream_{i}"]["phase"] for i in range(3)]
        print(f"  Step {step_idx+1:2d}: Stream phases = {phases}")

    integrated = cycle_outputs[-1]["integrated_state"]
    print(f"\n✓ Final integrated state norm: {np.linalg.norm(integrated):.4f}")
    print("✓ EchoBeats demo complete")


if __name__ == "__main__":
    main()
