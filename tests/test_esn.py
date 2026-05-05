"""Integration tests for EchoStateNetwork: Mackey-Glass NRMSE < 0.1."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from models.echo_state_network import EchoStateNetwork, ESNConfig


def generate_mackey_glass(n=2000, tau=17, beta=0.2, gamma=0.1, n_exp=10):
    x = np.zeros(n)
    x[0] = 1.2
    for t in range(tau, n - 1):
        x[t + 1] = x[t] + beta * x[t - tau] / (1 + x[t - tau]**n_exp) - gamma * x[t]
    return x


def make_dataset(data, input_len=5, horizon=1):
    n = len(data) - input_len - horizon + 1
    X = np.array([data[i:i + input_len] for i in range(n)])
    y = np.array([[data[i + input_len + horizon - 1]] for i in range(n)])
    return X, y


def nrmse(pred, target):
    return float(np.sqrt(np.mean((pred - target)**2)) / (np.std(target) + 1e-8))


def test_esn_mackey_glass():
    """NRMSE on Mackey-Glass should be < 0.1 with a well-configured ESN."""
    mg = generate_mackey_glass(n=2000)
    X, y = make_dataset(mg, input_len=5)

    split = 1000
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    config = ESNConfig(
        n_inputs=5, n_reservoir=200, n_outputs=1,
        spectral_radius=0.95, leak_rate=0.3, sparsity=0.1,
        ridge_lambda=1e-6, washout=100, random_seed=42,
    )
    esn = EchoStateNetwork(config)
    esn.train(X_train, y_train)

    # Don't reset state — continue from end-of-training state for warmup
    preds = esn.run(X_test)
    error = nrmse(preds, y_test)
    print(f"Mackey-Glass NRMSE: {error:.4f}")
    assert error < 0.1, f"NRMSE={error:.4f} >= 0.1"


def test_esn_topologies():
    """All 3 topologies should train and predict without error."""
    X = np.random.randn(200, 3)
    y = np.sum(X, axis=1, keepdims=True)

    for topo in ["random", "small_world", "scale_free"]:
        config = ESNConfig(n_inputs=3, n_reservoir=50, n_outputs=1,
                           topology=topo, washout=10, random_seed=1)
        esn = EchoStateNetwork(config)
        esn.train(X, y)
        preds = esn.run(X[:10])
        assert preds.shape == (10, 1), f"Topology {topo} failed"


def test_esn_save_load(tmp_path):
    config = ESNConfig(n_inputs=2, n_reservoir=20, n_outputs=1, random_seed=0)
    esn = EchoStateNetwork(config)
    X = np.random.randn(50, 2)
    y = np.random.randn(50, 1)
    esn.train(X, y)

    path = str(tmp_path / "esn.npz")
    esn.save(path)

    esn2 = EchoStateNetwork.load(path, config)
    np.testing.assert_array_equal(esn.W_out, esn2.W_out)
    np.testing.assert_array_equal(esn.W_in, esn2.W_in)
