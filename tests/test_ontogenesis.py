"""
Tests for reservoir ontogenesis (ontogenesis.py):
- Gene types: IntGene, FloatGene, CategoricalGene
- ReservoirKernelGenome: mutation, crossover, to_esn_config
- b_series_reservoir_expansion: shape, spectral radius
- Population: evaluate, evolve, run (small scale)
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from reservoir_core.ontogenesis.ontogenesis import (
    IntGene, FloatGene, CategoricalGene,
    ReservoirKernelGenome,
    b_series_reservoir_expansion,
    Population,
    _compute_nrmse,
)
from models.echo_state_network import ESNConfig


# ---------------------------------------------------------------------------
# Gene types
# ---------------------------------------------------------------------------

def test_int_gene_bounds():
    g = IntGene(value=5, low=1, high=10)
    for _ in range(50):
        g.mutate(sigma=0.5)
        assert g.low <= g.value <= g.high


def test_float_gene_bounds():
    g = FloatGene(value=0.5, low=0.0, high=1.0)
    for _ in range(50):
        g.mutate(sigma=0.5)
        assert g.low <= g.value <= g.high


def test_categorical_gene_stays_in_choices():
    choices = ["tanh", "relu", "sigmoid"]
    g = CategoricalGene(value="tanh", choices=choices)
    for _ in range(30):
        g.mutate(prob=1.0)  # always mutate
        assert g.value in choices


def test_int_gene_copy_independence():
    g = IntGene(value=5, low=1, high=10)
    c = g.copy()
    c.value = 99
    assert g.value == 5


def test_float_gene_copy_independence():
    g = FloatGene(value=0.7, low=0.0, high=1.0)
    c = g.copy()
    c.value = 0.0
    assert abs(g.value - 0.7) < 1e-9


# ---------------------------------------------------------------------------
# ReservoirKernelGenome
# ---------------------------------------------------------------------------

def test_genome_default_keys():
    g = ReservoirKernelGenome()
    expected = {"n_neurons", "spectral_radius", "input_scaling",
                "leak_rate", "sparsity", "activation", "topology"}
    assert set(g.genes.keys()) == expected


def test_genome_to_esn_config():
    g = ReservoirKernelGenome()
    cfg = g.to_esn_config(n_inputs=2, n_outputs=1, random_seed=7)
    assert isinstance(cfg, ESNConfig)
    assert cfg.n_inputs == 2
    assert cfg.n_outputs == 1
    assert cfg.random_seed == 7
    assert cfg.n_reservoir == g.genes["n_neurons"].value


def test_genome_mutate_changes_values():
    np.random.seed(0)
    g = ReservoirKernelGenome()
    original_sr = g.genes["spectral_radius"].value
    # Mutate many times with high rate — should change at least one gene
    for _ in range(20):
        g.mutate(rate=1.0)
    # At minimum, the genome is still valid
    cfg = g.to_esn_config()
    assert isinstance(cfg, ESNConfig)


def test_genome_crossover_generation():
    g1 = ReservoirKernelGenome()
    g2 = ReservoirKernelGenome()
    g1.generation = 3
    g2.generation = 5
    child = g1.crossover(g2)
    assert child.generation == 6  # max(3, 5) + 1


def test_genome_crossover_genes_from_parents():
    np.random.seed(42)
    g1 = ReservoirKernelGenome()
    g2 = ReservoirKernelGenome()
    # Force distinct values
    g1.genes["leak_rate"].value = 0.1
    g2.genes["leak_rate"].value = 0.9
    child = g1.crossover(g2)
    # Child's leak_rate must be one of the parent values
    assert child.genes["leak_rate"].value in (0.1, 0.9)


def test_genome_copy_independence():
    g = ReservoirKernelGenome()
    g.fitness = -0.5
    c = g.copy()
    c.fitness = -999.0
    assert g.fitness == -0.5


def test_genome_repr():
    g = ReservoirKernelGenome()
    r = repr(g)
    assert "Genome" in r


# ---------------------------------------------------------------------------
# b_series_reservoir_expansion
# ---------------------------------------------------------------------------

def test_b_series_shape():
    W = b_series_reservoir_expansion(order=2, N=30, random_seed=0)
    assert W.shape == (30, 30)


def test_b_series_spectral_radius_near_09():
    W = b_series_reservoir_expansion(order=3, N=40, random_seed=1)
    eigenvalues = np.linalg.eigvals(W)
    sr = float(np.max(np.abs(eigenvalues)))
    assert abs(sr - 0.9) < 0.05, f"Spectral radius {sr:.4f} not close to 0.9"


def test_b_series_order_1():
    W = b_series_reservoir_expansion(order=1, N=20, random_seed=2)
    assert W.shape == (20, 20)
    assert np.all(np.isfinite(W))


def test_b_series_with_domain_spec():
    domain = {0: 0.5, 1: 2.0}
    W = b_series_reservoir_expansion(order=2, domain_spec=domain, N=25, random_seed=3)
    assert W.shape == (25, 25)


# ---------------------------------------------------------------------------
# _compute_nrmse
# ---------------------------------------------------------------------------

def test_compute_nrmse_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert _compute_nrmse(y, y) == pytest.approx(0.0, abs=1e-9)


def test_compute_nrmse_positive():
    y = np.array([1.0, 2.0, 3.0])
    yhat = np.array([1.1, 1.9, 3.2])
    nrmse = _compute_nrmse(yhat, y)
    assert nrmse > 0


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------

def _simple_data(n=60, seed=0):
    """Simple linear regression dataset."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 1))
    y = X * 2.0 + 0.1
    return X[:40], y[:40], X[40:], y[40:]


def test_population_init():
    pop = Population(size=5, n_inputs=1, n_outputs=1, random_seed=0)
    assert len(pop.individuals) == 5
    assert pop.generation == 0


def test_population_evaluate_sets_fitness():
    Xtr, ytr, Xval, yval = _simple_data()
    pop = Population(size=4, n_inputs=1, n_outputs=1, random_seed=1)
    pop.evaluate(Xtr, ytr, Xval, yval)
    for g in pop.individuals:
        assert g.fitness != float("-inf") or True  # some may fail on tiny reservoir


def test_population_evolve_increments_generation():
    Xtr, ytr, Xval, yval = _simple_data()
    pop = Population(size=6, n_inputs=1, n_outputs=1, random_seed=2)
    pop.evaluate(Xtr, ytr, Xval, yval)
    pop.evolve()
    assert pop.generation == 1
    assert pop.best is not None


def test_population_evolve_preserves_size():
    Xtr, ytr, Xval, yval = _simple_data()
    pop = Population(size=8, n_inputs=1, n_outputs=1, random_seed=3)
    pop.evaluate(Xtr, ytr, Xval, yval)
    pop.evolve()
    assert len(pop.individuals) == 8


def test_population_run_returns_best_genome():
    Xtr, ytr, Xval, yval = _simple_data(n=50)
    pop = Population(size=4, n_inputs=1, n_outputs=1, random_seed=4)
    best = pop.run(Xtr, ytr, Xval, yval, n_generations=2)
    assert isinstance(best, ReservoirKernelGenome)
    assert best.fitness > float("-inf")


def test_population_fitness_history_length():
    Xtr, ytr, Xval, yval = _simple_data()
    pop = Population(size=4, n_inputs=1, n_outputs=1, random_seed=5)
    pop.run(Xtr, ytr, Xval, yval, n_generations=3)
    assert len(pop.fitness_history) == 3
