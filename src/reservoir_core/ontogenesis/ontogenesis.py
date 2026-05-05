"""
Reservoir Ontogenesis: Evolutionary self-generation of reservoir configurations.

Implements:
- Gene types: IntGene, FloatGene, CategoricalGene
- ReservoirKernelGenome: full reservoir configuration genome
- Population: generational evolution with fitness = -NRMSE
- b_series_reservoir_expansion: B-series connectivity patterns
"""
import copy
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from models.echo_state_network import EchoStateNetwork, ESNConfig


# =============================================================================
# Gene Types
# =============================================================================

@dataclass
class IntGene:
    value: int
    low: int
    high: int

    def mutate(self, sigma: float = 0.1) -> None:
        delta = int(np.round(np.random.randn() * sigma * (self.high - self.low)))
        self.value = int(np.clip(self.value + delta, self.low, self.high))

    def copy(self) -> "IntGene":
        return copy.copy(self)


@dataclass
class FloatGene:
    value: float
    low: float
    high: float

    def mutate(self, sigma: float = 0.05) -> None:
        delta = np.random.randn() * sigma * (self.high - self.low)
        self.value = float(np.clip(self.value + delta, self.low, self.high))

    def copy(self) -> "FloatGene":
        return copy.copy(self)


@dataclass
class CategoricalGene:
    value: Any
    choices: List[Any]

    def mutate(self, prob: float = 0.2) -> None:
        if np.random.random() < prob:
            self.value = np.random.choice(self.choices)

    def copy(self) -> "CategoricalGene":
        return copy.copy(self)


# =============================================================================
# Genome
# =============================================================================

class ReservoirKernelGenome:
    """Evolutionary genome for reservoir configurations."""

    def __init__(self):
        self.genes: Dict[str, Any] = {
            "n_neurons":       IntGene(200, 50, 2000),
            "spectral_radius": FloatGene(0.9, 0.5, 1.05),
            "input_scaling":   FloatGene(0.5, 0.01, 2.0),
            "leak_rate":       FloatGene(0.3, 0.01, 1.0),
            "sparsity":        FloatGene(0.1, 0.01, 0.5),
            "activation":      CategoricalGene("tanh", ["tanh", "relu", "sigmoid"]),
            "topology":        CategoricalGene("random", ["random", "small_world", "scale_free"]),
        }
        self.fitness: float = float("-inf")
        self.generation: int = 0

    def to_esn_config(self, n_inputs: int = 1, n_outputs: int = 1,
                       random_seed: Optional[int] = None) -> ESNConfig:
        """Convert genome to ESNConfig."""
        return ESNConfig(
            n_inputs=n_inputs,
            n_reservoir=self.genes["n_neurons"].value,
            n_outputs=n_outputs,
            spectral_radius=self.genes["spectral_radius"].value,
            input_scaling=self.genes["input_scaling"].value,
            leak_rate=self.genes["leak_rate"].value,
            sparsity=self.genes["sparsity"].value,
            activation=self.genes["activation"].value,
            topology=self.genes["topology"].value,
            random_seed=random_seed,
        )

    def mutate(self, rate: float = 0.1) -> None:
        """Apply random mutations to genome."""
        for gene in self.genes.values():
            if np.random.random() < rate:
                gene.mutate()

    def crossover(self, other: "ReservoirKernelGenome") -> "ReservoirKernelGenome":
        """Create offspring through uniform crossover."""
        child = ReservoirKernelGenome()
        for key in self.genes:
            child.genes[key] = (self.genes[key].copy() if np.random.random() < 0.5
                                 else other.genes[key].copy())
        child.generation = max(self.generation, other.generation) + 1
        return child

    def copy(self) -> "ReservoirKernelGenome":
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        return (f"Genome(n={self.genes['n_neurons'].value}, "
                f"sr={self.genes['spectral_radius'].value:.3f}, "
                f"lr={self.genes['leak_rate'].value:.3f}, "
                f"fitness={self.fitness:.4f})")


# =============================================================================
# B-Series Reservoir Expansion
# =============================================================================

def _elementary_differentials(order: int) -> List[List[int]]:
    """
    Generate elementary differential trees up to given order.
    These are rooted trees counted by A000081 sequence.
    Returns simplified representation as ordered degree sequences.
    """
    if order <= 0:
        return [[]]
    trees = [[]]  # order 0: empty tree
    if order >= 1:
        trees.append([1])  # order 1: single node
    if order >= 2:
        trees.append([2])  # order 2: node with 2 children
        trees.append([1, 1])
    if order >= 3:
        trees.append([3])
        trees.append([2, 1])
        trees.append([1, 1, 1])
    return trees[:order + 1]


def b_series_reservoir_expansion(order: int, domain_spec: Optional[Dict] = None,
                                  N: int = 100, random_seed: int = 0) -> np.ndarray:
    """
    Generate reservoir connectivity matrix using B-series expansion.

    The elementary differentials (rooted trees) map to structured
    connectivity patterns that encode computational trees.

    Parameters
    ----------
    order       : expansion order (1-5)
    domain_spec : optional domain-specific weights dict
    N           : reservoir size
    random_seed : RNG seed

    Returns
    -------
    W : (N, N) connectivity matrix
    """
    rng = np.random.default_rng(random_seed)
    trees = _elementary_differentials(order)
    default_weights = {i: 1.0 / max(1, len(t)) for i, t in enumerate(trees)}
    if domain_spec:
        weights = {**default_weights, **domain_spec}
    else:
        weights = default_weights

    W = np.zeros((N, N))
    for tree_idx, tree in enumerate(trees):
        if not tree:
            continue
        w = weights.get(tree_idx, 1.0 / len(tree))
        # Map tree structure to connectivity pattern
        if len(tree) == 1:
            # Single-degree: circulant shift
            shift = tree[0] % N
            W += w * np.roll(np.eye(N), shift, axis=1)
        else:
            # Multi-degree: random sparse pattern weighted by tree degree
            density = min(0.3, len(tree) * 0.05)
            mask = rng.random((N, N)) < density
            W += w * mask * rng.uniform(-1, 1, (N, N))

    # Normalise spectral radius to 0.9
    import scipy.sparse as _sp
    import scipy.sparse.linalg as _spla
    W_sparse = _sp.csr_matrix(W)
    try:
        k = min(6, W.shape[0] - 2)
        if k < 1:
            raise ValueError
        eigs = _spla.eigs(W_sparse, k=k, which="LM", return_eigenvectors=False)
        norm = float(np.max(np.abs(eigs))) + 1e-8
    except Exception:
        norm = float(np.max(np.abs(np.linalg.eigvals(W)))) + 1e-8
    W = W * (0.9 / norm)
    return W


# =============================================================================
# Population (Generational Evolution)
# =============================================================================

def _compute_nrmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    return float(rmse / (np.std(targets) + 1e-8))


class Population:
    """
    Generational evolutionary optimizer for reservoir configurations.
    Fitness = negative NRMSE on a provided validation set.
    """

    def __init__(self, size: int = 20, n_inputs: int = 1, n_outputs: int = 1,
                 elite_fraction: float = 0.2, mutation_rate: float = 0.1,
                 random_seed: Optional[int] = None):
        self.size = size
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.elite_fraction = elite_fraction
        self.mutation_rate = mutation_rate
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
        self.individuals: List[ReservoirKernelGenome] = [
            ReservoirKernelGenome() for _ in range(size)
        ]
        self.generation: int = 0
        self.best: Optional[ReservoirKernelGenome] = None
        self.fitness_history: List[float] = []

    def evaluate(self, train_inputs: np.ndarray, train_targets: np.ndarray,
                 val_inputs: np.ndarray, val_targets: np.ndarray) -> None:
        """Evaluate all individuals on the validation set."""
        for genome in self.individuals:
            config = genome.to_esn_config(
                n_inputs=self.n_inputs, n_outputs=self.n_outputs,
                random_seed=self.random_seed)
            try:
                esn = EchoStateNetwork(config)
                esn.train(train_inputs, train_targets)
                preds = esn.run(val_inputs)
                nrmse = _compute_nrmse(preds, val_targets)
                genome.fitness = -nrmse  # maximize fitness = minimize NRMSE
            except Exception:
                genome.fitness = float("-inf")

    def evolve(self) -> None:
        """One generation: selection, crossover, mutation."""
        self.individuals.sort(key=lambda g: g.fitness, reverse=True)
        self.best = self.individuals[0].copy()
        self.fitness_history.append(self.best.fitness)

        n_elite = max(1, int(self.size * self.elite_fraction))
        elites = [g.copy() for g in self.individuals[:n_elite]]

        # Fill rest with crossover + mutation
        new_pop = elites[:]
        while len(new_pop) < self.size:
            parents = np.random.choice(elites, size=2, replace=len(elites) < 2)
            child = parents[0].crossover(parents[1])
            child.mutate(self.mutation_rate)
            new_pop.append(child)

        self.individuals = new_pop
        self.generation += 1

    def run(self, train_inputs: np.ndarray, train_targets: np.ndarray,
            val_inputs: np.ndarray, val_targets: np.ndarray,
            n_generations: int = 10) -> ReservoirKernelGenome:
        """
        Run evolution for n_generations.

        Returns best genome found.
        """
        for gen in range(n_generations):
            self.evaluate(train_inputs, train_targets, val_inputs, val_targets)
            self.evolve()
            best_nrmse = -self.best.fitness if self.best else float("inf")
            print(f"[Gen {gen+1}/{n_generations}] Best NRMSE: {best_nrmse:.4f} | {self.best}")
        return self.best
