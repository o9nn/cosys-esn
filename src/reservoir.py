"""
COSYS-ESN: Reservoir Computing Implementation
==============================================

Cosmos System 5 model applied to Echo State Networks and Reservoir Computing.

This module implements the triadic reservoir architecture:
- Cerebral Triad: Output layer (readout functions)
- Somatic Triad: Reservoir layer (recurrent dynamics)
- Autonomic Triad: Input layer (preprocessing functions)

Author: Cosmos System Enhancement Project
Date: December 29, 2025
License: AGPL-3.0
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from cosmos_core import (
    BaseCosmosService, ServiceConfig, ServiceMessage,
    Triad, Polarity, ServicePosition, Dimension,
    TriadicCoordinator, create_message, setup_logging
)


# ============================================================================
# RESERVOIR CONFIGURATION
# ============================================================================

@dataclass
class ReservoirConfig:
    """Configuration for the reservoir computing system."""
    # Reservoir dimensions
    input_dim: int = 10
    reservoir_dim: int = 100
    output_dim: int = 5
    
    # Reservoir dynamics
    spectral_radius: float = 0.95  # Edge-of-chaos (≈1.0)
    leak_rate: float = 0.3         # Alpha in state update
    sparsity: float = 0.1          # Connection sparsity
    
    # Input/output scaling
    input_scaling: float = 1.0
    output_scaling: float = 1.0
    
    # Training
    ridge_lambda: float = 1e-6     # Ridge regression regularization
    
    # Activation function
    activation: str = "tanh"       # tanh, relu, sigmoid


# ============================================================================
# AUTONOMIC TRIAD: INPUT LAYER
# ============================================================================

class InputPreprocessingService(BaseCosmosService):
    """
    M-1: Input Monitoring
    Validates input, detects noise, scales signals.
    """
    
    def __init__(self, config: ServiceConfig, reservoir_config: ReservoirConfig):
        super().__init__(config)
        self.reservoir_config = reservoir_config
        self.input_scaling = reservoir_config.input_scaling
        
    async def initialize(self) -> None:
        self.log('info', 'Input Preprocessing Service initialized')
        self.initialized = True
        
    async def process(self, message: ServiceMessage) -> Optional[ServiceMessage]:
        if message.type == 'RAW_INPUT':
            # Scale and validate input
            raw_input = np.array(message.payload)
            
            # Input validation
            if raw_input.shape[0] != self.reservoir_config.input_dim:
                self.log('error', f'Invalid input dimension: {raw_input.shape}')
                return None
            
            # Scale input
            scaled_input = raw_input * self.input_scaling
            
            return create_message(
                'PREPROCESSED_INPUT',
                scaled_input,
                self.config.service_name,
                'somatic:M-1'
            )
        return None
    
    async def shutdown(self) -> None:
        self.log('info', 'Input Preprocessing Service shutdown')


class SignalStateService(BaseCosmosService):
    """
    S-8: Signal State Management
    Manages input buffers and temporal windowing.
    """
    
    def __init__(self, config: ServiceConfig, window_size: int = 10):
        super().__init__(config)
        self.window_size = window_size
        self.buffer = []
        
    async def initialize(self) -> None:
        self.log('info', 'Signal State Service initialized')
        self.initialized = True
        
    async def process(self, message: ServiceMessage) -> Optional[ServiceMessage]:
        if message.type == 'PREPROCESSED_INPUT':
            # Add to buffer
            self.buffer.append(message.payload)
            if len(self.buffer) > self.window_size:
                self.buffer.pop(0)
            
            return create_message(
                'BUFFERED_INPUT',
                np.array(self.buffer),
                self.config.service_name
            )
        return None
    
    async def shutdown(self) -> None:
        self.log('info', 'Signal State Service shutdown')


# ============================================================================
# SOMATIC TRIAD: RESERVOIR LAYER
# ============================================================================

class ReservoirDynamicsService(BaseCosmosService):
    """
    P-5: Recurrent Processing
    Core reservoir with sparse recurrent connections and nonlinear dynamics.
    """
    
    def __init__(self, config: ServiceConfig, reservoir_config: ReservoirConfig):
        super().__init__(config)
        self.reservoir_config = reservoir_config
        self.state = None
        self.W_in = None   # Input weights
        self.W = None      # Recurrent weights
        
    async def initialize(self) -> None:
        """Initialize reservoir weights."""
        np.random.seed(42)
        
        # Input weights: uniform random [-1, 1]
        self.W_in = np.random.uniform(
            -1, 1, 
            (self.reservoir_config.reservoir_dim, self.reservoir_config.input_dim)
        )
        
        # Recurrent weights: sparse random matrix
        self.W = self._create_sparse_reservoir()
        
        # Scale to desired spectral radius
        self._scale_spectral_radius()
        
        # Initialize state
        self.state = np.zeros(self.reservoir_config.reservoir_dim)
        
        self.log('info', f'Reservoir initialized: {self.reservoir_config.reservoir_dim} units, '
                        f'spectral radius={self.reservoir_config.spectral_radius:.3f}')
        self.initialized = True
        
    def _create_sparse_reservoir(self) -> np.ndarray:
        """Create sparse random recurrent weight matrix."""
        N = self.reservoir_config.reservoir_dim
        sparsity = self.reservoir_config.sparsity
        
        # Random matrix
        W = np.random.randn(N, N)
        
        # Apply sparsity mask
        mask = np.random.rand(N, N) < sparsity
        W = W * mask
        
        return W
    
    def _scale_spectral_radius(self):
        """Scale recurrent weights to desired spectral radius."""
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvals(self.W)
        current_radius = np.max(np.abs(eigenvalues))
        
        # Scale to target spectral radius
        if current_radius > 0:
            self.W = self.W * (self.reservoir_config.spectral_radius / current_radius)
        
        self.log('debug', f'Spectral radius scaled to {self.reservoir_config.spectral_radius:.3f}')
    
    def _activation(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.reservoir_config.activation == "tanh":
            return np.tanh(x)
        elif self.reservoir_config.activation == "relu":
            return np.maximum(0, x)
        elif self.reservoir_config.activation == "sigmoid":
            return 1 / (1 + np.exp(-x))
        else:
            return x
    
    async def process(self, message: ServiceMessage) -> Optional[ServiceMessage]:
        if message.type == 'PREPROCESSED_INPUT':
            u = message.payload  # Input vector
            
            # Echo state update equation:
            # x(t+1) = (1-α)x(t) + α·f(W_in·u(t+1) + W·x(t))
            alpha = self.reservoir_config.leak_rate
            
            pre_activation = self.W_in @ u + self.W @ self.state
            activated = self._activation(pre_activation)
            
            self.state = (1 - alpha) * self.state + alpha * activated
            
            return create_message(
                'RESERVOIR_STATE',
                self.state.copy(),
                self.config.service_name,
                'cerebral:P-5'
            )
        return None
    
    async def shutdown(self) -> None:
        self.log('info', 'Reservoir Dynamics Service shutdown')


class MembraneInterfaceService(BaseCosmosService):
    """
    M-1: Membrane Interface
    Manages input scaling and boundary conditions.
    """
    
    def __init__(self, config: ServiceConfig):
        super().__init__(config)
        
    async def initialize(self) -> None:
        self.log('info', 'Membrane Interface Service initialized')
        self.initialized = True
        
    async def process(self, message: ServiceMessage) -> Optional[ServiceMessage]:
        # Pass through to reservoir
        return message
    
    async def shutdown(self) -> None:
        self.log('info', 'Membrane Interface Service shutdown')


class ReservoirStateService(BaseCosmosService):
    """
    S-8: State Management
    Maintains reservoir state vector and echo persistence.
    """
    
    def __init__(self, config: ServiceConfig):
        super().__init__(config)
        self.state_history = []
        
    async def initialize(self) -> None:
        self.log('info', 'Reservoir State Service initialized')
        self.initialized = True
        
    async def process(self, message: ServiceMessage) -> Optional[ServiceMessage]:
        if message.type == 'RESERVOIR_STATE':
            # Store state history
            self.state_history.append(message.payload)
            return message
        return None
    
    async def shutdown(self) -> None:
        self.log('info', 'Reservoir State Service shutdown')


# ============================================================================
# CEREBRAL TRIAD: OUTPUT LAYER
# ============================================================================

class ReadoutProcessingService(BaseCosmosService):
    """
    P-5: Readout Processing
    Linear combination of reservoir states for output.
    """
    
    def __init__(self, config: ServiceConfig, reservoir_config: ReservoirConfig):
        super().__init__(config)
        self.reservoir_config = reservoir_config
        self.W_out = None  # Output weights (trained)
        
    async def initialize(self) -> None:
        # Initialize with zeros (will be trained)
        self.W_out = np.zeros((
            self.reservoir_config.output_dim,
            self.reservoir_config.reservoir_dim
        ))
        self.log('info', 'Readout Processing Service initialized')
        self.initialized = True
        
    def train(self, states: np.ndarray, targets: np.ndarray):
        """
        Train output weights using ridge regression.
        
        Args:
            states: (n_samples, reservoir_dim) reservoir states
            targets: (n_samples, output_dim) target outputs
        """
        ridge_lambda = self.reservoir_config.ridge_lambda
        
        # Ridge regression: W_out = Y^T X (X^T X + λI)^-1
        X = states
        Y = targets
        
        # Add bias term
        X_bias = np.hstack([X, np.ones((X.shape[0], 1))])
        
        # Solve ridge regression
        identity = np.eye(X_bias.shape[1])
        self.W_out = Y.T @ X_bias @ np.linalg.inv(
            X_bias.T @ X_bias + ridge_lambda * identity
        )
        
        self.log('info', f'Output weights trained: {self.W_out.shape}')
        
    async def process(self, message: ServiceMessage) -> Optional[ServiceMessage]:
        if message.type == 'RESERVOIR_STATE':
            state = message.payload
            
            # Add bias
            state_bias = np.append(state, 1.0)
            
            # Compute output
            output = self.W_out @ state_bias
            
            return create_message(
                'READOUT_OUTPUT',
                output,
                self.config.service_name,
                'cerebral:O-4'
            )
        return None
    
    async def shutdown(self) -> None:
        self.log('info', 'Readout Processing Service shutdown')


class OutputOrganizationService(BaseCosmosService):
    """
    O-4: Output Organization
    Structures and formats the final output.
    """
    
    def __init__(self, config: ServiceConfig):
        super().__init__(config)
        
    async def initialize(self) -> None:
        self.log('info', 'Output Organization Service initialized')
        self.initialized = True
        
    async def process(self, message: ServiceMessage) -> Optional[ServiceMessage]:
        if message.type == 'READOUT_OUTPUT':
            # Format output
            output = message.payload
            
            return create_message(
                'FINAL_OUTPUT',
                {
                    'output': output,
                    'timestamp': message.timestamp,
                    'source_triad': 'cerebral'
                },
                self.config.service_name
            )
        return None
    
    async def shutdown(self) -> None:
        self.log('info', 'Output Organization Service shutdown')


# ============================================================================
# RESERVOIR SYSTEM
# ============================================================================

class CosmosReservoirSystem:
    """
    Complete Cosmos System 5 Reservoir Computing implementation.
    
    Integrates all three triads:
    - Autonomic: Input preprocessing
    - Somatic: Reservoir dynamics
    - Cerebral: Output readout
    """
    
    def __init__(self, reservoir_config: ReservoirConfig):
        self.reservoir_config = reservoir_config
        self.coordinator = TriadicCoordinator()
        self.services = {}
        
    async def initialize(self):
        """Initialize all services."""
        # Autonomic Triad (Input)
        input_service = InputPreprocessingService(
            ServiceConfig(
                "input-preprocessing",
                Triad.AUTONOMIC,
                ServicePosition.M1,
                Polarity.SYMPATHETIC,
                Dimension.PERFORMANCE
            ),
            self.reservoir_config
        )
        await input_service.initialize()
        self.coordinator.register_service(input_service)
        self.services['input'] = input_service
        
        # Somatic Triad (Reservoir)
        reservoir_service = ReservoirDynamicsService(
            ServiceConfig(
                "reservoir-dynamics",
                Triad.SOMATIC,
                ServicePosition.P5,
                Polarity.SOMATIC,
                Dimension.COMMITMENT
            ),
            self.reservoir_config
        )
        await reservoir_service.initialize()
        self.coordinator.register_service(reservoir_service)
        self.services['reservoir'] = reservoir_service
        
        # Cerebral Triad (Output)
        readout_service = ReadoutProcessingService(
            ServiceConfig(
                "readout-processing",
                Triad.CEREBRAL,
                ServicePosition.P5,
                Polarity.SOMATIC,
                Dimension.COMMITMENT
            ),
            self.reservoir_config
        )
        await readout_service.initialize()
        self.coordinator.register_service(readout_service)
        self.services['readout'] = readout_service
        
        output_service = OutputOrganizationService(
            ServiceConfig(
                "output-organization",
                Triad.CEREBRAL,
                ServicePosition.O4,
                Polarity.SOMATIC,
                Dimension.COMMITMENT
            )
        )
        await output_service.initialize()
        self.coordinator.register_service(output_service)
        self.services['output'] = output_service
        
        print("✓ Cosmos Reservoir System initialized")
        print(f"  - Input dim: {self.reservoir_config.input_dim}")
        print(f"  - Reservoir dim: {self.reservoir_config.reservoir_dim}")
        print(f"  - Output dim: {self.reservoir_config.output_dim}")
        print(f"  - Spectral radius: {self.reservoir_config.spectral_radius}")
    
    async def process_input(self, input_data: np.ndarray) -> np.ndarray:
        """Process a single input through the reservoir."""
        # Create input message
        msg = create_message('RAW_INPUT', input_data, 'external')
        
        # Process through input service
        msg = await self.services['input'].process(msg)
        
        # Process through reservoir
        msg = await self.services['reservoir'].process(msg)
        
        # Process through readout
        msg = await self.services['readout'].process(msg)
        
        # Process through output organization
        msg = await self.services['output'].process(msg)
        
        return msg.payload['output']
    
    def train(self, inputs: np.ndarray, targets: np.ndarray):
        """
        Train the reservoir system.
        
        Args:
            inputs: (n_samples, input_dim) input sequences
            targets: (n_samples, output_dim) target outputs
        """
        import asyncio
        
        # Collect reservoir states
        states = []
        for inp in inputs:
            # Run through reservoir
            asyncio.run(self.services['input'].process(
                create_message('RAW_INPUT', inp, 'external')
            ))
            msg = asyncio.run(self.services['reservoir'].process(
                create_message('PREPROCESSED_INPUT', inp, 'input')
            ))
            states.append(msg.payload)
        
        states = np.array(states)
        
        # Train readout
        self.services['readout'].train(states, targets)
        
        print(f"✓ Reservoir trained on {len(inputs)} samples")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import asyncio
    
    setup_logging("INFO")
    
    print("=== COSYS-ESN: Cosmos Reservoir Computing Demo ===\n")
    
    # Create reservoir configuration
    config = ReservoirConfig(
        input_dim=5,
        reservoir_dim=50,
        output_dim=3,
        spectral_radius=0.95,
        leak_rate=0.3
    )
    
    # Create system
    system = CosmosReservoirSystem(config)
    asyncio.run(system.initialize())
    
    print("\n--- Generating training data ---")
    # Generate simple training data (sine waves)
    n_samples = 100
    t = np.linspace(0, 10, n_samples)
    inputs = np.column_stack([
        np.sin(t),
        np.cos(t),
        np.sin(2*t),
        np.cos(2*t),
        np.sin(3*t)
    ])
    targets = np.column_stack([
        np.sin(t + 0.5),
        np.cos(t + 0.5),
        np.sin(2*t + 0.5)
    ])
    
    print(f"Training data: {inputs.shape} → {targets.shape}")
    
    # Train
    print("\n--- Training reservoir ---")
    system.train(inputs, targets)
    
    # Test
    print("\n--- Testing reservoir ---")
    test_input = inputs[0]
    output = asyncio.run(system.process_input(test_input))
    
    print(f"Input: {test_input}")
    print(f"Output: {output}")
    print(f"Target: {targets[0]}")
    print(f"Error: {np.mean((output - targets[0])**2):.6f}")
    
    print("\n✓ COSYS-ESN demonstration complete")
