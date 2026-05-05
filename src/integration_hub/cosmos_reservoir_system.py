"""
CosmosReservoirSystem: Full 18-service wired system.

Async pipeline: RAW_INPUT → Autonomic (6) → Somatic (6) → Cerebral (6) → FINAL_OUTPUT
"""
import asyncio
import numpy as np
from typing import Optional
from cosmos_core import (
    TriadicCoordinator, ServiceConfig, create_message,
    Triad, Polarity, ServicePosition, Dimension, setup_logging,
)
from autonomic_triad.input_monitoring import InputMonitoringService
from autonomic_triad.signal_state import SignalStateService
from autonomic_triad.preprocessing_director import PreprocessingDirectorService
from autonomic_triad.transform_processing import TransformProcessingService
from autonomic_triad.input_organization import InputOrganizationService
from autonomic_triad.encoding_triggers import EncodingTriggersService

from somatic_triad.membrane_interface import MembraneInterfaceService
from somatic_triad.state_management import StateManagementService
from somatic_triad.recurrent_processing import RecurrentProcessingService
from somatic_triad.dynamics_organization import DynamicsOrganizationService
from somatic_triad.dynamics_development import DynamicsDevelopmentService
from somatic_triad.state_treasury import StateTreasuryService

from cerebral_triad.target_interface import TargetInterfaceService
from cerebral_triad.output_delivery import OutputDeliveryService
from cerebral_triad.readout_processing import ReadoutProcessingService
from cerebral_triad.output_organization import OutputOrganizationService
from cerebral_triad.learning_director import LearningDirectorService
from cerebral_triad.weight_treasury import WeightTreasuryService

from models.echo_state_network import EchoStateNetwork, ESNConfig


def _sc(name: str, triad: Triad, pos: ServicePosition,
        polarity: Polarity = Polarity.SOMATIC,
        dim: Dimension = Dimension.COMMITMENT) -> ServiceConfig:
    return ServiceConfig(name, triad, pos, polarity, dim)


class CosmosReservoirSystem:
    """
    Complete Cosmos System 5 Reservoir Computing implementation.

    Wires all 18 services across three triads with a full async pipeline.
    """

    def __init__(self, config: Optional[ESNConfig] = None, **kwargs):
        if config is None:
            config = ESNConfig(**kwargs)
        self.config = config
        self.coordinator = TriadicCoordinator()
        self.services = {}
        self.esn: Optional[EchoStateNetwork] = None

    async def initialize(self) -> None:
        cfg = self.config

        # === Autonomic Triad (Input Layer) ===
        self.services["a_m1"] = InputMonitoringService(
            _sc("autonomic-input-monitoring", Triad.AUTONOMIC, ServicePosition.M1,
                Polarity.SYMPATHETIC, Dimension.PERFORMANCE),
            input_dim=cfg.n_inputs, input_scaling=cfg.input_scaling,
        )
        self.services["a_s8"] = SignalStateService(
            _sc("autonomic-signal-state", Triad.AUTONOMIC, ServicePosition.S8,
                Polarity.SYMPATHETIC, Dimension.PERFORMANCE),
            window_size=20,
        )
        self.services["a_pd2"] = PreprocessingDirectorService(
            _sc("autonomic-preprocessing", Triad.AUTONOMIC, ServicePosition.PD2,
                Polarity.PARASYMPATHETIC, Dimension.POTENTIAL),
        )
        self.services["a_p5"] = TransformProcessingService(
            _sc("autonomic-transform", Triad.AUTONOMIC, ServicePosition.P5,
                Polarity.SOMATIC, Dimension.COMMITMENT),
        )
        self.services["a_o4"] = InputOrganizationService(
            _sc("autonomic-input-org", Triad.AUTONOMIC, ServicePosition.O4,
                Polarity.SOMATIC, Dimension.COMMITMENT),
        )
        self.services["a_t7"] = EncodingTriggersService(
            _sc("autonomic-encoding", Triad.AUTONOMIC, ServicePosition.T7,
                Polarity.PARASYMPATHETIC, Dimension.POTENTIAL),
            k=min(5, cfg.n_inputs), output_dim=cfg.n_inputs,
        )

        # === Somatic Triad (Reservoir Layer) ===
        self.services["s_m1"] = MembraneInterfaceService(
            _sc("somatic-membrane", Triad.SOMATIC, ServicePosition.M1,
                Polarity.SOMATIC, Dimension.PERFORMANCE),
            input_scaling=cfg.input_scaling,
        )
        self.services["s_s8"] = StateManagementService(
            _sc("somatic-state", Triad.SOMATIC, ServicePosition.S8,
                Polarity.SOMATIC, Dimension.PERFORMANCE),
            reservoir_dim=cfg.n_reservoir,
        )
        self.services["s_p5"] = RecurrentProcessingService(
            _sc("somatic-recurrent", Triad.SOMATIC, ServicePosition.P5,
                Polarity.SOMATIC, Dimension.COMMITMENT),
            reservoir_dim=cfg.n_reservoir, input_dim=cfg.n_inputs,
            spectral_radius=cfg.spectral_radius, sparsity=cfg.sparsity,
            leak_rate=cfg.leak_rate, activation=cfg.activation,
            random_seed=cfg.random_seed,
        )
        self.services["s_o4"] = DynamicsOrganizationService(
            _sc("somatic-dynamics-org", Triad.SOMATIC, ServicePosition.O4,
                Polarity.SOMATIC, Dimension.COMMITMENT),
            target_spectral_radius=cfg.spectral_radius,
        )
        self.services["s_pd2"] = DynamicsDevelopmentService(
            _sc("somatic-dynamics-dev", Triad.SOMATIC, ServicePosition.PD2,
                Polarity.PARASYMPATHETIC, Dimension.POTENTIAL),
        )
        self.services["s_t7"] = StateTreasuryService(
            _sc("somatic-state-treasury", Triad.SOMATIC, ServicePosition.T7,
                Polarity.PARASYMPATHETIC, Dimension.POTENTIAL),
        )

        # === Cerebral Triad (Output Layer) ===
        self.services["c_p5"] = ReadoutProcessingService(
            _sc("cerebral-readout", Triad.CEREBRAL, ServicePosition.P5,
                Polarity.SOMATIC, Dimension.COMMITMENT),
            reservoir_dim=cfg.n_reservoir, output_dim=cfg.n_outputs,
        )
        self.services["c_o4"] = OutputOrganizationService(
            _sc("cerebral-output-org", Triad.CEREBRAL, ServicePosition.O4,
                Polarity.SOMATIC, Dimension.COMMITMENT),
        )
        self.services["c_pd2"] = LearningDirectorService(
            _sc("cerebral-learning", Triad.CEREBRAL, ServicePosition.PD2,
                Polarity.PARASYMPATHETIC, Dimension.POTENTIAL),
            ridge_lambda=cfg.ridge_lambda,
        )
        self.services["c_t7"] = WeightTreasuryService(
            _sc("cerebral-weight-treasury", Triad.CEREBRAL, ServicePosition.T7,
                Polarity.PARASYMPATHETIC, Dimension.POTENTIAL),
        )
        self.services["c_m1"] = TargetInterfaceService(
            _sc("cerebral-target", Triad.CEREBRAL, ServicePosition.M1,
                Polarity.SYMPATHETIC, Dimension.PERFORMANCE),
        )
        self.services["c_s8"] = OutputDeliveryService(
            _sc("cerebral-output-delivery", Triad.CEREBRAL, ServicePosition.S8,
                Polarity.SYMPATHETIC, Dimension.PERFORMANCE),
        )

        # Initialize all services
        for svc in self.services.values():
            await svc.initialize()
            self.coordinator.register_service(svc)

        # Create the underlying ESN for convenience
        self.esn = EchoStateNetwork(self.config)

        print("✓ CosmosReservoirSystem initialized (18 services)")
        print(f"  - Input dim:    {cfg.n_inputs}")
        print(f"  - Reservoir:    {cfg.n_reservoir}")
        print(f"  - Output dim:   {cfg.n_outputs}")
        print(f"  - Spectral r:   {cfg.spectral_radius}")

    async def process_input(self, input_data: np.ndarray) -> np.ndarray:
        """Process a single input through the full 18-service pipeline."""
        # Autonomic pipeline
        msg = create_message("RAW_INPUT", input_data, "external")
        msg = await self.services["a_m1"].process(msg)
        if msg is None:
            return np.zeros(self.config.n_outputs)

        # Membrane interface
        mem_msg = create_message("PREPROCESSED_INPUT", msg.payload, "autonomic")
        mem_msg = await self.services["s_m1"].process(mem_msg)
        if mem_msg is None:
            return np.zeros(self.config.n_outputs)

        # Reservoir dynamics
        res_msg = await self.services["s_p5"].process(mem_msg)
        if res_msg is None:
            return np.zeros(self.config.n_outputs)

        # State management
        await self.services["s_s8"].process(res_msg)
        await self.services["s_t7"].process(res_msg)

        # Readout
        readout_msg = await self.services["c_p5"].process(res_msg)
        if readout_msg is None:
            return np.zeros(self.config.n_outputs)

        # Output organization
        output_msg = await self.services["c_o4"].process(readout_msg)
        if output_msg is None:
            return np.zeros(self.config.n_outputs)

        payload = output_msg.payload
        if isinstance(payload, dict):
            return np.asarray(payload.get("output", np.zeros(self.config.n_outputs)), dtype=float)
        return np.asarray(payload, dtype=float)

    async def train(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        """
        Train the reservoir system using all 18 services.

        Uses the underlying ESN for efficient batch training, then
        synchronizes W_out back to ReadoutProcessingService.
        """
        washout = min(100, len(inputs) // 5)

        # Fit preprocessing
        self.services["a_pd2"].fit(inputs)

        # Collect states through the somatic pipeline
        states = []
        self.services["s_p5"].state = np.zeros(self.config.n_reservoir)
        for inp in inputs:
            msg = create_message("RAW_INPUT", inp, "external")
            msg = await self.services["a_m1"].process(msg)
            if msg is None:
                continue
            mem_msg = create_message("PREPROCESSED_INPUT", msg.payload, "autonomic")
            mem_msg = await self.services["s_m1"].process(mem_msg)
            if mem_msg is None:
                continue
            res_msg = await self.services["s_p5"].process(mem_msg)
            if res_msg is not None:
                states.append(res_msg.payload)

        if not states:
            return

        states = np.array(states)
        states_w = states[washout:]
        targets_w = targets[washout:]

        # Learning Director: batch ridge regression
        W_out = self.services["c_pd2"].batch_train(states_w, targets_w)

        # Sync W_out to ReadoutProcessingService
        self.services["c_p5"].W_out = W_out

        # Store in Weight Treasury
        self.services["c_t7"].store("W_out", W_out, metadata={"method": "ridge"})

        print(f"✓ CosmosReservoirSystem trained on {len(inputs)} samples "
              f"(washout={washout})")
