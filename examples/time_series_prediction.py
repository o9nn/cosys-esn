"""
COSYS-ESN Example: Time Series Prediction
==========================================

Demonstrates the Cosmos Reservoir System for predicting chaotic time series
(Mackey-Glass equation) using the triadic architecture.

Author: Cosmos System Enhancement Project
Date: December 29, 2025
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from integration_hub.cosmos_reservoir_system import CosmosReservoirSystem
from models.echo_state_network import ESNConfig
from cosmos_core import setup_logging
import asyncio


def generate_mackey_glass(n_samples=2000, tau=17, beta=0.2, gamma=0.1, n=10):
    """
    Generate Mackey-Glass chaotic time series.
    
    dx/dt = beta * x(t-tau) / (1 + x(t-tau)^n) - gamma * x(t)
    
    This is a classic benchmark for reservoir computing.
    """
    x = np.zeros(n_samples)
    x[0] = 1.2  # Initial condition
    
    for t in range(tau, n_samples - 1):
        x[t + 1] = x[t] + beta * x[t - tau] / (1 + x[t - tau]**n) - gamma * x[t]
    
    return x


def create_time_series_dataset(data, input_length=5, prediction_horizon=1):
    """
    Create input-output pairs for time series prediction.
    
    Args:
        data: 1D time series
        input_length: Number of past time steps to use as input
        prediction_horizon: Number of steps ahead to predict
    
    Returns:
        inputs: (n_samples, input_length)
        targets: (n_samples, 1)
    """
    n_samples = len(data) - input_length - prediction_horizon + 1
    inputs = np.zeros((n_samples, input_length))
    targets = np.zeros((n_samples, 1))
    
    for i in range(n_samples):
        inputs[i] = data[i:i + input_length]
        targets[i] = data[i + input_length + prediction_horizon - 1]
    
    return inputs, targets


def evaluate_prediction(system, test_inputs, test_targets):
    """Evaluate prediction performance."""
    predictions = []
    
    for inp in test_inputs:
        output = asyncio.run(system.process_input(inp))
        predictions.append(output[0])
    
    predictions = np.array(predictions)
    test_targets = test_targets.flatten()
    
    # Compute metrics
    mse = np.mean((predictions - test_targets)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - test_targets))
    
    # Normalized RMSE
    nrmse = rmse / np.std(test_targets)
    
    return predictions, {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'nrmse': nrmse
    }


def plot_results(test_targets, predictions, metrics, save_path):
    """Plot prediction results."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Time series comparison
    n_plot = min(500, len(test_targets))
    axes[0].plot(test_targets[:n_plot], label='Target', linewidth=2, alpha=0.7)
    axes[0].plot(predictions[:n_plot], label='Prediction', linewidth=2, alpha=0.7)
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Value')
    axes[0].set_title('Mackey-Glass Time Series Prediction')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Error plot
    error = predictions[:n_plot] - test_targets[:n_plot]
    axes[1].plot(error, color='red', linewidth=1, alpha=0.7)
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Prediction Error')
    axes[1].set_title(f'Prediction Error (NRMSE: {metrics["nrmse"]:.4f})')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to {save_path}")


def main():
    """Main demonstration."""
    setup_logging("INFO")
    
    print("=" * 70)
    print("COSYS-ESN: Mackey-Glass Time Series Prediction")
    print("=" * 70)
    
    # Generate Mackey-Glass time series
    print("\n[1/5] Generating Mackey-Glass chaotic time series...")
    mg_data = generate_mackey_glass(n_samples=2000, tau=17)
    print(f"  ✓ Generated {len(mg_data)} time steps")
    print(f"  ✓ Mean: {np.mean(mg_data):.4f}, Std: {np.std(mg_data):.4f}")
    
    # Create dataset
    print("\n[2/5] Creating time series dataset...")
    input_length = 5
    prediction_horizon = 1
    
    inputs, targets = create_time_series_dataset(
        mg_data, 
        input_length=input_length,
        prediction_horizon=prediction_horizon
    )
    
    # Split train/test
    train_size = 1000
    train_inputs = inputs[:train_size]
    train_targets = targets[:train_size]
    test_inputs = inputs[train_size:]
    test_targets = targets[train_size:]
    
    print(f"  ✓ Training set: {train_inputs.shape} → {train_targets.shape}")
    print(f"  ✓ Test set: {test_inputs.shape} → {test_targets.shape}")
    
    # Create and initialize reservoir
    print("\n[3/5] Initializing Cosmos Reservoir System...")
    config = ESNConfig(
        n_inputs=input_length,
        n_reservoir=200,      # Larger reservoir for complex dynamics
        n_outputs=1,
        spectral_radius=0.95,   # Near edge-of-chaos
        leak_rate=0.3,
        sparsity=0.1,
        ridge_lambda=1e-6
    )
    
    system = CosmosReservoirSystem(config)
    asyncio.run(system.initialize())
    
    # Train
    print("\n[4/5] Training reservoir on Mackey-Glass data...")
    asyncio.run(system.train(train_inputs, train_targets))
    
    # Evaluate
    print("\n[5/5] Evaluating on test set...")
    predictions, metrics = evaluate_prediction(system, test_inputs, test_targets)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Mean Squared Error (MSE):       {metrics['mse']:.6f}")
    print(f"  Root Mean Squared Error (RMSE): {metrics['rmse']:.6f}")
    print(f"  Mean Absolute Error (MAE):      {metrics['mae']:.6f}")
    print(f"  Normalized RMSE (NRMSE):        {metrics['nrmse']:.6f}")
    print("=" * 70)
    
    # Interpretation
    print("\n📊 Performance Interpretation:")
    if metrics['nrmse'] < 0.1:
        print("  ✓ Excellent prediction (NRMSE < 0.1)")
    elif metrics['nrmse'] < 0.3:
        print("  ✓ Good prediction (NRMSE < 0.3)")
    elif metrics['nrmse'] < 0.5:
        print("  ⚠ Moderate prediction (NRMSE < 0.5)")
    else:
        print("  ✗ Poor prediction (NRMSE >= 0.5)")
    
    # Plot results
    print("\n📈 Generating visualization...")
    plot_results(
        test_targets.flatten(),
        predictions,
        metrics,
        os.path.join(os.path.dirname(__file__), 'mackey_glass_prediction.png')
    )
    
    print("\n" + "=" * 70)
    print("✓ COSYS-ESN Time Series Prediction Demo Complete")
    print("=" * 70)
    print("\nKey Insights:")
    print("  • Cosmos System 5 architecture successfully applied to reservoir computing")
    print("  • Triadic structure (Input → Reservoir → Output) enables temporal processing")
    print("  • Edge-of-chaos dynamics (spectral radius ≈ 1.0) maximize information capacity")
    print("  • Echo state property allows the reservoir to 'remember' past inputs")
    print("  • Linear readout layer learns to extract relevant features from reservoir states")
    print("\nArchitectural Mapping:")
    print("  • Autonomic Triad: Input preprocessing and validation")
    print("  • Somatic Triad: Recurrent reservoir dynamics (commitment to processing)")
    print("  • Cerebral Triad: Readout and output organization (potential extraction)")


if __name__ == "__main__":
    main()
