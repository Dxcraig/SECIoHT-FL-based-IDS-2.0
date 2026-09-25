"""
Federated Learning (FedAvg) and Differential Privacy (DP-SGD) simulation engine.
Replicates the Stage 3 and Stage 4 pipeline from SECIoHT-FL.
"""
import numpy as np
import math
from typing import List, Dict, Any, Generator


def calculate_dp_epsilon(
    rounds: int,
    local_epochs: int,
    noise_multiplier: float,
    delta: float = 1e-4,
    batch_size: int = 64,
    num_samples_per_client: int = 4000
) -> float:
    """
    Computes cumulative differential privacy epsilon (ε) via Gaussian Mechanism RDP bound.
    Demonstrates how epsilon accumulates across communication rounds.
    """
    if noise_multiplier <= 0:
        return float("inf")

    steps_per_epoch = max(1, num_samples_per_client // batch_size)
    total_steps = rounds * local_epochs * steps_per_epoch
    q = batch_size / num_samples_per_client  # subsampling ratio

    # Standard Moments Accountant / RDP approximation for Gaussian mechanism
    # eps = (q * sqrt(2 * log(1/delta) * total_steps)) / noise_multiplier
    eps = (q * math.sqrt(2.0 * math.log(1.0 / delta) * total_steps)) / noise_multiplier
    return round(float(eps), 3)


def simulate_fl_rounds(
    num_clients: int = 3,
    num_rounds: int = 5,
    local_epochs: int = 2,
    noise_multiplier: float = 1.5,
    max_grad_norm: float = 1e-4
) -> Generator[Dict[str, Any], None, None]:
    """
    Generator yielding step-by-step federated training metrics across communication rounds.
    Provides real-time feedback for the interactive Streamlit FL Lab.
    """
    rng = np.random.default_rng(42)
    
    # Initial global model baseline
    global_acc = 0.52
    global_loss = 0.69
    
    # Noise multiplier impact on convergence ceiling
    acc_ceiling = 0.88 if noise_multiplier == 0 else (0.84 if noise_multiplier <= 1.5 else 0.76)
    convergence_speed = 0.08 if noise_multiplier == 0 else 0.06

    for rnd in range(1, num_rounds + 1):
        # 1. Local client training phase
        client_metrics = []
        for c in range(num_clients):
            c_loss = max(0.20, global_loss - (rnd * 0.05) + rng.normal(0, 0.02))
            c_acc = min(acc_ceiling, global_acc + (rnd * convergence_speed) + rng.normal(0, 0.015))
            client_metrics.append({
                "client_id": f"Hospital / Edge Device {c+1}",
                "train_loss": round(float(c_loss), 4),
                "train_acc": round(float(c_acc * 100), 2)
            })

        # 2. Server FedAvg aggregation
        global_loss = max(0.22, global_loss - 0.045 + (0.01 if noise_multiplier > 0 else 0.0))
        global_acc = min(acc_ceiling, global_acc + convergence_speed)
        
        # 3. Privacy Accountant accumulation
        cumulative_eps = calculate_dp_epsilon(
            rounds=rnd,
            local_epochs=local_epochs,
            noise_multiplier=noise_multiplier
        )

        yield {
            "round": rnd,
            "global_accuracy": round(float(global_acc * 100), 2),
            "global_loss": round(float(global_loss), 4),
            "cumulative_epsilon": cumulative_eps,
            "clients": client_metrics
        }
