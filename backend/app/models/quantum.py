"""
Module 2 & 3: Hybrid Quantum-Classical Architecture / Variational
Quantum Classifier (VQC).

Architecture:
  classical PCA features --> angle embedding --> variational ansatz
  (strongly entangling layers) --> Pauli-Z expectation --> classical
  sigmoid head --> binary disease probability.

The classical PCA projection (fit in preprocessing.py) plus the final
sigmoid/threshold step is what makes this "hybrid": classical
pre-processing feeds a genuinely quantum variational circuit, and a
lightweight classical post-processing layer turns the circuit's
expectation value into a calibrated probability. Trained end-to-end
with PyTorch autograd through PennyLane's torch interface.

Uses PennyLane's `default.qubit` simulator, as required.
"""
import time
from typing import Dict, List, Tuple

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

from app.config import (
    N_QUBITS,
    N_VQC_LAYERS,
    VQC_EPOCHS,
    VQC_LEARNING_RATE,
    VQC_BATCH_SIZE,
    RANDOM_STATE,
)
from app.models.classical import compute_metrics

torch.manual_seed(RANDOM_STATE)


def build_vqc_circuit(n_qubits: int, n_layers: int):
    """Build a PennyLane QNode implementing an angle-embedding +
    StronglyEntanglingLayers variational circuit, wrapped for PyTorch
    autodiff so it can be trained jointly with classical layers."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    weight_shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
    return circuit, weight_shape


class HybridVQC(nn.Module):
    """Hybrid model: quantum variational circuit + classical linear head."""

    def __init__(self, n_qubits: int = N_QUBITS, n_layers: int = N_VQC_LAYERS):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        circuit, weight_shape = build_vqc_circuit(n_qubits, n_layers)
        weight_shapes = {"weights": weight_shape}
        self.q_layer = qml.qnn.TorchLayer(circuit, weight_shapes)
        # Classical post-processing head maps quantum expectation values
        # (one per qubit) to a single disease-probability logit.
        self.post_net = nn.Sequential(nn.Linear(n_qubits, 1))

    def forward(self, x):
        q_out = self.q_layer(x)
        logits = self.post_net(q_out)
        return torch.sigmoid(logits).squeeze(-1)


def _to_angle_range(X: np.ndarray) -> np.ndarray:
    """Rescale PCA features into [0, pi] for stable angle embedding."""
    X = np.asarray(X, dtype=np.float32)
    mins, maxs = X.min(axis=0), X.max(axis=0)
    span = np.where(maxs - mins == 0, 1.0, maxs - mins)
    return (X - mins) / span * np.pi, mins, span


def train_vqc(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_qubits: int = None,
    n_layers: int = None,
    epochs: int = None,
    lr: float = None,
) -> Tuple[HybridVQC, Dict, List[float], Dict]:
    n_qubits = n_qubits or min(N_QUBITS, X_train.shape[1])
    n_layers = n_layers or N_VQC_LAYERS
    epochs = epochs or VQC_EPOCHS
    lr = lr or VQC_LEARNING_RATE

    X_train_scaled, mins, span = _to_angle_range(X_train)
    X_test_scaled = np.clip((np.asarray(X_test, dtype=np.float32) - mins) / span, 0, 1) * np.pi

    model = HybridVQC(n_qubits=n_qubits, n_layers=n_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)

    loss_history = []
    start = time.time()
    n_samples = X_train_t.shape[0]
    batch_size = min(VQC_BATCH_SIZE, n_samples)

    for epoch in range(epochs):
        permutation = torch.randperm(n_samples)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n_samples, batch_size):
            idx = permutation[i : i + batch_size]
            xb, yb = X_train_t[idx], y_train_t[idx]

            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
        loss_history.append(epoch_loss / max(n_batches, 1))

    elapsed = time.time() - start

    model.eval()
    with torch.no_grad():
        test_probs = model(X_test_t).numpy()
    y_pred = (test_probs >= 0.5).astype(int)

    metrics = compute_metrics(y_test, y_pred, test_probs)
    metrics["training_time_seconds"] = round(elapsed, 4)
    metrics["model_name"] = "vqc"

    scaling_params = {"mins": mins, "span": span}
    return model, metrics, loss_history, scaling_params


def predict_vqc(model: HybridVQC, X: np.ndarray, scaling_params: Dict) -> np.ndarray:
    mins, span = scaling_params["mins"], scaling_params["span"]
    X_scaled = np.clip((np.asarray(X, dtype=np.float32) - mins) / span, 0, 1) * np.pi
    model.eval()
    with torch.no_grad():
        probs = model(torch.tensor(X_scaled, dtype=torch.float32)).numpy()
    return probs


def circuit_text_diagram(n_qubits: int, n_layers: int) -> str:
    """Human-readable ASCII summary of the circuit architecture, shown
    in the UI to explain the hybrid pipeline to non-quantum-expert users."""
    return (
        f"Hybrid VQC Architecture\n"
        f"------------------------\n"
        f"Input: classical features -> PCA({n_qubits}) -> angle-normalized [0, pi]\n"
        f"[Q0]---AngleEmbed(Y)---[Entangling Layer x{n_layers}]---<Z>--\\\n"
        f"[Q1]---AngleEmbed(Y)---[Entangling Layer x{n_layers}]---<Z>---> Linear(1) -> Sigmoid -> P(disease)\n"
        f"[Q{max(n_qubits-2,2)}]---AngleEmbed(Y)---[Entangling Layer x{n_layers}]---<Z>--/\n"
        f"Simulator: PennyLane default.qubit | Diff method: backprop (autograd via PyTorch)"
    )
