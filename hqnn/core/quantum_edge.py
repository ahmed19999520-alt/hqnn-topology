import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple
from hqnn.utils.gates import von_neumann_entropy, concurrence, negativity


@dataclass
class EdgeConfig:
    initial_weight_scale: float = 0.1
    entanglement_target: float = 0.8
    slime_decay: float = 0.1
    topology_protection: float = 0.5


class QuantumEdge:
    def __init__(self, node_i: int, node_j: int,
                 config: Optional[EdgeConfig] = None):
        self.i = node_i
        self.j = node_j
        self.config = config or EdgeConfig()

        scale = self.config.initial_weight_scale
        self.weight = complex(
            np.random.normal(0, scale),
            np.random.normal(0, scale)
        )

        self.entanglement: float = 0.0
        self.quantum_flow: float = 0.0
        self.slime_factor: float = 1.0
        self.mutual_information: float = 0.0
        self.concurrence: float = 0.0
        self.negativity: float = 0.0
        self.flow_history: list = []
        self.error_rate: float = 0.0

    def update_entanglement_metrics(self,
                                    rho_i: np.ndarray,
                                    rho_j: np.ndarray) -> None:
        S_i = von_neumann_entropy(rho_i)
        S_j = von_neumann_entropy(rho_j)
        rho_combined = 0.5 * (rho_i + rho_j)
        S_ij = von_neumann_entropy(rho_combined)
        self.mutual_information = max(0.0, S_i + S_j - 2 * S_ij)

        if rho_i.shape == (4, 4):
            self.concurrence = concurrence(rho_i)
            self.negativity = negativity(rho_i)

        self.entanglement = min(1.0, self.mutual_information)
        self.quantum_flow = self.entanglement * abs(self.weight)
        self.flow_history.append(self.quantum_flow)

    def apply_slime_update(self, noise_level: float, dt: float) -> None:
        growth = (self.quantum_flow - self.config.slime_decay) * self.slime_factor
        noise_suppression = 1.0 - min(0.95, noise_level)
        topology_term = self.entanglement * self.config.topology_protection

        d_slime = (growth * noise_suppression + topology_term) * dt
        self.slime_factor = max(0.01, self.slime_factor + d_slime)

        phase = np.angle(self.weight)
        new_mag = abs(self.weight) * (1.0 + 0.1 * d_slime)
        new_mag = np.clip(new_mag, 1e-6, 10.0)
        self.weight = new_mag * np.exp(1j * phase)

    def get_transmission_fidelity(self) -> float:
        return float(np.exp(-self.error_rate) * self.slime_factor / (1.0 + self.slime_factor))

    def reset(self) -> None:
        scale = self.config.initial_weight_scale
        self.weight = complex(
            np.random.normal(0, scale),
            np.random.normal(0, scale)
        )
        self.entanglement = 0.0
        self.quantum_flow = 0.0
        self.slime_factor = 1.0
        self.flow_history.clear()