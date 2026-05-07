import numpy as np
from scipy.linalg import expm
from dataclasses import dataclass, field
from typing import Optional, List
from hqnn.utils.gates import von_neumann_entropy, fidelity


@dataclass
class NodeConfig:
    n_qubits: int = 2
    initial_state: str = "superposition"
    decoherence_model: str = "lindblad"
    t1: float = 100e-6
    t2: float = 50e-6


class QuantumNode:
    def __init__(self, node_id: int, config: Optional[NodeConfig] = None):
        self.id = node_id
        self.config = config or NodeConfig()
        self.n_qubits = self.config.n_qubits
        self.dim = 2 ** self.n_qubits

        self.density_matrix = self._initialize_density_matrix()
        self.phase = np.random.uniform(0, 2 * np.pi)
        self.topological_health = 1.0
        self.coherence_time = 0.0
        self.error_count = 0
        self.measurement_history: List[float] = []

    def _initialize_density_matrix(self) -> np.ndarray:
        if self.config.initial_state == "superposition":
            state = np.ones(self.dim, dtype=complex) / np.sqrt(self.dim)
        elif self.config.initial_state == "zero":
            state = np.zeros(self.dim, dtype=complex)
            state[0] = 1.0
        elif self.config.initial_state == "ghz":
            state = np.zeros(self.dim, dtype=complex)
            state[0] = 1 / np.sqrt(2)
            state[-1] = 1 / np.sqrt(2)
        else:
            state = np.ones(self.dim, dtype=complex) / np.sqrt(self.dim)
        return np.outer(state, state.conj())

    def apply_gate(self, U: np.ndarray) -> None:
        assert U.shape == (self.dim, self.dim), f"Gate shape mismatch: {U.shape} vs {(self.dim, self.dim)}"
        self.density_matrix = U @ self.density_matrix @ U.conj().T
        self._renormalize()

    def apply_decoherence(self, gamma: float, dt: float) -> None:
        if self.config.decoherence_model == "lindblad":
            self._apply_lindblad(gamma, dt)
        elif self.config.decoherence_model == "amplitude_damping":
            self._apply_amplitude_damping(gamma, dt)
        elif self.config.decoherence_model == "phase_damping":
            self._apply_phase_damping(gamma, dt)

    def _apply_lindblad(self, gamma: float, dt: float) -> None:
        L = np.zeros((self.dim, self.dim), dtype=complex)
        for i in range(self.dim - 1):
            L[i, i + 1] = np.sqrt(gamma)
        Ldag = L.conj().T
        dissipator = (L @ self.density_matrix @ Ldag
                      - 0.5 * (Ldag @ L @ self.density_matrix
                                + self.density_matrix @ Ldag @ L))
        self.density_matrix += dissipator * dt
        self._renormalize()

    def _apply_amplitude_damping(self, gamma: float, dt: float) -> None:
        p = 1 - np.exp(-gamma * dt)
        K0 = np.array([[1, 0], [0, np.sqrt(1 - p)]], dtype=complex)
        K1 = np.array([[0, np.sqrt(p)], [0, 0]], dtype=complex)
        if self.dim == 2:
            rho_new = K0 @ self.density_matrix @ K0.conj().T + K1 @ self.density_matrix @ K1.conj().T
            self.density_matrix = rho_new
        else:
            self._apply_lindblad(gamma, dt)
        self._renormalize()

    def _apply_phase_damping(self, gamma: float, dt: float) -> None:
        p = 1 - np.exp(-gamma * dt)
        for i in range(self.dim):
            for j in range(self.dim):
                if i != j:
                    self.density_matrix[i, j] *= (1 - p)
        self._renormalize()

    def _renormalize(self) -> None:
        trace = np.trace(self.density_matrix)
        if np.abs(trace) > 1e-12:
            self.density_matrix /= trace
        self.density_matrix = (self.density_matrix + self.density_matrix.conj().T) / 2

    def measure_observable(self, observable: np.ndarray) -> float:
        expectation = float(np.real(np.trace(observable @ self.density_matrix)))
        self.measurement_history.append(expectation)
        return expectation

    def get_entropy(self) -> float:
        return von_neumann_entropy(self.density_matrix)

    def get_purity(self) -> float:
        return float(np.real(np.trace(self.density_matrix @ self.density_matrix)))

    def get_bloch_vector(self) -> np.ndarray:
        if self.dim != 2:
            return np.zeros(3)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        return np.array([
            float(np.real(np.trace(X @ self.density_matrix))),
            float(np.real(np.trace(Y @ self.density_matrix))),
            float(np.real(np.trace(Z @ self.density_matrix))),
        ])

    def reset(self) -> None:
        self.density_matrix = self._initialize_density_matrix()
        self.topological_health = 1.0
        self.error_count = 0
        self.measurement_history.clear()

    def clone(self) -> "QuantumNode":
        node = QuantumNode(self.id, self.config)
        node.density_matrix = self.density_matrix.copy()
        node.phase = self.phase
        node.topological_health = self.topological_health
        return node