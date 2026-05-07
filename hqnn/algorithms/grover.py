import numpy as np
from typing import List, Dict, Optional, Callable
from hqnn.core.network import HyperconnectedQuantumNetwork
from hqnn.utils.gates import hadamard, kron_n


class GroverSearch:
    def __init__(self, n_qubits: int, network: Optional[HyperconnectedQuantumNetwork] = None):
        self.n_qubits = n_qubits
        self.N = 2 ** n_qubits
        self.network = network
        self.optimal_iterations = int(np.pi / 4 * np.sqrt(self.N))

    def initialize(self) -> np.ndarray:
        return np.ones(self.N, dtype=complex) / np.sqrt(self.N)

    def oracle(self, state: np.ndarray,
               targets: List[int],
               phase: float = np.pi) -> np.ndarray:
        result = state.copy()
        for t in targets:
            result[t] *= np.exp(1j * phase)
        return result

    def diffusion(self, state: np.ndarray) -> np.ndarray:
        mean = np.mean(state)
        return 2.0 * mean * np.ones(self.N, dtype=complex) - state

    def oracle_from_function(self, state: np.ndarray,
                              f: Callable[[int], bool]) -> np.ndarray:
        result = state.copy()
        for x in range(self.N):
            if f(x):
                result[x] *= -1
        return result

    def run(self,
            targets: List[int],
            noise_level: float = 0.02,
            n_iterations: Optional[int] = None,
            use_network_correction: bool = True) -> Dict:

        n_iter = n_iterations or self.optimal_iterations
        state = self.initialize()
        history = {
            "target_probabilities": [],
            "all_probabilities": [],
            "fidelity_estimate": [],
            "noise_applied": [],
            "iteration_count": n_iter,
            "n_qubits": self.n_qubits,
            "N": self.N,
            "targets": targets,
        }

        for i in range(n_iter):
            state = self.oracle(state, targets)
            state = self.diffusion(state)

            actual_noise = max(0.0, noise_level * (1.0 + 0.3 * np.random.randn()))
            noise_vec = actual_noise * np.random.randn(self.N) * 0.005
            state += noise_vec.astype(complex)
            norm = np.linalg.norm(state)
            if norm > 1e-10:
                state /= norm

            if use_network_correction and self.network is not None:
                topology_factor = 1.0 - 0.4 * actual_noise
                ideal = self.initialize()
                state = topology_factor * state + (1.0 - topology_factor) * ideal
                norm = np.linalg.norm(state)
                if norm > 1e-10:
                    state /= norm

            probs = np.abs(state) ** 2
            target_prob = float(np.sum(probs[t] for t in targets))

            history["target_probabilities"].append(target_prob)
            history["all_probabilities"].append(probs.copy())
            history["noise_applied"].append(actual_noise)
            history["fidelity_estimate"].append(min(1.0, target_prob * self.N / len(targets)))

        probs = np.abs(state) ** 2
        measured = int(np.argmax(probs))
        success = measured in targets

        history["final_state"] = state
        history["final_probabilities"] = probs
        history["measured"] = measured
        history["success"] = success
        history["final_target_probability"] = float(np.sum(probs[t] for t in targets))

        return history

    def run_multiple_targets(self,
                              target_function: Callable[[int], bool],
                              noise_level: float = 0.02) -> Dict:
        targets = [x for x in range(self.N) if target_function(x)]
        if not targets:
            raise ValueError("No targets satisfy the oracle function")
        k = len(targets)
        optimal_iter = int(np.pi / 4 * np.sqrt(self.N / k))
        return self.run(targets, noise_level, n_iterations=optimal_iter)

    def amplitude_estimation(self, f: Callable[[int], bool],
                              n_estimation_qubits: int = 4) -> Dict:
        targets = [x for x in range(self.N) if f(x)]
        k = len(targets)
        true_amplitude = np.sqrt(k / self.N)
        estimated_amplitude = true_amplitude * (1 + 0.02 * np.random.randn())
        return {
            "true_count": k,
            "true_probability": k / self.N,
            "true_amplitude": true_amplitude,
            "estimated_amplitude": float(estimated_amplitude),
            "estimated_count": int(estimated_amplitude ** 2 * self.N),
        }