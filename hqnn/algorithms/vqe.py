import numpy as np
from typing import Dict, List, Optional, Callable, Tuple
from scipy.optimize import minimize
from hqnn.utils.gates import pauli_x, pauli_y, pauli_z, kron_n


class VQEAnsatz:
    def __init__(self, n_qubits: int, n_layers: int = 2):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dim = 2 ** n_qubits
        self.n_params = n_qubits * n_layers * 3

    def ry_gate(self, theta: float) -> np.ndarray:
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, -s], [s, c]], dtype=complex)

    def rz_gate(self, phi: float) -> np.ndarray:
        return np.array([[np.exp(-1j * phi / 2), 0],
                         [0, np.exp(1j * phi / 2)]], dtype=complex)

    def rx_gate(self, theta: float) -> np.ndarray:
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)

    def build_layer(self, params: np.ndarray, layer_idx: int) -> np.ndarray:
        offset = layer_idx * self.n_qubits * 3
        U = np.eye(self.dim, dtype=complex)

        for q in range(self.n_qubits):
            rx = self.rx_gate(params[offset + q * 3])
            ry = self.ry_gate(params[offset + q * 3 + 1])
            rz = self.rz_gate(params[offset + q * 3 + 2])
            single_gate = rz @ ry @ rx

            ops = [np.eye(2, dtype=complex)] * self.n_qubits
            ops[q] = single_gate
            U = kron_n(*ops) @ U

        for q in range(self.n_qubits - 1):
            cnot = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 0, 1],
                              [0, 0, 1, 0]], dtype=complex)
            ops = [np.eye(2, dtype=complex)] * self.n_qubits
            ops[q] = np.eye(2, dtype=complex)
            ops[q + 1] = np.eye(2, dtype=complex)
            full_cnot = np.eye(self.dim, dtype=complex)
            U = full_cnot @ U

        return U

    def build_circuit(self, params: np.ndarray) -> np.ndarray:
        state = np.zeros(self.dim, dtype=complex)
        state[0] = 1.0

        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        ops = [H] * self.n_qubits
        H_full = kron_n(*ops)
        state = H_full @ state

        for layer in range(self.n_layers):
            U = self.build_layer(params, layer)
            state = U @ state

        norm = np.linalg.norm(state)
        if norm > 1e-10:
            state /= norm
        return state

    def build_density_matrix(self, params: np.ndarray) -> np.ndarray:
        state = self.build_circuit(params)
        return np.outer(state, state.conj())


class MolecularHamiltonian:
    def __init__(self, n_qubits: int, molecule: str = "H2"):
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        self.molecule = molecule

    def build(self) -> np.ndarray:
        X, Y, Z = pauli_x(), pauli_y(), pauli_z()
        I = np.eye(2, dtype=complex)

        def op(operators: List) -> np.ndarray:
            return kron_n(*operators)

        H = np.zeros((self.dim, self.dim), dtype=complex)

        if self.molecule == "H2" and self.n_qubits == 4:
            H += -0.8105 * op([I, I, I, I])
            H += 0.1715 * op([Z, I, I, I])
            H += -0.2234 * op([I, Z, I, I])
            H += 0.1715 * op([I, I, Z, I])
            H += -0.2234 * op([I, I, I, Z])
            H += 0.1686 * op([Z, Z, I, I])
            H += 0.1200 * op([I, Z, Z, I])
            H += 0.1686 * op([I, I, Z, Z])
            H += 0.1747 * op([Z, I, I, Z])
            H += 0.1200 * op([Z, I, Z, I])
            H += 0.0663 * op([I, Z, I, Z])
            H += 0.0663 * op([X, X, Y, Y])
            H += -0.0663 * op([X, Y, Y, X])
            H += -0.0663 * op([Y, X, X, Y])
            H += 0.0663 * op([Y, Y, X, X])
        else:
            for i in range(self.n_qubits):
                ops = [I] * self.n_qubits
                ops[i] = Z
                H += -0.5 * op(ops)
            for i in range(self.n_qubits - 1):
                ops = [I] * self.n_qubits
                ops[i] = Z
                ops[i + 1] = Z
                H += 0.25 * op(ops)

        return H


class VQEAlgorithm:
    def __init__(self, n_qubits: int = 4, n_layers: int = 2,
                 molecule: str = "H2"):
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        self.ansatz = VQEAnsatz(n_qubits, n_layers)
        self.hamiltonian_builder = MolecularHamiltonian(n_qubits, molecule)
        self.H = self.hamiltonian_builder.build()
        self.energy_history: List[float] = []
        self.param_history: List[np.ndarray] = []
        self.gradient_history: List[np.ndarray] = []

    def exact_ground_energy(self) -> Tuple[float, np.ndarray]:
        eigenvalues, eigenvectors = np.linalg.eigh(self.H)
        return float(eigenvalues[0]), eigenvectors[:, 0]

    def energy_expectation(self, params: np.ndarray,
                            noise: float = 0.0) -> float:
        state = self.ansatz.build_circuit(params)
        energy = float(np.real(state.conj() @ self.H @ state))
        if noise > 0.0:
            energy += noise * np.random.randn() * 0.05
        self.energy_history.append(energy)
        self.param_history.append(params.copy())
        return energy

    def parameter_shift_gradient(self, params: np.ndarray,
                                   noise: float = 0.0) -> np.ndarray:
        grad = np.zeros_like(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += np.pi / 2
            params_minus = params.copy()
            params_minus[i] -= np.pi / 2
            e_plus = self.energy_expectation(params_plus, noise)
            e_minus = self.energy_expectation(params_minus, noise)
            grad[i] = (e_plus - e_minus) / 2.0
        return grad

    def optimize_gradient_descent(self, n_iterations: int = 100,
                                   lr: float = 0.05,
                                   noise: float = 0.02) -> Dict:
        params = np.random.uniform(-np.pi, np.pi, self.ansatz.n_params)
        exact_energy, _ = self.exact_ground_energy()
        convergence_history = []

        for it in range(n_iterations):
            grad = self.parameter_shift_gradient(params, noise)
            self.gradient_history.append(grad.copy())
            params = params - lr * grad
            params = np.clip(params, -2 * np.pi, 2 * np.pi)
            energy = self.energy_expectation(params, noise)
            error = abs(energy - exact_energy)
            convergence_history.append({
                "iteration": it,
                "energy": energy,
                "error": error,
                "grad_norm": float(np.linalg.norm(grad)),
            })

        final_energy = self.energy_expectation(params, 0.0)
        return {
            "final_energy": final_energy,
            "exact_energy": exact_energy,
            "final_params": params,
            "convergence_error": abs(final_energy - exact_energy),
            "relative_error": abs(final_energy - exact_energy) / abs(exact_energy),
            "convergence_history": convergence_history,
            "energy_history": self.energy_history,
            "n_function_evaluations": len(self.energy_history),
        }

    def optimize_scipy(self, method: str = "BFGS",
                        noise: float = 0.01) -> Dict:
        params0 = np.random.uniform(-np.pi, np.pi, self.ansatz.n_params)
        exact_energy, _ = self.exact_ground_energy()

        eval_count = [0]

        def objective(p):
            eval_count[0] += 1
            return self.energy_expectation(p, noise)

        result = minimize(objective, params0, method=method,
                          options={"maxiter": 500, "gtol": 1e-6})

        final_energy = self.energy_expectation(result.x, 0.0)
        return {
            "final_energy": final_energy,
            "exact_energy": exact_energy,
            "final_params": result.x,
            "convergence_error": abs(final_energy - exact_energy),
            "relative_error": abs(final_energy - exact_energy) / abs(exact_energy),
            "scipy_result": result,
            "n_function_evaluations": eval_count[0],
            "success": result.success,
        }