import numpy as np
from typing import List, Optional, Dict
from hqnn.utils.gates import pauli_x, pauli_y, pauli_z, kron_n


class NetworkHamiltonian:
    def __init__(self, n_nodes: int,
                 J: float = 1.0,
                 Gamma: float = 0.5,
                 lam: float = 0.3):
        self.n = n_nodes
        self.J = J
        self.Gamma = Gamma
        self.lam = lam
        self.dim = 2 ** n_nodes
        self._cache: Dict[str, np.ndarray] = {}

    def _single_site_op(self, op: np.ndarray, site: int) -> np.ndarray:
        key = f"site_{site}_{op.tobytes()}"
        if key in self._cache:
            return self._cache[key]
        ops = [np.eye(2, dtype=complex)] * self.n
        ops[site] = op
        result = kron_n(*ops)
        self._cache[key] = result
        return result

    def _two_site_op(self, op1: np.ndarray, op2: np.ndarray,
                      site1: int, site2: int) -> np.ndarray:
        ops = [np.eye(2, dtype=complex)] * self.n
        ops[site1] = op1
        ops[site2] = op2
        return kron_n(*ops)

    def build_ising(self, adjacency: np.ndarray) -> np.ndarray:
        Z = pauli_z()
        X = pauli_x()
        H = np.zeros((self.dim, self.dim), dtype=complex)

        for i in range(self.n):
            for j in range(i + 1, self.n):
                if abs(adjacency[i, j]) > 1e-10:
                    coupling = float(np.real(adjacency[i, j]))
                    H -= self.J * coupling * self._two_site_op(Z, Z, i, j)

        for i in range(self.n):
            H -= self.Gamma * self._single_site_op(X, i)

        return H

    def build_heisenberg(self, adjacency: np.ndarray) -> np.ndarray:
        X, Y, Z = pauli_x(), pauli_y(), pauli_z()
        H = np.zeros((self.dim, self.dim), dtype=complex)

        for i in range(self.n):
            for j in range(i + 1, self.n):
                if abs(adjacency[i, j]) > 1e-10:
                    J_ij = float(np.real(adjacency[i, j]))
                    H += J_ij * (self._two_site_op(X, X, i, j) +
                                  self._two_site_op(Y, Y, i, j) +
                                  self._two_site_op(Z, Z, i, j))

        return H

    def build_topological_term(self, adjacency: np.ndarray,
                                 euler_characteristic: int) -> np.ndarray:
        Z = pauli_z()
        H_topo = np.zeros((self.dim, self.dim), dtype=complex)

        for i in range(self.n):
            for j in range(i + 1, self.n):
                if abs(adjacency[i, j]) > 1e-10:
                    topo_strength = self.lam * (1.0 / (1.0 + abs(euler_characteristic)))
                    H_topo += topo_strength * self._two_site_op(Z, Z, i, j)

        return H_topo

    def build_full(self, adjacency: np.ndarray,
                    euler_characteristic: int) -> np.ndarray:
        H_ising = self.build_ising(adjacency)
        H_topo = self.build_topological_term(adjacency, euler_characteristic)
        return H_ising + H_topo

    def time_evolution_operator(self, H: np.ndarray, dt: float) -> np.ndarray:
        from scipy.linalg import expm
        return expm(-1j * H * dt)

    def ground_state(self, H: np.ndarray) -> np.ndarray:
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        return eigenvectors[:, 0]

    def energy_gap(self, H: np.ndarray) -> float:
        eigenvalues = np.sort(np.real(np.linalg.eigvalsh(H)))
        return float(eigenvalues[1] - eigenvalues[0]) if len(eigenvalues) > 1 else 0.0