import numpy as np
from typing import List, Dict, Optional, Tuple
from hqnn.utils.gates import (hadamard, cnot, pauli_x,
                                pauli_z, von_neumann_entropy)


class MatrixProductState:
    def __init__(self, n_sites: int, bond_dim: int = 32,
                 phys_dim: int = 2):
        self.n = n_sites
        self.D = bond_dim
        self.d = phys_dim
        self.tensors: List[np.ndarray] = self._initialize_mps()

    def _initialize_mps(self) -> List[np.ndarray]:
        tensors = []
        for i in range(self.n):
            D_left = 1 if i == 0 else min(self.D, self.d**i)
            D_right = 1 if i == self.n-1 else min(self.D, self.d**(i+1))
            D_left = min(D_left, self.D)
            D_right = min(D_right, self.D)
            T = np.zeros((D_left, self.d, D_right), dtype=complex)
            T[0, 0, 0] = 1.0
            tensors.append(T)
        return tensors

    def apply_single_site_gate(self, gate: np.ndarray, site: int) -> None:
        T = self.tensors[site]
        new_T = np.einsum('ab,LbR->LaR', gate, T)
        self.tensors[site] = new_T

    def apply_two_site_gate(self, gate: np.ndarray,
                             site: int, max_bond: Optional[int] = None) -> None:
        if site >= self.n - 1:
            return
        max_bond = max_bond or self.D
        TL = self.tensors[site]
        TR = self.tensors[site + 1]
        theta = np.einsum('LaM,MbR->LabR', TL, TR)
        DL, da, db, DR = theta.shape
        theta_mat = theta.reshape(DL * da, db * DR)
        U, S, Vh = np.linalg.svd(theta_mat, full_matrices=False)
        chi = min(len(S), max_bond)
        U = U[:, :chi]
        S = S[:chi]
        Vh = Vh[:chi, :]
        self.tensors[site] = U.reshape(DL, da, chi)
        self.tensors[site + 1] = (np.diag(S) @ Vh).reshape(chi, db, DR)

    def apply_hadamard_layer(self) -> None:
        H = hadamard()
        for i in range(self.n):
            self.apply_single_site_gate(H, i)

    def apply_entangling_layer(self, adjacency: np.ndarray,
                                max_bond: int = 32) -> None:
        CNOT = cnot()
        CNOT_reshaped = CNOT.reshape(2, 2, 2, 2)
        gate = CNOT_reshaped.transpose(0, 2, 1, 3).reshape(4, 4)
        for i in range(self.n - 1):
            if abs(adjacency[i, i+1]) > 1e-6:
                self.apply_two_site_gate(gate, i, max_bond)

    def get_local_density_matrix(self, site: int) -> np.ndarray:
        T = self.tensors[site]
        DL, d, DR = T.shape
        rho = np.einsum('LaR,LbR->ab', T, T.conj())
        norm = np.trace(rho)
        if abs(norm) > 1e-12:
            rho /= norm
        return rho

    def get_two_site_density_matrix(self, site: int) -> np.ndarray:
        if site >= self.n - 1:
            return np.eye(4, dtype=complex) / 4
        TL = self.tensors[site]
        TR = self.tensors[site + 1]
        theta = np.einsum('LaM,MbR->LabR', TL, TR)
        DL, da, db, DR = theta.shape
        theta_mat = theta.reshape(DL * da * db, DR)
        rho_full = np.einsum('iR,jR->ij', theta_mat, theta_mat.conj())
        rho_reshaped = rho_full.reshape(DL, da, db, DL, da, db)
        rho_2site = np.einsum('LabLcd->abcd', rho_reshaped)
        rho_2site = rho_2site.reshape(da * db, da * db)
        norm = np.trace(rho_2site)
        if abs(norm) > 1e-12:
            rho_2site /= norm
        return rho_2site

    def get_entanglement_entropy(self, bond: int) -> float:
        if bond >= self.n - 1:
            return 0.0
        TL = self.tensors[bond]
        TR = self.tensors[bond + 1]
        theta = np.einsum('LaM,MbR->LabR', TL, TR)
        DL, da, db, DR = theta.shape
        theta_mat = theta.reshape(DL * da, db * DR)
        _, S, _ = np.linalg.svd(theta_mat, full_matrices=False)
        S2 = S**2
        S2 = S2[S2 > 1e-12]
        S2 /= S2.sum()
        return float(-np.sum(S2 * np.log2(S2)))

    def get_bond_dimensions(self) -> List[int]:
        return [T.shape[2] for T in self.tensors[:-1]]

    def total_parameters(self) -> int:
        return sum(T.size for T in self.tensors)


class TensorNetworkHQNN:
    def __init__(self, n_nodes: int, bond_dim: int = 32):
        self.n = n_nodes
        self.D = bond_dim
        self.mps = MatrixProductState(n_nodes, bond_dim)
        self.entanglement_history: List[List[float]] = []

    def run_circuit_step(self, adjacency: np.ndarray) -> None:
        self.mps.apply_hadamard_layer()
        self.mps.apply_entangling_layer(adjacency, self.D)

    def get_all_local_densities(self) -> List[np.ndarray]:
        return [self.mps.get_local_density_matrix(i) for i in range(self.n)]

    def get_entanglement_profile(self) -> List[float]:
        profile = [self.mps.get_entanglement_entropy(bond)
                   for bond in range(self.n - 1)]
        self.entanglement_history.append(profile)
        return profile

    def get_network_fidelity(self) -> float:
        purities = []
        for i in range(self.n):
            rho = self.mps.get_local_density_matrix(i)
            purities.append(float(np.real(np.trace(rho @ rho))))
        return float(np.mean(purities))

    def apply_decoherence_mps(self, gamma: float, dt: float) -> None:
        for i in range(self.n):
            p = 1.0 - np.exp(-gamma * dt)
            rho = self.mps.get_local_density_matrix(i)
            for j in range(rho.shape[0]):
                for k in range(rho.shape[1]):
                    if j != k:
                        rho[j, k] *= (1.0 - p)
            DL, d, DR = self.mps.tensors[i].shape
            eigenvalues, eigenvectors = np.linalg.eigh(rho)
            eigenvalues = np.maximum(eigenvalues, 0)
            eigenvalues /= eigenvalues.sum() + 1e-12
            sqrt_rho = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.conj().T
            self.mps.tensors[i] = sqrt_rho.reshape(1, d, 1) if DL == 1 and DR == 1 \
                else self.mps.tensors[i]