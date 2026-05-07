import numpy as np
import pytest
from hqnn.core.hamiltonian import NetworkHamiltonian
from hqnn.core.network import HyperconnectedQuantumNetwork


class TestNetworkHamiltonian:
    def setup_method(self):
        self.n = 4
        self.H_builder = NetworkHamiltonian(n_nodes=self.n, J=1.0, Gamma=0.5, lam=0.3)
        self.net = HyperconnectedQuantumNetwork(n_nodes=self.n, seed=42)
        self.A = np.real(self.net.get_adjacency_matrix())

    def test_ising_hamiltonian_hermitian(self):
        H = self.H_builder.build_ising(self.A)
        assert np.allclose(H, H.conj().T, atol=1e-10)

    def test_ising_hamiltonian_real_eigenvalues(self):
        H = self.H_builder.build_ising(self.A)
        evals = np.linalg.eigvalsh(H)
        assert np.all(np.isreal(evals))

    def test_heisenberg_hamiltonian_hermitian(self):
        H = self.H_builder.build_heisenberg(self.A)
        assert np.allclose(H, H.conj().T, atol=1e-10)

    def test_topological_term_hermitian(self):
        H_topo = self.H_builder.build_topological_term(self.A, euler_characteristic=2)
        assert np.allclose(H_topo, H_topo.conj().T, atol=1e-10)

    def test_full_hamiltonian_hermitian(self):
        H = self.H_builder.build_full(self.A, euler_characteristic=2)
        assert np.allclose(H, H.conj().T, atol=1e-10)

    def test_full_hamiltonian_shape(self):
        H = self.H_builder.build_full(self.A, euler_characteristic=1)
        dim = 2 ** self.n
        assert H.shape == (dim, dim)

    def test_time_evolution_operator_unitary(self):
        H = self.H_builder.build_ising(self.A)
        U = self.H_builder.time_evolution_operator(H, dt=0.01)
        result = U @ U.conj().T
        dim = 2 ** self.n
        assert np.allclose(result, np.eye(dim), atol=1e-8)

    def test_ground_state_normalized(self):
        H = self.H_builder.build_ising(self.A)
        gs = self.H_builder.ground_state(H)
        assert abs(np.linalg.norm(gs) - 1.0) < 1e-8

    def test_ground_state_is_eigenvector(self):
        H = self.H_builder.build_ising(self.A)
        gs = self.H_builder.ground_state(H)
        Hgs = H @ gs
        evals = np.linalg.eigvalsh(H)
        E0 = evals[0]
        assert np.allclose(Hgs, E0 * gs, atol=1e-8)

    def test_energy_gap_positive(self):
        H = self.H_builder.build_ising(self.A)
        gap = self.H_builder.energy_gap(H)
        assert gap >= 0.0

    def test_topological_strength_decreases_with_euler(self):
        H_low = self.H_builder.build_topological_term(self.A, euler_characteristic=1)
        H_high = self.H_builder.build_topological_term(self.A, euler_characteristic=10)
        norm_low = np.linalg.norm(H_low)
        norm_high = np.linalg.norm(H_high)
        assert norm_low >= norm_high