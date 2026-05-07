import numpy as np
import pytest
from hqnn.core.quantum_node import QuantumNode, NodeConfig
from hqnn.core.quantum_edge import QuantumEdge, EdgeConfig
from hqnn.core.network import HyperconnectedQuantumNetwork
from hqnn.core.hamiltonian import NetworkHamiltonian
from hqnn.utils.gates import (hadamard, pauli_x, pauli_z, bell_state,
                                ghz_state, von_neumann_entropy, fidelity,
                                concurrence, density_from_state)


class TestQuantumGates:
    def test_hadamard_unitary(self):
        H = hadamard()
        result = H @ H.conj().T
        assert np.allclose(result, np.eye(2), atol=1e-10)

    def test_pauli_hermitian(self):
        for gate in [pauli_x(), pauli_z()]:
            assert np.allclose(gate, gate.conj().T, atol=1e-10)

    def test_bell_state_normalized(self):
        for i in range(4):
            bs = bell_state(i)
            assert abs(np.linalg.norm(bs) - 1.0) < 1e-10

    def test_ghz_state_normalized(self):
        for n in [2, 3, 4]:
            gs = ghz_state(n)
            assert abs(np.linalg.norm(gs) - 1.0) < 1e-10

    def test_von_neumann_entropy_pure_state(self):
        state = np.array([1, 0], dtype=complex)
        rho = density_from_state(state)
        S = von_neumann_entropy(rho)
        assert abs(S) < 1e-10

    def test_von_neumann_entropy_mixed_state(self):
        rho = np.eye(4, dtype=complex) / 4
        S = von_neumann_entropy(rho)
        assert abs(S - 2.0) < 1e-6

    def test_fidelity_identical_states(self):
        state = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        rho = density_from_state(state)
        f = fidelity(rho, rho)
        assert abs(f - 1.0) < 1e-8

    def test_concurrence_bell_state(self):
        bs = bell_state(0)
        rho = density_from_state(bs)
        c = concurrence(rho)
        assert abs(c - 1.0) < 1e-6

    def test_concurrence_product_state(self):
        state = np.array([1, 0, 0, 0], dtype=complex)
        rho = density_from_state(state)
        c = concurrence(rho)
        assert abs(c) < 1e-6


class TestQuantumNode:
    def test_initialization_superposition(self):
        config = NodeConfig(n_qubits=2, initial_state="superposition")
        node = QuantumNode(0, config)
        trace = np.trace(node.density_matrix)
        assert abs(trace - 1.0) < 1e-10

    def test_initialization_zero(self):
        config = NodeConfig(n_qubits=2, initial_state="zero")
        node = QuantumNode(0, config)
        assert abs(node.density_matrix[0, 0] - 1.0) < 1e-10

    def test_density_matrix_hermitian(self):
        node = QuantumNode(0)
        assert np.allclose(node.density_matrix,
                           node.density_matrix.conj().T, atol=1e-10)

    def test_density_matrix_positive_semidefinite(self):
        node = QuantumNode(0)
        eigenvalues = np.linalg.eigvalsh(node.density_matrix)
        assert np.all(eigenvalues >= -1e-10)

    def test_gate_application_unitary(self):
        node = QuantumNode(0, NodeConfig(n_qubits=2))
        H = hadamard()
        H2 = np.kron(H, H)
        initial_purity = node.get_purity()
        node.apply_gate(H2)
        assert abs(node.get_purity() - initial_purity) < 1e-8

    def test_decoherence_reduces_purity(self):
        node = QuantumNode(0)
        initial_purity = node.get_purity()
        for _ in range(10):
            node.apply_decoherence(gamma=0.1, dt=0.01)
        assert node.get_purity() <= initial_purity + 1e-10

    def test_entropy_increases_with_decoherence(self):
        config = NodeConfig(n_qubits=2, initial_state="zero")
        node = QuantumNode(0, config)
        initial_entropy = node.get_entropy()
        for _ in range(5):
            node.apply_decoherence(0.1, 0.01)
        final_entropy = node.get_entropy()
        assert final_entropy >= initial_entropy - 1e-8

    def test_reset(self):
        node = QuantumNode(0)
        initial_dm = node.density_matrix.copy()
        node.apply_decoherence(0.5, 0.1)
        node.reset()
        assert np.allclose(node.density_matrix, initial_dm, atol=1e-10)

    def test_bloch_vector_unit_sphere(self):
        config = NodeConfig(n_qubits=1, initial_state="zero")
        node = QuantumNode(0, config)
        bv = node.get_bloch_vector()
        assert np.linalg.norm(bv) <= 1.0 + 1e-10

    def test_purity_range(self):
        node = QuantumNode(0)
        for _ in range(20):
            node.apply_decoherence(0.05, 0.01)
            p = node.get_purity()
            assert 0.0 <= p <= 1.0 + 1e-8


class TestQuantumEdge:
    def test_initialization(self):
        edge = QuantumEdge(0, 1)
        assert edge.i == 0
        assert edge.j == 1
        assert 0.0 <= edge.entanglement <= 1.0

    def test_slime_update_convergence(self):
        edge = QuantumEdge(0, 1)
        edge.quantum_flow = 0.5
        initial_slime = edge.slime_factor
        for _ in range(50):
            edge.apply_slime_update(0.05, 0.01)
        assert edge.slime_factor != initial_slime

    def test_slime_factor_positive(self):
        edge = QuantumEdge(0, 1)
        for _ in range(100):
            edge.apply_slime_update(0.5, 0.1)
        assert edge.slime_factor > 0


class TestHyperconnectedNetwork:
    def test_initialization(self):
        net = HyperconnectedQuantumNetwork(n_nodes=5, seed=42)
        assert len(net.nodes) == 5
        assert len(net.edges) > 0

    def test_connectivity(self):
        net = HyperconnectedQuantumNetwork(n_nodes=6, connectivity=0.8, seed=42)
        adj = {i: [] for i in range(6)}
        for (i, j) in net.edges:
            adj[i].append(j)
            adj[j].append(i)
        visited = set()
        stack = [0]
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                stack.extend(adj[node])
        assert len(visited) == 6

    def test_adjacency_matrix_hermitian(self):
        net = HyperconnectedQuantumNetwork(n_nodes=4, seed=42)
        A = net.get_adjacency_matrix()
        assert np.allclose(A, A.conj().T, atol=1e-10)

    def test_euler_characteristic_computed(self):
        net = HyperconnectedQuantumNetwork(n_nodes=5, seed=42)
        chi = net._compute_euler_characteristic()
        assert isinstance(chi, int)

    def test_snapshot_returns_dict(self):
        net = HyperconnectedQuantumNetwork(n_nodes=4, seed=42)
        snap = net.snapshot()
        for key in ["time_step", "euler_characteristic", "network_fidelity",
                    "average_entanglement"]:
            assert key in snap

    def test_decoherence_reduces_fidelity(self):
        net = HyperconnectedQuantumNetwork(n_nodes=4, seed=42)
        initial_fidelity = net.get_network_fidelity()
        for _ in range(50):
            net.apply_global_decoherence(0.2, 0.01)
        final_fidelity = net.get_network_fidelity()
        assert final_fidelity <= initial_fidelity + 1e-6

    def test_laplacian_psd(self):
        net = HyperconnectedQuantumNetwork(n_nodes=5, seed=42)
        L = net.get_laplacian()
        eigenvalues = np.linalg.eigvalsh(L)
        assert np.all(eigenvalues >= -1e-8)