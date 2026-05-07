import numpy as np
import pytest
from hqnn.algorithms.grover import GroverSearch
from hqnn.algorithms.shor import ShorAlgorithm, QuantumFourierTransform
from hqnn.algorithms.vqe import VQEAlgorithm, VQEAnsatz, MolecularHamiltonian


class TestGrover:
    def test_initialization(self):
        g = GroverSearch(n_qubits=3)
        assert g.N == 8
        assert g.optimal_iterations == int(np.pi / 4 * np.sqrt(8))

    def test_initial_state_normalized(self):
        g = GroverSearch(n_qubits=4)
        state = g.initialize()
        assert abs(np.linalg.norm(state) - 1.0) < 1e-10

    def test_initial_state_uniform(self):
        g = GroverSearch(n_qubits=3)
        state = g.initialize()
        expected = 1.0 / np.sqrt(8)
        assert np.allclose(np.abs(state), expected, atol=1e-10)

    def test_oracle_flips_target(self):
        g = GroverSearch(n_qubits=3)
        state = g.initialize()
        target = 5
        flipped = g.oracle(state, [target])
        assert abs(flipped[target] + state[target]) < 1e-10
        for i in range(g.N):
            if i != target:
                assert abs(flipped[i] - state[i]) < 1e-10

    def test_diffusion_preserves_norm(self):
        g = GroverSearch(n_qubits=3)
        state = np.random.randn(8) + 1j * np.random.randn(8)
        state /= np.linalg.norm(state)
        diffused = g.diffusion(state)
        assert abs(np.linalg.norm(diffused) - 1.0) < 1e-8

    def test_run_finds_target(self):
        g = GroverSearch(n_qubits=4)
        target = 11
        result = g.run([target], noise_level=0.0, use_network_correction=False)
        assert result["final_target_probability"] > 0.8

    def test_run_with_noise_still_works(self):
        g = GroverSearch(n_qubits=3)
        result = g.run([3], noise_level=0.05)
        assert "success" in result
        assert result["final_target_probability"] > 0.3

    def test_multiple_targets(self):
        g = GroverSearch(n_qubits=4)
        targets = [2, 7, 12]
        result = g.run(targets, noise_level=0.0)
        assert result["final_target_probability"] > 0.5


class TestQFT:
    def test_qft_unitarity(self):
        qft = QuantumFourierTransform(n_qubits=3)
        F = qft.matrix()
        F_dag = F.conj().T
        result = F @ F_dag
        assert np.allclose(result, np.eye(8), atol=1e-8)

    def test_qft_inverse(self):
        qft = QuantumFourierTransform(n_qubits=3)
        state = np.zeros(8, dtype=complex)
        state[3] = 1.0
        forward = qft.apply(state, noise=0.0)
        backward = qft.apply_inverse(forward, noise=0.0)
        assert np.allclose(state, backward, atol=1e-8)

    def test_qft_preserves_norm(self):
        qft = QuantumFourierTransform(n_qubits=4)
        state = np.random.randn(16) + 1j * np.random.randn(16)
        state /= np.linalg.norm(state)
        transformed = qft.apply(state, noise=0.0)
        assert abs(np.linalg.norm(transformed) - 1.0) < 1e-8


class TestShor:
    def test_factor_15(self):
        shor = ShorAlgorithm(n_qubits=8)
        result = shor.factor(N=15, noise=0.01)
        if result["success"]:
            f1, f2 = result["factors"]
            assert f1 * f2 == 15
            assert f1 in [3, 5] and f2 in [3, 5]

    def test_factor_even_number(self):
        shor = ShorAlgorithm()
        result = shor.factor(N=14)
        assert result["method"] == "trivial_even"
        assert 2 in result["factors"]

    def test_factor_prime_detected(self):
        shor = ShorAlgorithm()
        result = shor.factor(N=7)
        assert result["method"] == "prime_input"

    def test_period_finding_returns_dict(self):
        shor = ShorAlgorithm(n_qubits=6)
        result = shor._quantum_period_finding(a=2, N=15)
        assert "classical_period" in result
        assert result["classical_period"] > 0


class TestVQE:
    def test_hamiltonian_hermitian(self):
        H_builder = MolecularHamiltonian(n_qubits=4, molecule="H2")
        H = H_builder.build()
        assert np.allclose(H, H.conj().T, atol=1e-10)

    def test_hamiltonian_real_eigenvalues(self):
        H_builder = MolecularHamiltonian(n_qubits=4, molecule="H2")
        H = H_builder.build()
        eigenvalues = np.linalg.eigvalsh(H)
        assert np.all(np.isreal(eigenvalues))

    def test_ansatz_normalized(self):
        ansatz = VQEAnsatz(n_qubits=4, n_layers=2)
        params = np.random.uniform(-np.pi, np.pi, ansatz.n_params)
        state = ansatz.build_circuit(params)
        assert abs(np.linalg.norm(state) - 1.0) < 1e-8

    def test_vqe_converges(self):
        vqe = VQEAlgorithm(n_qubits=4, n_layers=2, molecule="H2")
        result = vqe.optimize_gradient_descent(n_iterations=30, noise=0.01)
        assert result["convergence_error"] < 2.0
        assert len(result["energy_history"]) > 0

    def test_vqe_exact_energy(self):
        vqe = VQEAlgorithm(n_qubits=4, n_layers=2)
        E0, psi0 = vqe.exact_ground_energy()
        assert isinstance(E0, float)
        assert abs(np.linalg.norm(psi0) - 1.0) < 1e-8

    def test_parameter_shift_gradient_shape(self):
        vqe = VQEAlgorithm(n_qubits=2, n_layers=1)
        params = np.random.uniform(-np.pi, np.pi, vqe.ansatz.n_params)
        grad = vqe.parameter_shift_gradient(params, noise=0.0)
        assert grad.shape == params.shape