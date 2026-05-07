import numpy as np
import pytest
from hqnn.core.network import HyperconnectedQuantumNetwork
from hqnn.core.quantum_node import NodeConfig
from hqnn.core.hamiltonian import NetworkHamiltonian
from hqnn.correction.topological import TopologicalErrorCorrector
from hqnn.correction.surface_code import SurfaceCode
from hqnn.optimizer.slime import QuantumSlimeMoldOptimizer
from hqnn.algorithms.grover import GroverSearch
from hqnn.algorithms.vqe import VQEAlgorithm
from hqnn.cage.beam_cage import QuantumBeamCage, BeamParameters
from hqnn.utils.metrics import network_topology_metrics, correction_efficiency_metrics


class TestFullPipeline:
    def setup_method(self):
        np.random.seed(0)
        self.node_config = NodeConfig(n_qubits=2, decoherence_model="lindblad")
        self.net = HyperconnectedQuantumNetwork(
            n_nodes=6, connectivity=0.75,
            node_config=self.node_config, seed=0
        )
        self.corrector = TopologicalErrorCorrector(
            self.net, fidelity_threshold=0.75)
        self.slime = QuantumSlimeMoldOptimizer(
            self.net, decay_rate=0.08, adaptive=True)

    def test_network_survives_decoherence_and_correction(self):
        for step in range(30):
            self.net.apply_global_decoherence(0.1, 0.01)
            self.net.update_entanglement_all()
            self.slime.step(0.1)
            diag = self.corrector.diagnose()
            self.corrector.correct(diag)
        fidelity = self.net.get_network_fidelity()
        assert fidelity > 0.3

    def test_euler_characteristic_bounded(self):
        initial_euler = self.net.initial_euler
        for _ in range(20):
            self.net.apply_global_decoherence(0.05, 0.01)
            diag = self.corrector.diagnose()
            self.corrector.correct(diag)
        snapshots = self.net.euler_history
        for chi in snapshots:
            assert abs(chi - initial_euler) <= 10

    def test_slime_improves_flow(self):
        noise_profile = np.full(50, 0.05)
        self.slime.run(n_steps=50, noise_profile=noise_profile)
        health = self.slime.get_network_health_score()
        assert 0.0 <= health <= 1.0

    def test_correction_log_grows(self):
        for _ in range(15):
            self.net.apply_global_decoherence(0.3, 0.05)
            diag = self.corrector.diagnose()
            self.corrector.correct(diag)
        stats = self.corrector.get_correction_statistics()
        assert stats["total_detections"] == 15

    def test_hamiltonian_evolution_consistent(self):
        H_builder = NetworkHamiltonian(n_nodes=self.net.n)
        A = np.real(self.net.get_adjacency_matrix())
        H = H_builder.build_full(A, self.net.initial_euler)
        U = H_builder.time_evolution_operator(H, dt=0.001)
        assert np.allclose(U @ U.conj().T,
                           np.eye(U.shape[0]), atol=1e-6)

    def test_topology_metrics_after_simulation(self):
        for _ in range(10):
            self.net.apply_global_decoherence(0.05, 0.01)
        A = np.abs(np.real(self.net.get_adjacency_matrix()))
        metrics = network_topology_metrics(A)
        assert metrics["n_nodes"] == self.net.n
        assert metrics["spectral_gap"] >= 0


class TestAlgorithmsIntegration:
    def test_grover_with_network_correction(self):
        np.random.seed(1)
        net = HyperconnectedQuantumNetwork(n_nodes=5, seed=1)
        grover = GroverSearch(n_qubits=4, network=net)
        result = grover.run(targets=[9], noise_level=0.02,
                             use_network_correction=True)
        assert "success" in result
        assert result["final_target_probability"] > 0.1

    def test_vqe_finds_reasonable_energy(self):
        np.random.seed(2)
        vqe = VQEAlgorithm(n_qubits=4, n_layers=2, molecule="H2")
        result = vqe.optimize_gradient_descent(n_iterations=30, noise=0.02)
        assert result["final_energy"] < 0
        assert result["convergence_error"] < 5.0

    def test_surface_code_integration(self):
        sc = SurfaceCode(distance=3)
        results = sc.simulate_circuit_cycles(
            physical_error_rate=0.005, n_cycles=20)
        assert 0.0 <= results["logical_error_rate"] <= 1.0
        assert len(results["cycle_failures"]) == 20

    def test_beam_cage_integration(self):
        params = BeamParameters(beam_waist=2e-6, power=5e-3)
        cage = QuantumBeamCage(params)
        result = cage.simulate_atomic_motion(
            n_atoms=50, T_atom=1e-6, n_steps=30)
        assert result["trapped_fraction"][-1] > 0.5
        assert result["coherence"][-1] > 0.0


class TestEndToEnd:
    def test_full_pipeline_runs(self):
        np.random.seed(99)
        node_config = NodeConfig(n_qubits=2)
        net = HyperconnectedQuantumNetwork(
            n_nodes=9, connectivity=0.75,
            node_config=node_config, seed=99
        )
        corrector = TopologicalErrorCorrector(net)
        slime = QuantumSlimeMoldOptimizer(net, adaptive=True)

        t = np.linspace(0, 5, 60)
        noise_profile = np.clip(
            0.05 + 0.03 * np.sin(0.5 * t) + 0.01 * np.random.randn(60),
            0.001, 0.5
        ).astype(np.float32)

        fidelities = []
        entanglements = []
        for step, noise in enumerate(noise_profile):
            net.apply_global_decoherence(float(noise), dt=0.01)
            net.update_entanglement_all()
            net.apply_circuit_step()
            slime.step(float(noise))
            diag = corrector.diagnose()
            corrector.correct(diag)
            snap = net.snapshot()
            fidelities.append(snap["network_fidelity"])
            entanglements.append(snap["average_entanglement"])

        assert len(fidelities) == 60
        assert np.mean(fidelities) > 0.2
        assert corrector.total_detections == 60

    def test_grover_shor_vqe_all_run(self):
        np.random.seed(7)

        grover = GroverSearch(n_qubits=3)
        g_result = grover.run([5], noise_level=0.01)
        assert "final_target_probability" in g_result

        from hqnn.algorithms.shor import ShorAlgorithm
        shor = ShorAlgorithm(n_qubits=6)
        s_result = shor.factor(15, noise=0.01)
        assert "factors" in s_result

        vqe = VQEAlgorithm(n_qubits=2, n_layers=1)
        v_result = vqe.optimize_gradient_descent(n_iterations=10, noise=0.01)
        assert "final_energy" in v_result