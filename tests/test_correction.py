import numpy as np
import pytest
from hqnn.core.network import HyperconnectedQuantumNetwork
from hqnn.core.quantum_node import NodeConfig
from hqnn.correction.topological import TopologicalErrorCorrector
from hqnn.correction.surface_code import SurfaceCode
from hqnn.optimizer.slime import QuantumSlimeMoldOptimizer


class TestTopologicalCorrector:
    def setup_method(self):
        self.net = HyperconnectedQuantumNetwork(n_nodes=6, seed=42)
        self.corrector = TopologicalErrorCorrector(self.net)

    def test_diagnose_healthy_network(self):
        diagnosis = self.corrector.diagnose()
        assert "needs_correction" in diagnosis
        assert "euler_drift" in diagnosis
        assert "network_fidelity" in diagnosis

    def test_correction_after_decoherence(self):
        for _ in range(30):
            self.net.apply_global_decoherence(0.3, 0.05)
        diagnosis = self.corrector.diagnose()
        corrected = self.corrector.correct(diagnosis)
        post_diagnosis = self.corrector.diagnose()
        assert post_diagnosis["network_fidelity"] >= diagnosis["network_fidelity"] - 0.05

    def test_correction_statistics(self):
        for _ in range(10):
            for __ in range(5):
                self.net.apply_global_decoherence(0.2, 0.01)
            diag = self.corrector.diagnose()
            self.corrector.correct(diag)
        stats = self.corrector.get_correction_statistics()
        assert stats["total_detections"] == 10
        assert stats["total_corrections"] >= 0

    def test_euler_characteristic_tracked(self):
        for _ in range(5):
            self.net.snapshot()
        assert len(self.net.euler_history) == 6


class TestSurfaceCode:
    def test_initialization(self):
        sc = SurfaceCode(distance=3)
        assert sc.n_data == 9
        assert sc.d == 3

    def test_syndrome_measurement(self):
        sc = SurfaceCode(distance=3)
        x_syn, z_syn = sc.measure_syndrome()
        assert x_syn.dtype == int
        assert z_syn.dtype == int

    def test_error_injection_and_decode(self):
        sc = SurfaceCode(distance=3)
        sc.inject_errors(error_rate=0.05)
        result = sc.decode_and_correct()
        assert "logical_failure" in result
        assert "n_physical_errors" in result

    def test_logical_error_rate_below_threshold(self):
        sc = SurfaceCode(distance=3)
        rate = sc.logical_error_rate(physical_rate=0.005, n_trials=500)
        assert 0.0 <= rate <= 1.0

    def test_logical_error_rate_above_threshold(self):
        sc = SurfaceCode(distance=3)
        rate_below = sc.logical_error_rate(0.005, n_trials=300)
        rate_above = sc.logical_error_rate(0.02, n_trials=300)
        assert rate_above >= rate_below

    def test_resource_overhead(self):
        sc = SurfaceCode(distance=5)
        overhead = sc.resource_overhead()
        assert overhead["data_qubits"] == 25
        assert overhead["logical_qubits"] == 1

    def test_circuit_simulation(self):
        sc = SurfaceCode(distance=3)
        result = sc.simulate_circuit_cycles(physical_error_rate=0.01, n_cycles=10)
        assert len(result["cycle_failures"]) == 10
        assert "logical_error_rate" in result


class TestSlimeOptimizer:
    def test_flow_history_grows(self):
        net = HyperconnectedQuantumNetwork(n_nodes=5, seed=42)
        optimizer = QuantumSlimeMoldOptimizer(net)
        profile = np.ones(50) * 0.05
        result = optimizer.run(n_steps=50, noise_profile=profile)
        assert len(result["flow_history"]) == 50

    def test_slime_factors_positive(self):
        net = HyperconnectedQuantumNetwork(n_nodes=4, seed=42)
        optimizer = QuantumSlimeMoldOptimizer(net)
        for edge in net.edges.values():
            edge.quantum_flow = 0.3
        for _ in range(20):
            optimizer.step(noise_level=0.05)
        for edge in net.edges.values():
            assert edge.slime_factor > 0

    def test_adaptive_decay_changes_mu(self):
        net = HyperconnectedQuantumNetwork(n_nodes=5, seed=42)
        optimizer = QuantumSlimeMoldOptimizer(net, decay_rate=0.1, adaptive=True)
        initial_mu = optimizer.mu
        for _ in range(30):
            optimizer.step(noise_level=0.3)
        assert optimizer.mu != initial_mu or len(optimizer.adaptation_log) > 0

    def test_health_score_range(self):
        net = HyperconnectedQuantumNetwork(n_nodes=5, seed=42)
        optimizer = QuantumSlimeMoldOptimizer(net)
        profile = np.random.uniform(0.02, 0.1, 30)
        optimizer.run(n_steps=30, noise_profile=profile)
        health = optimizer.get_network_health_score()
        assert 0.0 <= health <= 1.0