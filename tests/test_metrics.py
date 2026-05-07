import numpy as np
import pytest
from hqnn.utils.metrics import (
    quantum_volume,
    trace_distance,
    bures_distance,
    quantum_coherence_l1,
    quantum_coherence_relative_entropy,
    relative_entropy,
    quantum_fisher_information,
    multipartite_entanglement_measure,
    network_topology_metrics,
    noise_prediction_metrics,
    correction_efficiency_metrics,
)
from hqnn.utils.gates import (
    density_from_state, bell_state, pauli_z, pauli_x
)


class TestQuantumMetrics:
    def test_quantum_volume_positive(self):
        qv = quantum_volume(n_qubits=4, depth=10, error_rate=0.001)
        assert qv > 0

    def test_quantum_volume_decreases_with_error(self):
        qv_low = quantum_volume(4, 10, 0.001)
        qv_high = quantum_volume(4, 10, 0.1)
        assert qv_low >= qv_high

    def test_trace_distance_identical_states(self):
        rho = density_from_state(np.array([1, 0], dtype=complex))
        assert abs(trace_distance(rho, rho)) < 1e-10

    def test_trace_distance_orthogonal_states(self):
        rho0 = density_from_state(np.array([1, 0], dtype=complex))
        rho1 = density_from_state(np.array([0, 1], dtype=complex))
        td = trace_distance(rho0, rho1)
        assert abs(td - 1.0) < 1e-8

    def test_trace_distance_range(self):
        rho = density_from_state(np.array([1, 0], dtype=complex))
        sigma = np.eye(2, dtype=complex) / 2
        td = trace_distance(rho, sigma)
        assert 0.0 <= td <= 1.0

    def test_bures_distance_identical(self):
        rho = density_from_state(np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2))
        assert abs(bures_distance(rho, rho)) < 1e-8

    def test_bures_distance_nonnegative(self):
        rho = density_from_state(np.array([1, 0], dtype=complex))
        sigma = np.eye(2, dtype=complex) / 2
        assert bures_distance(rho, sigma) >= 0

    def test_coherence_l1_zero_for_diagonal(self):
        rho = np.diag([0.5, 0.3, 0.2]).astype(complex)
        assert abs(quantum_coherence_l1(rho)) < 1e-10

    def test_coherence_l1_positive_for_superposition(self):
        state = np.array([1, 1], dtype=complex) / np.sqrt(2)
        rho = density_from_state(state)
        assert quantum_coherence_l1(rho) > 0

    def test_coherence_relative_entropy_nonnegative(self):
        state = np.array([1, 1], dtype=complex) / np.sqrt(2)
        rho = density_from_state(state)
        cre = quantum_coherence_relative_entropy(rho)
        assert cre >= -1e-10

    def test_relative_entropy_identical(self):
        rho = np.eye(4, dtype=complex) / 4
        re = relative_entropy(rho, rho)
        assert abs(re) < 1e-8

    def test_quantum_fisher_information_nonnegative(self):
        rho = np.eye(2, dtype=complex) / 2
        H = pauli_z()
        qfi = quantum_fisher_information(rho, H)
        assert qfi >= -1e-10

    def test_multipartite_entanglement_keys(self):
        states = [np.eye(4, dtype=complex) / 4 for _ in range(3)]
        result = multipartite_entanglement_measure(states)
        assert "global_entanglement" in result
        assert "subsystem_entropies" in result
        assert "pairwise_concurrence" in result

    def test_multipartite_entanglement_range(self):
        states = [np.eye(4, dtype=complex) / 4 for _ in range(4)]
        result = multipartite_entanglement_measure(states)
        assert 0.0 <= result["global_entanglement"] <= 1.0


class TestNetworkMetrics:
    def test_topology_metrics_keys(self):
        A = np.array([
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [0, 1, 1, 0],
        ], dtype=float)
        metrics = network_topology_metrics(A)
        for key in ["mean_degree", "spectral_gap", "clustering_coefficient",
                    "euler_characteristic", "n_nodes", "n_edges"]:
            assert key in metrics

    def test_topology_metrics_complete_graph(self):
        n = 5
        A = np.ones((n, n), dtype=float) - np.eye(n)
        metrics = network_topology_metrics(A)
        assert abs(metrics["mean_degree"] - (n - 1)) < 1e-10
        assert metrics["spectral_gap"] > 0

    def test_topology_metrics_disconnected(self):
        A = np.zeros((4, 4), dtype=float)
        metrics = network_topology_metrics(A)
        assert metrics["mean_degree"] == 0.0
        assert metrics["spectral_gap"] == 0.0


class TestNoisePredictionMetrics:
    def test_perfect_prediction(self):
        y = np.array([0.1, 0.2, 0.15, 0.05, 0.3])
        metrics = noise_prediction_metrics(y, y)
        assert abs(metrics["mae"]) < 1e-10
        assert abs(metrics["mse"]) < 1e-10
        assert abs(metrics["r2"] - 1.0) < 1e-8

    def test_metrics_keys(self):
        y_true = np.random.uniform(0.01, 0.3, 100)
        y_pred = y_true + 0.01 * np.random.randn(100)
        metrics = noise_prediction_metrics(y_true, y_pred)
        for key in ["mae", "mse", "rmse", "mape", "r2",
                    "pearson_r", "spearman_r"]:
            assert key in metrics

    def test_uncertainty_coverage_keys(self):
        y_true = np.random.uniform(0.01, 0.3, 100)
        y_pred = y_true + 0.01 * np.random.randn(100)
        y_unc = np.full(100, 0.05)
        metrics = noise_prediction_metrics(y_true, y_pred, y_unc)
        assert "coverage_90" in metrics
        assert "coverage_95" in metrics
        assert "sharpness" in metrics

    def test_r2_range(self):
        y_true = np.random.uniform(0.01, 0.3, 100)
        y_pred = y_true + 0.05 * np.random.randn(100)
        metrics = noise_prediction_metrics(y_true, y_pred)
        assert metrics["r2"] <= 1.0


class TestCorrectionMetrics:
    def test_empty_log(self):
        metrics = correction_efficiency_metrics([], total_steps=100)
        assert metrics["correction_rate"] == 0.0

    def test_correction_rate(self):
        log = [{"step": i, "error_type": "decoherence"} for i in range(0, 100, 10)]
        metrics = correction_efficiency_metrics(log, total_steps=100)
        assert abs(metrics["correction_rate"] - 0.1) < 1e-10

    def test_mean_interval(self):
        log = [{"step": i * 10, "error_type": "topological"} for i in range(5)]
        metrics = correction_efficiency_metrics(log, total_steps=50)
        assert abs(metrics["mean_correction_interval"] - 10.0) < 1e-10

    def test_error_type_distribution(self):
        log = (
            [{"step": i, "error_type": "decoherence"} for i in range(3)] +
            [{"step": i + 3, "error_type": "topological"} for i in range(2)]
        )
        metrics = correction_efficiency_metrics(log, total_steps=10)
        dist = metrics["error_type_distribution"]
        assert dist.get("decoherence", 0) == 3
        assert dist.get("topological", 0) == 2