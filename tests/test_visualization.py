import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tempfile
import os
from hqnn.utils.visualization import (
    plot_density_matrix,
    plot_bloch_sphere,
    plot_network_graph,
    plot_entanglement_matrix,
    plot_training_curves,
    plot_noise_prediction,
    plot_quantum_circuit_evolution,
    plot_error_correction_comparison,
    plot_slime_dynamics,
)
from hqnn.utils.gates import density_from_state, bell_state


class TestVisualizationFunctions:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def _path(self, name: str) -> str:
        return os.path.join(self.tmpdir, name)

    def test_plot_density_matrix_saves(self):
        rho = density_from_state(bell_state(0))
        save_path = self._path("dm.png")
        fig = plot_density_matrix(rho, save_path=save_path)
        assert os.path.exists(save_path)
        assert fig is not None

    def test_plot_density_matrix_returns_figure(self):
        rho = np.eye(4, dtype=complex) / 4
        fig = plot_density_matrix(rho)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_bloch_sphere(self):
        vecs = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]
        save_path = self._path("bloch.png")
        fig = plot_bloch_sphere(vecs, labels=["X", "Y", "Z"],
                                 save_path=save_path)
        assert os.path.exists(save_path)
        plt.close(fig)

    def test_plot_network_graph_no_labels(self):
        A = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ], dtype=complex) * 0.5
        save_path = self._path("network.png")
        fig = plot_network_graph(A, save_path=save_path)
        assert os.path.exists(save_path)
        plt.close(fig)

    def test_plot_network_graph_with_fidelities(self):
        A = np.array([
            [0, 0.5, 0.3],
            [0.5, 0, 0.7],
            [0.3, 0.7, 0],
        ], dtype=complex)
        fids = [0.9, 0.7, 0.85]
        fig = plot_network_graph(A, node_fidelities=fids)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_entanglement_matrix(self):
        E = np.array([
            [1.0, 0.8, 0.3],
            [0.8, 1.0, 0.5],
            [0.3, 0.5, 1.0],
        ])
        save_path = self._path("entanglement.png")
        fig = plot_entanglement_matrix(E, save_path=save_path)
        assert os.path.exists(save_path)
        plt.close(fig)

    def test_plot_training_curves(self):
        train_losses = list(np.exp(-np.linspace(0, 3, 50)))
        val_losses = list(np.exp(-np.linspace(0, 2.5, 50)) * 1.05)
        save_path = self._path("training.png")
        fig = plot_training_curves(train_losses, val_losses,
                                    save_path=save_path)
        assert os.path.exists(save_path)
        plt.close(fig)

    def test_plot_noise_prediction_with_uncertainty(self):
        n = 100
        actuals = list(0.05 + 0.02 * np.sin(np.linspace(0, 10, n)))
        predictions = [a + 0.005 * np.random.randn() for a in actuals]
        uncertainties = [0.01] * n
        save_path = self._path("noise_pred.png")
        fig = plot_noise_prediction(actuals, predictions, uncertainties,
                                     save_path=save_path)
        assert os.path.exists(save_path)
        plt.close(fig)

    def test_plot_noise_prediction_without_uncertainty(self):
        actuals = list(np.random.uniform(0.02, 0.1, 80))
        predictions = [a + 0.01 * np.random.randn() for a in actuals]
        fig = plot_noise_prediction(actuals, predictions)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_circuit_evolution(self):
        n = 80
        fidelity = list(0.85 + 0.05 * np.sin(np.linspace(0, 10, n)))
        entanglement = list(0.4 + 0.1 * np.cos(np.linspace(0, 10, n)))
        noise = list(np.random.uniform(0.03, 0.08, n))
        corrections = [5, 15, 30, 55, 72]
        save_path = self._path("circuit.png")
        fig = plot_quantum_circuit_evolution(
            fidelity, entanglement, None, corrections, noise,
            save_path=save_path
        )
        assert os.path.exists(save_path)
        plt.close(fig)

    def test_plot_circuit_evolution_with_entropy(self):
        n = 50
        fidelity = list(np.random.uniform(0.8, 0.95, n))
        entanglement = list(np.random.uniform(0.3, 0.6, n))
        entropy = list(np.random.uniform(0.5, 1.5, n))
        noise = list(np.random.uniform(0.02, 0.07, n))
        fig = plot_quantum_circuit_evolution(
            fidelity, entanglement, entropy, [], noise)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_error_correction_comparison(self):
        error_rates = np.logspace(-3, -1, 15)
        our_rates = list(error_rates * 0.3)
        sc_rates = list(error_rates * 0.8)
        our_fids = list(1.0 - np.array(our_rates))
        save_path = self._path("comparison.png")
        fig = plot_error_correction_comparison(
            error_rates, our_rates, sc_rates, our_fids,
            save_path=save_path
        )
        assert os.path.exists(save_path)
        plt.close(fig)

    def test_plot_slime_dynamics(self):
        n = 100
        flow = list(np.random.uniform(0.1, 0.5, n))
        slime = list(np.random.uniform(0.8, 2.0, n))
        mu = list(np.random.uniform(0.08, 0.12, n))
        noise = list(np.random.uniform(0.03, 0.1, n))
        save_path = self._path("slime.png")
        fig = plot_slime_dynamics(flow, slime, mu, noise, save_path=save_path)
        assert os.path.exists(save_path)
        plt.close(fig)