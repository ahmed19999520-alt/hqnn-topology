import numpy as np
import pytest
import torch
import os


def pytest_configure(config):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


@pytest.fixture(autouse=True)
def set_random_seeds():
    np.random.seed(42)
    torch.manual_seed(42)
    yield


@pytest.fixture
def small_network():
    from hqnn.core.network import HyperconnectedQuantumNetwork
    from hqnn.core.quantum_node import NodeConfig
    config = NodeConfig(n_qubits=2, initial_state="superposition")
    return HyperconnectedQuantumNetwork(n_nodes=4, connectivity=0.75,
                                        node_config=config, seed=42)


@pytest.fixture
def standard_network():
    from hqnn.core.network import HyperconnectedQuantumNetwork
    from hqnn.core.quantum_node import NodeConfig
    config = NodeConfig(n_qubits=2)
    return HyperconnectedQuantumNetwork(n_nodes=9, connectivity=0.75,
                                        node_config=config, seed=42)


@pytest.fixture
def corrector(standard_network):
    from hqnn.correction.topological import TopologicalErrorCorrector
    return TopologicalErrorCorrector(standard_network)


@pytest.fixture
def slime_optimizer(standard_network):
    from hqnn.optimizer.slime import QuantumSlimeMoldOptimizer
    return QuantumSlimeMoldOptimizer(standard_network, adaptive=True)


@pytest.fixture
def beam_cage():
    from hqnn.cage.beam_cage import QuantumBeamCage, BeamParameters
    params = BeamParameters(beam_waist=2e-6, wavelength=780e-9, power=5e-3)
    return QuantumBeamCage(params)


@pytest.fixture
def noise_sequence():
    t = np.linspace(0, 10, 500)
    noise = 0.05 + 0.03 * np.sin(0.5 * t) + 0.01 * np.random.randn(500)
    return np.clip(noise, 0.001, 0.5).astype(np.float32)


@pytest.fixture
def trained_torch_predictor(noise_sequence):
    from hqnn.ml.noise_predictor_torch import NoisePredictor, NoisePredictorTrainer
    model = NoisePredictor(seq_length=15, n_features=5,
                            hidden_dim=32, forecast_horizon=3)
    trainer = NoisePredictorTrainer(model, learning_rate=1e-3)
    trainer.train(noise_sequence, n_epochs=5, early_stopping_patience=5)
    return trainer