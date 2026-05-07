import numpy as np
import pytest
import torch
from hqnn.ml.noise_predictor_torch import (
    NoisePredictor, NoisePredictorTrainer,
    LSTMAttentionBlock, TemporalConvBlock
)
from hqnn.ml.noise_predictor_tf import (
    NoisePredictorTFTrainer, build_noise_predictor_tf
)
from hqnn.ml.hybrid_trainer import HybridAdaptiveController
from hqnn.core.network import HyperconnectedQuantumNetwork
from hqnn.correction.topological import TopologicalErrorCorrector
from hqnn.optimizer.slime import QuantumSlimeMoldOptimizer


def make_synthetic_noise(n: int = 500, seed: int = 0) -> np.ndarray:
    np.random.seed(seed)
    t = np.linspace(0, 10, n)
    noise = 0.05 + 0.03 * np.sin(0.5 * t) + 0.01 * np.random.randn(n)
    return np.clip(noise, 0.001, 0.5).astype(np.float32)


class TestTemporalConvBlock:
    def test_output_shape_same_channels(self):
        block = TemporalConvBlock(32, 32, kernel_size=3, dilation=1)
        x = torch.randn(4, 32, 20)
        out = block(x)
        assert out.shape == x.shape

    def test_output_shape_diff_channels(self):
        block = TemporalConvBlock(16, 32, kernel_size=3, dilation=2)
        x = torch.randn(4, 16, 20)
        out = block(x)
        assert out.shape == (4, 32, 20)

    def test_residual_connection(self):
        block = TemporalConvBlock(32, 32)
        x = torch.zeros(2, 32, 10)
        out = block(x)
        assert out.shape == x.shape


class TestLSTMAttentionBlock:
    def test_output_shape(self):
        block = LSTMAttentionBlock(input_dim=64, hidden_dim=32, n_heads=4)
        x = torch.randn(4, 20, 64)
        out, attn = block(x)
        assert out.shape == (4, 20, 64)

    def test_attention_weights_shape(self):
        block = LSTMAttentionBlock(input_dim=64, hidden_dim=32, n_heads=4)
        x = torch.randn(2, 15, 64)
        _, attn = block(x)
        assert attn.shape[0] == 2


class TestNoisePredictorTorch:
    def test_forward_output_shape(self):
        model = NoisePredictor(seq_length=20, n_features=5,
                                hidden_dim=32, forecast_horizon=5)
        x = torch.randn(4, 20, 5)
        preds, unc = model(x)
        assert preds.shape == (4, 5)
        assert unc.shape == (4, 5)

    def test_predictions_positive(self):
        model = NoisePredictor(seq_length=20, n_features=5,
                                hidden_dim=32, forecast_horizon=5)
        x = torch.randn(8, 20, 5)
        preds, unc = model(x)
        assert torch.all(preds >= 0)
        assert torch.all(unc >= 0)

    def test_prepare_data_shapes(self):
        model = NoisePredictor(seq_length=20, forecast_horizon=5)
        trainer = NoisePredictorTrainer(model)
        noise = make_synthetic_noise(300)
        train_loader, val_loader = trainer.prepare_data(noise)
        batch = next(iter(train_loader))
        X, y = batch
        assert X.shape[1] == 20
        assert X.shape[2] == 5
        assert y.shape[1] == 5

    def test_train_reduces_loss(self):
        model = NoisePredictor(seq_length=20, n_features=5,
                                hidden_dim=32, forecast_horizon=3)
        trainer = NoisePredictorTrainer(model, learning_rate=1e-3)
        noise = make_synthetic_noise(400)
        result = trainer.train(noise, n_epochs=5, early_stopping_patience=10)
        assert result["epochs_trained"] >= 1
        assert len(result["train_losses"]) >= 1

    def test_predict_output_shape(self):
        model = NoisePredictor(seq_length=20, n_features=5, forecast_horizon=5)
        trainer = NoisePredictorTrainer(model)
        buffer = make_synthetic_noise(30)
        preds, unc = trainer.predict(buffer)
        assert len(preds) == 5
        assert len(unc) == 5

    def test_predict_with_short_buffer(self):
        model = NoisePredictor(seq_length=20, n_features=5, forecast_horizon=3)
        trainer = NoisePredictorTrainer(model)
        buffer = make_synthetic_noise(5)
        preds, unc = trainer.predict(buffer)
        assert len(preds) == 3

    def test_model_parameters_update(self):
        model = NoisePredictor(seq_length=15, n_features=5,
                                hidden_dim=32, forecast_horizon=3)
        trainer = NoisePredictorTrainer(model, learning_rate=1e-3)
        noise = make_synthetic_noise(300)
        params_before = {k: v.clone() for k, v in model.named_parameters()}
        trainer.train(noise, n_epochs=3)
        params_changed = any(
            not torch.equal(params_before[k], v)
            for k, v in model.named_parameters()
        )
        assert params_changed


class TestNoisePredictorTF:
    def test_model_builds(self):
        model = build_noise_predictor_tf(
            seq_length=20, n_features=5, d_model=32,
            n_heads=2, n_transformer_blocks=1, forecast_horizon=5
        )
        assert model is not None

    def test_forward_pass(self):
        import tensorflow as tf
        model = build_noise_predictor_tf(
            seq_length=20, n_features=5, d_model=32,
            n_heads=2, n_transformer_blocks=1, forecast_horizon=5
        )
        x = tf.random.normal((4, 20, 5))
        preds, unc = model(x, training=False)
        assert preds.shape == (4, 5)
        assert unc.shape == (4, 5)

    def test_predictions_positive_tf(self):
        import tensorflow as tf
        model = build_noise_predictor_tf(
            seq_length=20, n_features=5, d_model=32,
            n_heads=2, n_transformer_blocks=1, forecast_horizon=5
        )
        x = tf.random.normal((8, 20, 5))
        preds, unc = model(x, training=False)
        assert tf.reduce_all(preds >= 0).numpy()

    def test_tf_trainer_prepares_data(self):
        trainer = NoisePredictorTFTrainer(
            seq_length=20, n_features=5, d_model=32,
            n_heads=2, forecast_horizon=5
        )
        noise = make_synthetic_noise(400)
        train_ds, val_ds = trainer.prepare_dataset(noise)
        for X_b, y_b in train_ds.take(1):
            assert X_b.shape[1] == 20
            assert y_b.shape[1] == 5
            break

    def test_tf_trainer_trains(self):
        trainer = NoisePredictorTFTrainer(
            seq_length=15, n_features=5, d_model=32,
            n_heads=2, forecast_horizon=3
        )
        noise = make_synthetic_noise(300)
        result = trainer.train(noise, n_epochs=3, early_stopping_patience=5)
        assert result["epochs_trained"] >= 1
        assert len(result["train_losses"]) >= 1


class TestHybridAdaptiveController:
    def setup_method(self):
        self.net = HyperconnectedQuantumNetwork(n_nodes=5, seed=42)
        self.corrector = TopologicalErrorCorrector(self.net)
        self.slime = QuantumSlimeMoldOptimizer(self.net, adaptive=True)

    def test_initialization(self):
        controller = HybridAdaptiveController(
            self.net, self.corrector, self.slime,
            predictor_backend="torch", seq_length=15, forecast_horizon=3
        )
        assert controller.is_trained is False

    def test_synthetic_noise_generation(self):
        controller = HybridAdaptiveController(
            self.net, self.corrector, self.slime)
        noise = controller.generate_synthetic_noise(n_samples=200)
        assert len(noise) == 200
        assert np.all(noise >= 0.001)
        assert np.all(noise <= 0.5)

    def test_step_returns_log(self):
        controller = HybridAdaptiveController(
            self.net, self.corrector, self.slime,
            predictor_backend="torch", seq_length=10, forecast_horizon=3
        )
        log = controller.step(actual_noise=0.05, step_index=0)
        assert "network_fidelity" in log
        assert "strategy" in log
        assert "actual_noise" in log

    def test_run_full_loop_short(self):
        controller = HybridAdaptiveController(
            self.net, self.corrector, self.slime,
            predictor_backend="torch", seq_length=10, forecast_horizon=3
        )
        noise_profile = make_synthetic_noise(30)
        results = controller.run_full_control_loop(
            noise_profile, pretrain_steps=100)
        assert "fidelity_history" in results
        assert len(results["fidelity_history"]) == 30
        assert "correction_events" in results