import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from hqnn.ml.noise_predictor_torch import NoisePredictor, NoisePredictorTrainer
from hqnn.ml.noise_predictor_tf import NoisePredictorTFTrainer
from hqnn.core.network import HyperconnectedQuantumNetwork
from hqnn.optimizer.slime import QuantumSlimeMoldOptimizer
from hqnn.correction.topological import TopologicalErrorCorrector


class HybridAdaptiveController:
    def __init__(self,
                 network: HyperconnectedQuantumNetwork,
                 corrector: TopologicalErrorCorrector,
                 slime_optimizer: QuantumSlimeMoldOptimizer,
                 predictor_backend: str = "torch",
                 seq_length: int = 20,
                 forecast_horizon: int = 5):
        self.network = network
        self.corrector = corrector
        self.slime = slime_optimizer
        self.backend = predictor_backend
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon

        if predictor_backend == "torch":
            model = NoisePredictor(seq_length=seq_length,
                                    forecast_horizon=forecast_horizon)
            self.trainer = NoisePredictorTrainer(model)
        else:
            self.trainer = NoisePredictorTFTrainer(
                seq_length=seq_length,
                forecast_horizon=forecast_horizon)

        self.noise_buffer: List[float] = []
        self.control_log: List[Dict] = []
        self.is_trained = False

    def pretrain(self, synthetic_noise: np.ndarray,
                  n_epochs: int = 80) -> Dict:
        result = self.trainer.train(synthetic_noise, n_epochs=n_epochs)
        self.is_trained = True
        return result

    def generate_synthetic_noise(self, n_samples: int = 2000,
                                   seed: int = 42) -> np.ndarray:
        np.random.seed(seed)
        t = np.linspace(0, 40, n_samples)
        base = 0.05 + 0.03 * np.sin(0.3 * t) + 0.02 * np.sin(1.7 * t)
        spikes = np.zeros(n_samples)
        spike_positions = np.random.choice(n_samples, size=30, replace=False)
        spikes[spike_positions] = np.random.uniform(0.1, 0.4, 30)
        noise = base + spikes + 0.01 * np.random.randn(n_samples)
        return np.clip(noise, 0.001, 0.5).astype(np.float32)

    def _get_predictions(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_trained or len(self.noise_buffer) < 5:
            dummy = np.full(self.forecast_horizon, 0.05, dtype=np.float32)
            return dummy, dummy * 0.1

        buffer_arr = np.array(self.noise_buffer, dtype=np.float32)
        return self.trainer.predict(buffer_arr)

    def _compute_preemptive_strategy(self,
                                       predictions: np.ndarray,
                                       uncertainty: np.ndarray) -> Dict:
        max_pred = float(np.max(predictions))
        mean_pred = float(np.mean(predictions))
        max_unc = float(np.max(uncertainty))
        risk = max_pred + 0.5 * max_unc

        if risk > 0.20:
            strategy = "high_risk_preemptive"
            self.slime.mu = 0.03
            self.slime.topology_protection = 0.8
            correction_frequency = 1
        elif risk > 0.12:
            strategy = "medium_risk_adaptive"
            self.slime.mu = 0.06
            self.slime.topology_protection = 0.6
            correction_frequency = 3
        else:
            strategy = "low_risk_normal"
            self.slime.mu = 0.10
            self.slime.topology_protection = 0.5
            correction_frequency = 5

        return {
            "strategy": strategy,
            "risk_score": risk,
            "correction_frequency": correction_frequency,
            "predicted_max": max_pred,
            "predicted_mean": mean_pred,
            "uncertainty_max": max_unc,
        }

    def step(self, actual_noise: float,
              step_index: int) -> Dict:
        self.noise_buffer.append(actual_noise)
        if len(self.noise_buffer) > self.seq_length * 3:
            self.noise_buffer.pop(0)

        predictions, uncertainty = self._get_predictions()
        strategy_info = self._compute_preemptive_strategy(predictions, uncertainty)

        self.network.apply_global_decoherence(gamma=actual_noise, dt=0.01)
        self.network.update_entanglement_all()
        self.network.apply_circuit_step()

        slime_result = self.slime.step(actual_noise)

        correct_now = (step_index % strategy_info["correction_frequency"] == 0)
        corrected = False
        if correct_now:
            diagnosis = self.corrector.diagnose()
            corrected = self.corrector.correct(diagnosis)
        else:
            diagnosis = {}

        snapshot = self.network.snapshot()
        log_entry = {
            "step": step_index,
            "actual_noise": actual_noise,
            "predictions": predictions.tolist(),
            "uncertainty": uncertainty.tolist(),
            "strategy": strategy_info["strategy"],
            "risk_score": strategy_info["risk_score"],
            "slime_flow": slime_result["average_flow"],
            "network_fidelity": snapshot["network_fidelity"],
            "average_entanglement": snapshot["average_entanglement"],
            "euler_drift": snapshot["euler_drift"],
            "corrected": corrected,
        }
        self.control_log.append(log_entry)
        return log_entry

    def run_full_control_loop(self,
                               noise_profile: np.ndarray,
                               pretrain_steps: int = 500) -> Dict:
        if not self.is_trained:
            synthetic = self.generate_synthetic_noise(n_samples=pretrain_steps * 4)
            self.pretrain(synthetic, n_epochs=60)

        results = {
            "fidelity_history": [],
            "entanglement_history": [],
            "noise_history": [],
            "prediction_history": [],
            "strategy_history": [],
            "correction_events": [],
            "euler_drift_history": [],
        }

        for step_idx, noise in enumerate(noise_profile):
            log = self.step(float(noise), step_idx)
            results["fidelity_history"].append(log["network_fidelity"])
            results["entanglement_history"].append(log["average_entanglement"])
            results["noise_history"].append(log["actual_noise"])
            results["prediction_history"].append(log["predictions"][0])
            results["strategy_history"].append(log["strategy"])
            results["euler_drift_history"].append(log["euler_drift"])
            if log["corrected"]:
                results["correction_events"].append(step_idx)

        stats = self.corrector.get_correction_statistics()
        results["correction_statistics"] = stats
        results["health_score"] = self.slime.get_network_health_score()
        results["control_log"] = self.control_log

        return results