import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import argparse
import json
import os
from pathlib import Path
from hqnn.ml.noise_predictor_torch import NoisePredictor, NoisePredictorTrainer
from hqnn.ml.noise_predictor_tf import NoisePredictorTFTrainer
from hqnn.utils.metrics import noise_prediction_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train noise predictor")
    parser.add_argument("--backend", type=str, default="torch",
                        choices=["torch", "tensorflow", "both"])
    parser.add_argument("--seq_length", type=int, default=20)
    parser.add_argument("--forecast_horizon", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--n_samples", type=int, default=3000)
    parser.add_argument("--noise_regime", type=str, default="mixed",
                        choices=["low", "medium", "high", "mixed", "spiky"])
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--output", type=str, default="outputs/predictor_results.png")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def generate_noise_dataset(n_samples: int, regime: str,
                             seed: int) -> np.ndarray:
    np.random.seed(seed)
    t = np.linspace(0, n_samples / 100, n_samples)

    if regime == "low":
        base = 0.02 + 0.01 * np.sin(0.3 * t)
        fluctuation = 0.005 * np.random.randn(n_samples)
        noise = base + fluctuation

    elif regime == "medium":
        base = 0.05 + 0.02 * np.sin(0.5 * t) + 0.01 * np.sin(1.7 * t + 0.5)
        fluctuation = 0.01 * np.random.randn(n_samples)
        noise = base + fluctuation

    elif regime == "high":
        base = 0.15 + 0.05 * np.sin(0.2 * t)
        fluctuation = 0.03 * np.random.randn(n_samples)
        noise = base + fluctuation

    elif regime == "spiky":
        base = 0.04 * np.ones(n_samples)
        spikes = np.zeros(n_samples)
        spike_idx = np.random.choice(n_samples, size=n_samples // 15, replace=False)
        spikes[spike_idx] = np.random.uniform(0.1, 0.4, size=len(spike_idx))
        noise = base + spikes + 0.005 * np.random.randn(n_samples)

    else:
        segments = []
        seg_len = n_samples // 4
        regimes_list = ["low", "medium", "high", "spiky"]
        for reg in regimes_list:
            seg = generate_noise_dataset(seg_len, reg, seed + hash(reg) % 100)
            segments.append(seg)
        noise = np.concatenate(segments)[:n_samples]

    return np.clip(noise, 0.001, 0.5).astype(np.float32)


def train_torch_predictor(noise_data: np.ndarray, args) -> dict:
    print("  Training PyTorch predictor...")
    model = NoisePredictor(
        seq_length=args.seq_length,
        n_features=5,
        hidden_dim=args.hidden_dim,
        n_lstm_layers=2,
        n_heads=4,
        forecast_horizon=args.forecast_horizon,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")

    trainer = NoisePredictorTrainer(
        model, learning_rate=args.lr, weight_decay=1e-4)

    checkpoint_path = None
    if args.checkpoint_dir:
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        checkpoint_path = f"{args.checkpoint_dir}/torch_predictor.pt"

    train_result = trainer.train(
        noise_data,
        n_epochs=args.n_epochs,
        early_stopping_patience=15,
        checkpoint_path=checkpoint_path,
    )

    test_start = int(len(noise_data) * 0.85)
    test_noise = noise_data[test_start:]
    predictions_list = []
    uncertainties_list = []
    actuals_list = []

    seq_len = args.seq_length
    horizon = args.forecast_horizon
    for i in range(seq_len, len(test_noise) - horizon):
        buf = test_noise[max(0, i - seq_len):i]
        pred, unc = trainer.predict(buf)
        predictions_list.append(float(pred[0]))
        uncertainties_list.append(float(unc[0]))
        actuals_list.append(float(test_noise[i]))

    preds_arr = np.array(predictions_list)
    acts_arr = np.array(actuals_list)
    unc_arr = np.array(uncertainties_list)

    metrics = noise_prediction_metrics(acts_arr, preds_arr, unc_arr)

    print(f"  Test MAE   : {metrics['mae']:.6f}")
    print(f"  Test R²    : {metrics['r2']:.4f}")
    print(f"  Coverage95 : {metrics.get('coverage_95', 'N/A')}")

    return {
        "backend": "torch",
        "train_result": train_result,
        "test_metrics": metrics,
        "predictions": predictions_list,
        "actuals": actuals_list,
        "uncertainties": uncertainties_list,
        "n_params": n_params,
    }


def train_tf_predictor(noise_data: np.ndarray, args) -> dict:
    print("  Training TensorFlow predictor...")
    trainer = NoisePredictorTFTrainer(
        seq_length=args.seq_length,
        n_features=5,
        d_model=args.hidden_dim,
        n_heads=4,
        forecast_horizon=args.forecast_horizon,
        learning_rate=args.lr,
    )

    checkpoint_dir = None
    if args.checkpoint_dir:
        checkpoint_dir = f"{args.checkpoint_dir}/tf_predictor"

    train_result = trainer.train(
        noise_data,
        n_epochs=args.n_epochs,
        checkpoint_dir=checkpoint_dir,
        early_stopping_patience=15,
    )

    test_start = int(len(noise_data) * 0.85)
    test_noise = noise_data[test_start:]
    predictions_list = []
    uncertainties_list = []
    actuals_list = []

    seq_len = args.seq_length
    horizon = args.forecast_horizon
    for i in range(seq_len, len(test_noise) - horizon):
        buf = test_noise[max(0, i - seq_len):i]
        pred, unc = trainer.predict(buf)
        predictions_list.append(float(pred[0]))
        uncertainties_list.append(float(unc[0]))
        actuals_list.append(float(test_noise[i]))

    preds_arr = np.array(predictions_list)
    acts_arr = np.array(actuals_list)
    unc_arr = np.array(uncertainties_list)

    metrics = noise_prediction_metrics(acts_arr, preds_arr, unc_arr)
    print(f"  Test MAE   : {metrics['mae']:.6f}")
    print(f"  Test R²    : {metrics['r2']:.4f}")

    return {
        "backend": "tensorflow",
        "train_result": train_result,
        "test_metrics": metrics,
        "predictions": predictions_list,
        "actuals": actuals_list,
        "uncertainties": uncertainties_list,
    }


def plot_predictor_results(results_list: list, noise_data: np.ndarray,
                            output_path: str) -> None:
    n_results = len(results_list)
    fig, axes = plt.subplots(n_results + 1, 2,
                              figsize=(14, 5 * (n_results + 1)))
    fig.suptitle("Noise Predictor Training Results",
                 fontsize=13, fontweight="bold")

    colors = {"torch": "#2980b9", "tensorflow": "#e74c3c"}

    for idx, result in enumerate(results_list):
        backend = result["backend"]
        color = colors.get(backend, "#27ae60")
        train_losses = result["train_result"]["train_losses"]
        val_losses = result["train_result"]["val_losses"]

        axes[idx, 0].semilogy(train_losses, color=color,
                               linewidth=2, label="Train")
        axes[idx, 0].semilogy(val_losses, color=color,
                               linewidth=2, linestyle="--", label="Val")
        axes[idx, 0].set_title(f"{backend.capitalize()} Training Loss")
        axes[idx, 0].set_xlabel("Epoch")
        axes[idx, 0].set_ylabel("Loss (log MSE)")
        axes[idx, 0].legend(fontsize=9)
        axes[idx, 0].grid(True, alpha=0.3)

        n_show = min(200, len(result["actuals"]))
        acts = result["actuals"][:n_show]
        preds = result["predictions"][:n_show]
        uncs = result["uncertainties"][:n_show]
        x = range(n_show)

        axes[idx, 1].plot(x, acts, color="red", linewidth=1.5,
                           alpha=0.7, label="Actual")
        axes[idx, 1].plot(x, preds, color=color, linewidth=1.5,
                           linestyle="--", label="Predicted")
        axes[idx, 1].fill_between(x,
                                    np.array(preds) - np.array(uncs),
                                    np.array(preds) + np.array(uncs),
                                    alpha=0.2, color=color)
        m = result["test_metrics"]
        axes[idx, 1].set_title(
            f"{backend.capitalize()} Predictions\n"
            f"MAE={m['mae']:.4f}, R²={m['r2']:.4f}"
        )
        axes[idx, 1].set_xlabel("Test Step")
        axes[idx, 1].set_ylabel("Noise Level γ")
        axes[idx, 1].legend(fontsize=9)
        axes[idx, 1].grid(True, alpha=0.3)

    ax_noise = axes[-1, 0]
    n_show = min(500, len(noise_data))
    ax_noise.plot(noise_data[:n_show], color="#27ae60", linewidth=1.5, alpha=0.8)
    ax_noise.set_title(f"Training Data Sample (first {n_show} steps)")
    ax_noise.set_xlabel("Step")
    ax_noise.set_ylabel("Noise Level γ")
    ax_noise.grid(True, alpha=0.3)

    ax_compare = axes[-1, 1]
    if len(results_list) >= 2:
        for result in results_list:
            backend = result["backend"]
            color = colors.get(backend, "#27ae60")
            preds = result["predictions"][:200]
            ax_compare.plot(range(len(preds)), preds, color=color,
                             linewidth=1.5, alpha=0.8, label=backend)
        ax_compare.plot(range(200), results_list[0]["actuals"][:200],
                         color="red", linewidth=1, alpha=0.6, label="Actual")
        ax_compare.set_title("Backend Comparison")
        ax_compare.set_xlabel("Test Step")
        ax_compare.legend(fontsize=9)
        ax_compare.grid(True, alpha=0.3)
    else:
        ax_compare.set_visible(False)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Predictor results saved: {output_path}")


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Noise Predictor Training")
    print(f"  Backend: {args.backend} | Regime: {args.noise_regime}")
    print(f"  Samples: {args.n_samples} | Epochs: {args.n_epochs}")

    noise_data = generate_noise_dataset(args.n_samples, args.noise_regime, args.seed)
    print(f"  Noise stats | mean={noise_data.mean():.4f} | "
          f"std={noise_data.std():.4f} | "
          f"max={noise_data.max():.4f}")

    results_list = []

    if args.backend in ("torch", "both"):
        r = train_torch_predictor(noise_data, args)
        results_list.append(r)

    if args.backend in ("tensorflow", "both"):
        r = train_tf_predictor(noise_data, args)
        results_list.append(r)

    plot_predictor_results(results_list, noise_data, args.output)

    summary = {
        "config": vars(args),
        "noise_stats": {
            "mean": float(noise_data.mean()),
            "std": float(noise_data.std()),
            "n_samples": len(noise_data),
        },
        "results": [
            {
                "backend": r["backend"],
                "test_metrics": r["test_metrics"],
                "epochs_trained": r["train_result"]["epochs_trained"],
                "best_val_loss": r["train_result"]["best_val_loss"],
            }
            for r in results_list
        ],
    }

    json_path = args.output.replace(".png", "_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {json_path}")


if __name__ == "__main__":
    main()