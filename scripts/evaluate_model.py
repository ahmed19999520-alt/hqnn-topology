import numpy as np
import torch
import argparse
import json
import h5py
from pathlib import Path
from hqnn.ml.noise_predictor_torch import NoisePredictor, NoisePredictorTrainer
from hqnn.utils.metrics import noise_prediction_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained noise predictor")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, required=True,
                        help="Path to HDF5 or numpy file")
    parser.add_argument("--dataset_name", type=str, default="mixed_regime")
    parser.add_argument("--seq_length", type=int, default=20)
    parser.add_argument("--forecast_horizon", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--output", type=str, default="outputs/eval_results.json")
    return parser.parse_args()


def load_data(data_path: str, dataset_name: str) -> np.ndarray:
    if data_path.endswith(".h5") or data_path.endswith(".hdf5"):
        with h5py.File(data_path, "r") as f:
            if dataset_name in f:
                return f[dataset_name][:]
            first_key = list(f.keys())[0]
            print(f"Dataset '{dataset_name}' not found, using '{first_key}'")
            return f[first_key][:]
    elif data_path.endswith(".npy"):
        return np.load(data_path)
    else:
        raise ValueError(f"Unsupported format: {data_path}")


def evaluate_checkpoint(checkpoint_path: str, noise_data: np.ndarray,
                         seq_length: int, forecast_horizon: int,
                         hidden_dim: int) -> dict:
    model = NoisePredictor(
        seq_length=seq_length,
        n_features=5,
        hidden_dim=hidden_dim,
        forecast_horizon=forecast_horizon,
    )
    trainer = NoisePredictorTrainer(model)

    ckpt = torch.load(checkpoint_path, map_location=trainer.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test_start = int(len(noise_data) * 0.8)
    test_data = noise_data[test_start:]

    predictions = []
    uncertainties = []
    actuals = []

    for i in range(seq_length, len(test_data) - forecast_horizon):
        buf = test_data[max(0, i - seq_length):i]
        pred, unc = trainer.predict(buf)
        predictions.append(float(pred[0]))
        uncertainties.append(float(unc[0]))
        actuals.append(float(test_data[i]))

    preds_arr = np.array(predictions)
    acts_arr = np.array(actuals)
    unc_arr = np.array(uncertainties)

    metrics = noise_prediction_metrics(acts_arr, preds_arr, unc_arr)

    return {
        "checkpoint": checkpoint_path,
        "n_test_samples": len(actuals),
        "metrics": metrics,
        "predictions_sample": predictions[:20],
        "actuals_sample": actuals[:20],
        "uncertainties_sample": uncertainties[:20],
        "checkpoint_info": {
            "epoch": ckpt.get("epoch", "unknown"),
            "val_loss": ckpt.get("val_loss", "unknown"),
        },
    }


def main():
    args = parse_args()

    print(f"Loading data from: {args.data}")
    noise_data = load_data(args.data, args.dataset_name)
    print(f"Data loaded: {len(noise_data)} samples")

    print(f"Evaluating checkpoint: {args.checkpoint}")
    results = evaluate_checkpoint(
        args.checkpoint, noise_data,
        args.seq_length, args.forecast_horizon, args.hidden_dim
    )

    print("\n" + "=" * 50)
    print("  Evaluation Results")
    print("=" * 50)
    m = results["metrics"]
    print(f"  Test samples : {results['n_test_samples']}")
    print(f"  MAE          : {m['mae']:.6f}")
    print(f"  RMSE         : {m['rmse']:.6f}")
    print(f"  R²           : {m['r2']:.4f}")
    print(f"  MAPE         : {m['mape']:.2f}%")
    if "coverage_95" in m:
        print(f"  Coverage 95% : {m['coverage_95']:.4f}")
    print(f"  Checkpoint   : epoch={results['checkpoint_info']['epoch']}, "
          f"val_loss={results['checkpoint_info']['val_loss']}")
    print("=" * 50)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {args.output}")


if __name__ == "__main__":
    main()