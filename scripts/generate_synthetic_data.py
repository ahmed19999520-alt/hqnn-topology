import numpy as np
import argparse
import json
import h5py
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic quantum noise datasets")
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--n_sequences", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--format", type=str, default="hdf5",
                        choices=["hdf5", "numpy", "json"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def generate_low_noise(n: int, seed: int) -> np.ndarray:
    np.random.seed(seed)
    t = np.linspace(0, n / 100, n)
    base = 0.02 + 0.008 * np.sin(0.3 * t) + 0.005 * np.sin(1.1 * t)
    return np.clip(base + 0.003 * np.random.randn(n), 0.001, 0.15).astype(np.float32)


def generate_medium_noise(n: int, seed: int) -> np.ndarray:
    np.random.seed(seed)
    t = np.linspace(0, n / 100, n)
    base = 0.05 + 0.02 * np.sin(0.5 * t) + 0.01 * np.sin(1.7 * t + 0.5)
    burst_idx = np.random.choice(n, size=n // 30, replace=False)
    bursts = np.zeros(n)
    bursts[burst_idx] = np.random.uniform(0.05, 0.15, len(burst_idx))
    return np.clip(base + bursts + 0.01 * np.random.randn(n), 0.001, 0.35).astype(np.float32)


def generate_high_noise(n: int, seed: int) -> np.ndarray:
    np.random.seed(seed)
    t = np.linspace(0, n / 100, n)
    base = 0.15 + 0.05 * np.sin(0.2 * t) + 0.03 * np.sin(0.8 * t)
    spike_idx = np.random.choice(n, size=n // 15, replace=False)
    spikes = np.zeros(n)
    spikes[spike_idx] = np.random.uniform(0.1, 0.4, len(spike_idx))
    return np.clip(base + spikes + 0.03 * np.random.randn(n), 0.001, 0.5).astype(np.float32)


def generate_drift_noise(n: int, seed: int) -> np.ndarray:
    np.random.seed(seed)
    drift = np.linspace(0.02, 0.15, n)
    oscillation = 0.01 * np.sin(np.linspace(0, 20, n))
    white = 0.005 * np.random.randn(n)
    return np.clip(drift + oscillation + white, 0.001, 0.5).astype(np.float32)


def generate_mixed_regime(n: int, seed: int) -> np.ndarray:
    np.random.seed(seed)
    seg = n // 4
    segs = [
        generate_low_noise(seg, seed),
        generate_medium_noise(seg, seed + 1),
        generate_high_noise(seg, seed + 2),
        generate_drift_noise(n - 3 * seg, seed + 3),
    ]
    return np.concatenate(segs).astype(np.float32)


GENERATORS = {
    "low_noise": generate_low_noise,
    "medium_noise": generate_medium_noise,
    "high_noise": generate_high_noise,
    "drift_noise": generate_drift_noise,
    "mixed_regime": generate_mixed_regime,
}


def compute_statistics(data: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "median": float(np.median(data)),
        "p25": float(np.percentile(data, 25)),
        "p75": float(np.percentile(data, 75)),
        "p95": float(np.percentile(data, 95)),
    }


def save_hdf5(datasets: dict, output_dir: str) -> str:
    path = f"{output_dir}/quantum_noise_dataset.h5"
    with h5py.File(path, "w") as f:
        for name, data in datasets.items():
            f.create_dataset(name, data=data, compression="gzip",
                              compression_opts=6)
            stats = compute_statistics(data)
            for k, v in stats.items():
                f[name].attrs[k] = v
    return path


def save_numpy(datasets: dict, output_dir: str) -> list:
    paths = []
    for name, data in datasets.items():
        path = f"{output_dir}/{name}.npy"
        np.save(path, data)
        paths.append(path)
    return paths


def save_json_metadata(datasets: dict, output_dir: str) -> str:
    metadata = {}
    for name, data in datasets.items():
        metadata[name] = {
            "n_samples": len(data),
            "dtype": str(data.dtype),
            "statistics": compute_statistics(data),
        }
    path = f"{output_dir}/dataset_metadata.json"
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
    return path


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.n_samples} samples per sequence")
    print(f"Output dir: {args.output_dir}")
    print(f"Format: {args.format}")

    datasets = {}
    for i, (name, generator) in enumerate(GENERATORS.items()):
        seed = args.seed + i * 100
        data = generator(args.n_samples, seed)
        datasets[name] = data
        stats = compute_statistics(data)
        print(f"  {name}: mean={stats['mean']:.4f} std={stats['std']:.4f} "
              f"min={stats['min']:.4f} max={stats['max']:.4f}")

    if args.format == "hdf5":
        path = save_hdf5(datasets, args.output_dir)
        print(f"Saved HDF5: {path}")
    elif args.format == "numpy":
        paths = save_numpy(datasets, args.output_dir)
        for p in paths:
            print(f"Saved: {p}")

    meta_path = save_json_metadata(datasets, args.output_dir)
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()