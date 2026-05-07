import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from hqnn.algorithms.shor import ShorAlgorithm, QuantumFourierTransform


def parse_args():
    parser = argparse.ArgumentParser(description="Run Shor factoring on HQNN")
    parser.add_argument("--numbers", type=int, nargs="+",
                        default=[15, 21, 35, 77],
                        help="Numbers to factor")
    parser.add_argument("--noise", type=float, default=0.02)
    parser.add_argument("--n_qubits", type=int, default=8)
    parser.add_argument("--max_attempts", type=int, default=8)
    parser.add_argument("--output", type=str, default="outputs/shor_results.png")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run_shor_batch(numbers: list, noise: float, n_qubits: int,
                    max_attempts: int, seed: int) -> dict:
    np.random.seed(seed)
    shor = ShorAlgorithm(n_qubits=n_qubits)
    results = {}

    for N in numbers:
        print(f"  Factoring {N}...")
        result = shor.factor(N=N, noise=noise, max_attempts=max_attempts)
        results[N] = result

        status = "SUCCESS" if result["success"] else "FAILED"
        factors = result["factors"] if result["factors"] else ["?"]
        print(f"    {status} | Method: {result['method']} | Factors: {factors}")

    return results


def demonstrate_qft(n_qubits: int = 4, output_dir: str = "outputs") -> None:
    qft = QuantumFourierTransform(n_qubits=n_qubits)
    N = 2 ** n_qubits

    states_to_test = {
        "Computational |0⟩": np.eye(N)[0],
        "Computational |1⟩": np.eye(N)[1],
        "Uniform superposition": np.ones(N) / np.sqrt(N),
        "GHZ-like": np.zeros(N, dtype=complex),
    }
    states_to_test["GHZ-like"][0] = 1 / np.sqrt(2)
    states_to_test["GHZ-like"][-1] = 1 / np.sqrt(2)

    fig, axes = plt.subplots(len(states_to_test), 2,
                              figsize=(14, 4 * len(states_to_test)))
    fig.suptitle(f"Quantum Fourier Transform (n={n_qubits})",
                 fontsize=13, fontweight="bold")

    for idx, (name, state) in enumerate(states_to_test.items()):
        state = state.astype(complex)
        state /= np.linalg.norm(state)
        transformed = qft.apply(state, noise=0.0)

        probs_in = np.abs(state) ** 2
        probs_out = np.abs(transformed) ** 2

        axes[idx, 0].bar(range(N), probs_in, color="#2980b9", alpha=0.8)
        axes[idx, 0].set_title(f"Input: {name}")
        axes[idx, 0].set_xlabel("State |k⟩")
        axes[idx, 0].set_ylabel("Probability")
        axes[idx, 0].grid(True, alpha=0.3)

        axes[idx, 1].bar(range(N), probs_out, color="#8e44ad", alpha=0.8)
        axes[idx, 1].set_title(f"QFT Output")
        axes[idx, 1].set_xlabel("Frequency k")
        axes[idx, 1].set_ylabel("Probability")
        axes[idx, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_path = f"{output_dir}/qft_demonstration.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"QFT demonstration saved: {save_path}")


def plot_shor_results(results: dict, output_path: str) -> None:
    numbers = list(results.keys())
    n = len(numbers)

    fig, axes = plt.subplots(1, min(n, 4), figsize=(5 * min(n, 4), 5))
    if n == 1:
        axes = [axes]
    fig.suptitle("Shor's Algorithm: QFT Probability Distributions",
                 fontsize=13, fontweight="bold")

    for idx, N_val in enumerate(numbers[:4]):
        ax = axes[idx]
        result = results[N_val]

        if result["attempts"]:
            for attempt in result["attempts"]:
                pr = attempt.get("period_result", {})
                qft_probs = pr.get("qft_probabilities")
                if qft_probs is not None and len(qft_probs) > 0:
                    ax.bar(range(len(qft_probs)), qft_probs,
                           alpha=0.7, color="#27ae60")
                    break
            else:
                ax.text(0.5, 0.5, "No QFT data", transform=ax.transAxes,
                        ha="center", va="center")

        status = "SUCCESS" if result["success"] else "FAILED"
        factors = result["factors"] if result["factors"] else ["?"]
        ax.set_title(f"N={N_val} | {status}\nFactors: {factors}")
        ax.set_xlabel("Measured Frequency")
        ax.set_ylabel("Probability")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Shor results saved: {output_path}")


def print_shor_summary(results: dict) -> None:
    print("\n" + "=" * 60)
    print("  Shor's Factoring Algorithm Results")
    print("=" * 60)
    print(f"  {'N':<8} {'Factors':<20} {'Method':<20} {'Attempts':<10}")
    print("  " + "-" * 58)
    for N_val, result in results.items():
        factors = str(result["factors"]) if result["factors"] else "FAILED"
        method = result["method"]
        n_attempts = len(result["attempts"])
        print(f"  {N_val:<8} {factors:<20} {method:<20} {n_attempts:<10}")
    success_rate = sum(1 for r in results.values() if r["success"]) / len(results)
    print(f"\n  Success rate: {success_rate:.1%}")
    print("=" * 60)


def main():
    args = parse_args()
    print(f"Shor Algorithm | Numbers: {args.numbers} | Noise: {args.noise}")

    results = run_shor_batch(
        numbers=args.numbers,
        noise=args.noise,
        n_qubits=args.n_qubits,
        max_attempts=args.max_attempts,
        seed=args.seed,
    )

    print_shor_summary(results)
    plot_shor_results(results, args.output)
    demonstrate_qft(n_qubits=4, output_dir=str(Path(args.output).parent))


if __name__ == "__main__":
    main()