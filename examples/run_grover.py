import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from hqnn.algorithms.grover import GroverSearch
from hqnn.core.network import HyperconnectedQuantumNetwork
from hqnn.utils.metrics import noise_prediction_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Run Grover search on HQNN")
    parser.add_argument("--n_qubits", type=int, default=5,
                        help="Number of qubits (search space = 2^n_qubits)")
    parser.add_argument("--targets", type=int, nargs="+", default=[17],
                        help="Target indices to search for")
    parser.add_argument("--noise", type=float, default=0.02,
                        help="Noise level (0 to 1)")
    parser.add_argument("--n_nodes", type=int, default=9,
                        help="Number of network nodes for topological correction")
    parser.add_argument("--output", type=str, default="outputs/grover_results.png",
                        help="Output plot path")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run_grover_comparison(n_qubits: int, targets: list, noise_levels: list,
                           n_nodes: int, seed: int) -> dict:
    np.random.seed(seed)
    network = HyperconnectedQuantumNetwork(n_nodes=n_nodes, seed=seed)
    results = {}

    for noise in noise_levels:
        grover_no_correction = GroverSearch(n_qubits=n_qubits, network=None)
        result_no_corr = grover_no_correction.run(
            targets=targets,
            noise_level=noise,
            use_network_correction=False,
        )

        network.reset()
        grover_with_correction = GroverSearch(n_qubits=n_qubits, network=network)
        result_with_corr = grover_with_correction.run(
            targets=targets,
            noise_level=noise,
            use_network_correction=True,
        )

        results[noise] = {
            "no_correction": result_no_corr,
            "with_correction": result_with_corr,
        }

    return results


def plot_grover_comparison(results: dict, targets: list,
                            output_path: str) -> None:
    noise_levels = sorted(results.keys())
    n_noise = len(noise_levels)

    if n_noise == 1:
        noise = noise_levels[0]
        r_no = results[noise]["no_correction"]
        r_yes = results[noise]["with_correction"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f"Grover Search: N={r_no['N']} | Targets={targets} | Noise={noise:.3f}",
            fontsize=13, fontweight="bold"
        )

        axes[0].plot(r_no["target_probabilities"], "r-o",
                     markersize=4, linewidth=2, label="No Correction")
        axes[0].plot(r_yes["target_probabilities"], "b-o",
                     markersize=4, linewidth=2, label="Topological Correction")
        axes[0].axhline(y=1.0 / r_no["N"], color="gray",
                        linestyle="--", alpha=0.7, label="Initial")
        axes[0].set_title("P(target) per Iteration")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("P(target)")
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3)

        final_probs_no = r_no["final_probabilities"]
        final_probs_yes = r_yes["final_probabilities"]
        x = range(len(final_probs_no))
        axes[1].bar(x, final_probs_no, alpha=0.5, color="red",
                    label="No Correction", width=0.8)
        axes[1].bar(x, final_probs_yes, alpha=0.5, color="blue",
                    label="With Correction", width=0.8)
        for t in targets:
            axes[1].axvline(x=t, color="orange", linewidth=2,
                            linestyle="--", label=f"Target {t}")
        axes[1].set_title("Final Probability Distribution")
        axes[1].set_xlabel("State Index")
        axes[1].set_ylabel("Probability")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)
    else:
        final_probs_no = [results[n]["no_correction"]["final_target_probability"]
                          for n in noise_levels]
        final_probs_yes = [results[n]["with_correction"]["final_target_probability"]
                           for n in noise_levels]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(noise_levels, final_probs_no, "r-o", linewidth=2,
                markersize=6, label="No Correction")
        ax.plot(noise_levels, final_probs_yes, "b-o", linewidth=2,
                markersize=6, label="Topological Correction")
        ax.set_title(f"Grover Search: Final P(target) vs Noise | Targets={targets}")
        ax.set_xlabel("Noise Level (γ)")
        ax.set_ylabel("P(target)")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {output_path}")


def print_grover_summary(results: dict, targets: list) -> None:
    print("\n" + "=" * 60)
    print("  Grover Search Results")
    print("=" * 60)
    print(f"  Targets: {targets}")
    print(f"  {'Noise':<10} {'No Corr P(t)':<18} {'With Corr P(t)':<18} "
          f"{'Improvement':<12}")
    print("  " + "-" * 58)
    for noise in sorted(results.keys()):
        p_no = results[noise]["no_correction"]["final_target_probability"]
        p_yes = results[noise]["with_correction"]["final_target_probability"]
        improvement = (p_yes - p_no) / (p_no + 1e-10) * 100
        print(f"  {noise:<10.4f} {p_no:<18.4f} {p_yes:<18.4f} "
              f"{improvement:+.1f}%")
    print("=" * 60)


def main():
    args = parse_args()
    print(f"Grover Search | n_qubits={args.n_qubits} | "
          f"N={2**args.n_qubits} | targets={args.targets}")

    noise_levels = [args.noise] if args.noise else [0.0, 0.01, 0.02, 0.05, 0.10]

    results = run_grover_comparison(
        n_qubits=args.n_qubits,
        targets=args.targets,
        noise_levels=noise_levels,
        n_nodes=args.n_nodes,
        seed=args.seed,
    )

    print_grover_summary(results, args.targets)
    plot_grover_comparison(results, args.targets, args.output)


if __name__ == "__main__":
    main()