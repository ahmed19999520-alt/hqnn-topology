import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import json
import argparse
from pathlib import Path
from hqnn.core.network import HyperconnectedQuantumNetwork
from hqnn.core.quantum_node import NodeConfig
from hqnn.correction.topological import TopologicalErrorCorrector
from hqnn.correction.surface_code import SurfaceCode
from hqnn.optimizer.slime import QuantumSlimeMoldOptimizer
from hqnn.utils.metrics import correction_efficiency_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark error correction")
    parser.add_argument("--n_nodes", type=int, default=9)
    parser.add_argument("--n_error_rates", type=int, default=20)
    parser.add_argument("--p_min", type=float, default=1e-3)
    parser.add_argument("--p_max", type=float, default=1e-1)
    parser.add_argument("--n_trials_per_rate", type=int, default=50)
    parser.add_argument("--sc_distances", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--output", type=str, default="outputs/benchmark_results.png")
    parser.add_argument("--save_json", type=str, default="outputs/benchmark_data.json")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def benchmark_hqnn(n_nodes: int, error_rates: np.ndarray,
                    n_trials: int, seed: int) -> dict:
    np.random.seed(seed)
    node_config = NodeConfig(n_qubits=2, decoherence_model="lindblad")
    results = {
        "logical_rates": [],
        "fidelities": [],
        "correction_rates": [],
        "timing": [],
    }

    for p in error_rates:
        trial_fidelities = []
        trial_corrections = []

        for trial in range(n_trials):
            net = HyperconnectedQuantumNetwork(
                n_nodes=n_nodes, connectivity=0.75,
                node_config=node_config, seed=seed + trial
            )
            corrector = TopologicalErrorCorrector(net, fidelity_threshold=0.78)
            slime = QuantumSlimeMoldOptimizer(net, adaptive=False)

            t0 = time.perf_counter()
            for step in range(20):
                net.apply_global_decoherence(p, dt=0.05)
                net.update_entanglement_all()
                slime.step(p)
                diag = corrector.diagnose()
                corrector.correct(diag)
            elapsed = time.perf_counter() - t0

            final_fid = net.get_network_fidelity()
            trial_fidelities.append(final_fid)
            trial_corrections.append(corrector.total_corrections)

        avg_fid = float(np.mean(trial_fidelities))
        logical_rate = max(0.0, 1.0 - avg_fid)
        results["logical_rates"].append(logical_rate)
        results["fidelities"].append(avg_fid)
        results["correction_rates"].append(float(np.mean(trial_corrections) / 20))
        results["timing"].append(elapsed)

    return results


def benchmark_surface_code(distances: list, error_rates: np.ndarray,
                             n_trials: int) -> dict:
    all_results = {}

    for d in distances:
        sc = SurfaceCode(distance=d)
        logical_rates = []

        for p in error_rates:
            rate = sc.logical_error_rate(p, n_trials=n_trials)
            logical_rates.append(rate)

        all_results[d] = {
            "logical_rates": logical_rates,
            "overhead": sc.resource_overhead(),
        }

    return all_results


def plot_benchmark(error_rates: np.ndarray, hqnn_results: dict,
                    sc_results: dict, n_nodes: int, output_path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Error Correction Benchmark: HQNN-Topology vs Surface Codes",
                 fontsize=13, fontweight="bold")

    colors_sc = ["#e74c3c", "#e67e22", "#f1c40f"]
    axes[0, 0].loglog(error_rates, hqnn_results["logical_rates"],
                       "b-o", markersize=6, linewidth=2.5,
                       label=f"HQNN ({n_nodes} nodes)")
    for idx, (d, sc_data) in enumerate(sc_results.items()):
        axes[0, 0].loglog(error_rates, sc_data["logical_rates"],
                           f"--s", markersize=5, linewidth=2,
                           color=colors_sc[idx % len(colors_sc)],
                           label=f"Surface Code d={d}")
    axes[0, 0].loglog(error_rates, error_rates, "k:", linewidth=1,
                       alpha=0.5, label="Break-even")
    axes[0, 0].axhline(y=0.01, color="gray", linestyle=":",
                        alpha=0.7, label="1% target")
    axes[0, 0].set_xlabel("Physical Error Rate p")
    axes[0, 0].set_ylabel("Logical Error Rate p_L")
    axes[0, 0].set_title("Logical vs Physical Error Rate")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3, which="both")

    axes[0, 1].semilogx(error_rates, hqnn_results["fidelities"],
                         "b-o", markersize=6, linewidth=2.5,
                         label=f"HQNN ({n_nodes} nodes)")
    for idx, (d, sc_data) in enumerate(sc_results.items()):
        sc_fids = [1.0 - r for r in sc_data["logical_rates"]]
        axes[0, 1].semilogx(error_rates, sc_fids, "--s",
                              markersize=5, linewidth=2,
                              color=colors_sc[idx % len(colors_sc)],
                              label=f"Surface Code d={d}")
    axes[0, 1].axhline(y=0.99, color="green", linestyle="--",
                        linewidth=1.5, label="99% target")
    axes[0, 1].set_xlabel("Physical Error Rate p")
    axes[0, 1].set_ylabel("Logical Qubit Fidelity")
    axes[0, 1].set_title("Fidelity vs Error Rate")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    qubit_counts = [n_nodes] + [sc_results[d]["overhead"]["total_qubits"]
                                  for d in sc_results]
    labels = [f"HQNN\n({n_nodes})"] + [f"SC d={d}\n({sc_results[d]['overhead']['total_qubits']})"
                                          for d in sc_results]
    bar_colors = ["#2980b9"] + colors_sc[:len(sc_results)]
    bars = axes[1, 0].bar(labels, qubit_counts, color=bar_colors, alpha=0.85)
    for bar, count in zip(bars, qubit_counts):
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.3, str(count),
                         ha="center", va="bottom", fontweight="bold")
    axes[1, 0].set_title("Physical Qubit Overhead")
    axes[1, 0].set_ylabel("Total Physical Qubits")
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    improvement = [
        (sc_results[d]["logical_rates"][i] /
         max(hqnn_results["logical_rates"][i], 1e-10))
        for d in list(sc_results.keys())[:1]
        for i in range(len(error_rates))
    ]
    axes[1, 1].semilogx(error_rates, improvement, "g-o",
                         markersize=5, linewidth=2)
    axes[1, 1].axhline(y=1.0, color="gray", linestyle="--",
                        alpha=0.7, label="Break-even")
    axes[1, 1].set_xlabel("Physical Error Rate p")
    axes[1, 1].set_ylabel("Improvement Factor")
    axes[1, 1].set_title(f"HQNN Improvement over SC d={list(sc_results.keys())[0]}")
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Benchmark plot saved: {output_path}")


def print_benchmark_summary(error_rates: np.ndarray, hqnn_results: dict,
                              sc_results: dict, n_nodes: int) -> None:
    print("\n" + "=" * 70)
    print("  Error Correction Benchmark Summary")
    print("=" * 70)
    print(f"  HQNN nodes: {n_nodes}")
    for d, data in sc_results.items():
        overhead = data["overhead"]
        print(f"  SC d={d}: {overhead['total_qubits']} total qubits")
    print()

    hqnn_avg = float(np.mean(hqnn_results["logical_rates"]))
    print(f"  {'Metric':<35} {'HQNN':<15}", end="")
    for d in sc_results:
        print(f"  {'SC d='+str(d):<15}", end="")
    print()
    print("  " + "-" * 65)

    sc_avgs = {d: float(np.mean(data["logical_rates"]))
               for d, data in sc_results.items()}
    print(f"  {'Avg logical error rate':<35} {hqnn_avg:<15.6f}", end="")
    for d in sc_results:
        print(f"  {sc_avgs[d]:<15.6f}", end="")
    print()

    hqnn_fid = float(np.mean(hqnn_results["fidelities"]))
    print(f"  {'Avg fidelity':<35} {hqnn_fid:<15.4f}", end="")
    for d in sc_results:
        sc_fid = float(np.mean([1 - r for r in sc_results[d]["logical_rates"]]))
        print(f"  {sc_fid:<15.4f}", end="")
    print()
    print("=" * 70)


def main():
    args = parse_args()
    print(f"Benchmark | n_nodes={args.n_nodes} | "
          f"SC distances={args.sc_distances} | "
          f"rates={args.n_error_rates}")

    error_rates = np.logspace(
        np.log10(args.p_min),
        np.log10(args.p_max),
        args.n_error_rates
    )

    print("  Benchmarking HQNN...")
    hqnn_results = benchmark_hqnn(
        n_nodes=args.n_nodes,
        error_rates=error_rates,
        n_trials=args.n_trials_per_rate,
        seed=args.seed,
    )

    print("  Benchmarking Surface Codes...")
    sc_results = benchmark_surface_code(
        distances=args.sc_distances,
        error_rates=error_rates,
        n_trials=args.n_trials_per_rate,
    )

    print_benchmark_summary(error_rates, hqnn_results, sc_results, args.n_nodes)
    plot_benchmark(error_rates, hqnn_results, sc_results, args.n_nodes, args.output)

    if args.save_json:
        data = {
            "error_rates": error_rates.tolist(),
            "hqnn": hqnn_results,
            "surface_codes": {
                str(d): {
                    "logical_rates": v["logical_rates"],
                    "overhead": v["overhead"],
                }
                for d, v in sc_results.items()
            },
            "config": vars(args),
        }
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Benchmark data saved: {args.save_json}")


if __name__ == "__main__":
    main()