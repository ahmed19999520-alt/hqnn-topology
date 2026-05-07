import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from hqnn.algorithms.vqe import VQEAlgorithm, MolecularHamiltonian


def parse_args():
    parser = argparse.ArgumentParser(description="Run VQE on HQNN")
    parser.add_argument("--n_qubits", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--molecule", type=str, default="H2")
    parser.add_argument("--n_iterations", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.04)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--optimizer", type=str, default="gradient_descent",
                        choices=["gradient_descent", "bfgs", "cobyla"])
    parser.add_argument("--n_runs", type=int, default=3,
                        help="Number of independent runs for statistics")
    parser.add_argument("--output", type=str, default="outputs/vqe_results.png")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run_vqe_multiple(n_qubits: int, n_layers: int, molecule: str,
                      n_iterations: int, lr: float, noise: float,
                      optimizer_type: str, n_runs: int, seed: int) -> dict:
    all_results = []

    for run in range(n_runs):
        np.random.seed(seed + run)
        vqe = VQEAlgorithm(n_qubits=n_qubits, n_layers=n_layers, molecule=molecule)

        if optimizer_type == "gradient_descent":
            result = vqe.optimize_gradient_descent(
                n_iterations=n_iterations, lr=lr, noise=noise)
        elif optimizer_type in ("bfgs", "cobyla"):
            result = vqe.optimize_scipy(method=optimizer_type.upper(), noise=noise)
        else:
            result = vqe.optimize_gradient_descent(
                n_iterations=n_iterations, lr=lr, noise=noise)

        result["run"] = run
        all_results.append(result)
        print(f"  Run {run+1}/{n_runs} | Energy: {result['final_energy']:.6f} Ha | "
              f"Error: {result['relative_error']:.4%}")

    energies = [r["final_energy"] for r in all_results]
    errors = [r["convergence_error"] for r in all_results]
    best_run = all_results[np.argmin(errors)]

    return {
        "all_results": all_results,
        "best_result": best_run,
        "mean_energy": float(np.mean(energies)),
        "std_energy": float(np.std(energies)),
        "mean_error": float(np.mean(errors)),
        "std_error": float(np.std(errors)),
        "exact_energy": best_run["exact_energy"],
    }


def plot_vqe_results(summary: dict, output_path: str) -> None:
    best = summary["best_result"]
    all_results = summary["all_results"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"VQE Results: {len(all_results)} runs | "
        f"Best energy = {best['final_energy']:.6f} Ha",
        fontsize=13, fontweight="bold"
    )

    for i, result in enumerate(all_results):
        hist = result["energy_history"]
        color = "#2980b9" if result is not best else "#e74c3c"
        lw = 1.0 if result is not best else 2.5
        label = f"Run {result['run']+1}" if result is not best else f"Run {result['run']+1} (best)"
        axes[0, 0].plot(hist[:300], color=color, linewidth=lw,
                        alpha=0.7, label=label)

    axes[0, 0].axhline(y=summary["exact_energy"], color="green",
                        linestyle="--", linewidth=2, label="Exact Ground State")
    axes[0, 0].set_title("Energy Convergence (all runs)")
    axes[0, 0].set_xlabel("Function Evaluation")
    axes[0, 0].set_ylabel("Energy (Hartree)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    energies = [r["final_energy"] for r in all_results]
    axes[0, 1].hist(energies, bins=max(3, len(energies)//2),
                     color="#8e44ad", alpha=0.8, edgecolor="white")
    axes[0, 1].axvline(x=summary["exact_energy"], color="green",
                        linestyle="--", linewidth=2, label="Exact")
    axes[0, 1].axvline(x=summary["mean_energy"], color="red",
                        linestyle="-", linewidth=2, label="Mean")
    axes[0, 1].set_title("Final Energy Distribution")
    axes[0, 1].set_xlabel("Final Energy (Ha)")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)

    if best.get("convergence_history"):
        conv_hist = best["convergence_history"]
        iters = [h["iteration"] for h in conv_hist]
        errors = [h["error"] for h in conv_hist]
        grad_norms = [h["grad_norm"] for h in conv_hist]

        axes[1, 0].semilogy(iters, errors, color="#e74c3c", linewidth=2)
        axes[1, 0].set_title("Convergence Error (best run)")
        axes[1, 0].set_xlabel("Iteration")
        axes[1, 0].set_ylabel("|E - E_exact| (Ha)")
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].semilogy(iters, grad_norms, color="#27ae60", linewidth=2)
        axes[1, 1].set_title("Gradient Norm (best run)")
        axes[1, 1].set_xlabel("Iteration")
        axes[1, 1].set_ylabel("‖∇E‖")
        axes[1, 1].grid(True, alpha=0.3)
    else:
        for ax in [axes[1, 0], axes[1, 1]]:
            ax.text(0.5, 0.5, "Convergence history\nnot available",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=11, color="gray")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"VQE results saved: {output_path}")


def print_vqe_summary(summary: dict, molecule: str) -> None:
    print("\n" + "=" * 60)
    print(f"  VQE Results: {molecule}")
    print("=" * 60)
    print(f"  Exact ground energy : {summary['exact_energy']:.8f} Ha")
    print(f"  Mean final energy   : {summary['mean_energy']:.8f} Ha")
    print(f"  Std final energy    : {summary['std_energy']:.8f} Ha")
    print(f"  Mean abs error      : {summary['mean_error']:.8f} Ha")
    print(f"  Best relative error : "
          f"{summary['best_result']['relative_error']:.6%}")
    n_evals = summary["best_result"].get("n_function_evaluations", "N/A")
    print(f"  Function evaluations: {n_evals}")
    print("=" * 60)


def main():
    args = parse_args()
    print(f"VQE | Molecule: {args.molecule} | n_qubits: {args.n_qubits} | "
          f"n_layers: {args.n_layers} | optimizer: {args.optimizer}")

    summary = run_vqe_multiple(
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        molecule=args.molecule,
        n_iterations=args.n_iterations,
        lr=args.lr,
        noise=args.noise,
        optimizer_type=args.optimizer,
        n_runs=args.n_runs,
        seed=args.seed,
    )

    print_vqe_summary(summary, args.molecule)
    plot_vqe_results(summary, args.output)


if __name__ == "__main__":
    main()