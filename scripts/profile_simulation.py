import numpy as np
import time
import argparse
import json
from pathlib import Path
from hqnn.core.network import HyperconnectedQuantumNetwork
from hqnn.core.quantum_node import NodeConfig
from hqnn.correction.topological import TopologicalErrorCorrector
from hqnn.optimizer.slime import QuantumSlimeMoldOptimizer


def parse_args():
    parser = argparse.ArgumentParser(description="Profile HQNN simulation")
    parser.add_argument("--n_nodes_list", type=int, nargs="+",
                        default=[4, 6, 9, 12, 16])
    parser.add_argument("--n_steps", type=int, default=50)
    parser.add_argument("--n_runs", type=int, default=3)
    parser.add_argument("--output", type=str, default="outputs/profile_results.json")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def profile_single(n_nodes: int, n_steps: int, seed: int) -> dict:
    node_config = NodeConfig(n_qubits=2)
    net = HyperconnectedQuantumNetwork(n_nodes=n_nodes, seed=seed,
                                       node_config=node_config)
    corrector = TopologicalErrorCorrector(net)
    slime = QuantumSlimeMoldOptimizer(net, adaptive=False)

    noise = np.full(n_steps, 0.05)
    timings = {
        "decoherence": [],
        "entanglement_update": [],
        "slime_step": [],
        "correction": [],
        "snapshot": [],
    }

    for step in range(n_steps):
        t0 = time.perf_counter()
        net.apply_global_decoherence(noise[step], dt=0.01)
        timings["decoherence"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        net.update_entanglement_all()
        timings["entanglement_update"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        slime.step(noise[step])
        timings["slime_step"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        diag = corrector.diagnose()
        corrector.correct(diag)
        timings["correction"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        net.snapshot()
        timings["snapshot"].append(time.perf_counter() - t0)

    stats = {}
    for name, times in timings.items():
        arr = np.array(times)
        stats[name] = {
            "mean_ms": float(np.mean(arr) * 1000),
            "std_ms": float(np.std(arr) * 1000),
            "min_ms": float(np.min(arr) * 1000),
            "max_ms": float(np.max(arr) * 1000),
            "total_ms": float(np.sum(arr) * 1000),
        }

    total = sum(s["total_ms"] for s in stats.values())
    return {
        "n_nodes": n_nodes,
        "n_edges": len(net.edges),
        "n_steps": n_steps,
        "timings": stats,
        "total_ms": total,
        "ms_per_step": total / n_steps,
    }


def main():
    args = parse_args()
    print("HQNN Simulation Profiler")
    print(f"Steps: {args.n_steps} | Runs: {args.n_runs}")
    print("=" * 60)

    all_results = {}

    for n_nodes in args.n_nodes_list:
        run_results = []
        for run in range(args.n_runs):
            result = profile_single(n_nodes, args.n_steps,
                                     args.seed + run)
            run_results.append(result)

        ms_per_step = [r["ms_per_step"] for r in run_results]
        total_ms = [r["total_ms"] for r in run_results]

        summary = {
            "n_nodes": n_nodes,
            "n_edges": run_results[0]["n_edges"],
            "mean_ms_per_step": float(np.mean(ms_per_step)),
            "std_ms_per_step": float(np.std(ms_per_step)),
            "mean_total_ms": float(np.mean(total_ms)),
            "component_timings": run_results[0]["timings"],
        }
        all_results[str(n_nodes)] = summary

        print(f"  N={n_nodes:3d} nodes | E={summary['n_edges']:3d} edges | "
              f"{summary['mean_ms_per_step']:.2f} ± "
              f"{summary['std_ms_per_step']:.2f} ms/step")

    print("=" * 60)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Profile data saved: {args.output}")


if __name__ == "__main__":
    main()