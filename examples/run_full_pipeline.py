import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import json
import time
from pathlib import Path

from hqnn.core.network import HyperconnectedQuantumNetwork
from hqnn.core.quantum_node import NodeConfig
from hqnn.core.quantum_edge import EdgeConfig
from hqnn.algorithms.grover import GroverSearch
from hqnn.algorithms.shor import ShorAlgorithm
from hqnn.algorithms.vqe import VQEAlgorithm
from hqnn.cage.beam_cage import QuantumBeamCage, BeamParameters
from hqnn.correction.topological import TopologicalErrorCorrector
from hqnn.correction.surface_code import SurfaceCode
from hqnn.optimizer.slime import QuantumSlimeMoldOptimizer
from hqnn.ml.noise_predictor_torch import NoisePredictor, NoisePredictorTrainer
from hqnn.ml.hybrid_trainer import HybridAdaptiveController


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def generate_realistic_noise(n_steps: int, seed: int = 0) -> np.ndarray:
    np.random.seed(seed)
    t = np.linspace(0, 20, n_steps)
    base = 0.05
    slow_drift = 0.02 * np.sin(0.2 * t)
    fast_fluctuation = 0.015 * np.sin(2.5 * t + 0.3)
    spikes = np.zeros(n_steps)
    spike_idx = np.random.choice(n_steps, size=n_steps // 20, replace=False)
    spikes[spike_idx] = np.random.uniform(0.08, 0.25, size=len(spike_idx))
    white_noise = 0.008 * np.random.randn(n_steps)
    profile = base + slow_drift + fast_fluctuation + spikes + white_noise
    return np.clip(profile, 0.001, 0.5).astype(np.float32)


def run_phase_a_algorithms(noise_level: float = 0.025) -> dict:
    results = {}

    grover = GroverSearch(n_qubits=5)
    g_result = grover.run(targets=[17, 23], noise_level=noise_level)
    results["grover"] = {
        "target_probabilities": g_result["target_probabilities"],
        "final_prob": g_result["final_target_probability"],
        "success": g_result["success"],
        "N": g_result["N"],
        "n_iterations": g_result["iteration_count"],
    }

    shor = ShorAlgorithm(n_qubits=8)
    for N_val in [15, 21, 35]:
        s_result = shor.factor(N=N_val, noise=noise_level)
        results[f"shor_{N_val}"] = {
            "factors": s_result["factors"],
            "success": s_result["success"],
            "method": s_result["method"],
        }

    vqe = VQEAlgorithm(n_qubits=4, n_layers=3, molecule="H2")
    v_result = vqe.optimize_gradient_descent(n_iterations=80, lr=0.04, noise=noise_level)
    results["vqe"] = {
        "final_energy": v_result["final_energy"],
        "exact_energy": v_result["exact_ground_energy"],
        "relative_error": v_result["relative_error"],
        "energy_history": v_result["energy_history"],
        "n_evals": v_result["n_function_evaluations"],
    }

    return results


def run_phase_b_beam_cage() -> dict:
    params = BeamParameters(
        beam_waist=1.5e-6,
        wavelength=780e-9,
        power=8e-3,
    )
    cage = QuantumBeamCage(params)
    result = cage.simulate_atomic_motion(n_atoms=800, T_atom=5e-7, n_steps=300)
    result["trap_params"] = result["trap_parameters"]
    return result


def run_phase_c_comparison(n_nodes: int = 9) -> dict:
    node_config = NodeConfig(n_qubits=2, initial_state="superposition",
                              decoherence_model="lindblad")
    net = HyperconnectedQuantumNetwork(n_nodes=n_nodes, connectivity=0.75,
                                       node_config=node_config, seed=42)
    corrector = TopologicalErrorCorrector(net, euler_tolerance=2,
                                           fidelity_threshold=0.78)
    sc = SurfaceCode(distance=3)

    error_rates = np.logspace(-3, -1, 20)
    our_logical_rates = []
    sc_logical_rates = []
    our_fidelities = []

    for p in error_rates:
        net.reset()
        fids = []
        for trial in range(30):
            net.apply_global_decoherence(p, dt=0.05)
            diag = corrector.diagnose()
            corrector.correct(diag)
            fids.append(diag["network_fidelity"])
        avg_fid = float(np.mean(fids))
        our_fidelities.append(avg_fid)
        our_logical_rates.append(max(0.0, 1.0 - avg_fid))

        sc_rate = sc.logical_error_rate(p, n_trials=200)
        sc_logical_rates.append(sc_rate)

    return {
        "error_rates": error_rates.tolist(),
        "our_logical_rates": our_logical_rates,
        "sc_logical_rates": sc_logical_rates,
        "our_fidelities": our_fidelities,
        "sc_overhead": sc.resource_overhead(),
    }


def run_phase_d_neural(n_steps: int = 200) -> dict:
    noise_profile = generate_realistic_noise(n_steps + 500, seed=1)

    model = NoisePredictor(seq_length=20, n_features=5,
                            hidden_dim=64, forecast_horizon=5)
    trainer = NoisePredictorTrainer(model, learning_rate=5e-4)
    train_result = trainer.train(noise_profile[:400], n_epochs=50,
                                  early_stopping_patience=10)

    predictions = []
    actuals = []
    uncertainties = []
    for i in range(20, min(200, len(noise_profile) - 5)):
        buf = noise_profile[max(0, i-20):i]
        pred, unc = trainer.predict(buf)
        predictions.append(float(pred[0]))
        uncertainties.append(float(unc[0]))
        actuals.append(float(noise_profile[i]))

    mae = float(np.mean(np.abs(np.array(predictions) - np.array(actuals))))
    return {
        "train_losses": train_result["train_losses"],
        "val_losses": train_result["val_losses"],
        "predictions": predictions,
        "actuals": actuals,
        "uncertainties": uncertainties,
        "mae": mae,
        "epochs_trained": train_result["epochs_trained"],
    }


def run_phase_e_unified_control(n_steps: int = 150) -> dict:
    noise_profile = generate_realistic_noise(n_steps, seed=2)

    node_config = NodeConfig(n_qubits=2, decoherence_model="lindblad")
    net = HyperconnectedQuantumNetwork(n_nodes=9, connectivity=0.75,
                                       node_config=node_config, seed=99)
    corrector = TopologicalErrorCorrector(net)
    slime = QuantumSlimeMoldOptimizer(net, decay_rate=0.08, adaptive=True)

    controller = HybridAdaptiveController(
        network=net, corrector=corrector,
        slime_optimizer=slime, predictor_backend="torch",
        seq_length=20, forecast_horizon=5,
    )

    results = controller.run_full_control_loop(noise_profile, pretrain_steps=400)
    return results


def plot_all_results(phase_a, phase_b, phase_c, phase_d, phase_e):
    fig = plt.figure(figsize=(22, 18))
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.42, wspace=0.35)
    fig.suptitle("HQNN Topology: Full System Dashboard", fontsize=16, fontweight="bold")

    ax1 = fig.add_subplot(gs[0, 0])
    grover_probs = phase_a["grover"]["target_probabilities"]
    ax1.plot(grover_probs, linewidth=2, color="#2980b9")
    ax1.axhline(y=1.0 / phase_a["grover"]["N"], color="red",
                linestyle="--", alpha=0.7, label="Initial")
    ax1.set_title(f"Grover Search (N={phase_a['grover']['N']})")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("P(target)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    vqe_history = phase_a["vqe"]["energy_history"]
    ax2.plot(vqe_history[:200], linewidth=1.5, color="#8e44ad", alpha=0.8)
    ax2.axhline(y=phase_a["vqe"]["exact_energy"], color="green",
                linestyle="--", linewidth=2, label="Exact")
    ax2.set_title(f"VQE Energy Convergence\nError={phase_a['vqe']['relative_error']:.4%}")
    ax2.set_xlabel("Function Evaluation")
    ax2.set_ylabel("Energy (Ha)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[0, 2])
    coherence = phase_b["coherence"]
    trapped = phase_b["trapped_fraction"]
    steps_b = range(len(coherence))
    ax3.plot(steps_b, coherence, color="#27ae60", linewidth=2, label="Coherence")
    ax3.plot(steps_b, trapped, color="#e67e22", linestyle="--",
             linewidth=2, label="Trapped Fraction")
    ax3.set_title("Quantum Beam Cage")
    ax3.set_xlabel("Time Step")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 0])
    er = phase_c["error_rates"]
    ax4.loglog(er, phase_c["our_logical_rates"], "b-o",
               markersize=4, linewidth=2, label="HQNN Hybrid")
    ax4.loglog(er, phase_c["sc_logical_rates"], "r-s",
               markersize=4, linewidth=2, label="Surface Code d=3")
    ax4.axhline(y=0.01, color="gray", linestyle=":", alpha=0.7)
    ax4.set_title("Error Correction Comparison")
    ax4.set_xlabel("Physical Error Rate")
    ax4.set_ylabel("Logical Error Rate")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(phase_d["train_losses"], color="#e74c3c", linewidth=2, label="Train")
    ax5.plot(phase_d["val_losses"], color="#3498db", linewidth=2, label="Validation")
    ax5.set_title(f"Neural Predictor Training\nMAE={phase_d['mae']:.5f}")
    ax5.set_xlabel("Epoch")
    ax5.set_ylabel("Loss (MSE)")
    ax5.set_yscale("log")
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(gs[1, 2])
    n_pts = min(150, len(phase_d["predictions"]))
    x_pts = range(n_pts)
    ax6.plot(x_pts, phase_d["actuals"][:n_pts], color="red",
             alpha=0.6, linewidth=1, label="Actual Noise")
    ax6.plot(x_pts, phase_d["predictions"][:n_pts], color="blue",
             linewidth=1.5, label="Predicted")
    unc = np.array(phase_d["uncertainties"][:n_pts])
    pred = np.array(phase_d["predictions"][:n_pts])
    ax6.fill_between(x_pts, pred - unc, pred + unc,
                     alpha=0.2, color="blue", label="Uncertainty")
    ax6.set_title("Noise Prediction")
    ax6.set_xlabel("Step")
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    ax7 = fig.add_subplot(gs[2, 0])
    fidelity_hist = phase_e["fidelity_history"]
    noise_hist = phase_e["noise_history"]
    ax7.plot(fidelity_hist, color="#2980b9", linewidth=2, label="Fidelity")
    ax7.plot(noise_hist, color="red", alpha=0.5, linewidth=1, label="Noise")
    for ev in phase_e["correction_events"]:
        ax7.axvline(x=ev, color="orange", alpha=0.3, linewidth=0.8)
    ax7.set_title("Unified Control: Fidelity vs Noise")
    ax7.set_xlabel("Step")
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)

    ax8 = fig.add_subplot(gs[2, 1])
    ent_hist = phase_e["entanglement_history"]
    ax8.plot(ent_hist, color="#8e44ad", linewidth=2)
    ax8.fill_between(range(len(ent_hist)), ent_hist, alpha=0.2, color="#8e44ad")
    ax8.set_title("Network Entanglement (Unified Control)")
    ax8.set_xlabel("Step")
    ax8.set_ylabel("Avg Entanglement")
    ax8.grid(True, alpha=0.3)

    ax9 = fig.add_subplot(gs[2, 2])
    euler_drift = phase_e["euler_drift_history"]
    ax9.plot(euler_drift, color="#e74c3c", linewidth=2)
    ax9.axhline(y=2, color="green", linestyle="--", label="Tolerance")
    ax9.set_title("Topological Drift (Euler)")
    ax9.set_xlabel("Step")
    ax9.set_ylabel("Euler Drift")
    ax9.legend(fontsize=8)
    ax9.grid(True, alpha=0.3)

    ax10 = fig.add_subplot(gs[3, :])
    categories = [
        "Grover\nAccuracy",
        "VQE\nConvergence",
        "Beam Cage\nCoherence",
        "Error\nCorrection",
        "Noise\nPrediction",
        "Unified\nControl",
    ]

    grover_score = min(100.0, phase_a["grover"]["final_prob"] * 100 * phase_a["grover"]["N"])
    vqe_score = max(0.0, (1.0 - phase_a["vqe"]["relative_error"]) * 100)
    cage_score = float(np.mean(phase_b["trapped_fraction"][-50:])) * 100
    correction_score = max(0.0, (1.0 - np.mean(phase_c["our_logical_rates"])) * 100)
    prediction_score = max(0.0, (1.0 - phase_d["mae"] / 0.05) * 100)
    control_score = float(np.mean(phase_e["fidelity_history"][-30:])) * 100

    scores = [grover_score, vqe_score, cage_score,
               correction_score, prediction_score, control_score]
    colors = ["#3498db", "#9b59b6", "#27ae60", "#e74c3c", "#f39c12", "#1abc9c"]

    bars = ax10.barh(categories, scores, color=colors, alpha=0.85, height=0.6)
    ax10.set_xlim(0, 115)
    ax10.set_xlabel("Performance Score (%)")
    ax10.set_title("System Performance Summary")
    for bar, score in zip(bars, scores):
        ax10.text(bar.get_width() + 1.0, bar.get_y() + bar.get_height() / 2,
                  f"{score:.1f}%", va="center", fontweight="bold", fontsize=10)
    ax10.axvline(x=85, color="green", linestyle="--", alpha=0.5, label="Target 85%")
    ax10.legend(fontsize=9)
    ax10.grid(True, alpha=0.2, axis="x")

    plt.savefig(OUTPUT_DIR / "hqnn_full_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_results(all_results: dict):
    serializable = {}
    for key, val in all_results.items():
        if isinstance(val, dict):
            serializable[key] = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in val.items()
                if not callable(v)
            }
        else:
            serializable[key] = val

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(serializable, f, indent=2, default=str)


def print_summary(phase_a, phase_b, phase_c, phase_d, phase_e):
    print("\n" + "=" * 65)
    print("  HQNN TOPOLOGY - FULL SIMULATION SUMMARY")
    print("=" * 65)

    print("\nPhase A: Quantum Algorithms")
    print(f"  Grover  | N={phase_a['grover']['N']} | "
          f"P(target)={phase_a['grover']['final_prob']:.4f} | "
          f"Success={phase_a['grover']['success']}")
    print(f"  Shor 15 | Factors={phase_a.get('shor_15', {}).get('factors', [])} | "
          f"Method={phase_a.get('shor_15', {}).get('method', 'N/A')}")
    print(f"  VQE     | Energy={phase_a['vqe']['final_energy']:.6f} Ha | "
          f"Error={phase_a['vqe']['relative_error']:.4%}")

    print("\nPhase B: Quantum Beam Cage")
    print(f"  Trapped Fraction : {np.mean(phase_b['trapped_fraction'][-50:]):.2%}")
    print(f"  Final Coherence  : {phase_b['coherence'][-1]:.4f}")
    trap_p = phase_b.get("trap_parameters", phase_b.get("trap_params", {}))
    print(f"  Trap Frequency   : {trap_p.get('trap_frequency_hz', 0):.2e} Hz")

    print("\nPhase C: Error Correction Comparison")
    print(f"  Our avg logical rate : {np.mean(phase_c['our_logical_rates']):.6f}")
    print(f"  SC  avg logical rate : {np.mean(phase_c['sc_logical_rates']):.6f}")
    print(f"  SC total qubits      : {phase_c['sc_overhead']['total_qubits']}")

    print("\nPhase D: Neural Noise Predictor (PyTorch)")
    print(f"  Epochs trained   : {phase_d['epochs_trained']}")
    print(f"  Best val loss    : {min(phase_d['val_losses']):.6f}")
    print(f"  Prediction MAE   : {phase_d['mae']:.6f}")

    print("\nPhase E: Unified Adaptive Control")
    print(f"  Avg Fidelity     : {np.mean(phase_e['fidelity_history']):.4f}")
    print(f"  Avg Entanglement : {np.mean(phase_e['entanglement_history']):.4f}")
    print(f"  Corrections      : {len(phase_e['correction_events'])}")
    print(f"  Health Score     : {phase_e.get('health_score', 0):.4f}")
    print("=" * 65)


def main():
    t0 = time.time()

    print("Phase A: Quantum Algorithms")
    phase_a = run_phase_a_algorithms(noise_level=0.02)

    print("Phase B: Quantum Beam Cage")
    phase_b = run_phase_b_beam_cage()

    print("Phase C: Error Correction Comparison")
    phase_c = run_phase_c_comparison(n_nodes=9)

    print("Phase D: Neural Noise Predictor")
    phase_d = run_phase_d_neural(n_steps=200)

    print("Phase E: Unified Adaptive Control")
    phase_e = run_phase_e_unified_control(n_steps=150)

    print_summary(phase_a, phase_b, phase_c, phase_d, phase_e)
    plot_all_results(phase_a, phase_b, phase_c, phase_d, phase_e)
    save_results({"phase_a": phase_a, "phase_b": phase_b,
                  "phase_c": phase_c, "phase_d": phase_d, "phase_e": phase_e})

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.1f}s")
    print(f"Outputs saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()