import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Optional, Tuple
from pathlib import Path


PALETTE = {
    "blue":   "#2980b9",
    "purple": "#8e44ad",
    "green":  "#27ae60",
    "red":    "#e74c3c",
    "orange": "#e67e22",
    "teal":   "#1abc9c",
    "gray":   "#7f8c8d",
    "yellow": "#f1c40f",
}


def plot_density_matrix(rho: np.ndarray,
                         title: str = "Density Matrix",
                         save_path: Optional[str] = None) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    im1 = axes[0].imshow(np.real(rho), cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im1, ax=axes[0])
    axes[0].set_title("Real Part")
    axes[0].set_xlabel("Column")
    axes[0].set_ylabel("Row")

    im2 = axes[1].imshow(np.imag(rho), cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im2, ax=axes[1])
    axes[1].set_title("Imaginary Part")
    axes[1].set_xlabel("Column")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_bloch_sphere(bloch_vectors: List[np.ndarray],
                       labels: Optional[List[str]] = None,
                       title: str = "Bloch Sphere",
                       save_path: Optional[str] = None) -> plt.Figure:
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(xs, ys, zs, alpha=0.05, color="cyan")
    ax.plot_wireframe(xs, ys, zs, alpha=0.1, color="gray", linewidth=0.3)

    for axis, label in [([1,0,0], "x"), ([0,1,0], "y"), ([0,0,1], "z")]:
        ax.quiver(0, 0, 0, *axis, length=1.2, color="black", arrow_length_ratio=0.1)
        ax.text(*(np.array(axis) * 1.3), f"|{label}⟩", fontsize=10, ha="center")

    colors = list(PALETTE.values())
    for idx, bv in enumerate(bloch_vectors):
        color = colors[idx % len(colors)]
        label = labels[idx] if labels else f"Node {idx}"
        ax.quiver(0, 0, 0, bv[0], bv[1], bv[2],
                  color=color, linewidth=2, arrow_length_ratio=0.15)
        ax.scatter(*bv, color=color, s=80, zorder=5, label=label)

    ax.set_xlim([-1.3, 1.3])
    ax.set_ylim([-1.3, 1.3])
    ax.set_zlim([-1.3, 1.3])
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_network_graph(adjacency_matrix: np.ndarray,
                        node_fidelities: Optional[List[float]] = None,
                        edge_entanglements: Optional[Dict] = None,
                        title: str = "Quantum Network",
                        save_path: Optional[str] = None) -> plt.Figure:
    import networkx as nx
    n = adjacency_matrix.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(n):
        for j in range(i + 1, n):
            w = abs(adjacency_matrix[i, j])
            if w > 1e-6:
                G.add_edge(i, j, weight=float(w))

    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42, k=2.0)

    node_colors = PALETTE["blue"]
    if node_fidelities:
        norm = Normalize(vmin=min(node_fidelities), vmax=max(node_fidelities))
        cmap = plt.cm.RdYlGn
        node_colors = [cmap(norm(f)) for f in node_fidelities]
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Node Fidelity", shrink=0.6)

    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(edge_weights) if edge_weights else 1.0
    edge_widths = [3.0 * w / max_w for w in edge_weights]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                            node_size=600, alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9,
                             font_color="white", font_weight="bold")
    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths,
                            edge_color=PALETTE["teal"], alpha=0.7)

    if edge_entanglements:
        edge_labels = {
            (i, j): f"{e:.2f}"
            for (i, j), e in edge_entanglements.items()
            if abs(adjacency_matrix[i, j]) > 1e-6
        }
        nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax, font_size=7)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_entanglement_matrix(entanglement_matrix: np.ndarray,
                              title: str = "Pairwise Entanglement",
                              save_path: Optional[str] = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 7))
    n = entanglement_matrix.shape[0]
    im = ax.imshow(entanglement_matrix, cmap="plasma", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Entanglement")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"N{i}" for i in range(n)])
    ax.set_yticklabels([f"N{i}" for i in range(n)])

    for i in range(n):
        for j in range(n):
            val = entanglement_matrix[i, j]
            color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color)

    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_training_curves(train_losses: List[float],
                          val_losses: List[float],
                          title: str = "Training Curves",
                          save_path: Optional[str] = None) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    epochs = range(len(train_losses))
    axes[0].plot(epochs, train_losses, color=PALETTE["red"],
                 linewidth=2, label="Train Loss")
    axes[0].plot(epochs, val_losses, color=PALETTE["blue"],
                 linewidth=2, label="Val Loss")
    axes[0].set_title("Loss (Linear Scale)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(epochs, train_losses, color=PALETTE["red"],
                     linewidth=2, label="Train Loss")
    axes[1].semilogy(epochs, val_losses, color=PALETTE["blue"],
                     linewidth=2, label="Val Loss")
    axes[1].set_title("Loss (Log Scale)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE Loss (log)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_noise_prediction(actuals: List[float],
                           predictions: List[float],
                           uncertainties: Optional[List[float]] = None,
                           title: str = "Noise Prediction",
                           save_path: Optional[str] = None) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    n = len(actuals)
    x = range(n)
    a = np.array(actuals)
    p = np.array(predictions)

    axes[0].plot(x, a, color=PALETTE["red"], linewidth=1.5,
                 alpha=0.8, label="Actual Noise")
    axes[0].plot(x, p, color=PALETTE["blue"], linewidth=1.5,
                 linestyle="--", label="Predicted")
    if uncertainties is not None:
        u = np.array(uncertainties)
        axes[0].fill_between(x, p - u, p + u,
                              alpha=0.2, color=PALETTE["blue"],
                              label="1-sigma Uncertainty")
        axes[0].fill_between(x, p - 2*u, p + 2*u,
                              alpha=0.1, color=PALETTE["blue"],
                              label="2-sigma Uncertainty")
    axes[0].set_ylabel("Noise Level (gamma)")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    residuals = a - p
    axes[1].bar(x, residuals, color=[
        PALETTE["red"] if r > 0 else PALETTE["blue"]
        for r in residuals
    ], alpha=0.7, width=1.0)
    axes[1].axhline(y=0, color="black", linewidth=1)
    axes[1].set_xlabel("Time Step")
    axes[1].set_ylabel("Residual")
    axes[1].set_title(f"Prediction Residuals (MAE={np.mean(np.abs(residuals)):.5f})")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_quantum_circuit_evolution(fidelity_history: List[float],
                                    entanglement_history: List[float],
                                    entropy_history: Optional[List[float]],
                                    correction_steps: List[int],
                                    noise_history: List[float],
                                    title: str = "Quantum Circuit Evolution",
                                    save_path: Optional[str] = None) -> plt.Figure:
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    steps = range(len(fidelity_history))
    for cs in correction_steps:
        for ax in axes:
            ax.axvline(x=cs, color=PALETTE["orange"], alpha=0.25,
                       linewidth=0.7, linestyle="--")

    axes[0].plot(steps, fidelity_history, color=PALETTE["blue"],
                 linewidth=2, label="Network Fidelity")
    axes[0].plot(steps, noise_history[:len(fidelity_history)],
                 color=PALETTE["red"], linewidth=1, alpha=0.6, label="Noise Level")
    axes[0].fill_between(steps, fidelity_history,
                          alpha=0.15, color=PALETTE["blue"])
    axes[0].set_ylabel("Fidelity / Noise")
    axes[0].set_ylim([-0.05, 1.15])
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, entanglement_history, color=PALETTE["purple"],
                 linewidth=2, label="Avg Entanglement")
    axes[1].fill_between(steps, entanglement_history,
                          alpha=0.15, color=PALETTE["purple"])
    axes[1].set_ylabel("Entanglement")
    axes[1].set_ylim([-0.05, 1.05])
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    if entropy_history:
        axes[2].plot(steps, entropy_history[:len(steps)],
                     color=PALETTE["teal"], linewidth=2, label="Network Entropy")
        axes[2].fill_between(steps, entropy_history[:len(steps)],
                              alpha=0.15, color=PALETTE["teal"])
    else:
        axes[2].plot(steps, [0] * len(steps), color=PALETTE["gray"])

    axes[2].set_xlabel("Simulation Step")
    axes[2].set_ylabel("Entropy (bits)")
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    correction_patch = mpatches.Patch(color=PALETTE["orange"],
                                       alpha=0.4, label="Correction Event")
    fig.legend(handles=[correction_patch], loc="upper right", fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_error_correction_comparison(error_rates: np.ndarray,
                                      our_rates: List[float],
                                      sc_rates: List[float],
                                      our_fidelities: List[float],
                                      title: str = "Error Correction Comparison",
                                      save_path: Optional[str] = None) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    axes[0].loglog(error_rates, our_rates, "b-o", markersize=5,
                    linewidth=2, label="HQNN Hybrid")
    axes[0].loglog(error_rates, sc_rates, "r-s", markersize=5,
                    linewidth=2, label="Surface Code d=3")
    axes[0].loglog(error_rates, error_rates, "g--",
                    linewidth=1.5, alpha=0.7, label="Break-even")
    axes[0].axhline(y=0.01, color=PALETTE["gray"], linestyle=":",
                     alpha=0.8, label="Target 1%")
    axes[0].set_xlabel("Physical Error Rate")
    axes[0].set_ylabel("Logical Error Rate")
    axes[0].set_title("Logical vs Physical Error Rate")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3, which="both")

    axes[1].semilogx(error_rates, our_fidelities, "b-o",
                      markersize=5, linewidth=2, label="HQNN Fidelity")
    sc_fidelities = [1.0 - r for r in sc_rates]
    axes[1].semilogx(error_rates, sc_fidelities, "r-s",
                      markersize=5, linewidth=2, label="SC Fidelity")
    axes[1].axhline(y=0.99, color=PALETTE["green"],
                     linestyle="--", linewidth=1.5, label="Target 99%")
    axes[1].set_xlabel("Physical Error Rate")
    axes[1].set_ylabel("Logical Qubit Fidelity")
    axes[1].set_title("Fidelity vs Physical Error Rate")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_beam_cage_analysis(cage_results: Dict,
                             trap_r: np.ndarray,
                             trap_z: np.ndarray,
                             potential: np.ndarray,
                             save_path: Optional[str] = None) -> plt.Figure:
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
    fig.suptitle("Quantum Beam Cage Analysis", fontsize=14, fontweight="bold")

    ax1 = fig.add_subplot(gs[0, 0])
    r_um = trap_r * 1e6
    z_um = trap_z * 1e6
    U_K = potential / 1.38e-23
    im = ax1.contourf(r_um, z_um, U_K, levels=25, cmap="RdBu_r")
    plt.colorbar(im, ax=ax1, label="Potential (K)")
    ax1.set_title("Trapping Potential U(r,z)")
    ax1.set_xlabel("r (μm)")
    ax1.set_ylabel("z (μm)")

    ax2 = fig.add_subplot(gs[0, 1])
    steps = range(len(cage_results["coherence"]))
    ax2.plot(steps, cage_results["coherence"],
             color=PALETTE["blue"], linewidth=2, label="Coherence")
    ax2.plot(steps, cage_results["trapped_fraction"],
             color=PALETTE["green"], linewidth=2,
             linestyle="--", label="Trapped Fraction")
    ax2.set_title("Coherence vs Trapped Fraction")
    ax2.set_xlabel("Time Step")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.semilogy(steps, cage_results["decoherence_rate"],
                  color=PALETTE["red"], linewidth=2)
    ax3.set_title("Decoherence Rate (Hz)")
    ax3.set_xlabel("Time Step")
    ax3.set_ylabel("Γ (Hz)")
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 0])
    temp = cage_results["temperature"]
    ax4.plot(steps, np.array(temp) * 1e6,
             color=PALETTE["orange"], linewidth=2)
    ax4.set_title("Atomic Temperature (μK)")
    ax4.set_xlabel("Time Step")
    ax4.set_ylabel("T (μK)")
    ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(gs[1, 1])
    if cage_results["positions_history"]:
        final_pos = cage_results["positions_history"][-1]
        trap_p = cage_results.get("trap_parameters", cage_results.get("trap_params", {}))
        w0 = trap_p.get("w0_um", 1.5)
        bins = np.linspace(-5 * w0, 5 * w0, 40)
        ax5.hist(final_pos * 1e6, bins=bins, color=PALETTE["purple"],
                 alpha=0.75, edgecolor="white", linewidth=0.5)
        ax5.axvline(x=0, color="red", linewidth=2, linestyle="--")
        ax5.axvline(x=w0, color="orange", linewidth=1.5, linestyle=":", label=f"w₀={w0:.1f}μm")
        ax5.axvline(x=-w0, color="orange", linewidth=1.5, linestyle=":")
    ax5.set_title("Final Atomic Position Distribution")
    ax5.set_xlabel("Position (μm)")
    ax5.set_ylabel("Count")
    ax5.legend(fontsize=8)

    ax6 = fig.add_subplot(gs[1, 2])
    trap_p = cage_results.get("trap_parameters", cage_results.get("trap_params", {}))
    param_names = ["U₀/kB (K)", "ω_trap/2π (kHz)",
                   "z_R (μm)", "w₀ (μm)", "T_ground (nK)"]
    param_vals = [
        trap_p.get("U0_kelvin", 0),
        trap_p.get("trap_frequency_hz", 0) / 1e3,
        trap_p.get("zR_um", 0),
        trap_p.get("w0_um", 0),
        trap_p.get("ground_state_size_nm", 0) / 1e3,
    ]
    bars = ax6.barh(param_names, param_vals,
                     color=list(PALETTE.values())[:5], alpha=0.8)
    ax6.set_title("Trap Parameters")
    for bar, val in zip(bars, param_vals):
        ax6.text(bar.get_width() * 1.02, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3g}", va="center", fontsize=9)
    ax6.grid(True, alpha=0.2, axis="x")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_slime_dynamics(flow_history: List[float],
                         slime_factors: List[float],
                         mu_history: List[float],
                         noise_history: List[float],
                         save_path: Optional[str] = None) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Slime Mold Optimizer Dynamics", fontsize=13, fontweight="bold")
    steps = range(len(flow_history))

    axes[0, 0].plot(steps, flow_history, color=PALETTE["teal"], linewidth=2)
    axes[0, 0].fill_between(steps, flow_history, alpha=0.2, color=PALETTE["teal"])
    axes[0, 0].set_title("Average Quantum Flow (Q_ij)")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Mutual Information")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(steps, slime_factors, color=PALETTE["green"], linewidth=2)
    axes[0, 1].set_title("Average Slime Factor (D_ij)")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("Slime Factor")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(steps, noise_history[:len(steps)],
                     color=PALETTE["red"], linewidth=1.5, alpha=0.7, label="Noise")
    axes[1, 0].plot(steps, mu_history[:len(steps)],
                     color=PALETTE["blue"], linewidth=2, label="mu (adaptive)")
    axes[1, 0].set_title("Noise vs Adaptive Decay Rate")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    if len(flow_history) > 10:
        window = 10
        smoothed = np.convolve(flow_history, np.ones(window) / window, mode="valid")
        axes[1, 1].plot(range(len(smoothed)), smoothed,
                         color=PALETTE["purple"], linewidth=2,
                         label=f"Flow (smoothed w={window})")
        axes[1, 1].plot(steps, np.array(noise_history[:len(steps)]) * max(flow_history + [1]),
                         color=PALETTE["red"], alpha=0.4, linewidth=1, label="Scaled Noise")
        axes[1, 1].set_title("Smoothed Flow Analysis")
        axes[1, 1].set_xlabel("Step")
        axes[1, 1].legend(fontsize=9)
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def generate_full_report(all_results: Dict,
                          output_dir: str = "outputs/figures") -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if "phase_a" in all_results:
        pa = all_results["phase_a"]
        if "vqe" in pa:
            plot_training_curves(
                pa["vqe"]["energy_history"],
                pa["vqe"]["energy_history"],
                title="VQE Energy Convergence",
                save_path=f"{output_dir}/vqe_convergence.png"
            )

    if "phase_b" in all_results:
        pb = all_results["phase_b"]
        from hqnn.cage.beam_cage import QuantumBeamCage, BeamParameters
        params = BeamParameters(beam_waist=1.5e-6)
        cage = QuantumBeamCage(params)
        r = np.linspace(-5 * params.w0, 5 * params.w0, 80)
        z = np.linspace(-3 * params.zR, 3 * params.zR, 80)
        U = cage.trapping_potential(r, z)
        plot_beam_cage_analysis(pb, r, z, U,
                                 save_path=f"{output_dir}/beam_cage.png")

    if "phase_c" in all_results:
        pc = all_results["phase_c"]
        plot_error_correction_comparison(
            np.array(pc["error_rates"]),
            pc["our_logical_rates"],
            pc["sc_logical_rates"],
            pc["our_fidelities"],
            save_path=f"{output_dir}/error_correction.png"
        )

    if "phase_d" in all_results:
        pd_r = all_results["phase_d"]
        plot_noise_prediction(
            pd_r["actuals"],
            pd_r["predictions"],
            pd_r.get("uncertainties"),
            save_path=f"{output_dir}/noise_prediction.png"
        )
        plot_training_curves(
            pd_r["train_losses"],
            pd_r["val_losses"],
            title="Neural Predictor Training",
            save_path=f"{output_dir}/training_curves.png"
        )

    if "phase_e" in all_results:
        pe = all_results["phase_e"]
        plot_quantum_circuit_evolution(
            pe["fidelity_history"],
            pe["entanglement_history"],
            None,
            pe["correction_events"],
            pe["noise_history"],
            save_path=f"{output_dir}/circuit_evolution.png"
        )