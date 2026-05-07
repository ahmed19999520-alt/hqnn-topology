import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.stats import pearsonr, spearmanr
from hqnn.utils.gates import von_neumann_entropy, fidelity, concurrence, negativity


def quantum_volume(n_qubits: int, depth: int, error_rate: float) -> float:
    effective_depth = min(depth, int(1.0 / (n_qubits * error_rate + 1e-12)))
    return float(2 ** min(n_qubits, effective_depth))


def process_fidelity(chi_ideal: np.ndarray, chi_actual: np.ndarray) -> float:
    d = chi_ideal.shape[0]
    return float(np.real(np.trace(chi_ideal.conj().T @ chi_actual)) / d)


def average_gate_fidelity(U_ideal: np.ndarray, U_noisy: np.ndarray) -> float:
    d = U_ideal.shape[0]
    diff = U_ideal - U_noisy
    return float(1.0 - np.real(np.trace(diff.conj().T @ diff)) / (d * (d + 1)))


def diamond_norm_approx(E_ideal: np.ndarray, E_noisy: np.ndarray) -> float:
    diff = E_ideal - E_noisy
    singular_values = np.linalg.svd(diff, compute_uv=False)
    return float(np.sum(singular_values))


def quantum_fisher_information(rho: np.ndarray,
                                H: np.ndarray) -> float:
    eigenvalues, eigenvectors = np.linalg.eigh(rho)
    qfi = 0.0
    d = rho.shape[0]
    for i in range(d):
        for j in range(d):
            pi = eigenvalues[i]
            pj = eigenvalues[j]
            denom = pi + pj
            if denom > 1e-12:
                hij = eigenvectors[:, i].conj() @ H @ eigenvectors[:, j]
                qfi += 2.0 * abs(hij) ** 2 * (pi - pj) ** 2 / denom
    return float(qfi)


def relative_entropy(rho: np.ndarray, sigma: np.ndarray,
                      eps: float = 1e-12) -> float:
    eigenvalues_rho = np.linalg.eigvalsh(rho)
    eigenvalues_sigma = np.linalg.eigvalsh(sigma)
    eigenvalues_rho = np.maximum(eigenvalues_rho, eps)
    eigenvalues_sigma = np.maximum(eigenvalues_sigma, eps)
    S_rho = -np.sum(eigenvalues_rho * np.log(eigenvalues_rho))
    cross = -np.sum(eigenvalues_rho * np.log(eigenvalues_sigma))
    return float(cross - S_rho)


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    diff = rho - sigma
    singular_values = np.linalg.svd(diff, compute_uv=False)
    return float(0.5 * np.sum(np.abs(singular_values)))


def bures_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    F = fidelity(rho, sigma)
    return float(np.sqrt(max(0.0, 2.0 - 2.0 * np.sqrt(F))))


def quantum_coherence_l1(rho: np.ndarray) -> float:
    d = rho.shape[0]
    coherence = 0.0
    for i in range(d):
        for j in range(d):
            if i != j:
                coherence += abs(rho[i, j])
    return float(coherence)


def quantum_coherence_relative_entropy(rho: np.ndarray,
                                        eps: float = 1e-12) -> float:
    diagonal_rho = np.diag(np.diag(rho))
    return relative_entropy(rho, diagonal_rho)


def multipartite_entanglement_measure(density_matrices: List[np.ndarray]) -> Dict:
    n = len(density_matrices)
    pairwise_concurrence = np.zeros((n, n))
    pairwise_negativity = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            if density_matrices[i].shape == (4, 4):
                c = concurrence(density_matrices[i])
                neg = negativity(density_matrices[i])
            else:
                c = 0.0
                neg = 0.0
            pairwise_concurrence[i, j] = c
            pairwise_concurrence[j, i] = c
            pairwise_negativity[i, j] = neg
            pairwise_negativity[j, i] = neg

    entropies = [von_neumann_entropy(rho) for rho in density_matrices]
    global_entanglement = float(np.mean([
        pairwise_concurrence[i, j]
        for i in range(n) for j in range(i + 1, n)
    ])) if n > 1 else 0.0

    return {
        "pairwise_concurrence": pairwise_concurrence,
        "pairwise_negativity": pairwise_negativity,
        "subsystem_entropies": entropies,
        "global_entanglement": global_entanglement,
        "mean_entropy": float(np.mean(entropies)),
    }


def network_topology_metrics(adjacency_matrix: np.ndarray) -> Dict:
    n = adjacency_matrix.shape[0]
    A_real = np.abs(adjacency_matrix)
    degree = A_real.sum(axis=1)
    mean_degree = float(np.mean(degree))
    degree_variance = float(np.var(degree))

    D = np.diag(degree)
    L = D - A_real
    eigenvalues_L = np.sort(np.linalg.eigvalsh(L))
    spectral_gap = float(eigenvalues_L[1]) if n > 1 else 0.0
    algebraic_connectivity = spectral_gap

    with np.errstate(divide="ignore", invalid="ignore"):
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degree, 1e-12)))
    L_norm = D_inv_sqrt @ L @ D_inv_sqrt
    eigenvalues_Ln = np.linalg.eigvalsh(L_norm)
    spectral_radius = float(np.max(np.abs(eigenvalues_Ln)))

    adjacency_binary = (A_real > 0).astype(float)
    triangles = float(np.trace(np.linalg.matrix_power(adjacency_binary, 3))) / 6.0
    possible_triangles = float(np.sum(degree * (degree - 1)) / 2.0)
    clustering_coefficient = (3.0 * triangles / possible_triangles
                               if possible_triangles > 0 else 0.0)

    V = n
    E = int(np.sum(A_real > 0)) // 2
    F = int(triangles)
    euler_characteristic = V - E + F

    return {
        "mean_degree": mean_degree,
        "degree_variance": degree_variance,
        "spectral_gap": spectral_gap,
        "algebraic_connectivity": algebraic_connectivity,
        "spectral_radius": spectral_radius,
        "clustering_coefficient": clustering_coefficient,
        "n_triangles": F,
        "euler_characteristic": euler_characteristic,
        "n_nodes": V,
        "n_edges": E,
    }


def noise_prediction_metrics(actuals: np.ndarray,
                              predictions: np.ndarray,
                              uncertainties: Optional[np.ndarray] = None) -> Dict:
    mae = float(np.mean(np.abs(actuals - predictions)))
    mse = float(np.mean((actuals - predictions) ** 2))
    rmse = float(np.sqrt(mse))
    mape = float(np.mean(np.abs((actuals - predictions) / (np.abs(actuals) + 1e-10))) * 100)

    ss_res = float(np.sum((actuals - predictions) ** 2))
    ss_tot = float(np.sum((actuals - np.mean(actuals)) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)

    pearson_r, pearson_p = pearsonr(actuals, predictions)
    spearman_r, spearman_p = spearmanr(actuals, predictions)

    result = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mape": mape,
        "r2": float(r2),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
    }

    if uncertainties is not None:
        coverage_90 = float(np.mean(
            np.abs(actuals - predictions) <= 1.645 * uncertainties
        ))
        coverage_95 = float(np.mean(
            np.abs(actuals - predictions) <= 1.96 * uncertainties
        ))
        sharpness = float(np.mean(uncertainties))
        result.update({
            "coverage_90": coverage_90,
            "coverage_95": coverage_95,
            "sharpness": sharpness,
        })

    return result


def correction_efficiency_metrics(correction_log: List[Dict],
                                   total_steps: int) -> Dict:
    if not correction_log:
        return {
            "correction_rate": 0.0,
            "mean_correction_interval": float(total_steps),
            "correction_overhead": 0.0,
        }

    n_corrections = len(correction_log)
    correction_rate = n_corrections / max(1, total_steps)

    steps = [entry.get("step", i) for i, entry in enumerate(correction_log)]
    intervals = np.diff(steps) if len(steps) > 1 else np.array([total_steps])
    mean_interval = float(np.mean(intervals))
    std_interval = float(np.std(intervals))

    error_types = [entry.get("error_type") for entry in correction_log]
    type_counts = {}
    for et in error_types:
        if et:
            type_counts[et] = type_counts.get(et, 0) + 1

    return {
        "total_corrections": n_corrections,
        "correction_rate": correction_rate,
        "mean_correction_interval": mean_interval,
        "std_correction_interval": std_interval,
        "error_type_distribution": type_counts,
        "correction_overhead": correction_rate * 0.05,
    }