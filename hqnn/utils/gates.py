import numpy as np
from typing import Optional
import torch
import tensorflow as tf


def pauli_x() -> np.ndarray:
    return np.array([[0, 1], [1, 0]], dtype=complex)


def pauli_y() -> np.ndarray:
    return np.array([[0, -1j], [1j, 0]], dtype=complex)


def pauli_z() -> np.ndarray:
    return np.array([[1, 0], [0, -1]], dtype=complex)


def hadamard() -> np.ndarray:
    return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def phase_gate(theta: float) -> np.ndarray:
    return np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=complex)


def rotation_x(theta: float) -> np.ndarray:
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def rotation_y(theta: float) -> np.ndarray:
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rotation_z(theta: float) -> np.ndarray:
    return np.array([[np.exp(-1j * theta / 2), 0],
                     [0, np.exp(1j * theta / 2)]], dtype=complex)


def cnot() -> np.ndarray:
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]], dtype=complex)


def toffoli() -> np.ndarray:
    T = np.eye(8, dtype=complex)
    T[6, 6] = 0
    T[7, 7] = 0
    T[6, 7] = 1
    T[7, 6] = 1
    return T


def swap() -> np.ndarray:
    return np.array([[1, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1]], dtype=complex)


def kron_n(*gates: np.ndarray) -> np.ndarray:
    result = gates[0]
    for g in gates[1:]:
        result = np.kron(result, g)
    return result


def apply_gate_to_qubit(state: np.ndarray, gate: np.ndarray,
                         qubit: int, n_qubits: int) -> np.ndarray:
    ops = []
    for i in range(n_qubits):
        if i == qubit:
            ops.append(gate)
        else:
            ops.append(np.eye(2, dtype=complex))
    full_gate = kron_n(*ops)
    return full_gate @ state


def density_from_state(state: np.ndarray) -> np.ndarray:
    return np.outer(state, state.conj())


def partial_trace(rho: np.ndarray, keep: int, n_qubits: int) -> np.ndarray:
    dim = 2 ** n_qubits
    dim_keep = 2 ** keep
    dim_trace = dim // dim_keep
    rho_reshaped = rho.reshape([dim_keep, dim_trace, dim_keep, dim_trace])
    return np.trace(rho_reshaped, axis1=1, axis2=3)


def von_neumann_entropy(rho: np.ndarray, eps: float = 1e-12) -> float:
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > eps]
    return float(-np.sum(eigenvalues * np.log2(eigenvalues)))


def fidelity(rho1: np.ndarray, rho2: np.ndarray) -> float:
    from scipy.linalg import sqrtm
    sqrt_rho1 = sqrtm(rho1)
    product = sqrt_rho1 @ rho2 @ sqrt_rho1
    sqrt_product = sqrtm(product)
    return float(np.real(np.trace(sqrt_product)) ** 2)


def concurrence(rho: np.ndarray) -> float:
    Y = pauli_y()
    YY = np.kron(Y, Y)
    rho_tilde = YY @ rho.conj() @ YY
    product = rho @ rho_tilde
    eigenvalues = np.sort(np.real(np.linalg.eigvals(product)))[::-1]
    eigenvalues = np.sqrt(np.maximum(eigenvalues, 0))
    return float(max(0, eigenvalues[0] - eigenvalues[1] - eigenvalues[2] - eigenvalues[3]))


def negativity(rho: np.ndarray) -> float:
    dim = int(np.sqrt(rho.shape[0]))
    rho_pt = rho.reshape(dim, dim, dim, dim).transpose(2, 1, 0, 3).reshape(rho.shape)
    eigenvalues = np.linalg.eigvalsh(rho_pt)
    return float(np.sum(np.abs(eigenvalues[eigenvalues < 0])))


def bell_state(kind: int = 0) -> np.ndarray:
    states = {
        0: np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2),
        1: np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2),
        2: np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2),
        3: np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2),
    }
    return states[kind]


def ghz_state(n: int) -> np.ndarray:
    dim = 2 ** n
    state = np.zeros(dim, dtype=complex)
    state[0] = 1 / np.sqrt(2)
    state[-1] = 1 / np.sqrt(2)
    return state


def w_state(n: int) -> np.ndarray:
    dim = 2 ** n
    state = np.zeros(dim, dtype=complex)
    for i in range(n):
        state[2 ** i] = 1 / np.sqrt(n)
    return state