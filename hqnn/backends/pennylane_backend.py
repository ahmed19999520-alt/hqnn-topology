import pennylane as qml
import numpy as np
from typing import List, Dict, Optional

class PennyLaneQuantumCircuit:
    def __init__(self, n_qubits: int, device: str = "default.qubit"):
        self.n_qubits = n_qubits
        self.device = qml.device(device, wires=n_qubits)

    def build_hqnn_circuit(self, params: np.ndarray,
                            adjacency: np.ndarray):
        @qml.qnode(self.device)
        def circuit():
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
            for layer in range(len(params) // (self.n_qubits * 3)):
                offset = layer * self.n_qubits * 3
                for q in range(self.n_qubits):
                    qml.RX(params[offset + q*3],     wires=q)
                    qml.RY(params[offset + q*3 + 1], wires=q)
                    qml.RZ(params[offset + q*3 + 2], wires=q)
                for i in range(self.n_qubits):
                    for j in range(i+1, self.n_qubits):
                        if abs(adjacency[i,j]) > 1e-6:
                            qml.CNOT(wires=[i, j])
            return qml.state()
        return circuit

    def get_density_matrix(self, params: np.ndarray,
                            adjacency: np.ndarray) -> np.ndarray:
        circuit = self.build_hqnn_circuit(params, adjacency)
        state = circuit()
        return np.outer(state, state.conj())

    def build_grover_circuit(self, n_qubits: int,
                              target: int, n_iterations: int):
        @qml.qnode(self.device)
        def circuit():
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
            for _ in range(n_iterations):
                target_bits = format(target, f'0{n_qubits}b')
                for i, bit in enumerate(target_bits):
                    if bit == '0':
                        qml.PauliX(wires=i)
                qml.ctrl(qml.PauliZ, control=list(range(n_qubits-1)))(wires=n_qubits-1)
                for i, bit in enumerate(target_bits):
                    if bit == '0':
                        qml.PauliX(wires=i)
                for i in range(n_qubits):
                    qml.Hadamard(wires=i)
                for i in range(n_qubits):
                    qml.PauliX(wires=i)
                qml.ctrl(qml.PauliZ, control=list(range(n_qubits-1)))(wires=n_qubits-1)
                for i in range(n_qubits):
                    qml.PauliX(wires=i)
                for i in range(n_qubits):
                    qml.Hadamard(wires=i)
            return qml.probs(wires=range(n_qubits))
        return circuit

    def vqe_energy(self, params: np.ndarray,
                    hamiltonian: qml.Hamiltonian) -> float:
        @qml.qnode(self.device)
        def circuit():
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
            for layer in range(2):
                offset = layer * self.n_qubits * 3
                for q in range(self.n_qubits):
                    qml.RY(params[offset + q*3 + 1], wires=q)
                    qml.RZ(params[offset + q*3 + 2], wires=q)
                for q in range(self.n_qubits - 1):
                    qml.CNOT(wires=[q, q+1])
            return qml.expval(hamiltonian)
        return float(circuit())

    def build_h2_hamiltonian(self) -> qml.Hamiltonian:
        coeffs = [-0.8105, 0.1715, -0.2234, 0.1715, -0.2234,
                   0.1686, 0.1200, 0.1686, 0.1747, 0.1200, 0.0663]
        ops = [
            qml.Identity(0),
            qml.PauliZ(0),
            qml.PauliZ(1),
            qml.PauliZ(2),
            qml.PauliZ(3),
            qml.PauliZ(0) @ qml.PauliZ(1),
            qml.PauliZ(1) @ qml.PauliZ(2),
            qml.PauliZ(2) @ qml.PauliZ(3),
            qml.PauliZ(0) @ qml.PauliZ(3),
            qml.PauliZ(0) @ qml.PauliZ(2),
            qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliY(2) @ qml.PauliY(3),
        ]
        return qml.Hamiltonian(coeffs, ops)