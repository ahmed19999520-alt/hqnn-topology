from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace
import numpy as np
from typing import Dict, List, Optional


class QiskitBackend:
    def __init__(self, backend_type: str = "aer_simulator",
                 noise_model: Optional[NoiseModel] = None):
        self.backend_type = backend_type
        self.noise_model = noise_model
        self.simulator = AerSimulator(noise_model=noise_model)

    def build_realistic_noise_model(self,
                                     t1: float = 100e-6,
                                     t2: float = 50e-6,
                                     gate_time_1q: float = 50e-9,
                                     gate_time_2q: float = 200e-9,
                                     readout_error: float = 0.01) -> NoiseModel:
        noise_model = NoiseModel()
        error_1q = thermal_relaxation_error(t1, t2, gate_time_1q)
        error_2q = thermal_relaxation_error(t1, t2, gate_time_2q).expand(
            thermal_relaxation_error(t1, t2, gate_time_2q))
        noise_model.add_all_qubit_quantum_error(error_1q, ['h', 'rx', 'ry', 'rz'])
        noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])
        readout = [[1 - readout_error, readout_error],
                   [readout_error, 1 - readout_error]]
        noise_model.add_all_qubit_readout_error(readout)
        return noise_model

    def hqnn_to_qiskit_circuit(self, n_qubits: int,
                                params: np.ndarray,
                                adjacency: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(n_qubits)
        for i in range(n_qubits):
            qc.h(i)
        n_layers = len(params) // (n_qubits * 3)
        for layer in range(n_layers):
            offset = layer * n_qubits * 3
            for q in range(n_qubits):
                qc.rx(params[offset + q*3],     q)
                qc.ry(params[offset + q*3 + 1], q)
                qc.rz(params[offset + q*3 + 2], q)
            for i in range(n_qubits):
                for j in range(i+1, n_qubits):
                    if abs(adjacency[i, j]) > 1e-6:
                        qc.cx(i, j)
        return qc

    def get_density_matrix_from_circuit(self,
                                         qc: QuantumCircuit) -> np.ndarray:
        qc_dm = qc.copy()
        qc_dm.save_density_matrix()
        job = self.simulator.run(
            transpile(qc_dm, self.simulator), shots=1)
        result = job.result()
        dm = result.data()['density_matrix']
        return np.array(dm.data)

    def run_grover_qiskit(self, n_qubits: int,
                           target: int,
                           n_iterations: int,
                           shots: int = 4096) -> Dict:
        n_iter = n_iterations or int(np.pi/4 * np.sqrt(2**n_qubits))
        qc = QuantumCircuit(n_qubits, n_qubits)
        for i in range(n_qubits):
            qc.h(i)
        target_bits = format(target, f'0{n_qubits}b')[::-1]
        for _ in range(n_iter):
            for i, bit in enumerate(target_bits):
                if bit == '0':
                    qc.x(i)
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
            for i, bit in enumerate(target_bits):
                if bit == '0':
                    qc.x(i)
            for i in range(n_qubits):
                qc.h(i)
            for i in range(n_qubits):
                qc.x(i)
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
            for i in range(n_qubits):
                qc.x(i)
            for i in range(n_qubits):
                qc.h(i)
        for i in range(n_qubits):
            qc.measure(i, i)
        transpiled = transpile(qc, self.simulator)
        job = self.simulator.run(transpiled, shots=shots)
        counts = job.result().get_counts()
        probs = {int(k[::-1], 2): v/shots
                 for k, v in counts.items()}
        measured = max(probs, key=probs.get)
        return {
            'counts': counts,
            'probabilities': probs,
            'measured': measured,
            'target': target,
            'success': measured == target,
            'target_probability': probs.get(target, 0.0),
        }

    def run_vqe_qiskit(self, n_qubits: int = 4,
                        n_iterations: int = 50,
                        shots: int = 8192) -> Dict:
        from scipy.optimize import minimize

        def energy_from_params(params):
            qc = QuantumCircuit(n_qubits)
            for i in range(n_qubits):
                qc.h(i)
            for layer in range(2):
                offset = layer * n_qubits * 3
                for q in range(n_qubits):
                    qc.ry(params[offset + q*3 + 1], q)
                    qc.rz(params[offset + q*3 + 2], q)
                for q in range(n_qubits - 1):
                    qc.cx(q, q+1)
            qc.save_statevector()
            job = self.simulator.run(transpile(qc, self.simulator), shots=1)
            sv = np.array(job.result().data()['statevector'])
            Z = np.array([[1,0],[0,-1]], dtype=complex)
            I = np.eye(2, dtype=complex)
            def op(ops):
                result = ops[0]
                for o in ops[1:]:
                    result = np.kron(result, o)
                return result
            H = (-0.8105 * op([I,I,I,I]) +
                  0.1715 * op([Z,I,I,I]) +
                 -0.2234 * op([I,Z,I,I]) +
                  0.1715 * op([I,I,Z,I]) +
                 -0.2234 * op([I,I,I,Z]) +
                  0.1686 * op([Z,Z,I,I]) +
                  0.1200 * op([I,Z,Z,I]))
            return float(np.real(sv.conj() @ H @ sv))

        params0 = np.random.uniform(-np.pi, np.pi, n_qubits * 3 * 2)
        result = minimize(energy_from_params, params0,
                          method='COBYLA',
                          options={'maxiter': n_iterations})
        return {
            'final_energy': float(result.fun),
            'final_params': result.x,
            'success': result.success,
            'n_evaluations': result.nfev,
        }

    def benchmark_against_ideal(self, n_qubits: int,
                                  params: np.ndarray,
                                  adjacency: np.ndarray) -> Dict:
        qc = self.hqnn_to_qiskit_circuit(n_qubits, params, adjacency)
        ideal_sim = AerSimulator()
        noisy_sim = AerSimulator(noise_model=self.noise_model)
        qc_sv = qc.copy()
        qc_sv.save_statevector()
        job_ideal = ideal_sim.run(transpile(qc_sv, ideal_sim), shots=1)
        sv_ideal = np.array(job_ideal.result().data()['statevector'])
        dm_ideal = np.outer(sv_ideal, sv_ideal.conj())
        qc_dm = qc.copy()
        qc_dm.save_density_matrix()
        job_noisy = noisy_sim.run(transpile(qc_dm, noisy_sim), shots=1)
        dm_noisy = np.array(job_noisy.result().data()['density_matrix'].data)
        from hqnn.utils.gates import fidelity
        F = fidelity(dm_ideal, dm_noisy)
        return {
            'fidelity': F,
            'ideal_purity': float(np.real(np.trace(dm_ideal @ dm_ideal))),
            'noisy_purity': float(np.real(np.trace(dm_noisy @ dm_noisy))),
            'trace_distance': float(0.5 * np.sum(np.abs(
                np.linalg.svd(dm_ideal - dm_noisy, compute_uv=False)))),
        }