import numpy as np
from typing import Dict, List, Tuple, Optional


class SurfaceCodeQubit:
    def __init__(self, qubit_type: str, row: int, col: int):
        self.qubit_type = qubit_type
        self.row = row
        self.col = col
        self.state: int = 0
        self.error_x: bool = False
        self.error_z: bool = False


class SurfaceCode:
    def __init__(self, distance: int = 3):
        self.d = distance
        self.n_data = distance ** 2
        self.n_ancilla = 2 * (distance ** 2 - 1)
        self.threshold_error_rate = 0.01
        self._build_layout()

    def _build_layout(self) -> None:
        self.data_qubits = [
            SurfaceCodeQubit("data", r, c)
            for r in range(self.d)
            for c in range(self.d)
        ]
        self.x_stabilizers = []
        self.z_stabilizers = []

        for r in range(self.d - 1):
            for c in range(self.d):
                if (r + c) % 2 == 0:
                    self.x_stabilizers.append((r, c, r + 1, c))
                else:
                    self.z_stabilizers.append((r, c, r + 1, c))

        for r in range(self.d):
            for c in range(self.d - 1):
                if (r + c) % 2 == 1:
                    self.x_stabilizers.append((r, c, r, c + 1))
                else:
                    self.z_stabilizers.append((r, c, r, c + 1))

    def inject_errors(self, error_rate: float) -> None:
        for qubit in self.data_qubits:
            if np.random.random() < error_rate:
                qubit.error_x = True
            if np.random.random() < error_rate:
                qubit.error_z = True

    def measure_syndrome(self) -> Tuple[np.ndarray, np.ndarray]:
        x_syndrome = np.zeros(len(self.x_stabilizers), dtype=int)
        z_syndrome = np.zeros(len(self.z_stabilizers), dtype=int)

        def get_qubit(r: int, c: int) -> Optional[SurfaceCodeQubit]:
            if 0 <= r < self.d and 0 <= c < self.d:
                return self.data_qubits[r * self.d + c]
            return None

        for idx, (r1, c1, r2, c2) in enumerate(self.x_stabilizers):
            q1 = get_qubit(r1, c1)
            q2 = get_qubit(r2, c2)
            syndrome = 0
            if q1 and q1.error_z:
                syndrome ^= 1
            if q2 and q2.error_z:
                syndrome ^= 1
            x_syndrome[idx] = syndrome

        for idx, (r1, c1, r2, c2) in enumerate(self.z_stabilizers):
            q1 = get_qubit(r1, c1)
            q2 = get_qubit(r2, c2)
            syndrome = 0
            if q1 and q1.error_x:
                syndrome ^= 1
            if q2 and q2.error_x:
                syndrome ^= 1
            z_syndrome[idx] = syndrome

        return x_syndrome, z_syndrome

    def minimum_weight_matching(self, syndrome: np.ndarray) -> List[int]:
        defects = list(np.where(syndrome != 0)[0])
        corrections = []
        used = set()
        for i in defects:
            if i in used:
                continue
            best_j = None
            best_dist = float("inf")
            for j in defects:
                if j != i and j not in used:
                    dist = abs(i - j)
                    if dist < best_dist:
                        best_dist = dist
                        best_j = j
            if best_j is not None:
                corrections.extend([i, best_j])
                used.add(i)
                used.add(best_j)
        return corrections

    def decode_and_correct(self) -> Dict:
        x_syn, z_syn = self.measure_syndrome()
        x_corrections = self.minimum_weight_matching(x_syn)
        z_corrections = self.minimum_weight_matching(z_syn)
        n_x_errors = sum(1 for q in self.data_qubits if q.error_x)
        n_z_errors = sum(1 for q in self.data_qubits if q.error_z)
        logical_failure = (n_x_errors + n_z_errors) >= (self.d + 1) // 2

        for qubit in self.data_qubits:
            qubit.error_x = False
            qubit.error_z = False

        return {
            "x_syndrome": x_syn.tolist(),
            "z_syndrome": z_syn.tolist(),
            "x_corrections": x_corrections,
            "z_corrections": z_corrections,
            "logical_failure": logical_failure,
            "n_physical_errors": n_x_errors + n_z_errors,
        }

    def logical_error_rate(self, physical_rate: float,
                            n_trials: int = 5000) -> float:
        failures = 0
        for _ in range(n_trials):
            for qubit in self.data_qubits:
                qubit.error_x = np.random.random() < physical_rate
                qubit.error_z = np.random.random() < physical_rate
            result = self.decode_and_correct()
            if result["logical_failure"]:
                failures += 1
        return failures / n_trials

    def resource_overhead(self) -> Dict:
        return {
            "data_qubits": self.n_data,
            "ancilla_qubits": self.n_ancilla,
            "total_qubits": self.n_data + self.n_ancilla,
            "code_distance": self.d,
            "error_threshold": self.threshold_error_rate,
            "logical_qubits": 1,
        }

    def simulate_circuit_cycles(self,
                                  physical_error_rate: float,
                                  n_cycles: int = 20) -> Dict:
        results = {
            "cycle_failures": [],
            "syndrome_weights": [],
            "cumulative_logical_errors": 0,
        }
        for _ in range(n_cycles):
            self.inject_errors(physical_error_rate)
            decode_result = self.decode_and_correct()
            results["cycle_failures"].append(decode_result["logical_failure"])
            syndrome_weight = (sum(decode_result["x_syndrome"]) +
                               sum(decode_result["z_syndrome"]))
            results["syndrome_weights"].append(syndrome_weight)
            if decode_result["logical_failure"]:
                results["cumulative_logical_errors"] += 1

        results["logical_error_rate"] = (results["cumulative_logical_errors"] / n_cycles)
        return results