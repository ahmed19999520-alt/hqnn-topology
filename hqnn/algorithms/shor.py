import numpy as np
from math import gcd, isqrt
from typing import Dict, List, Optional, Tuple
from fractions import Fraction


class QuantumFourierTransform:
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.N = 2 ** n_qubits

    def apply(self, state: np.ndarray, noise: float = 0.0) -> np.ndarray:
        N = len(state)
        result = np.zeros(N, dtype=complex)
        for k in range(N):
            for j in range(N):
                angle = 2.0 * np.pi * j * k / N
                phase_noise = noise * np.random.randn() * 0.05
                result[k] += state[j] * np.exp(1j * (angle + phase_noise))
            result[k] /= np.sqrt(N)
        return result

    def apply_inverse(self, state: np.ndarray, noise: float = 0.0) -> np.ndarray:
        N = len(state)
        result = np.zeros(N, dtype=complex)
        for k in range(N):
            for j in range(N):
                angle = -2.0 * np.pi * j * k / N
                phase_noise = noise * np.random.randn() * 0.05
                result[k] += state[j] * np.exp(1j * (angle + phase_noise))
            result[k] /= np.sqrt(N)
        return result

    def matrix(self) -> np.ndarray:
        N = self.N
        omega = np.exp(2j * np.pi / N)
        F = np.array([[omega ** (j * k) for k in range(N)]
                      for j in range(N)], dtype=complex) / np.sqrt(N)
        return F


class ShorAlgorithm:
    def __init__(self, n_qubits: int = 8):
        self.n_qubits = n_qubits
        self.N_reg = 2 ** n_qubits
        self.qft = QuantumFourierTransform(n_qubits)

    def _is_prime(self, n: int) -> bool:
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, isqrt(n) + 1, 2):
            if n % i == 0:
                return False
        return True

    def _classical_period_finding(self, a: int, N: int) -> int:
        r = 1
        x = a % N
        while x != 1 and r < N:
            x = (x * a) % N
            r += 1
        return r

    def _quantum_period_finding(self, a: int, N: int,
                                 noise: float = 0.0) -> Dict:
        state = np.ones(self.N_reg, dtype=complex) / np.sqrt(self.N_reg)

        for x in range(self.N_reg):
            f_x = pow(int(a), int(x), int(N))
            phase = 2.0 * np.pi * f_x / N
            state[x] *= np.exp(1j * phase)

        qft_state = self.qft.apply(state, noise=noise)
        probs = np.abs(qft_state) ** 2
        probs /= probs.sum()

        measured_peak = int(np.argmax(probs))

        if measured_peak == 0:
            r_estimate = N
        else:
            frac = Fraction(measured_peak, self.N_reg).limit_denominator(N)
            r_estimate = frac.denominator

        classical_r = self._classical_period_finding(a, N)

        return {
            "measured_frequency": measured_peak,
            "estimated_period": r_estimate,
            "classical_period": classical_r,
            "qft_probabilities": probs,
            "period_correct": abs(r_estimate - classical_r) <= max(1, classical_r // 4),
        }

    def factor(self, N: int, noise: float = 0.02,
               max_attempts: int = 8) -> Dict:
        result = {
            "N": N,
            "factors": [],
            "attempts": [],
            "success": False,
            "method": "quantum",
        }

        if N % 2 == 0:
            result["factors"] = [2, N // 2]
            result["success"] = True
            result["method"] = "trivial_even"
            return result

        if self._is_prime(N):
            result["factors"] = [N]
            result["success"] = False
            result["method"] = "prime_input"
            return result

        for attempt in range(max_attempts):
            a = np.random.randint(2, N - 1)
            g = gcd(int(a), int(N))

            if g != 1:
                result["factors"] = [g, N // g]
                result["success"] = True
                result["method"] = "lucky_gcd"
                break

            period_result = self._quantum_period_finding(a, N, noise)
            r = period_result["classical_period"]

            attempt_data = {
                "attempt": attempt + 1,
                "a": a,
                "r": r,
                "period_result": period_result,
                "success": False,
            }

            if r % 2 == 0 and r > 0:
                x = pow(int(a), r // 2, int(N))
                f1 = gcd(int(x + 1), int(N))
                f2 = gcd(int(x - 1), int(N))

                if 1 < f1 < N:
                    attempt_data["success"] = True
                    attempt_data["factors"] = [f1, N // f1]
                    result["factors"] = [f1, N // f1]
                    result["success"] = True
                    result["attempts"].append(attempt_data)
                    break
                elif 1 < f2 < N:
                    attempt_data["success"] = True
                    attempt_data["factors"] = [f2, N // f2]
                    result["factors"] = [f2, N // f2]
                    result["success"] = True
                    result["attempts"].append(attempt_data)
                    break

            result["attempts"].append(attempt_data)

        return result