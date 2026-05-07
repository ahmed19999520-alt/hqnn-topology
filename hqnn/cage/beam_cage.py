import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.special import j0 as bessel_j0


class BeamParameters:
    def __init__(self,
                 beam_waist: float = 2e-6,
                 wavelength: float = 780e-9,
                 power: float = 5e-3,
                 detuning: float = -2e9 * 2 * np.pi,
                 atomic_linewidth: float = 2 * np.pi * 6e6,
                 atomic_resonance: float = 2 * np.pi * 3.84e14,
                 atomic_mass: float = 1.44e-25):
        self.w0 = beam_waist
        self.wavelength = wavelength
        self.power = power
        self.detuning = detuning
        self.Gamma = atomic_linewidth
        self.omega_0 = atomic_resonance
        self.mass = atomic_mass
        self.k = 2 * np.pi / wavelength
        self.omega = 3e8 * self.k
        self.zR = np.pi * beam_waist ** 2 / wavelength


class QuantumBeamCage:
    def __init__(self, params: Optional[BeamParameters] = None):
        self.p = params or BeamParameters()
        self.kB = 1.380649e-23
        self.hbar = 1.054571817e-34
        self.I_peak = 2 * self.p.power / (np.pi * self.p.w0 ** 2)
        self.U0 = self._compute_trap_depth()
        self.trap_frequency = self._compute_trap_frequency()

    def _compute_trap_depth(self) -> float:
        alpha = (3 * (3e8) ** 2 * self.p.Gamma /
                 (2 * self.p.omega_0 ** 3 * abs(self.p.detuning)))
        return abs(alpha * self.I_peak / (2 * 8.854187817e-12 * 3e8))

    def _compute_trap_frequency(self) -> float:
        return np.sqrt(4 * self.U0 / (self.p.mass * self.p.w0 ** 2))

    def beam_waist_at_z(self, z: float) -> float:
        return self.p.w0 * np.sqrt(1.0 + (z / self.p.zR) ** 2)

    def intensity_profile(self, r: np.ndarray, z: float) -> np.ndarray:
        wz = self.beam_waist_at_z(z)
        return self.I_peak * (self.p.w0 / wz) ** 2 * np.exp(-2 * r ** 2 / wz ** 2)

    def bessel_beam_profile(self, r: np.ndarray, k_perp: float) -> np.ndarray:
        return self.I_peak * bessel_j0(k_perp * r) ** 2

    def trapping_potential(self, r: np.ndarray, z: np.ndarray) -> np.ndarray:
        R, Z = np.meshgrid(r, z)
        wz = self.p.w0 * np.sqrt(1.0 + (Z / self.p.zR) ** 2)
        return -self.U0 * (self.p.w0 / wz) ** 2 * np.exp(-2 * R ** 2 / wz ** 2)

    def trapping_force_radial(self, r: np.ndarray, z: float = 0.0) -> np.ndarray:
        wz = self.beam_waist_at_z(z)
        F_scale = -4 * self.U0 / wz ** 2
        return F_scale * r * np.exp(-2 * r ** 2 / wz ** 2)

    def decoherence_suppression_factor(self, r: float,
                                        T_environment: float = 300.0) -> float:
        kT = self.kB * T_environment
        wz = self.beam_waist_at_z(0.0)
        U_r = self.U0 * np.exp(-2 * r ** 2 / wz ** 2)
        return U_r / kT

    def thermal_occupation(self, T_atom: float) -> float:
        kT = self.kB * T_atom
        omega = self.trap_frequency
        return 1.0 / (np.exp(self.hbar * omega / kT) - 1.0 + 1e-30)

    def ground_state_size(self) -> float:
        return np.sqrt(self.hbar / (2 * self.p.mass * self.trap_frequency))

    def simulate_atomic_motion(self,
                                n_atoms: int = 500,
                                T_atom: float = 1e-6,
                                n_steps: int = 500,
                                dt: float = 1e-6) -> Dict:
        kT_atom = self.kB * T_atom
        sigma_r = min(
            self.p.w0 * np.sqrt(kT_atom / (2 * max(self.U0, 1e-30))),
            self.p.w0 * 3.0
        )
        sigma_v = np.sqrt(kT_atom / self.p.mass)

        positions = np.random.randn(n_atoms) * sigma_r
        velocities = np.random.randn(n_atoms) * sigma_v

        results = {
            "positions_history": [],
            "coherence": [],
            "trapped_fraction": [],
            "decoherence_rate": [],
            "temperature": [],
            "trap_parameters": {
                "U0_kelvin": self.U0 / self.kB,
                "trap_frequency_hz": self.trap_frequency / (2 * np.pi),
                "zR_um": self.p.zR * 1e6,
                "w0_um": self.p.w0 * 1e6,
                "ground_state_size_nm": self.ground_state_size() * 1e9,
            },
        }

        for step in range(n_steps):
            forces = self.trapping_force_radial(positions)
            velocities += forces / self.p.mass * dt
            positions += velocities * dt

            photon_recoil = np.sqrt(2 * (self.hbar * self.p.k) ** 2 *
                                     self.p.Gamma / self.p.mass * dt)
            velocities += photon_recoil * np.random.randn(n_atoms)

            trapped = np.abs(positions) < 5.0 * self.p.w0
            trapped_fraction = float(np.mean(trapped))

            if trapped.any():
                v_trapped = velocities[trapped]
                T_current = float(self.p.mass * np.mean(v_trapped ** 2) / self.kB)
            else:
                T_current = float(T_atom)

            avg_r = float(np.mean(np.abs(positions[trapped]))) if trapped.any() else self.p.w0
            eta = self.decoherence_suppression_factor(avg_r)
            gamma_dec = 1.0 / (1.0 + eta) * 1e4

            n_bar = self.thermal_occupation(max(T_current, 1e-9))
            coherence = trapped_fraction * np.exp(-gamma_dec * dt * (step + 1)) / (1.0 + n_bar * 0.1)
            coherence = float(np.clip(coherence, 0.0, 1.0))

            if step % 5 == 0:
                results["positions_history"].append(positions.copy())

            results["coherence"].append(coherence)
            results["trapped_fraction"].append(trapped_fraction)
            results["decoherence_rate"].append(gamma_dec)
            results["temperature"].append(T_current)

        return results

    def compute_quantum_volume(self) -> float:
        V_trap = (np.pi ** 1.5) * self.p.w0 ** 2 * self.p.zR / 8.0
        return V_trap

    def effective_decoherence_rate(self, T_environment: float = 300.0) -> float:
        kT = self.kB * T_environment
        if self.U0 < 1e-30:
            return 1e6
        return float(1e4 * np.exp(-self.U0 / kT))