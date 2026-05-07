import numpy as np
import pytest
from hqnn.cage.beam_cage import QuantumBeamCage, BeamParameters


class TestBeamParameters:
    def test_defaults(self):
        p = BeamParameters()
        assert p.w0 > 0
        assert p.wavelength > 0
        assert p.power > 0

    def test_custom(self):
        p = BeamParameters(beam_waist=3e-6, wavelength=852e-9, power=10e-3)
        assert abs(p.w0 - 3e-6) < 1e-18
        assert abs(p.wavelength - 852e-9) < 1e-18


class TestQuantumBeamCage:
    def setup_method(self):
        self.params = BeamParameters(
            beam_waist=2e-6,
            wavelength=780e-9,
            power=5e-3,
        )
        self.cage = QuantumBeamCage(self.params)

    def test_trap_depth_positive(self):
        assert self.cage.U0 > 0

    def test_trap_frequency_positive(self):
        assert self.cage.trap_frequency > 0

    def test_rayleigh_range_positive(self):
        assert self.params.zR > 0

    def test_beam_waist_at_focus(self):
        w = self.cage.beam_waist_at_z(0.0)
        assert abs(w - self.params.w0) < 1e-20

    def test_beam_waist_increases_with_z(self):
        w0 = self.cage.beam_waist_at_z(0.0)
        wz = self.cage.beam_waist_at_z(self.params.zR)
        assert wz > w0

    def test_beam_waist_at_rayleigh(self):
        wz = self.cage.beam_waist_at_z(self.params.zR)
        expected = self.params.w0 * np.sqrt(2)
        assert abs(wz - expected) < 1e-20

    def test_intensity_profile_peak_at_center(self):
        r = np.linspace(-5e-6, 5e-6, 100)
        I = self.cage.intensity_profile(r, z=0.0)
        peak_idx = np.argmax(I)
        assert abs(r[peak_idx]) < 2e-7

    def test_intensity_profile_nonnegative(self):
        r = np.linspace(-5e-6, 5e-6, 50)
        I = self.cage.intensity_profile(r, z=0.0)
        assert np.all(I >= 0)

    def test_trapping_potential_negative_at_center(self):
        r = np.array([0.0])
        z = np.array([0.0])
        U = self.cage.trapping_potential(r, z)
        assert U[0, 0] < 0

    def test_trapping_force_restoring(self):
        r_pos = np.array([1e-7])
        r_neg = np.array([-1e-7])
        F_pos = self.cage.trapping_force_radial(r_pos)
        F_neg = self.cage.trapping_force_radial(r_neg)
        assert F_pos[0] < 0
        assert F_neg[0] > 0

    def test_decoherence_suppression_positive(self):
        eta = self.cage.decoherence_suppression_factor(0.0)
        assert eta > 0

    def test_decoherence_suppression_decreases_with_r(self):
        eta_0 = self.cage.decoherence_suppression_factor(0.0)
        eta_r = self.cage.decoherence_suppression_factor(self.params.w0)
        assert eta_0 > eta_r

    def test_thermal_occupation_positive(self):
        n_bar = self.cage.thermal_occupation(T_atom=1e-6)
        assert n_bar >= 0

    def test_ground_state_size_positive(self):
        gs = self.cage.ground_state_size()
        assert gs > 0

    def test_quantum_volume_positive(self):
        V = self.cage.compute_quantum_volume()
        assert V > 0

    def test_effective_decoherence_rate_positive(self):
        rate = self.cage.effective_decoherence_rate()
        assert rate > 0

    def test_simulate_returns_required_keys(self):
        result = self.cage.simulate_atomic_motion(
            n_atoms=50, T_atom=1e-6, n_steps=20)
        for key in ["coherence", "trapped_fraction",
                    "decoherence_rate", "temperature", "trap_parameters"]:
            assert key in result

    def test_simulate_coherence_range(self):
        result = self.cage.simulate_atomic_motion(
            n_atoms=50, T_atom=1e-6, n_steps=20)
        for c in result["coherence"]:
            assert 0.0 <= c <= 1.0 + 1e-8

    def test_simulate_trapped_fraction_range(self):
        result = self.cage.simulate_atomic_motion(
            n_atoms=50, T_atom=1e-6, n_steps=20)
        for f in result["trapped_fraction"]:
            assert 0.0 <= f <= 1.0 + 1e-8

    def test_simulate_positions_history_nonempty(self):
        result = self.cage.simulate_atomic_motion(
            n_atoms=30, T_atom=1e-6, n_steps=10)
        assert len(result["positions_history"]) > 0

    def test_simulate_cold_atoms_high_trapped_fraction(self):
        result = self.cage.simulate_atomic_motion(
            n_atoms=100, T_atom=1e-8, n_steps=10)
        assert result["trapped_fraction"][0] > 0.8

    def test_bessel_beam_profile_nonnegative(self):
        r = np.linspace(0, 5e-6, 50)
        k_perp = 2e6
        I = self.cage.bessel_beam_profile(r, k_perp)
        assert np.all(I >= 0)