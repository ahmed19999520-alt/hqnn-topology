import numpy as np
from typing import Dict, List, Optional
from hqnn.core.network import HyperconnectedQuantumNetwork
from hqnn.utils.gates import von_neumann_entropy


class QuantumSlimeMoldOptimizer:
    def __init__(self,
                 network: HyperconnectedQuantumNetwork,
                 decay_rate: float = 0.1,
                 dt: float = 0.01,
                 topology_protection: float = 0.5,
                 adaptive: bool = True):
        self.network = network
        self.mu = decay_rate
        self.dt = dt
        self.topology_protection = topology_protection
        self.adaptive = adaptive
        self.flow_history: List[float] = []
        self.noise_history: List[float] = []
        self.adaptation_log: List[Dict] = []

    def _quantum_mutual_information(self, i: int, j: int) -> float:
        rho_i = self.network.nodes[i].density_matrix
        rho_j = self.network.nodes[j].density_matrix
        S_i = von_neumann_entropy(rho_i)
        S_j = von_neumann_entropy(rho_j)
        rho_mix = 0.5 * (rho_i + rho_j)
        S_ij = von_neumann_entropy(rho_mix)
        return max(0.0, S_i + S_j - 2.0 * S_ij)

    def _compute_effective_noise(self, noise_level: float) -> float:
        if self.adaptive and len(self.noise_history) >= 5:
            recent = np.array(self.noise_history[-10:])
            trend = np.polyfit(range(len(recent)), recent, 1)[0]
            effective = noise_level * (1.0 + 0.5 * np.sign(trend))
            return float(np.clip(effective, 0.001, 1.0))
        return noise_level

    def step(self, noise_level: float) -> Dict:
        effective_noise = self._compute_effective_noise(noise_level)
        total_flow = 0.0
        edge_updates = []

        for (i, j), edge in self.network.edges.items():
            Q_ij = self._quantum_mutual_information(i, j)
            edge.quantum_flow = Q_ij

            local_noise = effective_noise * (1.0 + 0.2 * np.random.exponential(1.0))
            local_noise = min(local_noise, 0.99)

            growth = (Q_ij - self.mu) * edge.slime_factor
            noise_suppression = 1.0 - local_noise
            topo_term = edge.entanglement * self.topology_protection

            d_slime = (growth * noise_suppression + topo_term) * self.dt
            new_slime = max(0.01, edge.slime_factor + d_slime)
            edge.slime_factor = min(10.0, new_slime)

            phase = np.angle(edge.weight)
            new_mag = abs(edge.weight) * (1.0 + 0.05 * d_slime)
            new_mag = np.clip(new_mag, 1e-6, 5.0)
            edge.weight = new_mag * np.exp(1j * phase)

            edge.entanglement = float(np.clip(
                edge.entanglement + 0.01 * (Q_ij - edge.entanglement),
                0.0, 1.0
            ))

            total_flow += Q_ij
            edge_updates.append({
                "edge": (i, j),
                "flow": Q_ij,
                "slime_factor": edge.slime_factor,
            })

        n_edges = max(1, len(self.network.edges))
        avg_flow = total_flow / n_edges
        self.flow_history.append(avg_flow)
        self.noise_history.append(noise_level)

        if self.adaptive:
            self._adapt_decay_rate(avg_flow, noise_level)

        return {
            "average_flow": avg_flow,
            "total_flow": total_flow,
            "effective_noise": effective_noise,
            "mu": self.mu,
            "edge_updates": edge_updates,
        }

    def _adapt_decay_rate(self, avg_flow: float, noise_level: float) -> None:
        if avg_flow < self.mu * 0.5:
            self.mu = max(0.01, self.mu * 0.95)
        elif avg_flow > self.mu * 2.0:
            self.mu = min(0.5, self.mu * 1.05)

        self.adaptation_log.append({
            "avg_flow": avg_flow,
            "noise": noise_level,
            "new_mu": self.mu,
        })

    def run(self, n_steps: int, noise_profile: np.ndarray,
            apply_decoherence: bool = True) -> Dict:
        results = {
            "flow_history": [],
            "noise_history": [],
            "slime_factors": [],
            "network_snapshots": [],
            "mu_history": [],
        }

        for step in range(n_steps):
            noise = float(noise_profile[step % len(noise_profile)])

            if apply_decoherence:
                self.network.apply_global_decoherence(gamma=noise, dt=self.dt)

            self.network.update_entanglement_all()
            step_result = self.step(noise)

            avg_slime = np.mean([e.slime_factor for e in self.network.edges.values()])
            results["flow_history"].append(step_result["average_flow"])
            results["noise_history"].append(noise)
            results["slime_factors"].append(avg_slime)
            results["mu_history"].append(self.mu)

            if step % 10 == 0:
                results["network_snapshots"].append(self.network.snapshot())

        return results

    def get_network_health_score(self) -> float:
        if not self.flow_history:
            return 0.0
        recent_flows = self.flow_history[-20:]
        avg_flow = np.mean(recent_flows)
        flow_stability = 1.0 - np.std(recent_flows) / (np.mean(recent_flows) + 1e-10)
        avg_entanglement = self.network.get_average_entanglement()
        fidelity = self.network.get_network_fidelity()
        health = 0.3 * min(1.0, avg_flow) + 0.3 * flow_stability + 0.2 * avg_entanglement + 0.2 * fidelity
        return float(np.clip(health, 0.0, 1.0))