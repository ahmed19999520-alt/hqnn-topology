import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple, Set
from scipy.linalg import expm
from hqnn.core.quantum_node import QuantumNode, NodeConfig
from hqnn.core.quantum_edge import QuantumEdge, EdgeConfig
from hqnn.utils.gates import (hadamard, cnot, kron_n, ghz_state,
                                von_neumann_entropy, pauli_x, pauli_z)


class HyperconnectedQuantumNetwork:
    def __init__(self,
                 n_nodes: int,
                 connectivity: float = 0.75,
                 node_config: Optional[NodeConfig] = None,
                 edge_config: Optional[EdgeConfig] = None,
                 seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)

        self.n = n_nodes
        self.connectivity = connectivity
        self.node_config = node_config or NodeConfig()
        self.edge_config = edge_config or EdgeConfig()
        self.time_step = 0

        self.nodes: List[QuantumNode] = [
            QuantumNode(i, self.node_config) for i in range(n_nodes)
        ]
        self.edges: Dict[Tuple[int, int], QuantumEdge] = {}

        self._build_graph()
        self.graph = self._build_networkx_graph()

        self.initial_euler = self._compute_euler_characteristic()
        self.euler_history: List[int] = [self.initial_euler]
        self.fidelity_history: List[float] = []
        self.entanglement_history: List[float] = []

    def _build_graph(self) -> None:
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if np.random.random() < self.connectivity:
                    self.edges[(i, j)] = QuantumEdge(i, j, self.edge_config)

        connected = self._check_connectivity()
        if not connected:
            self._ensure_connectivity()

    def _check_connectivity(self) -> bool:
        if self.n <= 1:
            return True
        visited: Set[int] = set()
        stack = [0]
        adj: Dict[int, List[int]] = {i: [] for i in range(self.n)}
        for (i, j) in self.edges:
            adj[i].append(j)
            adj[j].append(i)
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                stack.extend(adj[node])
        return len(visited) == self.n

    def _ensure_connectivity(self) -> None:
        for i in range(self.n - 1):
            if (i, i + 1) not in self.edges:
                self.edges[(i, i + 1)] = QuantumEdge(i, i + 1, self.edge_config)

    def _build_networkx_graph(self) -> nx.Graph:
        G = nx.Graph()
        G.add_nodes_from(range(self.n))
        for (i, j), edge in self.edges.items():
            G.add_edge(i, j, weight=abs(edge.weight))
        return G

    def _compute_euler_characteristic(self) -> int:
        V = self.n
        E = len(self.edges)
        F = 0
        nodes = list(range(self.n))
        for i in nodes:
            for j in nodes:
                for k in nodes:
                    if (i < j < k and
                            (i, j) in self.edges and
                            (j, k) in self.edges and
                            (i, k) in self.edges):
                        F += 1
        return V - E + F

    def get_adjacency_matrix(self) -> np.ndarray:
        A = np.zeros((self.n, self.n), dtype=complex)
        for (i, j), edge in self.edges.items():
            phase = np.exp(1j * edge.entanglement * np.pi)
            A[i, j] = edge.weight * phase
            A[j, i] = A[i, j].conj()
        return A

    def get_laplacian(self) -> np.ndarray:
        A = np.real(self.get_adjacency_matrix())
        D = np.diag(A.sum(axis=1))
        return D - A

    def get_spectral_gap(self) -> float:
        L = self.get_laplacian()
        eigenvalues = np.sort(np.linalg.eigvalsh(L))
        return float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0

    def apply_circuit_step(self, algorithm: str = "generic") -> None:
        H2 = kron_n(hadamard(), hadamard())
        for node in self.nodes:
            node.apply_gate(H2)

        CNOT = cnot()
        CNOT4 = kron_n(CNOT, np.eye(1, dtype=complex).repeat(1, axis=0))

        for (i, j), edge in self.edges.items():
            if abs(edge.weight) > 0.02:
                phase = np.angle(edge.weight)
                gate_2q = kron_n(hadamard(), hadamard())
                U = gate_2q * np.exp(1j * phase * edge.entanglement)
                self.nodes[i].apply_gate(U)

        self.time_step += 1

    def update_entanglement_all(self) -> None:
        for (i, j), edge in self.edges.items():
            rho_i = self.nodes[i].density_matrix
            rho_j = self.nodes[j].density_matrix
            edge.update_entanglement_metrics(rho_i, rho_j)

    def apply_global_decoherence(self, gamma: float, dt: float) -> None:
        for node in self.nodes:
            node.apply_decoherence(gamma, dt)

    def get_network_fidelity(self) -> float:
        purities = [node.get_purity() for node in self.nodes]
        return float(np.mean(purities))

    def get_average_entanglement(self) -> float:
        if not self.edges:
            return 0.0
        return float(np.mean([e.entanglement for e in self.edges.values()]))

    def get_total_quantum_flow(self) -> float:
        return float(np.sum([e.quantum_flow for e in self.edges.values()]))

    def get_network_entropy(self) -> float:
        entropies = [node.get_entropy() for node in self.nodes]
        return float(np.mean(entropies))

    def snapshot(self) -> Dict:
        euler = self._compute_euler_characteristic()
        fidelity = self.get_network_fidelity()
        entanglement = self.get_average_entanglement()
        self.euler_history.append(euler)
        self.fidelity_history.append(fidelity)
        self.entanglement_history.append(entanglement)
        return {
            "time_step": self.time_step,
            "euler_characteristic": euler,
            "euler_drift": abs(euler - self.initial_euler),
            "network_fidelity": fidelity,
            "average_entanglement": entanglement,
            "total_quantum_flow": self.get_total_quantum_flow(),
            "network_entropy": self.get_network_entropy(),
            "spectral_gap": self.get_spectral_gap(),
        }

    def reset(self) -> None:
        for node in self.nodes:
            node.reset()
        for edge in self.edges.values():
            edge.reset()
        self.time_step = 0
        self.fidelity_history.clear()
        self.entanglement_history.clear()