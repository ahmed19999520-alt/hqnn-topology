from hqnn.core.network import HyperconnectedQuantumNetwork
from hqnn.core.quantum_node import QuantumNode, NodeConfig
from hqnn.core.quantum_edge import QuantumEdge, EdgeConfig
from hqnn.core.hamiltonian import NetworkHamiltonian
from hqnn.algorithms.grover import GroverSearch
from hqnn.algorithms.shor import ShorAlgorithm, QuantumFourierTransform
from hqnn.algorithms.vqe import VQEAlgorithm, VQEAnsatz, MolecularHamiltonian
from hqnn.correction.topological import TopologicalErrorCorrector
from hqnn.correction.surface_code import SurfaceCode
from hqnn.optimizer.slime import QuantumSlimeMoldOptimizer
from hqnn.cage.beam_cage import QuantumBeamCage, BeamParameters
from hqnn.ml.noise_predictor_torch import NoisePredictor, NoisePredictorTrainer
from hqnn.ml.noise_predictor_tf import NoisePredictorTFTrainer, build_noise_predictor_tf
from hqnn.ml.hybrid_trainer import HybridAdaptiveController
from hqnn.utils.metrics import (
    quantum_volume,
    trace_distance,
    bures_distance,
    quantum_coherence_l1,
    multipartite_entanglement_measure,
    network_topology_metrics,
    noise_prediction_metrics,
    correction_efficiency_metrics,
)
from hqnn.utils.visualization import (
    plot_density_matrix,
    plot_bloch_sphere,
    plot_network_graph,
    plot_entanglement_matrix,
    plot_training_curves,
    plot_noise_prediction,
    plot_quantum_circuit_evolution,
    plot_error_correction_comparison,
    plot_beam_cage_analysis,
    plot_slime_dynamics,
    generate_full_report,
)

__version__ = "1.0.0"
__author__ = "HQNN Research Group"
__license__ = "MIT"

__all__ = [
    "HyperconnectedQuantumNetwork",
    "QuantumNode", "NodeConfig",
    "QuantumEdge", "EdgeConfig",
    "NetworkHamiltonian",
    "GroverSearch",
    "ShorAlgorithm", "QuantumFourierTransform",
    "VQEAlgorithm", "VQEAnsatz", "MolecularHamiltonian",
    "TopologicalErrorCorrector",
    "SurfaceCode",
    "QuantumSlimeMoldOptimizer",
    "QuantumBeamCage", "BeamParameters",
    "NoisePredictor", "NoisePredictorTrainer",
    "NoisePredictorTFTrainer", "build_noise_predictor_tf",
    "HybridAdaptiveController",
    "quantum_volume", "trace_distance", "bures_distance",
    "quantum_coherence_l1", "multipartite_entanglement_measure",
    "network_topology_metrics", "noise_prediction_metrics",
    "correction_efficiency_metrics",
    "plot_density_matrix", "plot_bloch_sphere", "plot_network_graph",
    "plot_entanglement_matrix", "plot_training_curves",
    "plot_noise_prediction", "plot_quantum_circuit_evolution",
    "plot_error_correction_comparison", "plot_beam_cage_analysis",
    "plot_slime_dynamics", "generate_full_report",
]