# API Reference

## hqnn.core.network.HyperconnectedQuantumNetwork

### Constructor

```python
HyperconnectedQuantumNetwork(
    n_nodes: int,
    connectivity: float = 0.75,
    node_config: Optional[NodeConfig] = None,
    edge_config: Optional[EdgeConfig] = None,
    seed: Optional[int] = None
)
```

**Parameters**
- `n_nodes`: Number of quantum nodes (logical qubits)
- `connectivity`: Probability that any two nodes are connected (0 to 1)
- `node_config`: Configuration for each quantum node
- `edge_config`: Configuration for each quantum edge
- `seed`: Random seed for reproducibility

**Key Methods**
- `apply_circuit_step()`: Apply one step of the quantum circuit
- `apply_global_decoherence(gamma, dt)`: Apply Lindblad decoherence
- `update_entanglement_all()`: Recompute all edge entanglement metrics
- `get_adjacency_matrix()`: Returns complex adjacency matrix
- `get_network_fidelity()`: Returns mean purity across all nodes
- `get_average_entanglement()`: Returns mean entanglement across all edges
- `snapshot()`: Returns full diagnostic dictionary

## hqnn.algorithms.grover.GroverSearch

```python
GroverSearch(n_qubits: int, network: Optional[HyperconnectedQuantumNetwork] = None)
```

**Methods**
- `run(targets, noise_level, n_iterations, use_network_correction)`: Run full search
- `oracle(state, targets, phase)`: Apply oracle to state
- `diffusion(state)`: Apply Grover diffusion operator
- `amplitude_estimation(f, n_estimation_qubits)`: Estimate count of solutions

## hqnn.algorithms.vqe.VQEAlgorithm

```python
VQEAlgorithm(n_qubits: int = 4, n_layers: int = 2, molecule: str = "H2")
```

**Methods**
- `optimize_gradient_descent(n_iterations, lr, noise)`: GD optimization
- `optimize_scipy(method, noise)`: SciPy-based optimization
- `parameter_shift_gradient(params, noise)`: Compute gradient via parameter shift
- `exact_ground_energy()`: Return exact eigenvalue for benchmarking

## hqnn.correction.topological.TopologicalErrorCorrector

```python
TopologicalErrorCorrector(
    network: HyperconnectedQuantumNetwork,
    euler_tolerance: int = 2,
    fidelity_threshold: float = 0.80,
    entanglement_threshold: float = 0.25
)
```

**Methods**
- `diagnose()`: Run full network diagnosis
- `correct(diagnosis)`: Apply error correction if needed
- `get_correction_statistics()`: Return correction performance metrics

## hqnn.ml.noise_predictor_torch.NoisePredictorTrainer

```python
NoisePredictorTrainer(model: NoisePredictor, device: str = "auto", learning_rate: float = 1e-3)
```

**Methods**
- `train(noise_sequence, n_epochs, early_stopping_patience, checkpoint_path)`: Train model
- `predict(noise_buffer)`: Return (predictions, uncertainties) tuple
- `prepare_data(noise_sequence, val_split)`: Create DataLoaders

## hqnn.ml.hybrid_trainer.HybridAdaptiveController

```python
HybridAdaptiveController(
    network, corrector, slime_optimizer,
    predictor_backend: str = "torch",
    seq_length: int = 20,
    forecast_horizon: int = 5
)
```

**Methods**
- `pretrain(synthetic_noise, n_epochs)`: Pre-train predictor on synthetic data
- `step(actual_noise, step_index)`: Execute one control step
- `run_full_control_loop(noise_profile, pretrain_steps)`: Run complete pipeline