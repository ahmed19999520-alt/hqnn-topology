# Changelog

## [1.0.0] - 2026-06-06

### Added
- HyperconnectedQuantumNetwork: N-node quantum network with complex edge weights
- QuantumNode: Local density matrix with Lindblad, amplitude damping, phase damping models
- QuantumEdge: Entanglement-weighted edges with slime mold dynamics
- NetworkHamiltonian: Ising, Heisenberg, and topological Hamiltonian builders
- GroverSearch: N-qubit Grover search with topological noise correction
- ShorAlgorithm: QFT-based period finding and integer factoring
- VQEAlgorithm: Variational quantum eigensolver with H2 molecular Hamiltonian
- TopologicalErrorCorrector: Euler characteristic monitoring and stabilizer correction
- SurfaceCode: Distance-d surface code with MWPM decoding
- QuantumSlimeMoldOptimizer: Physarum-inspired quantum flow optimizer
- QuantumBeamCage: Gaussian beam optical trap with atomic motion simulation
- NoisePredictor (PyTorch): TCN + BiLSTM + MultiHeadAttention predictor
- NoisePredictorTFTrainer (TensorFlow): Transformer-based predictor
- HybridAdaptiveController: Unified predictive control loop
- Full metrics suite: QFI, trace distance, Bures distance, coherence measures
- Full visualization suite: density matrix, Bloch sphere, network graph, etc.
- C# core library: QuantumNode, QuantumNetwork, TopologicalCorrector
- C# simulation runner: full pipeline in .NET 8
- 84 unit and integration tests with >94% coverage
- GitHub Actions CI/CD pipeline
- Docker support
- Pre-commit hooks
- Synthetic data generator
- Model evaluation script
- Performance profiler