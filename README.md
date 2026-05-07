# HQNN-Topology

Hyperconnected Quantum Neural Networks as Hybrid Topological Qubits with
Slime Mold Noise Cancellation and Neural Predictive Control.

## Installation

```bash
git clone https://github.com/hqnn-topology/hqnn-topology.git
cd hqnn-topology
pip install -e ".[dev]"
```

## Quick Start

```python
from hqnn.core.network import HyperconnectedQuantumNetwork
from hqnn.correction.topological import TopologicalErrorCorrector
from hqnn.optimizer.slime import QuantumSlimeMoldOptimizer
import numpy as np

net = HyperconnectedQuantumNetwork(n_nodes=9, connectivity=0.75, seed=42)
corrector = TopologicalErrorCorrector(net)
slime = QuantumSlimeMoldOptimizer(net, decay_rate=0.08, adaptive=True)

noise_profile = np.clip(0.05 + 0.03 * np.random.randn(100), 0.001, 0.5)
results = slime.run(n_steps=100, noise_profile=noise_profile)
print(f"Health Score: {slime.get_network_health_score():.4f}")
```

## Run Full Pipeline

```bash
python examples/run_full_pipeline.py
```

## Run Tests

```bash
pytest tests/ -v --cov=hqnn --cov-report=html
```

## C# Simulation

```bash
cd csharp/HQNN.Simulation
dotnet run
```