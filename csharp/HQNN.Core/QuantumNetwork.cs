using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;

namespace HQNN.Core
{
    public class EdgeKey : IEquatable<EdgeKey>
    {
        public int I { get; }
        public int J { get; }
        public EdgeKey(int i, int j) { I = i; J = j; }
        public bool Equals(EdgeKey other) => other != null && I == other.I && J == other.J;
        public override int GetHashCode() => HashCode.Combine(I, J);
    }

    public class QuantumEdge
    {
        public int NodeI { get; }
        public int NodeJ { get; }
        public Complex Weight { get; set; }
        public double Entanglement { get; set; }
        public double QuantumFlow { get; set; }
        public double SlimeFactor { get; set; }

        private static readonly Random Rng = new Random();

        public QuantumEdge(int i, int j, double weightScale = 0.1)
        {
            NodeI = i;
            NodeJ = j;
            Weight = new Complex(
                Rng.NextGaussian() * weightScale,
                Rng.NextGaussian() * weightScale
            );
            Entanglement = 0.0;
            QuantumFlow = 0.0;
            SlimeFactor = 1.0;
        }

        public void ApplySlimeUpdate(double noiseLevel, double dt,
                                      double topologyProtection = 0.5)
        {
            double growth = (QuantumFlow - 0.1) * SlimeFactor;
            double noiseSuppression = 1.0 - Math.Min(0.95, noiseLevel);
            double topoTerm = Entanglement * topologyProtection;
            double dSlime = (growth * noiseSuppression + topoTerm) * dt;
            SlimeFactor = Math.Max(0.01, Math.Min(10.0, SlimeFactor + dSlime));
            double phase = Math.Atan2(Weight.Imaginary, Weight.Real);
            double newMag = Math.Max(1e-6, Math.Min(5.0, Weight.Magnitude * (1.0 + 0.05 * dSlime)));
            Weight = new Complex(newMag * Math.Cos(phase), newMag * Math.Sin(phase));
        }
    }

    public class HyperconnectedQuantumNetwork
    {
        public int N { get; private set; }
        public List<QuantumNode> Nodes { get; private set; }
        public Dictionary<EdgeKey, QuantumEdge> Edges { get; private set; }
        public int InitialEulerCharacteristic { get; private set; }
        public int TimeStep { get; private set; }
        public List<double> FidelityHistory { get; private set; }
        public List<double> EntanglementHistory { get; private set; }

        private readonly Random _rng;
        private readonly NodeConfig _nodeConfig;

        public HyperconnectedQuantumNetwork(int nNodes, double connectivity = 0.75,
                                             NodeConfig nodeConfig = null, int seed = 42)
        {
            N = nNodes;
            _rng = new Random(seed);
            _nodeConfig = nodeConfig ?? new NodeConfig();
            Nodes = new List<QuantumNode>();
            Edges = new Dictionary<EdgeKey, QuantumEdge>();
            FidelityHistory = new List<double>();
            EntanglementHistory = new List<double>();
            TimeStep = 0;

            for (int i = 0; i < nNodes; i++)
                Nodes.Add(new QuantumNode(i, _nodeConfig));

            for (int i = 0; i < nNodes; i++)
                for (int j = i + 1; j < nNodes; j++)
                    if (_rng.NextDouble() < connectivity)
                        Edges[new EdgeKey(i, j)] = new QuantumEdge(i, j);

            EnsureConnectivity();
            InitialEulerCharacteristic = ComputeEulerCharacteristic();
        }

        private void EnsureConnectivity()
        {
            for (int i = 0; i < N - 1; i++)
            {
                var key = new EdgeKey(i, i + 1);
                if (!Edges.ContainsKey(key))
                    Edges[key] = new QuantumEdge(i, i + 1);
            }
        }

        public int ComputeEulerCharacteristic()
        {
            int V = N;
            int E = Edges.Count;
            int F = 0;

            for (int i = 0; i < N; i++)
                for (int j = i + 1; j < N; j++)
                    for (int k = j + 1; k < N; k++)
                        if (Edges.ContainsKey(new EdgeKey(i, j)) &&
                            Edges.ContainsKey(new EdgeKey(j, k)) &&
                            Edges.ContainsKey(new EdgeKey(i, k)))
                            F++;

            return V - E + F;
        }

        public void ApplyGlobalDecoherence(double gamma, double dt)
        {
            foreach (var node in Nodes)
                node.ApplyDecoherence(gamma, dt);
        }

        public double GetNetworkFidelity()
        {
            return Nodes.Average(n => n.GetPurity());
        }

        public double GetAverageEntanglement()
        {
            if (!Edges.Any()) return 0.0;
            return Edges.Values.Average(e => e.Entanglement);
        }

        public NetworkSnapshot TakeSnapshot()
        {
            int euler = ComputeEulerCharacteristic();
            double fidelity = GetNetworkFidelity();
            double entanglement = GetAverageEntanglement();
            FidelityHistory.Add(fidelity);
            EntanglementHistory.Add(entanglement);

            return new NetworkSnapshot
            {
                TimeStep = TimeStep,
                EulerCharacteristic = euler,
                EulerDrift = Math.Abs(euler - InitialEulerCharacteristic),
                NetworkFidelity = fidelity,
                AverageEntanglement = entanglement,
            };
        }

        public void Step()
        {
            TimeStep++;
        }
    }

    public class NetworkSnapshot
    {
        public int TimeStep { get; set; }
        public int EulerCharacteristic { get; set; }
        public int EulerDrift { get; set; }
        public double NetworkFidelity { get; set; }
        public double AverageEntanglement { get; set; }
    }

    public static class RandomExtensions
    {
        public static double NextGaussian(this Random rng)
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = 1.0 - rng.NextDouble();
            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        }
    }
}