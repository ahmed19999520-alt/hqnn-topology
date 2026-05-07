using System;
using System.Collections.Generic;
using System.Linq;
using HQNN.Core;

namespace HQNN.Simulation
{
    public class SimulationConfig
    {
        public int NNodes { get; set; } = 9;
        public double Connectivity { get; set; } = 0.75;
        public int NSteps { get; set; } = 200;
        public double BaseNoise { get; set; } = 0.05;
        public double NoiseAmplitude { get; set; } = 0.03;
        public int Seed { get; set; } = 42;
    }

    public class SimulationResult
    {
        public List<double> FidelityHistory { get; set; } = new();
        public List<double> EntanglementHistory { get; set; } = new();
        public List<double> NoiseHistory { get; set; } = new();
        public List<int> EulerHistory { get; set; } = new();
        public List<int> CorrectionSteps { get; set; } = new();
        public int TotalCorrections { get; set; }
        public double FinalFidelity { get; set; }
        public double FinalEntanglement { get; set; }
    }

    public class SimulationRunner
    {
        private readonly SimulationConfig _config;
        private readonly HyperconnectedQuantumNetwork _network;
        private readonly Random _rng;

        public SimulationRunner(SimulationConfig config = null)
        {
            _config = config ?? new SimulationConfig();
            _rng = new Random(_config.Seed);
            _network = new HyperconnectedQuantumNetwork(
                _config.NNodes,
                _config.Connectivity,
                seed: _config.Seed
            );
        }

        private double[] GenerateNoiseProfile()
        {
            var profile = new double[_config.NSteps];
            for (int t = 0; t < _config.NSteps; t++)
            {
                double base_noise = _config.BaseNoise;
                double variation = _config.NoiseAmplitude * Math.Sin(0.3 * t);
                double random_part = 0.01 * (_rng.NextDouble() * 2 - 1);
                profile[t] = Math.Max(0.001, Math.Min(0.5, base_noise + variation + random_part));
            }
            return profile;
        }

        private bool NeedsCorrection(NetworkSnapshot snapshot,
                                      double eulerTolerance = 2,
                                      double fidelityThreshold = 0.80)
        {
            return snapshot.EulerDrift > eulerTolerance ||
                   snapshot.NetworkFidelity < fidelityThreshold;
        }

        private void ApplyCorrection()
        {
            foreach (var edge in _network.Edges.Values)
            {
                if (edge.Entanglement < 0.25)
                    edge.Entanglement = Math.Min(1.0, edge.Entanglement + 0.15);
                if (edge.Weight.Magnitude < 0.01)
                {
                    var rng = new Random();
                    edge.Weight = new System.Numerics.Complex(
                        rng.NextGaussian() * 0.05,
                        rng.NextGaussian() * 0.05
                    );
                }
            }
        }

        private void ApplySlimeStep(double noiseLevel)
        {
            foreach (var edge in _network.Edges.Values)
                edge.ApplySlimeUpdate(noiseLevel, dt: 0.01);
        }

        public SimulationResult Run()
        {
            var result = new SimulationResult();
            var noiseProfile = GenerateNoiseProfile();

            Console.WriteLine("HQNN Simulation Starting");
            Console.WriteLine($"Nodes: {_config.NNodes} | Steps: {_config.NSteps}");
            Console.WriteLine(new string('-', 55));

            for (int step = 0; step < _config.NSteps; step++)
            {
                double noise = noiseProfile[step];
                _network.ApplyGlobalDecoherence(noise, dt: 0.01);
                ApplySlimeStep(noise);
                _network.Step();

                var snapshot = _network.TakeSnapshot();
                result.FidelityHistory.Add(snapshot.NetworkFidelity);
                result.EntanglementHistory.Add(snapshot.AverageEntanglement);
                result.NoiseHistory.Add(noise);
                result.EulerHistory.Add(snapshot.EulerCharacteristic);

                if (NeedsCorrection(snapshot))
                {
                    ApplyCorrection();
                    result.CorrectionSteps.Add(step);
                    result.TotalCorrections++;
                }

                if (step % 25 == 0)
                {
                    Console.WriteLine(
                        $"Step {step,4} | Fidelity: {snapshot.NetworkFidelity:F4} | " +
                        $"Entanglement: {snapshot.AverageEntanglement:F4} | " +
                        $"Noise: {noise:F4} | " +
                        $"Euler Drift: {snapshot.EulerDrift}"
                    );
                }
            }

            result.FinalFidelity = result.FidelityHistory.LastOrDefault();
            result.FinalEntanglement = result.EntanglementHistory.LastOrDefault();

            Console.WriteLine(new string('-', 55));
            Console.WriteLine($"Simulation Complete");
            Console.WriteLine($"Final Fidelity     : {result.FinalFidelity:F4}");
            Console.WriteLine($"Final Entanglement : {result.FinalEntanglement:F4}");
            Console.WriteLine($"Total Corrections  : {result.TotalCorrections}");

            return result;
        }
    }

    public class Program
    {
        public static void Main(string[] args)
        {
            var config = new SimulationConfig
            {
                NNodes = 9,
                Connectivity = 0.75,
                NSteps = 200,
                BaseNoise = 0.05,
                NoiseAmplitude = 0.03,
                Seed = 42,
            };

            var runner = new SimulationRunner(config);
            var result = runner.Run();
        }
    }
}