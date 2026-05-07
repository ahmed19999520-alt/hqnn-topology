using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;

namespace HQNN.Core
{
    public class CorrectionRecord
    {
        public int Step { get; set; }
        public string ErrorType { get; set; } = string.Empty;
        public List<string> CorrectionsApplied { get; set; } = new();
        public double PreFidelity { get; set; }
        public double PostFidelity { get; set; }
    }

    public class DiagnosisResult
    {
        public int TimeStep { get; set; }
        public int EulerCharacteristic { get; set; }
        public int EulerDrift { get; set; }
        public double NetworkFidelity { get; set; }
        public double AverageEntanglement { get; set; }
        public bool NeedsCorrection { get; set; }
        public string? ErrorType { get; set; }
    }

    public class TopologicalCorrector
    {
        private readonly HyperconnectedQuantumNetwork _network;
        private readonly int _eulerTolerance;
        private readonly double _fidelityThreshold;
        private readonly double _entanglementThreshold;
        private readonly List<CorrectionRecord> _log;
        private int _totalCorrections;
        private int _totalDetections;
        private static readonly Random Rng = new Random(42);

        public IReadOnlyList<CorrectionRecord> Log => _log.AsReadOnly();
        public int TotalCorrections => _totalCorrections;

        public TopologicalCorrector(
            HyperconnectedQuantumNetwork network,
            int eulerTolerance = 2,
            double fidelityThreshold = 0.80,
            double entanglementThreshold = 0.25)
        {
            _network = network;
            _eulerTolerance = eulerTolerance;
            _fidelityThreshold = fidelityThreshold;
            _entanglementThreshold = entanglementThreshold;
            _log = new List<CorrectionRecord>();
            _totalCorrections = 0;
            _totalDetections = 0;
        }

        public DiagnosisResult Diagnose()
        {
            var snapshot = _network.TakeSnapshot();
            _totalDetections++;

            string? errorType = null;
            bool needsCorrection = false;

            if (snapshot.EulerDrift > _eulerTolerance)
            {
                errorType = "topological";
                needsCorrection = true;
            }
            else if (snapshot.NetworkFidelity < _fidelityThreshold)
            {
                errorType = "decoherence";
                needsCorrection = true;
            }
            else if (snapshot.AverageEntanglement < _entanglementThreshold)
            {
                errorType = "entanglement_loss";
                needsCorrection = true;
            }

            return new DiagnosisResult
            {
                TimeStep = snapshot.TimeStep,
                EulerCharacteristic = snapshot.EulerCharacteristic,
                EulerDrift = snapshot.EulerDrift,
                NetworkFidelity = snapshot.NetworkFidelity,
                AverageEntanglement = snapshot.AverageEntanglement,
                NeedsCorrection = needsCorrection,
                ErrorType = errorType,
            };
        }

        public bool Correct(DiagnosisResult diagnosis)
        {
            if (!diagnosis.NeedsCorrection)
                return false;

            double preFidelity = diagnosis.NetworkFidelity;
            var corrections = new List<string>();

            if (diagnosis.ErrorType == "topological" || diagnosis.EulerDrift > _eulerTolerance)
            {
                RestoreTopology();
                corrections.Add("topology_restore");
            }

            if (diagnosis.ErrorType == "decoherence" || diagnosis.NetworkFidelity < _fidelityThreshold)
            {
                ApplyDecoherenceCorrection();
                corrections.Add("decoherence_correction");
            }

            if (diagnosis.ErrorType == "entanglement_loss" || diagnosis.AverageEntanglement < _entanglementThreshold)
            {
                ReinforceEntanglement();
                corrections.Add("entanglement_reinforce");
            }

            _totalCorrections++;
            _log.Add(new CorrectionRecord
            {
                Step = diagnosis.TimeStep,
                ErrorType = diagnosis.ErrorType ?? "unknown",
                CorrectionsApplied = corrections,
                PreFidelity = preFidelity,
                PostFidelity = _network.GetNetworkFidelity(),
            });

            return true;
        }

        private void RestoreTopology()
        {
            foreach (var edge in _network.Edges.Values)
            {
                if (edge.Weight.Magnitude < 0.01)
                {
                    edge.Weight = new Complex(
                        Rng.NextGaussian() * 0.05,
                        Rng.NextGaussian() * 0.05
                    );
                    edge.SlimeFactor = Math.Max(edge.SlimeFactor, 0.5);
                }
            }
        }

        private void ApplyDecoherenceCorrection()
        {
            foreach (var node in _network.Nodes)
            {
                double purity = node.GetPurity();
                if (purity < 0.6)
                {
                    double alpha = (0.7 - purity) * 0.4;
                    int dim = node.Dim;
                    var idealDm = new Complex[dim, dim];
                    double val = 1.0 / dim;
                    for (int i = 0; i < dim; i++)
                        idealDm[i, i] = new Complex(val, 0);

                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j < dim; j++)
                            node.DensityMatrix[i, j] =
                                (1 - alpha) * node.DensityMatrix[i, j] +
                                alpha * idealDm[i, j];

                    node.ErrorCount++;
                }
            }
        }

        private void ReinforceEntanglement()
        {
            foreach (var edge in _network.Edges.Values)
            {
                if (edge.Entanglement < _entanglementThreshold)
                {
                    edge.Entanglement = Math.Min(1.0, edge.Entanglement + 0.15);
                    edge.SlimeFactor = Math.Min(2.0, edge.SlimeFactor * 1.2);
                }
            }
        }

        public Dictionary<string, object> GetStatistics()
        {
            var stats = new Dictionary<string, object>
            {
                ["total_corrections"] = _totalCorrections,
                ["total_detections"] = _totalDetections,
                ["correction_rate"] = _totalDetections > 0
                    ? (double)_totalCorrections / _totalDetections
                    : 0.0,
            };

            if (_log.Any())
            {
                stats["avg_pre_fidelity"] = _log.Average(l => l.PreFidelity);
                stats["avg_post_fidelity"] = _log.Average(l => l.PostFidelity);
                stats["avg_fidelity_improvement"] = _log.Average(l => l.PostFidelity - l.PreFidelity);
            }

            return stats;
        }
    }
}