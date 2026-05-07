using System;
using System.Numerics;
using System.Collections.Generic;
using System.Linq;

namespace HQNN.Core
{
    public class NodeConfig
    {
        public int NQubits { get; set; } = 2;
        public string InitialState { get; set; } = "superposition";
        public string DecoherenceModel { get; set; } = "lindblad";
        public double T1 { get; set; } = 100e-6;
        public double T2 { get; set; } = 50e-6;
    }

    public class QuantumNode
    {
        public int Id { get; private set; }
        public NodeConfig Config { get; private set; }
        public int NQubits { get; private set; }
        public int Dim { get; private set; }
        public Complex[,] DensityMatrix { get; private set; }
        public double Phase { get; set; }
        public double TopologicalHealth { get; set; }
        public int ErrorCount { get; set; }
        public List<double> MeasurementHistory { get; private set; }

        private static readonly Random Rng = new Random();

        public QuantumNode(int nodeId, NodeConfig config = null)
        {
            Id = nodeId;
            Config = config ?? new NodeConfig();
            NQubits = Config.NQubits;
            Dim = 1 << NQubits;
            DensityMatrix = InitializeDensityMatrix();
            Phase = Rng.NextDouble() * 2 * Math.PI;
            TopologicalHealth = 1.0;
            ErrorCount = 0;
            MeasurementHistory = new List<double>();
        }

        private Complex[,] InitializeDensityMatrix()
        {
            var state = new Complex[Dim];
            double norm = 1.0 / Math.Sqrt(Dim);

            switch (Config.InitialState)
            {
                case "superposition":
                    for (int i = 0; i < Dim; i++)
                        state[i] = new Complex(norm, 0);
                    break;
                case "zero":
                    state[0] = Complex.One;
                    break;
                case "ghz":
                    state[0] = new Complex(norm * Math.Sqrt(2), 0);
                    state[Dim - 1] = new Complex(norm * Math.Sqrt(2), 0);
                    break;
                default:
                    for (int i = 0; i < Dim; i++)
                        state[i] = new Complex(norm, 0);
                    break;
            }

            var dm = new Complex[Dim, Dim];
            for (int i = 0; i < Dim; i++)
                for (int j = 0; j < Dim; j++)
                    dm[i, j] = state[i] * Complex.Conjugate(state[j]);

            return dm;
        }

        public void ApplyGate(Complex[,] U)
        {
            if (U.GetLength(0) != Dim || U.GetLength(1) != Dim)
                throw new ArgumentException($"Gate dimension mismatch: expected {Dim}x{Dim}");

            var temp = MatMul(U, DensityMatrix);
            var Udag = ConjugateTranspose(U);
            DensityMatrix = MatMul(temp, Udag);
            Renormalize();
        }

        public void ApplyDecoherence(double gamma, double dt)
        {
            switch (Config.DecoherenceModel)
            {
                case "lindblad":
                    ApplyLindblad(gamma, dt);
                    break;
                case "phase_damping":
                    ApplyPhaseDamping(gamma, dt);
                    break;
                default:
                    ApplyLindblad(gamma, dt);
                    break;
            }
        }

        private void ApplyLindblad(double gamma, double dt)
        {
            var L = new Complex[Dim, Dim];
            for (int i = 0; i < Dim - 1; i++)
                L[i, i + 1] = new Complex(Math.Sqrt(gamma), 0);

            var Ldag = ConjugateTranspose(L);
            var LdagL = MatMul(Ldag, L);

            var LrhoLdag = MatMul(MatMul(L, DensityMatrix), Ldag);
            var LdagLrho = MatMul(LdagL, DensityMatrix);
            var rhoLdagL = MatMul(DensityMatrix, LdagL);

            for (int i = 0; i < Dim; i++)
                for (int j = 0; j < Dim; j++)
                {
                    var dissipator = LrhoLdag[i, j]
                        - 0.5 * LdagLrho[i, j]
                        - 0.5 * rhoLdagL[i, j];
                    DensityMatrix[i, j] += dissipator * dt;
                }

            Renormalize();
        }

        private void ApplyPhaseDamping(double gamma, double dt)
        {
            double p = 1.0 - Math.Exp(-gamma * dt);
            for (int i = 0; i < Dim; i++)
                for (int j = 0; j < Dim; j++)
                    if (i != j)
                        DensityMatrix[i, j] *= (1.0 - p);
            Renormalize();
        }

        private void Renormalize()
        {
            Complex trace = Complex.Zero;
            for (int i = 0; i < Dim; i++)
                trace += DensityMatrix[i, i];

            if (trace.Magnitude > 1e-12)
                for (int i = 0; i < Dim; i++)
                    for (int j = 0; j < Dim; j++)
                        DensityMatrix[i, j] /= trace;
        }

        public double GetPurity()
        {
            var rho2 = MatMul(DensityMatrix, DensityMatrix);
            double purity = 0.0;
            for (int i = 0; i < Dim; i++)
                purity += rho2[i, i].Real;
            return purity;
        }

        public double GetEntropy()
        {
            var eigenvalues = ComputeEigenvalues();
            double entropy = 0.0;
            foreach (var ev in eigenvalues)
            {
                double v = ev.Real;
                if (v > 1e-12)
                    entropy -= v * Math.Log2(v);
            }
            return entropy;
        }

        public double MeasureObservable(Complex[,] observable)
        {
            var product = MatMul(observable, DensityMatrix);
            double result = 0.0;
            for (int i = 0; i < Dim; i++)
                result += product[i, i].Real;
            MeasurementHistory.Add(result);
            return result;
        }

        private Complex[] ComputeEigenvalues()
        {
            var eigenvalues = new Complex[Dim];
            for (int i = 0; i < Dim; i++)
                eigenvalues[i] = new Complex(DensityMatrix[i, i].Real, 0);
            return eigenvalues;
        }

        public static Complex[,] MatMul(Complex[,] A, Complex[,] B)
        {
            int m = A.GetLength(0), n = B.GetLength(1), k = A.GetLength(1);
            var C = new Complex[m, n];
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    for (int l = 0; l < k; l++)
                        C[i, j] += A[i, l] * B[l, j];
            return C;
        }

        public static Complex[,] ConjugateTranspose(Complex[,] M)
        {
            int m = M.GetLength(0), n = M.GetLength(1);
            var R = new Complex[n, m];
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    R[j, i] = Complex.Conjugate(M[i, j]);
            return R;
        }

        public void Reset()
        {
            DensityMatrix = InitializeDensityMatrix();
            TopologicalHealth = 1.0;
            ErrorCount = 0;
            MeasurementHistory.Clear();
        }
    }
}