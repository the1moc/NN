using MathNet.Numerics.LinearAlgebra;
using NNWithoutGUI.Network;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNWithoutGUI
{
	class Startup
	{
		static void Main(string[] args)
		{
			NeuralNetwork network = new NeuralNetwork(3, 3, 3, 0.5);
			Vector<double> inputs = Vector<double>.Build.DenseOfArray(new double[] { 0.01, 0.99, 0.70});
			Vector<double> outputs = Vector<double>.Build.DenseOfArray(new double[] { 0.01, 0.99, 0.70 });

			network.InitialiseWeights();

			for (int i = 0; i < 600; i++)
			{
				network.Train(inputs, outputs);
			}

			Console.WriteLine(network.Query(inputs));

			Console.Read();
		}
	}
}
