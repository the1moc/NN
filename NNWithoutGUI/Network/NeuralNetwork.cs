using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Random;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra.Complex;

namespace NNWithoutGUI.Network
{
	class NeuralNetwork
	{
		// The number of input parameters.
		private int _inputCount;

		// The number of hidden nodes.
		private int _hiddenNodesCount;

		// The number of output nodes.
		private int _outputNodesCount;

		// The learning rate for the neural network.
		private double _learningRate;

		// The weights from from the input layer to the hidden layer.
		private Matrix<double> _inputWeights;

		// The weights going from the hidden layer to the output layer.
		private Matrix<double> _hiddenWeights;

		/// <summary>
		/// Constructor for the Neural Network.
		/// </summary>
		/// <param name="inputCount">The number of input parameters.</param>
		/// <param name="hiddenNodeCount">The number of hidden nodes.</param>
		/// <param name="outputNodeCount">The number of output nodes.</param>
		/// <param name="learningRate">The learning rate for the network.</param>
		public NeuralNetwork(int inputCount, int hiddenNodeCount, int outputNodeCount, double learningRate)
		{
			this._inputCount        = inputCount;
			this._hiddenNodesCount  = hiddenNodeCount;
			this._outputNodesCount  = outputNodeCount;
			this._learningRate = learningRate;
		}

		/// <summary>
		/// Initialises the weights.
		/// </summary>
		public void InitialiseWeights()
		{
			Random rng = new Random();

			// Generate random weights
			this._inputWeights  = Matrix<double>.Build.Dense(_inputCount, _hiddenNodesCount, new double[] { 0.01,0.5,0.9, 0.01, 0.5, 0.9, 0.01, 0.5, 0.9});
			this._hiddenWeights = Matrix<double>.Build.Dense(_hiddenNodesCount, _outputNodesCount, new double[] { 0.01, 0.5, 0.9, 0.01, 0.5, 0.9, 0.01, 0.5, 0.9 });
		}

		/// <summary>
		/// Feed the network some input data and run it through.
		/// </summary>
		/// <param name="inputs">The inputs.</param>
		/// <returns></returns>
		public Vector<double> Query(Vector<double> inputs)
		{
			// Input layer -> Hidden Layer
			Vector<double> hiddenNodes = Vector<double>.Build.DenseOfEnumerable(this._inputWeights.EnumerateRows().Select(row => (row * inputs)));
			hiddenNodes                = hiddenNodes.Map(value => SpecialFunctions.Logistic(value));

			// Hidden layer -> Output layer
			Vector<double> outputNodes = Vector<double>.Build.DenseOfEnumerable(this._hiddenWeights.EnumerateRows().Select(row => (row * hiddenNodes)));
			outputNodes                = outputNodes.Map(value => SpecialFunctions.Logistic(value));

			return outputNodes;
		}

		/// <summary>
		/// Feed the network some input data and run it through.
		/// </summary>
		/// <param name="inputs">The inputs.</param>
		/// <returns></returns>
		public void Train(Vector<double> inputs, Vector<double> target)
		{
			// Pass the values forward
			// Input layer -> Hidden Layer
			Vector<double> hiddenNodes        = this._inputWeights * inputs;
			Vector<double> sigmoidHiddenNodes = hiddenNodes.Map(value => SpecialFunctions.Logistic(value));

			// Hidden layer -> Output layer
			Vector<double> outputNodes        = this._hiddenWeights * sigmoidHiddenNodes;
			Vector<double> sigmoidOutputNodes = outputNodes.Map(value => SpecialFunctions.Logistic(value));

			// Calculate the total error for the network
			double totalNetworkError = sigmoidOutputNodes.Select((node, index) => Math.Pow((target[index] - node), 2) * 0.5).Sum();

			// Pass the error backwards
			// Hidden -> Output layer weight update

			// The change in error against each output node
			Vector<double> errorAgainstOutputNodes  = Vector<double>.Build.DenseOfEnumerable(sigmoidOutputNodes.Select((nodeValue, index) => -(target[index] - nodeValue)));

			// The change in the output node output against the input
			Vector<double> outputNodesOutputAgainstInput  = Vector<double>.Build.DenseOfEnumerable(sigmoidOutputNodes.Select((nodeValue) => nodeValue * (1 - nodeValue)));

			// The change in error against each output nodes input
			Vector<double> errorAgainstOutputNodesInput = errorAgainstOutputNodes.PointwiseMultiply(outputNodesOutputAgainstInput);

			// The amount to update the hidden weights
			Vector<double> hiddenWeightErrors = errorAgainstOutputNodesInput.PointwiseMultiply(sigmoidOutputNodes);

			// Updated weight matrix
			Matrix<double> updatedHiddenWeights = Matrix<double>.Build.DenseOfRowVectors(this._hiddenWeights.EnumerateRows().Select((vector, index) => { return vector - (this._learningRate * hiddenWeightErrors[index]); }));

			// Input -> Hidden layer weight update

			// The change in error against each hidden node
			Vector<double> errorAgainstHiddenNodesOutputs = Vector<double>.Build.DenseOfEnumerable(this._inputWeights.Transpose().EnumerateRows().Select(row => (row * errorAgainstOutputNodesInput)));

			// The change in the hidden nodes output against its net input
			Vector<double> outputAgainstInput = Vector<double>.Build.DenseOfEnumerable(sigmoidHiddenNodes.Select((nodeValue) => nodeValue * (1 - nodeValue)));

			// The amount to update the hidden weights
			Vector<double> inputWeightErrors = outputAgainstInput.PointwiseMultiply(inputs);

			// Updated weight matrix
			Matrix<double> updatedInputWeights = Matrix<double>.Build.DenseOfRowVectors(this._inputWeights.Transpose().EnumerateRows().Select((vector, index) => { return vector - (this._learningRate * inputWeightErrors[index]); })).Transpose();

			this._inputWeights  = updatedInputWeights;
			this._hiddenWeights = updatedHiddenWeights;

			Console.WriteLine(totalNetworkError);
		}
	}
}
