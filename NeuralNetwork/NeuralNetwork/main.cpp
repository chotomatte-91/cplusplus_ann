#include "kernel.h"
#include "utils\MathUtils.h"
#include "ann/neuralnet.h"
#include <cmath>
#include <algorithm>

float mysigmoid(float val)
{
	float denom = 1.f + std::exp(-val);
	return 1.f / denom;
}

float mysigmoidPrime(float val)
{
	float temp = mysigmoid(val);
	return temp * (1.f - temp);
}

float mytanH(float val)
{
	return std::tanh(val);
}

float mytanHPrime(float val)
{
	return 1.f - (val * val);
}

void ann_cpu_test()
{
	//two neurons input, two neurons hidden, one neuron output
	std::vector<unsigned> config{ 2, 2, 1 };
	NeuralNet<float> nn(config);

	//hardcoded XOR input and labels
	/*XOR TABLE
	0 0 0
	0 1 1
	1 0 1
	1 1 0
	*/

	std::vector<std::vector<float>> inputs = {
	{ 0, 0 },
	{ 0, 1 },
	{ 1, 0 },
	{ 1, 1 }
	};
	std::vector<float> labels = { 0, 1, 1, 0 };

	//hardcoded initialization of weights and bias for testing
	float i1_h1 = -0.7706f;
	float i2_h1 = 0.6257f;
	float h1_bias = 0.1859f;
	float i1_h2 = 0.5607f;
	float i2_h2 = 0.2109f;
	float h2_bias = -0.7984f;
	float h1_o1 = 0.5951f;
	float h2_o1 = 0.3433f;
	float o1_bias = 0.1328f;

	//set weights

	//input layer to first neuron in hidden layer
	nn.getNeuron(0, 0).setWeight(0, i1_h1);
	nn.getNeuron(0, 1).setWeight(0, i2_h1);
	nn.getNeuron(0, 2).setWeight(0, h1_bias);

	//input layer to second neuron in hidden layer
	nn.getNeuron(0, 0).setWeight(1, i1_h2);
	nn.getNeuron(0, 1).setWeight(1, i2_h2);
	nn.getNeuron(0, 2).setWeight(0, h2_bias);

	//hidden to output layer
	nn.getNeuron(1, 0).setWeight(0, h1_o1);
	nn.getNeuron(1, 1).setWeight(0, h2_o1);
	nn.getNeuron(1, 2).setWeight(0, o1_bias);

	//tanh is activation function for input and hidden layer
	nn.getNeuron(1, 0).setActivationFunctions(mytanH, mytanHPrime);
	nn.getNeuron(1, 1).setActivationFunctions(mytanH, mytanHPrime);

	//sigmoid is activation function for output layer
	nn.getNeuron(2, 0).setActivationFunctions(mysigmoid, mysigmoidPrime);

	//nn.train(inputs, labels, 0.5f, 1);
}

int main(int argc, char **argv)
{
	//ann_cpu_test();
#if 1

	const uint inputHeight = std::atoi(argv[1]);
	const uint numStream = std::atoi(argv[2]);
	const uint numIter = std::atoi(argv[3]);

	cudaStream_t* streams;
	streams = new cudaStream_t[numStream];
	cudaStreamCreateWithFlags(streams, cudaStreamNonBlocking);

	//Test test
	std::vector<unsigned> test{ 2, 2, 1 };
	NeuralNet<float> testNetwork(test);

	//hardcoded initialization of weights and bias for testing
	float i1_h1 = -0.7706f;
	float i2_h1 = 0.6257f;
	float h1_bias = 0.1859f;
	float i1_h2 = 0.5607f;
	float i2_h2 = 0.2109f;
	float h2_bias = -0.7984f;
	float h1_o1 = 0.5951f;
	float h2_o1 = 0.3433f;
	float o1_bias = 0.1328f;

	//set weights

	//input layer to first neuron in hidden layer
	testNetwork.getNeuron(0, 0).setWeight(0, i1_h1);
	testNetwork.getNeuron(0, 0).setWeight(1, i1_h1);
	testNetwork.getNeuron(0, 1).setWeight(0, i2_h1);
	testNetwork.getNeuron(0, 1).setWeight(1, i2_h1);
	testNetwork.getNeuron(0, 2).setWeight(0, 0);
	testNetwork.getNeuron(0, 2).setWeight(1, 0);

	testNetwork.getNeuron(1, 0).setWeight(0, h2_bias);
	testNetwork.getNeuron(1, 1).setWeight(0, i2_h1);
	testNetwork.getNeuron(1, 2).setWeight(0, 0);

	////input layer to second neuron in hidden layer
	//testNetwork.getNeuron(0, 0).setWeight(0, i1_h2);
	//testNetwork.getNeuron(0, 1).setWeight(1, i2_h2);
	//testNetwork.getNeuron(0, 2).setWeight(0, h2_bias);
	//
	////hidden to output layer
	//testNetwork.getNeuron(1, 0).setWeight(0, h1_o1);
	//testNetwork.getNeuron(1, 1).setWeight(0, h2_o1);
	//testNetwork.getNeuron(1, 2).setWeight(0, o1_bias);


	int hiddenLayerCount = test.size();
	std::vector<float> weights;
	std::vector<float> layerSizes;

	// get first layer size
	layerSizes.push_back(testNetwork.numNeurons(0) - 1);
	// get the rest of the layer
	for (int i = 1; i < testNetwork.numLayers(); ++i)
	{
		NeuralNet<float>::Layer& prevLayer = testNetwork.GetLayer(i-1);
		NeuralNet<float>::Layer& curLayer = testNetwork.GetLayer(i);
		layerSizes.push_back(testNetwork.numNeurons(i) - 1);

		// for all neurons in current layer
		for (unsigned l = 0; l < curLayer.size() - 1; ++l) {
			unsigned index = curLayer[l].getIndex();
			for (size_t n = 0; n < prevLayer.size()-1; ++n)
				weights.push_back(prevLayer[n].getWeight(index));
		}
	}

	testNetwork.getNeuron(0, 0).setOutput(1.0f);
	testNetwork.getNeuron(0, 1).setOutput(0.0f);

	std::vector<float> inputVal;
	for (int i = 0; i < test[0]; ++i)
	{
		//get us float* vector; the input values
		inputVal.push_back(testNetwork.getNeuron(0, i).getOutput());
	}

	//get the sizes of the inputMatrix and weightMatrix's height.
	//weightMatrix size startVecLength * startMatHeight
	//going forward, we can push in test/config ptr into GPU then we access the length value directly from GPU call
	int startVecLength = layerSizes[0], startMatHeight = layerSizes[1];
	int prevVecLength = 0, prevMatHeight = 0;
	float maxLayerSize = *std::max_element(std::begin(layerSizes), std::end(layerSizes));

	//Weightsize is so as num of weights per layer = next layer's count of neurons * curr layer's count of neurons
	uint weightSize = maxLayerSize * maxLayerSize;
	float* h_inputCUDA, *h_weights, *h_finalOutputCUDA, *h_weightCUDA;
	float* d_inputCUDA, *d_hiddenLayerCUDA, *d_weights, *d_finalOutputCUDA;

	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);

	//pin memory for streams
	//h_input length is just the size of the layer
	checkCudaErrors(cudaHostAlloc((void **)&h_inputCUDA, maxLayerSize * maxLayerSize * sizeof(float), cudaHostAllocDefault));
	checkCudaErrors(cudaHostAlloc((void **)&h_weights, weights.size() * sizeof(float), cudaHostAllocDefault));
	checkCudaErrors(cudaHostAlloc((void **)&h_weightCUDA, weights.size() * sizeof(float), cudaHostAllocDefault));
	checkCudaErrors(cudaHostAlloc((void **)&h_finalOutputCUDA, maxLayerSize * maxLayerSize * sizeof(float), cudaHostAllocDefault));

	//ptr to the first element
	checkCudaErrors(cudaMemcpy(h_inputCUDA, &inputVal[0], inputVal.size() * sizeof(float), cudaMemcpyHostToHost));

	for (int i = 0; i < inputVal.size(); ++i)
	{
		std::cout << "inputVal[" << i << "] = " << inputVal[i] << std::endl;
		std::cout << "h_inputCUDA[" << i << "] = " << h_inputCUDA[i] << std::endl << std::endl;
	}

	//offset outselves to get correct set of weights
	checkCudaErrors(cudaMemcpy(h_weights, &weights[0], weights.size() * sizeof(float), cudaMemcpyHostToHost));

	for (int i = 0; i < weights.size(); ++i)
	{
		std::cout << "weights[" << i << "] = " << weights[i] << std::endl;
		std::cout << "h_weights[" << i << "] = " << h_weights[i] << std::endl << std::endl;
	}

	checkCudaErrors(cudaMalloc((void **)&d_inputCUDA, maxLayerSize * maxLayerSize * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_hiddenLayerCUDA, maxLayerSize * maxLayerSize * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_weights, weights.size() * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_finalOutputCUDA, maxLayerSize * maxLayerSize * sizeof(float)));

	//gpu set up for stream calling for feed forward
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);

	for (int iteration = 0; iteration < numIter; ++iteration)
	{
		uint rowPtsPerStream = ceilf((float)startVecLength / (float)numStream);
		uint elementsPerStream = rowPtsPerStream;
		uint elePerBatch;

		for (int j = 0; j < numStream; ++j)
		{
			int offset = elementsPerStream * j;
			if (j == numStream - 1)
			{
				elePerBatch = (startVecLength - (rowPtsPerStream * j));
			}
			else
			{
				elePerBatch = elementsPerStream;
			}
			//create Streams and copy data from host into device using streams
			cudaStreamCreate(&streams[j]);
			checkCudaErrors(cudaMemcpyAsync(d_inputCUDA + offset, h_inputCUDA + offset, elePerBatch * startVecLength * sizeof(float), cudaMemcpyHostToDevice, streams[j]));
		}

		for (int k = 0; k < numStream; ++k)
		{
			cudaStreamSynchronize(streams[k]);
		}

		//0 is input layer
		for (int i = 1; i < hiddenLayerCount; ++i)
		{
			//to iterate through the sizes for the layer to get the new vecLength and matHeight
			//i - 1 since we start from i = 1
			startVecLength = layerSizes[i - 1];
			startMatHeight = layerSizes[i];

			uint weightOffset = prevVecLength * prevMatHeight * (i - 1);

			checkCudaErrors(cudaMemset(d_hiddenLayerCUDA, 0, maxLayerSize * maxLayerSize * sizeof(float)));

			//uint largestOffset = inputHeight < startMatHeight ? startMatHeight : inputHeight;
			auto numberOfRows = (unsigned)ceilf(float(startVecLength) / BLOCKSIZEX);
			auto numOfCol = (unsigned)ceilf(float(startMatHeight) / BLOCKSIZEY);
			auto numOfDataset = (unsigned)ceilf(float(inputHeight) / BLOCKSIZEZ);
			dim3 gridSize(numberOfRows, numOfCol, numOfDataset);
			dim3 blockSize(BLOCKSIZEX, BLOCKSIZEY, BLOCKSIZEZ);
			/////

			uint weightsPerStream = ceilf((float)startMatHeight / (float)numStream);
			for (int j = 0; j < numStream; ++j)
			{
				int offset = startMatHeight * j;
				if (j == numStream - 1)
				{
					elePerBatch = (startMatHeight - (weightsPerStream * j));
				}
				else
				{
					elePerBatch = startMatHeight;
				}
				//update weights every loop
				checkCudaErrors(cudaMemcpyAsync(d_weights + offset, h_weights + offset + weightOffset, elePerBatch * startVecLength * sizeof(float), cudaMemcpyHostToDevice, streams[j]));
			}

			for (int k = 0; k < numStream; ++k)
			{
				cudaStreamSynchronize(streams[k]);
			}

			rowPtsPerStream = ceilf((float)(startVecLength - 2) / (float)numStream);
			uint amountPerBatch = rowPtsPerStream + 2;
			elePerBatch = (startVecLength - 2 - (rowPtsPerStream * (numStream - 1)));
			uint amountLastBatch = elePerBatch + 2;
			uint amountRows;

			for (int j = 0; j < numStream; ++j)
			{
				//shift away the ptr to point at correct elements
				int offset = rowPtsPerStream * j;
				if (j == numStream - 1)
				{
					amountRows = amountLastBatch;
				}
				else
				{
					amountRows = amountPerBatch;
				}
				//Kernel func call//
				feedForward << < gridSize, blockSize, 0, streams[j] >> > (d_inputCUDA + offset, d_weights + offset, d_hiddenLayerCUDA + offset, startVecLength, inputHeight, startMatHeight, 0);
				getLastCudaError("JI BA BOOM\n");
				////
			}

			for (int k = 0; k < numStream; ++k)
			{
				cudaStreamSynchronize(streams[k]);
			}

			float* justToSwap = d_inputCUDA;
			//swap ptr for next iter
			d_inputCUDA = d_hiddenLayerCUDA;
			d_hiddenLayerCUDA = justToSwap;

			prevVecLength = startVecLength;
			prevMatHeight = startMatHeight;
		}

		for (int j = 0; j < numStream; ++j)
		{
			if ((hiddenLayerCount % 2) == 0)
			{
				//d_finalOutputCUDA = d_hiddenLayerCUDA;
				checkCudaErrors(cudaMemcpyAsync(d_finalOutputCUDA, d_hiddenLayerCUDA, maxLayerSize * startVecLength * sizeof(float), cudaMemcpyDeviceToDevice, streams[j]));
			}
			else
			{
				//d_finalOutputCUDA = d_inputCUDA;
				checkCudaErrors(cudaMemcpyAsync(d_finalOutputCUDA, d_inputCUDA, maxLayerSize * startVecLength * sizeof(float), cudaMemcpyDeviceToDevice, streams[j]));
			}
			checkCudaErrors(cudaMemcpyAsync(h_finalOutputCUDA, d_finalOutputCUDA, maxLayerSize * startVecLength * sizeof(float), cudaMemcpyDeviceToHost, streams[j]));
		}

		for (int k = 0; k < numStream; ++k)
		{
			cudaStreamSynchronize(streams[k]);
			cudaStreamDestroy(streams[k]);
		}

		////

		//if ((hiddenLayerCount % 2) == 0)
		//{
		//  //d_finalOutputCUDA = d_hiddenLayerCUDA;
		//  checkCudaErrors(cudaMemcpy(d_finalOutputCUDA, d_hiddenLayerCUDA, maxLayerSize * startVecLength * sizeof(float), cudaMemcpyDeviceToDevice));
		//}
		//else
		//{
		//  //d_finalOutputCUDA = d_inputCUDA;
		//  checkCudaErrors(cudaMemcpy(d_finalOutputCUDA, d_inputCUDA, maxLayerSize * startVecLength * sizeof(float), cudaMemcpyDeviceToDevice));
		//}
		//checkCudaErrors(cudaMemcpy(h_finalOutputCUDA, d_finalOutputCUDA, maxLayerSize * startVecLength * sizeof(float), cudaMemcpyDeviceToHost));

		//checkCudaErrors(cudaMemcpy(h_weightCUDA, d_weights, weightSize * sizeof(float), cudaMemcpyDeviceToHost));
	}
	for (int i = 0; i < maxLayerSize * startVecLength; ++i)
	{
		std::cout << "h_finalOutputCUDA[" << i << "] = " << h_finalOutputCUDA[i] << std::endl << std::endl;
	}

	//for (int i = 0; i < weightSize; ++i)
	//{
	//  std::cout << "h_weightCUDA[" << i << "] = " << h_weightCUDA[i] << std::endl << std::endl;
	//}

	std::vector<std::vector<float>> inputs = {
	{ 1, 0 }
	//{ 1, 1 },
	//{ 1, 1 },
	//{ 1, 1 }
	};
	std::vector<float> labels = { 0 };
	std::vector<float> myOutput;
	testNetwork.SetOutputAtLayer(0, inputs[0]);
	testNetwork.Forward(1);
	testNetwork.Forward(2);
	testNetwork.GetOutputAtLayer(2, myOutput);

	std::cout << "Checking result:\n";
	bool result = true;
	for (int i = 0; i < myOutput.size(); ++i) {
		if (abs(h_finalOutputCUDA[i] - myOutput[i]) > EPSILON) {
			std::cout << "Result dont match at iteration " << i << "\n";
			std::cout << "h_finalOutputCUDA[" << i << "] = " << h_finalOutputCUDA[i] << std::endl;
			std::cout << "myOutput [" << i << "] = " << myOutput[i] << std::endl;
			result = false;
			//break;
		}
	}
	if (result) {
		std::cout << "Results match\n";
		//	std::cout << "Speedup: " << CPUTime / GPUTime << std::endl;
	}

	delete[] streams;
	//free cuda malloc memory
	checkCudaErrors(cudaFree(d_inputCUDA));
	checkCudaErrors(cudaFree(d_weights));
	checkCudaErrors(cudaFree(d_hiddenLayerCUDA));
	checkCudaErrors(cudaFree(d_finalOutputCUDA));

	//free pinned memory
	checkCudaErrors(cudaFreeHost(h_inputCUDA));
	checkCudaErrors(cudaFreeHost(h_weights));
	checkCudaErrors(cudaFreeHost(h_finalOutputCUDA));
#endif
}
