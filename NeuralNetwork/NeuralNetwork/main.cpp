#include "kernel.h"
#include "utils\MathUtils.h"
#include "ann/neuralnet.h"
#include <cmath>
#include <algorithm>
#include <random>

#include <iostream>

//#define PRINT_FUNC_USED

//activation functions
template <typename T>
T mysigmoid(T val)
{
#ifdef PRINT_FUNC_USED
  std::cout << "Executing sigmoid function" << std::endl;
#endif
  assert(val > static_cast<T>(-709));
	T denom = static_cast<T>(1.f + std::exp(-val));
	return (1 / denom);
}

template <typename T>
T mysigmoidPrime(T val)
{
#ifdef PRINT_FUNC_USED
  std::cout << "Executing sigmoid derivative" << std::endl;
#endif
	T temp = mysigmoid(val);
	return temp * (1 - temp);
}

template <typename T>
T mytanH(T val)
{
#ifdef PRINT_FUNC_USED
  std::cout << "Executing tanh function" << std::endl;
#endif
	return static_cast<T>(std::tanh((double)val));
}

template <typename T>
T mytanHPrime(T val)
{
#ifdef PRINT_FUNC_USED
  std::cout << "Executing tanh derivative" << std::endl;
#endif
  T temp = mytanH(val);
  return static_cast<T>(1.f - (temp * temp));
}

//error functions
template <typename T>
T mean_squared_error(const std::vector<T>& predicted, const std::vector<T>& expected)
{
  assert(predicted.size() == expected.size());
  size_t N = expected.size();

  T error = T();
  for (size_t i = 0; i < N; ++i)
  {
    T d = expected[i] - predicted[i];
    error += (d * d);
  }
  return (error / (T)(2 * N));
}

#define TYPE float
void ann_cpu_test()
{
	//two neurons input, two neurons hidden, one neuron output
	std::vector<unsigned> config{ 2, 2, 1 };
	NeuralNet<TYPE> nn(config);
  nn.setErrorFunction(mean_squared_error);

	//hardcoded XOR input and labels
	/*XOR TABLE
	0 0 0
	0 1 1
	1 0 1
	1 1 0
	*/

	std::vector<std::vector<TYPE>> inputs = {
	{ 0, 0 },
	{ 0, 1 },
	{ 1, 0 },
	{ 1, 1 }
	};
	std::vector<TYPE> labels = { 0, 1, 1, 0 };

	//hardcoded initialization of weights and bias for testing
	float i1_h1 = (TYPE) -0.7706f;
	float i2_h1 = (TYPE) 0.6257f;
	float h1_bias = (TYPE) 0.1859f;
	float i1_h2 = (TYPE) 0.5607f;
	float i2_h2 = (TYPE) 0.2109f;
	float h2_bias = (TYPE)-0.7984f;
	float h1_o1 = (TYPE) 0.5951f;
	float h2_o1 = (TYPE) 0.3433f;
	float o1_bias = (TYPE) 0.1328f;

	//set weights

	//input layer to first neuron in hidden layer
	nn.getNeuron(0, 0).setWeight(0, i1_h1);
	nn.getNeuron(0, 1).setWeight(0, i2_h1);
	nn.getNeuron(0, 2).setWeight(0, h1_bias);

	//input layer to second neuron in hidden layer
	nn.getNeuron(0, 0).setWeight(1, i1_h2);
	nn.getNeuron(0, 1).setWeight(1, i2_h2);
	nn.getNeuron(0, 2).setWeight(1, h2_bias);

	//hidden to output layer
	nn.getNeuron(1, 0).setWeight(0, h1_o1);
	nn.getNeuron(1, 1).setWeight(0, h2_o1);
	nn.getNeuron(1, 2).setWeight(0, o1_bias);

	//tanh is activation function for hidden layer
	nn.getNeuron(1, 0).setActivationFunctions(mytanH, mytanHPrime);
	nn.getNeuron(1, 1).setActivationFunctions(mytanH, mytanHPrime);
  nn.getNeuron(1, 2).setActivationFunctions(mytanH, mytanHPrime);

	//sigmoid is activation function for output layer
	nn.getNeuron(2, 0).setActivationFunctions(mysigmoid, mysigmoidPrime);

  /* expected outputs after 10000 iteration alpha = 0.5f
  std::vector<float> expectedOutputs = { 0.0169, 0.9782, 0.9782, 0.0150 };
  hidden layer weights:
  [0.1785, -0.7730, 0.6199], [0.7984, 0.5606, 0.2083]
  [0.1303, 0.5908, 0.3435]
  h1_bias = 0.1785
  h2_bias = -0.7984
  o1_bias = 0.1303
  i1_h1 = -0.7730
  i2_h1 = 0.6199
  i1_h2 = 0.5606
  i2_h2 = 0.2083
  h1_o1 = 0.5908
  h2_o1 = 0.3435
  */

	nn.train(inputs, labels, 0.5f, 3000) ;
  nn.printWeights();
}


int main(int argc, char **argv)
{
	ann_cpu_test();
#if 0
	if (argc < 6) {
		std::cout << "Usage: [filename] [stream] [layers] [maxheight]\n";
		return -1;
	}

	const uint inputHeight = 1;
	const uint numStream = std::atoi(argv[2]);
	const uint numIter = std::atoi(argv[3]);
	const uint numLays = std::atoi(argv[4]);
	const uint maxHeight = std::atoi(argv[5]);

	cudaStream_t* streams;
	streams = new cudaStream_t[numStream];
	cudaStreamCreateWithFlags(streams, cudaStreamNonBlocking);
	float GPUTime = 0.0f;
	float CPUTime = 0.0f;

	std::cout << "Reading file...\n";

	// Read input from file
	bmp_header header;
	unsigned char* imageData;
	std::string name(argv[1]);
	bool suc = bmp_read(argv[1], &header, &imageData);
	int imageWidth = header.width;
	int imageHeight = header.height;
	int imageChannels = 3;
	uint byteCount = imageWidth * imageHeight;
	
	if (suc) std::cout << "File: " << name << std::endl;
	else return -1;

	std::cout << "Pre-process neuro network model...\n";
	std::vector<float> inputVal;
	inputVal.reserve(byteCount);
	for (int i = 0; i < byteCount; ++i)
		inputVal.push_back((float)imageData[i] / 255.0f);

	std::random_device rd;
	std::mt19937 engine(rd());
	std::uniform_real_distribution<> dist(1, 5);

	// generate strings of random number
	std::vector<uint> stringLayer;
	stringLayer.reserve(numLays);
	stringLayer.push_back(byteCount);
	for (uint i = 1; i < numLays -1; ++i)
		stringLayer.push_back(dist(engine));
	stringLayer.push_back(1);

	// Init Neural network
	NeuralNet<float> testNetwork(stringLayer);

	// Set input neurons
	for (int i = 0; i < inputVal.size(); ++i)
		testNetwork.getNeuron(0, i).setOutput(inputVal[i]);

	int hiddenLayerCount = stringLayer.size();
	std::vector<float> weights;
	std::vector<unsigned> layerSizes;

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
			for (size_t n = 0; n < prevLayer.size() - 1; ++n)
				weights.push_back(prevLayer[n].getWeight(index));
		}
	}

	std::cout << "Setting up for GPU computation...\n\n\n";
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

	//for (int i = 0; i < inputVal.size(); ++i)
	//{
	//	std::cout << "inputVal[" << i << "] = " << inputVal[i] << std::endl;
	//	std::cout << "h_inputCUDA[" << i << "] = " << h_inputCUDA[i] << std::endl << std::endl;
	//}

	//offset outselves to get correct set of weights
	checkCudaErrors(cudaMemcpy(h_weights, &weights[0], weights.size() * sizeof(float), cudaMemcpyHostToHost));

	//for (int i = 0; i < weights.size(); ++i)
	//{
	//	std::cout << "weights[" << i << "] = " << weights[i] << std::endl;
	//	std::cout << "h_weights[" << i << "] = " << h_weights[i] << std::endl << std::endl;
	//}

	//gpu set up for stream calling for feed forward
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);
	checkCudaErrors(cudaMalloc((void **)&d_inputCUDA, maxLayerSize * maxLayerSize * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_hiddenLayerCUDA, maxLayerSize * maxLayerSize * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_weights, weights.size() * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_finalOutputCUDA, maxLayerSize * maxLayerSize * sizeof(float)));


	std::cout << "Running GPU...\n";
	for (int iteration = 0; iteration < numIter; ++iteration)
	{
		uint rowPtsPerStream = ceilf((float)startVecLength / (float)numStream);
		uint elementsPerStream = rowPtsPerStream;
		uint elePerBatch;

		for (uint j = 0; j < numStream; ++j)
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

		for (uint k = 0; k < numStream; ++k)
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
			for (uint j = 0; j < numStream; ++j)
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

			for (uint j = 0; j < numStream; ++j)
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
				////
			}

			for (uint k = 0; k < numStream; ++k)
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



		for (uint j = 0; j < numStream; ++j)
			checkCudaErrors(cudaMemcpyAsync(h_finalOutputCUDA, ((hiddenLayerCount % 2) == 0) ? d_hiddenLayerCUDA : d_inputCUDA,
							maxLayerSize * startVecLength * sizeof(float), cudaMemcpyDeviceToHost, streams[j]));

		for (uint k = 0; k < numStream; ++k)
		{
			cudaStreamSynchronize(streams[k]);
			cudaStreamDestroy(streams[k]);
		}

		sdkStopTimer(&hTimer);
		GPUTime += 1.0e-3 * (double)sdkGetTimerValue(&hTimer);
	}::cout << "h_finalOutputCUDA[" << i << "] = " << h_finalOutputCUDA[i] << std::endl << std::endl;
	//}
	std::cout << "GPU time taken on average: " << (GPUTime / (float)numIter) << std::endl << std::endl;

	//std::cout << "h_finalOutputCUDA[0] = " << h_finalOutputCUDA[0] << std::endl << std::endl;
	//for (int i = 0; i < maxLayerSize * startVecLength; ++i)
	//{
	//	std
	//for (int i = 0; i < weightSize; ++i)
	//{
	//  std::cout << "h_weightCUDA[" << i << "] = " << h_weightCUDA[i] << std::endl << std::endl;
	//}

	std::cout << "Running CPU...\n";
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);
	std::vector<float> myOutput;
	std::vector<std::vector<float>> inpu;
	std::vector<float> labels{ 0.0f };
	inpu.push_back(inputVal);

	// simulate iteration
	for (unsigned i = 1; i < stringLayer.size(); ++i)
		testNetwork.train(inpu, labels, 0.5f, numIter); //.Forward(i)

	//testNetwork.GetOutputAtLayer(2, myOutput);

	sdkStopTimer(&hTimer);
	CPUTime += 1.0e-3 * (double)sdkGetTimerValue(&hTimer);
	std::cout << "CPU time taken on average: " << (CPUTime / (float)numIter) << std::endl << std::endl;

	//std::cout << "Checking forwarded result:\n";
	//bool result = true;
	//for (int i = 0; i < myOutput.size(); ++i) {
	//	if (abs(h_finalOutputCUDA[i] - myOutput[i]) > EPSILON) {
	//		std::cout << "Result dont match at iteration " << i << "\n";
	//		std::cout << "h_finalOutputCUDA[" << i << "] = " << h_finalOutputCUDA[i] << std::endl;
	//		std::cout << "myOutput [" << i << "] = " << myOutput[i] << std::endl;
	//		std::cout << "Difference = " << abs(h_finalOutputCUDA[i] - myOutput[i]) << std::endl;
	//		result = false;
	//		break;
	//	}
	//}
	//if (result) {
	//	std::cout << "Results match\n";
	//}
	
	std::cout << "Speedup: " << CPUTime / GPUTime << std::endl;
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
