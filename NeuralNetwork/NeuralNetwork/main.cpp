#include "kernel.h"
#include "ann/neuralnet.h"
#include <cmath>

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
  std::vector<unsigned> config{2, 2, 1};
  NeuralNet nn(config);

  Neuron& neuron = nn.getNeuron(1, 1);
  float y = neuron.getOutput();
}


int main(int argc, char **argv)
{
  ann_cpu_test();
#if 0
  ann_cpu_test();
	if (argc != 2) printf("Usage: [image file]");

	StopWatchInterface *hTimer = NULL;
	cudaDeviceProp deviceProp;
	deviceProp.major = 0;
	deviceProp.minor = 0;

	// For this ANN, we are only dealing with 3 layers
	//bmp_header header;
	//unsigned char* imageData;
	//float* h_input;
	//float* d_inLayer, d_outLayer;

	//Use command-line specified CUDA device, otherwise use device with highest Gflops/s
	int dev = findCudaDevice(argc, (const char **)argv);

	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
	cudaGetDevice(&dev);

	printf("CUDA device [%s] has %d Multi-Processors, Compute %d.%d\n",
		deviceProp.name, deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

	//bmp_read(argv[1], &header, &imageData);
	//int imageWidth = header.width;
	//int imageHeight = header.height;
	//int imageChannels = 3;
	//double dAvgSecs;
	//uint byteCount = imageWidth * imageHeight * imageChannels * sizeof(unsigned char);

	//// convert char to float
	//h_input = new float[byteCount];
	//for (int i = 0; i < byteCount; ++i)
	//	h_input[i] = (float)imageData[i];
	//delete[] h_input;

	/*
		--------------------
		BELOW IS ALL TESTING
		--------------------
	*/

	// YANWEN TEST
	float *d_arr1, *d_arr2, *h_arr1, *h_arr2;
	float *d_output;
	const uint testSize = 100000;
	h_arr1 = new float[testSize];
	h_arr2 = new float[testSize];

	for (int i = 0; i < testSize; ++i)
		h_arr1[i] = (float)((i + 3) % 8) * 0.5f;
	
	float ii = -1.0f;
	for (int i = testSize - 1; i > -1; --i) {
		h_arr2[i] = (float)((i - 3) % 5) * ii;
		ii *= ii;
	}

	checkCudaErrors(cudaMalloc((void **)&(d_arr1), testSize * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&(d_arr2), testSize * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&(d_output), sizeof(float)));
	checkCudaErrors(cudaMemcpy(d_arr1, h_arr1, testSize * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_arr2, h_arr2, testSize * sizeof(float), cudaMemcpyHostToDevice));

	dim3 gridSize(ceilf(float(testSize - 1) / BLOCKSIZE), 1, 1);
	dim3 blockSize(BLOCKSIZE, BLOCKSIZE, 1);

	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);

	simpleDotProduct <<< gridSize, blockSize >>>(d_arr1, d_arr2, d_output, testSize);
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemcpy(h_arr1, d_output, sizeof(float), cudaMemcpyDeviceToHost));

	sdkStopTimer(&hTimer);

	std::cout << h_arr1[0] << std::endl;

	checkCudaErrors(cudaFree(d_arr1));
	checkCudaErrors(cudaFree(d_arr2));
	checkCudaErrors(cudaFree(d_output));
	delete[] h_arr1;
	delete[] h_arr2;
#endif
}