#include "kernel.h"
#include "utils\MathUtils.h"
#include "ann/neuralnet.h"
#include <cmath>

	/*
	----------------------------
	VECTOR TO MATRIX DOT PRODUCT
	---------------------------- */
int main(int argc, char **argv)
{
	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);
	cudaDeviceProp deviceProp;
	deviceProp.major = 0;
	deviceProp.minor = 0;

	//Use command-line specified CUDA device, otherwise use device with highest Gflops/s
	int dev = findCudaDevice(argc, (const char **)argv);

	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
	cudaGetDevice(&dev);

	printf("CUDA device [%s] has %d Multi-Processors, Compute %d.%d\n",
		deviceProp.name, deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

	std::cout << "Starting GPU vector to matrix dot product: \n";
	std::cout << "Length: " << std::atoi(argv[1]) << std::endl;

	// YANWEN TEST
	float *d_arr1, *d_arr2, *h_arr1, *h_arr2;
	float *d_output, *h_output, *dh_output;
	const uint inputSize = std::atoi(argv[1]);
	const uint inputHeight = std::atoi(argv[2]);
	const uint matrixHeight = std::atoi(argv[3]);
	h_arr1 = new float[inputSize * inputHeight];
	h_arr2 = new float[inputSize * matrixHeight];
	h_output = new float[inputHeight * matrixHeight];
	dh_output = new float[inputHeight * matrixHeight];

	for (int i = 0; i < inputSize*inputHeight; ++i)
		h_arr1[i] = 2;
	
	for(int i = 0; i < inputSize * matrixHeight; ++i)
		h_arr2[i] = 1;

	for (int i = 0; i < matrixHeight * inputHeight; ++i)
		h_output[i] = 0.0f;


	checkCudaErrors(cudaMalloc((void **)&(d_arr1), inputSize * inputHeight * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&(d_arr2), inputSize * matrixHeight * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&(d_output), matrixHeight * inputHeight * sizeof(float)));
	checkCudaErrors(cudaMemcpy(d_arr1, h_arr1, inputSize * inputHeight * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_arr2, h_arr2, inputSize * matrixHeight * sizeof(float), cudaMemcpyHostToDevice));

	auto numberOfRows = (unsigned)ceilf(float(inputSize) / BLOCKSIZEX);
	auto numOfCol = (unsigned)ceilf(float(matrixHeight) / BLOCKSIZEY);
	auto numOfDataset = (unsigned)ceilf(float(inputHeight) / BLOCKSIZEZ);
	dim3 gridSize(numberOfRows, numOfCol, numOfDataset);
	dim3 blockSize(BLOCKSIZEX, BLOCKSIZEY, BLOCKSIZEZ);
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);

	//dotProduct << < gridSize, blockSize >> > (d_arr1, d_arr2, d_output, inputSize);
	feedForward << < gridSize, blockSize >> > (d_arr1, d_arr2, d_output, inputSize, inputHeight, matrixHeight, 0);
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemcpy(dh_output, d_output, inputHeight * matrixHeight * sizeof(float), cudaMemcpyDeviceToHost));

	sdkStopTimer(&hTimer);
	float GPUTime = 1.0e-3 * (double)sdkGetTimerValue(&hTimer);
	std::cout << "GPU time taken: " << GPUTime << std::endl << std::endl;

	for (int i = 0; i < inputHeight; ++i) {
		for (int j = 0; j < matrixHeight; ++j)
			std::cout << dh_output[j + matrixHeight * i] << " ";
		std::cout << "\n";
	}

	//std::cout << "Starting CPU vector to matrix dot product: \n";
	//sdkResetTimer(&hTimer);
	//sdkStartTimer(&hTimer);
	//cpu_MatrixDotProduct(h_arr1, h_arr2, h_output, inputSize, 12, 12);
	//sdkStopTimer(&hTimer);
	//float CPUTime = 1.0e-3 * (double)sdkGetTimerValue(&hTimer);
	//std::cout << "CPU time taken: " << CPUTime << std::endl << std::endl;

	//std::cout << "Checking result:\n";
	//bool result = true;
	//for (int i = 0; i < inputSize * 12; ++i) {
	//	if (abs(dh_output[i] - h_output[i]) > EPSILON) {
	//		std::cout << "Result dont match at iteration " << i << "\n";
	//		std::cout << "dh_output[" << i << "] = " << dh_output[i] << std::endl;
	//		std::cout << "h_output [" << i << "] = " << h_output[i] << std::endl;
	//		std::cout << "difference: " << dh_output[i] - h_output[i] << std::endl;
	//		result = false;
	//		break;
	//	}
	//}
	//if (result) {
	//	std::cout << "Results match\n";
	//	std::cout << "Speedup: " << CPUTime / GPUTime << std::endl;
	//}

	// Clean up
	checkCudaErrors(cudaFree(d_arr1));
	checkCudaErrors(cudaFree(d_arr2));
	checkCudaErrors(cudaFree(d_output));
	delete[] h_arr1;
	delete[] h_arr2;
	delete[] h_output;
#endif
}
