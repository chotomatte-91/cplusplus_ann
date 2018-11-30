#include "kernel.h"
#include "utils\MathUtils.h"

void ann_cpu_test()
{

}


int main(int argc, char **argv)
{
  //ann_cpu_test();
#if 1
	//if (argc != 2) {
	//	printf("Usage: [image file]");
	//	return -1;
	//}

	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);
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

	/*
		----------------------------
		VECTOR TO MATRIX DOT PRODUCT
		----------------------------
	*/
	std::cout << "Starting GPU vector to matrix dot product: \n";
	std::cout << "Length: " << std::atoi(argv[1]) << std::endl;

	// YANWEN TEST
	float *d_arr1, *d_arr2, *h_arr1, *h_arr2;
	float *d_output, *h_output, *dh_output;
	const uint vmSize = std::atoi(argv[1]);
	const uint debugMode = std::atoi(argv[2]);
	h_arr1 = new float[vmSize];
	h_arr2 = new float[vmSize * vmSize];
	h_output = new float[vmSize];
	dh_output = new float[vmSize];

	for (int i = 0; i < vmSize; ++i) {
		h_arr1[i] = debugMode == 1 ? 1 : 1.0f / (i + 1);
		h_output[i] = 0.0f;
	}
	for (int i = 0; i < vmSize * vmSize; ++i)
		h_arr2[i] = 1;

	checkCudaErrors(cudaMalloc((void **)&(d_arr1), vmSize * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&(d_arr2), vmSize * vmSize * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&(d_output), vmSize * sizeof(float)));
	checkCudaErrors(cudaMemcpy(d_arr1, h_arr1, vmSize * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_arr2, h_arr2, vmSize * vmSize * sizeof(float), cudaMemcpyHostToDevice));

	auto numberOfRows = (unsigned) ceilf(float(vmSize) / BLOCKSIZE);
	dim3 gridSize(numberOfRows, numberOfRows, 1);
	dim3 blockSize(BLOCKSIZE, BLOCKSIZE, 1);
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);

	dotProduct <<< gridSize, blockSize >>>(d_arr1, d_arr2, d_output, vmSize);
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemcpy(dh_output, d_output, vmSize * sizeof(float), cudaMemcpyDeviceToHost));

	sdkStopTimer(&hTimer);
	float GPUTime = 1.0e-3 * (double)sdkGetTimerValue(&hTimer);
	std::cout << "GPU time taken: " << GPUTime << std::endl << std::endl;


	std::cout << "Starting CPU vector to matrix dot product: \n";
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);
	cpu_VectorMatrixDotProduct(h_arr1, h_arr2, h_output, vmSize);
	sdkStopTimer(&hTimer);
	float CPUTime = 1.0e-3 * (double)sdkGetTimerValue(&hTimer);
	std::cout << "CPU time taken: " << CPUTime << std::endl << std::endl;

	std::cout << "Checking result:\n";
	bool result = true;
	for (int i = 0; i < vmSize; ++i) {
		if (abs(dh_output[i] - h_output[i]) > EPSILON) {
			std::cout << "Result dont match at iteration " << i << "\n";
			std::cout << "dh_output[" << i << "] = " << dh_output[i] << std::endl;
			std::cout << "h_output [" << i << "] = " << h_output[i] << std::endl;
			std::cout << "difference: " << dh_output[i] - h_output[i] << std::endl;
			result = false;
			break;
		}
	}
	if (result) {
		std::cout << "Results match\n";
		std::cout << "Speedup: " << CPUTime / GPUTime << std::endl;
	}


	// Clean up
	checkCudaErrors(cudaFree(d_arr1));
	checkCudaErrors(cudaFree(d_arr2));
	checkCudaErrors(cudaFree(d_output));
	delete[] h_arr1;
	delete[] h_arr2;
	delete[] h_output;
#endif
}