#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "kernel.h"


int main(int argc, char **argv)
{
	StopWatchInterface *hTimer = NULL;
	cudaDeviceProp deviceProp;
	deviceProp.major = 0;
	deviceProp.minor = 0;

	//Use command-line specified CUDA device, otherwise use device with highest Gflops/s
	int dev = findCudaDevice(argc, (const char **)argv);

	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
	cudaGetDevice(&dev);

	printf("CUDA device [%s] has %d Multi-Processors, Compute %d.%d\n",
		deviceProp.name, deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);


}