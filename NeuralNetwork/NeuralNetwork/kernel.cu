#include "kernel.h"

__global__ float kernel_sigmoid(float x)
{
	// fast sigmoid function
	return x / (1 + abs(x));
}

__device__ float kernel_tanh(float x)
{
	return tanhf(x);
}

__device__ float kernel_Relu(float x)
{
	return x > 0 ? x : 0;
}

__global__ void simpleDotProduct(float* input, float* input2, float* output, uint length)
{
	float sum1 = 0.0f, sum2 = 0.0f;
	int index = threadIdx.x + blockDim.x * blockIdx.x;

	if (index < length) {
		sum1 = input[index];
		sum2 = input2[index];
	}

	// NAIVE WAY
	if (index < length)
	{
		float sum = sum1 * sum2;
		atomicAdd(output, sum);
	}
}

/*
	vectorIndex uses thread index x and is use to access vector
	mtxIndex make use of vectorIndex to compute it 2-dimensional position to access the matrix
*/
__global__ void dotProduct(float* vector, float* __restrict__ matrix, float* output, uint vectorLength, uint matrixlength)
{
	float __shared__ vectorShared[BLOCKSIZE];
	float __shared__ partialResult[BLOCKSIZE][BLOCKSIZE];
	int vecIndex = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;
	int mtxIndex = vecIndex + row * vectorLength;

	// load 2 element at once
	if (vecIndex < vectorLength && threadIdx.y == 0)
		vectorShared[threadIdx.x] = vector[vecIndex];
	__syncthreads();

	// parallel reduction scan
	if (vecIndex < vectorLength && row < matrixlength)
	{
		partialResult[threadIdx.y][threadIdx.x] = vectorShared[threadIdx.x] * matrix[mtxIndex];
		__syncthreads();

		// Complete unrolling
		int BlockDim = blockDim.x >> 1;
		partialResult[threadIdx.y][threadIdx.x] += (threadIdx.x < BlockDim && (vecIndex + BlockDim) < vectorLength) ?
			partialResult[threadIdx.y][threadIdx.x + BlockDim] : 0.0f;
		__syncthreads();

		BlockDim = blockDim.x >> 2;
		partialResult[threadIdx.y][threadIdx.x] += (threadIdx.x < BlockDim && (vecIndex + BlockDim) < vectorLength) ?
			partialResult[threadIdx.y][threadIdx.x + BlockDim] : 0.0f;
		__syncthreads();

		BlockDim = blockDim.x >> 3;
		partialResult[threadIdx.y][threadIdx.x] += (threadIdx.x < BlockDim && (vecIndex + BlockDim) < vectorLength) ?
			partialResult[threadIdx.y][threadIdx.x + BlockDim] : 0.0f;
		__syncthreads();

		BlockDim = blockDim.x >> 4;
		partialResult[threadIdx.y][threadIdx.x] += (threadIdx.x < BlockDim && (vecIndex + BlockDim) < vectorLength) ?
			partialResult[threadIdx.y][threadIdx.x + BlockDim] : 0.0f;
		__syncthreads();

		BlockDim = blockDim.x >> 5;
		partialResult[threadIdx.y][threadIdx.x] += (threadIdx.x < BlockDim && (vecIndex + BlockDim) < vectorLength) ?
			partialResult[threadIdx.y][threadIdx.x + BlockDim] : 0.0f;
		__syncthreads();

		//for (unsigned i = blockDim.x >> 1; i > 0; i >>= 1) {
		//	__syncthreads();
		//	if (threadIdx.x < i && (vecIndex + i) < length)
		//		partialResult[threadIdx.y][threadIdx.x] += partialResult[threadIdx.y][threadIdx.x + i];
		//}

		if (threadIdx.x == 0)
			atomicAdd(output + row, partialResult[threadIdx.y][threadIdx.x]);
	}

}

__global__ void feedForward(float* inputMatrix, float* __restrict__ weightMatrix,
	float* intermediateOutput, uint layerIndex, uint length)
{
	//__shared__ float inputShared[BLOCKSIZE];
	//__shared__ float partialResShared[BLOCKSIZE][BLOCKSIZE];
	//int vecIndex = threadIdx.x + blockDim.x * blockIdx.x;
	//int row      = threadIdx.y + blockDim.y * blockIdx.y;
	//int mtxIndex = vecIndex + row * length;

	//if (vecIndex < length && threadIdx.y == 0)
	//	inputShared[threadIdx.x] = inputMatrix[vecIndex + layerIndex * length];
	//__syncthreads();


	//// Parallel scan reduction
	//if (vecIndex < length && row < length)
	//{
	//	// step 1: do multiplication
	//	partialResShared[threadIdx.y][threadIdx.x] = inputShared[threadIdx.x] * weightMatrix[mtxIndex];
	//	
	//	// step2: parallel reduction to complete dot product
	//	for (unsigned i = 1; i <= BLOCKSIZE; i<<=1) {
	//		int index = (threadIdx.x + 1) * (i << 1) - 1;
	//		if (index < BLOCKSIZE)
	//			partialResShared[threadIdx.y][index] += partialResShared[threadIdx.y][index - i];
	//		__syncthreads();
	//	}
	//}
}


//#include <stdio.h>
//
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
//
//__global__ void addKernel(int *c, const int *a, const int *b)
//{
//    int i = threadIdx.x;
//    c[i] = a[i] + b[i];
//}
//
//int main()
//{
//    const int arraySize = 5;
//    const int a[arraySize] = { 1, 2, 3, 4, 5 };
//    const int b[arraySize] = { 10, 20, 30, 40, 50 };
//    int c[arraySize] = { 0 };
//
//    // Add vectors in parallel.
//    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addWithCuda failed!");
//        return 1;
//    }
//
//    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
//        c[0], c[1], c[2], c[3], c[4]);
//
//    // cudaDeviceReset must be called before exiting in order for profiling and
//    // tracing tools such as Nsight and Visual Profiler to show complete traces.
//    cudaStatus = cudaDeviceReset();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceReset failed!");
//        return 1;
//    }
//
//    return 0;
//}
//
//// Helper function for using CUDA to add vectors in parallel.
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
//{
//    int *dev_a = 0;
//    int *dev_b = 0;
//    int *dev_c = 0;
//    cudaError_t cudaStatus;
//
//    // Choose which GPU to run on, change this on a multi-GPU system.
//    cudaStatus = cudaSetDevice(0);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//        goto Error;
//    }
//
//    // Allocate GPU buffers for three vectors (two input, one output)    .
//    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    // Copy input vectors from host memory to GPU buffers.
//    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    // Launch a kernel on the GPU with one thread for each element.
//    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);
//
//    // Check for any errors launching the kernel
//    cudaStatus = cudaGetLastError();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//        goto Error;
//    }
//    
//    // cudaDeviceSynchronize waits for the kernel to finish, and returns
//    // any errors encountered during the launch.
//    cudaStatus = cudaDeviceSynchronize();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
//        goto Error;
//    }
//
//    // Copy output vector from GPU buffer to host memory.
//    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//Error:
//    cudaFree(dev_c);
//    cudaFree(dev_a);
//    cudaFree(dev_b);
//    
//    return cudaStatus;
//}
