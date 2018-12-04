#include "kernel.h"

__device__ float kernel_identity(float x)
{
	return x;
}

__device__ float kernel_sigmoid(float x)
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


__device__ float(*acvt[4])(float) 
{
	kernel_identity,
	kernel_sigmoid,
	kernel_tanh,
	kernel_Relu
};


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
									SAMPLE / TEST CODE
	vectorIndex uses thread index x and is use to access vector
	mtxIndex make use of vectorIndex to compute it 2-dimensional position to access the matrix
*/
__global__ void dotProduct(float* vector, float* __restrict__ matrix, float* output, uint length)
{
	float __shared__ vectorShared[BLOCKSIZE];
	float __shared__ partialResult[BLOCKSIZE][BLOCKSIZE];
	int vecIndex = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;
	int mtxIndex = vecIndex + row * length;

	// load input element into shared memory
	if (vecIndex < length && threadIdx.y == 0)
		vectorShared[threadIdx.x] = vector[vecIndex];
	__syncthreads();

	// parallel reduction scan
	if (vecIndex < length && row < length)
	{
		partialResult[threadIdx.y][threadIdx.x] = vectorShared[threadIdx.x] * matrix[mtxIndex];
		__syncthreads();

		// Complete unrolling
		int BlockDim = blockDim.x >> 1;
		partialResult[threadIdx.y][threadIdx.x] += (threadIdx.x < BlockDim && (vecIndex + BlockDim) < length) ?
			partialResult[threadIdx.y][threadIdx.x + BlockDim] : 0.0f;
		__syncthreads();

		BlockDim = blockDim.x >> 2;
		partialResult[threadIdx.y][threadIdx.x] += (threadIdx.x < BlockDim && (vecIndex + BlockDim) < length) ?
			partialResult[threadIdx.y][threadIdx.x + BlockDim] : 0.0f;
		__syncthreads();

		BlockDim = blockDim.x >> 3;
		partialResult[threadIdx.y][threadIdx.x] += (threadIdx.x < BlockDim && (vecIndex + BlockDim) < length) ?
			partialResult[threadIdx.y][threadIdx.x + BlockDim] : 0.0f;
		__syncthreads();

		BlockDim = blockDim.x >> 4;
		partialResult[threadIdx.y][threadIdx.x] += (threadIdx.x < BlockDim && (vecIndex + BlockDim) < length) ?
			partialResult[threadIdx.y][threadIdx.x + BlockDim] : 0.0f;
		__syncthreads();

		BlockDim = blockDim.x >> 5;
		partialResult[threadIdx.y][threadIdx.x] += (threadIdx.x < BlockDim && (vecIndex + BlockDim) < length) ?
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

/*
	Recommended block size:
	(16 * 16 * 4) more balance approached (maybe)
	(8 * 8 * 16) dataset size is larger
	(4 * 4 * 64) dataset size is large and no. of neuron small

	x and y controls how many neurons in input & hidden
	layer. Z controls how many different set of inputs(*fnc)

	NOTE: Block size shoud not be more than 10x10x10 cuz it will exceed total limit of cuda cores in GTX 1060 GPU
*/
__global__ void feedForward(float* __restrict__ inputMatrix, float* __restrict__ weightMatrix, float* output,
							uint inputLength, uint inputHeight, uint matrixHeight, uint acvtFnc) /* NOTE: matrixLength = inputlength */
{
	/*
		Variable setup
	*/
	float __shared__ inputShared[BLOCKSIZEZ][BLOCKSIZEX];
	float __shared__ partialResult[BLOCKSIZEZ][BLOCKSIZEY][BLOCKSIZEX];
	int vecIndex   = threadIdx.x + blockDim.x * blockIdx.x;
	int yRow       = threadIdx.y + blockDim.y * blockIdx.y;
	int zRow       = threadIdx.z + blockDim.z * blockIdx.z;
	int inputIndex = vecIndex + zRow * inputLength;
	int mtxIndex   = vecIndex + yRow * inputLength;

	/* 
		load input elements into shared memory
		inputShared is a a 3d shared memory that
		contains different sets of input, each set
		of input is stored in different z index
		each input element is applied with an activation
		function from the previous layer (deferred activation)
	*/
	if (vecIndex < inputLength && zRow < inputHeight && threadIdx.y == 0) {
		inputShared[threadIdx.z][threadIdx.x] = acvt[acvtFnc](inputMatrix[inputIndex]);
	}
	__syncthreads();


	// parallel reduction scan
	if (vecIndex < inputLength && yRow < matrixHeight && zRow < inputHeight)
	{
		// compute dot product for each row in input matrix into the 3d shared memory
		partialResult[threadIdx.z][threadIdx.y][threadIdx.x] = inputShared[threadIdx.z][threadIdx.x] * weightMatrix[mtxIndex];
		__syncthreads();

		// Complete unrolling for reduction of blockSize (16 * 16 * 4) where x and z = 16 * 16
		int BlockDim = blockDim.x >> 1;
		partialResult[threadIdx.z][threadIdx.y][threadIdx.x] += (threadIdx.x < BlockDim && (vecIndex + BlockDim) < inputLength) ?
			partialResult[threadIdx.z][threadIdx.y][threadIdx.x + BlockDim] : 0.0f;
		__syncthreads();

		BlockDim = blockDim.x >> 2;
		partialResult[threadIdx.z][threadIdx.y][threadIdx.x] += (threadIdx.x < BlockDim && (vecIndex + BlockDim) < inputLength) ?
			partialResult[threadIdx.z][threadIdx.y][threadIdx.x + BlockDim] : 0.0f;
		__syncthreads();

		BlockDim = blockDim.x >> 3;
		partialResult[threadIdx.z][threadIdx.y][threadIdx.x] += (threadIdx.x < BlockDim && (vecIndex + BlockDim) < inputLength) ?
			partialResult[threadIdx.z][threadIdx.y][threadIdx.x + BlockDim] : 0.0f;
		__syncthreads();

		BlockDim = blockDim.x >> 4;
		partialResult[threadIdx.z][threadIdx.y][threadIdx.x] += (threadIdx.x < BlockDim && (vecIndex + BlockDim) < inputLength) ?
			partialResult[threadIdx.z][threadIdx.y][threadIdx.x + BlockDim] : 0.0f;
		__syncthreads();


		int newOffset = yRow + zRow * inputHeight;
		if (threadIdx.x == 0)
			atomicAdd(output + newOffset, partialResult[threadIdx.z][threadIdx.y][threadIdx.x]);
	}
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
