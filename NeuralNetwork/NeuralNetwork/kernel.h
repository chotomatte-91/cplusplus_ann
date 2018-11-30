#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "bmp.h"
#include <stdlib.h>
#include <stdio.h>
#include <helper_cuda.h>
#include <helper_functions.h>

using uint = unsigned int;

#define BLOCKSIZE 32

__global__ void simpleDotProduct(float* input, float* input2, float* output, uint length);

/*
	Parameter description:
	float* vector - array of float (1D) of size N
	float* matrix - array of float (2D) of size N
	float* output - array of float (1D) of size N
	uint length   - size N
*/
__global__ void dotProduct(float* __restrict__ vector, float* __restrict__ matrix, float* output, uint length);