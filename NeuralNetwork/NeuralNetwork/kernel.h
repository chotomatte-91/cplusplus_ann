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
