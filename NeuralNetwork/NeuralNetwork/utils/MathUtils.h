#pragma once
#include "../kernel.h"

#define EPSILON 1e-6

void cpu_VectorMatrixDotProduct(const float* vector, const float* matrix, float* output, uint length);