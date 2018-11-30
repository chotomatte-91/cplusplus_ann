#include "MathUtils.h"

void cpu_VectorMatrixDotProduct(const float * vector, const float * matrix, 
								float * output, uint length)
{
	for (unsigned i = 0; i < length; ++i) {
		for (unsigned j = 0; j < length; ++j) 
			output[i] += vector[j] * matrix[j + i * length];
	}
}
