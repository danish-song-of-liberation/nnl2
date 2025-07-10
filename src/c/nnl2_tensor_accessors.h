#include <stdlib.h>
#include <string.h>
#include <stddef.h>

#include "nnl2_core.h"
#include "nnl2_tensor_backend.h"

#ifndef NNL2_TENSOR_ACCESSORS
#define NNL2_TENSOR_ACCESSORS

// NNL2

/**
 * @file nnl2_tensor_accessors.h
 * @brief The fastest possible operations with tensors (without error checking)
 */

/**
 * @brief Gets the size of a tensor element in bytes for a given type
 * @param dtype Tensor data type (INT32, FLOAT32, FLOAT64)
 * @return The size of one element in bytes
 * @warning Does not check dtype validity (array bounds may be exceeded)
 */
inline size_t get_dtype_size(TensorType dtype) {
	static const size_t sizes[] = {sizeof(int), sizeof(float), sizeof(double)};
		
	return sizes[dtype]; // Unsafe but fast
}	

/**
 * @brief Calculates the product of the array elements
 * @param lst pointer to array of dimension sizes
 * @param len number of dimensions
 * @return Total number of elements in the tensor
 * @note No overflow checks (aggresive optimization)
 */
inline size_t product(const int* lst, int len) {
	size_t acc = 1;
	
	while (len--) acc *= *lst++;
	
	return acc;
}

/**
 * @brief Creates a new tensor
 * @param shape - an array of dimensions for each dimension
 * @param rank - number of dimensions 
 * @param dtype - data type of the tensor's elements
 * @return Pointer to created tensor
 * @warning Does not perform checks
 * @note The calling code must ensure that the parameters are correct
 */
Tensor* fast_make_tensor(const int* shape, int rank, TensorType dtype) {
	Tensor* tensor = malloc(sizeof(Tensor));
	
	tensor->rank = rank;
	tensor->dtype = dtype;
	
	tensor->shape = malloc(rank * sizeof(int));
	memcpy(tensor->shape, shape, rank * sizeof(int));
	
	size_t total_elems = product(shape, rank);
	tensor->data = malloc(total_elems * get_dtype_size(dtype));
	
	return tensor;
}

Tensor* make_tensor(const int* shape, int rank, TensorType dtype) {
	Tensor* tensor = malloc(sizeof(Tensor));
	
	tensor->rank = rank;
	tensor->dtype = dtype;
	
	tensor->shape = malloc(rank * sizeof(int));
	memcpy(tensor->shape, shape, rank * sizeof(int));
	
	size_t total_elems = product(shape, rank);
	tensor->data = calloc(total_elems, get_dtype_size(dtype));
	
	return tensor;
}

void* at(Tensor* tensor, const int32_t* indices, uint8_t sum_indices) {
    // THE CLOCK IS TICKING... TICKING...
	
	if (sum_indices == tensor->rank) {
		size_t offset = 0;
		size_t stride = 1;
		
		for(uint32_t i = tensor->rank; i-- > 0;) {
			offset += indices[i] * stride;
			stride *= tensor->shape[i];
		}
		
		switch(tensor->dtype) {
			case INT32:   return (int32_t*)tensor->data + offset;
			case FLOAT32: return (float*)tensor->data + offset;
			case FLOAT64: return (double*)tensor->data + offset;
			default: return NULL;
		}
	} else if (sum_indices < tensor->rank) {
		Tensor* subtensor = (Tensor*)malloc(sizeof(Tensor));
		
		subtensor->dtype = tensor->dtype;
		subtensor->rank = tensor->rank - sum_indices;
		subtensor->shape = (int*)malloc(subtensor->rank * sizeof(int));
		
		memcpy(subtensor->shape, tensor->shape + sum_indices, subtensor->rank * sizeof(int32_t));
		
		size_t offset = 0;
		size_t stride = 1;
		
		for(uint32_t i = sum_indices; i-- > 0;) {
			offset += indices[i] * stride;
			stride *= tensor->shape[i];
		}
		
		subtensor->data = (uint8_t*)tensor->data + offset * get_dtype_size(tensor->dtype);
		
		return subtensor;
	} else {
		fprintf(stderr, "Incorrent indices");
		
		return NULL;
	}
}

void get_tensor_metadata(Tensor* t, int* dtype, int* rank, int** shape) {
    *dtype = t->dtype;
    *rank = t->rank;
    *shape = t->shape;
}

#endif
