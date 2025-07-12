#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include <errno.h>
#include <stdio.h> 

#ifdef _WIN32

#include <malloc.h>

#define ALLOC_ALIGNED(ptr, alignment, size) \
    do { \
        ptr = _aligned_malloc(size, alignment); \
        if (ptr == NULL) { fprintf(stderr, "Error (Hello from C!): Failed to allocate memory\n"); } \
    } while(0)
#define FREE_ALIGNED(ptr) _aligned_free(ptr)

#else
	
#define _POSIX_C_SOURCE 200809L

#include <stdlib.h> 

#define ALLOC_ALIGNED(ptr, alignment, size)  \
	do { \
        int err = posix_memalign(&ptr, alignment, size); \
        if (err != 0) { fprintf(stderr, "Error (Hello from C!): Failed to allocate memory\n"); } \
    } while(0)
#define FREE_ALIGNED(ptr) free(ptr)

#endif

#include "nnl2_core.h"
#include "nnl2_tensor_backend.h"

#ifndef NNL2_TENSOR_ACCESSORS
#define NNL2_TENSOR_ACCESSORS

#define NUM_TENSOR_TYPES 3 // int32, float32, float64

// NNL2

/**
 * @file nnl2_tensor_accessors.h
 * @brief The fastest possible operations with tensors (without error checking)
 */

/**
 * @brief Gets the size of a tensor element in bytes for a given type
 * @param dtype Tensor data type (INT32, FLOAT32, FLOAT64)
 * @return The size of one element in bytes
 * @warning Does not check dtype validity (array bounds may be exceeded) but creating tensors already checks this
 */
inline size_t get_dtype_size(TensorType dtype) {
	static const size_t sizes[] = {sizeof(int), sizeof(float), sizeof(double)};
		
	return sizes[dtype]; 
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
 * @brief Creates a new tensor with trash value (but faster!)
 * @param shape - an array of dimensions for each dimension
 * @param rank - number of dimensions 
 * @param dtype - data type of the tensor's elements
 * @return Pointer to created tensor
 * @warning Does not perform checks
 * @note The calling code must ensure that the parameters are correct. otherwise, you'll end up with something very strange that somehow works
 */
Tensor* fast_make_tensor(const int* shape, int rank, TensorType dtype) {
	/** @brief
     * organicist technospecialization, pedagogical 
	 * authoritarianism, and territorial sectorization 
	 * end in numerical illiteracy and mass innumeracy
	 * (nick land)
	 */
	 
	if (shape == NULL) {
        fprintf(stderr, "Error (Hello from C!): Bad shape pointer\n");
        return NULL;
    } 
	
	if (rank <= 0) {
		fprintf(stderr, "Error (Hello from C!): Bad rank (%d). Rank must be positive\n", rank);
		return NULL;
	}
	
	for (int i = 0; i < rank; i++) {
        if (shape[i] <= 0) {
            fprintf(stderr, "Error (Hello from C!): Bad shape dimension at %d (%d). Dimensions must be positive\n",  i, shape[i]);
            return NULL;
        }
    }
	
	if (dtype < 0 || dtype >= NUM_TENSOR_TYPES) {
        fprintf(stderr, "Error (Hello from C!): Bad tensor type (%d)\n", dtype);
        return NULL;
    }
	
	Tensor* tensor = malloc(sizeof(Tensor));
	
	if (tensor == NULL) {
        fprintf(stderr, "Error (Hello from C!): failed to allocate memory for Tensor structure\n");
        return NULL;
    }
	
	tensor->rank = rank;
	tensor->dtype = dtype;
	
	tensor->shape = malloc(rank * sizeof(int));
	
	if (tensor->shape == NULL) {
        fprintf(stderr, "Error (Hello from C!): failed to allocate memory for shape\n");
        free(tensor);
        return NULL;
    }
	
	memcpy(tensor->shape, shape, rank * sizeof(int));
	
	size_t total_elems = product(shape, rank);
	size_t type_size = get_dtype_size(dtype);
	size_t total_size = total_elems * type_size;
	
	if (total_elems == 0) {
        fprintf(stderr, "Error (Hello from C!): total elements calculation resulted in 0\n");
        free(tensor->shape);
        free(tensor);
        return NULL;
    }
	
	void* data;
	
	ALLOC_ALIGNED(data, (size_t)64, total_size);
	
	if (data == NULL) {
        fprintf(stderr, "Error (Hello from C!): failed to allocate aligned memory\n");
        free(tensor->shape);
        free(tensor);
        return NULL;
    }
	
	tensor->data = data;
	
	return tensor;
}

/**
 * @brief Creates a new tensor with checks
 * @param shape - an array of dimensions for each dimension
 * @param rank - number of dimensions 
 * @param dtype - data type of the tensor's elements
 * @return Pointer to created tensor
 */
Tensor* make_tensor(const int* shape, int rank, TensorType dtype) {
	/** @brief
	 *
	 * imagine a biological creature with a fully developed 
	 * nervous system, including nociceptors.
	 *
	 * Now imagine that, similar to artificial neural networks, 
	 * this creature's physical pain is measured on a scale from 0.0 to 1.0.
	 *
	 * if 0.0 - no pain
	 * 0.1 - incredibly painful (comparable to being shot)
	 * 1.0 - oh my god
	 *
	 * what if we set the pain level in all the nerves to 0.5
	 * and prevent the creature from losing consciousness, going crazy, 
	 * or reducing the pain naturally (through neurochemistry)?
	 *
	 * replace the pain with pleasure.
	 * 0.0 - lack of pleasure
	 * 0.1 - the happiest moment in life
	 * 1.0 - absolute form
	 *
	 * now tell me which is worse, 0.5 on eternal pain or 0.5 on eternal pleasure?
	 *
	 */
	
	if (shape == NULL) {
        fprintf(stderr, "Error (Hello from C!): Bad shape pointer\n");
        return NULL;
    }
	
	if (rank <= 0) {
		fprintf(stderr, "Error (Hello from C!): Bad rank (%d). Rank must be positive\n", rank);
		return NULL;
	}
	
	for (int i = 0; i < rank; i++) {
        if (shape[i] <= 0) {
            fprintf(stderr, "Error (Hello from C!): Bad shape dimension at %d (%d). Dimensions must be positive\n",  i, shape[i]);
            return NULL;
        }
    }
	
	if (dtype < 0 || dtype >= NUM_TENSOR_TYPES) {
        fprintf(stderr, "Error (Hello from C!): Bad tensor type (%d)\n", dtype);
        return NULL;
    }
	
	Tensor* tensor = malloc(sizeof(Tensor));
	
	if (tensor == NULL) {
        fprintf(stderr, "Error (Hello from C!): failed to allocate memory for Tensor structure\n");
        return NULL;
    }
	
	tensor->rank = rank;
	tensor->dtype = dtype;
	
	tensor->shape = malloc(rank * sizeof(int));
	
	if (tensor->shape == NULL) {
        fprintf(stderr, "Error (Hello from C!): failed to allocate memory for shape\n");
        free(tensor);
        return NULL;
    }
	
	memcpy(tensor->shape, shape, rank * sizeof(int));
		
	size_t total_elems = product(shape, rank);
	
	if (total_elems == 0) {
        fprintf(stderr, "Error (Hello from C!): total elements calculation resulted in 0\n");
        free(tensor->shape);
        free(tensor);
        return NULL;
    }
	
	size_t type_size = get_dtype_size(dtype);
	
	if (type_size == 0) {
        fprintf(stderr, "Error (Hello from C!): invalid dtype size (0)\n");
        free(tensor->shape);
        free(tensor);
        return NULL;
    }
	
	size_t total_size = total_elems * type_size;
	
	void* data;
	
	ALLOC_ALIGNED(data, (size_t)64, total_size);
	
	if (data == NULL) {
        fprintf(stderr, "Error (Hello from C!): failed to allocate aligned memory\n");
        free(tensor->shape);
        free(tensor);
        return NULL;
    }
	
	tensor->data = data;
	memset(tensor->data, 0, total_size);
	
	return tensor;
}

/**
 * @brief Accesses an element or creates a subtensor within a tensor.
 * @param tensor - Pointer to the target tensor
 * @param indices - An array of integer(pointer) indices used to specify the element or subtensor
 * @param sum_indices - Length of indices
 * @return Element or subtensor
 * @warning Does not perform checks (for performance)
 */
void* fast_at(Tensor* tensor, const int32_t* indices, uint8_t sum_indices) {
	const int32_t tensor_rank = tensor->rank;
	
	if (sum_indices == tensor_rank) {
		size_t offset = 0;
		size_t stride = 1;
		
		for(uint32_t i = tensor_rank; i-- > 0;) {
			offset += indices[i] * stride;
			stride *= tensor->shape[i];
		}
		
		switch(tensor->dtype) {
			case INT32:   return (int32_t*)tensor->data + offset;
			case FLOAT32: return (float*)tensor->data + offset;
			case FLOAT64: return (double*)tensor->data + offset;
			default: return NULL;
		}
	} else if (sum_indices < tensor_rank) {
		Tensor* subtensor = (Tensor*)malloc(sizeof(Tensor));
		
		subtensor->dtype = tensor->dtype;
		subtensor->rank = tensor_rank - sum_indices;
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
		fprintf(stderr, "Error (Hello from C!): Incorrent indices\n");
		
		return NULL;
	}
}

/**
 * @brief Accesses an element or creates a subtensor within a tensor.
 * @param tensor - Pointer to the target tensor
 * @param indices - An array of integer(pointer) indices used to specify the element or subtensor
 * @param sum_indices - Length of indices
 * @return Element or subtensor
 */
void* at(Tensor* tensor, const int32_t* indices, uint8_t sum_indices) {
	if (tensor == NULL) {
        fprintf(stderr, "Error (Hello from C!): Bad tensor pointer\n");
        return NULL;
    }
	
	if (indices == NULL) {
        fprintf(stderr, "Error (Hello from C!): Bad indices pointer\n");
        return NULL;
    }
	
	const int32_t tensor_rank = tensor->rank;
	
	if (tensor_rank <= 0) {
		fprintf(stderr, "Error (Hello from C!): Bad rank (%d). Rank must be positive\n", tensor_rank);
		return NULL;
	}	
	
	for (int i = 0; i < tensor_rank; i++) {
        if (indices[i] < 0) {
            fprintf(stderr, "Error (Hello from C!): Bad shape dimension at %d (%d). Dimensions must be positive\n",  i, indices[i]);
            return NULL;
        }
    }
	
	if (sum_indices == tensor_rank) {
		size_t offset = 0;
		size_t stride = 1;
		
		for(uint32_t i = tensor_rank; i-- > 0;) {
			offset += indices[i] * stride;
			stride *= tensor->shape[i];
		}
		
		switch(tensor->dtype) {
			case INT32:   return (int32_t*)tensor->data + offset;
			case FLOAT32: return (float*)tensor->data + offset;
			case FLOAT64: return (double*)tensor->data + offset;
			default: return NULL;
		}
	} else if (sum_indices < tensor_rank) {
		Tensor* subtensor = (Tensor*)malloc(sizeof(Tensor));
		
		if (subtensor == NULL) {
            fprintf(stderr, "Error (Hello from C!): Failed to allocate subtensor\n");
            return NULL;
        }
		
		subtensor->dtype = tensor->dtype;
		subtensor->rank = tensor_rank - sum_indices;
		subtensor->shape = (int*)malloc(subtensor->rank * sizeof(int));
		
		if (subtensor->shape == NULL) {
            fprintf(stderr, "Error (Hello from C!): Failed to allocate subtensor shape\n");
            free(subtensor);
            return NULL;
        }
		
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
		fprintf(stderr, "Error (Hello from C!): Incorrent indices\n");
		
		return NULL;
	}
}

void get_tensor_metadata(Tensor* t, int* dtype, int* rank, int** shape) {
	/** @brief
	 * in fact, my fascination with ML and framework creation
	 * is nothing more than an act of selfishness and escapism
	 */
	
    *dtype = t->dtype;
    *rank = t->rank;
    *shape = t->shape;
}

void function_of_liberation(Tensor* tensor) {
	free(tensor->shape); // Well, see? Even the death of a tensor is a liberation
	FREE_ALIGNED(tensor->data); // their goal has been achieved, and their will to live is fading
	
	/** @brief
	 * life in general is a miserable, miserable thing, 
	 * it has always been miserable and unhappy, and it will always 
	 * be miserable and unhappy, and nonexistence is better than existence 
	 * (Philipp Mainländer)
	 */
}

void debug_tensor(Tensor tensor) {
	printf("Hello from C to Lisp with love!\nIt is debugging for the output value directly from C.\n\n");
	
	size_t total_elems = product(tensor.shape, tensor.rank);
	
	printf("Total elements: %zu\n\n", total_elems);
	
	double* data = (double*)tensor.data;
	
	for (size_t it = 0; it < total_elems; it++) {
		int indices[] = {it};
		double* tensor_tref = (double*)at(&tensor, indices, 1);
		
		printf("Element %zu (converted to double) according to index: %f; according to tref: %f\n", it + 1, data[it], *tensor_tref);
	}
}

#endif
