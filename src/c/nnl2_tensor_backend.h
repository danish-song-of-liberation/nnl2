#include "nnl2_core.h"

#ifndef NNL2_TENSOR_BACKEND
#define NNL2_TENSOR_BACKEND

// NNL2

/**
 * @file nnl2_tensor_backend.h
 * @brief Contains the tensor and implemenets structures
 */

/**
 * @brief Enumerations of available tensor types (INT32, FLOAT32/FLOAT, FLOAT64/DOUBLE)
 */
typedef enum {
	INT32,
	FLOAT32,
	FLOAT64
} TensorType;

/**
 * @brief Tensor structure with an arbitrary number of dimensions
 * 		  - dtype : Type of tensor (INT32, FLOAT32/FLOAT, FLOAT64/DOUBLE)
 *		  - data : Void pointer to tensor data
 * 		  - shape : Pointer to an int array of tensor dimensions
 * 		  - rank : The number of dimensions of the tensor
 */
typedef struct {
	TensorType dtype;
	void* data;
	int* shape;
	int rank;
} Tensor;

/**
 * @brief Structure for creating an implementation of a linear algebra function (e.t.c. matmul)
 * 		  - fn: A voidtype pointer to a function
 *		  - speed: Priority for speed, the higher the better
 *		  - available: Flag for availability of implementation
 * 		  - name: Name of implementation
 */
typedef struct {
	void* fn;
	int speed;
	bool available;
	const char* name;
} Implementation;

#endif
