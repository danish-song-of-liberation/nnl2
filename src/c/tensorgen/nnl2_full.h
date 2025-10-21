#ifndef NNL2_FULL_H
#define NNL2_FULL_H

/** @brief 
 * Creates a new tensor filled with a specified value
 *
 ** @param shape
 * Pointer to integer array specifying the dimensions of the tensor
 *
 ** @param rank 
 * Number of dimensions (length of shape array)
 *
 ** @param dtype 
 * Data type of the tensor elements
 *
 ** @param filler
 * Pointer to the value used to fill the tensor. Must match the specified data type
 * For example: for FLOAT32, pass pointer to float; for INT32, pass pointer to int32_t
 *
 ** @return 
 * Pointer to the newly allocated tensor filled with specified value, or NULL on failure
 *
 ** @note
 * The returned tensor must be freed using nnl2_free_tensor() to avoid memory leaks
 *
 ** @note
 * The filler parameter must point to a valid value of the correct type matching dtype
 *
 ** @note
 * For optimal performance with floating-point types, ensure proper memory alignment
 *
 ** @note
 * May conduct additional checks depending on the safety level configuration
 *
 ** @example
 * // Create a 2x3 matrix filled with 5.5 (float32)
 * int shape[] = {2, 3};
 * float value = 5.5f;
 * Tensor* filled_tensor = nnl2_full(shape, 2, FLOAT32, &value);
 *
 * // Create a 1D vector filled with 42 (int32)
 * int shape1d[] = {10};
 * int32_t int_value = 42;
 * Tensor* int_tensor = nnl2_full(shape1d, 1, INT32, &int_value);
 *
 ** @exception NNL2_ERROR_SHAPE_INVALID
 * Shape is NULL or contains invalid dimensions
 *
 ** @exception NNL2_ERROR_RANK_INVALID
 * Rank is zero or negative
 *
 ** @exception NNL2_ERROR_FILLER_NULL
 * Filler pointer is NULL
 *
 ** @exception NNL2_ERROR_ALLOCATION_FAILED
 * Tensor allocation fails
 *
 ** @exception NNL2_ERROR_UNSUPPORTED_TYPE
 * Unsupported data type is specified
 *
 ** @see inplace_fill()
 **/
Tensor* nnl2_full(const int* shape, int rank, TensorType dtype, void* filler) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
		NNL2_FUNC_ENTER();
	#endif
	
	// Additional validation of input parameters in maximal safety level
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX 
		if (!shape || rank <= 0 || !filler) {
			NNL2_ERROR("Invalid tensor structure in full");
			return NULL;
		}
		
		// Validate each dimension of the shape
		for (int i = 0; i < rank; i++) {
			if (shape[i] <= 0) {
				NNL2_ERROR("Invalid shape dimension at index %d: %d", i, shape[i]);
				return NULL;
			}
		}
	#endif

	// Create empty tensor with specified shape and data type
	Tensor* tensor_t = nnl2_empty(shape, rank, dtype);
	
	// Check if tensor creation was successful
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		if (!tensor_t) {
			fprintf(stderr, "Failed to allocate tensor\n");
			return NULL;
		}
	#endif
	
	// Fill the tensor with the specified value
	if(!inplace_fill(tensor_t, filler, dtype)) {
		// Error handle
		NNL2_ERROR("Function completed failed");
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
		NNL2_FUNC_EXIT();
	#endif
	
	return tensor_t;
}

#endif
