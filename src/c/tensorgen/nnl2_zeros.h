#ifndef NNL2_ZEROS_H
#define NNL2_ZEROS_H

/** @brief 
 * Creates a new tensor filled with ones
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
 ** @return
 * Pointer to the newly allocated tensor filled with ones, or NULL on failure
 *
 ** @note 
 * The returned tensor must be freed using nnl2_free_tensor() to avoid memory leaks
 *
 ** @note 
 * For optimal performance with floating-point types, ensure proper memory alignment
 *
 ** @note
 * May conduct additional checks depending on the safety level
 *
 ** @example
 * // Create a 3x3 matrix of ones with float32 type
 * int shape[] = {3, 3};
 * Tensor* ones_matrix = nnl2_ones(shape, 2, FLOAT32); 
 *
 ** @exception NNL2Error
 * Shape is NULL or rank is invalid
 *
 ** @exception NNL2Error
 * New tensor allocation fails
 *
 ** @exception NNL2Error
 * Unsupported data type is specified
 *
 ** @see nnl2_empty()
 ** @see nnl2_zeros()
 ** @see inplace_fill()
 **/
Tensor* nnl2_zeros(int32_t* shape, int32_t rank, TensorType dtype) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
		NNL2_FUNC_ENTER();
	#endif
	
	// Additional validation of input parameters in maximal safety level
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		if (shape == NULL || rank <= 0) {
			NNL2_ERROR("Invalid shape or rank in ones");
			return NULL;
		}
	#endif
	
	// Creating an empty tensor with a specified shape and data type
    Tensor* tensor_t = nnl2_empty(shape, rank, dtype);
	
	// Checking the success of tensor creation
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if (tensor_t == NULL) {
			NNL2_ERROR("Failed to allocate empty tensor");
			return NULL;
		}
	#endif

	bool success;

	// Filling the tensor with units depending on the data type
    switch(dtype) {
        case INT32:    success = inplace_fill(tensor_t, &(int32_t){0}, dtype);   break;
        case FLOAT32:  success = inplace_fill(tensor_t, &(float){0.0f}, dtype);  break;     
        case FLOAT64:  success = inplace_fill(tensor_t, &(double){0.0}, dtype);  break;

		// Processing unsupported data types
        default: {
            NNL2_TYPE_ERROR(dtype);
            nnl2_free_tensor(tensor_t);  
            return NULL;
        }
    }
	
	if(!success) {
		NNL2_ERROR("Function completed failed");
	}
    
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
		NNL2_FUNC_EXIT();
	#endif
	
    return tensor_t;
}

#endif /** NNL2_ZEROS_H **/
