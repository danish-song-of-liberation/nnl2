#ifndef NNL2_SUM_H
#define NNL2_SUM_H

/** @brief
 * Computes the sum of all elements in a tensor without axis reduction
 * 
 ** @param tensor 
 * Pointer to the input tensor structure
 *
 ** @param result 
 * Pointer to the memory where the sum result will be stored
 */
void naive_sum_without_axis(Tensor* tensor, void* result) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	size_t total_elems = product(tensor->shape, tensor->rank);
	if(total_elems == 0) return; // If tensor if empty then return empty result
		
	switch(tensor->dtype) {
		case FLOAT64: {
            double* cast_data = (double*)tensor->data;
            double acc = 0.0;
            for (size_t it = 0; it < total_elems; it++) acc += cast_data[it];
            *((double*)result) = acc; // Store result in output pointer
            break; 
        }
	
		case FLOAT32: {
            float* cast_data = (float*)tensor->data;
            float acc = 0.0f;
            for (size_t it = 0; it < total_elems; it++) acc += cast_data[it];
            *((float*)result) = acc; 
            break;
        }
			
		case INT32: {
            int32_t* cast_data = (int32_t*)tensor->data;
            int32_t acc = 0;
            for (size_t it = 0; it < total_elems; it++) acc += cast_data[it];
            *((int32_t*)result) = acc; 
            break;
        }
			
		default: {
			NNL2_TYPE_ERROR(tensor->dtype);
			return;
		}
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for sum without axis operation
 * @details
 * Array follows the common backend registration pattern for element-wise
 * summation operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for tensor summation
 * 
 * @see nnl2_naive
 * @see naive_sum_without_axis
 */
Implementation sum_without_axis_backends[] = {
    REGISTER_BACKEND(naive_sum_without_axis, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for sum without axis operation
 * @ingroup backend_system 
 */
sumwithoutaxisfn nnl2_sum_without_axis;

/** 
 * @brief Makes the sum without axis backend current
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(sum_without_axis);

/** 
 * @brief Sets the backend for sum without axis operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for summation
 * @see ESET_BACKEND_BY_NAME
 */
void set_sum_without_axis_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(sum_without_axis_backends, nnl2_sum_without_axis, backend_name, CURRENT_BACKEND(sum_without_axis));
}

/** 
 * @brief Gets the name of the active backend for sum without axis operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_sum_without_axis_backend() {
    return CURRENT_BACKEND(sum_without_axis);
}

/** 
 * @brief Function declaration for getting all available sum without axis backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(sum_without_axis);

/**
 * @brief Function declaration for getting the number of available sum without axis backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(sum_without_axis);

/** @brief
 * Calculates the source index in the tensor for the axis-wise summation operation
 *
 ** @param tensor
 * Original tensor
 *
 ** @param result
 * Result tensor
 *
 ** @param axis
 * The axis along which the summation is performed
 *
 ** @param result_index
 * Index in the resulting tensor
 * 
 ** @param axis_index
 * Position along the summation axis
 * 
 ** @return 
 * Calculated index in the original tensor
 */
NNL2_FORCE_INLINE static size_t nnl2_naive_calculate_original_index_for_sum_with_axis(Tensor* tensor, Tensor* result, int axis, size_t result_index, int axis_index);

/** @brief
 * Performs axis summation for a FLOAT64 tensor
 *
 ** @param tensor
 * Original tensor with double (float64) type
 *
 ** @param result
 * The preallocated resulting tensor
 *
 ** @param axis
 * Sum axis (0-based)
 *
 ** @param result_numel
 * The number of elements in the result
 *
 ** @param elements_along_axis
 * The number of elements along the summation axis
 *
 ** @return
 * Tensor* Pointer to the resulting tensor
 */
NNL2_FORCE_INLINE static Tensor* nnl2_naive_sum_with_axis_float64(Tensor* tensor, Tensor* result, int axis, size_t result_numel, int elements_along_axis);

/** @brief
 * The documentation is identical to the 
 * nnl2_naive_sum_with_axis_float64 but with the float32 type
 *
 ** @see naive_sum_with_axis_float64
 **/
NNL2_FORCE_INLINE static Tensor* nnl2_naive_sum_with_axis_float32(Tensor* tensor, Tensor* result, int axis, size_t result_numel, int elements_along_axis);

/** @brief
 * The documentation is identical to the 
 * nnl2_naive_sum_with_axis_float64 but with the int32 type
 *
 ** @see naive_sum_with_axis_float64
 **/
NNL2_FORCE_INLINE static Tensor* nnl2_naive_sum_with_axis_int32(Tensor* tensor, Tensor* result, int axis, size_t result_numel, int elements_along_axis);

/** @brief
 * Computes the sum of tensor elements along the specified axis
 *
 ** @param tensor
 * Input tensor to perform summation on
 *
 ** @param axis
 * Axis along which to sum (0-based index)
 *
 ** @return
 * Tensor* New tensor with reduced rank containing the sums
 */
Tensor* naive_sum_with_axis(Tensor* tensor, int axis) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	// Validate axis parameter
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if (axis < 0 || axis >= tensor->rank) {
			NNL2_ERROR("Invalid axis %d for tensor of rank %d", axis, tensor->rank);
			return NULL;
		}
	#endif
    
	// Allocate memory for result shape (rank reduced by 1)
    int* result_shape = malloc((tensor->rank - 1) * sizeof(int));
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(result_shape == NULL) {
			NNL2_ERROR("Failed to allocate memory");
			return NULL;
		}
	#endif
	
	// Construct result shape by excluding the summation axis
    int j = 0;
    for (int i = 0; i < tensor->rank; i++) {
        if (i != axis) {
            result_shape[j++] = tensor->shape[i];
        }
    }

    // Create empty result tensor with appropriate shape and dtype
    Tensor* result = nnl2_empty(result_shape, tensor->rank - 1, tensor->dtype);
    free(result_shape);
    
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(result == NULL) {
			NNL2_ERROR("Failed to create result tensor");
			return NULL;
		}
	#endif
	
	// Select appropriate summation function based on data type
	Tensor* (*sum_func)(Tensor*, Tensor*, int, size_t, int) = NULL;
	
	switch(tensor->dtype) {
        case FLOAT32: {
            sum_func = nnl2_naive_sum_with_axis_float32;
            break;
		}
		
        case FLOAT64: {
            sum_func = nnl2_naive_sum_with_axis_float64;
            break;
		}
		
        case INT32: {
            sum_func = nnl2_naive_sum_with_axis_int32;
            break;
		}
		
        default: {
            NNL2_TYPE_ERROR(tensor->dtype);
            nnl2_free_tensor(result);
            return NULL;
		}
    }
    
	// Calculate dimensions for the summation operation
    size_t result_numel = product(result->shape, result->rank);
    int elements_along_axis = tensor->shape[axis];
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
	// Perform the actual summation
    return sum_func(tensor, result, axis, result_numel, elements_along_axis);
}

/** @brief
 * See docs at declaration
 *
 ** @see naive_calculate_original_index_for_sum_with_axis
 **/
NNL2_FORCE_INLINE static size_t nnl2_naive_calculate_original_index_for_sum_with_axis(Tensor* tensor, Tensor* result, int axis, size_t result_index, int axis_index) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
	
	size_t original_index = 0;
    int temp = result_index;
    
	// Reconstruct original coordinates from flattened result index
    for (int dim = result->rank - 1; dim >= 0; dim--) {
        int coord = temp % result->shape[dim];
        temp /= result->shape[dim];

		// Adjust dimension index to account for removed axis
        int original_dim = dim;
        if (original_dim >= axis) {
            original_dim++; 
        }
        
		// Add contribution from this dimension using original strides
        original_index += coord * tensor->strides[original_dim];
    }
    
	// Add contribution from the summation axis
    original_index += axis_index * tensor->strides[axis];
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
	
    return original_index;
}

/** @brief
 * See docs at declaration
 *
 ** @see naive_sum_with_axis_float64
 **/
NNL2_FORCE_INLINE static Tensor* nnl2_naive_sum_with_axis_float64(Tensor* tensor, Tensor* result, int axis, size_t result_numel, int elements_along_axis) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
	
    double* data = (double*)tensor->data;
    double* result_data = (double*)result->data;
    
	// Iterate over all elements in the result tensor
    for (size_t i = 0; i < result_numel; i++) {
        double sum = 0.0;
		
		// Sum along the specified axis
        for (int k = 0; k < elements_along_axis; k++) {
            size_t original_index = nnl2_naive_calculate_original_index_for_sum_with_axis(tensor, result, axis, i, k);
            sum += data[original_index];
        }
		
		// Store the computed sum
        result_data[i] = sum;
    }
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
    
    return result;
}

/** @brief
 * See docs at declaration
 *
 ** @see naive_sum_with_axis_float32
 **/
NNL2_FORCE_INLINE static Tensor* nnl2_naive_sum_with_axis_float32(Tensor* tensor, Tensor* result, int axis, size_t result_numel, int elements_along_axis) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
	
    float* data = (float*)tensor->data;
    float* result_data = (float*)result->data;
    
	// Iterate over all elements in the result tensor
    for (size_t i = 0; i < result_numel; i++) {
        float sum = 0.0f;
		
		// Sum along the specified axis
        for (int k = 0; k < elements_along_axis; k++) {
            size_t original_index = nnl2_naive_calculate_original_index_for_sum_with_axis(tensor, result, axis, i, k);
            sum += data[original_index];
        }
		
		// Store the computed sum
        result_data[i] = sum;
    }
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
    
    return result;
}

/** @brief
 * See docs at declaration
 *
 ** @see naive_sum_with_axis_int32
 **/
NNL2_FORCE_INLINE static Tensor* nnl2_naive_sum_with_axis_int32(Tensor* tensor, Tensor* result, int axis, size_t result_numel, int elements_along_axis) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
	
	int32_t* data = (int32_t*)tensor->data;
    int32_t* result_data = (int32_t*)result->data;
    
	// Iterate over all elements in the result tensor
    for (size_t i = 0; i < result_numel; i++) {
        int32_t sum = 0;
		
		// Sum along the specified axis
        for (int k = 0; k < elements_along_axis; k++) {
            size_t original_index = nnl2_naive_calculate_original_index_for_sum_with_axis(tensor, result, axis, i, k);
            sum += data[original_index];
        }
		
		// Store the computed sum
        result_data[i] = sum;
    }
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
    
    return result;
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for sum with axis operation
 * @details
 * Array follows the common backend registration pattern for axis-wise
 * summation operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for tensor summation along specified axes
 * 
 * @see nnl2_naive
 * @see naive_sum_with_axis
 */
Implementation sum_with_axis_backends[] = {
    REGISTER_BACKEND(naive_sum_with_axis, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for sum with axis operation
 * @ingroup backend_system 
 * 
 * This function pointer is used to call the currently active backend
 * implementation for summing tensor elements along specified dimensions.
 */
sumwithaxisfn nnl2_sum_with_axis;

/** 
 * @brief Makes the sum with axis backend current
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(sum_with_axis);

/** 
 * @brief Sets the backend for sum with axis operation
 * @ingroup backend_system
 * @param[in] backend_name Name of the backend to activate for axis-wise summation
 * @see ESET_BACKEND_BY_NAME
 * 
 * This function allows dynamic switching between different backend implementations
 * for the sum with axis operation at runtime.
 */
void set_sum_with_axis_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(sum_with_axis_backends, nnl2_sum_with_axis, backend_name, CURRENT_BACKEND(sum_with_axis));
}

#endif /** NNL2_SUM_H **/
