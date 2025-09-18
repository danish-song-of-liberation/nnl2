#ifndef NNL2_MAX_H
#define NNL2_MAX_H

/** @brief
 * Performs element-wise maximum of two tensors (naive implementation)
 *
 ** @details
 * The function creates a new tensor containing the result of taking the maximum
 * of each element from the first tensor and the corresponding element in the second tensor.
 * It supports various data types with automatic casting to a higher type in the hierarchy.
 *
 ** @param tensora
 * Pointer to the first tensor
 *
 ** @param tensorb
 * Pointer to the second tensor
 *
 ** @return 
 * Pointer to a new tensor with the maximum values
 *
 ** @note
 * Uses volatile pointers to prevent compiler optimizations
 *
 ** @note
 * Returns NULL in case of failure or unsupported data type
 *
 ** @note
 * The result tensor has the same shape as input tensors and the highest data type
 *
 ** @see nnl2_empty
 ** @see get_dtype_size()
 ** @see nnl2_convert_to_float64()
 ** @see nnl2_convert_to_float32()
 ** @see nnl2_convert_to_int32()
 **/
Tensor* naive_max(const Tensor* tensora, const Tensor* tensorb) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	// Additional checks on maximum safety level
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensora, "First tensor is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensora->data, "First tensor data is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensorb, "Second tensor is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensorb->data, "Second tensor data is NULL", NULL);
	#endif
	
    // Calculate the total number of elements in the tensors
    size_t len = product(tensora->shape, tensora->rank);
    
    TensorType dtype_a = tensora->dtype;
    TensorType dtype_b = tensorb->dtype;
    
    // Selecting the winning type (higher in the hierarchy)
    TensorType winner_in_the_type_hierarchy = MAX(dtype_a, dtype_b);

    // Create an output tensor with the same shape and winning data type
    Tensor* result = nnl2_empty(tensora->shape, tensora->rank, winner_in_the_type_hierarchy);
    
    if (result == NULL) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return NULL;
    }
    
    if (len == 0) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return result;
    }
    
    if (dtype_a == dtype_b) {
        // Handling the case if the data types match
        
        switch (dtype_a) {
            case FLOAT64: {
                volatile double* data_a = (double*)tensora->data;
                volatile double* data_b = (double*)tensorb->data;
                volatile double* data_result = (double*)result->data;
            
                // Element-wise maximum calculation
                for (size_t i = 0; i < len; i++) {
                    data_result[i] = MAX(data_a[i], data_b[i]);
                }
                
                break;
            }
            
            case FLOAT32: {
                volatile float* data_a = (float*)tensora->data;
                volatile float* data_b = (float*)tensorb->data;
                volatile float* data_result = (float*)result->data;
        
                // Element-wise maximum calculation
                for (size_t i = 0; i < len; i++) {
                    data_result[i] = MAX(data_a[i], data_b[i]);
                }
                
                break;
            }
            
            case INT32: {
                volatile int32_t* data_a = (int32_t*)tensora->data;
                volatile int32_t* data_b = (int32_t*)tensorb->data;
                volatile int32_t* data_result = (int32_t*)result->data;
        
                // Element-wise maximum calculation
                for (size_t i = 0; i < len; i++) {
                    data_result[i] = MAX(data_a[i], data_b[i]);
                }
                
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(dtype_a);
                return NULL;
            }
        }
    } else {
        // Handling the case if the data types are NOT match
        switch (winner_in_the_type_hierarchy) {
            case FLOAT64: {
                volatile double* data_result = (double*)result->data;
                
                for (size_t i = 0; i < len; i++) {
                    // Calculate the pointers to the current elements, taking into account the size of the type
                    void* elem_a = (char*)tensora->data + i * get_dtype_size(dtype_a);
                    void* elem_b = (char*)tensorb->data + i * get_dtype_size(dtype_b);
                    
                    data_result[i] = MAX(nnl2_convert_to_float64(elem_a, dtype_a), nnl2_convert_to_float64(elem_b, dtype_b));
                }
                
                break;
            }
            
            case FLOAT32: {
                volatile float* data_result = (float*)result->data;
                
                for (size_t i = 0; i < len; i++) {
                    // Calculate the pointers to the current elements, taking into account the size of the type
                    void* elem_a = (char*)tensora->data + i * get_dtype_size(dtype_a);
                    void* elem_b = (char*)tensorb->data + i * get_dtype_size(dtype_b);
                    
                    data_result[i] = MAX(nnl2_convert_to_float32(elem_a, dtype_a), nnl2_convert_to_float32(elem_b, dtype_b));
                }
                
                break;
            }
        
            case INT32: {
                volatile int32_t* data_result = (int32_t*)result->data;
                
                for (size_t i = 0; i < len; i++) {
                    // Calculate the pointers to the current elements, taking into account the size of the type
                    void* elem_a = (char*)tensora->data + i * get_dtype_size(dtype_a);
                    void* elem_b = (char*)tensorb->data + i * get_dtype_size(dtype_b);
                    
                    data_result[i] = MAX(nnl2_convert_to_int32(elem_a, dtype_a), nnl2_convert_to_int32(elem_b, dtype_b));
                }
                
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(winner_in_the_type_hierarchy);
                return NULL;
            }
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return result;
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for max operation
 * @details
 * Array follows the common backend registration pattern for element-wise maximum operations.
 * Currently registered backends:
 *  - nnl2_naive: Basic reference implementation
 * 
 * @see nnl2_naive
 */
Implementation max_backends[] = {
	REGISTER_BACKEND(naive_max, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for max operation
 * @ingroup backend_system 
 */
maxfn nnl2_max;

/** 
 * @brief Makes the max backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(max);

/** 
 * @brief Sets the backend for max operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see ESET_BACKEND_BY_NAME
 */
void set_max_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(max_backends, nnl2_max, backend_name, current_backend(max));
}

/** 
 * @brief Gets the name of the active backend for max operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_max_backend() {
	return current_backend(max);
}

/** 
 * @brief Function declaration for getting all available max backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(max);

/**
 * @brief Function declaration for getting the number of available max backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(max);

#endif /** NNL2_MAX_H **/
