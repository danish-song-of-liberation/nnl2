#ifndef NNL2_POW_H
#define NNL2_POW_H

/** @brief
 * Performs element-wise exponentiation of two tensors (naive implementation)
 *
 ** @details
 * The function creates a new tensor containing the result of raising each element
 * of the base tensor to the power of the corresponding element in the exponent tensor.
 * It supports various data types with automatic casting to a higher type in the hierarchy.
 *
 ** @param base
 * Pointer to the base tensor
 *
 ** @param exponent
 * Pointer to the exponent tensor
 *
 ** @return 
 * Pointer to a new tensor with the exponentiation result
 *
 ** @note
 * Uses volatile pointers to prevent compiler optimizations
 *
 ** @note
 * Returns NULL in case of failure or unsupported data type
 *
 ** @note
 * For integer types, the result is cast back to integer which may truncate values
 *
 ** @see nnl2_empty
 ** @see get_dtype_size()
 ** @see nnl2_convert_to_float64()
 ** @see nnl2_convert_to_float32()
 ** @see nnl2_convert_to_int32()
 **/
nnl2_tensor* naive_pow(const nnl2_tensor* base, const nnl2_tensor* exponent) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	// Additional checks on maximum safety level
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(base, "Base tensor is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(base, "Exponent tensor is NULL", NULL);
	#endif
	
    // Calculate the total number of elements in the tensors
    size_t len = nnl2_product(base->shape, base->rank);
    
    nnl2_tensor_type dtype_base = base->dtype;
    nnl2_tensor_type dtype_exponent = exponent->dtype;
    
    // Selecting the winning type (higher in the hierarchy)
    nnl2_tensor_type winner_in_the_type_hierarchy = MAX(dtype_base, dtype_exponent);

    // Create an output tensor with the same shape and winning data type
    nnl2_tensor* result = nnl2_empty(base->shape, base->rank, winner_in_the_type_hierarchy);
    
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
    
    if (dtype_base == dtype_exponent) {
        // Handling the case if the data types match
        
        switch (dtype_base) {
            case FLOAT64: {
                volatile double* data_base = (double*)base->data;
                volatile double* data_exponent = (double*)exponent->data;
                volatile double* data_result = (double*)result->data;
            
                // Element-wise exponentiation
                for (size_t i = 0; i < len; i++) {
                    data_result[i] = pow(data_base[i], data_exponent[i]);
                }
                
                break;
            }
            
            case FLOAT32: {
                volatile float* data_base = (float*)base->data;
                volatile float* data_exponent = (float*)exponent->data;
                volatile float* data_result = (float*)result->data;
        
                // Element-wise exponentiation
                for (size_t i = 0; i < len; i++) {
                    data_result[i] = powf(data_base[i], data_exponent[i]);
                }
                
                break;
            }
            
            case INT32: {
                volatile int32_t* data_base = (int32_t*)base->data;
                volatile int32_t* data_exponent = (int32_t*)exponent->data;
                volatile int32_t* data_result = (int32_t*)result->data;
        
                // Element-wise exponentiation
                for (size_t i = 0; i < len; i++) {
                    data_result[i] = (int32_t)pow(data_base[i], data_exponent[i]);
                }
                
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(dtype_base);
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
                    void* elem_base = (char*)base->data + i * get_dtype_size(dtype_base);
                    void* elem_exponent = (char*)exponent->data + i * get_dtype_size(dtype_exponent);
                    
                    data_result[i] = pow(nnl2_convert_to_float64(elem_base, dtype_base), nnl2_convert_to_float64(elem_exponent, dtype_exponent));
                }
                
                break;
            }
            
            case FLOAT32: {
                volatile float* data_result = (float*)result->data;
                
                for (size_t i = 0; i < len; i++) {
                    // Calculate the pointers to the current elements, taking into account the size of the type
                    void* elem_base = (char*)base->data + i * get_dtype_size(dtype_base);
                    void* elem_exponent = (char*)exponent->data + i * get_dtype_size(dtype_exponent);
                    
                    data_result[i] = powf(nnl2_convert_to_float32(elem_base, dtype_base), nnl2_convert_to_float32(elem_exponent, dtype_exponent));
                }
                
                break;
            }
        
            case INT32: {
                volatile int32_t* data_result = (int32_t*)result->data;
                
                for (size_t i = 0; i < len; i++) {
                    // Calculate the pointers to the current elements, taking into account the size of the type
                    void* elem_base = (char*)base->data + i * get_dtype_size(dtype_base);
                    void* elem_exponent = (char*)exponent->data + i * get_dtype_size(dtype_exponent);
                    
                    data_result[i] = (int32_t)pow(nnl2_convert_to_int32(elem_base, dtype_base), nnl2_convert_to_int32(elem_exponent, dtype_exponent));
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
 * @brief Backend implementations for power operation
 * @details
 * Array follows the common backend registration pattern.
 * Currently registered backends:
 *  - naive_pow: Basic reference implementation
 * 
 * @see naive_pow
 */
nnl2_runtime_implementation pow_backends[] = {
	REGISTER_BACKEND(naive_pow, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for power operation
 * @ingroup backend_system 
 */
powfn nnl2_pow;

/** 
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(pow);

/** 
 * @brief Sets the backend for power operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void set_pow_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(pow_backends, pow, backend_name, CURRENT_BACKEND(pow));
}

/** 
 * @brief Gets the name of the active backend for power operation
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* get_pow_backend() {
	return CURRENT_BACKEND(pow);
}

/** 
 * @brief Function declaration for getting all `pow` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(pow);

/**
 * @brief Function declaration for getting the number of all `pow` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(pow);

#endif /** NNL2_POW_H **/
