#ifndef NNL2_POW_INPLACE_H
#define NNL2_POW_INPLACE_H

/** @brief 
 * Performs element-wise exponentiation of two tensors (naive implementation)
 * 
 * Raises the elements of the base tensor to the power of the corresponding elements 
 * of the exponent tensor, modifying the base tensor in place
 *
 ** @param base 
 * Pointer to the tensor that will be modified (receives the exponentiation result)
 *
 ** @param exponent 
 * Pointer to the tensor whose values will be used as exponents
 *
 ** @note 
 * Supports different data types through automatic conversion
 * The exponent elements are converted to the base's data type
 *
 ** @note 
 * In max safety mode, validates tensor pointers and data pointers before access
 *
 ** @warning 
 * This function modifies the base tensor directly
 *
 * @example
 * // Create two tensors with the same shape
 * Tensor* a = nnl2_tensor((float[]){2.0, 3.0, 4.0}, (int[]){3}, 1, FLOAT32);
 * Tensor* b = nnl2_tensor((float[]){2.0, 1.0, 0.5}, (int[]){3}, 1, FLOAT32);
 * 
 * // Raise a to the power of b (a becomes a^b)
 * naive_powinplace(a, b);
 * 
 * // Now a contains [4.0, 3.0, 2.0]
 * nnl2_quick_print_tensor(a);
 * 
 * // Cleanup
 * nnl2_free_tensor(a);
 * nnl2_free_tensor(b);
 */
void naive_powinplace(Tensor* base, const Tensor* exponent) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks at the maximum safety level
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(base, "Base tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(base->data, "Base tensor's data is NULL");
        
        NNL2_CHECK_NULL_IF_ERR_RETURN(exponent, "Exponent tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(exponent->data, "Exponent tensor's data is NULL");
    #endif
    
    // Calculating the total number of elements in the base tensor
    size_t len_base = product(base->shape, base->rank);
    
    // If the tensor is empty, exit the function
    if(len_base == 0) return;
    
    TensorType dtype_base = base->dtype;
    TensorType dtype_exponent = exponent->dtype;
    
    if(dtype_base == dtype_exponent) {
        // Handling case when the tensors have the same type
        
        switch(dtype_base) {
            case FLOAT64: {
                volatile double* data_base = (double*)base->data;
                volatile double* data_exponent = (double*)exponent->data;
                
                // Element-wise exponentiation
                for(size_t i = 0; i < len_base; i++) {
                    data_base[i] = pow(data_base[i], data_exponent[i]);
                }
                break;
            }
            
            case FLOAT32: {
                volatile float* data_base = (float*)base->data;
                volatile float* data_exponent = (float*)exponent->data;
                
                // Element-wise exponentiation
                for(size_t i = 0; i < len_base; i++) {
                    data_base[i] = powf(data_base[i], data_exponent[i]);
                }    
                break;
            }
            
            case INT32: {
                volatile int32_t* data_base = (int32_t*)base->data;
                volatile int32_t* data_exponent = (int32_t*)exponent->data;
                
                // Element-wise exponentiation
                for(size_t i = 0; i < len_base; i++) {
                    data_base[i] = (int32_t)pow(data_base[i], data_exponent[i]);
                }        
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(dtype_base);
                return;
            }
        }
    } else {
        // Handling the case when tensors have different data types
        // Calculating the step size for accessing exponent tensor elements
        size_t exponent_step = get_dtype_size(dtype_exponent);
        
        // Casting exponent data to char* for byte access
        char* exponent_data = (char*)exponent->data;
        
        switch(dtype_base) {
            case FLOAT64: {
                volatile double* data_base = (double*)base->data;
                
                // For each element, convert the exponent element to FLOAT64 and use it as exponent
                for(size_t i = 0; i < len_base; i++) {
                    void* exponent_elem = exponent_data + i * exponent_step;
                    data_base[i] = pow(data_base[i], nnl2_convert_to_float64(exponent_elem, dtype_exponent));
                }
                
                break; 
            }
            
            case FLOAT32: {
                volatile float* data_base = (float*)base->data;
                
                // For each element, convert the exponent element to FLOAT32 and use it as exponent
                for(size_t i = 0; i < len_base; i++) {
                    void* exponent_elem = exponent_data + i * exponent_step;
                    data_base[i] = powf(data_base[i], nnl2_convert_to_float32(exponent_elem, dtype_exponent));
                }
                
                break; 
            }
            
            case INT32: {
                volatile int32_t* data_base = (int32_t*)base->data;
                
                // For each element, convert the exponent element to INT32 and use it as exponent
                for(size_t i = 0; i < len_base; i++) {
                    void* exponent_elem = exponent_data + i * exponent_step;
                    data_base[i] = (int32_t)pow(data_base[i], nnl2_convert_to_int32(exponent_elem, dtype_exponent));
                }
                
                break; 
            }
            
            default: {
                NNL2_TYPE_ERROR(dtype_base);
                return;
            }
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

/** 
 * @ingroup backend_system
 * @brief Backend implementations for in-place power operation
 * @details
 * Array follows the common backend registration pattern.
 * Currently registered backends:
 *  - naive_powinplace: Basic reference implementation
 * 
 * @see naive_powinplace
 */
Implementation powinplace_backends[] = {
	REGISTER_BACKEND(naive_powinplace, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for in-place power operation
 * @ingroup backend_system 
 */
powinplacefn powinplace;

/** 
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(powinplace);

/** 
 * @brief Sets the backend for in-place power operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void set_powinplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(powinplace_backends, powinplace, backend_name, CURRENT_BACKEND(powinplace));
}

/** 
 * @brief Gets the name of the active backend for in-place power operation
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* get_powinplace_backend() {
	return CURRENT_BACKEND(powinplace);
}

/** 
 * @brief Function declaration for getting all `powinplace` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(powinplace);

/**
 * @brief Function declaration for getting the number of all `powinplace` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(powinplace);

#endif
