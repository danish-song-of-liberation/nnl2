#ifndef NNL2_DIV_H
#define NNL2_DIV_H

/** @brief
 * Performs element-wise division of two tensors (naive implementation)
 *
 ** @details
 * The function creates a new tensor containing the quotient of the corresponding elements
 * of the two input tensors. It supports various data types with automatic
 * casting to a higher type in the hierarchy. Checks for division by zero.
 *
 ** @param dividend
 * Pointer to the dividend tensor
 *
 ** @param divisor
 * Pointer to the divisor tensor
 *
 ** @return 
 * Pointer to a new tensor with the division result
 *
 ** @note
 * Uses volatile pointers to prevent compiler optimizations
 *
 ** @note
 * Returns NULL in case of failure or division by zero
 *
 ** @see nnl2_empty
 ** @see get_dtype_size()
 ** @see nnl2_convert_to_float64()
 ** @see nnl2_convert_to_float32()
 ** @see nnl2_convert_to_int32()
 **/
Tensor* nnl2_naive_div(const Tensor* dividend, const Tensor* divisor) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Calculate the total number of elements in the tensors
    size_t len = product(dividend->shape, dividend->rank);
    
    TensorType dtype_dividend = dividend->dtype;
    TensorType dtype_divisor = divisor->dtype;
    
    // Selecting the winning type (higher in the hierarchy)
    TensorType winner_in_the_type_hierarchy = MAX(dtype_dividend, dtype_divisor);

    // Create an output tensor with the same shape and winning data type
    Tensor* quotient = nnl2_empty(dividend->shape, dividend->rank, winner_in_the_type_hierarchy);
    
    if (quotient == NULL) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return NULL;
    }
    
    if (len == 0) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return quotient;
    }
    
    if (dtype_dividend == dtype_divisor) {
        // Handling the case if the data types match
        
        switch (dtype_dividend) {
            case FLOAT64: {
                volatile double* data_dividend = (double*)dividend->data;
                volatile double* data_divisor = (double*)divisor->data;
                volatile double* data_quotient = (double*)quotient->data;
            
                // Element-wise division with zero check
                for (size_t i = 0; i < len; i++) {
                    if (data_divisor[i] == 0.0) {
                        NNL2_ERROR("Division by zero at index %zu\n", i);
                        nnl2_free_tensor(quotient);
                        return NULL;
                    }
                    data_quotient[i] = data_dividend[i] / data_divisor[i];
                }
                
                break;
            }
            
            case FLOAT32: {
                volatile float* data_dividend = (float*)dividend->data;
                volatile float* data_divisor = (float*)divisor->data;
                volatile float* data_quotient = (float*)quotient->data;
        
                // Element-wise division with zero check
                for (size_t i = 0; i < len; i++) {
                    if (data_divisor[i] == 0.0f) {
                        NNL2_ERROR("Division by zero at index %zu\n", i);
                        nnl2_free_tensor(quotient);
                        return NULL;
                    }
                    data_quotient[i] = data_dividend[i] / data_divisor[i];
                }
                
                break;
            }
            
            case INT32: {
                volatile int32_t* data_dividend = (int32_t*)dividend->data;
                volatile int32_t* data_divisor = (int32_t*)divisor->data;
                volatile int32_t* data_quotient = (int32_t*)quotient->data;
        
                // Element-wise division with zero check
                for (size_t i = 0; i < len; i++) {
                    if (data_divisor[i] == 0) {
                        NNL2_ERROR("Division by zero at index %zu\n", i);
                        nnl2_free_tensor(quotient);
                        return NULL;
                    }
                    data_quotient[i] = data_dividend[i] / data_divisor[i];
                }
                
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(dtype_dividend);
                nnl2_free_tensor(quotient);
                return NULL;
            }
        }
    } else {
        // Handling the case if the data types are NOT match
        switch (winner_in_the_type_hierarchy) {
            case FLOAT64: {
                volatile double* data_quotient = (double*)quotient->data;
                
                for (size_t i = 0; i < len; i++) {
                    // Calculate the pointers to the current elements, taking into account the size of the type
                    void* elem_dividend = (char*)dividend->data + i * get_dtype_size(dtype_dividend);
                    void* elem_divisor = (char*)divisor->data + i * get_dtype_size(dtype_divisor);
                    
                    double divisor_val = nnl2_convert_to_float64(elem_divisor, dtype_divisor);
                    if (divisor_val == 0.0) {
                        fprintf(stderr, "Error (Hello from C!): Division by zero at index %zu\n", i);
                        nnl2_free_tensor(quotient);
                        return NULL;
                    }
                    
                    data_quotient[i] = nnl2_convert_to_float64(elem_dividend, dtype_dividend) / divisor_val;
                }
                
                break;
            }
            
            case FLOAT32: {
                volatile float* data_quotient = (float*)quotient->data;
                
                for (size_t i = 0; i < len; i++) {
                    // Calculate the pointers to the current elements, taking into account the size of the type
                    void* elem_dividend = (char*)dividend->data + i * get_dtype_size(dtype_dividend);
                    void* elem_divisor = (char*)divisor->data + i * get_dtype_size(dtype_divisor);
                    
                    float divisor_val = nnl2_convert_to_float32(elem_divisor, dtype_divisor);
                    if (divisor_val == 0.0f) {
                        fprintf(stderr, "Error (Hello from C!): Division by zero at index %zu\n", i);
                        nnl2_free_tensor(quotient);
                        return NULL;
                    }
                    
                    data_quotient[i] = nnl2_convert_to_float32(elem_dividend, dtype_dividend) / divisor_val;
                }
                
                break;
            }
        
            case INT32: {
                volatile int32_t* data_quotient = (int32_t*)quotient->data;
                
                for (size_t i = 0; i < len; i++) {
                    // Calculate the pointers to the current elements, taking into account the size of the type
                    void* elem_dividend = (char*)dividend->data + i * get_dtype_size(dtype_dividend);
                    void* elem_divisor = (char*)divisor->data + i * get_dtype_size(dtype_divisor);
                    
                    int32_t divisor_val = nnl2_convert_to_int32(elem_divisor, dtype_divisor);
                    if (divisor_val == 0) {
                        fprintf(stderr, "Error (Hello from C!): Division by zero at index %zu\n", i);
                        nnl2_free_tensor(quotient);
                        return NULL;
                    }
                    
                    data_quotient[i] = nnl2_convert_to_int32(elem_dividend, dtype_dividend) / divisor_val;
                }
                
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(winner_in_the_type_hierarchy);
                nnl2_free_tensor(quotient);
                return NULL;
            }
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return quotient;
}

/** 
 * @ingroup backend_system
 * @brief Backend implementations for division operation
 * @details
 * Array follows the common backend registration pattern.
 * Currently registered backends:
 *  - nnl2_naive_div: Basic reference implementation
 * 
 * @see nnl2_naive_div
 */
Implementation div_backends[] = {
	REGISTER_BACKEND(nnl2_naive_div, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for division operation
 * @ingroup backend_system 
 */
divfn nnl2_div;

/** 
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(div);

/** 
 * @brief Sets the backend for division operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void set_div_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(div_backends, div, backend_name, CURRENT_BACKEND(div));
}

/** 
 * @brief Gets the name of the active backend for division operation
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* get_div_backend() {
	return CURRENT_BACKEND(div);
}

/** 
 * @brief Function declaration for getting all `div` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(div);

/**
 * @brief Function declaration for getting the number of all `div` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(div);

#endif /** NNL2_DIV_H **/
