#ifndef NNL2_DIV_INPLACE_H
#define NNL2_DIV_INPLACE_H

/** @brief 
 * Performs element-wise division of two tensors (naive implementation)
 * 
 * Divides the elements of the dividend tensor by the corresponding elements 
 * of the divisor tensor, modifying the dividend tensor in place
 *
 ** @param dividend 
 * Pointer to the tensor that will be modified (receives the division result)
 *
 ** @param divisor 
 * Pointer to the tensor whose values will divide the dividend
 *
 ** @note 
 * Supports different data types through automatic conversion
 * The divisor elements are converted to the dividend's data type
 *
 ** @note 
 * In max safety mode, validates tensor pointers and data pointers before access
 *
 ** @warning 
 * This function modifies the dividend tensor directly
 * Division by zero may result in undefined behavior depending on data type
 *
 * @example
 * // Create two tensors with the same shape
 * Tensor* a = nnl2_ones((int[]){2, 3}, 2, FLOAT32);
 * Tensor* b = nnl2_ones((int[]){2, 3}, 2, FLOAT32);
 * 
 * // Divide a by b (a becomes a / b)
 * nnl2_naive_divinplace(a, b);
 * 
 * // Now a contains 1.0 in all elements
 * nnl2_quick_print_tensor(a);
 * 
 * // Cleanup
 * nnl2_free_tensor(a);
 * nnl2_free_tensor(b);
 */
void nnl2_naive_divinplace(Tensor* dividend, const Tensor* divisor) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks at the maximum safety level
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(dividend, "Dividend tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(dividend->data, "Dividend tensor's data is NULL");
        
        NNL2_CHECK_NULL_IF_ERR_RETURN(divisor, "Divisor tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(divisor->data, "Divisor tensor's data is NULL");
    #endif
    
    // Calculating the total number of elements in the dividend tensor
    size_t len_dividend = product(dividend->shape, dividend->rank);
    
    // If the tensor is empty, exit the function
    if(len_dividend == 0) return;
    
    TensorType dtype_dividend = dividend->dtype;
    TensorType dtype_divisor = divisor->dtype;
    
    if(dtype_dividend == dtype_divisor) {
        // Handling case when the tensors have the same type
        
        switch(dtype_dividend) {
            case FLOAT64: {
                volatile double* data_dividend = (double*)dividend->data;
                volatile double* data_divisor = (double*)divisor->data;
                
                // Element-wise division
                for(size_t i = 0; i < len_dividend; i++) data_dividend[i] /= data_divisor[i];
                break;
            }
            
            case FLOAT32: {
                volatile float* data_dividend = (float*)dividend->data;
                volatile float* data_divisor = (float*)divisor->data;
                
                // Element-wise division
                for(size_t i = 0; i < len_dividend; i++) data_dividend[i] /= data_divisor[i];    
                break;
            }
            
            case INT32: {
                volatile int32_t* data_dividend = (int32_t*)dividend->data;
                volatile int32_t* data_divisor = (int32_t*)divisor->data;
                
                // Element-wise division
                for(size_t i = 0; i < len_dividend; i++) data_dividend[i] /= data_divisor[i];        
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(dtype_dividend);
                return;
            }
        }
    } else {
        // Handling the case when tensors have different data types
        // Calculating the step size for accessing divisor tensor elements
        size_t divisor_step = get_dtype_size(dtype_divisor);
        
        // Casting divisor data to char* for byte access
        char* divisor_data = (char*)divisor->data;
        
        switch(dtype_dividend) {
            case FLOAT64: {
                volatile double* data_dividend = (double*)dividend->data;
                
                // For each element, convert the divisor element to float64 and divide by it
                for(size_t i = 0; i < len_dividend; i++) {
                    void* divisor_elem = divisor_data + i * divisor_step;
                    data_dividend[i] /= nnl2_convert_to_float64(divisor_elem, dtype_divisor);
                }
                
                break; 
            }
            
            case FLOAT32: {
                volatile float* data_dividend = (float*)dividend->data;
                
                // For each element, convert the divisor element to float32 and divide by it
                for(size_t i = 0; i < len_dividend; i++) {
                    void* divisor_elem = divisor_data + i * divisor_step;
                    data_dividend[i] /= nnl2_convert_to_float32(divisor_elem, dtype_divisor);
                }
                
                break; 
            }
            
            case INT32: {
                volatile int32_t* data_dividend = (int32_t*)dividend->data;
                
                // For each element, convert the divisor element to int32 and divide by it
                for(size_t i = 0; i < len_dividend; i++) {
                    void* divisor_elem = divisor_data + i * divisor_step;
                    data_dividend[i] /= nnl2_convert_to_int32(divisor_elem, dtype_divisor);
                }
                
                break; 
            }
            
            default: {
                NNL2_TYPE_ERROR(dtype_dividend);
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
 * @brief Backend implementations for division operation
 * @details
 * Array follows the common backend registration pattern.
 * Currently registered backends:
 *  - nnl2_naive_divinplace: Basic reference implementation
 * 
 * @see nnl2_naive_divinplace
 */
Implementation divinplace_backends[] = {
	REGISTER_BACKEND(nnl2_naive_divinplace, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for division operation
 * @ingroup backend_system 
 */
divinplacefn divinplace;

/** 
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
make_current_backend(divinplace);

/** 
 * @brief Sets the backend for division operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void set_divinplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(divinplace_backends, divinplace, backend_name, CURRENT_BACKEND(divinplace));
}

/** 
 * @brief Gets the name of the active backend for division operation
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* get_divinplace_backend() {
	return CURRENT_BACKEND(divinplace);
}

/** 
 * @brief Function declaration for getting all `divinplace` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(divinplace);

/**
 * @brief Function declaration for getting the number of all `divinplace` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(divinplace);

#endif /** NNL2_DIV_INPLACE_H **/
