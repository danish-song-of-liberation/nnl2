#ifndef NNL2_AXPY_H
#define NNL2_AXPY_H

/** @brief
 * Performs element-wise AXPY operation (naive implementation)
 * Computes: result = summand + alpha * sumend
 *
 ** @details
 * The function creates a new tensor containing the result of the AXPY operation
 * on the two input tensors. It supports various data types with automatic
 * casting to a higher type in the hierarchy
 *
 ** @param summand
 * Pointer to the summand tensor
 *
 ** @param sumend
 * Pointer to the sumend tensor to be scaled and added
 *
 ** @param alpha
 * Scalar multiplier for the sumend tensor
 *
 ** @return 
 * Pointer to a new tensor with the AXPY operation result
 *
 ** @note
 * Uses volatile pointers to prevent compiler optimizations
 *
 ** @note
 * Returns NULL in case of failure
 *
 ** @see nnl2_empty
 ** @see get_dtype_size()
 ** @see nnl2_convert_to_float64()
 ** @see nnl2_convert_to_float32()
 ** @see nnl2_convert_to_int32()
 **/
Tensor* naive_axpy(Tensor* summand, Tensor* sumend, float alpha) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks at the maximum safety level
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(summand, "Summand tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(summand->data, "Summand tensor's data is NULL", NULL);
        
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(sumend, "Sumend tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(sumend->data, "Sumend tensor's data is NULL", NULL);
    #endif
    
    // Calculate the total number of elements in the tensors
    size_t len = product(summand->shape, summand->rank);
    
    TensorType dtype_summand = summand->dtype;
    TensorType dtype_sumend = sumend->dtype;
    
    // Selecting the winning type (higher in the hierarchy)
    TensorType winner_in_the_type_hierarchy = MAX(dtype_summand, dtype_sumend);

    // Create an output tensor with the same shape and winning data type
    Tensor* result = nnl2_empty(summand->shape, summand->rank, winner_in_the_type_hierarchy);
    
    if(len == 0) return result;
    
    if(dtype_summand == dtype_sumend) {
        // Handling the case if the data types match
        
        switch(dtype_summand) {
            case FLOAT64: {
                volatile double* data_summand = (double*)summand->data;
                volatile double* data_sumend = (double*)sumend->data;
                volatile double* data_result = (double*)result->data;
                double alpha_double = (double)alpha;
            
                // Element-wise AXPY operation
                for(size_t i = 0; i < len; i++) {
                    data_result[i] = data_summand[i] + (data_sumend[i] * alpha_double);
                }
                
                break;
            }
            
            case FLOAT32: {
                volatile float* data_summand = (float*)summand->data;
                volatile float* data_sumend = (float*)sumend->data;
                volatile float* data_result = (float*)result->data;
        
                // Element-wise AXPY operation
                for(size_t i = 0; i < len; i++) {
                    data_result[i] = data_summand[i] + (data_sumend[i] * alpha);
                }
                
                break;
            }
            
            case INT32: {
                volatile int32_t* data_summand = (int32_t*)summand->data;
                volatile int32_t* data_sumend = (int32_t*)sumend->data;
                volatile int32_t* data_result = (int32_t*)result->data;
                int32_t alpha_int = (int32_t)alpha;
        
                // Element-wise AXPY operation
                for(size_t i = 0; i < len; i++) {
                    data_result[i] = data_summand[i] + (data_sumend[i] * alpha_int);
                }
                
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(dtype_summand);
                nnl2_free_tensor(result);
                return NULL;
            }
        }
    } else {
        // Handling the case if the data types are NOT match
        switch(winner_in_the_type_hierarchy) {
            case FLOAT64: {
                volatile double* data_result = (double*)result->data;
                double alpha_double = (double)alpha;
                
                for(size_t i = 0; i < len; i++) {
                    // Calculate the pointers to the current elements, taking into account the size of the type
                    void* elem_summand = (char*)summand->data + i * get_dtype_size(dtype_summand);
                    void* elem_sumend = (char*)sumend->data + i * get_dtype_size(dtype_sumend);
                    
                    data_result[i] = nnl2_convert_to_float64(elem_summand, dtype_summand) + (nnl2_convert_to_float64(elem_sumend, dtype_sumend) * alpha_double);
                }
                
                break;
            }
            
            case FLOAT32: {
                volatile float* data_result = (float*)result->data;
                
                for(size_t i = 0; i < len; i++) {
                    // Calculate the pointers to the current elements, taking into account the size of the type
                    void* elem_summand = (char*)summand->data + i * get_dtype_size(dtype_summand);
                    void* elem_sumend = (char*)sumend->data + i * get_dtype_size(dtype_sumend);
                    
                    data_result[i] = nnl2_convert_to_float32(elem_summand, dtype_summand) + (nnl2_convert_to_float32(elem_sumend, dtype_sumend) * alpha);
                }
                
                break;
            }
        
            case INT32: {
                volatile int32_t* data_result = (int32_t*)result->data;
                int32_t alpha_int = (int32_t)alpha;
                
                for(size_t i = 0; i < len; i++) {
                    // Calculate the pointers to the current elements, taking into account the size of the type
                    void* elem_summand = (char*)summand->data + i * get_dtype_size(dtype_summand);
                    void* elem_sumend = (char*)sumend->data + i * get_dtype_size(dtype_sumend);
                    
                    data_result[i] = nnl2_convert_to_int32(elem_summand, dtype_summand) + 
                                    (nnl2_convert_to_int32(elem_sumend, dtype_sumend) * alpha_int);
                }
                
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(winner_in_the_type_hierarchy);
                nnl2_free_tensor(result);
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
 * @brief Backend implementations for AXPY operation
 */
Implementation axpy_backends[] = {
    REGISTER_BACKEND(naive_axpy, nnl2_naive, NAIVE_BACKEND_NAME),
};  

/**
 * @brief Function pointer for AXPY operation
 * @ingroup backend_system
 */
axpyfn axpy;
make_current_backend(axpy);

/**
 * @brief Sets the backend for AXPY operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for AXPY operation
 */
void set_axpy_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(axpy_backends, axpy, backend_name, current_backend(axpy));
}

/**
 * @brief Gets the name of the current backend for AXPY operation
 * @ingroup backend_system
 * @return const char* Name of the current backend
 */
const char* get_axpy_backend() {
    return current_backend(axpy);
}

/**
 * @brief Gets the list of available backends for AXPY operation
 * @ingroup backend_system
 * @return const char** Array of backend names
 */
DEFINE_GET_BACKENDS_FUNCTION(axpy);

/**
 * @brief Gets the number of available backends for AXPY operation
 * @ingroup backend_system
 * @return size_t Number of available backends
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(axpy);

#endif /** NNL2_AXPY_H **/
