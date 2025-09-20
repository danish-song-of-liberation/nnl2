#ifndef NNL2_AXPY_INPLACE_H
#define NNL2_AXPY_INPLACE_H

/** @brief 
 * Performs element-wise AXPY operation (naive implementation)
 * 
 * Computes: summand = summand + alpha * sumend
 * Performs the scaled vector addition operation on two tensors,
 * modifying the summand tensor in place
 *
 ** @param summand 
 * Pointer to the tensor that will be modified (receives the AXPY result)
 *
 ** @param sumend 
 * Pointer to the tensor whose values will be scaled and added to the summand
 *
 ** @param alpha
 * Scalar multiplier for the sumend tensor values
 *
 ** @note 
 * Supports different data types through automatic conversion
 * The sumend elements are converted to the summand's data type
 *
 ** @note 
 * In max safety mode, validates tensor pointers and data pointers before access
 *
 ** @warning 
 * This function modifies the summand tensor directly
 * Both tensors must have the same shape
 *
 * @example
 * // Create two tensors with the same shape
 * Tensor* a = nnl2_ones((int[]){2, 3}, 2, FLOAT32);
 * Tensor* b = nnl2_ones((int[]){2, 3}, 2, FLOAT32);
 * 
 * // Compute a = a + 2.5 * b
 * naive_axpy_inplace(a, b, 2.5f);
 * 
 * // Now a contains 3.5 in all elements
 * nnl2_quick_print_tensor(a);
 * 
 * // Cleanup
 * nnl2_free_tensor(a);
 * nnl2_free_tensor(b);
 */
void naive_axpy_inplace(Tensor* summand, Tensor* sumend, float alpha) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks at the maximum safety level
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(summand, "Summand tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(summand->data, "Summand tensor's data is NULL");
        
        NNL2_CHECK_NULL_IF_ERR_RETURN(sumend, "Sumend tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(sumend->data, "Sumend tensor's data is NULL");
    #endif
    
    // Calculating the total number of elements in the summand tensor
    size_t total_elems = product(summand->shape, summand->rank);
    
    // If the tensor is empty, exit the function
    if(total_elems == 0) return;
    
    TensorType dtype_summand = summand->dtype;
    TensorType dtype_sumend = sumend->dtype;
    
    if(dtype_summand == dtype_sumend) {
        // Handling case when the tensors have the same type
        
        switch(dtype_summand) {
            case FLOAT64: {
                volatile double* data_summand = (double*)summand->data;
                volatile double* data_sumend = (double*)sumend->data;
                double alpha_double = (double)alpha;
                
                // Element-wise AXPY operation
                for(size_t i = 0; i < total_elems; i++) {
                    data_summand[i] += data_sumend[i] * alpha_double;
                }
				
                break;
            }
            
            case FLOAT32: {
                volatile float* data_summand = (float*)summand->data;
                volatile float* data_sumend = (float*)sumend->data;
                
                // Element-wise AXPY operation
                for(size_t i = 0; i < total_elems; i++) {
                    data_summand[i] += data_sumend[i] * alpha;
                }    
				
                break;
            }
            
            case INT32: {
                volatile int32_t* data_summand = (int32_t*)summand->data;
                volatile int32_t* data_sumend = (int32_t*)sumend->data;
                int32_t alpha_int = (int32_t)alpha;
                
                // Element-wise AXPY operation
                for(size_t i = 0; i < total_elems; i++) {
                    data_summand[i] += data_sumend[i] * alpha_int;
                }        
				
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(dtype_summand);
                return;
            }
        }
    } else {
        // Handling the case when tensors have different data types
        // Calculating the step size for accessing sumend tensor elements
        size_t sumend_step = get_dtype_size(dtype_sumend);
        
        // Casting sumend data to char* for byte access
        char* sumend_data = (char*)sumend->data;
        
        switch(dtype_summand) {
            case FLOAT64: {
                volatile double* data_summand = (double*)summand->data;
                double alpha_double = (double)alpha;
                
                // For each element, convert the sumend element to FLOAT64 and perform AXPY
                for(size_t i = 0; i < total_elems; i++) {
                    void* sumend_elem = sumend_data + i * sumend_step;
                    data_summand[i] += nnl2_convert_to_float64(sumend_elem, dtype_sumend) * alpha_double;
                }
                
                break; 
            }
            
            case FLOAT32: {
                volatile float* data_summand = (float*)summand->data;
                
                // For each element, convert the sumend element to FLOAT32 and perform AXPY
                for(size_t i = 0; i < total_elems; i++) {
                    void* sumend_elem = sumend_data + i * sumend_step;
                    data_summand[i] += nnl2_convert_to_float32(sumend_elem, dtype_sumend) * alpha;
                }
                
                break; 
            }
            
            case INT32: {
                volatile int32_t* data_summand = (int32_t*)summand->data;
                int32_t alpha_int = (int32_t)alpha;
                
                // For each element, convert the sumend element to INT32 and perform AXPY
                for(size_t i = 0; i < total_elems; i++) {
                    void* sumend_elem = sumend_data + i * sumend_step;
                    data_summand[i] += nnl2_convert_to_int32(sumend_elem, dtype_sumend) * alpha_int;
                }
                
                break; 
            }
            
            default: {
                NNL2_TYPE_ERROR(dtype_summand);
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
 * @brief Backend implementations for AXPY in-place operation
 */
Implementation axpy_inplace_backends[] = {
    REGISTER_BACKEND(naive_axpy_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};  

/**
 * @brief Function pointer for AXPY in-place operation
 * @ingroup backend_system
 */
axpyinplacefn axpy_inplace;
make_current_backend(axpy_inplace);

/**
 * @brief Sets the backend for AXPY in-place operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for AXPY in-place operation
 */
void set_axpy_inplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(axpy_inplace_backends, axpy_inplace, backend_name, current_backend(axpy_inplace));
}

/**
 * @brief Gets the name of the current backend for AXPY in-place operation
 * @ingroup backend_system
 * @return const char* Name of the current backend
 */
const char* get_axpy_inplace_backend() {
    return current_backend(axpy_inplace);
}

/**
 * @brief Gets the list of available backends for AXPY in-place operation
 * @ingroup backend_system
 * @return const char** Array of backend names
 */
DEFINE_GET_BACKENDS_FUNCTION(axpy_inplace);

/**
 * @brief Gets the number of available backends for AXPY in-place operation
 * @ingroup backend_system
 * @return size_t Number of available backends
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(axpy_inplace);

#endif /** NNL2_AXPY_INPLACE_H **/
