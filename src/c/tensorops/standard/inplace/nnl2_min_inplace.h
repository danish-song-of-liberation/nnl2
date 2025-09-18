#ifndef NNL2_MIN_INPLACE_H
#define NNL2_MIN_INPLACE_H

/** @brief 
 * Performs element-wise minimum of two tensors (naive implementation)
 * 
 * Compares elements of the first tensor with corresponding elements 
 * of the second tensor, storing the minimum value in the first tensor
 *
 ** @param tensora 
 * Pointer to the tensor that will be modified (receives the minimum values)
 *
 ** @param tensorb 
 * Pointer to the tensor whose values will be used for comparison
 *
 ** @note 
 * Supports different data types through automatic conversion
 * The second tensor's elements are converted to the first tensor's data type
 *
 ** @note 
 * In max safety mode, validates tensor pointers and data pointers before access
 *
 ** @warning 
 * This function modifies the first tensor directly
 *
 * @example
 * // Create two tensors with the same shape
 * Tensor* a = nnl2_tensor((float[]){2.0, 3.0, 4.0}, (int[]){3}, 1, FLOAT32);
 * Tensor* b = nnl2_tensor((float[]){3.0, 1.0, 5.0}, (int[]){3}, 1, FLOAT32);
 * 
 * // Store element-wise minimum in tensor a
 * naive_mininplace(a, b);
 * 
 * // Now a contains [2.0, 1.0, 4.0]
 * nnl2_quick_print_tensor(a);
 * 
 * // Cleanup
 * nnl2_free_tensor(a);
 * nnl2_free_tensor(b);
 */
void naive_mininplace(Tensor* tensora, const Tensor* tensorb) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks at the maximum safety level
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensora, "First tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensora->data, "First tensor's data is NULL");
        
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensorb, "Second tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensorb->data, "Second tensor's data is NULL");
    #endif
    
    // Calculating the total number of elements in the first tensor
    size_t total_elems = product(tensora->shape, tensora->rank);
    
    // If the tensor is empty, exit the function
    if(total_elems == 0) return;
    
    TensorType typea = tensora->dtype;
    TensorType typeb = tensorb->dtype;
    
    if(typea == typeb) {
        // Handling case when the tensors have the same type
        
        switch(typea) {
            case FLOAT64: {
                volatile double* data_a = (double*)tensora->data;
                volatile double* data_b = (double*)tensorb->data;
                
                // Element-wise minimum calculation
                for(size_t i = 0; i < total_elems; i++) {
                    data_a[i] = MIN(data_a[i], data_b[i]);
                }
                break;
            }
            
            case FLOAT32: {
                volatile float* data_a = (float*)tensora->data;
                volatile float* data_b = (float*)tensorb->data;
                
                // Element-wise minimum calculation
                for(size_t i = 0; i < total_elems; i++) {
                    data_a[i] = MIN(data_a[i], data_b[i]);
                }
                break;
            }
            
            case INT32: {
                volatile int32_t* data_a = (int32_t*)tensora->data;
                volatile int32_t* data_b = (int32_t*)tensorb->data;
                
                // Element-wise minimum calculation
                for(size_t i = 0; i < total_elems; i++) {
                    data_a[i] = MIN(data_a[i], data_b[i]);
                }
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(typea);
                return;
            }
        }
    } else {
        // Handling the case when tensors have different data types
        // Calculating the step size for accessing second tensor elements
        size_t typeb_step = get_dtype_size(typeb);
        
        // Casting second tensor data to char* for byte access
        char* data_b = (char*)tensorb->data;
        
        switch(typea) {
            case FLOAT64: {
                volatile double* data_a = (double*)tensora->data;
                
                // For each element, convert the second tensor element to FLOAT64 and compare
                for(size_t i = 0; i < total_elems; i++) {
                    void* elem_b = data_b + i * typeb_step;
                    double converted_b = nnl2_convert_to_float64(elem_b, typeb);
                    data_a[i] = MIN(data_a[i], converted_b);
                }
                break;
            }
            
            case FLOAT32: {
                volatile float* data_a = (float*)tensora->data;
                
                // For each element, convert the second tensor element to FLOAT32 and compare
                for(size_t i = 0; i < total_elems; i++) {
                    void* elem_b = data_b + i * typeb_step;
                    float converted_b = nnl2_convert_to_float32(elem_b, typeb);
                    data_a[i] = MIN(data_a[i], converted_b);
                }
                break;
            }
            
            case INT32: {
                volatile int32_t* data_a = (int32_t*)tensora->data;
                
                // For each element, convert the second tensor element to INT32 and compare
                for(size_t i = 0; i < total_elems; i++) {
                    void* elem_b = data_b + i * typeb_step;
                    int32_t converted_b = nnl2_convert_to_int32(elem_b, typeb);
                    data_a[i] = MIN(data_a[i], converted_b);
                }
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(typea);
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
 * @brief Backend implementations for mininplace operation
 * @details
 * Array follows the common backend registration pattern for element-wise minimum operations.
 * Currently registered backends:
 *  - nnl2_naive: Basic reference implementation
 * 
 * @see nnl2_naive
 */
Implementation mininplace_backends[] = {
	REGISTER_BACKEND(naive_mininplace, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for mininplace operation
 * @ingroup backend_system 
 */
mininplacefn mininplace;

/** 
 * @brief Makes the mininplace backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(mininplace);

/** 
 * @brief Sets the backend for mininplace operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see ESET_BACKEND_BY_NAME
 */
void set_mininplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(mininplace_backends, mininplace, backend_name, current_backend(mininplace));
}

/** 
 * @brief Gets the name of the active backend for mininplace operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_mininplace_backend() {
	return current_backend(mininplace);
}

/** 
 * @brief Function declaration for getting all available mininplace backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(mininplace);

/**
 * @brief Function declaration for getting the number of available mininplace backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(mininplace);

#endif /** NNL2_MIN_INPLACE_H **/
