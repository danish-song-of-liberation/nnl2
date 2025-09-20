#ifndef NNL2_BROADCASTING_SUB_INPLACE_H
#define NNL2_BROADCASTING_SUB_INPLACE_H

/** @brief
 * Performs element-wise subtraction with broadcasting (in place)
 *
 ** @details
 * Subtracts subtrahend tensor from minuend tensor with broadcasting support
 *
 ** @param minuend
 * Pointer to minuend tensor (will be modified in place)
 *
 ** @param subtrahend
 * Pointer to subtrahend tensor
 */
void naive_sub_broadcasting_inplace(Tensor* minuend, const Tensor* subtrahend) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks for max safety level 
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX     
        NNL2_CHECK_NULL_IF_ERR_RETURN(minuend, "Minuend tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(subtrahend, "Subtrahend tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(minuend->shape, "Minuend shape is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(subtrahend->shape, "Subtrahend shape is NULL");
    #endif
    
    // Calculate the total number of elements in each tensor
    size_t numel_minuend = product(minuend->shape, minuend->rank);
    size_t numel_subtrahend = product(subtrahend->shape, subtrahend->rank);
    
    // Getting the tensor data types
    TensorType minuend_dtype = minuend->dtype;
    TensorType subtrahend_dtype = subtrahend->dtype;
    
    // Checking the possibility of broadcasting (numel_minuend must be a multiple of numel_subtrahend)
    if((numel_minuend % numel_subtrahend) == 0) {
        // Handling the case where the data types match (more efficiently)
        if(minuend_dtype == subtrahend_dtype) {
            switch(minuend_dtype) {
                case FLOAT64: {
                    double* cast_minuend_data = (double*)minuend->data;
                    double* cast_subtrahend_data = (double*)subtrahend->data;
                    
                    for(size_t i = 0; i < (numel_minuend / numel_subtrahend); i++) {
                        for(size_t j = 0; j < numel_subtrahend; j++) {
                            cast_minuend_data[i * numel_subtrahend + j] -= cast_subtrahend_data[j];
                        }
                    }
                    
                    break;
                }
                
                case FLOAT32: {
                    float* cast_minuend_data = (float*)minuend->data;
                    float* cast_subtrahend_data = (float*)subtrahend->data;
                    
                    for(size_t i = 0; i < (numel_minuend / numel_subtrahend); i++) {
                        for(size_t j = 0; j < numel_subtrahend; j++) {
                            cast_minuend_data[i * numel_subtrahend + j] -= cast_subtrahend_data[j];
                        }
                    }
                    
                    break;
                }
                
                case INT32: {
                    int32_t* cast_minuend_data = (int32_t*)minuend->data;
                    int32_t* cast_subtrahend_data = (int32_t*)subtrahend->data;
                    
                    for(size_t i = 0; i < (numel_minuend / numel_subtrahend); i++) {
                        for(size_t j = 0; j < numel_subtrahend; j++) {
                            cast_minuend_data[i * numel_subtrahend + j] -= cast_subtrahend_data[j];
                        }
                    }
                    
                    break;
                }
                
                default: {
                    NNL2_TYPE_ERROR(minuend_dtype);
                    return;
                }
            }    
        } else {
            // Handling a case with different data types (conversion required)
            size_t subtrahend_step = get_dtype_size(subtrahend_dtype); // The size of the element in bytes
            char* subtrahend_data = (char*)subtrahend->data; // Byte pointer for accessing data
            
            switch(minuend_dtype) {
                case FLOAT64: {
                    double* data_minuend = (double*)minuend->data;
                
                    for(size_t i = 0; i < (numel_minuend / numel_subtrahend); i++) {
                        for(size_t j = 0; j < numel_subtrahend; j++) {
                            // Get a pointer to the subtrahend element and convert its type
                            void* subtrahend_elem = subtrahend_data + j * subtrahend_step;
                            data_minuend[i * numel_subtrahend + j] -= nnl2_convert_to_float64(subtrahend_elem, subtrahend_dtype);
                        }
                    }
                
                    break; 
                }
                
                case FLOAT32: {
                    float* data_minuend = (float*)minuend->data;
                
                    for(size_t i = 0; i < (numel_minuend / numel_subtrahend); i++) {
                        for(size_t j = 0; j < numel_subtrahend; j++) {
                            void* subtrahend_elem = subtrahend_data + j * subtrahend_step;
                            data_minuend[i * numel_subtrahend + j] -= nnl2_convert_to_float32(subtrahend_elem, subtrahend_dtype);
                        }
                    }
                
                    break; 
                }
                
                case INT32: {
                    int32_t* data_minuend = (int32_t*)minuend->data;
                
                    for(size_t i = 0; i < (numel_minuend / numel_subtrahend); i++) {
                        for(size_t j = 0; j < numel_subtrahend; j++) {
                            void* subtrahend_elem = subtrahend_data + j * subtrahend_step;
                            data_minuend[i * numel_subtrahend + j] -= nnl2_convert_to_int32(subtrahend_elem, subtrahend_dtype);
                        }
                    }
                
                    break; 
                }
                
                default: {
                    NNL2_TYPE_ERROR(minuend_dtype);
                    return;
                }
            }
        }
    } 
    
    else {
        NNL2_ERROR("Cannot broadcast subtrahend tensor");
        return;
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for subtraction with broadcasting (in place)
 * @details
 * Array follows the common backend registration pattern for subtraction
 * with broadcasting operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for subtraction with broadcasting
 * 
 * @see nnl2_naive
 * @see naive_sub_broadcasting_inplace
 */
Implementation sub_broadcasting_inplace_backends[] = {
    REGISTER_BACKEND(naive_sub_broadcasting_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for subtraction with broadcasting operation (in place)
 * @ingroup backend_system
 */
subbroadcastinginplacefn sub_broadcasting_inplace;

/**
 * @brief Sets the backend for subtraction with broadcasting operation (in place)
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for subtraction with broadcasting
 * @see SET_BACKEND_BY_NAME
 * @see sub_broadcasting_inplace_backends
 */
void set_sub_broadcasting_inplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(sub_broadcasting_inplace_backends, sub_broadcasting_inplace, backend_name);
}

#endif /** NNL2_BROADCASTING_SUB_INPLACE_H **/
