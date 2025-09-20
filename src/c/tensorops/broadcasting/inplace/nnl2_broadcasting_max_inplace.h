#ifndef NNL2_BROADCASTING_MAX_INPLACE_H
#define NNL2_BROADCASTING_MAX_INPLACE_H

/** @brief
 * Performs element-wise maximum with broadcasting (in place)
 *
 ** @param x
 * Pointer to first tensor (will be modified in place)
 *
 ** @param y
 * Pointer to second tensor
 */
void naive_max_broadcasting_inplace(Tensor* x, const Tensor* y) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks for max safety level 
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(x, "X tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(y, "Y tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(x->shape, "X shape is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(y->shape, "Y shape is NULL");
    #endif
    
    size_t numel_x = product(x->shape, x->rank);
    size_t numel_y = product(y->shape, y->rank);
    
    // Getting the tensor data types
    TensorType x_dtype = x->dtype;
    TensorType y_dtype = y->dtype;

    if((numel_x % numel_y) == 0) {
        // Handling the case where the data types match (more efficiently)
        if(x_dtype == y_dtype) {
            switch(x_dtype) {
                case FLOAT64: {
                    double* cast_x_data = (double*)x->data;
                    double* cast_y_data = (double*)y->data;

                    for(size_t i = 0; i < (numel_x / numel_y); i++) {
                        for(size_t j = 0; j < numel_y; j++) {
                            cast_x_data[i * numel_y + j] = MAX(cast_x_data[i * numel_y + j], cast_y_data[j]);
                        }
                    }

                    break;
                }

                case FLOAT32: {
                    float* cast_x_data = (float*)x->data;
                    float* cast_y_data = (float*)y->data;

                    for(size_t i = 0; i < (numel_x / numel_y); i++) {
                        for(size_t j = 0; j < numel_y; j++) {
                            cast_x_data[i * numel_y + j] = MAX(cast_x_data[i * numel_y + j], cast_y_data[j]);
                        }
                    }

                    break;
                }

                case INT32: {
                    int32_t* cast_x_data = (int32_t*)x->data;
                    int32_t* cast_y_data = (int32_t*)y->data;

                    for(size_t i = 0; i < (numel_x / numel_y); i++) {
                        for(size_t j = 0; j < numel_y; j++) {
                            cast_x_data[i * numel_y + j] = MAX(cast_x_data[i * numel_y + j], cast_y_data[j]);
                        }
                    }

                    break;
                }

                default: {
                    NNL2_TYPE_ERROR(x_dtype);
                    return;
                }
            }
        } else {
            // Handling a case with different data types (conversion required)
            size_t y_step = get_dtype_size(y_dtype);
            char* y_data = (char*)y->data;
            
            switch(x_dtype) {
                case FLOAT64: {
                    double* data_x = (double*)x->data;
                
                    for(size_t i = 0; i < (numel_x / numel_y); i++) {
                        for(size_t j = 0; j < numel_y; j++) {
                            void* y_elem = y_data + j * y_step;
                            double y_val = nnl2_convert_to_float64(y_elem, y_dtype);
                            data_x[i * numel_y + j] = MAX(data_x[i * numel_y + j], y_val);
                        }
                    }
                
                    break; 
                }
                
                case FLOAT32: {
                    float* data_x = (float*)x->data;
                
                    for(size_t i = 0; i < (numel_x / numel_y); i++) {
                        for(size_t j = 0; j < numel_y; j++) {
                            void* y_elem = y_data + j * y_step;
                            float y_val = nnl2_convert_to_float32(y_elem, y_dtype);
                            data_x[i * numel_y + j] = MAX(data_x[i * numel_y + j], y_val);
                        }
                    }
                
                    break; 
                }
                
                case INT32: {
                    int32_t* data_x = (int32_t*)x->data;
                
                    for(size_t i = 0; i < (numel_x / numel_y); i++) {
                        for(size_t j = 0; j < numel_y; j++) {
                            void* y_elem = y_data + j * y_step;
                            int32_t y_val = nnl2_convert_to_int32(y_elem, y_dtype);
                            data_x[i * numel_y + j] = MAX(data_x[i * numel_y + j], y_val);
                        }
                    }
                
                    break; 
                }
                
                default: {
                    NNL2_TYPE_ERROR(x_dtype);
                    return;
                }
            }
        }
    } else {
        NNL2_ERROR("Cannot broadcast y tensor");
        return;
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for maximum operation with broadcasting (in place)
 */
Implementation max_broadcasting_inplace_backends[] = {
    REGISTER_BACKEND(naive_max_broadcasting_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};  

/**
 * @brief Function pointer for maximum operation with broadcasting (in place)
 * @ingroup backend_system
 */
maxbroadcastinginplacefn max_broadcasting_inplace;

/**
 * @brief Sets the backend for maximum operation with broadcasting (in place)
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for maximum operation with broadcasting
 */
void set_max_broadcasting_inplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(max_broadcasting_inplace_backends, max_broadcasting_inplace, backend_name);
}

#endif /** NNL2_BROADCASTING_MAX_INPLACE_H **/
