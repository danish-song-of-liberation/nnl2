#ifndef NNL2_BROADCASTING_ATAN2_INPLACE_H
#define NNL2_BROADCASTING_ATAN2_INPLACE_H

/** @brief
 * Performs element-wise atan2 operation with broadcasting (in place)
 *
 ** @param y
 * Pointer to y-coordinate tensor (will be modified in place)
 *
 ** @param x
 * Pointer to x-coordinate tensor
 */
void naive_atan2_broadcasting_inplace(nnl2_tensor* y, const nnl2_tensor* x) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(y, "y tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(x, "x tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(y->shape, "y shape is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(x->shape, "x shape is NULL");
    #endif
    
    size_t numel_y = nnl2_product(y->shape, y->rank);
    size_t numel_x = nnl2_product(x->shape, x->rank);
    
    nnl2_tensor_type y_dtype = y->dtype;
    nnl2_tensor_type x_dtype = x->dtype;

    if((numel_y % numel_x) == 0) {
        if(y_dtype == x_dtype) {
            switch(y_dtype) {
                case FLOAT64: {
                    double* data_y = (double*)y->data;
                    double* data_x = (double*)x->data;

                    for(size_t i = 0; i < (numel_y / numel_x); i++) {
                        for(size_t j = 0; j < numel_x; j++) {
                            data_y[i * numel_x + j] = atan2(data_y[i * numel_x + j], data_x[j]);
                        }
                    }

                    break;
                }

                case FLOAT32: {
                    float* data_y = (float*)y->data;
                    float* data_x = (float*)x->data;

                    for(size_t i = 0; i < (numel_y / numel_x); i++) {
                        for(size_t j = 0; j < numel_x; j++) {
                            data_y[i * numel_x + j] = atan2f(data_y[i * numel_x + j], data_x[j]);
                        }
                    }

                    break;
                }

                case INT32: {
                    int32_t* data_y = (int32_t*)y->data;
                    int32_t* data_x = (int32_t*)x->data;

                    for(size_t i = 0; i < (numel_y / numel_x); i++) {
                        for(size_t j = 0; j < numel_x; j++) {
                            if (data_y[i * numel_x + j] != 0 || data_x[j] != 0) {
                                NNL2_FATAL("Can't apply atan2 to non-zero INT32 tensors");
                            }
                        }
                    }

                    break;
                }

                default: {
                    NNL2_TYPE_ERROR(y_dtype);
                    return;
                }
            }
        } else {
            size_t x_step = get_dtype_size(x_dtype);
            char* x_data = (char*)x->data;
            
            switch(y_dtype) {
                case FLOAT64: {
                    double* data_y = (double*)y->data;
                
                    for(size_t i = 0; i < (numel_y / numel_x); i++) {
                        for(size_t j = 0; j < numel_x; j++) {
                            void* x_elem = x_data + j * x_step;
                            data_y[i * numel_x + j] = atan2(data_y[i * numel_x + j], nnl2_convert_to_float64(x_elem, x_dtype));
                        }
                    }
                
                    break; 
                }
                
                case FLOAT32: {
                    float* data_y = (float*)y->data;
                
                    for(size_t i = 0; i < (numel_y / numel_x); i++) {
                        for(size_t j = 0; j < numel_x; j++) {
                            void* x_elem = x_data + j * x_step;
                            data_y[i * numel_x + j] = atan2f(data_y[i * numel_x + j], nnl2_convert_to_float32(x_elem, x_dtype));
                        }
                    }
                
                    break; 
                }
                
                case INT32: {
                    int32_t* data_y = (int32_t*)y->data;
                
                    for(size_t i = 0; i < (numel_y / numel_x); i++) {
                        for(size_t j = 0; j < numel_x; j++) {
                            void* x_elem = x_data + j * x_step;
                            int32_t x_val = nnl2_convert_to_int32(x_elem, x_dtype);
                            
                            if (data_y[i * numel_x + j] != 0 || x_val != 0) {
                                NNL2_FATAL("Can't apply atan2 to non-zero INT32 tensors");
                            }
                        }
                    }
                
                    break; 
                }
                
                default: {
                    NNL2_TYPE_ERROR(y_dtype);
                    return;
                }
            }
        }
    } else {
        NNL2_ERROR("Cannot broadcast x tensor");
        return;
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for atan2 operation with broadcasting (in place)
 */
nnl2_runtime_implementation atan2_broadcasting_inplace_backends[] = {
    REGISTER_BACKEND(naive_atan2_broadcasting_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};  

/**
 * @brief Function pointer for atan2 operation with broadcasting (in place)
 * @ingroup backend_system
 */
atan2broadcastinginplacefn nnl2_atan2_broadcasting_inplace;

/**
 * @brief Sets the backend for atan2 operation with broadcasting (in place)
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for atan2 operation with broadcasting
 */
void set_atan2inplace_broadcasting_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(atan2_broadcasting_inplace_backends, nnl2_atan2_broadcasting_inplace, backend_name);
}

#endif /** NNL2_BROADCASTING_ATAN2_INPLACE_H **/
