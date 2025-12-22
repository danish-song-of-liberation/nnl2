#ifndef NNL2_BROADCASTING_ATAN2_H
#define NNL2_BROADCASTING_ATAN2_H

/** @brief
 * Performs element-wise atan2 operation with broadcasting support
 *
 ** @param y
 * y-coordinate tensor (numerator)
 *
 ** @param x
 * x-coordinate tensor (denominator)
 * 
 ** @return
 * New tensor containing the result of atan2 operation
 *
 ** @note
 * Contains type conversion
 */
nnl2_tensor* naive_atan2_broadcasting(nnl2_tensor* y, nnl2_tensor* x) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(y, "y tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(x, "x tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(y->shape, "y shape is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(x->shape, "x shape is NULL", NULL);
    #endif
 
    size_t numel_y = nnl2_product(y->shape, y->rank);
    size_t numel_x = nnl2_product(x->shape, x->rank);
    
    nnl2_tensor_type y_dtype = y->dtype;
    nnl2_tensor_type x_dtype = x->dtype;
    
    nnl2_tensor_type result_dtype = MAX(y_dtype, x_dtype);
    
    nnl2_tensor* result = nnl2_empty(y->shape, y->rank, result_dtype);

    if((numel_y % numel_x) == 0) {
        if(y_dtype == x_dtype) {
            switch(y_dtype) {
                case FLOAT64: {
                    double* cast_y_data = (double*)y->data;
                    double* cast_x_data = (double*)x->data;
                    double* cast_result_data = (double*)result->data;

                    for(size_t i = 0; i < (numel_y / numel_x); i++) {
                        for(size_t j = 0; j < numel_x; j++) {
                            cast_result_data[i * numel_x + j] = atan2(cast_y_data[i * numel_x + j], cast_x_data[j]);
                        }
                    }

                    break;
                }

                case FLOAT32: {
                    float* cast_y_data = (float*)y->data;
                    float* cast_x_data = (float*)x->data;
                    float* cast_result_data = (float*)result->data;

                    for(size_t i = 0; i < (numel_y / numel_x); i++) {
                        for(size_t j = 0; j < numel_x; j++) {
                            cast_result_data[i * numel_x + j] = atan2f(cast_y_data[i * numel_x + j], cast_x_data[j]);
                        }
                    }

                    break;
                }

                case INT32: {
                    int32_t* cast_y_data = (int32_t*)y->data;
                    int32_t* cast_x_data = (int32_t*)x->data;
                    double* cast_result_data = (double*)result->data;

                    for(size_t i = 0; i < (numel_y / numel_x); i++) {
                        for(size_t j = 0; j < numel_x; j++) {
                            cast_result_data[i * numel_x + j] = atan2((double)cast_y_data[i * numel_x + j], (double)cast_x_data[j]);
                        }
                    }

                    break;
                }

                default: {
                    NNL2_TYPE_ERROR(y_dtype);
                    return NULL;
                }
            }
            
            #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                NNL2_FUNC_EXIT();
            #endif
            
            return result;
        } else {
            size_t y_step = get_dtype_size(y_dtype);
            size_t x_step = get_dtype_size(x_dtype);
            
            switch(result_dtype) {
                case FLOAT64: {
                    double* cast_data_result = (double*)result->data;
                    
                    char* cast_y_data = (char*)y->data;
                    char* cast_x_data = (char*)x->data;
                    
                    for(size_t i = 0; i < (numel_y / numel_x); i++) {
                        for(size_t j = 0; j < numel_x; j++) {
                            void* elem_y = cast_y_data + (i * numel_x + j) * y_step;
                            void* elem_x = cast_x_data + j * x_step; 
                            
                            cast_data_result[i * numel_x + j] = atan2(nnl2_convert_to_float64(elem_y, y_dtype), nnl2_convert_to_float64(elem_x, x_dtype));
                        }
                    }
                    
                    break;
                }
                
                case FLOAT32: {
                    float* cast_data_result = (float*)result->data;
                    
                    char* cast_y_data = (char*)y->data;
                    char* cast_x_data = (char*)x->data;
                    
                    for(size_t i = 0; i < (numel_y / numel_x); i++) {
                        for(size_t j = 0; j < numel_x; j++) {
                            void* elem_y = cast_y_data + (i * numel_x + j) * y_step;
                            void* elem_x = cast_x_data + j * x_step;
                            
                            cast_data_result[i * numel_x + j] = atan2f(nnl2_convert_to_float32(elem_y, y_dtype), nnl2_convert_to_float32(elem_x, x_dtype));
                        }
                    }
                    
                    break;
                }
                
                case INT32: {
                    double* cast_data_result = (double*)result->data;
                    
                    char* cast_y_data = (char*)y->data;
                    char* cast_x_data = (char*)x->data;
                    
                    for(size_t i = 0; i < (numel_y / numel_x); i++) {                    
                        for(size_t j = 0; j < numel_x; j++) {
                            void* elem_y = cast_y_data + (i * numel_x + j) * y_step;
                            void* elem_x = cast_x_data + j * x_step;
                        
                            cast_data_result[i * numel_x + j] = atan2((double)nnl2_convert_to_int32(elem_y, y_dtype), (double)nnl2_convert_to_int32(elem_x, x_dtype));
                        }
                    }
                    
                    break;
                }
                
                default: {
                    NNL2_TYPE_ERROR(result_dtype);
                    nnl2_free_tensor(result);
                    return NULL;
                }
            }
            
            #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                NNL2_FUNC_EXIT();
            #endif
            
            return result;
        }
    } else {
        NNL2_ERROR("Cannot broadcast x tensor");
        return NULL;
    }
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for atan2 operation with broadcasting
 */
nnl2_runtime_implementation atan2_broadcasting_backends[] = {
    REGISTER_BACKEND(naive_atan2_broadcasting, nnl2_naive, NAIVE_BACKEND_NAME),
};  

/**
 * @brief Function pointer for atan2 operation with broadcasting
 * @ingroup backend_system
 */
atan2broadcastingfn nnl2_atan2_broadcasting;

/**
 * @brief Sets the backend for atan2 operation with broadcasting
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for atan2 operation with broadcasting
 */
void set_atan2_broadcasting_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(atan2_broadcasting_backends, nnl2_atan2_broadcasting, backend_name);
}

#endif /** NNL2_BROADCASTING_ATAN2_H **/
