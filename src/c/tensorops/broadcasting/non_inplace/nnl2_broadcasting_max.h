#ifndef NNL2_BROADCASTING_MAX_H
#define NNL2_BROADCASTING_MAX_H

/** @brief
 * Performs element-wise maximum with broadcasting support
 *
 ** @param x
 * First tensor
 *
 ** @param y
 * Second tensor
 * 
 ** @return
 * New tensor containing the result of maximum operation
 *
 ** @note
 * Contains type conversion
 */
Tensor* naive_max_broadcasting(Tensor* x, Tensor* y) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks for max safety level 
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(x, "X tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(y, "Y tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(x->shape, "X shape is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(y->shape, "Y shape is NULL", NULL);
    #endif
 
    size_t numel_x = product(x->shape, x->rank);
    size_t numel_y = product(y->shape, y->rank);
    
    // Getting the tensor data types
    TensorType x_dtype = x->dtype;
    TensorType y_dtype = y->dtype;
    
    TensorType winner_in_the_type_hierarchy = MAX(x_dtype, y_dtype);
    
    // Сreating a resultant tensor
    Tensor* result = nnl2_empty(x->shape, x->rank, winner_in_the_type_hierarchy);

    if((numel_x % numel_y) == 0) {
        if(x_dtype == y_dtype) {
            // Handling the case where the data types match (more efficiently)
            switch(x_dtype) {
                case FLOAT64: {
                    double* cast_x_data = (double*)x->data;
                    double* cast_y_data = (double*)y->data;
                    double* cast_result_data = (double*)result->data;

                    for(size_t i = 0; i < (numel_x / numel_y); i++) {
                        for(size_t j = 0; j < numel_y; j++) {
                            cast_result_data[i * numel_y + j] = MAX(cast_x_data[i * numel_y + j], cast_y_data[j]);
                        }
                    }

                    break;
                }

                case FLOAT32: {
                    float* cast_x_data = (float*)x->data;
                    float* cast_y_data = (float*)y->data;
                    float* cast_result_data = (float*)result->data;

                    for(size_t i = 0; i < (numel_x / numel_y); i++) {
                        for(size_t j = 0; j < numel_y; j++) {
                            cast_result_data[i * numel_y + j] = MAX(cast_x_data[i * numel_y + j], cast_y_data[j]);
                        }
                    }

                    break;
                }

                case INT32: {
                    int32_t* cast_x_data = (int32_t*)x->data;
                    int32_t* cast_y_data = (int32_t*)y->data;
                    int32_t* cast_result_data = (int32_t*)result->data;

                    for(size_t i = 0; i < (numel_x / numel_y); i++) {
                        for(size_t j = 0; j < numel_y; j++) {
                            cast_result_data[i * numel_y + j] = MAX(cast_x_data[i * numel_y + j], cast_y_data[j]);
                        }
                    }

                    break;
                }

                default: {
                    NNL2_TYPE_ERROR(x_dtype);
                    return NULL;
                }
            }
            
            #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                NNL2_FUNC_EXIT();
            #endif
            
            return result;
        } else {
            // Handling a case with different data types (conversion required)
            size_t x_step = get_dtype_size(x_dtype);
            size_t y_step = get_dtype_size(y_dtype);
            
            switch(winner_in_the_type_hierarchy) {
                case FLOAT64: {
                    double* cast_data_result = (double*)result->data;
                    
                    char* cast_x_data = (char*)x->data;
                    char* cast_y_data =  (char*)y->data;
                    
                    for(size_t i = 0; i < (numel_x / numel_y); i++) {
                        for(size_t j = 0; j < numel_y; j++) {
                            void* elem_x = cast_x_data + (i * numel_y + j) * x_step;
                            void* elem_y = cast_y_data + j * y_step; 
                            
                            double x_val = nnl2_convert_to_float64(elem_x, x_dtype);
                            double y_val = nnl2_convert_to_float64(elem_y, y_dtype);
                            cast_data_result[i * numel_y + j] = MAX(x_val, y_val);
                        }
                    }
                    
                    break;
                }
                
                case FLOAT32: {
                    float* cast_data_result = (float*)result->data;
                    
                    char* cast_x_data = (char*)x->data;
                    char* cast_y_data =  (char*)y->data;
                    
                    for(size_t i = 0; i < (numel_x / numel_y); i++) {
                        for(size_t j = 0; j < numel_y; j++) {
                            void* elem_x = cast_x_data + (i * numel_y + j) * x_step;
                            void* elem_y = cast_y_data + j * y_step;
                            
                            float x_val = nnl2_convert_to_float32(elem_x, x_dtype);
                            float y_val = nnl2_convert_to_float32(elem_y, y_dtype);
                            cast_data_result[i * numel_y + j] = MAX(x_val, y_val);
                        }
                    }
                    
                    break;
                }
                
                case INT32: {
                    int32_t* cast_data_result = (int32_t*)result->data;
                    
                    char* cast_x_data = (char*)x->data;
                    char* cast_y_data =  (char*)y->data;
                    
                    for(size_t i = 0; i < (numel_x / numel_y); i++) {                    
                        for(size_t j = 0; j < numel_y; j++) {
                            void* elem_x = cast_x_data + (i * numel_y + j) * x_step;
                            void* elem_y = cast_y_data + j * y_step;
                        
                            int32_t x_val = nnl2_convert_to_int32(elem_x, x_dtype);
                            int32_t y_val = nnl2_convert_to_int32(elem_y, y_dtype);
                            cast_data_result[i * numel_y + j] = MAX(x_val, y_val);
                        }
                    }
                    
                    break;
                }
                
                default: {
                    NNL2_TYPE_ERROR(winner_in_the_type_hierarchy);
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
        NNL2_ERROR("Cannot broadcast y tensor");
        return NULL;
    }
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for maximum operation with broadcasting
 */
Implementation max_broadcasting_backends[] = {
    REGISTER_BACKEND(naive_max_broadcasting, nnl2_naive, NAIVE_BACKEND_NAME),
};  

/**
 * @brief Function pointer for maximum operation with broadcasting
 * @ingroup backend_system
 */
maxbroadcastingfn max_broadcasting;

/**
 * @brief Sets the backend for maximum operation with broadcasting
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for maximum operation with broadcasting
 */
void set_max_broadcasting_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(max_broadcasting_backends, max_broadcasting, backend_name);
}

#endif /** NNL2_BROADCASTING_MAX_H **/
