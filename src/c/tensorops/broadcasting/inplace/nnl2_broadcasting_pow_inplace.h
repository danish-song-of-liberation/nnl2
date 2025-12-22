#ifndef NNL2_BROADCASTING_POW_INPLACE_H
#define NNL2_BROADCASTING_POW_INPLACE_H


/** @brief
 * Performs element-wise power operation with broadcasting (in place)
 *
 ** @param base
 * Pointer to base tensor (will be modified in place)
 *
 ** @param exponent
 * Pointer to exponent tensor
 */
void naive_pow_broadcasting_inplace(nnl2_tensor* base, const nnl2_tensor* exponent) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks for max safety level 
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(base, "Base tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(exponent, "Exponent tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(base->shape, "Base shape is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(exponent->shape, "Exponent shape is NULL");
    #endif
    
    size_t numel_base = product(base->shape, base->rank);
    size_t numel_exponent = product(exponent->shape, exponent->rank);
    
    // Getting the tensor data types
    nnl2_tensor_type base_dtype = base->dtype;
    nnl2_tensor_type exponent_dtype = exponent->dtype;

    if((numel_base % numel_exponent) == 0) {
        // Handling the case where the data types match (more efficiently)
        if(base_dtype == exponent_dtype) {
            switch(base_dtype) {
                case FLOAT64: {
                    double* cast_base_data = (double*)base->data;
                    double* cast_exponent_data = (double*)exponent->data;

                    for(size_t i = 0; i < (numel_base / numel_exponent); i++) {
                        for(size_t j = 0; j < numel_exponent; j++) {
                            cast_base_data[i * numel_exponent + j] = pow(cast_base_data[i * numel_exponent + j], cast_exponent_data[j]);
                        }
                    }

                    break;
                }

                case FLOAT32: {
                    float* cast_base_data = (float*)base->data;
                    float* cast_exponent_data = (float*)exponent->data;

                    for(size_t i = 0; i < (numel_base / numel_exponent); i++) {
                        for(size_t j = 0; j < numel_exponent; j++) {
                            cast_base_data[i * numel_exponent + j] = powf(cast_base_data[i * numel_exponent + j], cast_exponent_data[j]);
                        }
                    }

                    break;
                }

                case INT32: {
                    int32_t* cast_base_data = (int32_t*)base->data;
                    int32_t* cast_exponent_data = (int32_t*)exponent->data;

                    for(size_t i = 0; i < (numel_base / numel_exponent); i++) {
                        for(size_t j = 0; j < numel_exponent; j++) {
                            cast_base_data[i * numel_exponent + j] = (int32_t)pow((double)cast_base_data[i * numel_exponent + j], (double)cast_exponent_data[j]);
                        }
                    }

                    break;
                }

                default: {
                    NNL2_TYPE_ERROR(base_dtype);
                    return;
                }
            }
        } else {
            // Handling a case with different data types (conversion required)
            size_t exponent_step = get_dtype_size(exponent_dtype);
            char* exponent_data = (char*)exponent->data;
            
            switch(base_dtype) {
                case FLOAT64: {
                    double* data_base = (double*)base->data;
                
                    for(size_t i = 0; i < (numel_base / numel_exponent); i++) {
                        for(size_t j = 0; j < numel_exponent; j++) {
                            void* exponent_elem = exponent_data + j * exponent_step;
                            data_base[i * numel_exponent + j] = pow(data_base[i * numel_exponent + j], nnl2_convert_to_float64(exponent_elem, exponent_dtype));
                        }
                    }
                
                    break; 
                }
                
                case FLOAT32: {
                    float* data_base = (float*)base->data;
                
                    for(size_t i = 0; i < (numel_base / numel_exponent); i++) {
                        for(size_t j = 0; j < numel_exponent; j++) {
                            void* exponent_elem = exponent_data + j * exponent_step;
                            data_base[i * numel_exponent + j] = powf(data_base[i * numel_exponent + j], nnl2_convert_to_float32(exponent_elem, exponent_dtype));
                        }
                    }
                
                    break; 
                }
                
                case INT32: {
                    int32_t* data_base = (int32_t*)base->data;
                
                    for(size_t i = 0; i < (numel_base / numel_exponent); i++) {
                        for(size_t j = 0; j < numel_exponent; j++) {
                            void* exponent_elem = exponent_data + j * exponent_step;
                            data_base[i * numel_exponent + j] = (int32_t)pow((double)data_base[i * numel_exponent + j], (double)nnl2_convert_to_int32(exponent_elem, exponent_dtype));
                        }
                    }
                
                    break; 
                }
                
                default: {
                    NNL2_TYPE_ERROR(base_dtype);
                    return;
                }
            }
        }
    } else {
        NNL2_ERROR("Cannot broadcast exponent tensor");
        return;
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for power operation with broadcasting (in place)
 */
nnl2_runtime_implementation pow_broadcasting_inplace_backends[] = {
    REGISTER_BACKEND(naive_pow_broadcasting_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};  

/**
 * @brief Function pointer for power operation with broadcasting (in place)
 * @ingroup backend_system
 */
powbroadcastinginplacefn pow_broadcasting_inplace;

/**
 * @brief Sets the backend for power operation with broadcasting (in place)
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for power operation with broadcasting
 */
void set_pow_broadcasting_inplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(pow_broadcasting_inplace_backends, pow_broadcasting_inplace, backend_name);
}

#endif /** NNL2_BROADCASTING_POW_INPLACE_H **/
