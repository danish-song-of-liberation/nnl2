#ifndef NNL2_BROADCASTING_POW_H
#define NNL2_BROADCASTING_POW_H

/** @brief
 * Performs element-wise power operation with broadcasting support
 *
 ** @param base
 * Base tensor
 *
 ** @param exponent
 * Exponent tensor
 * 
 ** @return
 * New tensor containing the result of power operation
 *
 ** @note
 * Contains type conversion
 */
nnl2_tensor* naive_pow_broadcasting(nnl2_tensor* base, nnl2_tensor* exponent) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks for max safety level 
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(base, "Base tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(exponent, "Exponent tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(base->shape, "Base shape is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(exponent->shape, "Exponent shape is NULL", NULL);
    #endif
 
    size_t numel_base = nnl2_product(base->shape, base->rank);
    size_t numel_exponent = nnl2_product(exponent->shape, exponent->rank);
    
    // Getting the tensor data types
    nnl2_tensor_type base_dtype = base->dtype;
    nnl2_tensor_type exponent_dtype = exponent->dtype;
    
    nnl2_tensor_type winner_in_the_type_hierarchy = MAX(base_dtype, exponent_dtype);
    
    // Сreating a resultant tensor
    nnl2_tensor* result = nnl2_empty(base->shape, base->rank, winner_in_the_type_hierarchy);

    if((numel_base % numel_exponent) == 0) {
        if(base_dtype == exponent_dtype) {
            // Handling the case where the data types match (more efficiently)
            switch(base_dtype) {
                case FLOAT64: {
                    double* cast_base_data = (double*)base->data;
                    double* cast_exponent_data = (double*)exponent->data;
                    double* cast_result_data = (double*)result->data;

                    for(size_t i = 0; i < (numel_base / numel_exponent); i++) {
                        for(size_t j = 0; j < numel_exponent; j++) {
                            cast_result_data[i * numel_exponent + j] = pow(cast_base_data[i * numel_exponent + j], cast_exponent_data[j]);
                        }
                    }

                    break;
                }

                case FLOAT32: {
                    float* cast_base_data = (float*)base->data;
                    float* cast_exponent_data = (float*)exponent->data;
                    float* cast_result_data = (float*)result->data;

                    for(size_t i = 0; i < (numel_base / numel_exponent); i++) {
                        for(size_t j = 0; j < numel_exponent; j++) {
                            cast_result_data[i * numel_exponent + j] = powf(cast_base_data[i * numel_exponent + j], cast_exponent_data[j]);
                        }
                    }

                    break;
                }

                case INT32: {
                    int32_t* cast_base_data = (int32_t*)base->data;
                    int32_t* cast_exponent_data = (int32_t*)exponent->data;
                    int32_t* cast_result_data = (int32_t*)result->data;

                    for(size_t i = 0; i < (numel_base / numel_exponent); i++) {
                        for(size_t j = 0; j < numel_exponent; j++) {
                            cast_result_data[i * numel_exponent + j] = (int32_t)pow((double)cast_base_data[i * numel_exponent + j], (double)cast_exponent_data[j]);
                        }
                    }

                    break;
                }

                default: {
                    NNL2_TYPE_ERROR(base_dtype);
                    return NULL;
                }
            }
            
            #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                NNL2_FUNC_EXIT();
            #endif
            
            return result;
        } else {
            // Handling a case with different data types (conversion required)
            size_t base_step = get_dtype_size(base_dtype);
            size_t exponent_step = get_dtype_size(exponent_dtype);
            
            switch(winner_in_the_type_hierarchy) {
                case FLOAT64: {
                    double* cast_data_result = (double*)result->data;
                    
                    char* cast_base_data = (char*)base->data;
                    char* cast_exponent_data =  (char*)exponent->data;
                    
                    for(size_t i = 0; i < (numel_base / numel_exponent); i++) {
                        for(size_t j = 0; j < numel_exponent; j++) {
                            void* elem_base = cast_base_data + (i * numel_exponent + j) * base_step;
                            void* elem_exponent = cast_exponent_data + j * exponent_step; 
                            
                            cast_data_result[i * numel_exponent + j] = pow(nnl2_convert_to_float64(elem_base, base_dtype), nnl2_convert_to_float64(elem_exponent, exponent_dtype));
                        }
                    }
                    
                    break;
                }
                
                case FLOAT32: {
                    float* cast_data_result = (float*)result->data;
                    
                    char* cast_base_data = (char*)base->data;
                    char* cast_exponent_data =  (char*)exponent->data;
                    
                    for(size_t i = 0; i < (numel_base / numel_exponent); i++) {
                        for(size_t j = 0; j < numel_exponent; j++) {
                            void* elem_base = cast_base_data + (i * numel_exponent + j) * base_step;
                            void* elem_exponent = cast_exponent_data + j * exponent_step;
                            
                            cast_data_result[i * numel_exponent + j] = powf(nnl2_convert_to_float32(elem_base, base_dtype), nnl2_convert_to_float32(elem_exponent, exponent_dtype));
                        }
                    }
                    
                    break;
                }
                
                case INT32: {
                    int32_t* cast_data_result = (int32_t*)result->data;
                    
                    char* cast_base_data = (char*)base->data;
                    char* cast_exponent_data =  (char*)exponent->data;
                    
                    for(size_t i = 0; i < (numel_base / numel_exponent); i++) {                    
                        for(size_t j = 0; j < numel_exponent; j++) {
                            void* elem_base = cast_base_data + (i * numel_exponent + j) * base_step;
                            void* elem_exponent = cast_exponent_data + j * exponent_step;
                        
                            cast_data_result[i * numel_exponent + j] = (int32_t)pow((double)nnl2_convert_to_int32(elem_base, base_dtype), (double)nnl2_convert_to_int32(elem_exponent, exponent_dtype));
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
        NNL2_ERROR("Cannot broadcast exponent tensor");
        return NULL;
    }
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for power operation with broadcasting
 */
nnl2_runtime_implementation pow_broadcasting_backends[] = {
    REGISTER_BACKEND(naive_pow_broadcasting, nnl2_naive, NAIVE_BACKEND_NAME),
};  

/**
 * @brief Function pointer for power operation with broadcasting
 * @ingroup backend_system
 */
powbroadcastingfn pow_broadcasting;

/**
 * @brief Sets the backend for power operation with broadcasting
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for power operation with broadcasting
 */
void set_pow_broadcasting_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(pow_broadcasting_backends, pow_broadcasting, backend_name);
}

#endif /** NNL2_BROADCASTING_POW_H **/
