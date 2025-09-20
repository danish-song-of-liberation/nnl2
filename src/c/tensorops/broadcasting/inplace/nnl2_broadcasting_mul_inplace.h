#ifndef NNL2_BROADCASTING_MUL_INPLACE_H
#define NNL2_BROADCASTING_MUL_INPLACE_H

/** @brief
 * Performs element-wise multiplication with broadcasting (in place)
 *
 ** @param multiplicand
 * Pointer to multiplicand tensor (will be modified in place)
 *
 ** @param multiplier
 * Pointer to multiplier tensor
 */
void naive_mul_broadcasting_inplace(Tensor* multiplicand, const Tensor* multiplier) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks for max safety level 
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(multiplicand, "Multiplicand tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(multiplier, "Multiplier tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(multiplicand->shape, "Multiplicand shape is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(multiplier->shape, "Multiplier shape is NULL");
    #endif
    
    size_t numel_multiplicand = product(multiplicand->shape, multiplicand->rank);
    size_t numel_multiplier = product(multiplier->shape, multiplier->rank);
    
    // Getting the tensor data types
    TensorType multiplicand_dtype = multiplicand->dtype;
    TensorType multiplier_dtype = multiplier->dtype;

    if((numel_multiplicand % numel_multiplier) == 0) {
        // Handling the case where the data types match (more efficiently)
        if(multiplicand_dtype == multiplier_dtype) {
            switch(multiplicand_dtype) {
                case FLOAT64: {
                    double* cast_multiplicand_data = (double*)multiplicand->data;
                    double* cast_multiplier_data = (double*)multiplier->data;

                    for(size_t i = 0; i < (numel_multiplicand / numel_multiplier); i++) {
                        for(size_t j = 0; j < numel_multiplier; j++) {
                            cast_multiplicand_data[i * numel_multiplier + j] *= cast_multiplier_data[j];
                        }
                    }

                    break;
                }

                case FLOAT32: {
                    float* cast_multiplicand_data = (float*)multiplicand->data;
                    float* cast_multiplier_data = (float*)multiplier->data;

                    for(size_t i = 0; i < (numel_multiplicand / numel_multiplier); i++) {
                        for(size_t j = 0; j < numel_multiplier; j++) {
                            cast_multiplicand_data[i * numel_multiplier + j] *= cast_multiplier_data[j];
                        }
                    }

                    break;
                }

                case INT32: {
                    int32_t* cast_multiplicand_data = (int32_t*)multiplicand->data;
                    int32_t* cast_multiplier_data = (int32_t*)multiplier->data;

                    for(size_t i = 0; i < (numel_multiplicand / numel_multiplier); i++) {
                        for(size_t j = 0; j < numel_multiplier; j++) {
                            cast_multiplicand_data[i * numel_multiplier + j] *= cast_multiplier_data[j];
                        }
                    }

                    break;
                }

                default: {
                    NNL2_TYPE_ERROR(multiplicand_dtype);
                    return;
                }
            }
        } else {
            // Handling a case with different data types (conversion required)
            size_t multiplier_step = get_dtype_size(multiplier_dtype);
            char* multiplier_data = (char*)multiplier->data;
            
            switch(multiplicand_dtype) {
                case FLOAT64: {
                    double* data_multiplicand = (double*)multiplicand->data;
                
                    for(size_t i = 0; i < (numel_multiplicand / numel_multiplier); i++) {
                        for(size_t j = 0; j < numel_multiplier; j++) {
                            void* multiplier_elem = multiplier_data + j * multiplier_step;
                            data_multiplicand[i * numel_multiplier + j] *= nnl2_convert_to_float64(multiplier_elem, multiplier_dtype);
                        }
                    }
                
                    break; 
                }
                
                case FLOAT32: {
                    float* data_multiplicand = (float*)multiplicand->data;
                
                    for(size_t i = 0; i < (numel_multiplicand / numel_multiplier); i++) {
                        for(size_t j = 0; j < numel_multiplier; j++) {
                            void* multiplier_elem = multiplier_data + j * multiplier_step;
                            data_multiplicand[i * numel_multiplier + j] *= nnl2_convert_to_float32(multiplier_elem, multiplier_dtype);
                        }
                    }
                
                    break; 
                }
                
                case INT32: {
                    int32_t* data_multiplicand = (int32_t*)multiplicand->data;
                
                    for(size_t i = 0; i < (numel_multiplicand / numel_multiplier); i++) {
                        for(size_t j = 0; j < numel_multiplier; j++) {
                            void* multiplier_elem = multiplier_data + j * multiplier_step;
                            data_multiplicand[i * numel_multiplier + j] *= nnl2_convert_to_int32(multiplier_elem, multiplier_dtype);
                        }
                    }
                
                    break; 
                }
                
                default: {
                    NNL2_TYPE_ERROR(multiplicand_dtype);
                    return;
                }
            }
        }
    } else {
        NNL2_ERROR("Cannot broadcast multiplier tensor");
        return;
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for multiplication with broadcasting (in place)
 */
Implementation mul_broadcasting_inplace_backends[] = {
    REGISTER_BACKEND(naive_mul_broadcasting_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};  

/**
 * @brief Function pointer for multiplication with broadcasting operation (in place)
 * @ingroup backend_system
 */
mulbroadcastinginplacefn mul_broadcasting_inplace;

/**
 * @brief Sets the backend for multiplication with broadcasting operation (in place)
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for multiplication with broadcasting
 */
void set_mul_broadcasting_inplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(mul_broadcasting_inplace_backends, mul_broadcasting_inplace, backend_name);
}

#endif /** NNL2_BROADCASTING_MUL_INPLACE_H **/
