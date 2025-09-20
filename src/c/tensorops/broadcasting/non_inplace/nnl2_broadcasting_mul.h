#ifndef NNL2_BROADCASTING_MUL_H
#define NNL2_BROADCASTING_MUL_H

/** @brief
 * Performs element-wise multiplication with broadcasting support
 *
 ** @param multiplicand
 * First tensor to multiply
 *
 ** @param multiplier
 * Second tensor to multiply
 * 
 ** @return
 * New tensor containing the result of multiplication
 *
 ** @note
 * Contains type conversion
 */
Tensor* naive_mul_broadcasting(Tensor* multiplicand, Tensor* multiplier) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks for max safety level 
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(multiplicand, "Multiplicand tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(multiplier, "Multiplier tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(multiplicand->shape, "Multiplicand shape is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(multiplier->shape, "Multiplier shape is NULL", NULL);
    #endif
 
    size_t numel_multiplicand = product(multiplicand->shape, multiplicand->rank);
    size_t numel_multiplier = product(multiplier->shape, multiplier->rank);
    
    // Getting the tensor data types
    TensorType multiplicand_dtype = multiplicand->dtype;
    TensorType multiplier_dtype = multiplier->dtype;
    
    TensorType winner_in_the_type_hierarchy = MAX(multiplicand_dtype, multiplier_dtype);
    
    // Сreating a resultant tensor
    Tensor* result = nnl2_empty(multiplicand->shape, multiplicand->rank, winner_in_the_type_hierarchy);

    if((numel_multiplicand % numel_multiplier) == 0) {
        if(multiplicand_dtype == multiplier_dtype) {
            // Handling the case where the data types match (more efficiently)
            switch(multiplicand_dtype) {
                case FLOAT64: {
                    double* cast_multiplicand_data = (double*)multiplicand->data;
                    double* cast_multiplier_data = (double*)multiplier->data;
                    double* cast_result_data = (double*)result->data;

                    for(size_t i = 0; i < (numel_multiplicand / numel_multiplier); i++) {
                        for(size_t j = 0; j < numel_multiplier; j++) {
                            cast_result_data[i * numel_multiplier + j] = cast_multiplicand_data[i * numel_multiplier + j] * cast_multiplier_data[j];
                        }
                    }

                    break;
                }

                case FLOAT32: {
                    float* cast_multiplicand_data = (float*)multiplicand->data;
                    float* cast_multiplier_data = (float*)multiplier->data;
                    float* cast_result_data = (float*)result->data;

                    for(size_t i = 0; i < (numel_multiplicand / numel_multiplier); i++) {
                        for(size_t j = 0; j < numel_multiplier; j++) {
                            cast_result_data[i * numel_multiplier + j] = cast_multiplicand_data[i * numel_multiplier + j] * cast_multiplier_data[j];
                        }
                    }

                    break;
                }

                case INT32: {
                    int32_t* cast_multiplicand_data = (int32_t*)multiplicand->data;
                    int32_t* cast_multiplier_data = (int32_t*)multiplier->data;
                    int32_t* cast_result_data = (int32_t*)result->data;

                    for(size_t i = 0; i < (numel_multiplicand / numel_multiplier); i++) {
                        for(size_t j = 0; j < numel_multiplier; j++) {
                            cast_result_data[i * numel_multiplier + j] = cast_multiplicand_data[i * numel_multiplier + j] * cast_multiplier_data[j];
                        }
                    }

                    break;
                }

                default: {
                    NNL2_TYPE_ERROR(multiplicand_dtype);
                    return NULL;
                }
            }
            
            #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                NNL2_FUNC_EXIT();
            #endif
            
            return result;
        } else {
            // Handling a case with different data types (conversion required)
            size_t multiplicand_step = get_dtype_size(multiplicand_dtype);
            size_t multiplier_step = get_dtype_size(multiplier_dtype);
            
            switch(winner_in_the_type_hierarchy) {
                case FLOAT64: {
                    double* cast_data_result = (double*)result->data;
                    
                    char* cast_multiplicand_data = (char*)multiplicand->data;
                    char* cast_multiplier_data =  (char*)multiplier->data;
                    
                    for(size_t i = 0; i < (numel_multiplicand / numel_multiplier); i++) {
                        for(size_t j = 0; j < numel_multiplier; j++) {
                            void* elem_multiplicand = cast_multiplicand_data + (i * numel_multiplier + j) * multiplicand_step;
                            void* elem_multiplier = cast_multiplier_data + j * multiplier_step; 
                            
                            cast_data_result[i * numel_multiplier + j] = nnl2_convert_to_float64(elem_multiplicand, multiplicand_dtype) * nnl2_convert_to_float64(elem_multiplier, multiplier_dtype);
                        }
                    }
                    
                    break;
                }
                
                case FLOAT32: {
                    float* cast_data_result = (float*)result->data;
                    
                    char* cast_multiplicand_data = (char*)multiplicand->data;
                    char* cast_multiplier_data =  (char*)multiplier->data;
                    
                    for(size_t i = 0; i < (numel_multiplicand / numel_multiplier); i++) {
                        for(size_t j = 0; j < numel_multiplier; j++) {
                            void* elem_multiplicand = cast_multiplicand_data + (i * numel_multiplier + j) * multiplicand_step;
                            void* elem_multiplier = cast_multiplier_data + j * multiplier_step;
                            
                            cast_data_result[i * numel_multiplier + j] = nnl2_convert_to_float32(elem_multiplicand, multiplicand_dtype) * nnl2_convert_to_float32(elem_multiplier, multiplier_dtype);
                        }
                    }
                    
                    break;
                }
                
                case INT32: {
                    int32_t* cast_data_result = (int32_t*)result->data;
                    
                    char* cast_multiplicand_data = (char*)multiplicand->data;
                    char* cast_multiplier_data =  (char*)multiplier->data;
                    
                    for(size_t i = 0; i < (numel_multiplicand / numel_multiplier); i++) {                    
                        for(size_t j = 0; j < numel_multiplier; j++) {
                            void* elem_multiplicand = cast_multiplicand_data + (i * numel_multiplier + j) * multiplicand_step;
                            void* elem_multiplier = cast_multiplier_data + j * multiplier_step;
                        
                            cast_data_result[i * numel_multiplier + j] = nnl2_convert_to_int32(elem_multiplicand, multiplicand_dtype) * nnl2_convert_to_int32(elem_multiplier, multiplier_dtype);
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
        NNL2_ERROR("Cannot broadcast multiplier tensor");
        return NULL;
    }
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for multiplication with broadcasting
 */
Implementation mul_broadcasting_backends[] = {
    REGISTER_BACKEND(naive_mul_broadcasting, nnl2_naive, NAIVE_BACKEND_NAME),
};  

/**
 * @brief Function pointer for multiplication with broadcasting operation
 * @ingroup backend_system
 */
mulbroadcastingfn mul_broadcasting;

/**
 * @brief Sets the backend for multiplication with broadcasting operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for multiplication with broadcasting
 */
void set_mul_broadcasting_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(mul_broadcasting_backends, mul_broadcasting, backend_name);
}

#endif /** NNL2_BROADCASTING_MUL_H **/
