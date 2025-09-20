#ifndef NNL2_BROADCASTING_DIV_H
#define NNL2_BROADCASTING_DIV_H

/** @brief
 * Performs element-wise division with broadcasting support
 *
 ** @param dividend
 * First tensor to divide
 *
 ** @param divisor
 * Second tensor to divide by
 * 
 ** @return
 * New tensor containing the result of division
 *
 ** @note
 * Contains type conversion
 */
Tensor* naive_div_broadcasting(Tensor* dividend, Tensor* divisor) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks for max safety level 
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(dividend, "Dividend tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(divisor, "Divisor tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(dividend->shape, "Dividend shape is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(divisor->shape, "Divisor shape is NULL", NULL);
    #endif
 
    size_t numel_dividend = product(dividend->shape, dividend->rank);
    size_t numel_divisor = product(divisor->shape, divisor->rank);
    
    // Getting the tensor data types
    TensorType dividend_dtype = dividend->dtype;
    TensorType divisor_dtype = divisor->dtype;
    
    TensorType winner_in_the_type_hierarchy = MAX(dividend_dtype, divisor_dtype);
    
    // Сreating a resultant tensor
    Tensor* result = nnl2_empty(dividend->shape, dividend->rank, winner_in_the_type_hierarchy);

    if((numel_dividend % numel_divisor) == 0) {
        if(dividend_dtype == divisor_dtype) {
            // Handling the case where the data types match (more efficiently)
            switch(dividend_dtype) {
                case FLOAT64: {
                    double* cast_dividend_data = (double*)dividend->data;
                    double* cast_divisor_data = (double*)divisor->data;
                    double* cast_result_data = (double*)result->data;

                    for(size_t i = 0; i < (numel_dividend / numel_divisor); i++) {
                        for(size_t j = 0; j < numel_divisor; j++) {
                            cast_result_data[i * numel_divisor + j] = cast_dividend_data[i * numel_divisor + j] / cast_divisor_data[j];
                        }
                    }

                    break;
                }

                case FLOAT32: {
                    float* cast_dividend_data = (float*)dividend->data;
                    float* cast_divisor_data = (float*)divisor->data;
                    float* cast_result_data = (float*)result->data;

                    for(size_t i = 0; i < (numel_dividend / numel_divisor); i++) {
                        for(size_t j = 0; j < numel_divisor; j++) {
                            cast_result_data[i * numel_divisor + j] = cast_dividend_data[i * numel_divisor + j] / cast_divisor_data[j];
                        }
                    }

                    break;
                }

                case INT32: {
                    int32_t* cast_dividend_data = (int32_t*)dividend->data;
                    int32_t* cast_divisor_data = (int32_t*)divisor->data;
                    int32_t* cast_result_data = (int32_t*)result->data;

                    for(size_t i = 0; i < (numel_dividend / numel_divisor); i++) {
                        for(size_t j = 0; j < numel_divisor; j++) {
                            cast_result_data[i * numel_divisor + j] = cast_dividend_data[i * numel_divisor + j] / cast_divisor_data[j];
                        }
                    }

                    break;
                }

                default: {
                    NNL2_TYPE_ERROR(dividend_dtype);
                    return NULL;
                }
            }
            
            #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                NNL2_FUNC_EXIT();
            #endif
            
            return result;
        } else {
            // Handling a case with different data types (conversion required)
            size_t dividend_step = get_dtype_size(dividend_dtype);
            size_t divisor_step = get_dtype_size(divisor_dtype);
            
            switch(winner_in_the_type_hierarchy) {
                case FLOAT64: {
                    double* cast_data_result = (double*)result->data;
                    
                    char* cast_dividend_data = (char*)dividend->data;
                    char* cast_divisor_data =  (char*)divisor->data;
                    
                    for(size_t i = 0; i < (numel_dividend / numel_divisor); i++) {
                        for(size_t j = 0; j < numel_divisor; j++) {
                            void* elem_dividend = cast_dividend_data + (i * numel_divisor + j) * dividend_step;
                            void* elem_divisor = cast_divisor_data + j * divisor_step; 
                            
                            cast_data_result[i * numel_divisor + j] = nnl2_convert_to_float64(elem_dividend, dividend_dtype) / nnl2_convert_to_float64(elem_divisor, divisor_dtype);
                        }
                    }
                    
                    break;
                }
                
                case FLOAT32: {
                    float* cast_data_result = (float*)result->data;
                    
                    char* cast_dividend_data = (char*)dividend->data;
                    char* cast_divisor_data =  (char*)divisor->data;
                    
                    for(size_t i = 0; i < (numel_dividend / numel_divisor); i++) {
                        for(size_t j = 0; j < numel_divisor; j++) {
                            void* elem_dividend = cast_dividend_data + (i * numel_divisor + j) * dividend_step;
                            void* elem_divisor = cast_divisor_data + j * divisor_step;
                            
                            cast_data_result[i * numel_divisor + j] = nnl2_convert_to_float32(elem_dividend, dividend_dtype) / nnl2_convert_to_float32(elem_divisor, divisor_dtype);
                        }
                    }
                    
                    break;
                }
                
                case INT32: {
                    int32_t* cast_data_result = (int32_t*)result->data;
                    
                    char* cast_dividend_data = (char*)dividend->data;
                    char* cast_divisor_data =  (char*)divisor->data;
                    
                    for(size_t i = 0; i < (numel_dividend / numel_divisor); i++) {                    
                        for(size_t j = 0; j < numel_divisor; j++) {
                            void* elem_dividend = cast_dividend_data + (i * numel_divisor + j) * dividend_step;
                            void* elem_divisor = cast_divisor_data + j * divisor_step;
                        
                            cast_data_result[i * numel_divisor + j] = nnl2_convert_to_int32(elem_dividend, dividend_dtype) / nnl2_convert_to_int32(elem_divisor, divisor_dtype);
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
        NNL2_ERROR("Cannot broadcast divisor tensor");
        return NULL;
    }
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for division with broadcasting
 */
Implementation div_broadcasting_backends[] = {
    REGISTER_BACKEND(naive_div_broadcasting, nnl2_naive, NAIVE_BACKEND_NAME),
};  

/**
 * @brief Function pointer for division with broadcasting operation
 * @ingroup backend_system
 */
divbroadcastingfn div_broadcasting;

/**
 * @brief Sets the backend for division with broadcasting operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for division with broadcasting
 */
void set_div_broadcasting_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(div_broadcasting_backends, div_broadcasting, backend_name);
}

#endif /** NNL2_BROADCASTING_DIV_H **/
