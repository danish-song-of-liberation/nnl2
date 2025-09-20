#ifndef NNL2_BROADCASTING_DIV_INPLACE_H
#define NNL2_BROADCASTING_DIV_INPLACE_H

/** @brief
 * Performs element-wise division with broadcasting (in place)
 *
 ** @param dividend
 * Pointer to dividend tensor (will be modified in place)
 *
 ** @param divisor
 * Pointer to divisor tensor
 */
void naive_div_broadcasting_inplace(Tensor* dividend, const Tensor* divisor) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks for max safety level 
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(dividend, "Dividend tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(divisor, "Divisor tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(dividend->shape, "Dividend shape is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(divisor->shape, "Divisor shape is NULL");
    #endif
    
    size_t numel_dividend = product(dividend->shape, dividend->rank);
    size_t numel_divisor = product(divisor->shape, divisor->rank);
    
    // Getting the tensor data types
    TensorType dividend_dtype = dividend->dtype;
    TensorType divisor_dtype = divisor->dtype;

    if((numel_dividend % numel_divisor) == 0) {
        // Handling the case where the data types match (more efficiently)
        if(dividend_dtype == divisor_dtype) {
            switch(dividend_dtype) {
                case FLOAT64: {
                    double* cast_dividend_data = (double*)dividend->data;
                    double* cast_divisor_data = (double*)divisor->data;

                    for(size_t i = 0; i < (numel_dividend / numel_divisor); i++) {
                        for(size_t j = 0; j < numel_divisor; j++) {
                            cast_dividend_data[i * numel_divisor + j] /= cast_divisor_data[j];
                        }
                    }

                    break;
                }

                case FLOAT32: {
                    float* cast_dividend_data = (float*)dividend->data;
                    float* cast_divisor_data = (float*)divisor->data;

                    for(size_t i = 0; i < (numel_dividend / numel_divisor); i++) {
                        for(size_t j = 0; j < numel_divisor; j++) {
                            cast_dividend_data[i * numel_divisor + j] /= cast_divisor_data[j];
                        }
                    }

                    break;
                }

                case INT32: {
                    int32_t* cast_dividend_data = (int32_t*)dividend->data;
                    int32_t* cast_divisor_data = (int32_t*)divisor->data;

                    for(size_t i = 0; i < (numel_dividend / numel_divisor); i++) {
                        for(size_t j = 0; j < numel_divisor; j++) {
                            cast_dividend_data[i * numel_divisor + j] /= cast_divisor_data[j];
                        }
                    }

                    break;
                }

                default: {
                    NNL2_TYPE_ERROR(dividend_dtype);
                    return;
                }
            }
        } else {
            // Handling a case with different data types (conversion required)
            size_t divisor_step = get_dtype_size(divisor_dtype);
            char* divisor_data = (char*)divisor->data;
            
            switch(dividend_dtype) {
                case FLOAT64: {
                    double* data_dividend = (double*)dividend->data;
                
                    for(size_t i = 0; i < (numel_dividend / numel_divisor); i++) {
                        for(size_t j = 0; j < numel_divisor; j++) {
                            void* divisor_elem = divisor_data + j * divisor_step;
                            data_dividend[i * numel_divisor + j] /= nnl2_convert_to_float64(divisor_elem, divisor_dtype);
                        }
                    }
                
                    break; 
                }
                
                case FLOAT32: {
                    float* data_dividend = (float*)dividend->data;
                
                    for(size_t i = 0; i < (numel_dividend / numel_divisor); i++) {
                        for(size_t j = 0; j < numel_divisor; j++) {
                            void* divisor_elem = divisor_data + j * divisor_step;
                            data_dividend[i * numel_divisor + j] /= nnl2_convert_to_float32(divisor_elem, divisor_dtype);
                        }
                    }
                
                    break; 
                }
                
                case INT32: {
                    int32_t* data_dividend = (int32_t*)dividend->data;
                
                    for(size_t i = 0; i < (numel_dividend / numel_divisor); i++) {
                        for(size_t j = 0; j < numel_divisor; j++) {
                            void* divisor_elem = divisor_data + j * divisor_step;
                            data_dividend[i * numel_divisor + j] /= nnl2_convert_to_int32(divisor_elem, divisor_dtype);
                        }
                    }
                
                    break; 
                }
                
                default: {
                    NNL2_TYPE_ERROR(dividend_dtype);
                    return;
                }
            }
        }
    } else {
        NNL2_ERROR("Cannot broadcast divisor tensor");
        return;
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for division with broadcasting (in place)
 */
Implementation div_broadcasting_inplace_backends[] = {
    REGISTER_BACKEND(naive_div_broadcasting_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};  

/**
 * @brief Function pointer for division with broadcasting operation (in place)
 * @ingroup backend_system
 */
divbroadcastinginplacefn div_broadcasting_inplace;

/**
 * @brief Sets the backend for division with broadcasting operation (in place)
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for division with broadcasting
 */
void set_div_broadcasting_inplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(div_broadcasting_inplace_backends, div_broadcasting_inplace, backend_name);
}

#endif /** NNL2_BROADCASTING_DIV_INPLACE_H **/
