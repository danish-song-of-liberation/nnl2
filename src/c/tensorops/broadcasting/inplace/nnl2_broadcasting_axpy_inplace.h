#ifndef NNL2_BROADCASTING_AXPY_INPLACE_H
#define NNL2_BROADCASTING_AXPY_INPLACE_H


/** @brief
 * Performs element-wise AXPY operation with broadcasting (in place)
 * Computes: summand = summand + alpha * sumend
 *
 ** @param summand
 * Pointer to summand tensor (will be modified in place)
 *
 ** @param sumend
 * Pointer to sumend tensor
 *
 ** @param alpha
 * Scalar multiplier
 */
void naive_axpy_broadcasting_inplace(Tensor* summand, const Tensor* sumend, float alpha) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks for max safety level 
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(summand, "Summand tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(sumend, "Sumend tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(summand->shape, "Summand shape is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(sumend->shape, "Sumend shape is NULL");
    #endif
    
    size_t numel_summand = product(summand->shape, summand->rank);
    size_t numel_sumend = product(sumend->shape, sumend->rank);
    
    // Getting the tensor data types
    TensorType summand_dtype = summand->dtype;
    TensorType sumend_dtype = sumend->dtype;

    if((numel_summand % numel_sumend) == 0) {
        // Handling the case where the data types match (more efficiently)
        if(summand_dtype == sumend_dtype) {
            switch(summand_dtype) {
                case FLOAT64: {
                    double* cast_summand_data = (double*)summand->data;
                    double* cast_sumend_data = (double*)sumend->data;
                    double alpha_double = (double)alpha;

                    for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
                        for(size_t j = 0; j < numel_sumend; j++) {
                            cast_summand_data[i * numel_sumend + j] += cast_sumend_data[j] * alpha_double;
                        }
                    }

                    break;
                }

                case FLOAT32: {
                    float* cast_summand_data = (float*)summand->data;
                    float* cast_sumend_data = (float*)sumend->data;

                    for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
                        for(size_t j = 0; j < numel_sumend; j++) {
                            cast_summand_data[i * numel_sumend + j] += cast_sumend_data[j] * alpha;
                        }
                    }

                    break;
                }

                case INT32: {
                    int32_t* cast_summand_data = (int32_t*)summand->data;
                    int32_t* cast_sumend_data = (int32_t*)sumend->data;
                    int32_t alpha_int = (int32_t)alpha;

                    for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
                        for(size_t j = 0; j < numel_sumend; j++) {
                            cast_summand_data[i * numel_sumend + j] += cast_sumend_data[j] * alpha_int;
                        }
                    }

                    break;
                }

                default: {
                    NNL2_TYPE_ERROR(summand_dtype);
                    return;
                }
            }
        } else {
            // Handling a case with different data types (conversion required)
            size_t sumend_step = get_dtype_size(sumend_dtype);
            char* sumend_data = (char*)sumend->data;
            
            switch(summand_dtype) {
                case FLOAT64: {
                    double* data_summand = (double*)summand->data;
                    double alpha_double = (double)alpha;
                
                    for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
                        for(size_t j = 0; j < numel_sumend; j++) {
                            void* sumend_elem = sumend_data + j * sumend_step;
                            data_summand[i * numel_sumend + j] += nnl2_convert_to_float64(sumend_elem, sumend_dtype) * alpha_double;
                        }
                    }
                
                    break; 
                }
                
                case FLOAT32: {
                    float* data_summand = (float*)summand->data;
                
                    for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
                        for(size_t j = 0; j < numel_sumend; j++) {
                            void* sumend_elem = sumend_data + j * sumend_step;
                            data_summand[i * numel_sumend + j] += nnl2_convert_to_float32(sumend_elem, sumend_dtype) * alpha;
                        }
                    }
                
                    break; 
                }
                
                case INT32: {
                    int32_t* data_summand = (int32_t*)summand->data;
                    int32_t alpha_int = (int32_t)alpha;
                
                    for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
                        for(size_t j = 0; j < numel_sumend; j++) {
                            void* sumend_elem = sumend_data + j * sumend_step;
                            data_summand[i * numel_sumend + j] += nnl2_convert_to_int32(sumend_elem, sumend_dtype) * alpha_int;
                        }
                    }
                
                    break; 
                }
                
                default: {
                    NNL2_TYPE_ERROR(summand_dtype);
                    return;
                }
            }
        }
    } else {
        NNL2_ERROR("Cannot broadcast sumend tensor");
        return;
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for AXPY operation with broadcasting (in place)
 */
Implementation axpy_broadcasting_inplace_backends[] = {
    REGISTER_BACKEND(naive_axpy_broadcasting_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};  

/**
 * @brief Function pointer for AXPY operation with broadcasting (in place)
 * @ingroup backend_system
 */
axpybroadcastinginplacefn axpy_broadcasting_inplace;

/**
 * @brief Sets the backend for AXPY operation with broadcasting (in place)
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for AXPY operation with broadcasting
 */
void set_axpy_broadcasting_inplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(axpy_broadcasting_inplace_backends, axpy_broadcasting_inplace, backend_name);
}

#endif /** NNL2_BROADCASTING_AXPY_INPLACE_H **/
