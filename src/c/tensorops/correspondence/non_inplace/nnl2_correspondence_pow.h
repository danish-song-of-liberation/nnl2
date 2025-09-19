#ifndef NNL2_CORRESPONDENCE_POW_H
#define NNL2_CORRESPONDENCE_POW_H

/** @brief
 * Performs element-wise exponentiation of a tensor by a scalar exponent
 *
 ** @param tensor
 * Pointer to the input tensor to be exponentiated
 *
 ** @param exponent
 * Pointer to the scalar exponent value
 *
 ** @return
 * Pointer to a new tensor containing the result of the exponentiation operation 
 * (or NULL in case of failure)
 */
Tensor* naive_pow_powf(const Tensor* tensor, void* exponent) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    Tensor* result = nnl2_empty(tensor->shape, tensor->rank, tensor->dtype);
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(result == NULL) {
            NNL2_ERROR("Failed to allocate new tensor");
            return NULL;
        }
    #endif
    
    size_t total_elems = product(tensor->shape, tensor->rank);
    if(total_elems == 0) return result;
    
    switch(tensor->dtype) {
        case FLOAT64: {
            double* cast_data_original = (double*)tensor->data;
            double* cast_data_result = (double*)result->data;
            double pow_val = *((double*)exponent);
            for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = pow(cast_data_original[i], pow_val);
            break;
        }
        
        case FLOAT32: {
            float* cast_data_original = (float*)tensor->data;
            float* cast_data_result = (float*)result->data;
            float pow_val = *((float*)exponent);
            for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = powf(cast_data_original[i], pow_val);
            break;
        }
        
        case INT32: {
            int32_t* cast_data_original = (int32_t*)tensor->data;
            int32_t* cast_data_result = (int32_t*)result->data;
            int32_t pow_val = *((int32_t*)exponent);
            for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = (int32_t)pow((double)cast_data_original[i], pow_val);
            break;
        }
        
        default: {
            NNL2_TYPE_ERROR(tensor->dtype);
            nnl2_free_tensor(result);
            return NULL;
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return result;
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for exponentiation operation
 * @details
 * Array follows the common backend registration pattern for exponentiation
 * operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for exponentiation
 * 
 * @see nnl2_naive
 * @see naive_pow_powf
 */
Implementation pow_powf_backends[] = {
    REGISTER_BACKEND(naive_pow_powf, nnl2_naive, NAIVE_BACKEND_NAME),
};    

/**
 * @brief Function pointer for exponentiation operation
 * @ingroup backend_system
 */
powpowffn pow_powf;

/** 
 * @brief Sets the backend for exponentiation operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for exponentiation
 * @see SET_BACKEND_BY_NAME
 * @see pow_powf_backends
 */
void set_pow_powf_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(pow_powf_backends, pow_powf, backend_name);
}

#endif /** NNL2_CORRESPONDENCE_POW_H **/