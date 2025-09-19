#ifndef NNL2_CORRESPONDENCE_MAX_H
#define NNL2_CORRESPONDENCE_MAX_H

/** @brief
 * Performs element-wise maximum operation between tensor elements and a scalar value
 *
 ** @param tensor
 * Pointer to the input tensor to be processed
 *
 ** @param threshold
 * Pointer to the scalar threshold value for maximum operation
 *
 ** @return
 * Pointer to a new tensor containing the result of the maximum operation 
 * (or NULL in case of failure)
 */
Tensor* naive_max_maxf(const Tensor* tensor, void* threshold) {
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
            double max_val = *((double*)threshold);
            for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = MAX(cast_data_original[i], max_val);
            break;
        }
        
        case FLOAT32: {
            float* cast_data_original = (float*)tensor->data;
            float* cast_data_result = (float*)result->data;
            float max_val = *((float*)threshold);
            for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = MAX(cast_data_original[i], max_val);
            break;
        }
        
        case INT32: {
            int32_t* cast_data_original = (int32_t*)tensor->data;
            int32_t* cast_data_result = (int32_t*)result->data;
            int32_t max_val = *((int32_t*)threshold);
            for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = MAX(cast_data_original[i], max_val);
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
 * @brief Backend implementations for element-wise maximum operation
 * @details
 * Array follows the common backend registration pattern for element-wise maximum
 * operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for element-wise maximum
 * 
 * @see nnl2_naive
 * @see naive_max_maxf
 */
Implementation max_maxf_backends[] = {
    REGISTER_BACKEND(naive_max_maxf, nnl2_naive, NAIVE_BACKEND_NAME),
};    

/**
 * @brief Function pointer for element-wise maximum operation
 * @ingroup backend_system
 */
maxmaxffn max_maxf;

/** 
 * @brief Sets the backend for element-wise maximum operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for maximum operation
 * @see SET_BACKEND_BY_NAME
 * @see max_maxf_backends
 */
void set_max_maxf_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(max_maxf_backends, max_maxf, backend_name);
}

#endif /** NNL2_CORRESPONDENCE_MAX_H **/
