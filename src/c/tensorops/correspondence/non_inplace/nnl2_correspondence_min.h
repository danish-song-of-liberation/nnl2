#ifndef NNL2_CORRESPONDENCE_MIN_H
#define NNL2_CORRESPONDENCE_MIN_H

/** @brief
 * Performs element-wise minimum operation between tensor elements and a scalar value
 *
 ** @param tensor
 * Pointer to the input tensor to be processed
 *
 ** @param threshold
 * Pointer to the scalar threshold value for minimum operation
 *
 ** @return
 * Pointer to a new tensor containing the result of the minimum operation 
 * (or NULL in case of failure)
 */
Tensor* naive_min_minf(const Tensor* tensor, void* threshold) {
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
            double min_val = *((double*)threshold);
            for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = MIN(cast_data_original[i], min_val);
            break;
        }
        
        case FLOAT32: {
            float* cast_data_original = (float*)tensor->data;
            float* cast_data_result = (float*)result->data;
            float min_val = *((float*)threshold);
            for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = MIN(cast_data_original[i], min_val);
            break;
        }
        
        case INT32: {
            int32_t* cast_data_original = (int32_t*)tensor->data;
            int32_t* cast_data_result = (int32_t*)result->data;
            int32_t min_val = *((int32_t*)threshold);
            for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = MIN(cast_data_original[i], min_val);
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
 * @brief Backend implementations for element-wise minimum operation
 * @details
 * Array follows the common backend registration pattern for element-wise minimum
 * operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for element-wise minimum
 * 
 * @see nnl2_naive
 * @see naive_min_minf
 */
Implementation min_minf_backends[] = {
    REGISTER_BACKEND(naive_min_minf, nnl2_naive, NAIVE_BACKEND_NAME),
};    

/**
 * @brief Function pointer for element-wise minimum operation
 * @ingroup backend_system
 */
minminffn min_minf;

/** 
 * @brief Sets the backend for element-wise minimum operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for minimum operation
 * @see SET_BACKEND_BY_NAME
 * @see min_minf_backends
 */
void set_min_minf_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(min_minf_backends, min_minf, backend_name);
}

#endif /** NNL2_CORRESPONDENCE_MIN_H **/
