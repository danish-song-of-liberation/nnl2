#ifndef NNL2_CORRESPONDENCE_MAX_INPLACE_H
#define NNL2_CORRESPONDENCE_MAX_INPLACE_H

/** @brief 
 * Applies element-wise maximum between tensor elements and a scalar value (in-place).
 * Each element is replaced by the maximum of its current value and the specified scalar.
 * 
 ** @param tensor 
 * Pointer to the tensor whose elements will be processed
 * 
 ** @param threshold 
 * Pointer to the scalar threshold value for maximum operation
 */
void naive_max_maxf_inplace(Tensor* tensor, void* threshold) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    size_t total_elems = product(tensor->shape, tensor->rank);
    if(total_elems == 0) return;
    
    switch(tensor->dtype) {
        case FLOAT64: {
            double* cast_data = (double*)tensor->data;
            double max_val = *((double*)threshold);
            for(size_t i = 0; i < total_elems; i++) cast_data[i] = MAX(cast_data[i], max_val);
            break;
        }
        
        case FLOAT32: {
            float* cast_data = (float*)tensor->data;
            float max_val = *((float*)threshold);
            for(size_t i = 0; i < total_elems; i++) cast_data[i] = MAX(cast_data[i], max_val);
            break;
        }
        
        case INT32: {
            int32_t* cast_data = (int32_t*)tensor->data;
            int32_t max_val = *((int32_t*)threshold);
            for(size_t i = 0; i < total_elems; i++) cast_data[i] = MAX(cast_data[i], max_val);
            break;
        }
        
        default: {
            NNL2_TYPE_ERROR(tensor->dtype);
            return;
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}    

/**
 * @ingroup backend_system
 * @brief Backend implementations for in-place element-wise maximum operation
 * @details
 * Array follows the common backend registration pattern for in-place maximum
 * operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for element-wise maximum
 * 
 * @see nnl2_naive
 * @see naive_max_maxf_inplace
 */
Implementation max_maxf_inplace_backends[] = {
    REGISTER_BACKEND(naive_max_maxf_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};    

/**
 * @brief Function pointer for in-place element-wise maximum operation
 * @ingroup backend_system 
 */
maxmaxfinplacefn max_maxf_inplace;

/** 
 * @brief Sets the backend for in-place element-wise maximum operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for maximum operation
 * @see SET_BACKEND_BY_NAME
 */
void set_max_maxf_inplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(max_maxf_inplace_backends, max_maxf_inplace, backend_name);
}


#endif /** NNL2_CORRESPONDENCE_MAX_INPLACE_H **/
