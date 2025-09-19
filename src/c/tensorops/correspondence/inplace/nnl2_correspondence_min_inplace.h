#ifndef NNL2_CORRESPONDENCE_MIN_INPLACE_H
#define NNL2_CORRESPONDENCE_MIN_INPLACE_H

/** @brief 
 * Applies element-wise minimum between tensor elements and a scalar value (in-place).
 * Each element is replaced by the minimum of its current value and the specified scalar.
 * 
 ** @param tensor 
 * Pointer to the tensor whose elements will be processed
 * 
 ** @param threshold 
 * Pointer to the scalar threshold value for minimum operation
 */
void naive_min_minf_inplace(Tensor* tensor, void* threshold) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    size_t total_elems = product(tensor->shape, tensor->rank);
    if(total_elems == 0) return;
    
    switch(tensor->dtype) {
        case FLOAT64: {
            double* cast_data = (double*)tensor->data;
            double min_val = *((double*)threshold);
            for(size_t i = 0; i < total_elems; i++) cast_data[i] = MIN(cast_data[i], min_val);
            break;
        }
        
        case FLOAT32: {
            float* cast_data = (float*)tensor->data;
            float min_val = *((float*)threshold);
            for(size_t i = 0; i < total_elems; i++) cast_data[i] = MIN(cast_data[i], min_val);
            break;
        }
        
        case INT32: {
            int32_t* cast_data = (int32_t*)tensor->data;
            int32_t min_val = *((int32_t*)threshold);
            for(size_t i = 0; i < total_elems; i++) cast_data[i] = MIN(cast_data[i], min_val);
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
 * @brief Backend implementations for in-place element-wise minimum operation
 * @details
 * Array follows the common backend registration pattern for in-place minimum
 * operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for element-wise minimum
 * 
 * @see nnl2_naive
 * @see naive_min_minf_inplace
 */
Implementation min_minf_inplace_backends[] = {
    REGISTER_BACKEND(naive_min_minf_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};    

/**
 * @brief Function pointer for in-place element-wise minimum operation
 * @ingroup backend_system 
 */
minminfinplacefn min_minf_inplace;

/** 
 * @brief Sets the backend for in-place element-wise minimum operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for minimum operation
 * @see SET_BACKEND_BY_NAME
 */
void set_min_minf_inplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(min_minf_inplace_backends, min_minf_inplace, backend_name);
}

#endif /** NNL2_CORRESPONDENCE_MIN_INPLACE_H **/
