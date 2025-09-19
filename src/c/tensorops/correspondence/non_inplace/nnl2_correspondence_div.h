#ifndef NNL2_CORRESPONDENCE_DIV_H
#define NNL2_CORRESPONDENCE_DIV_H

/** @brief
 * Performs element-wise division of a tensor by a scalar divisor
 *
 ** @param tensor
 * Pointer to the input tensor to be divided
 *
 ** @param divisor
 * Pointer to the scalar divisor value
 *
 ** @return
 * Pointer to a new tensor containing the result of the division operation 
 * (or NULL in case of failure)
 */
Tensor* naive_div_divf(const Tensor* tensor, void* divisor) {
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
            double div = *((double*)divisor);
            for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = cast_data_original[i] / div;
            break;
        }
        
        case FLOAT32: {
            float* cast_data_original = (float*)tensor->data;
            float* cast_data_result = (float*)result->data;
            float div = *((float*)divisor);
            for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = cast_data_original[i] / div;
            break;
        }
        
        case INT32: {
            int32_t* cast_data_original = (int32_t*)tensor->data;
            int32_t* cast_data_result = (int32_t*)result->data;
            int32_t div = *((int32_t*)divisor);
            for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = cast_data_original[i] / div;
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
 * @brief Backend implementations for scalar division operation
 * @details
 * Array follows the common backend registration pattern for scalar division
 * operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for scalar division
 * 
 * @see nnl2_naive
 * @see naive_div_divf
 */
Implementation div_divf_backends[] = {
    REGISTER_BACKEND(naive_div_divf, nnl2_naive, NAIVE_BACKEND_NAME),
};    

/**
 * @brief Function pointer for scalar division operation
 * @ingroup backend_system
 */
divdivffn div_divf;

/** 
 * @brief Sets the backend for scalar division operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for scalar division
 * @see SET_BACKEND_BY_NAME
 * @see div_divf_backends
 */
void set_div_divf_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(div_divf_backends, div_divf, backend_name);
}

#endif /** NNL2_CORRESPONDENCE_DIV_H **/
