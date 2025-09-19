#ifndef NNL2_CORRESPONDENCE_MUL_H
#define NNL2_CORRESPONDENCE_MUL_H

/** @brief
 * Performs element-wise multiplication of a tensor by a scalar multiplier
 *
 ** @param tensor
 * Pointer to the input tensor to be multiplied
 *
 ** @param multiplier
 * Pointer to the scalar multiplier value
 *
 ** @return
 * Pointer to a new tensor containing the result of the multiplication operation 
 * (or NULL in case of failure)
 */
Tensor* naive_mul_mulf(const Tensor* tensor, void* multiplier) {
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
            double multiply = *((double*)multiplier);
            for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = cast_data_original[i] * multiply;
            break;
        }
        
        case FLOAT32: {
            float* cast_data_original = (float*)tensor->data;
            float* cast_data_result = (float*)result->data;
            float multiply = *((float*)multiplier);
            for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = cast_data_original[i] * multiply;
            break;
        }
        
        case INT32: {
            int32_t* cast_data_original = (int32_t*)tensor->data;
            int32_t* cast_data_result = (int32_t*)result->data;
            int32_t multiply = *((int32_t*)multiplier);
            for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = cast_data_original[i] * multiply;
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
 * @brief Backend implementations for scalar multiplication operation
 * @details
 * Array follows the common backend registration pattern for scalar multiplication
 * operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for scalar multiplication
 * 
 * @see nnl2_naive
 * @see naive_mul_mulf
 */
Implementation mul_mulf_backends[] = {
    REGISTER_BACKEND(naive_mul_mulf, nnl2_naive, NAIVE_BACKEND_NAME),
};    

/**
 * @brief Function pointer for scalar multiplication operation
 * @ingroup backend_system
 */
mulmulffn mul_mulf;

/** 
 * @brief Sets the backend for scalar multiplication operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for scalar multiplication
 * @see SET_BACKEND_BY_NAME
 * @see mul_mulf_backends
 */
void set_mul_mulf_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(mul_mulf_backends, mul_mulf, backend_name);
}

#endif /** NNL2_CORRESPONDENCE_MUL_H **/