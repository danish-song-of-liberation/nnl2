#ifndef NNL2_CORRESPONDENCE_MUL_INPLACE_H
#define NNL2_CORRESPONDENCE_MUL_INPLACE_H

/** @brief 
 * Multiplies each element of a tensor by a scalar value (in-place).
 * 
 ** @param tensor 
 * Pointer to the tensor whose elements will be multiplied
 * 
 ** @param multiplier 
 * Pointer to the scalar value to multiply by
 */
void naive_mul_mulf_inplace(Tensor* tensor, void* multiplier) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    size_t total_elems = product(tensor->shape, tensor->rank);
    if(total_elems == 0) return;
    
    switch(tensor->dtype) {
        case FLOAT64: {
            double* cast_data = (double*)tensor->data;
            double multiply = *((double*)multiplier);
            for(size_t i = 0; i < total_elems; i++) cast_data[i] *= multiply;
            break;
        }
        
        case FLOAT32: {
            float* cast_data = (float*)tensor->data;
            float multiply = *((float*)multiplier);
            for(size_t i = 0; i < total_elems; i++) cast_data[i] *= multiply;
            break;
        }
        
        case INT32: {
            int32_t* cast_data = (int32_t*)tensor->data;
            int32_t multiply = *((int32_t*)multiplier);
            for(size_t i = 0; i < total_elems; i++) cast_data[i] *= multiply;
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
 * @brief Backend implementations for in-place scalar multiplication operation
 * @details
 * Array follows the common backend registration pattern for in-place scalar multiplication
 * operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for scalar multiplication
 * 
 * @see nnl2_naive
 * @see naive_mul_mulf_inplace
 */
Implementation mul_mulf_inplace_backends[] = {
    REGISTER_BACKEND(naive_mul_mulf_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};    

/**
 * @brief Function pointer for in-place scalar multiplication operation
 * @ingroup backend_system 
 */
mulmulfinplacefn mul_mulf_inplace;

/** 
 * @brief Sets the backend for in-place scalar multiplication operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for scalar multiplication
 * @see SET_BACKEND_BY_NAME
 */
void set_mul_mulf_inplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(mul_mulf_inplace_backends, mul_mulf_inplace, backend_name);
}

#endif /** NNL2_CORRESPONDENCE_MUL_INPLACE_H **/
