#ifndef NNL2_CORRESPONDENCE_DIV_INPLACE_H
#define NNL2_CORRESPONDENCE_DIV_INPLACE_H

/** @brief 
 * Divides each element of a tensor by a scalar value (in-place).
 * 
 ** @param tensor 
 * Pointer to the tensor whose elements will be divided
 * 
 ** @param divisor 
 * Pointer to the scalar value to divide by
 */
void naive_div_divf_inplace(Tensor* tensor, void* divisor) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    size_t total_elems = product(tensor->shape, tensor->rank);
    if(total_elems == 0) return;
    
    switch(tensor->dtype) {
        case FLOAT64: {
            double* cast_data = (double*)tensor->data;
            double div = *((double*)divisor);
            for(size_t i = 0; i < total_elems; i++) cast_data[i] /= div;
            break;
        }
        
        case FLOAT32: {
            float* cast_data = (float*)tensor->data;
            float div = *((float*)divisor);
            for(size_t i = 0; i < total_elems; i++) cast_data[i] /= div;
            break;
        }
        
        case INT32: {
            int32_t* cast_data = (int32_t*)tensor->data;
            int32_t div = *((int32_t*)divisor);
            for(size_t i = 0; i < total_elems; i++) cast_data[i] /= div;
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
 * @brief Backend implementations for in-place scalar division operation
 * @details
 * Array follows the common backend registration pattern for in-place scalar division
 * operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for scalar division
 * 
 * @see nnl2_naive
 * @see naive_div_divf_inplace
 */
Implementation div_divf_inplace_backends[] = {
    REGISTER_BACKEND(naive_div_divf_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};    

/**
 * @brief Function pointer for in-place scalar division operation
 * @ingroup backend_system 
 */
divdivfinplacefn div_divf_inplace;

/** 
 * @brief Sets the backend for in-place scalar division operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for scalar division
 * @see SET_BACKEND_BY_NAME
 */
void set_div_divf_inplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(div_divf_inplace_backends, div_divf_inplace, backend_name);
}

#endif /** NNL2_CORRESPONDENCE_DIV_INPLACE_H **/
