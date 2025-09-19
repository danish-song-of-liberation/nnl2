#ifndef NNL2_CORRESPONDENCE_SUB_INPLACE_H
#define NNL2_CORRESPONDENCE_SUB_INPLACE_H

/** @brief 
 * Subtracts a scalar value from each element of a tensor (in-place).
 * 
 ** @param tensor 
 * Pointer to the tensor from which the value will be subtracted
 * 
 ** @param dec 
 * Pointer to the scalar value to subtract
 */
void naive_sub_decf_inplace(Tensor* tensor, void* dec) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    size_t total_elems = product(tensor->shape, tensor->rank);
    if(total_elems == 0) return;
    
    switch(tensor->dtype) {
        case FLOAT64: {
            double* cast_data = (double*)tensor->data;  // Casting
            double decrement = *((double*)dec);
            for(size_t i = 0; i < total_elems; i++) cast_data[i] -= decrement;
            break;
        }
        
        case FLOAT32: {
            float* cast_data = (float*)tensor->data;
            float decrement = *((float*)dec);
            for(size_t i = 0; i < total_elems; i++) cast_data[i] -= decrement;
            break;
        }
        
        case INT32: {
            int32_t* cast_data = (int32_t*)tensor->data;
            int32_t decrement = *((int32_t*)dec);
            for(size_t i = 0; i < total_elems; i++) cast_data[i] -= decrement;
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
 * @brief Backend implementations for in-place scalar subtraction operation
 * @details
 * Array follows the common backend registration pattern for in-place scalar subtraction
 * operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for scalar subtraction
 * 
 * @see nnl2_naive
 * @see naive_sub_decf_inplace
 */
Implementation sub_decf_inplace_backends[] = {
    REGISTER_BACKEND(naive_sub_decf_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};    

/**
 * @brief Function pointer for in-place scalar subtraction operation
 * @ingroup backend_system 
 */
subdecfinplacefn sub_decf_inplace;

/** 
 * @brief Sets the backend for in-place scalar subtraction operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for scalar subtraction
 * @see SET_BACKEND_BY_NAME
 */
void set_sub_decf_inplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(sub_decf_inplace_backends, sub_decf_inplace, backend_name);
}

#endif /** NNL2_CORRESPONDENCE_SUB_INPLACE_H **/
