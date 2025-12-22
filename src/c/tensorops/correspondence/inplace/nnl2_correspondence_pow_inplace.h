#ifndef NNL2_CORRESPONDENCE_POW_INPLACE_H
#define NNL2_CORRESPONDENCE_POW_INPLACE_H	

/** @brief 
 * Raises each element of a tensor to a scalar power (in-place)
 * 
 ** @param tensor 
 * Pointer to the tensor whose elements will be exponentiated
 * 
 ** @param exponent 
 * Pointer to the scalar exponent value
 */
void naive_pow_powf_inplace(nnl2_tensor* tensor, void* exponent) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    size_t total_elems = product(tensor->shape, tensor->rank);
    if(total_elems == 0) return;
    
    switch(tensor->dtype) {
        case FLOAT64: {
            double* cast_data = (double*)tensor->data;
            double pow_val = *((double*)exponent);
            for(size_t i = 0; i < total_elems; i++) cast_data[i] = pow(cast_data[i], pow_val);
            break;
        }
        
        case FLOAT32: {
            float* cast_data = (float*)tensor->data;
            float pow_val = *((float*)exponent);
            for(size_t i = 0; i < total_elems; i++) cast_data[i] = powf(cast_data[i], pow_val);
            break;
        }
        
        case INT32: {
            int32_t* cast_data = (int32_t*)tensor->data;
            int32_t pow_val = *((int32_t*)exponent);
            for(size_t i = 0; i < total_elems; i++) cast_data[i] = (int32_t)pow((double)cast_data[i], pow_val);
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
 * @brief Backend implementations for in-place exponentiation operation
 * @details
 * Array follows the common backend registration pattern for in-place exponentiation
 * operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for exponentiation
 * 
 * @see nnl2_naive
 * @see naive_pow_powf_inplace
 */
nnl2_runtime_implementation pow_powf_inplace_backends[] = {
    REGISTER_BACKEND(naive_pow_powf_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};    

/**
 * @brief Function pointer for in-place exponentiation operation
 * @ingroup backend_system 
 */
powpowfinplacefn pow_powf_inplace;

/** 
 * @brief Sets the backend for in-place exponentiation operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for exponentiation
 * @see SET_BACKEND_BY_NAME
 */
void set_pow_powf_inplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(pow_powf_inplace_backends, pow_powf_inplace, backend_name);
}

#endif /** NNL2_CORRESPONDENCE_POW_INPLACE_H **/
