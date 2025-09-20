#ifndef NNL2_CORRESPONDENCE_AXPY_H
#define NNL2_CORRESPONDENCE_AXPY_H

/** @brief
 * Performs element-wise AXPF operation (scalar AXPY)
 * Computes: result = summand + alpha * sumend (where sumend is a scalar)
 *
 ** @param summand
 * Pointer to the input tensor
 *
 ** @param sumend
 * Pointer to the scalar value to be scaled and added
 *
 ** @param alpha
 * Scalar multiplier for the sumend value
 *
 ** @return
 * Pointer to a new tensor containing the result of the AXPF operation 
 * (or NULL in case of fail)
 */
Tensor* naive_axpf(Tensor* summand, void* sumend, float alpha) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    Tensor* result = nnl2_empty(summand->shape, summand->rank, summand->dtype);
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(result == NULL) {
            NNL2_ERROR("Failed to allocate new tensor");
        }
    #endif
    
    size_t total_elems = product(summand->shape, summand->rank);
    if(total_elems == 0) return result;
    
    switch(summand->dtype) {
        case FLOAT64: {
            double* cast_data_original = (double*)summand->data; 
            double* cast_data_result = (double*)result->data;
            double cast_sumend = *((double*)sumend);
            double alpha_double = (double)alpha;
            for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = cast_data_original[i] + (cast_sumend * alpha_double);
            break;
        }
        
        case FLOAT32: {
            float* cast_data_original = (float*)summand->data;
            float* cast_data_result = (float*)result->data;
            float cast_sumend = *((float*)sumend);
            for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = cast_data_original[i] + (cast_sumend * alpha);
            break;
        }
        
        case INT32: {
            int32_t* cast_data_original = (int32_t*)summand->data;
            int32_t* cast_data_result = (int32_t*)result->data;
            int32_t cast_sumend = *((int32_t*)sumend);
            int32_t alpha_int = (int32_t)alpha;
            for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = cast_data_original[i] + (cast_sumend * alpha_int);
            break;
        }
        
        default: {
            NNL2_TYPE_ERROR(summand->dtype);
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
 * @brief Backend implementations for AXPF operation
 */
Implementation axpf_backends[] = {
    REGISTER_BACKEND(naive_axpf, nnl2_naive, NAIVE_BACKEND_NAME),
};  

/**
 * @brief Function pointer for AXPF operation
 * @ingroup backend_system
 */
axpffn axpf;

/**
 * @brief Sets the backend for AXPF operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for AXPF operation
 */
void set_axpf_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(axpf_backends, axpf, backend_name);
}

#endif /** NNL2_CORRESPONDENCE_AXPY_H **/
