#ifndef NNL2_CORRESPONDENCE_AXPY_INPLACE_H
#define NNL2_CORRESPONDENCE_AXPY_INPLACE_H

/** @brief 
 * Performs element-wise AXPF operation (scalar AXPY) in place
 * Computes: summand = summand + alpha * sumend (where sumend is a scalar)
 * 
 ** @param summand 
 * Pointer to the tensor that will be modified in place
 * 
 ** @param sumend 
 * Pointer to the scalar value to be scaled and added
 * 
 ** @param alpha
 * Scalar multiplier for the sumend value
 */
void naive_axpf_inplace(Tensor* summand, void* sumend, float alpha) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    size_t total_elems = product(summand->shape, summand->rank);
    if(total_elems == 0) return; 
    
    switch(summand->dtype) {
        case FLOAT64: {
            double* cast_summand = (double*)summand->data;
            double cast_sumend = *((double*)sumend);
            double alpha_double = (double)alpha;
            for(size_t i = 0; i < total_elems; i++) cast_summand[i] += cast_sumend * alpha_double;
            break;
        }
        
        case FLOAT32: {
            float* cast_summand = (float*)summand->data;
            float cast_sumend = *((float*)sumend);
            for(size_t i = 0; i < total_elems; i++) cast_summand[i] += cast_sumend * alpha;
            break;
        }
        
        case INT32: {
            int32_t* cast_summand = (int32_t*)summand->data;
            int32_t cast_sumend = *((int32_t*)sumend);
            int32_t alpha_int = (int32_t)alpha;
            for(size_t i = 0; i < total_elems; i++) cast_summand[i] += cast_sumend * alpha_int;
            break;
        }
        
        default: {
            NNL2_TYPE_ERROR(summand->dtype);
            return;
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for AXPF in-place operation
 */
Implementation axpf_inplace_backends[] = {
    REGISTER_BACKEND(naive_axpf_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};  

/**
 * @brief Function pointer for AXPF in-place operation
 * @ingroup backend_system
 */
axpfinplacefn axpf_inplace;

/**
 * @brief Sets the backend for AXPF in-place operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for AXPF in-place operation
 */
void set_axpf_inplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(axpf_inplace_backends, axpf_inplace, backend_name);
}

#endif /** NNL2_CORRESPONDENCE_AXPY_INPLACE_H **/
