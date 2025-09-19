#ifndef NNL2_CORRESPONDENCE_SUB_H
#define NNL2_CORRESPONDENCE_SUB_H

/** @brief
 * Performs element-wise subtraction of a scalar decrement from a tensor
 *
 ** @param tensor
 * Pointer to the input tensor from which the decrement will be subtracted
 *
 ** @param dec
 * Pointer to the scalar decrement value
 *
 ** @return
 * Pointer to a new tensor containing the result of the subtraction operation 
 * (or NULL in case of failure)
 */
Tensor* naive_sub_decf(const Tensor* tensor, void* dec) {
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
            double* cast_data_result = (double*)result->data;	// Casting
            double decrement = *((double*)dec); 
            for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = cast_data_original[i] - decrement; // Assigment
            break;
        }
        
        case FLOAT32: {
            float* cast_data_original = (float*)tensor->data;
            float* cast_data_result = (float*)result->data;
            float decrement = *((float*)dec);
            for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = cast_data_original[i] - decrement;
            break;
        }
        
        case INT32: {
            int32_t* cast_data_original = (int32_t*)tensor->data;
            int32_t* cast_data_result = (int32_t*)result->data;
            int32_t decrement = *((int32_t*)dec);
            for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = cast_data_original[i] - decrement;
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
 * @brief Backend implementations for scalar subtraction with decrement operation
 * @details
 * Array follows the common backend registration pattern for scalar subtraction with
 * decrement operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for scalar subtraction with decrement
 * 
 * @see nnl2_naive
 * @see naive_sub_decf
 */
Implementation sub_decf_backends[] = {
    REGISTER_BACKEND(naive_sub_decf, nnl2_naive, NAIVE_BACKEND_NAME),
};    

/**
 * @brief Function pointer for scalar subtraction with decrement operation
 * @ingroup backend_system
 */
subdecffn sub_decf;

/** 
 * @brief Sets the backend for scalar subtraction with decrement operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for scalar subtraction with decrement
 * @see SET_BACKEND_BY_NAME
 * @see sub_decf_backends
 */
void set_sub_decf_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(sub_decf_backends, sub_decf, backend_name);
}

#endif /** NNL2_CORRESPONDENCE_SUB_H **/
