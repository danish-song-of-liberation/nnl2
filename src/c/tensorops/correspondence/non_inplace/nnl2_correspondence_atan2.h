#ifndef NNL2_CORRESPONDENCE_ATAN2_H
#define NNL2_CORRESPONDENCE_ATAN2_H

/** @brief
 * Performs element-wise atan2(y/x) where x is a scalar value
 *
 ** @param y
 * Pointer to the input y-coordinate tensor
 *
 ** @param x
 * Pointer to the scalar x-coordinate value
 *
 ** @return
 * Pointer to a new tensor containing the result of atan2 operation 
 */
Tensor* naive_correspondence_atan2(const Tensor* y, void* x) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    Tensor* result = nnl2_empty(y->shape, y->rank, y->dtype);
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(result == NULL) {
            NNL2_ERROR("Failed to allocate new tensor");
            return NULL;
        }
    #endif
    
    size_t total_elems = product(y->shape, y->rank);
    if(total_elems == 0) return result;
    
    switch(y->dtype) {
        case FLOAT64: {
            double* y_data = (double*)y->data;
            double* result_data = (double*)result->data;
            double x_val = *((double*)x);
            for(size_t i = 0; i < total_elems; i++) result_data[i] = atan2(y_data[i], x_val);
            break;
        }
        
        case FLOAT32: {
            float* y_data = (float*)y->data;
            float* result_data = (float*)result->data;
            float x_val = *((float*)x);
            for(size_t i = 0; i < total_elems; i++) result_data[i] = atan2f(y_data[i], x_val);
            break;
        }
        
        case INT32: {
            int32_t* y_data = (int32_t*)y->data;
            double* result_data = (double*)result->data;
            int32_t x_val = *((int32_t*)x);
            
            bool all_zeros = true;
            for(size_t i = 0; i < total_elems; i++) {
                if(y_data[i] != 0 || x_val != 0) {
                    all_zeros = false;
                    break;
                }
            }
            
            if(all_zeros) {
                for(size_t i = 0; i < total_elems; i++) result_data[i] = 0.0;
            } else {
                for(size_t i = 0; i < total_elems; i++) result_data[i] = atan2((double)y_data[i], (double)x_val);
            }
			
            break;
        }
        
        default: {
            NNL2_TYPE_ERROR(y->dtype);
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
 * @brief Backend implementations for atan2 operation with scalar x
 */
Implementation atan2_correspondence_backends[] = {
    REGISTER_BACKEND(naive_correspondence_atan2, nnl2_naive, NAIVE_BACKEND_NAME),
};    

/**
 * @brief Function pointer for atan2 operation with scalar x
 * @ingroup backend_system
 */
atan2correspondencefn nnl2_atan2_correspondence;

/** 
 * @brief Sets the backend for atan2 operation with scalar x
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for atan2
 */
void set_atan2_correspondence_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(atan2_correspondence_backends, nnl2_atan2_correspondence, backend_name);
}

#endif /** NNL2_CORRESPONDENCE_ATAN2_H **/
