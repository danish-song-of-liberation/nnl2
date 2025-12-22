#ifndef NNL2_CORRESPONDENCE_ATAN2_INPLACE_H
#define NNL2_CORRESPONDENCE_ATAN2_INPLACE_H	

/** @brief 
 * Performs atan2(y/x) with scalar x (in-place)
 * 
 ** @param y 
 * Pointer to the y-coordinate tensor
 * 
 ** @param x 
 * Pointer to the scalar x-coordinate value
 */
void naive_correspondence_atan2_inplace(nnl2_tensor* y, void* x) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    size_t total_elems = nnl2_product(y->shape, y->rank);
    if(total_elems == 0) return;
    
    switch(y->dtype) {
        case FLOAT64: {
            double* data_y = (double*)y->data;
            double x_val = *((double*)x);
            for(size_t i = 0; i < total_elems; i++) data_y[i] = atan2(data_y[i], x_val);
            break;
        }
        
        case FLOAT32: {
            float* data_y = (float*)y->data;
            float x_val = *((float*)x);
            for(size_t i = 0; i < total_elems; i++) data_y[i] = atan2f(data_y[i], x_val);
            break;
        }
        
        case INT32: {
            int32_t* data_y = (int32_t*)y->data;
            int32_t x_val = *((int32_t*)x);
            
            for (size_t i = 0; i < total_elems; i++) {
                if (data_y[i] != 0 || x_val != 0) {
                    NNL2_FATAL("Can't apply atan2 to non-zero INT32 tensor with non-zero scalar");
                }
            }
            
            for(size_t i = 0; i < total_elems; i++) data_y[i] = 0;
            break;
        }
        
        default: {
            NNL2_TYPE_ERROR(y->dtype);
            return;
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}    

/**
 * @ingroup backend_system
 * @brief Backend implementations for in-place atan2 operation with scalar x
 */
nnl2_runtime_implementation atan2_correspondence_inplace_backends[] = {
    REGISTER_BACKEND(naive_correspondence_atan2_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};    

/**
 * @brief Function pointer for in-place atan2 operation with scalar x
 * @ingroup backend_system 
 */
atan2correspondenceinplacefn nnl2_atan2_correspondence_inplace;

/** 
 * @brief Sets the backend for in-place atan2 operation with scalar x
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for atan2
 */
void set_atan2inplace_correspondence_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(atan2_correspondence_inplace_backends, nnl2_atan2_correspondence_inplace, backend_name);
}

#endif /** NNL2_CORRESPONDENCE_ATAN2_INPLACE_H **/
