#ifndef NNL2_ATAN2_INPLACE_H
#define NNL2_ATAN2_INPLACE_H

/** @brief 
 * Performs element-wise atan2(y/x) calculation in place
 * 
 ** @param y 
 * Pointer to the tensor that will be modified (receives atan2 result)
 *
 ** @param x 
 * Pointer to the tensor whose values are used as denominator
 */
void nnl2_naive_atan2inplace(nnl2_tensor* y, const nnl2_tensor* x) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    size_t len = nnl2_product(y->shape, y->rank);
    
    if(len == 0) return;
    
    nnl2_tensor_type dtype_y = y->dtype;
    nnl2_tensor_type dtype_x = x->dtype;
    
    if(dtype_y == dtype_x) {
        switch(dtype_y) {
            case FLOAT64: {
                volatile double* data_y = (double*)y->data;
                volatile double* data_x = (double*)x->data;
                for(size_t i = 0; i < len; i++) data_y[i] = atan2(data_y[i], data_x[i]);
                break;
            }
            
            case FLOAT32: {
                volatile float* data_y = (float*)y->data;
                volatile float* data_x = (float*)x->data;
                for(size_t i = 0; i < len; i++) data_y[i] = atan2f(data_y[i], data_x[i]);
                break;
            }
            
            case INT32: {
                volatile int32_t* data_y = (int32_t*)y->data;
                volatile int32_t* data_x = (int32_t*)x->data;
                
                for (size_t i = 0; i < len; i++) {
                    if (data_y[i] != 0 || data_x[i] != 0) {
                        NNL2_FATAL("Can't apply atan2inplace to non-zero INT32 tensors");
                    }
                }
                
                for (size_t i = 0; i < len; i++) {
                    data_y[i] = 0;
                }
				
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(dtype_y);
                return;
            }
        }
    } else {
        size_t x_step = get_dtype_size(dtype_x);
        char* x_data = (char*)x->data;
        
        switch(dtype_y) {
            case FLOAT64: {
                volatile double* data_y = (double*)y->data;
                
                for(size_t i = 0; i < len; i++) {
                    void* x_elem = x_data + i * x_step;
                    double y_val = data_y[i];
                    double x_val = nnl2_convert_to_float64(x_elem, dtype_x);
                    data_y[i] = atan2(y_val, x_val);
                }
				
                break;
            }
            
            case FLOAT32: {
                volatile float* data_y = (float*)y->data;
                
                for(size_t i = 0; i < len; i++) {
                    void* x_elem = x_data + i * x_step;
                    float y_val = data_y[i];
                    float x_val = nnl2_convert_to_float32(x_elem, dtype_x);
                    data_y[i] = atan2f(y_val, x_val);
                }
				
                break;
            }
            
            case INT32: {
                volatile int32_t* data_y = (int32_t*)y->data;
                
                for(size_t i = 0; i < len; i++) {
                    void* x_elem = x_data + i * x_step;
                    int32_t x_val = nnl2_convert_to_int32(x_elem, dtype_x);
                    
                    if (data_y[i] != 0 || x_val != 0) {
                        NNL2_FATAL("Can't apply atan2inplace to non-zero INT32 tensors");
                    }
                }
                
                for(size_t i = 0; i < len; i++) {
                    data_y[i] = 0;
                }
				
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(dtype_y);
                return;
            }
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

/** 
 * @ingroup backend_system
 * @brief Backend implementations for atan2 in-place operation
 * @details
 * Array follows the common backend registration pattern.
 * Currently registered backends:
 *  - nnl2_naive_atan2inplace: Basic reference implementation
 * 
 * @see nnl2_naive_atan2inplace
 */
nnl2_runtime_implementation atan2inplace_backends[] = {
    REGISTER_BACKEND(nnl2_naive_atan2inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};    

/**
 * @brief Function pointer for atan2 in-place operation
 * @ingroup backend_system 
 */
atan2inplacefn nnl2_atan2inplace;

/** 
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(atan2inplace);

/** 
 * @brief Sets the backend for atan2 in-place operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void set_atan2inplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(atan2inplace_backends, nnl2_atan2inplace, backend_name, CURRENT_BACKEND(atan2inplace));
}

/** 
 * @brief Gets the name of the active backend for atan2 in-place operation
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* get_atan2inplace_backend() {
    return CURRENT_BACKEND(atan2inplace);
}

/** 
 * @brief Function declaration for getting all `atan2inplace` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(atan2inplace);

/**
 * @brief Function declaration for getting the number of all `atan2inplace` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(atan2inplace);

#endif /** NNL2_ATAN2_INPLACE_H **/
