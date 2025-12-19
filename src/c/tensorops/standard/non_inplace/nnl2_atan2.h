#ifndef NNL2_ATAN2_H
#define NNL2_ATAN2_H

/** @brief 
 * Naive implementation of atan2 operation
 *
 ** @details
 * Computes element-wise atan2(y/x) for corresponding elements in two input tensors
 * Returns angle in radians between [-π, π]
 *
 ** @param y 
 * Input tensor for y-coordinate (numerator)
 *
 ** @param x 
 * Input tensor for x-coordinate (denominator)
 *
 ** @return nnl2_tensor*
 * New tensor with atan2 values applied element-wise
 *
 ** @note 
 * sorry for the poor quality of the code, I'm in a hurry
 */
nnl2_tensor* naive_atan2(nnl2_tensor* y, nnl2_tensor* x) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif

    size_t len = product(y->shape, y->rank);
    
    TensorType dtype_y = y->dtype;
    TensorType dtype_x = x->dtype;
    
    TensorType result_dtype;
    if(dtype_y == INT32 && dtype_x == INT32) {
        int32_t* y_data = (int32_t*)y->data;
        int32_t* x_data = (int32_t*)x->data;
        
        bool all_zeros = true;
        for (size_t it = 0; it < len; it++) {
            if (y_data[it] != 0 || x_data[it] != 0) {
                all_zeros = false;
                break;
            }
        }
        
        if (all_zeros) {
            Tensor* result = nnl2_zeros(y->shape, y->rank, FLOAT64);
            #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                NNL2_FUNC_EXIT();
            #endif
			
            return result;
        }
		
        result_dtype = FLOAT64;
    } else {
        result_dtype = (dtype_y == FLOAT64 || dtype_x == FLOAT64) ? FLOAT64 : FLOAT32;
    }
    
    Tensor* result = nnl2_empty(y->shape, y->rank, result_dtype);
    if (len == 0) return result;
    
    if (result_dtype == FLOAT64) {
        volatile double* result_data = (double*)result->data;
        
        if (dtype_y == FLOAT64 && dtype_x == FLOAT64) {
            volatile double* y_data = (double*)y->data;
            volatile double* x_data = (double*)x->data;
			
            for(size_t it = 0; it < len; it++) {
                result_data[it] = atan2(y_data[it], x_data[it]);
            }
			
        } else if (dtype_y == FLOAT64) {
            volatile double* y_data = (double*)y->data;
            size_t x_step = get_dtype_size(dtype_x);
            char* x_data = (char*)x->data;
            
            for(size_t it = 0; it < len; it++) {
                void* x_elem = x_data + it * x_step;
                double y_val = y_data[it];
                double x_val = nnl2_convert_to_float64(x_elem, dtype_x);
                result_data[it] = atan2(y_val, x_val);
            }
			
        } else if (dtype_x == FLOAT64) {
            volatile double* x_data = (double*)x->data;
            size_t y_step = get_dtype_size(dtype_y);
            char* y_data = (char*)y->data;
            
            for(size_t it = 0; it < len; it++) {
                void* y_elem = y_data + it * y_step;
                double y_val = nnl2_convert_to_float64(y_elem, dtype_y);
                double x_val = x_data[it];
                result_data[it] = atan2(y_val, x_val);
            }
			
        } else {
            size_t y_step = get_dtype_size(dtype_y);
            size_t x_step = get_dtype_size(dtype_x);
            char* y_data = (char*)y->data;
            char* x_data = (char*)x->data;
            
            for(size_t it = 0; it < len; it++) {
                void* y_elem = y_data + it * y_step;
                void* x_elem = x_data + it * x_step;
                double y_val = nnl2_convert_to_float64(y_elem, dtype_y);
                double x_val = nnl2_convert_to_float64(x_elem, dtype_x);
                result_data[it] = atan2(y_val, x_val);
            }
        }
    } else { // FLOAT32
        volatile float* result_data = (float*)result->data;
        
        if (dtype_y == FLOAT32 && dtype_x == FLOAT32) {
            volatile float* y_data = (float*)y->data;
            volatile float* x_data = (float*)x->data;
            for(size_t it = 0; it < len; it++) {
                result_data[it] = atan2f(y_data[it], x_data[it]);
            }
			
        } else if (dtype_y == FLOAT32) {
            volatile float* y_data = (float*)y->data;
            size_t x_step = get_dtype_size(dtype_x);
            char* x_data = (char*)x->data;
            
            for(size_t it = 0; it < len; it++) {
                void* x_elem = x_data + it * x_step;
                float y_val = y_data[it];
                float x_val = nnl2_convert_to_float32(x_elem, dtype_x);
                result_data[it] = atan2f(y_val, x_val);
            }
			
        } else if (dtype_x == FLOAT32) {
            volatile float* x_data = (float*)x->data;
            size_t y_step = get_dtype_size(dtype_y);
            char* y_data = (char*)y->data;
            
            for(size_t it = 0; it < len; it++) {
                void* y_elem = y_data + it * y_step;
                float y_val = nnl2_convert_to_float32(y_elem, dtype_y);
                float x_val = x_data[it];
                result_data[it] = atan2f(y_val, x_val);
            }
			
        } else {
            size_t y_step = get_dtype_size(dtype_y);
            size_t x_step = get_dtype_size(dtype_x);
            char* y_data = (char*)y->data;
            char* x_data = (char*)x->data;
            
            for(size_t it = 0; it < len; it++) {
                void* y_elem = y_data + it * y_step;
                void* x_elem = x_data + it * x_step;
                float y_val = nnl2_convert_to_float32(y_elem, dtype_y);
                float x_val = nnl2_convert_to_float32(x_elem, dtype_x);
                result_data[it] = atan2f(y_val, x_val);
            }
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return result;
}

/** 
 * @ingroup backend_system
 * @brief Backend implementations for atan2 operation
 * @details
 * Array follows the common backend registration pattern.
 * Currently registered backends:
 *  - naive_atan2: Basic reference implementation
 * 
 * @see naive_atan2
 */
Implementation atan2_backends[] = {
    REGISTER_BACKEND(naive_atan2, nnl2_naive, NAIVE_BACKEND_NAME),
};    

/**
 * @brief Function pointer for atan2 operation
 * @ingroup backend_system 
 */
atan2fn nnl2_atan2;

/** 
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
make_current_backend(atan2);

/** 
 * @brief Sets the backend for atan2 operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void set_atan2_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(atan2_backends, nnl2_atan2, backend_name, current_backend(atan2));
}

/** 
 * @brief Gets the name of the active backend for atan2 operation
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* get_atan2_backend() {
    return current_backend(atan2);
}

/** 
 * @brief Function declaration for getting all `atan2` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(atan2);

/**
 * @brief Function declaration for getting the number of all `atan2` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(atan2);

#endif /** NNL2_ATAN2_H **/
