#ifndef NNL2_LINSPACE_H
#define NNL2_LINSPACE_H

/** @brief 
 * Creates a 1D tensor with a linear sequence of numbers
 *
 ** @param start 
 * Starting value of the sequence (inclusive)
 *
 ** @param stop 
 * Ending value of the sequence (inclusive when endpoint=true)
 *
 ** @param num 
 * Number of samples to generate (must be >= 0)
 *
 ** @param endpoint 
 * If true, stop is the last sample. Otherwise, it's not included.
 *
 ** @param dtype 
 * Data type of the tensor (only FLOAT32 or FLOAT64)
 * 
 ** @return nnl2_tensor* 
 * Pointer to the created tensor or NULL on error
 * 
 ** @note Unlike arange, linspace includes both endpoints by default
 ** @warning For num=0 returns empty tensor, for num=1 returns [start]
 */
nnl2_tensor* nnl2_naive_float_linspace(float start, float stop, int64_t num, bool endpoint, nnl2_tensor_type dtype) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
        NNL2_FUNC_ENTER();
    #endif 
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if (num < 0) {
            NNL2_ERROR("In function nnl2_naive_float_linspace, num cannot be negative. Got: %lld", num);
            return NULL;
        }
    #endif
    
    size_t capacity = (size_t)num;
    
    if (capacity == 0) {
        NNL2_WARN("In function nnl2_naive_float_linspace, empty tensor requested (num=0)");
    }
    
    int32_t shape[] = { (int32_t)capacity };
    nnl2_tensor* result = nnl2_empty(shape, 1, dtype);
    
    if (result == NULL) {
        NNL2_ERROR("In function nnl2_naive_float_linspace, failed to allocate tensor");
        return NULL;
    }
    
    switch(dtype) {
        case FLOAT32: {
            nnl2_float32* data = (nnl2_float32*)result->data;
            
            if(capacity == 1) {
                data[0] = start;
            } else if(capacity > 1) {
                float step;
                if (endpoint) {
                    step = (stop - start) / ((float)capacity - 1.0f);
                } else {
                    step = (stop - start) / (float)capacity;
                }
                
                float value = start;
                for (size_t it = 0; it < capacity; it++) {
                    data[it] = value;
                    value += step;
                }
            }
			
            //nothing to fill
            
            break;
        }
        
        case FLOAT64: {
            nnl2_float64* data = (nnl2_float64*)result->data;
            
            if(capacity == 1) {
                data[0] = (double)start;
            } else if(capacity > 1) {
                double step;
                if (endpoint) {
                    step = ((double)stop - (double)start) / ((double)capacity - 1.0);
                } else {
                    step = ((double)stop - (double)start) / (double)capacity;
                }
                
                double value = (double)start;
                for (size_t it = 0; it < capacity; it++) {
                    data[it] = value;
                    value += step;
                }
            }
			
            // nothing to fill
            
            break;
        }
        
        default: {
			NNL2_WARN("You are trying to pass an int type to the arange fleet. Use :float32/:float64 or change the :start/:stop to int");
            nnl2_free_tensor(result);
            NNL2_TYPE_ERROR(dtype);
            return NULL;
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
        NNL2_FUNC_EXIT();
    #endif 
    
    return result;
}

/** @brief 
 * Creates a 1D tensor with an integer linear sequence
 *
 ** @param start 
 * Starting value of the sequence (inclusive)
 *
 ** @param stop 
 * Ending value of the sequence (inclusive when endpoint=true)
 *
 ** @param num 
 * Number of samples to generate (must be >= 0)
 *
 ** @param endpoint 
 * If true, stop is the last sample. Otherwise, it's not included.
 *
 ** @param dtype 
 * Data type of the tensor (only INT32 or INT64)
 * 
 ** @return nnl2_tensor* 
 * Pointer to the created tensor or NULL on error
 * 
 ** @note Integer linspace uses linear interpolation and rounding
 ** @warning Values are rounded to nearest integer
 */
nnl2_tensor* nnl2_naive_int_linspace(int64_t start, int64_t stop, int64_t num, bool endpoint, nnl2_tensor_type dtype) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
        NNL2_FUNC_ENTER();
    #endif 
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if (num < 0) {
            NNL2_ERROR("In function nnl2_naive_int_linspace, num cannot be negative. Got: %lld", num);
            return NULL;
        }
    #endif
    
    int64_t capacity = num;
    
    if (capacity <= 0) {
        NNL2_WARN("In function nnl2_naive_int_linspace, empty range. capacity is %lld", capacity);
        capacity = 0;
    }
    
    int32_t shape[] = { (int32_t)capacity };
    nnl2_tensor* result = nnl2_empty(shape, 1, dtype);
    
    if (result == NULL) {
        NNL2_ERROR("In function nnl2_naive_int_linspace, failed to allocate tensor");
        return NULL;
    }
    
    switch(dtype) {
        case INT32: {
            nnl2_int32* data = (nnl2_int32*)result->data;
            
            if (capacity == 1) {
                if (start > INT32_MAX) {
                    data[0] = INT32_MAX;
                } else if (start < INT32_MIN) {
                    data[0] = INT32_MIN;
                } else {
                    data[0] = (nnl2_int32)start;
                }
            } else if (capacity > 1) {
                double step;
                if (endpoint) {
                    step = ((double)stop - (double)start) / ((double)capacity - 1.0);
                } else {
                    step = ((double)stop - (double)start) / (double)capacity;
                }
                
                double value = (double)start;
                for (int64_t it = 0; it < capacity; it++) {
                    int64_t rounded = (int64_t)llround(value);
                    
                    if (rounded > INT32_MAX) {
                        data[it] = INT32_MAX;
                    } else if (rounded < INT32_MIN) {
                        data[it] = INT32_MIN;
                    } else {
                        data[it] = (nnl2_int32)rounded;
                    }
                    
                    value += step;
                }
            }
			
            // nothing to fill
            
            break;
        }
        
        case INT64: {
            nnl2_int64* data = (nnl2_int64*)result->data;
            
            if (capacity == 1) {
                data[0] = start;
            } else if (capacity > 1) {
                double step;
                if (endpoint) {
                    step = ((double)stop - (double)start) / ((double)capacity - 1.0);
                } else {
                    step = ((double)stop - (double)start) / (double)capacity;
                }
                
                double value = (double)start;
                for (int64_t it = 0; it < capacity; it++) {
                    data[it] = (nnl2_int64)llround(value);
                    value += step;
                }
            }
			
            //nothing to fill
            
            break;
        }
        
        default: {
			NNL2_WARN("You are trying to pass an float type to the arange fleet. Use :int32/:int64 or change the :start/:stop to float");
            nnl2_free_tensor(result);
            NNL2_TYPE_ERROR(dtype);
            return NULL;
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
        NNL2_FUNC_EXIT();
    #endif 
    
    return result;
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for integer linspace operation
 */
nnl2_runtime_implementation int_linspace_backends[] = {
    REGISTER_BACKEND(nnl2_naive_int_linspace, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @ingroup backend_system
 * @brief Backend implementations for float linspace operation
 */
nnl2_runtime_implementation float_linspace_backends[] = {
    REGISTER_BACKEND(nnl2_naive_float_linspace, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for integer linspace operation
 * @ingroup backend_system 
 */
nnl2_int_linspacefn nnl2_int_linspace;

/**
 * @brief Function pointer for float linspace operation
 * @ingroup backend_system 
 */
nnl2_float_linspacefn nnl2_float_linspace;

/** 
 * @brief Sets the backend for integer linspace operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for integer linspace
 */
void set_int_linspace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(int_linspace_backends, nnl2_int_linspace, backend_name);
}

/** 
 * @brief Sets the backend for float linspace operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for float linspace
 */
void set_float_linspace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(float_linspace_backends, nnl2_float_linspace, backend_name);
}

#endif /** NNL2_LINSPACE_H **/
