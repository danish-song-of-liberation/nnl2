#ifndef NNL2_ARANGE_H
#define NNL2_ARANGE_H

/** @brief 
 * Creates a 1D tensor with an integer arithmetic progression
 *
 ** @param from 
 * Starting value of the sequence
 *
 ** @param to 
 * Ending value of the sequence (not included)
 *
 ** @param step 
 * Step size (cannot be zero)
 *
 ** @param dtype 
 * Data type of the tensor (only INT32 or INT64)
 * 
 ** @return nnl2_tensor* 
 * Pointer to the created tensor or NULL on error
 * 
 ** @note For positive step from < to, for negative step from > to
 ** @warning When INT32 overflow occurs, values are clamped to INT32_MAX
 */
nnl2_tensor* nnl2_naive_int_arange(int64_t from, int64_t to, int64_t step, nnl2_tensor_type dtype) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
        NNL2_FUNC_ENTER();
    #endif 
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if (step == 0) {
            NNL2_ERROR("In function nnl2_naive_int_arange, step cannot be zero");
            return NULL;
        }
        
        if (step > 0 && from >= to) {
            NNL2_ERROR("In function nnl2_naive_int_arange, invalid range. from >= to with positive step");
            return NULL;
        }
        
        if (step < 0 && from <= to) {
            NNL2_ERROR("In function nnl2_naive_int_arange, invalid range. from <= to with negative step");
            return NULL;
        }
    #endif
    
    int64_t capacity;
    if (step > 0) {
        capacity = (to - from + step - 1) / step; 
    } else {
        capacity = (from - to - step - 1) / (-step);  
    }
    
    if (capacity <= 0) {
        NNL2_WARN("In function nnl2_naive_int_arange, empty range. capacity is %lld", capacity);
        capacity = 0;
    }
    
    int32_t shape[] = { (int32_t)capacity };
    nnl2_tensor* result = nnl2_empty(shape, 1, dtype);
    
    if (result == NULL) {
        NNL2_ERROR("In function nnl2_naive_int_arange, failed to allocate tensor");
        return NULL;
    }
    
    switch(dtype) {
        case INT32: {
            nnl2_int32* data = (nnl2_int32*)result->data;
            for (int64_t it = 0, val = from; it < capacity; it++, val += step) {
                if (val > INT32_MAX) {
                    data[it] = INT32_MAX;
                } else if (val < INT32_MIN) {
                    data[it] = INT32_MIN;
                } else {
                    data[it] = (nnl2_int32)val;
                }
            }
            
            break;
        }
        
        case INT64: {
            nnl2_int64* data = (nnl2_int64*)result->data;
            for (int64_t it = 0, val = from; it < capacity; it++, val += step) {
                data[it] = (nnl2_int64)val;
            }
            
            break;
        }
        
        default: {
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
 * Creates a 1D tensor with a floating-point arithmetic progression
 *
 ** @param from 
 * Starting value of the sequence
 *
 ** @param to 
 * Ending value of the sequence (not included)
 *
 ** @param step 
 * Step size (cannot be zero or too small)
 *
 ** @param dtype 
 * Data type of the tensor (only FLOAT32 or FLOAT64)
 *
 ** @return nnl2_tensor* 
 * Pointer to the created tensor or NULL on error
 *
 ** @warning 
 * Floating-point precision errors may cause unexpected results
 * near boundaries
 */
nnl2_tensor* nnl2_naive_float_arange(float from, float to, float step, nnl2_tensor_type dtype) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
		NNL2_FUNC_ENTER();
	#endif 
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if (fabsf(step) < FLT_EPSILON) {
			NNL2_ERROR("In function nnl2_naive_float_arange, step cannot be zero or too small");
			return NULL;
		}
		
		if (from >= to && step > 0) {
			NNL2_ERROR("In function nnl2_naive_float_arange, invalid range. from >= to with positive step");
			return NULL;
		}
		
		if (from <= to && step < 0) {
			NNL2_ERROR("In function nnl2_naive_float_arange, invalid range. from <= to with negative step");
			return NULL;
		}
	#endif
    
    size_t capacity;
    if (step > 0) {
        capacity = (size_t)ceilf((to - from - 1e-6f) / step);
    } else {
        capacity = (size_t)ceilf((from - to - 1e-6f) / (-step));
    }
    
    if (capacity == 0) {
        NNL2_WARN("In function nnl2_naive_float_arange, empty range. capacity is zero");
    }
    
    int32_t shape[] = { (int32_t)capacity };
    nnl2_tensor* result = nnl2_empty(shape, 1, dtype);
    
    if (result == NULL) {
        NNL2_ERROR("In function nnl2_naive_float_arange, failed to allocate tensor");
        return NULL;
    }
    
    switch(dtype) {
        case FLOAT32: {
            nnl2_float32* data = (nnl2_float32*)result->data;
			
            float value = from;
            for (size_t it = 0; it < capacity; it++, value += step) {
                data[it] = value;
            }
			
            break;
        }
        
        case FLOAT64: {
            nnl2_float64* data = (nnl2_float64*)result->data;
			
            double value = (double)from;
            double dstep = (double)step;
            for (size_t it = 0; it < capacity; it++, value += dstep) {
                data[it] = value;
            }
			
            break;
        }
        
        default: {
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
 * @brief Backend implementations for integer arange operation
 */
nnl2_runtime_implementation int_arange_backends[] = {
    REGISTER_BACKEND(nnl2_naive_int_arange, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @ingroup backend_system
 * @brief Backend implementations for float arange operation
 */
nnl2_runtime_implementation float_arange_backends[] = {
    REGISTER_BACKEND(nnl2_naive_float_arange, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for integer arange operation
 * @ingroup backend_system 
 */
nnl2_int_arangefn nnl2_int_arange;

/**
 * @brief Function pointer for float arange operation
 * @ingroup backend_system 
 */
nnl2_float_arangefn nnl2_float_arange;

/** 
 * @brief Sets the backend for integer arange operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for integer arange
 */
void set_int_arange_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(int_arange_backends, nnl2_int_arange, backend_name);
}

/** 
 * @brief Sets the backend for float arange operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for float arange
 */
void set_float_arange_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(float_arange_backends, nnl2_float_arange, backend_name);
}

#endif /** NNL2_ARANGE_H **/
