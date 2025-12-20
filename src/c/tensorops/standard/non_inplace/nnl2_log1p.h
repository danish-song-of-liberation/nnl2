#ifndef NNL2_LOG1P_H
#define NNL2_LOG1P_H

/** @brief
 * Calculates the natural logarithm of (1 + x) for the elements of the input tensor
 *
 ** @param tensor
 * Pointer to the input tensor
 *
 ** @param save_type 
 * Flag to save data type for special case (all elements = 0)
 * true - return INT32 tensor with zeros
 * false - return FLOAT64 tensor with zeros
 *
 ** @return
 * A pointer to a new tensor with the result of calculating log(1 + x)
 */
Tensor* naive_log1p(const Tensor* tensor, bool save_type) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	size_t len = product(tensor->shape, tensor->rank);
	
    if(tensor->dtype == INT32) {
        int32_t* tensor_data = (int32_t*)tensor->data;
        int has_non_zeros = 0;
        
        // Checking if there are elements other than 0
        for(size_t it = 0; it < len; it++) {
            if (tensor_data[it] != 0) {
                has_non_zeros = 1;
                break;
            }
        }
        
        if(has_non_zeros) {
            // Check if any values are <= -1 (invalid for log1p)
            for (size_t it = 0; it < len; it++) {
                if (tensor_data[it] <= -1) {
                    NNL2_ERROR("Can't apply .log1p to passed tensor. values must be > -1");
                    return NULL;
                }
            }
            
            Tensor* result = nnl2_empty(tensor->shape, tensor->rank, FLOAT64);
            double* result_data = (double*)result->data;
            
            for (size_t it = 0; it < len; it++) {
                result_data[it] = log1p((double)tensor_data[it]);
            }
			
            return result;
        } else {
			if(save_type) {
				return nnl2_zeros(tensor->shape, tensor->rank, INT32);
			} else {
				return nnl2_zeros(tensor->shape, tensor->rank, FLOAT64);
			}
        }
    }
	
	Tensor* result = nnl2_empty(tensor->shape, tensor->rank, tensor->dtype);
	
	switch(tensor->dtype) {
		case FLOAT64: {
			double* tensor_data = (double*)tensor->data;
			double* result_data = (double*)result->data;
			
			// Check for values <= -1
			for(size_t it = 0; it < len; it++) {
				if (tensor_data[it] <= -1.0) {
					NNL2_ERROR("Can't apply .log1p to passed tensor. values must be > -1");
					nnl2_free_tensor(result);
					return NULL;
				}
			}
			
			for(size_t it = 0; it < len; it++) {
				result_data[it] = log1p(tensor_data[it]);
			}
			break;
		}
		
		case FLOAT32: {
			float* tensor_data = (float*)tensor->data;
			float* result_data = (float*)result->data;
			
			// Check for values <= -1
			for(size_t it = 0; it < len; it++) {
				if (tensor_data[it] <= -1.0f) {
					NNL2_ERROR("Can't apply .log1p to passed tensor. values must be > -1");
					nnl2_free_tensor(result);
					return NULL;
				}
			}
			
			for(size_t it = 0; it < len; it++) {
				result_data[it] = log1pf(tensor_data[it]);
			}
			
			break;
		}
		
		default: {
			NNL2_TYPE_ERROR(tensor->dtype);
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
 * @brief Backend implementations for log1p (log(1+x)) operation
 * @details
 * Array follows the common backend registration pattern for log1p operations.
 * Currently registered backends:
 *  - nnl2_naive: Basic reference implementation
 * 
 * @see nnl2_naive
 */
Implementation log1p_backends[] = {
	REGISTER_BACKEND(naive_log1p, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for log1p (log(1+x)) operation
 * @ingroup backend_system 
 */
log1pfn nnl2_log1p;

/** 
 * @brief Makes the log1p backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(log1p);

/** 
 * @brief Sets the backend for log1p operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see ESET_BACKEND_BY_NAME
 */
void set_log1p_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(log1p_backends, nnl2_log1p, backend_name, current_backend(log1p));
}

/** 
 * @brief Gets the name of the active backend for log1p operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_log1p_backend() {
	return current_backend(log1p);
}

/** 
 * @brief Function declaration for getting all available log1p backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(log1p);

/**
 * @brief Function declaration for getting the number of available log1p backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(log1p);

#endif /** NNL2_LOG1P_H **/
