#ifndef NNL2_LOG10_H
#define NNL2_LOG10_H

/** @brief
 * Calculates the base-10 logarithm of the elements of the input tensor
 *
 ** @param tensor
 * Pointer to the input tensor
 *
 ** @param save_type 
 * Flag to save data type for special case (all elements = 1)
 * true - return INT32 tensor with zeros
 * false - return FLOAT64 tensor with zeros
 *
 ** @return
 * A pointer to a new tensor with the result of calculating the base-10 logarithm
 */
nnl2_tensor* naive_log10(const nnl2_tensor* tensor, bool save_type) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	size_t len = nnl2_product(tensor->shape, tensor->rank);
	
    if (tensor->dtype == INT32 || tensor->dtype == INT64) {
        bool has_non_ones = false;
        
        // Checking if there are elements other than 1
        if (tensor->dtype == INT32) {
            int32_t* tensor_data = (int32_t*)tensor->data;
            for (size_t it = 0; it < len; it++) {
                if (tensor_data[it] != 1) {
                    has_non_ones = true;
                    break;
                }
            }
        } else { // INT64
            int64_t* tensor_data = (int64_t*)tensor->data;
            for (size_t it = 0; it < len; it++) {
                if (tensor_data[it] != 1) {
                    has_non_ones = true;
                    break;
                }
            }
        }
        
        if (has_non_ones) {
            nnl2_tensor* result = nnl2_empty(tensor->shape, tensor->rank, FLOAT64);
            double* result_data = (double*)result->data;
            
            if (tensor->dtype == INT32) {
                int32_t* tensor_data = (int32_t*)tensor->data;
                for (size_t it = 0; it < len; it++) {
                    if (tensor_data[it] <= 0) {
                        NNL2_ERROR("Log10 of non-positive value at index %zu\n", it);
                        nnl2_free_tensor(result);
                        return NULL;
                    }
                    result_data[it] = log10((double)tensor_data[it]);
                }
            } else { // INT64
                int64_t* tensor_data = (int64_t*)tensor->data;
                for (size_t it = 0; it < len; it++) {
                    if (tensor_data[it] <= 0) {
                        NNL2_ERROR("Log10 of non-positive value at index %zu\n", it);
                        nnl2_free_tensor(result);
                        return NULL;
                    }
                    result_data[it] = log10((double)tensor_data[it]);
                }
            }
			
            return result;
        } else {
			if(save_type) {
				if (tensor->dtype == INT32) {
					return nnl2_zeros(tensor->shape, tensor->rank, INT32);
				} else { // INT64
					return nnl2_zeros(tensor->shape, tensor->rank, INT64);
				}
			} else {
				return nnl2_zeros(tensor->shape, tensor->rank, FLOAT64);
			}
        }
    }
	
	nnl2_tensor* result = nnl2_empty(tensor->shape, tensor->rank, tensor->dtype);
    if(len == 0) return result;
	
	switch(tensor->dtype) {
		case FLOAT64: {
			volatile double* tensor_data = (double*)tensor->data;
			volatile double* result_data = (double*)result->data;
			for(size_t it = 0; it < len; it++) {
                if (tensor_data[it] <= 0.0) {
                    NNL2_ERROR("Log10 of non-positive value at index %zu\n", it);
                    nnl2_free_tensor(result);
                    return NULL;
                }
                result_data[it] = log10(tensor_data[it]);
            }
			break;
		}
		
		case FLOAT32: {
			volatile float* tensor_data = (float*)tensor->data;
			volatile float* result_data = (float*)result->data;
			for(size_t it = 0; it < len; it++) {
                if (tensor_data[it] <= 0.0f) {
                    NNL2_ERROR("Log10 of non-positive value at index %zu\n", it);
                    nnl2_free_tensor(result);
                    return NULL;
                }
                result_data[it] = log10f(tensor_data[it]);
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
 * @brief Backend implementations for base-10 logarithm operation
 * @details
 * Array follows the common backend registration pattern for base-10 logarithm operations.
 * Currently registered backends:
 *  - nnl2_naive: Basic reference implementation
 * 
 * @see nnl2_naive
 */
nnl2_runtime_implementation log10_backends[] = {
	REGISTER_BACKEND(naive_log10, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for base-10 logarithm operation
 * @ingroup backend_system 
 */
log10fn nnl2_log10;

/** 
 * @brief Makes the base-10 logarithm backend current
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(log10);

/** 
 * @brief Sets the backend for base-10 logarithm operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see ESET_BACKEND_BY_NAME
 */
void set_log10_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(log10_backends, nnl2_log10, backend_name, CURRENT_BACKEND(log10));
}

/** 
 * @brief Gets the name of the active backend for base-10 logarithm operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_log10_backend() {
	return CURRENT_BACKEND(log10);
}

/** 
 * @brief Function declaration for getting all available base-10 logarithm backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(log10);

/**
 * @brief Function declaration for getting the number of available base-10 logarithm backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(log10);

#endif /** NNL2_LOG10_H **/
