#ifndef NNL2_LOG2_H
#define NNL2_LOG2_H

/** @brief
 * Calculates the base-2 logarithm of the elements of the input tensor
 *
 ** @param tensor
 * Pointer to the input tensor
 *
 ** @param save_type 
 * Flag to save data type for special case (all elements are powers of two)
 * true - return INT32 tensor with log2 values
 * false - return FLOAT64 tensor with log2 values
 *
 ** @return
 * A pointer to a new tensor with the result of calculating the base-2 logarithm
 */
nnl2_tensor* naive_log2(const nnl2_tensor* tensor, bool save_type) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	size_t len = product(tensor->shape, tensor->rank);
	
    if (tensor->dtype == INT32) {
        int32_t* tensor_data = (int32_t*)tensor->data;
        int all_powers_of_two = 1;
        
        // Checking if all elements are positive powers of two
        for (size_t it = 0; it < len; it++) {
            int32_t value = tensor_data[it];
            if (value <= 0 || (value & (value - 1)) != 0) {
                all_powers_of_two = 0;
                break;
            }
        }
        
        if (!all_powers_of_two) {
            nnl2_tensor* result = nnl2_empty(tensor->shape, tensor->rank, FLOAT64);
            double* result_data = (double*)result->data;
            
            for (size_t it = 0; it < len; it++) {
                result_data[it] = log2((double)tensor_data[it]);
            }
			
            return result;
        } else {
            if(save_type) {
                nnl2_tensor* result = nnl2_empty(tensor->shape, tensor->rank, INT32);
                int32_t* result_data = (int32_t*)result->data;
                
                for (size_t it = 0; it < len; it++) {
                    int32_t value = tensor_data[it];
                    result_data[it] = 0;
                    // Calculate log2 for powers of two
                    while (value >>= 1) {
                        result_data[it]++;
                    }
                }
                return result;
            } else {
                nnl2_tensor* result = nnl2_empty(tensor->shape, tensor->rank, FLOAT64);
                double* result_data = (double*)result->data;
                
                for (size_t it = 0; it < len; it++) {
                    int32_t value = tensor_data[it];
                    int32_t log_val = 0;
                    // Calculate log2 for powers of two
                    while (value >>= 1) {
                        log_val++;
                    }
                    result_data[it] = (double)log_val;
                }
                return result;
            }
        }
    }
	
	nnl2_tensor* result = nnl2_empty(tensor->shape, tensor->rank, tensor->dtype);
	
	switch(tensor->dtype) {
		case FLOAT64: {
			volatile double* tensor_data = (double*)tensor->data;
			volatile double* result_data = (double*)result->data;
			for(size_t it = 0; it < len; it++) result_data[it] = log2(tensor_data[it]);
			break;
		}
		
		case FLOAT32: {
			volatile float* tensor_data = (float*)tensor->data;
			volatile float* result_data = (float*)result->data;
			for(size_t it = 0; it < len; it++) result_data[it] = log2f(tensor_data[it]);
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
 * @brief Backend implementations for base-2 logarithm operation
 * @details
 * Array follows the common backend registration pattern for base-2 logarithm operations.
 * Currently registered backends:
 *  - nnl2_naive: Basic reference implementation
 * 
 * @see nnl2_naive
 */
nnl2_runtime_implementation log2_backends[] = {
	REGISTER_BACKEND(naive_log2, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for base-2 logarithm operation
 * @ingroup backend_system 
 */
log2fn nnl2_log2;

/** 
 * @brief Makes the base-2 logarithm backend current
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(log2);

/** 
 * @brief Sets the backend for base-2 logarithm operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see ESET_BACKEND_BY_NAME
 */
void set_log2_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(log2_backends, nnl2_log2, backend_name, CURRENT_BACKEND(log2));
}

/** 
 * @brief Gets the name of the active backend for base-2 logarithm operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_log2_backend() {
	return CURRENT_BACKEND(log2);
}

/** 
 * @brief Function declaration for getting all available base-2 logarithm backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(log2);

/**
 * @brief Function declaration for getting the number of available base-2 logarithm backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(log2);

#endif /** NNL2_LOG2_H **/
