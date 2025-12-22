#ifndef NNL2_LOGARITHM_H
#define NNL2_LOGARITHM_H

/** @brief
 * Calculates the natural logarithm of the elements of the input tensor
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
 * A pointer to a new tensor with the result of calculating the logarithm
 */
nnl2_tensor* naive_log(const nnl2_tensor* tensor, bool save_type) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	size_t len = nnl2_product(tensor->shape, tensor->rank);
	
    if (tensor->dtype == INT32) {
        int32_t* tensor_data = (int32_t*)tensor->data;
        int has_non_ones = 0;
        
        // Checking if there are elements other than 1
        for (size_t it = 0; it < len; it++) {
            if (tensor_data[it] != 1) {
                has_non_ones = 1;
                break;
            }
        }
        
        if (has_non_ones) {
            nnl2_tensor* result = nnl2_empty(tensor->shape, tensor->rank, FLOAT64);
            double* result_data = (double*)result->data;
            
            for (size_t it = 0; it < len; it++) {
                result_data[it] = log((double)tensor_data[it]);
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
	
	nnl2_tensor* result = nnl2_empty(tensor->shape, tensor->rank, tensor->dtype);
	
	switch(tensor->dtype) {
		case FLOAT64: {
			volatile double* tensor_data = (double*)tensor->data;
			volatile double* result_data = (double*)result->data;
			for(size_t it = 0; it < len; it++) result_data[it] = log(tensor_data[it]);
			break;
		}
		
		case FLOAT32: {
			volatile float* tensor_data = (float*)tensor->data;
			volatile float* result_data = (float*)result->data;
			for(size_t it = 0; it < len; it++) result_data[it] = logf(tensor_data[it]);
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
 * @brief Backend implementations for logarithm operation
 * @details
 * Array follows the common backend registration pattern for logarithm operations.
 * Currently registered backends:
 *  - nnl2_naive: Basic reference implementation
 * 
 * @see nnl2_naive
 */
nnl2_runtime_implementation log_backends[] = {
	REGISTER_BACKEND(naive_log, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for logarithm operation
 * @ingroup backend_system 
 */
logfn nnl2_logarithm;

/** 
 * @brief Makes the logarithm backend current
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(log);

/** 
 * @brief Sets the backend for logarithm operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see ESET_BACKEND_BY_NAME
 */
void set_log_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(log_backends, log, backend_name, CURRENT_BACKEND(log));
}

/** 
 * @brief Gets the name of the active backend for logarithm operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_log_backend() {
	return CURRENT_BACKEND(log);
}

/** 
 * @brief Function declaration for getting all available logarithm backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(log);

/**
 * @brief Function declaration for getting the number of available logarithm backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(log);

#endif /** NNL2_LOGARITHM_H **/
