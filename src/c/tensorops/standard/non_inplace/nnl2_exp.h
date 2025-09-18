#ifndef NNL2_EXP_H
#define NNL2_EXP_H

/** @brief 
 * Naive implementation of exponential operation
 *
 ** @details
 * Computes element-wise exponential function e^x for each element in the input tensor
 *
 ** @param tensor 
 * Input tensor for exponential operation
 *
 ** @param save_type 
 * Flag to save data type for special case (all elements = 0)
 * true - return INT32 tensor with ones
 * false - return FLOAT64 tensor with ones
 *
 ** @return 
 * New tensor with exponential values applied element-wise
 */
Tensor* naive_exp(Tensor* tensor, bool save_type) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
		
	size_t len = product(tensor->shape, tensor->rank);
	
	// Processing a tensor with an integer data type of INT32.
	// Calculates the exponent for each element, but only if the tensor
	// has at least one non-zero element
	if (tensor->dtype == INT32) {
		int32_t* tensor_data = (int32_t*)tensor->data;
		bool has_non_zero = false;
		
		// Check whether the tensor has at least one non-zero element
		for (size_t it = 0; it < len; it++) {
			if (tensor_data[it] != 0) {
				has_non_zero = true;
				break;
			}
		}
		
		if (has_non_zero) {
			Tensor* result = nnl2_empty(tensor->shape, tensor->rank, FLOAT64);
			double* result_data = (double*)result->data;
			
			for (size_t it = 0; it < len; it++) {
				result_data[it] = exp((double)tensor_data[it]);
			}
			
			#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
				NNL2_FUNC_EXIT();
			#endif
			
			return result;
		} else {
			if(save_type) {
				return nnl2_ones(tensor->shape, tensor->rank, INT32);
			} else {
				return nnl2_ones(tensor->shape, tensor->rank, FLOAT64);
			}
		}
	}
	
	Tensor* result = nnl2_empty(tensor->shape, tensor->rank, tensor->dtype);
	if(len == 0) return result;
	
	switch(tensor->dtype) {
		case FLOAT64: {
			volatile double* tensor_data = (double*)tensor->data;
			volatile double* result_data = (double*)result->data;
			for(size_t it = 0; it < len; it++) result_data[it] = exp(tensor_data[it]);
			break;
		}
		
		case FLOAT32: {
			volatile float* tensor_data = (float*)tensor->data;
			volatile float* result_data = (float*)result->data;
			for(size_t it = 0; it < len; it++) result_data[it] = expf(tensor_data[it]);
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
 * @brief Backend implementations for exponential operation
 * @details
 * Array follows the common backend registration pattern.
 * Currently registered backends:
 *  - naive_exp: Basic reference implementation
 * 
 * @see naive_exp
 */
Implementation exp_backends[] = {
	REGISTER_BACKEND(naive_exp, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for exponential operation
 * @ingroup backend_system 
 */
expfn nnl2_exp;

/** 
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
make_current_backend(exp);

/** 
 * @brief Sets the backend for exponential operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void set_exp_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(exp_backends, exp, backend_name, current_backend(exp));
}

/** 
 * @brief Gets the name of the active backend for exponential operation
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* get_exp_backend() {
	return current_backend(exp);
}

/** 
 * @brief Function declaration for getting all `exp` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(exp);

/**
 * @brief Function declaration for getting the number of all `exp` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(exp);

#endif /** NNL2_EXP_H **/
