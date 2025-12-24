#ifndef NNL2_LOG_INPLACE_H
#define NNL2_LOG_INPLACE_H

/** @brief
 * Calculates the natural logarithm of each tensor element in place
 *
 ** @details
 * The function applies the natural logarithm function (ln(x)) to each element of the tensor,
 * replacing the original values with the calculated results
 *
 ** @param tensor
 * Pointer to a tensor for processing. The function modifies
 * the tensor data in place
 *
 ** @see log
 ** @see logf
 **/
void nnl2_naive_loginplace(nnl2_tensor* tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	size_t len = nnl2_product(tensor->shape, tensor->rank);
	
	// If tensor empty, exiting the function
	if(len == 0) return;
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor, "A NULL tensor was passed to .log!");
	#endif
	
	switch(tensor->dtype) {
		case FLOAT64: {
			volatile double* tensor_data = (double*)tensor->data;		
			for(size_t it = 0; it < len; it++) tensor_data[it] = log(tensor_data[it]);
			break;
		}
			
		case FLOAT32: {
			volatile float* tensor_data = (float*)tensor->data;		
			for(size_t it = 0; it < len; it++) tensor_data[it] = logf(tensor_data[it]);
			break;	
		}
		
		case INT64: {
			volatile int64_t* tensor_data = (int64_t*)tensor->data;	
			
			// 1 is the only positive integer for which log gives an integer (log(1) = 0)
			for(size_t it = 0; it < len; it++) {
				if (tensor_data[it] != 1) {
					NNL2_FATAL("Can't apply .log! to passed INT64 tensor with values not equal to 1");
				}
			}
			
			// Set all values to 0 (since log(1) = 0)
			for(size_t it = 0; it < len; it++) {
				tensor_data[it] = 0;
			}
			
			break;	
		}
			
		case INT32: {
			volatile int32_t* tensor_data = (int32_t*)tensor->data;	
			
			// 1 is the only integer for which log gives an integer
			for(size_t it = 0; it < len; it++) {
				if (tensor_data[it] != 1) {
					NNL2_FATAL("Can't apply .log! to passed tensor");
				}
			}
			
			for(size_t it = 0; it < len; it++) {
				tensor_data[it] = 0;
			}
			
			break;	
		}
			
		default: {
			NNL2_TYPE_ERROR(tensor->dtype);
			return;
		}
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}	

/** 
 * @ingroup backend_system
 * @brief Backend implementations for logarithm in-place operation
 * @details
 * Array follows the common backend registration pattern.
 * Currently registered backends:
 *  - nnl2_naive_loginplace: Basic reference implementation
 * 
 * @see nnl2_naive_loginplace
 */
nnl2_runtime_implementation loginplace_backends[] = {
	REGISTER_BACKEND(nnl2_naive_loginplace, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for logarithm in-place operation
 * @ingroup backend_system 
 */
loginplacefn loginplace;

/** 
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(loginplace);

/** 
 * @brief Sets the backend for logarithm in-place operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void set_loginplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(loginplace_backends, loginplace, backend_name, CURRENT_BACKEND(loginplace));
}

/** 
 * @brief Gets the name of the active backend for logarithm in-place operation
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* get_loginplace_backend() {
	return CURRENT_BACKEND(loginplace);
}

/** 
 * @brief Function declaration for getting all `loginplace` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(loginplace);

/**
 * @brief Function declaration for getting the number of all `loginplace` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(loginplace);

#endif /** NNL2_LOG_INPLACE_H **/
