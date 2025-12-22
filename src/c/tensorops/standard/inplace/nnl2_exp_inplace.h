#ifndef NNL2_EXP_INPLACE_H
#define NNL2_EXP_INPLACE_H

/** @brief
 * Calculates the exponent of each tensor element in place
 *
 ** @details
 * The function applies the exponential function (e^x) to each element of the tensor,
 * replacing the original values with the calculated results
 *
 ** @param tensor
 * Pointer to a tensor for processing. The function modifies
 * the tensor data in place
 *
 ** @see exp
 ** @see expf
 **/
void nnl2_naive_expinplace(nnl2_tensor* tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	size_t len = product(tensor->shape, tensor->rank);
	
	// If tensor empty, exiting the function
	if(len == 0) return;
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor, "A NULL tensor was passed to .exp!");
	#endif
	
	switch(tensor->dtype) {
		case FLOAT64: {
			volatile double* tensor_data = (double*)tensor->data;		
			for(size_t it = 0; it < len; it++) tensor_data[it] = exp(tensor_data[it]);
			break;
		}
			
		case FLOAT32: {
			volatile float* tensor_data = (float*)tensor->data;		
			for(size_t it = 0; it < len; it++) tensor_data[it] = expf(tensor_data[it]);
			break;	
		}
			
		case INT32: {
			volatile int32_t* tensor_data = (int32_t*)tensor->data;	
			
			// 0 is the only integer for which exp gives an integer
			for (size_t it = 0; it < len; it++) {
				if (tensor_data[it] != 0) {
					NNL2_FATAL("Can't apply .exp! to a passed tensor");
				}
			}
			
			for (size_t it = 0; it < len; it++) {
				tensor_data[it] = 1;
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
 * @brief Backend implementations for exponential in-place operation
 * @details
 * Array follows the common backend registration pattern.
 * Currently registered backends:
 *  - nnl2_naive_expinplace: Basic reference implementation
 * 
 * @see nnl2_naive_expinplace
 */
nnl2_runtime_implementation expinplace_backends[] = {
	REGISTER_BACKEND(nnl2_naive_expinplace, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for exponential in-place operation
 * @ingroup backend_system 
 */
expinplacefn expinplace;

/** 
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(expinplace);

/** 
 * @brief Sets the backend for exponential in-place operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void set_expinplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(expinplace_backends, expinplace, backend_name, CURRENT_BACKEND(expinplace));
}

/** 
 * @brief Gets the name of the active backend for exponential in-place operation
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* get_expinplace_backend() {
	return CURRENT_BACKEND(expinplace);
}

/** 
 * @brief Function declaration for getting all `expinplace` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(expinplace);

/**
 * @brief Function declaration for getting the number of all `expinplace` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(expinplace);

#endif /** NNL2_EXP_INPLACE_H **/
