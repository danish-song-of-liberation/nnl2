#ifndef NNL2_LOG1P_INPLACE_H
#define NNL2_LOG1P_INPLACE_H

/** @brief
 * Calculates the natural logarithm of (1 + x) for each tensor element in place
 *
 ** @details
 * The function applies the log1p function (log(1 + x)) to each element of the tensor,
 * replacing the original values with the calculated results.
 * log1p is more accurate than log(1 + x) for small values of x.
 *
 ** @param tensor
 * Pointer to a tensor for processing. The function modifies
 * the tensor data in place
 *
 ** @see log1p
 ** @see log1pf
 **/
void nnl2_naive_log1pinplace(nnl2_tensor* tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	size_t len = nnl2_product(tensor->shape, tensor->rank);
	
	// If tensor empty, exiting the function
	if(len == 0) return;
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor, "A NULL tensor was passed to .log1p!");
	#endif
	
	switch(tensor->dtype) {
		case FLOAT64: {
			double* tensor_data = (double*)tensor->data;		
			for(size_t it = 0; it < len; it++) {
				if (tensor_data[it] <= -1.0) {
					NNL2_FATAL("Can't apply .log1p! to passed tensor - values must be > -1");
				}
				tensor_data[it] = log1p(tensor_data[it]);
			}
			break;
		}
			
		case FLOAT32: {
			float* tensor_data = (float*)tensor->data;		
			for(size_t it = 0; it < len; it++) {
				if (tensor_data[it] <= -1.0f) {
					NNL2_FATAL("Can't apply .log1p! to passed tensor - values must be > -1");
				}
				tensor_data[it] = log1pf(tensor_data[it]);
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
 * @brief Backend implementations for log1p (log(1+x)) in-place operation
 * @details
 * Array follows the common backend registration pattern.
 * Currently registered backends:
 *  - nnl2_naive_log1pinplace: Basic reference implementation
 * 
 * @see nnl2_naive_log1pinplace
 */
nnl2_runtime_implementation log1pinplace_backends[] = {
	REGISTER_BACKEND(nnl2_naive_log1pinplace, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for log1p (log(1+x)) in-place operation
 * @ingroup backend_system 
 */
log1pinplacefn log1pinplace;

/** 
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(log1pinplace);

/** 
 * @brief Sets the backend for log1p (log(1+x)) in-place operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void set_log1pinplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(log1pinplace_backends, log1pinplace, backend_name, CURRENT_BACKEND(log1pinplace));
}

/** 
 * @brief Gets the name of the active backend for log1p (log(1+x)) in-place operation
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* get_log1pinplace_backend() {
	return CURRENT_BACKEND(log1pinplace);
}

/** 
 * @brief Function declaration for getting all `log1pinplace` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(log1pinplace);

/**
 * @brief Function declaration for getting the number of all `log1pinplace` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(log1pinplace);

#endif /** NNL2_LOG1P_INPLACE_H **/
