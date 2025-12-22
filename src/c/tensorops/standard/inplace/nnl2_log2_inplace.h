#ifndef NNL2_LOG2_INPLACE_H
#define NNL2_LOG2_INPLACE_H

/** @brief
 * Calculates the base-2 logarithm of each tensor element in place
 *
 ** @details
 * The function applies the base-2 logarithm function (log2(x)) to each element of the tensor,
 * replacing the original values with the calculated results
 *
 ** @param tensor
 * Pointer to a tensor for processing. The function modifies
 * the tensor data in place
 *
 ** @see log2
 ** @see log2f
 **/
void nnl2_naive_log2inplace(nnl2_tensor* tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	size_t len = nnl2_product(tensor->shape, tensor->rank);
	
	// If tensor empty, exiting the function
	if(len == 0) return;
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor, "A NULL tensor was passed to .log2!");
	#endif
	
	switch(tensor->dtype) {
		case FLOAT64: {
			double* tensor_data = (double*)tensor->data;		
			for(size_t it = 0; it < len; it++) tensor_data[it] = log2(tensor_data[it]);
			break;
		}
			
		case FLOAT32: {
			float* tensor_data = (float*)tensor->data;		
			for(size_t it = 0; it < len; it++) tensor_data[it] = log2f(tensor_data[it]);
			break;	
		}
			
		case INT32: {
			int32_t* tensor_data = (int32_t*)tensor->data;	
			
			// Check if all values are powers of two
			for(size_t it = 0; it < len; it++) {
				int32_t value = tensor_data[it];
				if (value <= 0 || (value & (value - 1)) != 0) {
					NNL2_FATAL("Can't apply .log2! to passed tensor - values must be positive powers of two");
				}
			}
			
			for(size_t it = 0; it < len; it++) {
				int32_t value = tensor_data[it];
				tensor_data[it] = 0;
				// Calculate log2 for powers of two
				while (value >>= 1) {
					tensor_data[it]++;
				}
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
 * @brief Backend implementations for base-2 logarithm in-place operation
 * @details
 * Array follows the common backend registration pattern.
 * Currently registered backends:
 *  - nnl2_naive_log2inplace: Basic reference implementation
 * 
 * @see nnl2_naive_log2inplace
 */
nnl2_runtime_implementation log2inplace_backends[] = {
	REGISTER_BACKEND(nnl2_naive_log2inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for base-2 logarithm in-place operation
 * @ingroup backend_system 
 */
log2inplacefn log2inplace;

/** 
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(log2inplace);

/** 
 * @brief Sets the backend for base-2 logarithm in-place operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void set_log2inplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(log2inplace_backends, log2inplace, backend_name, CURRENT_BACKEND(log2inplace));
}

/** 
 * @brief Gets the name of the active backend for base-2 logarithm in-place operation
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* get_log2inplace_backend() {
	return CURRENT_BACKEND(log2inplace);
}

/** 
 * @brief Function declaration for getting all `log2inplace` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(log2inplace);

/**
 * @brief Function declaration for getting the number of all `log2inplace` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(log2inplace);

#endif /** NNL2_LOG2_INPLACE_H **/
