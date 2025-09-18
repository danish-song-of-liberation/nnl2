#ifndef NNL2_COPY_H
#define NNL2_COPY_H

/** @brief
 * Creates a copy of a tensor with possible data type conversion
 *
 ** @param tensor
 * Pointer to the source tensor for copying
 *
 ** @param copy_type
 * The target data type for the copy
 *
 ** @return 
 * A pointer to a new tensor copy, or NULL if an error occurs
 *
 ** @note
 * Can perform additional checks depending on the safety level
 *
 ** @see nnl2_empty
 ** @see nnl2_convert_to_float64
 ** @see nnl2_convert_to_float32
 ** @see nnl2_convert_to_int32
 **/
Tensor* naive_copy(Tensor* tensor, TensorType copy_type) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Additional checks depending on the safety level
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		// NULL checks
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor, "Passed tensor is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor->data, "Tensor data is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor->shape, "Tensor shape is NULL", NULL);
	#endif
	
	TensorType dtype = tensor->dtype;
	
	size_t total_elems = product(tensor->shape, tensor->rank);
	Tensor* result;
	
	if(dtype == copy_type) {
		result = nnl2_empty(tensor->shape, tensor->rank, dtype);
		
		// Element-by-element copying based on data type
		switch(dtype) {
			case FLOAT64: {
				double* cast_data_original = (double*)tensor->data;
				double* cast_data_copy = (double*)result->data;	
				for(size_t it = 0; it < total_elems; it++) cast_data_copy[it] = cast_data_original[it];
				break;
			}
			
			case FLOAT32: {
				float* cast_data_original = (float*)tensor->data;
				float* cast_data_copy = (float*)result->data;	
				for(size_t it = 0; it < total_elems; it++) cast_data_copy[it] = cast_data_original[it];
				break;
			}
			
			case INT32: {
				int32_t* cast_data_original = (int32_t*)tensor->data;
				int32_t* cast_data_copy = (int32_t*)result->data;	
				for(size_t it = 0; it < total_elems; it++) cast_data_copy[it] = cast_data_original[it];
				break;
			} 
			
			default: {
				NNL2_TYPE_ERROR(dtype);
				return NULL;
			}
		}
	} else {
		// Create a tensor with the target data type
		result = nnl2_empty(tensor->shape, tensor->rank, copy_type);
		
		// Data conversion and copying
		switch(copy_type) {
			case FLOAT64: {
				double* cast_data_copy = (double*)result->data;
				
				for(size_t it = 0; it < total_elems; it++) {
					// Getting a pointer to the current element of the source tensor
					void* original_elem = (char*)tensor->data + it * get_dtype_size(dtype);
					
					// Convert and copy the element
					cast_data_copy[it] = nnl2_convert_to_float64(original_elem, dtype);
				}
				
				break;
			}
			
			case FLOAT32: {
				float* cast_data_copy = (float*)result->data;
				
				for(size_t it = 0; it < total_elems; it++) {
					// Getting a pointer to the current element of the source tensor
					void* original_elem = (char*)tensor->data + it * get_dtype_size(dtype);
					
					// Convert and copy the element
					cast_data_copy[it] = nnl2_convert_to_float32(original_elem, dtype);
				}
				
				break;
			}
			
			case INT32: {
				int32_t* cast_data_copy = (int32_t*)result->data;
				
				for(size_t it = 0; it < total_elems; it++) {
					// Getting a pointer to the current element of the source tensor
					void* original_elem = (char*)tensor->data + it * get_dtype_size(dtype);
					
					// Convert and copy the element
					cast_data_copy[it] = nnl2_convert_to_int32(original_elem, dtype);
				}
				
				break;
			}
			
			default: {
				NNL2_TYPE_ERROR(copy_type);
				return NULL;
			}
		}
	}

	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif

	return result;
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for copy operation
 * @details
 * Array follows the common backend registration pattern for copy operations.
 * Currently registered backends:
 *  - nnl2_naive: Basic reference implementation
 * 
 * @see nnl2_naive
 */
Implementation copy_backends[] = {
	REGISTER_BACKEND(naive_copy, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for copy operation
 * @ingroup backend_system 
 */
copyfn nnl2_copy;

/** 
 * @brief Makes the copy backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(copy);

/** 
 * @brief Sets the backend for copy operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see ESET_BACKEND_BY_NAME
 */
void set_copy_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(copy_backends, nnl2_copy, backend_name, current_backend(copy));
}

/** 
 * @brief Gets the name of the active backend for copy operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_copy_backend() {
	return current_backend(copy);
}

/** 
 * @brief Function declaration for getting all available copy backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(copy);

/**
 * @brief Function declaration for getting the number of available copy backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(copy);

#endif /** NNL2_COPY_H **/
