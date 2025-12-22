#ifndef NNL2_SCALE_H
#define NNL2_SCALE_H

/** @brief
 * Multiplies a tensor by a scalar factor
 *
 * Special cases (multiplication by 0 and 1) are handled optimally by creating
 * a tensor with zeros or copying the original tensor
 *
 ** @param tensor
 * Pointer to the input tensor
 *
 ** @param multiplier 
 * A scalar multiplier for multiplying tensor elements
 *
 ** @param save_type 
 * The flag for saving the data type (for example, by default, int32 
 * is cast to float64. save_type attempts to preserve the type if possible)
 *
 ** @see nnl2_copy
 ** @see nnl2_zeros
 ** @see nnl2_empty
 ** @see nnl2_free_tensor
 **/
nnl2_tensor* nnl2_naive_scale(const nnl2_tensor* tensor, float multiplier, bool save_type) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor, "Passed tensor is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor->data, "nnl2_tensor data is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor->shape, "nnl2_tensor shape is NULL", NULL);
	#endif
	
	nnl2_tensor* result;
	void* data_original = tensor->data;
	
	// Calculate the total number of elements in the tensor
	size_t num_elems = nnl2_product(tensor->shape, tensor->rank);
	nnl2_tensor_type dtype = tensor->dtype;
	
	if (multiplier == 1.0f) {
		// For multiplication by 1, just return a copy
		nnl2_tensor_type output_dtype = (dtype == INT32 && !save_type) ? FLOAT64 : dtype;
		return nnl2_copy(tensor, output_dtype);
	}
	
	if (multiplier == 0.0f) {
		// For multiplication by 0, return a tensor of zeros
		nnl2_tensor_type output_dtype = (dtype == INT32 && !save_type) ? FLOAT64 : dtype;
		return nnl2_zeros(tensor->shape, tensor->rank, output_dtype);
	}
	
	if(dtype == INT32) {
		int32_t* data_t = (int32_t*)data_original;
		
		if(save_type) {
			// Scale integer data, round, and keep as INT32
			result = nnl2_empty(tensor->shape, tensor->rank, INT32);
			int32_t* data_o = (int32_t*)result->data;
			for(size_t i = 0; i < num_elems; i++) {
				data_o[i] = (int32_t)lround(data_t[i] * multiplier);
			}
			return result;
		} else {
			// Scale integer data and promote to FLOAT64
			result = nnl2_empty(tensor->shape, tensor->rank, FLOAT64);
			double* data_o = (double*)result->data;
			for(size_t i = 0; i < num_elems; i++) {
				data_o[i] = data_t[i] * multiplier;
			}
		}
		
		return result;
	}
	
	result = nnl2_empty(tensor->shape, tensor->rank, tensor->dtype);
	void* data_result = result->data;
	
	switch(dtype) {
		case FLOAT64: {
			double* data_t = (double*)data_original;
			double* data_o = (double*)data_result;
			for(size_t i = 0; i < num_elems; i++) data_o[i] = data_t[i] * (double)multiplier;
			break;
		}
		
		case FLOAT32: {
			float* data_t = (float*)data_original;
			float* data_o = (float*)data_result;
			for(size_t i = 0; i < num_elems; i++) data_o[i] = data_t[i] * multiplier;
			break;
		}
		
		default: {
			NNL2_TYPE_ERROR(dtype);
			nnl2_free_tensor(result);
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
 * @brief Backend implementations for scale operation
 * @details
 * Array follows the common backend registration pattern for scale operations.
 * Currently registered backends:
 *  - nnl2_naive: Basic reference implementation
 * 
 * @see nnl2_naive
 */
nnl2_runtime_implementation scale_backends[] = {
	REGISTER_BACKEND(nnl2_naive_scale, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for scale operation
 * @ingroup backend_system 
 */
scalefn scale;

/** 
 * @brief Makes the scale backend current
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(scale);

/** 
 * @brief Sets the backend for scale operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see ESET_BACKEND_BY_NAME
 */
void set_scale_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(scale_backends, scale, backend_name, CURRENT_BACKEND(scale));
}

/** 
 * @brief Gets the name of the active backend for scale operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_scale_backend() {
	return CURRENT_BACKEND(scale);
}

/** 
 * @brief Function declaration for getting all available scale backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(scale);

/**
 * @brief Function declaration for getting the number of available scale backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(scale);

#endif /** NNL2_SCALE_H **/
