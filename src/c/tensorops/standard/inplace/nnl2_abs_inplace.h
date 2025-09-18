#ifndef NNL2_ABS_INPLACE_H
#define NNL2_ABS_INPLACE_H

/** @brief 
 * Calculates the absolute values of the tensor elements in place
 *
 ** @param tensor
 * A pointer to a tensor that will be modified
 *
 ** @see fabs
 ** @see fabsf
 ** @see abs
 **/
void naive_absinplace(Tensor* tensor) {	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif

	size_t total_elems = product(tensor->shape, tensor->rank);
	if(total_elems == 0) return; // If tensor is empty then return
	
	void* data = tensor->data;
	
	switch(tensor->dtype) {
		case FLOAT64: {
			double* cast_data = (double*)data;	
			for(size_t i = 0; i < total_elems; i++) cast_data[i] = fabs(cast_data[i]);
			break;
		}
		
		case FLOAT32: {
			float* cast_data = (float*)data;	
			for(size_t i = 0; i < total_elems; i++) cast_data[i] = fabsf(cast_data[i]);
			break;
		}
		
		case INT32: {
			int32_t* cast_data = (int32_t*)data;	
			for(size_t i = 0; i < total_elems; i++) cast_data[i] = abs(cast_data[i]);
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
 * @brief Backend implementations for absinplace operation
 * @details
 * Array follows the common backend registration pattern for in-place absolute value operations.
 * Currently registered backends:
 *  - nnl2_naive: Basic reference implementation
 * 
 * @see nnl2_naive
 */
Implementation absinplace_backends[] = {
	REGISTER_BACKEND(naive_absinplace, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for absinplace operation
 * @ingroup backend_system 
 */
absinplacefn absinplace;

/** 
 * @brief Makes the absinplace backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(absinplace);

/** 
 * @brief Sets the backend for absinplace operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see ESET_BACKEND_BY_NAME
 */
void set_absinplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(absinplace_backends, absinplace, backend_name, current_backend(absinplace));
}

/** 
 * @brief Gets the name of the active backend for absinplace operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_absinplace_backend() {
	return current_backend(absinplace);
}

/** 
 * @brief Function declaration for getting all available absinplace backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(absinplace);

/**
 * @brief Function declaration for getting the number of available absinplace backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(absinplace);

#endif /** NNL2_ABS_INPLACE_H **/
