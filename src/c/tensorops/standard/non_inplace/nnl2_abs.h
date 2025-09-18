#ifndef NNL2_ABS_H
#define NNL2_ABS_H

/** @brief
 * Naive implementation of absolute value operation
 *
 ** @param tensor 
 * Input tensor
 *
 ** @return 
 * New tensor with absolute values of input elements
 */
Tensor* naive_abs(Tensor* tensor) {	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif

	size_t total_elems = product(tensor->shape, tensor->rank);	
	
	Tensor* result = nnl2_empty(tensor->shape, tensor->rank, tensor->dtype);
	if(total_elems == 0) return result;
	
	void* data_t = tensor->data;
	void* data_r = result->data;
	
	switch(tensor->dtype) {
		case FLOAT64: {
			double* cast_data_t = (double*)data_t;	
			double* cast_data_r = (double*)data_r;
			for(size_t i = 0; i < total_elems; i++) cast_data_r[i] = fabs(cast_data_t[i]);
			break;
		}
		
		case FLOAT32: {
			float* cast_data_t = (float*)data_t;	
			float* cast_data_r = (float*)data_r;
			for(size_t i = 0; i < total_elems; i++) cast_data_r[i] = fabsf(cast_data_t[i]);
			break;
		}
		
		case INT32: {
			int32_t* cast_data_t = (int32_t*)data_t;	
			int32_t* cast_data_r = (int32_t*)data_r;
			for(size_t i = 0; i < total_elems; i++) cast_data_r[i] = abs(cast_data_t[i]);
			break;
		}
		
		default: {
			NNL2_TYPE_ERROR(tensor->dtype);
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
 * @brief Backend implementations for abs operation
 * @details
 * Array follows the common backend registration pattern for absolute value operations.
 * Currently registered backends:
 *  - nnl2_naive: Basic reference implementation
 * 
 * @see nnl2_naive
 */
Implementation abs_backends[] = {
	REGISTER_BACKEND(naive_abs, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for abs operation
 * @ingroup backend_system 
 */
absfn nnl2_abs;

/** 
 * @brief Makes the abs backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(abs);

/** 
 * @brief Sets the backend for abs operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see ESET_BACKEND_BY_NAME
 */
void set_abs_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(abs_backends, nnl2_abs, backend_name, current_backend(abs));
}

/** 
 * @brief Gets the name of the active backend for abs operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_abs_backend() {
	return current_backend(abs);
}

/** 
 * @brief Function declaration for getting all available abs backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(abs);

/**
 * @brief Function declaration for getting the number of available abs backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(abs);

#endif /** NNL2_ABS_H **/
