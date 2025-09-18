#ifndef NNL2_TANH_H
#define NNL2_TANH_H

/** @brief
 * Calculates the hyperbolic tangent (tanh) for each tensor element
 *
 ** @details
 * You can see all the details in nnl2_naive_tanhinplace including the full formula
 *
 ** @param tensor
 * Pointer to the input tensor
 *
 ** @note
 * For integer types, always returns float64
 *
 ** @return
 * Pointer to the resulting tensor
 *
 ** @see nnl2_naive_tanhinplace
 ** @see tanh
 ** @see tanhf
 **/
Tensor* naive_tanh(Tensor* tensor) {	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif

	int total_elems = product(tensor->shape, tensor->rank);
	TensorType dtype = tensor->dtype;
	
	if(dtype == INT32) dtype = FLOAT64;
	
	Tensor* result = nnl2_empty(tensor->shape, tensor->rank, dtype);
	if(total_elems == 0) return result; // If tensor is empty, return empty result
	
	void* data_t = tensor->data;
	void* data_r = result->data;
	
	switch(tensor->dtype) {
		case INT32: {
			int32_t* cast_data_t = (int32_t*)data_t;
			double* cast_data_r = (double*)data_r;
			for(int i = 0; i < total_elems; i++) cast_data_r[i] = tanh((double)cast_data_t[i]); 
			break;
		}
		
		case FLOAT64: {
			double* cast_data_t = (double*)data_t;	
			double* cast_data_r = (double*)data_r;
			for(int i = 0; i < total_elems; i++) cast_data_r[i] = tanh(cast_data_t[i]);
			break;
		}
		
		case FLOAT32: {
			float* cast_data_t = (float*)data_t;	
			float* cast_data_r = (float*)data_r;
			for(int i = 0; i < total_elems; i++) cast_data_r[i] = tanhf(cast_data_t[i]);
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
 * @brief Backend implementations for tanh operation
 * @details
 * Array follows the common backend registration pattern for hyperbolic tangent 
 * operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for tanh activation function
 * 
 * @see nnl2_naive
 * @see naive_tanh
 */
Implementation tanh_backends[] = {
	REGISTER_BACKEND(naive_tanh, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for tanh operation
 * @ingroup backend_system 
 */
tanhfn nnl2_tanh;

/** 
 * @brief Makes the tanh backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(tanh);

/** 
 * @brief Sets the backend for tanh operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for tanh
 * @see ESET_BACKEND_BY_NAME
 */
void set_tanh_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(tanh_backends, nnl2_tanh, backend_name, current_backend(tanh));
}

/** 
 * @brief Gets the name of the active backend for tanh operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_tanh_backend() {
	return current_backend(tanh);
}

/** 
 * @brief Function declaration for getting all available tanh backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(tanh);

/**
 * @brief Function declaration for getting the number of available tanh backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(tanh);

#endif /** NNL2_TANH_H **/
