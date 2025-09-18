#ifndef NNL2_TANH_INPLACE_H
#define NNL2_TANH_INPLACE_H


/** @brief
 * Calculates the hyperbolic tangent for all tensor elements in place
 *
 ** @details
 * Look up the hyperbolic tangent formula on the internet
 * I'm too lazy to write it manually
 *
 ** @param tensor
 * A pointer to a tensor for processing
 *
 ** @note
 * Does not work with integer data types
 * 
 ** @see nnl2_product
 ** @see tanh
 ** @see tanhf
 **/
void naive_tanhinplace(Tensor* tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	int total_elems = product(tensor->shape, tensor->rank);	
	if(total_elems == 0) return; // If tensor is empty then return
	
	void* data = tensor->data;
	
	switch(tensor->dtype) {
		case FLOAT64: {
			double* cast_data = (double*)data;	
			for(int i = 0; i < total_elems; i++) cast_data[i] = tanh(cast_data[i]);
			break;
		}
		
		case FLOAT32: {
			float* cast_data = (float*)data;	
			for(int i = 0; i < total_elems; i++) cast_data[i] = tanhf(cast_data[i]);
			break;
		}
		
		case INT32: {
			NNL2_FATAL("Tanh (in-place) cannot be applied to the provided tensor");
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
 * @brief Backend implementations for tanh in-place operation
 * @details
 * Array follows the common backend registration pattern for hyperbolic tangent 
 * in-place operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for tanh activation function
 * 
 * @see nnl2_naive
 * @see naive_tanhinplace
 */
Implementation tanhinplace_backends[] = {
	REGISTER_BACKEND(naive_tanhinplace, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for tanh in-place operation
 * @ingroup backend_system 
 */
tanhinplacefn tanhinplace;

/** 
 * @brief Makes the tanh in-place backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(tanhinplace);

/** 
 * @brief Sets the backend for tanh in-place operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for tanh in-place
 * @see ESET_BACKEND_BY_NAME
 */
void set_tanhinplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(tanhinplace_backends, tanhinplace, backend_name, current_backend(tanhinplace));
}

/** 
 * @brief Gets the name of the active backend for tanh in-place operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_tanhinplace_backend() {
	return current_backend(tanhinplace);
}

/** 
 * @brief Function declaration for getting all available tanh in-place backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(tanhinplace);

/**
 * @brief Function declaration for getting the number of available tanh in-place backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(tanhinplace);

#endif /** NNL2_TANH_INPLACE_H **/
