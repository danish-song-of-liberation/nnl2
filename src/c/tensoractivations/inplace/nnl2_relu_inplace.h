#ifndef NNL2_RELU_INPLACE_H
#define NNL2_RELU_INPLACE_H

/** @brief
 * Applies ReLU (ReLU(x) = max(x, 0)) activation function in-place to a tensor (naive implementation)
 *
 ** @param tensor
 * Pointer to the input tensor that will be modified in-place
 *
 ** @return
 * None (void function)
 *
 ** @see nnl2_relu_float64_inplace
 ** @see nnl2_relu_float32_inplace
 ** @see nnl2_relu_int32_inplace
 ** @see product
 **/
void naive_reluinplace(Tensor* tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	int total_elems = product(tensor->shape, tensor->rank);	
	if(total_elems == 0) return; // If tensor is empty return
	
	void* data = tensor->data;
	
	switch(tensor->dtype) {
		case FLOAT64: {
			double* cast_data = (double*)data;	
			for(int i = 0; i < total_elems; i++) nnl2_relu_float64_inplace(&cast_data[i]);
			break;
		}
		
		case FLOAT32: {
			float* cast_data = (float*)data;	
			for(int i = 0; i < total_elems; i++) nnl2_relu_float32_inplace(&cast_data[i]);
			break;
		}
		
		case INT32: {
			int32_t* cast_data = (int32_t*)data;	
			for(int i = 0; i < total_elems; i++) nnl2_relu_int32_inplace(&cast_data[i]);
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
 * @brief Backend implementations for in-place ReLU operation
 * @details
 * Array follows the common backend registration pattern for in-place ReLU operations.
 * Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for in-place ReLU activation
 * 
 * @see nnl2_naive
 * @see naive_reluinplace
 */
Implementation reluinplace_backends[] = {
	REGISTER_BACKEND(naive_reluinplace, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for in-place ReLU operation
 * @ingroup backend_system 
 */
reluinplacefn reluinplace;

/** 
 * @brief Makes the reluinplace backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(reluinplace);

/** 
 * @brief Sets the backend for in-place ReLU operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for in-place ReLU
 * @see ESET_BACKEND_BY_NAME
 */
void set_reluinplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(reluinplace_backends, reluinplace, backend_name, current_backend(reluinplace));
}

/** 
 * @brief Gets the name of the active backend for in-place ReLU operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_reluinplace_backend() {
	return current_backend(reluinplace);
}

/** 
 * @brief Function declaration for getting all available in-place ReLU backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(reluinplace);

/**
 * @brief Function declaration for getting the number of available in-place ReLU backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(reluinplace);

#endif /** NNL2_RELU_INPLACE_H **/
