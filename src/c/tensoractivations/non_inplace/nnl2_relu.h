#ifndef NNL2_RELU_H
#define NNL2_RELU_H

/** @brief
 * Applies ReLU (ReLU(x) = max(x, 0)) activation function to a tensor, returning a new tensor (naive implementation)
 *
 ** @param tensor
 * Pointer to the input tensor
 *
 ** @return
 * Pointer to a new tensor containing the ReLU-activated values
 * Returns NULL in case of failure
 *
 ** @see nnl2_relu_float64
 ** @see nnl2_relu_float32
 ** @see nnl2_relu_int32
 ** @see nnl2_empty
 ** @see product
 **/
Tensor* naive_relu(Tensor* tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		if(tensor == NULL) {
			NNL2_ERROR("Passed tensor is NULL");
		}
	#endif

	int total_elems = product(tensor->shape, tensor->rank);	
	Tensor* result = nnl2_empty(tensor->shape, tensor->rank, tensor->dtype);
	
	if(total_elems == 0) return result; // If tensor is empty return tensor with 0 elements
	
	void* data_t = tensor->data;
	void* data_r = result->data;
	
	switch(tensor->dtype) {
		case FLOAT64: {
			double* cast_data_t = (double*)data_t;	
			double* cast_data_r = (double*)data_r;
			for(int i = 0; i < total_elems; i++) cast_data_r[i] = nnl2_relu_float64(cast_data_t[i]);
			break;
		}
		
		case FLOAT32: {
			float* cast_data_t = (float*)data_t;	
			float* cast_data_r = (float*)data_r;
			for(int i = 0; i < total_elems; i++) cast_data_r[i] = nnl2_relu_float32(cast_data_t[i]);
			break;
		}
		
		case INT32: {
			int32_t* cast_data_t = (int32_t*)data_t;	
			int32_t* cast_data_r = (int32_t*)data_r;
			for(int i = 0; i < total_elems; i++) cast_data_r[i] = nnl2_relu_int32(cast_data_t[i]);
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
 * @brief Backend implementations for ReLU operation
 * @details
 * Array follows the common backend registration pattern for ReLU operations.
 * Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for ReLU activation function
 * 
 * @see nnl2_naive
 * @see naive_relu
 */
Implementation relu_backends[] = {
	REGISTER_BACKEND(naive_relu, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for ReLU operation
 * @ingroup backend_system 
 */
relufn relu;

/** 
 * @brief Makes the relu backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(relu);

/** 
 * @brief Sets the backend for ReLU operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for ReLU
 * @see ESET_BACKEND_BY_NAME
 */
void set_relu_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(relu_backends, relu, backend_name, current_backend(relu));
}

/** 
 * @brief Gets the name of the active backend for ReLU operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_relu_backend() {
	return current_backend(relu);
}

/** 
 * @brief Function declaration for getting all available ReLU backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(relu);

/**
 * @brief Function declaration for getting the number of available ReLU backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(relu);

#endif /** NNL2_RELU_H **/
