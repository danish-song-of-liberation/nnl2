#ifndef NNL2_SIGMOID_H
#define NNL2_SIGMOID_H

/** @brief
 * Calculates the sigmoid function for each tensor element
 *
 ** @details
 * Sigmoid function: sigmoid(x) = sigmoid(x) = 1 / (1 + exp(-x))
 *
 * Example 1: sigmoid(1) = 1 / (1 + exp(-1)) = 0.731
 * Example 2: sigmoid(0.5) = 1 / (1 + exp(-0.5)) = 0.622
 *
 ** @note
 * int32 Is automatically converted to float64
 *
 ** @return
 * Pointer to a new tensor with the result of applying a sigmoid function
 *
 ** @see nnl2_sigmoid_float64
 ** @see nnl2_sigmoid_float32
 ** @see nnl2_empty
 **/
Tensor* naive_sigmoid(Tensor* tensor) {	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	size_t total_elems = product(tensor->shape, tensor->rank);	
	TensorType dtype = tensor->dtype;
	
	if(dtype == INT32) dtype = FLOAT64;
	
	Tensor* result = nnl2_empty(tensor->shape, tensor->rank, dtype);
	
	// Ultra mega super optimization
	if(total_elems == 0) return result; // Increases speed by 100-150% (If tensor is empty then return empty result)
	
	void* data_t = tensor->data;
	void* data_r = result->data;
	
	switch(tensor->dtype) {
		case INT32: {
			int32_t* cast_data_t = (int32_t*)data_t;
			double* cast_data_r = (double*)data_r;
			
			// For int32 return float64 tensor
			for(size_t i = 0; i < total_elems; i++) cast_data_r[i] = nnl2_sigmoid_float64((double)cast_data_t[i]); 
			break;
		}
		
		case FLOAT64: {
			double* cast_data_t = (double*)data_t;	
			double* cast_data_r = (double*)data_r;
			for(size_t i = 0; i < total_elems; i++) cast_data_r[i] = nnl2_sigmoid_float64(cast_data_t[i]);
			break;
		}
		
		case FLOAT32: {
			float* cast_data_t = (float*)data_t;	
			float* cast_data_r = (float*)data_r;
			for(size_t i = 0; i < total_elems; i++) cast_data_r[i] = nnl2_sigmoid_float32(cast_data_t[i]);
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
 * @brief Backend implementations for sigmoid operation
 * @details
 * Array follows the common backend registration pattern for sigmoid operations.
 * Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for sigmoid activation function
 * 
 * @see nnl2_naive
 * @see naive_sigmoid
 */
Implementation sigmoid_backends[] = {
	REGISTER_BACKEND(naive_sigmoid, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for sigmoid operation
 * @ingroup backend_system 
 */
sigmoidfn sigmoid;

/** 
 * @brief Makes the sigmoid backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(sigmoid);

/** 
 * @brief Sets the backend for sigmoid operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for sigmoid
 * @see ESET_BACKEND_BY_NAME
 */
void set_sigmoid_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(sigmoid_backends, sigmoid, backend_name, current_backend(sigmoid));
}

/** 
 * @brief Gets the name of the active backend for sigmoid operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_sigmoid_backend() {
	return current_backend(sigmoid);
}

/** 
 * @brief Function declaration for getting all available sigmoid backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(sigmoid);

/**
 * @brief Function declaration for getting the number of available sigmoid backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(sigmoid);

#endif /** NNL2_SIGMOID_H **/
