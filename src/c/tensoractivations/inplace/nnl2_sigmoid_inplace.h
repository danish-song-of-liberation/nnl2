#ifndef NNL2_SIGMOID_INPLACE_H
#define NNL2_SIGMOID_INPLACE_H

/** @brief
 * Applies a sigmoid function to a tensor in place
 *
 ** @details
 * Sigmoid function (I write from memory): sigmoid(x) = 1/exp(-x)
 *
 * Example 1: sigmoid(1) = 1/exp(-1) = 0.731
 * Example 2: sigmoid(0.5) = 1/exp(-0.5) = 0.622
 *
 * oh, look, I think I've miscalculated the formula. Okay. Here's the correct formula:
 * sigmoid(x) = 1 / (1 + exp(-x))
 *
 ** @param tensor
 * A pointer to a tensor for conversion
 *
 * This @param is very useful, and without its documentation, 
 * you probably wouldn't know what it does or why it's needed
 *
 ** @see nnl2_product
 **/
void naive_sigmoidinplace(Tensor* tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Calculate total number of elements
	int total_elems = product(tensor->shape, tensor->rank);	
	if(total_elems == 0) return; // If 0 elems then return
	
	void* data = tensor->data;
	
	switch(tensor->dtype) {
		case FLOAT64: {
			double* cast_data = (double*)data;	
			for(int i = 0; i < total_elems; i++) nnl2_sigmoid_float64_inplace(&cast_data[i]); // Maximum efficiency
			break;
		}
		
		case FLOAT32: {
			float* cast_data = (float*)data;	
			for(int i = 0; i < total_elems; i++) nnl2_sigmoid_float32_inplace(&cast_data[i]); // Maximum efficiency
			break;
		}
		
		case INT32: {
			NNL2_FATAL("Sigmoid in-place cannot be applied to the provided tensor");
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
 * @brief Backend implementations for sigmoidinplace operation
 * @details
 * Array follows the common backend registration pattern for sigmoidinplace operations.
 * Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for in-place sigmoid activation function
 * 
 * @see nnl2_naive
 * @see naive_sigmoidinplace
 */
Implementation sigmoidinplace_backends[] = {
	REGISTER_BACKEND(naive_sigmoidinplace, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for sigmoidinplace operation
 * @ingroup backend_system 
 */
sigmoidinplacefn sigmoidinplace;

/** 
 * @brief Makes the sigmoidinplace backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(sigmoidinplace);

/** 
 * @brief Sets the backend for sigmoidinplace operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for sigmoidinplace
 * @see ESET_BACKEND_BY_NAME
 */
void set_sigmoidinplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(sigmoidinplace_backends, sigmoidinplace, backend_name, current_backend(sigmoidinplace));
}

/** 
 * @brief Gets the name of the active backend for sigmoidinplace operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_sigmoidinplace_backend() {
	return current_backend(sigmoidinplace);
}

/** 
 * @brief Function declaration for getting all available sigmoidinplace backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(sigmoidinplace);

/**
 * @brief Function declaration for getting the number of available sigmoidinplace backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(sigmoidinplace);

#endif /** NNL2_SIGMOID_INPLACE_H **/
