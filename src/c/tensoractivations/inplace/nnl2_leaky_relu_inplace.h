#ifndef NNL2_LEAKY_RELU_INPLACE_H
#define NNL2_LEAKY_RELU_INPLACE_H

/** @brief
 * Applies Leaky ReLU (max(alpha * x, x)) function to an in-place tensor
 *
 * Example 1: leakyrelu(1) -> 1
 * Example 2: leakyrelu(-1, alpha=0.01) -> -0.01
 *	
 ** @param tensor
 * A pointer to a tensor for modification]
 *
 ** @param alpha
 * Slope coefficient for negative values (usually a small positive number)
 */
void naive_leakyreluinplace(Tensor* tensor, float alpha) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	int total_elems = product(tensor->shape, tensor->rank);	
	if(total_elems == 0) return; // If tensor is empty return
	
	void* data = tensor->data;
	
	switch(tensor->dtype) {
		case FLOAT64: {
			double* cast_data = (double*)data;	
			for(int i = 0; i < total_elems; i++) nnl2_leaky_relu_float64_inplace(&cast_data[i], alpha);
			break;
		}
		
		case FLOAT32: {
			float* cast_data = (float*)data;	
			for(int i = 0; i < total_elems; i++) nnl2_leaky_relu_float32_inplace(&cast_data[i], alpha);
			break;
		}
		
		case INT32: {
			int32_t* cast_data = (int32_t*)data;	
			for(int i = 0; i < total_elems; i++) nnl2_leaky_relu_int32_inplace(&cast_data[i], alpha);
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
 * @brief Backend implementations for LeakyReLU in-place operation
 * @details
 * Array follows the common backend registration pattern for LeakyReLU in-place operations.
 * Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for LeakyReLU activation function
 * 
 * @see nnl2_naive
 * @see naive_leakyreluinplace
 */
Implementation leakyreluinplace_backends[] = {
	REGISTER_BACKEND(naive_leakyreluinplace, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for LeakyReLU in-place operation
 * @ingroup backend_system 
 */
leakyreluinplacefn leakyreluinplace;

/** 
 * @brief Makes the leakyreluinplace backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(leakyreluinplace);

/** 
 * @brief Sets the backend for LeakyReLU in-place operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for LeakyReLU in-place
 * @see ESET_BACKEND_BY_NAME
 */
void set_leakyreluinplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(leakyreluinplace_backends, leakyreluinplace, backend_name, current_backend(leakyreluinplace));
}

/** 
 * @brief Gets the name of the active backend for LeakyReLU in-place operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_leakyreluinplace_backend() {
	return current_backend(leakyreluinplace);
}

/** 
 * @brief Function declaration for getting all available LeakyReLU in-place backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(leakyreluinplace);

/**
 * @brief Function declaration for getting the number of available LeakyReLU in-place backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(leakyreluinplace);

#endif /** NNL2_LEAKY_RELU_INPLACE_H **/
