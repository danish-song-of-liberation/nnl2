#ifndef NNL2_XAVIER_H
#define NNL2_XAVIER_H

#define NNL2_XAVIER_NO_GAIN 1.0f

#define NNL2_XAVIER_NORMAL_DIST 2.0f
#define NNL2_XAVIER_UNIFORM_DIST 6.0f

/** @brief
 * Initializing a tensor using the Xavier distribution
 *
 * Standard deviation is calculated as: gain * sqrt(distribution / (in + out))
 *
 ** @param shape
 * Pointer to an array of tensor dimensions
 *
 ** @param rank
 * The rank of the tensor
 *
 ** @param dtype
 * Tensor data type
 *
 ** @param in
 * Number of input neurons
 *
 ** @param out
 * Number of output neurons
 *
 ** @param gain
 * Gain factor
 *
 ** @param distribution
 * Distribution parameter (usually 2.0 or 6.0)
 *
 ** @note
 * Integer data types are not supported
 *
 ** @see RAND_MAX
 ** @see nnl2_empty
 **/
Tensor* naive_xavier(int* shape, int rank, TensorType dtype, int in, int out, float gain, float distribution) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	if(dtype == INT32) {
		NNL2_FATAL("INT32 Can't be used for xavier distribution");
		return NULL;
	}
	
	Tensor* result = nnl2_empty(shape, rank, dtype);
	
	size_t total_elems = nnl2_product(shape, rank);
	if(total_elems == 0) return result; // If tensor is empty return empty result

	float scale_param = gain * sqrtf(distribution / (in + out));
	
	if(fabsf(distribution - 6.0f) < 1e-6f) {
		double from = (double)-scale_param;
		double to = (double)scale_param;
		
        uniform_inplace(result, &from, &to);
		
    } else {
        randn_inplace(result, 0.0, (double)scale_param);
    }
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
	
	return result;
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for xavier operation
 * @details
 * Array follows the common backend registration pattern for xavier initialization
 * operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for xavier initialization
 * 
 * @see nnl2_naive
 * @see naive_xavier
 */
Implementation xavier_backends[] = {
	REGISTER_BACKEND(naive_xavier, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for xavier operation
 * @ingroup backend_system 
 */
xavierfn xavier;

/** 
 * @brief Makes the xavier backend current
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(xavier);

/** 
 * @brief Sets the backend for xavier operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for xavier
 * @see ESET_BACKEND_BY_NAME
 */
void set_xavier_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(xavier_backends, xavier, backend_name, CURRENT_BACKEND(xavier));
}

/** 
 * @brief Gets the name of the active backend for xavier operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_xavier_backend() {
	return CURRENT_BACKEND(xavier);
}

/** 
 * @brief Function declaration for getting all available xavier backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(xavier);

/**
 * @brief Function declaration for getting the number of available xavier backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(xavier);

#endif /** NNL2_XAVIER_H **/
