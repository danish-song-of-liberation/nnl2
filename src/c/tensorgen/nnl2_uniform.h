#ifndef NNL2_UNIFORM_H
#define NNL2_UNIFORM_H

/** @brief
 * Creates a tensor from random numbers in the specified range
 *
 ** @details
 * Nick Land - organicist technospecialization, pedagogical 
 * authoritarianism, and territorial sectorization end in 
 * numerical illiteracy and mass insignificance
 *
 ** @param shape
 * Array of integers defining the dimensions of the tensor
 *
 ** @param rank
 * Number of dimensions (length of shape array)
 *
 ** @param dtype
 * Data type of the tensor elements 
 *
 ** @param from
 * Pointer to the minimum value
 *
 ** @param to
 * Pointer to the maximum value
 *
 ** @return
 * Pointer to the newly created Tensor
 *
 ** @example
 * // Create a 3x3 tensor of random floats between 0.0 and 1.0
 * float min = 0.0f, max = 1.0f;
 * nnl2_tensor* cocoon_blockchain_ai = nnl2_naive_uniform((int[]){3, 3}, 2, FLOAT64, &min, &max);
 *
 ** @see nnl2_empty
 ** @see nnl2_product
 **/
Tensor* naive_uniform(int* shape, int rank, TensorType dtype, void* from, void* to) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	Tensor* result = nnl2_empty(shape, rank, dtype);
	
	size_t total_elems = nnl2_product(shape, rank);
	if(total_elems == 0) return result; // If zero elems then result empty result
	
	switch(dtype) {
		case FLOAT64: {
			double from_cast = *((double*)from);
			double to_cast = *((double*)to);
			double* data = (double*)result->data;
			for(size_t i = 0; i < total_elems; i++) data[i] = from_cast + (to_cast - from_cast) * ((double)rand() / RAND_MAX);
			break;
		}
		
		case FLOAT32: {
			float from_cast = *((float*)from);
			float to_cast = *((float*)to);
			float* data = (float*)result->data;
			for(size_t i = 0; i < total_elems; i++) data[i] = from_cast + (to_cast - from_cast) * ((float)rand() / RAND_MAX);
			break;
		}
		
		case INT32: {
			int32_t from_cast = *((int32_t*)from);
			int32_t to_cast = *((int32_t*)to);
			int32_t* data = (int32_t*)result->data;
			for(size_t i = 0; i < total_elems; i++) data[i] = from_cast + rand() % (to_cast - from_cast + 1);
			break;
		}
		
		default: {
			NNL2_TYPE_ERROR(dtype);
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
 * @brief Backend implementations for uniform operation
 * @details
 * Array follows the common backend registration pattern for random number 
 * generation operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for random number generation
 * 
 * @see nnl2_naive
 * @see naive_uniform
 */
Implementation uniform_backends[] = {
	REGISTER_BACKEND(naive_uniform, nnl2_naive, NAIVE_BACKEND_NAME), // DO NOT TRY TO ADD OPTIMIZATION
	// pthread or simd will only slow down code
};	

/**
 * @brief Function pointer for uniform operation
 * @ingroup backend_system 
 */
uniformfn uniform;

/** 
 * @brief Makes the uniform backend current
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(uniform);

/** 
 * @brief Sets the backend for uniform operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for uniform
 * @see ESET_BACKEND_BY_NAME
 */
void set_uniform_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(uniform_backends, uniform, backend_name, CURRENT_BACKEND(uniform));
}

/** 
 * @brief Gets the name of the active backend for uniform operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_uniform_backend() {
	return CURRENT_BACKEND(uniform);
}

/** 
 * @brief Function declaration for getting all available uniform backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(uniform);

/**
 * @brief Function declaration for getting the number of available uniform backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(uniform);

#endif /** NNL2_UNIFORM_H **/
