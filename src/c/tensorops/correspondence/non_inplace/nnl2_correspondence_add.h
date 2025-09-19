#ifndef NNL2_CORRESPONDENCE_ADD_H
#define NNL2_CORRESPONDENCE_ADD_H

/** @brief
 * Performs element-wise addition of a scalar increment to a tensor
 *
 ** @param tensor
 * Pointer to the input tensor to which the increment will be added
 *
 ** @param inc
 * Pointer to the scalar increment value
 *
 ** @return
 * Pointer to a new tensor containing the result of the addition operation 
 * (or NULL in case of fail)
 */
Tensor* naive_add_incf(Tensor* tensor, void* inc) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	Tensor* result = nnl2_empty(tensor->shape, tensor->rank, tensor->dtype);
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(result == NULL) {
			NNL2_ERROR("Failed to allocate new tensor");
		}
	#endif
	
	size_t total_elems = product(tensor->shape, tensor->rank);
	if(total_elems == 0) return result;
	
	switch(tensor->dtype) {
		case FLOAT64: {
			double* cast_data_original = (double*)tensor->data; 
			double* cast_data_result = (double*)result->data;    // Casting
			double increment = *((double*)inc);
			for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = cast_data_original[i] + increment; // Assigment
			break;
		}
		
		case FLOAT32: {
			float* cast_data_original = (float*)tensor->data;
			float* cast_data_result = (float*)result->data;
			float increment = *((float*)inc);
			for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = cast_data_original[i] + increment;
			break;
		}
		
		case INT32: {
			int32_t* cast_data_original = (int32_t*)tensor->data;
			int32_t* cast_data_result = (int32_t*)result->data;
			int32_t increment = *((int32_t*)inc);
			for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = cast_data_original[i] + increment;
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
 * @brief Backend implementations for scalar addition with increment operation
 * @details
 * Array follows the common backend registration pattern for scalar addition with
 * increment operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for scalar addition with increment
 * 
 * @see nnl2_naive
 * @see naive_add_incf
 */
Implementation add_incf_backends[] = {
    REGISTER_BACKEND(naive_add_incf, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for scalar addition with increment operation
 * @ingroup backend_system
 */
addincffn add_incf;

/** 
 * @brief Sets the backend for scalar addition with increment operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for scalar addition with increment
 * @see SET_BACKEND_BY_NAME
 * @see add_incf_backends
 */
void set_add_incf_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(add_incf_backends, add_incf, backend_name);
}

#endif /** NNL2_CORRESPONDENCE_ADD_H **/
