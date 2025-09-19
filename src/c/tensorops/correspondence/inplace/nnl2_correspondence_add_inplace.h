#ifndef NNL2_CORRESPONDENCE_ADD_INPLACE_H
#define NNL2_CORRESPONDENCE_ADD_INPLACE_H

/** @brief 
 * Adds a scalar value to each element of a tensor (in-place).
 * 
 ** @param tensor 
 * Pointer to the tensor to which the value will be added
 * 
 ** @param inc 
 * Pointer to the scalar value to add
 */
void naive_add_incf_inplace(Tensor* tensor, void* inc) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	size_t total_elems = product(tensor->shape, tensor->rank);
	if(total_elems == 0) return; 
	
	switch(tensor->dtype) {
		case FLOAT64: {
			double* cast_data = (double*)tensor->data;
			double increment = *((double*)inc);
			for(size_t i = 0; i < total_elems; i++) cast_data[i] += increment;
			break;
		}
		
		case FLOAT32: {
			float* cast_data = (float*)tensor->data;
			float increment = *((float*)inc);
			for(size_t i = 0; i < total_elems; i++) cast_data[i] += increment;
			break;
		}
		
		case INT32: {
			int32_t* cast_data = (int32_t*)tensor->data;
			int32_t increment = *((int32_t*)inc);
			for(size_t i = 0; i < total_elems; i++) cast_data[i] += increment;
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
 * @brief Backend implementations for in-place scalar addition operation
 * @details
 * Array follows the common backend registration pattern for in-place scalar addition
 * operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for scalar addition
 * 
 * @see nnl2_naive
 * @see naive_add_incf_inplace
 */
Implementation add_incf_inplace_backends[] = {
	REGISTER_BACKEND(naive_add_incf_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for in-place scalar addition operation
 * @ingroup backend_system 
 */
addincfinplacefn add_incf_inplace;

/** 
 * @brief Sets the backend for in-place scalar addition operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for scalar addition
 * @see ESET_BACKEND_BY_NAME
 */
void set_add_incf_inplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(add_incf_inplace_backends, add_incf_inplace, backend_name);
}

#endif /** NNL2_CORRESPONDENCE_ADD_INPLACE_H **/
