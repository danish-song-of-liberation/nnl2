#ifndef NNL2_FILL_WITH_DATA_H
#define NNL2_FILL_WITH_DATA_H

/** @brief 
 * Fills tensor with data from provided array 
 * 
 ** @param tensor
 * Tensor to fill with data
 *
 ** @param data 
 * Pointer to data array
 *
 ** @param num_elems 
 * Number of elements to copy
 */
inline static void naive_fill_tensor_with_data(Tensor* tensor, void* data, size_t num_elems) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_ENTER();
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor, "Passed tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(data, "Passed data pointer is NULL");
	#endif
	
	if(num_elems == 0) return;
	
	switch(tensor->dtype) {
		case FLOAT64: {
			double* tensor_data = (double*)tensor->data;
			double* cast_data = (double*)data;
			for(size_t it = 0; it < num_elems; it++) tensor_data[it] = cast_data[it];			
			break;
		}
		
		case FLOAT32: {
			float* tensor_data = (float*)tensor->data;
			float* cast_data = (float*)data;
			for(size_t it = 0; it < num_elems; it++) tensor_data[it] = cast_data[it];			
			break;
		}
		
		case INT32: {
			int32_t* tensor_data = (int32_t*)tensor->data;
			int32_t* cast_data = (int32_t*)data;
			for(size_t it = 0; it < num_elems; it++) tensor_data[it] = cast_data[it];			
			break;
		}
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_EXIT();
	#endif
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for filling tensor with data
 */
Implementation fill_tensor_with_data_backends[] = {
    REGISTER_BACKEND(naive_fill_tensor_with_data, nnl2_naive, NAIVE_BACKEND_NAME),
};  

/**
 * @brief Function pointer for filling tensor with data operation
 * @ingroup backend_system
 */
filltensorwithdatafn fill_tensor_with_data;

/**
 * @brief Sets the backend for filling tensor with data operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for filling tensor with data
 */
void set_fill_tensor_with_data_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(fill_tensor_with_data_backends, fill_tensor_with_data, backend_name);
}

#endif /** NNL2_FILL_WITH_DATA_H **/
