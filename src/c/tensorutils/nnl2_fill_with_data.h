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
		case BOOL: {
			nnl2_bool* tensor_data = (nnl2_bool*)tensor->data;
			nnl2_bool* cast_data = (nnl2_bool*)data;
			for(size_t it = 0; it < num_elems; it++) tensor_data[it] = cast_data[it];
			break;
		}
		
		case INT8: {
			nnl2_int8* tensor_data = (nnl2_int8*)tensor->data;
			nnl2_int8* cast_data = (nnl2_int8*)data;
			for(size_t it = 0; it < num_elems; it++) tensor_data[it] = cast_data[it];
			break;
		}
		
		case UINT8: {
			nnl2_uint8* tensor_data = (nnl2_uint8*)tensor->data;
			nnl2_uint8* cast_data = (nnl2_uint8*)data;
			for(size_t it = 0; it < num_elems; it++) tensor_data[it] = cast_data[it];
			break;
		}
		
		case INT16: {
			nnl2_int16* tensor_data = (nnl2_int16*)tensor->data;
			nnl2_int16* cast_data = (nnl2_int16*)data;
			for(size_t it = 0; it < num_elems; it++) tensor_data[it] = cast_data[it];
			break;
		}
		
		case UINT16: {
			nnl2_uint16* tensor_data = (nnl2_uint16*)tensor->data;
			nnl2_uint16* cast_data = (nnl2_uint16*)data;
			for(size_t it = 0; it < num_elems; it++) tensor_data[it] = cast_data[it];
			break;
		}
		
		case INT32: {
			nnl2_int32* tensor_data = (nnl2_int32*)tensor->data;
			nnl2_int32* cast_data = (nnl2_int32*)data;
			for(size_t it = 0; it < num_elems; it++) tensor_data[it] = cast_data[it];
			break;
		}
		
		case UINT32: {
			nnl2_uint32* tensor_data = (nnl2_uint32*)tensor->data;
			nnl2_uint32* cast_data = (nnl2_uint32*)data;
			for(size_t it = 0; it < num_elems; it++) tensor_data[it] = cast_data[it];
			break;
		}
		
		case INT64: {
			nnl2_int64* tensor_data = (nnl2_int64*)tensor->data;
			nnl2_int64* cast_data = (nnl2_int64*)data;
			for(size_t it = 0; it < num_elems; it++) tensor_data[it] = cast_data[it];
			break;
		}
		
		case UINT64: {
			nnl2_uint64* tensor_data = (nnl2_uint64*)tensor->data;
			nnl2_uint64* cast_data = (nnl2_uint64*)data;
			for(size_t it = 0; it < num_elems; it++) tensor_data[it] = cast_data[it];
			break;
		}
		
		case INT128: {
			#if NNL2_INT128_SUPPORTED
				nnl2_int128* tensor_data = (nnl2_int128*)tensor->data;
				nnl2_int128* cast_data = (nnl2_int128*)data;
				for(size_t it = 0; it < num_elems; it++) tensor_data[it] = cast_data[it];
			#else
				NNL2_INT128_FATAL();
			#endif
			
			break;
		}
		
		case UINT128: {
			#if NNL2_UINT128_SUPPORTED
				nnl2_uint128* tensor_data = (nnl2_uint128*)tensor->data;
				nnl2_uint128* cast_data = (nnl2_uint128*)data;
				for(size_t it = 0; it < num_elems; it++) tensor_data[it] = cast_data[it];
			#else
				NNL2_UINT128_FATAL();
			#endif
			
			break;
		}
		
		case FLOAT32: {
			nnl2_float32* tensor_data = (nnl2_float32*)tensor->data;
			nnl2_float32* cast_data = (nnl2_float32*)data;
			for(size_t it = 0; it < num_elems; it++) tensor_data[it] = cast_data[it];
			break;
		}
		
		case FLOAT64: {
			nnl2_float64* tensor_data = (nnl2_float64*)tensor->data;
			nnl2_float64* cast_data = (nnl2_float64*)data;
			for(size_t it = 0; it < num_elems; it++) tensor_data[it] = cast_data[it];
			break;
		}
		
		case FLOAT128: {
			#if NNL2_FLOAT128_SUPPORTED
				nnl2_float128* tensor_data = (nnl2_float128*)tensor->data;
				nnl2_float128* cast_data = (nnl2_float128*)data;
				for(size_t it = 0; it < num_elems; it++) tensor_data[it] = cast_data[it];
			#else
				NNL2_FLOAT128_FATAL();
			#endif
			
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
