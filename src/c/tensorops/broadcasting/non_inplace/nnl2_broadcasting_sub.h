#ifndef NNL2_BROADCASTING_SUB_H
#define NNL2_BROADCASTING_SUB_H

/** @brief
 * Performs element-wise subtraction with broadcasting support
 *
 ** @param minuend
 * First tensor to subtract from
 *
 ** @param subtrahend
 * Second tensor to subtract
 * 
 ** @return
 * New tensor containing the result of subtraction
 *
 ** @note
 * Contains type conversion
 */
Tensor* naive_sub_broadcasting(Tensor* minuend, Tensor* subtrahend) {	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	// Additional checks for max safety level 
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX	
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(minuend, "Minuend tensor is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(subtrahend, "Subtrahend tensor is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(minuend->shape, "Minuend shape is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(subtrahend->shape, "Subtrahend shape is NULL", NULL);
	#endif
 
	// Calculate the total number of elements in each tensor
	size_t numel_minuend = product(minuend->shape, minuend->rank);
	size_t numel_subtrahend = product(subtrahend->shape, subtrahend->rank);
	
	// Getting the tensor data types
	TensorType minuend_dtype = minuend->dtype;
	TensorType subtrahend_dtype = subtrahend->dtype;
	
	TensorType winner_in_the_type_hierarchy = MAX(minuend_dtype, subtrahend_dtype);
	
	// Сreating a resultant tensor
	Tensor* result = nnl2_empty(minuend->shape, minuend->rank, winner_in_the_type_hierarchy);
	
	// Checking the possibility of broadcasting (numel_minuend must be a multiple of numel_subtrahend)
	if((numel_minuend % numel_subtrahend) == 0) {
		if(minuend_dtype == subtrahend_dtype) {
			// Handling the case where the data types match (more efficiently)
			switch(minuend_dtype) {
				case FLOAT64: {
					double* cast_minuend_data = (double*)minuend->data;
					double* cast_subtrahend_data = (double*)subtrahend->data;
					double* cast_result_data = (double*)result->data;
					
					for(size_t i = 0; i < (numel_minuend / numel_subtrahend); i++) {
						for(size_t j = 0; j < numel_subtrahend; j++) {
							cast_result_data[i * numel_subtrahend + j] = cast_minuend_data[i * numel_subtrahend + j] - cast_subtrahend_data[j];
						}
					}
					
					break;
				}
				
				case FLOAT32: {
					float* cast_minuend_data = (float*)minuend->data;
					float* cast_subtrahend_data = (float*)subtrahend->data;
					float* cast_result_data = (float*)result->data;
					
					for(size_t i = 0; i < (numel_minuend / numel_subtrahend); i++) {
						for(size_t j = 0; j < numel_subtrahend; j++) {
							cast_result_data[i * numel_subtrahend + j] = cast_minuend_data[i * numel_subtrahend + j] - cast_subtrahend_data[j];
						}
					}
					
					break;
				}
				
				case INT32: {
					int32_t* cast_minuend_data = (int32_t*)minuend->data;
					int32_t* cast_subtrahend_data = (int32_t*)subtrahend->data;
					int32_t* cast_result_data = (int32_t*)result->data;
					
					for(size_t i = 0; i < (numel_minuend / numel_subtrahend); i++) {
						for(size_t j = 0; j < numel_subtrahend; j++) {
							cast_result_data[i * numel_subtrahend + j] = cast_minuend_data[i * numel_subtrahend + j] - cast_subtrahend_data[j];
						}
					}
					
					break;
				}
				
				default: {
					NNL2_TYPE_ERROR(minuend_dtype);
					return NULL;
				}
			}
			
			#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
				NNL2_FUNC_EXIT();
			#endif
			
			return result;
		} 
		
		else {
			// Handling a case with different data types (conversion required)
			size_t minuend_step = get_dtype_size(minuend_dtype);
			size_t subtrahend_step = get_dtype_size(subtrahend_dtype);
			
			switch(winner_in_the_type_hierarchy) {
				case FLOAT64: {
					double* cast_data_result = (double*)result->data;
					
					char* cast_minuend_data = (char*)minuend->data;
					char* cast_subtrahend_data =  (char*)subtrahend->data;
					
					for(size_t i = 0; i < (numel_minuend / numel_subtrahend); i++) {
						for(size_t j = 0; j < numel_subtrahend; j++) {
							// Get a pointer to minuend element, subtrahend element and convert its type
							void* elem_minuend = cast_minuend_data + (i * numel_subtrahend + j) * minuend_step;
							void* elem_subtrahend = cast_subtrahend_data + j * subtrahend_step; 
							
							cast_data_result[i * numel_subtrahend + j] = nnl2_convert_to_float64(elem_minuend, minuend_dtype) - nnl2_convert_to_float64(elem_subtrahend, subtrahend_dtype);
						}
					}
					
					break;
				}
				
				case FLOAT32: {
					float* cast_data_result = (float*)result->data;
					
					char* cast_minuend_data = (char*)minuend->data;
					char* cast_subtrahend_data =  (char*)subtrahend->data;
					
					for(size_t i = 0; i < (numel_minuend / numel_subtrahend); i++) {
						for(size_t j = 0; j < numel_subtrahend; j++) {
							void* elem_minuend = cast_minuend_data + (i * numel_subtrahend + j) * minuend_step;
							void* elem_subtrahend =  cast_subtrahend_data + j * subtrahend_step;
							
							cast_data_result[i * numel_subtrahend + j] = nnl2_convert_to_float32(elem_minuend, minuend_dtype) - nnl2_convert_to_float32(elem_subtrahend, subtrahend_dtype);
						}
					}
					
					break;
				}
				
				case INT32: {
					int32_t* cast_data_result = (int32_t*)result->data;
					
					char* cast_minuend_data = (char*)minuend->data;
					char* cast_subtrahend_data =  (char*)subtrahend->data;
					
					for(size_t i = 0; i < (numel_minuend / numel_subtrahend); i++) {					
						for(size_t j = 0; j < numel_subtrahend; j++) {
							void* elem_minuend = cast_minuend_data + (i * numel_subtrahend + j) * minuend_step;
							void* elem_subtrahend =  cast_subtrahend_data + j * subtrahend_step;
						
							cast_data_result[i * numel_subtrahend + j] = nnl2_convert_to_int32(elem_minuend, minuend_dtype) - nnl2_convert_to_int32(elem_subtrahend, subtrahend_dtype);
						}
					}
					
					break;
				}
				
				default: {
					NNL2_TYPE_ERROR(winner_in_the_type_hierarchy);
					nnl2_free_tensor(result);
					return NULL;
				}
			}
			
			#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
				NNL2_FUNC_EXIT();
			#endif
			
			return result;
		}
	} 
	
	else {
		NNL2_ERROR("Cannot broadcast subtrahend tensor");
		return NULL;
	}
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for subtraction with broadcasting
 */
Implementation sub_broadcasting_backends[] = {
    REGISTER_BACKEND(naive_sub_broadcasting, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for subtraction with broadcasting operation
 * @ingroup backend_system
 */
subbroadcastingfn sub_broadcasting;

/**
 * @brief Sets the backend for subtraction with broadcasting operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for subtraction with broadcasting
 */
void set_sub_broadcasting_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(sub_broadcasting_backends, sub_broadcasting, backend_name);
}

#endif /** NNL2_BROADCASTING_SUB_H **/
