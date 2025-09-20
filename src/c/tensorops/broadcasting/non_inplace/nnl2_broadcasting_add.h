#ifndef NNL2_BROADCASTING_ADD_H
#define NNL2_BROADCASTING_ADD_H

/** @brief
 * Performs element-wise addition with broadcasting support
 *
 ** @param summand
 * First tensor to add
 *
 ** @param sumend
 * Second tensor to add
 * 
 ** @return
 * New tensor containing the result of addition
 *
 ** @note
 * Contains type conversion
 */
Tensor* naive_add_broadcasting(Tensor* summand, Tensor* sumend) {	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	// Additional checks for max safety level 
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(summand, "Summand tensor is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(sumend, "Sumend tensor is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(summand->shape, "Summand shape is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(sumend->shape, "Sumend shape is NULL", NULL);
	#endif
 
	// Calculate the total number of elements in each tensor
	size_t numel_summand = product(summand->shape, summand->rank);
	size_t numel_sumend = product(sumend->shape, sumend->rank);
	
	// Getting the tensor data types
	TensorType summand_dtype = summand->dtype;
	TensorType sumend_dtype = sumend->dtype;
	
	TensorType winner_in_the_type_hierarchy = MAX(summand_dtype, sumend_dtype);
	
	// Сreating a resultant tensor
	Tensor* result = nnl2_empty(summand->shape, summand->rank, winner_in_the_type_hierarchy);
	
	// Checking the possibility of broadcasting (numel_summand must be a multiple of numel_sumend)
	if((numel_summand % numel_sumend) == 0) {
		if(summand_dtype == sumend_dtype) {
			// Handling the case where the data types match (more efficiently)
			switch(summand_dtype) {
				case FLOAT64: {
					double* cast_summand_data = (double*)summand->data;
					double* cast_sumend_data = (double*)sumend->data;
					double* cast_result_data = (double*)result->data;
					
					for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
						for(size_t j = 0; j < numel_sumend; j++) {
							cast_result_data[i * numel_sumend + j] = cast_summand_data[i * numel_sumend + j] + cast_sumend_data[j];
						}
					}
					
					break;
				}
				
				case FLOAT32: {
					float* cast_summand_data = (float*)summand->data;
					float* cast_sumend_data = (float*)sumend->data;
					float* cast_result_data = (float*)result->data;
					
					for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
						for(size_t j = 0; j < numel_sumend; j++) {
							cast_result_data[i * numel_sumend + j] = cast_summand_data[i * numel_sumend + j] + cast_sumend_data[j];
						}
					}
					
					break;
				}
				
				case INT32: {
					int32_t* cast_summand_data = (int32_t*)summand->data;
					int32_t* cast_sumend_data = (int32_t*)sumend->data;
					int32_t* cast_result_data = (int32_t*)result->data;
					
					for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
						for(size_t j = 0; j < numel_sumend; j++) {
							cast_result_data[i * numel_sumend + j] = cast_summand_data[i * numel_sumend + j] + cast_sumend_data[j];
						}
					}
					
					break;
				}
				
				default: {
					NNL2_TYPE_ERROR(summand_dtype);
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
			size_t summand_step = get_dtype_size(summand_dtype);
			size_t sumend_step = get_dtype_size(sumend_dtype);
			
			switch(winner_in_the_type_hierarchy) {
				case FLOAT64: {
					double* cast_data_result = (double*)result->data;
					
					char* cast_summand_data = (char*)summand->data;
					char* cast_sumend_data =  (char*)sumend->data;
					
					for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
						for(size_t j = 0; j < numel_sumend; j++) {
							// Get a pointer to summand element, sumend element and convert its type
							void* elem_summand = cast_summand_data + (i * numel_sumend + j) * summand_step;
							void* elem_sumend = cast_sumend_data + j * sumend_step; 
							
							cast_data_result[i * numel_sumend + j] = nnl2_convert_to_float64(elem_summand, summand_dtype) + nnl2_convert_to_float64(elem_sumend, sumend_dtype);
						}
					}
					
					break;
				}
				
				case FLOAT32: {
					float* cast_data_result = (float*)result->data;
					
					char* cast_summand_data = (char*)summand->data;
					char* cast_sumend_data =  (char*)sumend->data;
					
					for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
						for(size_t j = 0; j < numel_sumend; j++) {
							void* elem_summand = cast_summand_data + (i * numel_sumend + j) * summand_step;
							void* elem_sumend =  cast_sumend_data + j * sumend_step;
							
							cast_data_result[i * numel_sumend + j] = nnl2_convert_to_float32(elem_summand, summand_dtype) + nnl2_convert_to_float32(elem_sumend, sumend_dtype);
						}
					}
					
					break;
				}
				
				case INT32: {
					int32_t* cast_data_result = (int32_t*)result->data;
					
					char* cast_summand_data = (char*)summand->data;
					char* cast_sumend_data =  (char*)sumend->data;
					
					for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {					
						for(size_t j = 0; j < numel_sumend; j++) {
							void* elem_summand = cast_summand_data + (i * numel_sumend + j) * summand_step;
							void* elem_sumend =  cast_sumend_data + j * sumend_step;
						
							cast_data_result[i * numel_sumend + j] = nnl2_convert_to_int32(elem_summand, summand_dtype) + nnl2_convert_to_int32(elem_sumend, sumend_dtype);
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
		NNL2_ERROR("Cannot broadcast sumend tensor");
		return NULL;
	}
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for addition with broadcasting
 * @details
 * Array follows the common backend registration pattern for addition
 * with broadcasting operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for addition with broadcasting
 * 
 * @see nnl2_naive
 * @see naive_add_broadcasting
 */
Implementation add_broadcasting_backends[] = {
    REGISTER_BACKEND(naive_add_broadcasting, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for addition with broadcasting operation
 * @ingroup backend_system
 */
addbroadcastingfn add_broadcasting;

/**
 * @brief Sets the backend for addition with broadcasting operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for addition with broadcasting
 * @see SET_BACKEND_BY_NAME
 * @see add_broadcasting_backends
 */
void set_add_broadcasting_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(add_broadcasting_backends, add_broadcasting, backend_name);
}

#endif /** NNL2_BROADCASTING_ADD_H **/
