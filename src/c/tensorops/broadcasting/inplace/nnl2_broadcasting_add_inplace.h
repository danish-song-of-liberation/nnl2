#ifndef NNL2_BROADCASTING_ADD_INPLACE_H
#define NNL2_BROADCASTING_ADD_INPLACE_H

/** @brief
 * Performs element-wise addition with broadcasting (in place)
 *
 ** @details
 * Nick land - Keep the war going. It's pointless.
 *
 ** @param summand
 * Pointer to summand tensor 
 *
 ** @param sumend
 * Pointer to sumend tensor
 */
void naive_add_broadcasting_inplace(Tensor* summand, Tensor* sumend) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	// Additional checks for max safety level 
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(summand, "Summand tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(sumend, "Sumend tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(summand->shape, "Summand shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(sumend->shape, "Sumend shape is NULL");
	#endif
	
	// Calculate the total number of elements in each tensor
	size_t numel_summand = product(summand->shape, summand->rank);
	size_t numel_sumend = product(sumend->shape, sumend->rank);
	
	// Getting the tensor data types
	TensorType summand_dtype = summand->dtype;
	TensorType sumend_dtype = sumend->dtype;
	
	// Checking the possibility of broadcasting (numel_summand must be a multiple of numel_sumend)
	if((numel_summand % numel_sumend) == 0) {
		// Handling the case where the data types match (more efficiently)
		if(summand_dtype == sumend_dtype) {
			switch(summand_dtype) {
				case FLOAT64: {
					double* cast_summand_data = (double*)summand->data;
					double* cast_sumend_data = (double*)sumend->data;
					
					for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
						for(size_t j = 0; j < numel_sumend; j++) {
							cast_summand_data[i * numel_sumend + j] += cast_sumend_data[j];
						}
					}
					
					break;
				}
				
				case FLOAT32: {
					float* cast_summand_data = (float*)summand->data;
					float* cast_sumend_data = (float*)sumend->data;
					
					for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
						for(size_t j = 0; j < numel_sumend; j++) {
							cast_summand_data[i * numel_sumend + j] += cast_sumend_data[j];
						}
					}
					
					break;
				}
				
				case INT32: {
					int32_t* cast_summand_data = (int32_t*)summand->data;
					int32_t* cast_sumend_data = (int32_t*)sumend->data;
					
					for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
						for(size_t j = 0; j < numel_sumend; j++) {
							cast_summand_data[i * numel_sumend + j] += cast_sumend_data[j];
						}
					}
					
					break;
				}
				
				default: {
					NNL2_TYPE_ERROR(summand_dtype);
					return;
				}
			}	
		} else {
			// Handling a case with different data types (conversion required)
			size_t sumend_step = get_dtype_size(sumend_dtype); // The size of the element in bytes
			char* sumend_data = (char*)sumend->data; // Byte pointer for accessing data
			
			switch(summand_dtype) {
				case FLOAT64: {
					double* data_minuend = (double*)summand->data;
                
					for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
						for(size_t j = 0; j < numel_sumend; j++) {
							// Get a pointer to the sumend element and convert its type
							void* sumend_elem = sumend_data + i * sumend_step;
							data_minuend[i * numel_sumend + j] += nnl2_convert_to_float64(sumend_elem, sumend_dtype);
						}
					}
                
					break; 
				}
				
				case FLOAT32: {
					float* data_minuend = (float*)summand->data;
                
					for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
						for(size_t j = 0; j < numel_sumend; j++) {
							void* sumend_elem = sumend_data + i * sumend_step;
							data_minuend[i * numel_sumend + j] += nnl2_convert_to_float32(sumend_elem, sumend_dtype);
						}
					}
                
					break; 
				}
				
				case INT32: {
					int32_t* data_minuend = (int32_t*)summand->data;
                
					for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
						for(size_t j = 0; j < numel_sumend; j++) {
							void* sumend_elem = sumend_data + i * sumend_step;
							data_minuend[i * numel_sumend + j] += nnl2_convert_to_int32(sumend_elem, sumend_dtype);
						}
					}
                
					break; 
				}
				
				default: {
					NNL2_TYPE_ERROR(summand_dtype);
					return;
				}
			}
		}
	} 
	
	else {
		NNL2_ERROR("Cannot broadcast sumend tensor");
		return;
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for in-place addition with broadcasting
 * @details
 * Array follows the common backend registration pattern for in-place addition
 * with broadcasting operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for in-place addition with broadcasting
 * 
 * @see nnl2_naive
 * @see naive_add_broadcasting_inplace
 */
Implementation add_broadcasting_inplace_backends[] = {
    REGISTER_BACKEND(naive_add_broadcasting_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};    

/**
 * @brief Function pointer for in-place addition with broadcasting operation
 * @ingroup backend_system
 */
addbroadcastinginplacefn add_broadcasting_inplace;

/**
 * @brief Sets the backend for in-place addition with broadcasting operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for in-place addition with broadcasting
 * @see SET_BACKEND_BY_NAME
 * @see add_broadcasting_inplace_backends
 */
void set_add_broadcasting_inplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(add_broadcasting_inplace_backends, add_broadcasting_inplace, backend_name);
}

#endif /** NNL2_BROADCASTING_ADD_INPLACE_H **/
