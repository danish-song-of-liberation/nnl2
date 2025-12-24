#ifndef NNL2_NEG_INPLACE_H
#define NNL2_NEG_INPLACE_H

/** @brief
 * Applies to each element of the tensor: tensor[i] = -(tensor[i]) 
 *
 ** @param tensor
 * Input tensor
 *
 ** @note
 * Works in-place
 *
 ** @see nnl2_product (maybe already nnl2_product)
 ** @see nnl2_tensor
 **/
void nnl2_naive_neg_inplace(nnl2_tensor* tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	// Additional checks at the maximum safety level
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensor, "nnl2_tensor is NULL (.neg!)");
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensor->data, "nnl2_tensor's data is NULL (.neg!)");
	#endif
	
	// Calculating the total number of elements in the base tensor
    size_t numel = nnl2_product(tensor->shape, tensor->rank);
	
	// If the tensor is empty, exit the function
    if(numel == 0) return;
	
	switch(tensor->dtype) {
		case FLOAT64: {
            nnl2_float64* tensor_data = (nnl2_float64*)tensor->data;
                
            // Element-wise neg
            for(size_t i = 0; i < numel; i++) {
                tensor_data[i] = -(tensor_data[i]);
            }
			
            break;
		}
		
		case FLOAT32: {
            nnl2_float32* tensor_data = (nnl2_float32*)tensor->data;
                
            // Element-wise neg
            for(size_t i = 0; i < numel; i++) {
                tensor_data[i] = -(tensor_data[i]);
            }
			
            break;
		}
		
		case INT64: {
			nnl2_int64* tensor_data = (nnl2_int64*)tensor->data;
				
			// Element-wise neg
			for(size_t i = 0; i < numel; i++) {
				tensor_data[i] = -(tensor_data[i]);
			}

			break;
		}
		
		case INT32: {
            nnl2_int32* tensor_data = (nnl2_int32*)tensor->data;
                
            // Element-wise neg
            for(size_t i = 0; i < numel; i++) {
                tensor_data[i] = -(tensor_data[i]);
            }
			
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
 * @brief Backend implementations for in-place neg operation
 * @details
 * Array follows the common backend registration pattern.
 * Currently registered backends:
 *  - nnl2_naive_neg_inplace: Basic reference implementation
 * 
 * @see nnl2_naive_neg_inplace
 */
nnl2_runtime_implementation neginplace_backends[] = {
    REGISTER_BACKEND(nnl2_naive_neg_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for in-place new operation
 * @ingroup backend_system 
 */
neginplacefn nnl2_neginplace;

/** 
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(neginplace);

/** 
 * @brief Sets the backend for in-place neg operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void set_neginplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(neginplace_backends, nnl2_neginplace, backend_name, CURRENT_BACKEND(neginplace));
}

/** 
 * @brief Gets the name of the active backend for in-place neg operation
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* get_neginplace_backend() {
    return CURRENT_BACKEND(neginplace);
}

/** 
 * @brief Function declaration for getting all `neginplace` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(neginplace);

/**
 * @brief Function declaration for getting the number of all `neginplace` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(neginplace);

#endif /** NNL2_NEG_INPLACE_H **/
