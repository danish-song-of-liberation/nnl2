#ifndef NNL2_SIN_INPLACE_H
#define NNL2_SIN_INPLACE_H

#include <math.h>

/** @brief
 * Applies to each element of the tensor: tensor[i] = sin(tensor[i]) 
 *
 ** @param tensor
 * Input tensor
 *
 ** @note
 * Works in-place
 *
 ** @see product (maybe already nnl2_product)
 ** @see nnl2_tensor
 **/
void nnl2_naive_sin_inplace(nnl2_tensor* tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	// Additional checks at the maximum safety level
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensor, "Tensor is NULL (.sin!)");
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensor->data, "Tensor's data is NULL (.sin!)");
	#endif
	
	// Calculating the total number of elements in the base tensor
    size_t numel = product(tensor->shape, tensor->rank);
	
	// If the tensor is empty, exit the function
    if(numel == 0) return;
	
	switch(tensor->dtype) {
		case FLOAT64: {
            nnl2_float64* tensor_data = (nnl2_float64*)tensor->data;
                
            // Element-wise sin
            for(size_t i = 0; i < numel; i++) {
                tensor_data[i] = sin(tensor_data[i]);
            }
			
            break;
		}
		
		case FLOAT32: {
            nnl2_float32* tensor_data = (nnl2_float32*)tensor->data;
                
            // Element-wise sin
            for(size_t i = 0; i < numel; i++) {
                tensor_data[i] = sinf(tensor_data[i]);
            }
			
            break;
		}
		
		case INT32: {
            nnl2_int32* tensor_data = (nnl2_int32*)tensor->data;
                
            // Element-wise sin
            for(size_t i = 0; i < numel; i++) {
                tensor_data[i] = (nnl2_int32)sin((double)tensor_data[i]);
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
 * @brief Backend implementations for in-place sin operation
 * @details
 * Array follows the common backend registration pattern.
 * Currently registered backends:
 *  - nnl2_naive_sin_inplace: Basic reference implementation
 * 
 * @see nnl2_naive_sin_inplace
 */
Implementation sininplace_backends[] = {
    REGISTER_BACKEND(nnl2_naive_sin_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for in-place sin operation
 * @ingroup backend_system 
 */
sininplacefn nnl2_sininplace;

/** 
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(sininplace);

/** 
 * @brief Sets the backend for in-place sin operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void set_sininplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(sininplace_backends, nnl2_sininplace, backend_name, CURRENT_BACKEND(sininplace));
}

/** 
 * @brief Gets the name of the active backend for in-place sin operation
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* get_sininplace_backend() {
    return CURRENT_BACKEND(sininplace);
}

/** 
 * @brief Function declaration for getting all `sininplace` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(sininplace);

/**
 * @brief Function declaration for getting the number of all `sininplace` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(sininplace);

#endif /** NNL2_SIN_INPLACE_H **/
