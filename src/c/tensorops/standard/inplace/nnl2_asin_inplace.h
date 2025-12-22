#ifndef NNL2_ASIN_INPLACE_H
#define NNL2_ASIN_INPLACE_H

#include <math.h>

/** @brief
 * Applies to each element of the tensor: tensor[i] = asin(tensor[i]) 
 *
 ** @param tensor
 * Input tensor
 *
 ** @note
 * Works in-place
 * Input values must be in the range [-1, 1] for real results
 *
 ** @see nnl2_product (maybe already nnl2_product)
 ** @see nnl2_tensor
 **/
void nnl2_naive_asin_inplace(nnl2_tensor* tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	// Additional checks at the maximum safety level
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensor, "nnl2_tensor is NULL (.asin!)");
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensor->data, "nnl2_tensor's data is NULL (.asin!)");
	#endif
	
	// Calculating the total number of elements in the base tensor
    size_t numel = nnl2_product(tensor->shape, tensor->rank);
	
	// If the tensor is empty, exit the function
    if(numel == 0) return;
	
	switch(tensor->dtype) {
		case FLOAT64: {
            nnl2_float64* tensor_data = (nnl2_float64*)tensor->data;
                
            // Element-wise asin
            for(size_t i = 0; i < numel; i++) {
                tensor_data[i] = asin(tensor_data[i]);
            }
			
            break;
		}
		
		case FLOAT32: {
            nnl2_float32* tensor_data = (nnl2_float32*)tensor->data;
                
            // Element-wise asin
            for(size_t i = 0; i < numel; i++) {
                tensor_data[i] = asinf(tensor_data[i]);
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
 * @brief Backend implementations for in-place asin operation
 * @details
 * Array follows the common backend registration pattern.
 * Currently registered backends:
 *  - nnl2_naive_asin_inplace: Basic reference implementation
 * 
 * @see nnl2_naive_asin_inplace
 */
nnl2_runtime_implementation asininplace_backends[] = {
    REGISTER_BACKEND(nnl2_naive_asin_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for in-place asin operation
 * @ingroup backend_system 
 */
asininplacefn nnl2_asininplace;

/** 
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(asininplace);

/** 
 * @brief Sets the backend for in-place asin operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void set_asininplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(asininplace_backends, nnl2_asininplace, backend_name, CURRENT_BACKEND(asininplace));
}

/** 
 * @brief Gets the name of the active backend for in-place asin operation
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* get_asininplace_backend() {
    return CURRENT_BACKEND(asininplace);
}

/** 
 * @brief Function declaration for getting all `asininplace` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(asininplace);

/**
 * @brief Function declaration for getting the number of all `asininplace` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(asininplace);

#endif /** NNL2_ASIN_INPLACE_H **/
