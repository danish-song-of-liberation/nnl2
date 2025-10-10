#ifndef NNL2_TRANSPOSITION_INPLACE_H
#define NNL2_TRANSPOSITION_INPLACE_H

/** @brief 
 * Performs pseudo-inplace transposition of a 2D tensor (matrix) in O(1) time
 *
 ** @param tensor
 * Pointer to the 2D tensor to be transposed in-place
 *
 ** @warning
 * MATHEMATICALLY INCORRECT FOR MOST OPERATIONS!
 * This function modifies the tensor's shape and strides to represent
 * a transposed view, but the underlying data remains unchanged in
 * row-major order. Most tensor operations will produce mathematically
 * incorrect results
 *
 ** @warning  
 * This is a DESTRUCTIVE operation that modifies the input tensor.
 * The original shape and strides are lost after this call
 *
 ** @example
 * // Transposes tensor in-place without data copying
 * Tensor* matrix = nnl2_empty((int[]){3, 2}, 2, FLOAT64);
 * nnl2_naive_transposition_inplace(matrix);  // Now shape [2, 3]
 *
 ** @see nnl2_naive_transposition
 **/
void nnl2_naive_transposition_inplace(Tensor* tensor) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensor, "Passed tensor in inplace transposition is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensor->shape, "Passed tensor shape in inplace transposition is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensor->strides, "Passed tensor stride in inplace transposition is NULL");
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(tensor->rank != 2) {
            NNL2_ERROR("Tensor has incorrect rank: %d (expected 2)", tensor->rank);
            return;
        }
    #endif
    
    // Swap shape dimensions in-place
    int32_t temp_shape = tensor->shape[0];
    tensor->shape[0] = tensor->shape[1];
    tensor->shape[1] = temp_shape;
    
    // Swap strides in-place  
    int32_t temp_stride = tensor->strides[0];
    tensor->strides[0] = tensor->strides[1];
    tensor->strides[1] = temp_stride;
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for transposition_inplace (view) operation
 * @details
 * Array follows the common backend registration pattern for transposition_inplace (view)
 * operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for transposition_inplace (view)
 * 
 * @see nnl2_naive
 * @see nnl2_naive_transposition_inplace
 */
Implementation transposition_inplace_backends[] = {
    REGISTER_BACKEND(nnl2_naive_transposition_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for transposition-inplace operation (view)
 * @ingroup backend_system 
 */
transpositioninplacefn nnl2_transposition_inplace;

/** 
 * @brief Makes the transposition-inplace backend current
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(transposition_inplace);

/** 
 * @brief Sets the backend for transposition-inplace operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for transposition-inplace
 * @see ESET_BACKEND_BY_NAME
 */
void set_transposition_inplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(transposition_inplace_backends, nnl2_transposition, backend_name, CURRENT_BACKEND(transposition_inplace));
}

/** 
 * @brief Gets the name of the active backend for transposition_inplace operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_transposition_inplace_backend() {
	return CURRENT_BACKEND(transposition_inplace);
}

/** 
 * @brief Function declaration for getting all available transposition (inplace) backends (view)
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(transposition_inplace);

/**
 * @brief Function declaration for getting the number of available transposition inplace backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(transposition_inplace);

#endif /** NNL2_TRANSPOSITION_INPLACE_H **/
