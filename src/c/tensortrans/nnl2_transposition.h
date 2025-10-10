#ifndef NNL2_TRANSPOSITION_H
#define NNL2_TRANSPOSITION_H

/** @brief 
 * Creates a transposed view of a 2D tensor (matrix) in O(1) time
 *
 ** @param tensor
 * Pointer to the original 2D tensor to be transposed
 *
 ** @return
 * Pointer to a new tensor view with transposed shape and strides
 *
 ** @warning
 * This operation creates a VIEW, not a copy! The returned tensor
 * shares data with the original tensor. Modifications to either
 * will affect both
 *
 ** @warning
 * MATHEMATICALLY INCORRECT FOR MOST OPERATIONS!
 *
 * While the visual representation is correct, most tensor operations
 * (addition, multiplication, etc.) will produce mathematically 
 * incorrect results when applied to transposed views
 * 
 * This implementation only swaps shape and strides. the underlying
 * data remains in row-major order. Operations that access data 
 * linearly will not respect the transposed layout
 *
 ** @see nnl2_empty
 ** @see nnl2_free_tensor
 **/
Tensor* nnl2_naive_transposition(const Tensor* tensor) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor, "Passed tensor in transposition is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor->data, "Passed tensor data in transposition is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor->shape, "Passed tensor shape in transposition is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor->strides, "Passed tensor stride in transposition is NULL", NULL);
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(tensor->rank != 2) {
            NNL2_ERROR("Tensor has incorrect rank: %d (expected 2)", tensor->rank);
            return NULL;
        }
    #endif
    
    Tensor* result = (Tensor*)malloc(sizeof(Tensor));
    if (!result) {
        NNL2_ERROR("Memory allocation failed for tensor structure");
        return NULL;
    }
    
	// Copyin metadata
    result->dtype = tensor->dtype;
    result->rank = tensor->rank;
    result->is_view = true;
    result->data = tensor->data;  
	
    result->shape = (int32_t*)malloc(result->rank * sizeof(int32_t));
    result->strides = (int32_t*)malloc(result->rank * sizeof(int32_t));
    
    if (!result->shape || !result->strides) {
        if (result->shape) free(result->shape);
        if (result->strides) free(result->strides);
        free(result);
        NNL2_ERROR("Memory allocation failed for shape/strides");
        return NULL;
    }
    
    result->shape[0] = tensor->shape[1];  // cols -> rows
    result->shape[1] = tensor->shape[0];  // rows -> cols
    
    result->strides[0] = tensor->strides[1];  
    result->strides[1] = tensor->strides[0];  
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return result;
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for transposition (view) operation
 * @details
 * Array follows the common backend registration pattern for transposition (view)
 * operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for transposition (view)
 * 
 * @see nnl2_naive
 * @see nnl2_naive_transposition
 */
Implementation transposition_backends[] = {
    REGISTER_BACKEND(nnl2_naive_transposition, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for transposition operation (view)
 * @ingroup backend_system 
 */
transpositionfn nnl2_transposition;

/** 
 * @brief Makes the transposition backend current
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(transposition);

/** 
 * @brief Sets the backend for transposition operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for transposition
 * @see ESET_BACKEND_BY_NAME
 */
void set_transposition_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(transposition_backends, nnl2_transposition, backend_name, CURRENT_BACKEND(transposition));
}

/** 
 * @brief Gets the name of the active backend for transposition operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_transposition_backend() {
	return current_backend(transposition);
}

/** 
 * @brief Function declaration for getting all available transposition backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(transposition);

/**
 * @brief Function declaration for getting the number of available transposition backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(transposition);

#endif /** NNL2_TRANSPOSITION_H **/
