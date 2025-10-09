#ifndef NNL2_CUT_H
#define NNL2_CUT_H

/** @brief
 * Returns a slice of the tensor using the specified indices (view)
 *
 ** @param tensor
 * Input tensor to slice
 *
 ** @param cut_from
 * Indices from cut to
 *
 ** @param cut_to 
 * Indicies to cut to
 *
 ** @code
 * // Example (lisp)
 *
 * (nnl2.hli.ts:tlet* ((a (nnl2.hli.ts:ones #(5 5)))
 *					   (b (nnl2.hli.ts:cut a :from #(0 0) :to #(5 3))))
 * 
 *   ...)
 ** @endcode
 ** @see Tensor (struct)
 **/
Tensor* nnl2_naive_cut(Tensor* tensor, int32_t* cut_from, int32_t* cut_to) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        // Check for NULL pointers in input parameters
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor, "Passed tensor in cut is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(cut_from, "Incorrect cut structure (cut_from is NULL)", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(cut_to, "Incorrect cut structure (cut_to is NULL)", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor->data, "Passed tensor data in cut is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor->shape, "Passed tensor shape in cut is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor->strides, "Passed tensor stride in cut is NULL", NULL);
    #endif
    
    // Correctness checks
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN   
        for (int i = 0; i < tensor->rank; i++) {
            // Check for negative indices in cut_from
            if (cut_from[i] < 0) {
                NNL2_ERROR("cut_from[%d] = %d is negative", i, cut_from[i]);
                return NULL;
            }

            // Check if cut_to exceeds tensor dimension bounds
            if (cut_to[i] > tensor->shape[i]) {
                NNL2_ERROR("cut_to[%d] = %d exceeds tensor dimension size %d", i, cut_to[i], tensor->shape[i]);
                return NULL;
            }
         
            // Check if cut range is valid (from < to)
            if (cut_from[i] >= cut_to[i]) {
                NNL2_ERROR("Invalid cut range: from[%d] = %d >= to[%d] = %d", i, cut_from[i], i, cut_to[i]);
                return NULL;
            }
			
			#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
				// Check if resulting dimension size is positive
				if (cut_to[i] - cut_from[i] <= 0) {
					NNL2_ERROR("Resulting dimension %d has zero or negative size: %d", i, cut_to[i] - cut_from[i]);
					return NULL;
				}
			#endif
        }
    #endif
    
    // Allocate memory for the result tensor structure
    Tensor* result = (Tensor*)malloc(sizeof(Tensor));
    if (!result) return NULL;
    
    // Copy basic metadata from input tensor
    result->dtype = tensor->dtype;
    result->rank = tensor->rank;
    result->is_view = true;  // This is a view, not a copy
    
    // Allocate memory for shape and strides arrays
    result->shape = (int32_t*)malloc(result->rank * sizeof(int32_t));
    result->strides = (int32_t*)malloc(result->rank * sizeof(int32_t));
    
    // Check if shape/strides allocation succeeded
    if (!result->shape || !result->strides) {
        if (result->shape)   free(result->shape);
        if (result->strides) free(result->strides);
        free(result);
        return NULL;
    }
    
    // Calculate result tensor shape
    for (int i = 0; i < tensor->rank; i++) {
        result->shape[i] = cut_to[i] - cut_from[i];
    }
    
    // Copy strides from original tensor (they remain the same for a view)
    memcpy(result->strides, tensor->strides, result->rank * sizeof(int32_t));	
    
    // Calculate the byte offset to the start of the sliced region
    size_t byte_offset = 0;
    for (int i = 0; i < tensor->rank; i++) {
        byte_offset += cut_from[i] * tensor->strides[i];
    }
    
    // Set the data pointer to the sliced region (no data copying)
    result->data = (char*)tensor->data + byte_offset;
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return result;
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for cut operation
 * @details
 * Array follows the common backend registration pattern for cut operation
 * operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation
 * 
 * @see nnl2_naive
 * @see nnl2_naive_cut
 * @see NAIVE_BACKEND_NAME
 */
Implementation cut_backends[] = {
	REGISTER_BACKEND(nnl2_naive_cut, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for cut operation
 * @ingroup backend_system 
 */
cutfn nnl2_cut;

/** 
 * @brief Sets the backend for cut operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for cut operation
 * @see SET_BACKEND_BY_NAME
 */
MAKE_CURRENT_BACKEND(cut);

/**
 * @brief Sets the backend for cut operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for cut operation
 */
void set_cut_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(cut_backends, nnl2_cut, backend_name, CURRENT_BACKEND(cut));
}

/**
 * @brief Gets the name of the current backend for cut operation
 * @ingroup backend_system
 * @return const char* Name of the current backend
 */
const char* get_cut_backend() {
    return CURRENT_BACKEND(cut);
}

/**
 * @brief Gets the list of available backends for cut operation
 * @ingroup backend_system
 * @return const char** Array of backend names
 */
DEFINE_GET_BACKENDS_FUNCTION(cut);

/**
 * @brief Gets the number of available backends for cut operation
 * @ingroup backend_system
 * @return size_t Number of available backends
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(cut);

#endif /** NNL2_CUT_H **/
