#ifndef NNL2_REINTERPRET_H
#define NNL2_REINTERPRET_H

/** @brief
 * Helper function to create a view of the entire tensor
 * Used internally by reinterpret_reshape for identical shape case
 *
 ** @param tensor
 * Pointer to the source tensor
 *
 ** @param indices
 * Optional indices for partial view (NULL for full view)
 *
 ** @param num_indices
 * Number of indices (0 for full view)
 **/
static Tensor* nnl2_create_view(Tensor* tensor, int32_t* indices, uint8_t num_indices);

/** @brief
 * Reshapes a tensor to a new shape with optional wildcard support, returning a view
 * This function creates a new tensor that shares data with the original tensor
 *
 ** @param tensor 
 * Pointer to the input tensor to be reshaped
 *
 ** @param new_shape
 * Array containing the target shape dimensions
 *
 ** @param new_shape_len
 * Number of dimensions in the new shape
 *
 ** @param force
 * If true, bypasses element count validation
 *
 ** @note
 * Wildcard dimension (-1) must appear at most once in the new_shape array
 * The returned tensor shares data with the original tensor - modifications affect both
 *
 ** @warning
 * Using force=true can lead to undefined behavior if shapes are incompatible
 * The original tensor must not be freed while reinterpreted views exist
 *
 ** @exception NNL2_ERROR_SHAPE_OVERFLOW 
 * Shape nnl2_product would exceed maximum size
 *
 ** @exception NNL2_ERROR_WILDCARD_COUNT 
 * More than one wildcard dimension found
 *
 ** @exception NNL2_ERROR_WILDCARD_COMPUTE 
 * Cannot compute wildcard dimension
 *
 ** @exception NNL2_ERROR_SHAPE_MISMATCH 
 * Element count doesn't match and force=false
 *
 ** @code
 * // Example: Create view with different shape
 * Tensor* original = nnl2_zeros((int[]){2, 3}, 2, FLOAT64);
 * Tensor* view = nnl2_naive_reinterpret(original, (int[]){3, 2}, 2, false);
 * // Both tensors share the same underlying data
 ** @endcode
 **
 ** @code
 * // Example: Create view with different shape with wildcard
 * Tensor* original = nnl2_zeros((int[]){2, 3}, 2, FLOAT64);
 * Tensor* view = nnl2_naive_reinterpret(original, (int[]){3, -1}, 2, false); // -1 Is wildcard. New shape: [3, 2]
 ** @endcode
 ** @see nnl2_naive_reshape
 ** @see nnl2_naive_view
 **/
Tensor* nnl2_naive_reinterpret(Tensor* tensor, int32_t* new_shape, int32_t new_shape_len, bool force) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor, "Passed tensor is NULL", NULL);
    #endif
    
    // Early return if shapes are identical
    if (tensor->rank == new_shape_len && memcmp(tensor->shape, new_shape, new_shape_len * sizeof(int32_t)) == 0) {
        return nnl2_create_view(tensor, NULL, 0); 
    }
    
    // Calculate total elements from original tensor
    size_t total_elems = nnl2_product(tensor->shape, tensor->rank);
    
    // Process shape and handle wildcards
    int32_t wildcard_index = -1; 
    size_t wildcard_count = 0;
    size_t wildcard_shape_product = 1;
    
    for(int32_t i = 0; i < new_shape_len; i++) {
        if(new_shape[i] == -1) { // Wildcard dimension
            wildcard_index = i;
            wildcard_count++;
        } else if (new_shape[i] < 0) { // Negative but not wildcard
            NNL2_ERROR("Invalid shape dimension: %d", new_shape[i]);
            return NULL;
        } else {
            #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
                if (new_shape[i] > 0) {
                    if (wildcard_shape_product > SIZE_MAX / new_shape[i]) {
                        NNL2_ERROR("Shape nnl2_product overflow at dimension %d", i);
                        return NULL;
                    }
                }
            #endif
			
            wildcard_shape_product *= new_shape[i];
        }
    }
    
    // Validate wildcard count
    if(wildcard_count > 1) {
        NNL2_ERROR("Must have at most one wildcard (-1), found %d", wildcard_count);
        return NULL;
    }
    
    // Resolve wildcard dimension if present
    int32_t* resolved_shape = new_shape;
    
    if(wildcard_count == 1) {
        if(wildcard_shape_product == 0 || total_elems % wildcard_shape_product != 0) {
            NNL2_ERROR("Cannot compute wildcard: %d %% %d != 0", total_elems, wildcard_shape_product);
            return NULL;
        }
        
        // Create temporary resolved shape using malloc
        int32_t* temp_shape = (int32_t*)malloc(new_shape_len * sizeof(int32_t));
        if (!temp_shape) {
            NNL2_ERROR("Failed to allocate temporary shape");
            return NULL;
        }
        
        memcpy(temp_shape, new_shape, new_shape_len * sizeof(int32_t));
        temp_shape[wildcard_index] = total_elems / wildcard_shape_product;
        resolved_shape = temp_shape;
    } else {
        // No wildcard
        if(total_elems != wildcard_shape_product && !force) {
            NNL2_ERROR("Number of elements for reshape does not match: expected %d, got %d", total_elems, wildcard_shape_product);
            return NULL;
        }
    }
    
    // Create view tensor 
    Tensor* view_tensor = (Tensor*)malloc(sizeof(Tensor));
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!view_tensor) {
            NNL2_ERROR("Failed to allocate view tensor");
            if (wildcard_count == 1) free((void*)resolved_shape);
            return NULL;
        }
    #endif
    
    // Initialize view tensor structure
    view_tensor->dtype = tensor->dtype;
    view_tensor->rank = new_shape_len;
    view_tensor->data = tensor->data; // Shared data
	view_tensor->is_view = true;
    
    // Allocate and copy shape
    view_tensor->shape = (int32_t*)malloc(new_shape_len * sizeof(int32_t));
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!view_tensor->shape) {
            NNL2_ERROR("Failed to allocate shape for view tensor");
            if (wildcard_count == 1) free((void*)resolved_shape);
            free(view_tensor);
            return NULL;
        }
    #endif
    
    memcpy(view_tensor->shape, resolved_shape, new_shape_len * sizeof(int32_t));
    
    // Free temporary shape if allocated it for wildcard resolution
    if (wildcard_count == 1) {
        free((void*)resolved_shape);
    }
    
    // Calculate strides for the new shape 
    view_tensor->strides = (int32_t*)malloc(new_shape_len * sizeof(int32_t));
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!view_tensor->strides) {
            NNL2_ERROR("Failed to allocate strides for view tensor");
            free(view_tensor->shape);
            free(view_tensor);
            return NULL;
        }
    #endif
    
    // Compute strides for the new shape
    view_tensor->strides[new_shape_len - 1] = 1;
    for (int32_t i = new_shape_len - 2; i >= 0; i--) {
        view_tensor->strides[i] = view_tensor->strides[i + 1] * view_tensor->shape[i + 1];
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return view_tensor;
}

/** @brief
 * See docs at declaration
 *
 ** @see nnl2_create_view
 **/
static Tensor* nnl2_create_view(Tensor* tensor, int32_t* indices, uint8_t num_indices) {
    Tensor* view = (Tensor*)malloc(sizeof(Tensor));
    if (!view) return NULL;
    
    view->dtype = tensor->dtype;
    view->rank = tensor->rank;

    // Allocate and copy shape
    view->shape = (int32_t*)malloc(tensor->rank * sizeof(int32_t));
    if (!view->shape) {
        free(view);
        return NULL;
    }
    memcpy(view->shape, tensor->shape, tensor->rank * sizeof(int32_t));
    
    // Allocate and copy strides
    view->strides = (int32_t*)malloc(tensor->rank * sizeof(int32_t));
    if (!view->strides) {
        free(view->shape);
        free(view);
        return NULL;
    }
    memcpy(view->strides, tensor->strides, tensor->rank * sizeof(int32_t));
    
    // Calculate data offset if indices are provided
    size_t offset = 0;
    if (indices && num_indices > 0) {
        for (uint8_t i = 0; i < num_indices; i++) {
            offset += indices[i] * tensor->strides[i];
        }
    }
    
    const size_t element_size = get_dtype_size(tensor->dtype);
    view->data = (char*)tensor->data + offset * element_size;
    
    return view;
}

/** 
 * @ingroup backend_system
 * @brief Backend implementations for reinterpret operation
 * @details
 * Array follows the common backend registration pattern.
 * Currently registered backends:
 *  - nnl2_naive_reinterpret: Basic reference implementation
 * 
 * @see nnl2_naive_reinterpret
 */
Implementation reinterpret_backends[] = {
    REGISTER_BACKEND(nnl2_naive_reinterpret, nnl2_naive, NAIVE_BACKEND_NAME),
};  

/**
 * @brief Function pointer for reinterpret operation
 * @ingroup backend_system 
 */
reinterpretfn nnl2_reinterpret;

/** 
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(reinterpret);

/** 
 * @brief Sets the backend for reinterpret operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void set_reinterpret_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(reinterpret_backends, nnl2_reinterpret, backend_name, CURRENT_BACKEND(reinterpret));
}

/** 
 * @brief Gets the name of the active backend for reinterpret operation
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* get_reinterpret_backend() {
    return CURRENT_BACKEND(reinterpret);
}

/** 
 * @brief Function declaration for getting all `reinterpret` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(reinterpret);

/**
 * @brief Function declaration for getting the number of all `reinterpret` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(reinterpret);

#endif /** NNL2_REINTERPRET_H **/
