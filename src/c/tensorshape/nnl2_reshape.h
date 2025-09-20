#ifndef NNL2_RESHAPE_H
#define NNL2_RESHAPE_H

/** @brief
 * Reshapes a tensor to a new shape with optional wildcard support
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
 *
 ** @warning
 * Using force=true can lead to undefined behavior if shapes are incompatible
 *
 ** @exception NNL2_ERROR_MEMORY_ALLOCATION
 * Failed to allocate memory for shape buffer
 *
 ** @exception NNL2_ERROR_SHAPE_OVERFLOW 
 * Shape product would exceed maximum size
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
 ** @exception NNL2_ERROR_TENSOR_ALLOCATION 
 * Failed to allocate new tensor
 *
 ** @exception NNL2_ERROR_UNSUPPORTED_TYPE 
 * Unsupported tensor data type
 *
 ** @code
 * // Example 1: Without wildcard
 * // Original tensor shape: [2, 3] (6 elements)
 * Tensor* original_tensor_no_wildcard = nnl2_zeros((int[]){2, 3}, 2, FLOAT64) // [[0, 0, 0], [0, 0, 0]]
 * Tensor* reshaped_tensor_no_wildcard = nnl2_naive_reshape(original_tensor_no_wildcard, (int[]){3, 2}, 2, false) // [[0, 0], [0, 0], [0, 0]]
 * // Result: shape [2, 3] -> [3, 2] (element count matches: 2 * 3 = 6 and 3 * 2 = 6)
 ** @endcode
 **
 ** @code
 * // Example 2: With wildcard
 * // Original tensor shape: [3, 4] (12 elements)
 * Tensor* original_tensor_with_wildcard = nnl2_zeros((int[]){3, 4}, 2, FLOAT64) // [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]
 * Tensor* reshaped_tensor_with_wildcard = nnl2_naive_reshape(original_tensor_with_wildcard, (int[]){4, -1}, 2, false) // -1 -> 3 ([4, 3]), [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
 ** @endcode
 **
 ** @see nnl2_empty
 ** @see nnl2_free_tensor
 ** @see product
 **/
Tensor* nnl2_naive_reshape(Tensor* tensor, int32_t* new_shape, int32_t new_shape_len, bool force) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor, "Passed tensor is NULL", NULL);
	#endif
	
	// Early return if shapes are identical
    if (tensor->rank == new_shape_len && memcmp(tensor->shape, new_shape, new_shape_len * sizeof(int32_t)) == 0) {
        return nnl2_copy(tensor, tensor->dtype); 
    }
	
	// Calculate total elements from original tensor
    size_t total_elems = product(tensor->shape, tensor->rank);
	
	// Allocate temporary buffer for shape processing (including wildcard resolution)
    int32_t* wildcard_shape = malloc(new_shape_len * sizeof(int32_t));
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if (!wildcard_shape) {
			NNL2_ERROR("Memory allocation failed");
			return NULL;
		}
	#endif
    
    int32_t wildcard_index = NNL2_WILDCARD_DIM;		// Index of wildcard dimension (-1 if not found)
    size_t wildcard_count = 0;						// Number of wildcard dimensions found
    size_t wildcard_shape_product = 1;  			// Product of non-wildcard dimensions
    
    for(int32_t i = 0; i < new_shape_len; i++) {
        wildcard_shape[i] = new_shape[i];
        if(new_shape[i] == NNL2_WILDCARD_DIM) {  // if(new_shape[i] == -1)
			// Found wildcard dimension
            wildcard_index = i;
            wildcard_count++;
        } else if (new_shape[i] < NNL2_WILDCARD_DIM) {
			NNL2_ERROR("Invalid shape dimension: %d", new_shape[i]);
			free(wildcard_shape);
			return NULL;
		} else {
			// Non-wildcard dimension
			
			#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
				if (new_shape[i] > 0) {
					// Check for multiplication overflow
					if (wildcard_shape_product > SIZE_MAX / new_shape[i]) {
						NNL2_ERROR("Shape product overflow at dimension %d: %d * %d would exceed maximum size", i, wildcard_shape_product, new_shape[i]);
						free(wildcard_shape);
						return NULL;
					}
				}
			#endif
			
            wildcard_shape_product *= new_shape[i];
        }
    }
    
	// Validate wildcard count (0 or 1 allowed)
    if(wildcard_count > 1) {
        NNL2_ERROR("Must have at most one wildcard (-1), found %d", wildcard_count);
        free(wildcard_shape);
        return NULL;
    }
    
    if(wildcard_count == 1) {
		// Handle wildcard dimension case
        if(wildcard_shape_product == 0 || total_elems % wildcard_shape_product != 0) {
            NNL2_ERROR("Cannot compute wildcard: %d %% %d != 0", total_elems, wildcard_shape_product);
            free(wildcard_shape);
            return NULL;
        }
        
		// Calculate and set the wildcard dimension value
        int32_t wildcard_value = total_elems / wildcard_shape_product;
        wildcard_shape[wildcard_index] = wildcard_value;
    } else {
		// No wildcard case
        if(total_elems != wildcard_shape_product && !force) {
            NNL2_ERROR("Number of elements for reshape does not match: expected %d, got %d", total_elems, wildcard_shape_product);
            free(wildcard_shape);
            return NULL;
        }
    }
    
	// Create new tensor with the resolved shape
    Tensor* new_tensor = nnl2_empty(wildcard_shape, new_shape_len, tensor->dtype);
    free(wildcard_shape); // Free temporary shape buffer
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(new_tensor == NULL) {
			NNL2_ERROR("Tensor allocation failed");
		}
	#endif
    
	// Copy data from original tensor to reshaped tensor
    switch(tensor->dtype) {
        case FLOAT64: {
            double* reshape_data = (double*)new_tensor->data;  // Casting
            double* original_data = (double*)tensor->data;
            for(size_t it = 0; it < total_elems; it++) reshape_data[it] = original_data[it]; // Copying
            break;
        }
        
        case FLOAT32: {
            float* reshape_data = (float*)new_tensor->data;
            float* original_data = (float*)tensor->data;
            for(size_t it = 0; it < total_elems; it++) reshape_data[it] = original_data[it];
            break;
        }
        
        case INT32: {
            int32_t* reshape_data = (int32_t*)new_tensor->data;
            int32_t* original_data = (int32_t*)tensor->data;
            for(size_t it = 0; it < total_elems; it++) reshape_data[it] = original_data[it];
            break;
        }
        
        default: {
            NNL2_TYPE_ERROR(tensor->dtype);
            nnl2_free_tensor(new_tensor);
            return NULL;
        }
    }
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return new_tensor;
}
 
/** 
 * @ingroup backend_system
 * @brief Backend implementations for reshape operation
 * @details
 * Array follows the common backend registration pattern.
 * Currently registered backends:
 *  - nnl2_naive_reshape: Basic reference implementation
 * 
 * @see nnl2_naive_reshape
 */
Implementation reshape_backends[] = {
    REGISTER_BACKEND(nnl2_naive_reshape, nnl2_naive, NAIVE_BACKEND_NAME),
};  

/**
 * @brief Function pointer for reshape operation
 * @ingroup backend_system 
 */
reshapefn nnl2_reshape;

/** 
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(reshape);

/** 
 * @brief Sets the backend for reshape operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void set_reshape_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(reshape_backends, nnl2_reshape, backend_name, CURRENT_BACKEND(reshape));
}

/** 
 * @brief Gets the name of the active backend for reshape operation
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* get_reshape_backend() {
    return CURRENT_BACKEND(reshape);
}

/** 
 * @brief Function declaration for getting all `reshape` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(reshape);

/**
 * @brief Function declaration for getting the number of all `reshape` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(reshape);

#endif /** NNL2_RESHAPE_H **/
