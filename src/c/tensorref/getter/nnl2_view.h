#ifndef NNL2_VIEW_H
#define NNL2_VIEW_H

/** @brief
 * Gets an element or a subtensor from a tensor using the specified indices
 * Uses precomputed strides for optimal performance
 *
 ** @param tensor
 * Pointer to the source tensor from which to extract elements or subtensors
 * Must be a valid tensor with properly initialized strides and shape arrays
 *
 ** @param indices
 * An array of indices for accessing tensor elements along each dimension
 * Indices are applied in row-major order (first index for outermost dimension)
 * Partial indexing is supported for creating subtensors
 *
 ** @param num_indices
 * The number of indices in the indices array
 * Can range from 0 (return full tensor view) to tensor->rank (return single element)
 *
 ** @return
 * If num_indices == tensor->rank returns pointer to the specific element in tensor data
 * If num indices < tensor->rank returns pointer to a subtensor
 * NULL in case of any error or invalid parameters
 *
 ** @note
 * When returning a subtensor, it creates a view that shares data with the original tensor
 * Modifications to the subtensor will affect the original tensor and vice versa
 *
 ** @note
 * The function performs index boundary checks based on the safety level
 *
 ** @note
 * Uses tensor->strides for efficient offset calculation
 *
 ** @details
 * The function firstly:
 *
 *** Сhecks the parameters for correctness
 ** Then
 *** Calculates the shift using steps
 ** Finally
 *** Returns a result based on the received 
 *** data, namely a scalar or a subtensor
 *
 ** @code
 * // Example 1: Access single element from 3D tensor
 * int indices[] = {1, 2, 3};
 * float* element = (float*)nnl2_naive_view(tensor3d, indices, 3);
 *
 * // Example 2: Create 2D slice from 3D tensor
 * int slice_indices[] = {1};
 * Tensor* slice = (Tensor*)nnl2_naive_view(tensor3d, slice_indices, 1);
 ** @endcode
 **
 ** @warning
 * The returned subtensor is a view that shares data with the original tensor
 * Freeing the original tensor while subtensors exist will cause dangling pointers
 *
 ** @see nnl2_free_tensor()
 ** @see get_dtype_size()
 ** @see NNL2_DEBUG_MODE
 ** @see NNL2_SAFETY_MODE
 **
 ** @exception[invalid_argument]
 * If tensor is NULL, indices is NULL, or tensor structure is invalid
 *
 ** @exception[out_of_range]
 * If num_indices exceeds tensor rank or any index is out of bounds
 *
 ** @exception[out_of_memory]
 * If memory allocation fails for subtensor structure or arrays
 *
 **/
void* nnl2_naive_view(Tensor* tensor, const int32_t* indices, uint8_t num_indices) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Parameter validation and safety checks
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		if (tensor == NULL) {
			NNL2_ERROR("Null tensor pointer in view");
			return NULL;
		}
    
		if (indices == NULL) {
			NNL2_ERROR("Null indices pointer in view");
			return NULL;
		}
	#endif
    
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		if (tensor->rank <= 0 || tensor->shape == NULL || tensor->data == NULL) {
			NNL2_ERROR("Invalid tensor structure in view");
			return NULL;
		}
	#endif
    
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if (num_indices > tensor->rank) {
			NNL2_ERROR("Too many indices (%u > %d) in view", num_indices, tensor->rank);
			return NULL;
		}
		
		// Validate each index against corresponding dimension bounds	
		for (uint8_t i = 0; i < num_indices; i++) {
			if (indices[i] < 0 || indices[i] >= tensor->shape[i]) {
				NNL2_ERROR("Index %u (%d) out of bounds for dimension %u (size %d) in view",
							i, indices[i], i, tensor->shape[i]);
						
				return NULL;
			}
		}
	#endif
	
	// Offset calculation using precomputed strides
    size_t offset = 0;

    for (uint8_t i = 0; i < num_indices; i++) {
        offset += indices[i] * tensor->strides[i];	
    }
	
	// Result construction
    if (num_indices == tensor->rank) {
        const size_t element_size = get_dtype_size(tensor->dtype);
		return (char*)tensor->data + offset * element_size;
    }

    Tensor* subtensor = (Tensor*)malloc(sizeof(Tensor));
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if (!subtensor) {
			NNL2_ERROR("Failed to allocate subtensor in view");
			return NULL;
		}
	#endif

    subtensor->dtype = tensor->dtype;
    subtensor->rank = tensor->rank - num_indices;
	subtensor->is_view = true;
        
	 // Allocate and copy remaining shape dimensions	
    subtensor->shape = (int32_t*)malloc(subtensor->rank * sizeof(int32_t));
    
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if (!subtensor->shape) {
			NNL2_ERROR("Failed to allocate subtensor shape in view");
			free(subtensor);
			return NULL;
		}
	#endif
    
	// Copy shape information for non-indexed dimensions
	memcpy(subtensor->shape, tensor->shape + num_indices, subtensor->rank * sizeof(int32_t));

	// Allocate and copy corresponding strides
	// Strides for remaining dimensions remain valid since they're precomputed
	subtensor->strides = (int32_t*)malloc(subtensor->rank * sizeof(int32_t));
	
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if (!subtensor->strides) {
            NNL2_ERROR("Failed to allocate subtensor strides in view");
            free(subtensor->shape);
            free(subtensor);
            return NULL;
        }
    #endif
	
	// Copy stride information for non-indexed dimensions
	memcpy(subtensor->strides, tensor->strides + num_indices, subtensor->rank * sizeof(int32_t));

	// Set data pointer to shared memory with calculated offset
    const size_t element_size = get_dtype_size(tensor->dtype);
    subtensor->data = (char*)tensor->data + offset * element_size;

	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif

    return subtensor;
}

/** @ingroup backend_system
 ** @brief Backend implementations for view
 ** @details
 * Array follows the common backend registration pattern
 * Currently register backends:
 *  - nnl2_naive_tref_getter: Basic reference implementation
 *
 ** @see REGISTER_BACKEND
 ** @see NAIVE_BACKEND_NAME
 ** @see nnl2_naive_view
 ** @see nnl2_naive
 **/
Implementation nnl2_view_backends[] = {
	REGISTER_BACKEND(nnl2_naive_view, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for the active view backend 
 * @ingroup backend_system 
 */
viewfn nnl2_view;

/** 
 * @brief Sets the backend for view
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void nnl2_set_view_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(nnl2_view_backends, nnl2_view, backend_name);
}

/** 
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(nnl2_view);

/** 
 * @brief Gets the name of the active backend for view
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* nnl2_get_view_backend() {
	return CURRENT_BACKEND(nnl2_view);
}

/** 
 * @brief Function declaration for getting all `view` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(nnl2_view);

/**
 * @brief Function declaration for getting the number of all `view` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(nnl2_view);

#endif /** NNL2_VIEW_H **/
