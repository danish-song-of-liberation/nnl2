#ifndef NNL2_TREF_H
#define NNL2_TREF_H

/** @brief
 * Gets an independent copy of an element or a subtensor from a tensor using the specified indices
 * Creates deep copies instead of views for complete data independence
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
 * Can range from 0 (return full tensor copy) to tensor->rank (return single element copy)
 *
 ** @return
 * If num_indices == tensor->rank returns pointer to a copy of the specific element
 * If num_indices < tensor->rank returns pointer to an independent subtensor copy
 * NULL in case of any error or invalid parameters
 *
 ** @note
 * When returning a subtensor, it creates an independent copy that does not share data
 * with the original tensor. Modifications to the copy will not affect the original tensor
 *
 ** @note
 * The function performs index boundary checks based on the safety level
 *
 ** @note
 * Uses tensor->strides for efficient offset calculation
 *
 ** @details
 * The function:
 * 1. Validates parameters and performs safety checks
 * 2. Calculates the memory offset using strides
 * 3. Creates independent copies of data instead of views
 *
 ** @code
 * // Example 1: Copy single element from 3D tensor
 * int indices[] = {1, 2, 3};
 * float* element_copy = (float*)nnl2_naive_copy(tensor3d, indices, 3);
 *
 * // Example 2: Create independent 2D slice from 3D tensor
 * int slice_indices[] = {1};
 * Tensor* slice_copy = (Tensor*)nnl2_naive_copy(tensor3d, slice_indices, 1);
 ** @endcode
 **
 ** @warning
 * The returned data must be manually freed when no longer needed
 * For subtensors use nnl2_free_tensor()
 * For single elements use FREE_ALIGNED()
 *
 ** @see nnl2_free_tensor()
 ** @see get_dtype_size()
 ** @see ALLOC_ALIGNED
 ** @see FREE_ALIGNED
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
 * If memory allocation fails for data copying
 *
 **/
void* nnl2_naive_tref_getter(Tensor* tensor, const int32_t* indices, uint8_t num_indices) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Parameter validation and safety checks
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        if (tensor == NULL) {
            NNL2_ERROR("Null tensor pointer in copy");
            return NULL;
        }
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        if (indices == NULL) {
            NNL2_ERROR("Null indices pointer in copy");
            return NULL;
        }
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
        if (tensor->rank <= 0 || tensor->shape == NULL || tensor->data == NULL) {
            NNL2_ERROR("Invalid tensor structure in copy");
            return NULL;
        }
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if (num_indices > tensor->rank) {
            NNL2_ERROR("Too many indices (%u > %d) in copy", num_indices, tensor->rank);
            return NULL;
        }

        for (uint8_t i = 0; i < num_indices; i++) {
            if (indices[i] < 0 || indices[i] >= tensor->shape[i]) {
                NNL2_ERROR("Index %u (%d) out of bounds for dimension %u (size %d) in copy",
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
    
    const size_t element_size = get_dtype_size(tensor->dtype);

    // Result construction with independent copies
    if (num_indices == tensor->rank) {
        // Copy single element using aligned allocation
        void* element_copy = NULL;
        ALLOC_ALIGNED(element_copy, TENSOR_MEM_ALIGNMENT, element_size);
		
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
            if (!element_copy) {
                NNL2_ERROR("Failed to allocate element copy");
				free(element_copy);
                return NULL;
            }
        #endif
        
        memcpy(element_copy, (char*)tensor->data + offset * element_size, element_size);
        return element_copy;
    }

    // Create independent subtensor copy
    Tensor* subtensor = (Tensor*)malloc(sizeof(Tensor));
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if (!subtensor) {
            NNL2_ERROR("Failed to allocate subtensor in copy");
            return NULL;
        }
    #endif

    subtensor->dtype = tensor->dtype;
    subtensor->rank = tensor->rank - num_indices;
    
    // Allocate and copy shape dimensions
    subtensor->shape = (int32_t*)malloc(subtensor->rank * sizeof(int32_t));
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if (!subtensor->shape) {
            NNL2_ERROR("Failed to allocate subtensor shape in copy");
            free(subtensor);
            return NULL;
        }
    #endif
    
    memcpy(subtensor->shape, tensor->shape + num_indices, subtensor->rank * sizeof(int32_t));

    // Allocate and copy strides
    subtensor->strides = (int32_t*)malloc(subtensor->rank * sizeof(int32_t));
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if (!subtensor->strides) {
            NNL2_ERROR("Failed to allocate subtensor strides in copy");
            free(subtensor->shape);
            free(subtensor);
            return NULL;
        }
    #endif
    
    memcpy(subtensor->strides, tensor->strides + num_indices, subtensor->rank * sizeof(int32_t));

    // Calculate total elements in subtensor
    size_t numel = 1;
    for (int i = 0; i < subtensor->rank; i++) {
        numel *= subtensor->shape[i];
    }

    // Allocate independent aligned memory for data and copy from source
    size_t data_size = numel * element_size;
    void* data = NULL;
    ALLOC_ALIGNED(data, TENSOR_MEM_ALIGNMENT, data_size);
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if (!data) {
            NNL2_ERROR("Failed to allocate subtensor data in copy");
            free(subtensor->strides);
            free(subtensor->shape);
            free(subtensor);
            return NULL;
        }
    #endif
    
    subtensor->data = data;
    
    // Copy the contiguous block of data for the subtensor
    char* source_ptr = (char*)tensor->data + offset * element_size;
    memcpy(subtensor->data, source_ptr, data_size);

    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif

    return subtensor;
}

/** @ingroup backend_system
 ** @brief Backend implementations for tref (getter)
 ** @details
 * Array follows the common backend registration pattern
 * Currently register backends:
 *  - nnl2_naive_tref_getter: Basic reference implementation
 *
 ** @see REGISTER_BACKEND
 ** @see NAIVE_BACKEND_NAME
 ** @see nnl2_naive_tref_getter
 ** @see nnl2_naive
 **/
Implementation nnl2_tref_getter_backends[] = {
	REGISTER_BACKEND(nnl2_naive_tref_getter, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for the active tref (getter) 
 * @ingroup backend_system 
 */
trefgetterfn nnl2_tref_getter;

/** 
 * @brief Sets the backend for tref (getter)
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void nnl2_set_tref_getter_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(nnl2_tref_getter_backends, nnl2_tref_getter, backend_name);
}

/** 
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(nnl2_tref_getter);

/** 
 * @brief Gets the name of the active backend for tref (getter)
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* nnl2_get_tref_getter_backend() {
	return CURRENT_BACKEND(nnl2_tref_getter);
}

/** 
 * @brief Function declaration for getting all `tref (getter)` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(nnl2_tref_getter);

/**
 * @brief Function declaration for getting the number of all `tref (getter)` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(nnl2_tref_getter);

#endif /** NNL2_TREF_H **/
