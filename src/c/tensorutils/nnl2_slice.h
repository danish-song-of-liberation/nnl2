#ifndef NNL2_SLICE_H
#define NNL2_SLICE_H

/** @brief
 * Returns a slice of the tensor using the specified indices (copy)
 *
 ** @param tensor
 * Input tensor to slice
 *
 ** @param slice_from
 * Indices from slice to
 *
 ** @param slice_to 
 * Indicies to slice to
 *
 ** @code
 * // Example (lisp)
 *
 * (nnl2.hli.ts:tlet* ((a (nnl2.hli.ts:ones #(5 5)))
 *					   (b (nnl2.hli.ts:slice a :from #(0 0) :to #(5 3))))
 * 
 *   ...)
 ** @endcode
 ** @see Tensor (struct)
 **/
Tensor* nnl2_naive_slice(Tensor* tensor, int32_t* slice_from, int32_t* slice_to) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		// Check for NULL pointers in input parameters
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor, "Passed tensor in slice is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(slice_from, "Incorrect slice structure (slice_from is NULL)", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(slice_to, "Incorrect slice structure (slice_to is NULL)", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor->data, "Passed tensor data in slice is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor->shape, "Passed tensor shape in slice is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor->strides, "Passed tensor stride in slice is NULL", NULL);
	#endif
	
	// Correctness checks
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN   
        for (int i = 0; i < tensor->rank; i++) {
			// Check for negative indices in slice_from
            if (slice_from[i] < 0) {
                NNL2_ERROR("slice_from[%d] = %d is negative", i, slice_from[i]);
                return NULL;
            }

			// Check if slice_to exceeds tensor dimension bounds
            if (slice_to[i] > tensor->shape[i]) {
                NNL2_ERROR("slice_to[%d] = %d exceeds tensor dimension size %d", i, slice_to[i], tensor->shape[i]);
                return NULL;
            }
         
		    // Check if slice range is valid (from < to)
            if (slice_from[i] >= slice_to[i]) {
                NNL2_ERROR("Invalid slice range: from[%d] = %d >= to[%d] = %d", i, slice_from[i], i, slice_to[i]);
                return NULL;
            }

			// Check if resulting dimension size is positive
            if (slice_to[i] - slice_from[i] <= 0) {
                NNL2_ERROR("Resulting dimension %d has zero or negative size: %d", i, slice_to[i] - slice_from[i]);
                return NULL;
            }
        }
    #endif
    
	// Allocate memory for the result tensor structure
    Tensor* result = (Tensor*)malloc(sizeof(Tensor));
    if (!result) return NULL;
    
	// Copy basic metadata from input tensor
    result->dtype = tensor->dtype;
    result->rank = tensor->rank;
    result->is_view = false;  // This is a copy, not a view
    
	// Allocate memory for shape and strides arrays
    result->shape = (int32_t*)malloc(result->rank * sizeof(int32_t));
    result->strides = (int32_t*)malloc(result->rank * sizeof(int32_t));
    
	// Check if shape/strides allocation succeeded
    if (!result->shape || !result->strides) {
        if (result->shape) free(result->shape);
        if (result->strides) free(result->strides);
        free(result);
        return NULL;
    }
    
	// Element size calculation
    size_t elem_size;
    switch (tensor->dtype) {
        case INT32: elem_size = sizeof(int32_t); break;
        case FLOAT32: elem_size = sizeof(float); break;
        case FLOAT64: elem_size = sizeof(double); break;
        default: 
            free(result->shape);
            free(result->strides);
            free(result);
            return NULL;
    }
	
	// Calculate result tensor shape and total elements
    int32_t total_elements = 1;
    for (int i = 0; i < tensor->rank; i++) {
		// Additional safety checks 
        if (slice_from[i] < 0 || slice_to[i] > tensor->shape[i] || 
            slice_from[i] >= slice_to[i]) {
            free(result->shape);
            free(result->strides);
            free(result);
            return NULL;
        }
        
		// Calculate new dimension size: to - from
        result->shape[i] = slice_to[i] - slice_from[i];
		
		// Accumulate total number of elements in result tensor
        total_elements *= result->shape[i];
    }
    
	// Data buffer allocation
    result->data = malloc(total_elements * elem_size);
    if (!result->data) {
        free(result->shape);
        free(result->strides);
        free(result);
        return NULL;
    }
	
	// Strides calculation for result tensor
	
    result->strides[result->rank - 1] = 1;
    for (int i = result->rank - 2; i >= 0; i--) {
        result->strides[i] = result->strides[i + 1] * result->shape[i + 1];
    }
    
    for (int i = 0; i < result->rank; i++) {
        result->strides[i] *= elem_size;
    }
    
    char* dest_ptr = (char*)result->data;
    
	// Allocate and initialize indices array for tracking current position
    int32_t* indices = (int32_t*)calloc(tensor->rank, sizeof(int32_t));
    if (!indices) {
        free(result->data);
        free(result->shape);
        free(result->strides);
        free(result);
        return NULL;
    }
    
	// Iterate through all elements in the result tensor
    for (int i = 0; i < total_elements; i++) {
		// Calculate source offset in original tensor
        int32_t src_offset = 0;
        for (int dim = 0; dim < tensor->rank; dim++) {
			// Convert result indices to original tensor indices
            src_offset += (slice_from[dim] + indices[dim]) * (tensor->strides[dim] / elem_size);
        }
        
		// Copy single element from source to destination
        memcpy(dest_ptr, (char*)tensor->data + src_offset * elem_size, elem_size);
        
		// Move destination pointer to next element
        dest_ptr += elem_size;
        
		// Update indices
        for (int dim = tensor->rank - 1; dim >= 0; dim--) {
            indices[dim]++;
            if (indices[dim] < result->shape[dim]) {
                break; // No carry-over needed
            }
			
            indices[dim] = 0; // Reset dimension and carry to next
        }
    }
    
    free(indices);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
	
    return result;
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for slice operation
 * @details
 * Array follows the common backend registration pattern for slice operation
 * operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation
 * 
 * @see nnl2_naive
 * @see nnl2_naive_slice
 * @see NAIVE_BACKEND_NAME
 */
Implementation slice_backends[] = {
	REGISTER_BACKEND(nnl2_naive_slice, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for slice operation
 * @ingroup backend_system 
 */
slicefn nnl2_slice;

/** 
 * @brief Sets the backend for slice operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for slice operation
 * @see SET_BACKEND_BY_NAME
 */
MAKE_CURRENT_BACKEND(slice);

/**
 * @brief Sets the backend for slice operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for slice operation
 */
void set_slice_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(slice_backends, nnl2_slice, backend_name, CURRENT_BACKEND(slice));
}

/**
 * @brief Gets the name of the current backend for slice operation
 * @ingroup backend_system
 * @return const char* Name of the current backend
 */
const char* get_slice_backend() {
    return CURRENT_BACKEND(slice);
}

/**
 * @brief Gets the list of available backends for slice operation
 * @ingroup backend_system
 * @return const char** Array of backend names
 */
DEFINE_GET_BACKENDS_FUNCTION(slice);

/**
 * @brief Gets the number of available backends for slice operation
 * @ingroup backend_system
 * @return size_t Number of available backends
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(slice);

#endif /** NNL2_SLICE_H **/
