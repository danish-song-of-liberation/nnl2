#ifndef NNL2_CONCAT_H
#define NNL2_CONCAT_H

/** @brief
 * Performs concatenation of two tensors along specified axis (naive implementation)
 * Supports different data types with automatic type promotion
 *
 ** @details 
 * This function performs concatenation of two tensors along the given axis
 * The function supports automatic type promotion - if input tensors have
 * different data types, the result will use the "winner" type (higher precision)
 *
 ** @param tensora 
 * Pointer to the first input tensor
 *
 ** @param tensorb
 * Pointer to the second input tensor
 *
 ** @param axis
 * Axis along which to concatenate (0-based)
 *
 ** @return 
 * Pointer to a new tensor containing the concatenated result
 *
 ** @example
 * Tensor* a = ones(shape_a, rank, FLOAT32);
 * Tensor* b = ones(shape_b, rank, FLOAT32);
 * Tensor* result = naive_concat(a, b, 1); // Concatenate along axis 1
 * if (result != NULL) {
 *     // Use concatenated result
 *     nnl2_free_tensor(result);
 * }
 *
 * @retval NULL If ranks of input tensors does not match
 * @retval NULL If axis is out of valid range
 * @retval NULL If non-concatenation dimensions have incompatible shapes
 * @retval NULL If memory allocation fails
 * @retval NULL If unsupported data type encountered
 *
 ** @warning
 * DRY final boss 
 *
 ** @see get_dtype_size()
 ** @see nnl2_empty
 ** @see nnl2_convert_to_float64()
 ** @see nnl2_convert_to_float32()
 ** @see nnl2_convert_to_int32()
 ** @see nnl2_free_tensor()
 **/
Tensor* naive_concat(Tensor* tensora, Tensor* tensorb, int axis) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    TensorType typea = tensora->dtype;
    TensorType typeb = tensorb->dtype;
	
    int rank = tensora->rank;
	
    TensorType winner_type = MAX(typea, typeb);
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if (tensora->rank != tensorb->rank) {
            NNL2_ERROR("Ranks are different (%d != %d) in concat", tensora->rank, tensorb->rank);
            return NULL;
        }
		
        if (axis < 0 || axis >= rank) {
            NNL2_ERROR("Invalid axis %d (must be 0-%d) in concat", axis, rank - 1);
            return NULL;
        }
		
        for (int i = 0; i < rank; i++) {
            if (i != axis && tensora->shape[i] != tensorb->shape[i]) {
                NNL2_ERROR("Incompatible shapes along axis %d (%d != %d) in concat", i, tensora->shape[i], tensorb->shape[i]);
                return NULL;
            }
        }
    #endif
    
    // Calculate result shape
    int* shape_result = malloc(rank * sizeof(int));
    if (shape_result == NULL) {
        NNL2_ERROR("Memory allocation failed for shape in concat");
        return NULL;
    }
    
    for (int i = 0; i < rank; i++) {
        shape_result[i] = (i == axis) ? (tensora->shape[i] + tensorb->shape[i]) : tensora->shape[i];
    }
    
    Tensor* result = nnl2_empty(shape_result, rank, winner_type);
    free(shape_result);
    
    if (result == NULL) {
        return NULL;
    }
    
    size_t item_size = get_dtype_size(winner_type);
    size_t a_axis_size = tensora->shape[axis];
   
    size_t total_elements = 1;
    for (int i = 0; i < rank; i++) {
        total_elements *= result->shape[i];
    }
    
    // Create index array
    size_t* indices = malloc(rank * sizeof(size_t));
    if (indices == NULL) {
        NNL2_ERROR("Memory allocation failed for indices in concat");
        nnl2_free_tensor(result);
        return NULL;
    }
    
    if (typea == typeb && typea == winner_type) {
        // Fast path: same types
        char* c_data = (char*)result->data;
        
        for (size_t linear_idx = 0; linear_idx < total_elements; linear_idx++) {
            // Convert linear index to multi-dimensional indices
            size_t temp = linear_idx;
            for (int i = rank - 1; i >= 0; i--) {
                indices[i] = temp % result->shape[i];
                temp /= result->shape[i];
            }
            
            // Calculate source tensor and adjusted indices
            Tensor* source_tensor;
            size_t* source_indices = malloc(rank * sizeof(size_t));
            memcpy(source_indices, indices, rank * sizeof(size_t));
            
            if (indices[axis] < a_axis_size) {
                source_tensor = tensora;
                // Use original indices for tensor A
            } else {
                source_tensor = tensorb;
                source_indices[axis] = indices[axis] - a_axis_size; // Adjust index for tensor B
            }
            
            // Calculate source offset
            size_t source_offset = 0;
            for (int i = 0; i < rank; i++) {
                source_offset += source_indices[i] * source_tensor->strides[i];
            }
            
            // Calculate destination offset
            size_t dest_offset = 0;
            for (int i = 0; i < rank; i++) {
                dest_offset += indices[i] * result->strides[i];
            }
            
            // Copy data
            char* source_ptr = (char*)source_tensor->data + source_offset * item_size;
            char* dest_ptr = c_data + dest_offset * item_size;
            memcpy(dest_ptr, source_ptr, item_size);
            
            free(source_indices);
        }
    } else {
        // Type conversion path
        switch(winner_type) {
            case FLOAT64: {
                double* dst = (double*)result->data;
                
                for (size_t linear_idx = 0; linear_idx < total_elements; linear_idx++) {
                    // Convert linear index to multi-dimensional indices
                    size_t temp = linear_idx;
                    for (int i = rank - 1; i >= 0; i--) {
                        indices[i] = temp % result->shape[i];
                        temp /= result->shape[i];
                    }
                    
                    // Determine source and convert
                    if (indices[axis] < a_axis_size) {
                        size_t source_offset = 0;
                        for (int i = 0; i < rank; i++) {
                            source_offset += indices[i] * tensora->strides[i];
                        }
						
                        void* src = (char*)tensora->data + source_offset * get_dtype_size(typea);
                        
                        size_t dest_offset = 0;
                        for (int i = 0; i < rank; i++) {
                            dest_offset += indices[i] * result->strides[i];
                        }
                        dst[dest_offset] = nnl2_convert_to_float64(src, typea);
                    } else {
                        size_t source_indices[rank];
                        memcpy(source_indices, indices, rank * sizeof(size_t));
                        source_indices[axis] = indices[axis] - a_axis_size;
                        
                        size_t source_offset = 0;
                        for (int i = 0; i < rank; i++) {
                            source_offset += source_indices[i] * tensorb->strides[i];
                        }
						
                        void* src = (char*)tensorb->data + source_offset * get_dtype_size(typeb);
                        
                        size_t dest_offset = 0;
                        for (int i = 0; i < rank; i++) {
                            dest_offset += indices[i] * result->strides[i];
                        }
                        dst[dest_offset] = nnl2_convert_to_float64(src, typeb);
                    }
                }
                break;
            }
            
            case FLOAT32: {
                float* dst = (float*)result->data;
                
                for (size_t linear_idx = 0; linear_idx < total_elements; linear_idx++) {
                    size_t temp = linear_idx;
					
                    for (int i = rank - 1; i >= 0; i--) {
                        indices[i] = temp % result->shape[i];
                        temp /= result->shape[i];
                    }
                    
                    if (indices[axis] < a_axis_size) {
                        size_t source_offset = 0;
                        for (int i = 0; i < rank; i++) {
                            source_offset += indices[i] * tensora->strides[i];
                        }
						
                        void* src = (char*)tensora->data + source_offset * get_dtype_size(typea);
                        
                        size_t dest_offset = 0;
                        for (int i = 0; i < rank; i++) {
                            dest_offset += indices[i] * result->strides[i];
                        }
						
                        dst[dest_offset] = nnl2_convert_to_float32(src, typea);
                    } else {
                        size_t source_indices[rank];
                        memcpy(source_indices, indices, rank * sizeof(size_t));
                        source_indices[axis] = indices[axis] - a_axis_size;
                        
                        size_t source_offset = 0;
                        for (int i = 0; i < rank; i++) {
                            source_offset += source_indices[i] * tensorb->strides[i];
                        }
						
                        void* src = (char*)tensorb->data + source_offset * get_dtype_size(typeb);
                        
                        size_t dest_offset = 0;
                        for (int i = 0; i < rank; i++) {
                            dest_offset += indices[i] * result->strides[i];
                        }
						
                        dst[dest_offset] = nnl2_convert_to_float32(src, typeb);
                    }
                }
                break;
            }
            
            case INT32: {
                int32_t* dst = (int32_t*)result->data;
                
                for (size_t linear_idx = 0; linear_idx < total_elements; linear_idx++) {
                    size_t temp = linear_idx;
                    for (int i = rank - 1; i >= 0; i--) {
                        indices[i] = temp % result->shape[i];
                        temp /= result->shape[i];
                    }
                    
                    if (indices[axis] < a_axis_size) {
                        size_t source_offset = 0;
                        for (int i = 0; i < rank; i++) {
                            source_offset += indices[i] * tensora->strides[i];
                        }
						
                        void* src = (char*)tensora->data + source_offset * get_dtype_size(typea);
                        
                        size_t dest_offset = 0;
                        for (int i = 0; i < rank; i++) {
                            dest_offset += indices[i] * result->strides[i];
                        }
                        dst[dest_offset] = nnl2_convert_to_int32(src, typea);
                    } else {
                        size_t source_indices[rank];
                        memcpy(source_indices, indices, rank * sizeof(size_t));
                        source_indices[axis] = indices[axis] - a_axis_size;
                        
                        size_t source_offset = 0;
                        for (int i = 0; i < rank; i++) {
                            source_offset += source_indices[i] * tensorb->strides[i];
                        }
						
                        void* src = (char*)tensorb->data + source_offset * get_dtype_size(typeb);
                        
                        size_t dest_offset = 0;
                        for (int i = 0; i < rank; i++) {
                            dest_offset += indices[i] * result->strides[i];
                        }
                        dst[dest_offset] = nnl2_convert_to_int32(src, typeb);
                    }
                }
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(winner_type);
                nnl2_free_tensor(result);
                free(indices);
                return NULL;
            }
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
 * @brief Backend implementations for concat operation
 * @details
 * Array follows the common backend registration pattern for concatenation 
 * operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for tensor concatenation
 * 
 * @see nnl2_naive
 * @see naive_concat
 */
Implementation concat_backends[] = {
	REGISTER_BACKEND(naive_concat, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for concat operation
 * @ingroup backend_system 
 */
concatfn nnl2_concat;

/** 
 * @brief Makes the concat backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(concat);

/** 
 * @brief Sets the backend for concat operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for concat
 * @see ESET_BACKEND_BY_NAME
 */
void set_concat_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(concat_backends, nnl2_concat, backend_name, current_backend(concat));
}

/** 
 * @brief Gets the name of the active backend for concat operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_concat_backend() {
	return current_backend(concat);
}

/** 
 * @brief Function declaration for getting all available concat backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(concat);

/**
 * @brief Function declaration for getting the number of available concat backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(concat);

#endif /** NNL2_CONCAT_H **/
