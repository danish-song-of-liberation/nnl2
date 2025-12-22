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

#ifdef NNL2_PTHREAD_AVAILABLE

/** @brief
 * Threshold for enabling parallel execution of concat operation
 */
#define NNL2_CONCAT_PARALLEL_THRESHOLD 100000

/** @brief
 * Worker function for parallel concat with same data types
 * 
 ** @param arg 
 * Pointer to concat_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 */
void* nnl2_own_pconcat_same_type(void* arg);

/** @brief
 * Worker function for parallel concat with type conversion
 * 
 ** @param arg 
 * Pointer to concat_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 */
void* nnl2_own_pconcat_convert(void* arg);

/** @brief
 * Optimized function for 1D concatenation
 */
Tensor* nnl2_own_concat_1d(Tensor* tensora, Tensor* tensorb, TensorType winner_type, bool same_type);

/** @brief
 * Optimized function for 2D concatenation along axis 0
 */
Tensor* nnl2_own_concat_2d_axis0(Tensor* tensora, Tensor* tensorb, TensorType winner_type, bool same_type);

/** @brief
 * Optimized function for 2D concatenation along axis 1
 */
Tensor* nnl2_own_concat_2d_axis1(Tensor* tensora, Tensor* tensorb, TensorType winner_type, bool same_type);

/** @brief
 * High-performance parallel implementation of tensor concatenation
 * 
 ** @param tensora 
 * Pointer to first source tensor
 *
 ** @param tensorb
 * Pointer to second source tensor
 *
 ** @param axis
 * Axis along which to concatenate
 * 
 ** @return
 * Pointer to new concatenated tensor, or NULL if error occurs
 *
 ** @details
 * Uses multi-threading with pthread and optimized memory access patterns
 * for maximum performance on modern CPU architectures. Automatically
 * selects optimal strategy based on tensor rank and concatenation axis.
 * 
 ** @note
 * Uses NNL2_NUM_THREADS for parallelization configuration
 * Falls back to naive implementation for small tensors or complex cases
 * 
 ** @warning
 * Requires pthread support
 */
Tensor* nnl2_own_concat(Tensor* tensora, Tensor* tensorb, int axis) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
	int rank = tensora->rank;
	
    // Additional checks at the maximum safety level
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensora, "First tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensorb, "Second tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensora->data, "First tensor data is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensorb->data, "Second tensor data is NULL", NULL);

        if (rank != tensorb->rank) {
            NNL2_ERROR("Ranks are different (%d != %d) in concat", rank, tensorb->rank);
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
    
    TensorType typea = tensora->dtype;
    TensorType typeb = tensorb->dtype;

    TensorType winner_type = MAX(typea, typeb);
    bool same_type = (typea == typeb && typea == winner_type);
    
    size_t total_elements_a = nnl2_product(tensora->shape, rank);
    size_t total_elements_b = nnl2_product(tensorb->shape, rank);
    size_t total_elements = total_elements_a + total_elements_b;
    
    // Fallback to naive implementation for small tensors
    if (total_elements < NNL2_CONCAT_PARALLEL_THRESHOLD) {
        return naive_concat(tensora, tensorb, axis);
    }
    
    // Optimized paths for common cases
    if (rank == 1) {
        return nnl2_own_concat_1d(tensora, tensorb, winner_type, same_type);
    }
    
    if (rank == 2) {
        if (axis == 0) {
            return nnl2_own_concat_2d_axis0(tensora, tensorb, winner_type, same_type);
        } else if (axis == 1) {
            return nnl2_own_concat_2d_axis1(tensora, tensorb, winner_type, same_type);
        }
    }
    
    // General case - use parallel implementation
    int* shape_result = malloc(rank * sizeof(int));
    if (shape_result == NULL) {
        NNL2_ERROR("Memory allocation failed for shape in concat");
        return NULL;
    }
    
    for (int i = 0; i < rank; i++) {
        shape_result[i] = (i == axis) ? (tensora->shape[i] + tensorb->shape[i]) : tensora->shape[i];
    }
    
    Tensor* result = nnl2_empty(shape_result, rank, winner_type);
    if (result == NULL) {
        free(shape_result);
        return NULL;
    }
    
    bool src_a_aligned = NNL2_IS_ALIGNED(tensora->data, NNL2_TENSOR_ALIGNMENT_32);
    bool src_b_aligned = NNL2_IS_ALIGNED(tensorb->data, NNL2_TENSOR_ALIGNMENT_32);
    bool dst_aligned = NNL2_IS_ALIGNED(result->data, NNL2_TENSOR_ALIGNMENT_32);
    bool is_aligned = src_a_aligned && src_b_aligned && dst_aligned;
    
    // Warning for unaligned memory in safety modes
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!is_aligned) {
            NNL2_WARN("In nnl2_own_concat, tensor memory is not aligned to 32 bytes. Performance may be suboptimal");
        }
    #endif
    
    size_t num_threads = NNL2_NUM_THREADS;
    pthread_t threads[num_threads];
    concat_ptask tasks[num_threads];
    
    // Используем готовые strides из структур тензоров (в элементах)
    int32_t* result_strides = result->strides;
    int32_t* a_strides = tensora->strides;
    int32_t* b_strides = tensorb->strides;
    
    // Configure tasks for parallel processing
    size_t chunk = total_elements / num_threads;
    size_t remainder = total_elements % num_threads;
    
    size_t current_start = 0;
    for (size_t i = 0; i < num_threads; i++) {
        size_t current_chunk = chunk + (i < remainder ? 1 : 0);
        
        tasks[i].src_a = tensora->data;
        tasks[i].src_b = tensorb->data;
        tasks[i].dst = result->data;
        tasks[i].start_idx = current_start;
        tasks[i].end_idx = current_start + current_chunk;
        tasks[i].total_elements = total_elements;
        tasks[i].a_axis_size = (size_t)tensora->shape[axis];
        tasks[i].item_size = get_dtype_size(winner_type);
        tasks[i].rank = rank;
        tasks[i].axis = axis;
        tasks[i].result_shape = shape_result;
        tasks[i].result_strides = result_strides;
        tasks[i].a_strides = a_strides;
        tasks[i].b_strides = b_strides;
        tasks[i].type_a = typea;
        tasks[i].type_b = typeb;
        tasks[i].result_type = winner_type;
        tasks[i].aligned = is_aligned;
        tasks[i].same_type = same_type;
        
        current_start += current_chunk;
    }
    
    // Create and run threads
    for (size_t i = 0; i < num_threads; i++) {
        void* (*worker_func)(void*) = tasks[i].same_type ? nnl2_own_pconcat_same_type : nnl2_own_pconcat_convert;
        
        int status = pthread_create(&threads[i], NULL, worker_func, &tasks[i]);
        if(status != 0) {
            NNL2_THREAD_CREATE_ERROR(status, "nnl2_own_concat");
            num_threads = i;
            break;
        }
    }
    
    // Wait for all threads to complete
    for (size_t i = 0; i < num_threads; i++) {
        int join_status = pthread_join(threads[i], NULL);
        if(join_status != 0) {
            NNL2_THREAD_JOIN_ERROR(join_status, "nnl2_own_concat");
        }
    }
    
    free(shape_result);
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return result;
}

// Optimized implementations for common cases

/** @brief
 * Optimized 1D concatenation
 */
Tensor* nnl2_own_concat_1d(Tensor* tensora, Tensor* tensorb, TensorType winner_type, bool same_type) {
    int shape_result[1] = {tensora->shape[0] + tensorb->shape[0]};
    Tensor* result = nnl2_empty(shape_result, 1, winner_type);
    
    if (result == NULL) {
        return NULL;
    }
    
    if (same_type) {
        size_t item_size = get_dtype_size(winner_type);
        size_t size_a = (size_t)tensora->shape[0] * item_size;
        size_t size_b = (size_t)tensorb->shape[0] * item_size;
        
        memcpy(result->data, tensora->data, size_a);
        memcpy((char*)result->data + size_a, tensorb->data, size_b);
    } else {
        // Type conversion needed
        size_t len_a = (size_t)tensora->shape[0];
        size_t len_b = (size_t)tensorb->shape[0];
        
        switch(winner_type) {
            case FLOAT64: {
                double* dst = (double*)result->data;
                for (size_t i = 0; i < len_a; i++) {
                    void* src = (char*)tensora->data + i * get_dtype_size(tensora->dtype);
                    dst[i] = nnl2_convert_to_float64(src, tensora->dtype);
                }
				
                for (size_t i = 0; i < len_b; i++) {
                    void* src = (char*)tensorb->data + i * get_dtype_size(tensorb->dtype);
                    dst[len_a + i] = nnl2_convert_to_float64(src, tensorb->dtype);
                }
				
                break;
            }
			
            case FLOAT32: {
                float* dst = (float*)result->data;
                for (size_t i = 0; i < len_a; i++) {
                    void* src = (char*)tensora->data + i * get_dtype_size(tensora->dtype);
                    dst[i] = nnl2_convert_to_float32(src, tensora->dtype);
                }
				
                for (size_t i = 0; i < len_b; i++) {
                    void* src = (char*)tensorb->data + i * get_dtype_size(tensorb->dtype);
                    dst[len_a + i] = nnl2_convert_to_float32(src, tensorb->dtype);
                }
				
                break;
            }
			
            case INT32: {
                int32_t* dst = (int32_t*)result->data;
                for (size_t i = 0; i < len_a; i++) {
                    void* src = (char*)tensora->data + i * get_dtype_size(tensora->dtype);
                    dst[i] = nnl2_convert_to_int32(src, tensora->dtype);
                }
				
                for (size_t i = 0; i < len_b; i++) {
                    void* src = (char*)tensorb->data + i * get_dtype_size(tensorb->dtype);
                    dst[len_a + i] = nnl2_convert_to_int32(src, tensorb->dtype);
                }
				
                break;
            }
			
            default: {
                nnl2_free_tensor(result);
                return NULL;
			}
        }
    }
    
    return result;
}

/** @brief
 * Optimized 2D concatenation along axis 0
 */
Tensor* nnl2_own_concat_2d_axis0(Tensor* tensora, Tensor* tensorb, TensorType winner_type, bool same_type) {
    int rows_a = tensora->shape[0];
    int rows_b = tensorb->shape[0];
    int cols = tensora->shape[1];
    
    int shape_result[2] = {rows_a + rows_b, cols};
    Tensor* result = nnl2_empty(shape_result, 2, winner_type);
    
    if (result == NULL) {
        return NULL;
    }
    
    if (same_type) {
        size_t item_size = get_dtype_size(winner_type);
        size_t row_size = (size_t)cols * item_size;
        
        memcpy(result->data, tensora->data, (size_t)rows_a * row_size);
        memcpy((char*)result->data + (size_t)rows_a * row_size, tensorb->data, (size_t)rows_b * row_size);
    } else {
        // Type conversion needed
        size_t total_rows = (size_t)(rows_a + rows_b);
        
        switch(winner_type) {
            case FLOAT64: {
                double* dst = (double*)result->data;
                for (size_t i = 0; i < total_rows; i++) {
                    for (size_t j = 0; j < (size_t)cols; j++) {
                        size_t dst_idx = i * cols + j;
                        if (i < (size_t)rows_a) {
                            void* src = (char*)tensora->data + (i * cols + j) * get_dtype_size(tensora->dtype);
                            dst[dst_idx] = nnl2_convert_to_float64(src, tensora->dtype);
                        } else {
                            size_t src_i = i - rows_a;
                            void* src = (char*)tensorb->data + (src_i * cols + j) * get_dtype_size(tensorb->dtype);
                            dst[dst_idx] = nnl2_convert_to_float64(src, tensorb->dtype);
                        }
                    }
                }
				
                break;
            }
			
            case FLOAT32: {
                float* dst = (float*)result->data;
                for (size_t i = 0; i < total_rows; i++) {
                    for (size_t j = 0; j < (size_t)cols; j++) {
                        size_t dst_idx = i * cols + j;
                        if (i < (size_t)rows_a) {
                            void* src = (char*)tensora->data + (i * cols + j) * get_dtype_size(tensora->dtype);
                            dst[dst_idx] = nnl2_convert_to_float32(src, tensora->dtype);
                        } else {
                            size_t src_i = i - rows_a;
                            void* src = (char*)tensorb->data + (src_i * cols + j) * get_dtype_size(tensorb->dtype);
                            dst[dst_idx] = nnl2_convert_to_float32(src, tensorb->dtype);
                        }
                    }
                }
				
                break;
            }
			
            case INT32: {
                int32_t* dst = (int32_t*)result->data;
                for (size_t i = 0; i < total_rows; i++) {
                    for (size_t j = 0; j < (size_t)cols; j++) {
                        size_t dst_idx = i * cols + j;
                        if (i < (size_t)rows_a) {
                            void* src = (char*)tensora->data + (i * cols + j) * get_dtype_size(tensora->dtype);
                            dst[dst_idx] = nnl2_convert_to_int32(src, tensora->dtype);
                        } else {
                            size_t src_i = i - rows_a;
                            void* src = (char*)tensorb->data + (src_i * cols + j) * get_dtype_size(tensorb->dtype);
                            dst[dst_idx] = nnl2_convert_to_int32(src, tensorb->dtype);
                        }
                    }
                }
				
                break;
            }
			
            default: {
                nnl2_free_tensor(result);
                return NULL;
			}
        }
    }
    
    return result;
}

/** @brief
 * Optimized 2D concatenation along axis 1
 */
Tensor* nnl2_own_concat_2d_axis1(Tensor* tensora, Tensor* tensorb, TensorType winner_type, bool same_type) {
    int rows = tensora->shape[0];
    int cols_a = tensora->shape[1];
    int cols_b = tensorb->shape[1];
    
    int shape_result[2] = {rows, cols_a + cols_b};
    Tensor* result = nnl2_empty(shape_result, 2, winner_type);
    
    if (result == NULL) {
        return NULL;
    }
    
    if (same_type) {
        size_t item_size = get_dtype_size(winner_type);
        size_t row_size_a = (size_t)cols_a * item_size;
        size_t row_size_b = (size_t)cols_b * item_size;
        size_t row_size_result = row_size_a + row_size_b;
        
        char* dst = (char*)result->data;
        char* src_a = (char*)tensora->data;
        char* src_b = (char*)tensorb->data;
        
        for (int i = 0; i < rows; i++) {
            memcpy(dst, src_a, row_size_a);
            memcpy(dst + row_size_a, src_b, row_size_b);
            dst += row_size_result;
            src_a += row_size_a;
            src_b += row_size_b;
        }
    } else {
        // Type conversion needed
        switch(winner_type) {
            case FLOAT64: {
                double* dst = (double*)result->data;
                for (size_t i = 0; i < (size_t)rows; i++) {
                    for (size_t j = 0; j < (size_t)(cols_a + cols_b); j++) {
                        size_t dst_idx = i * (cols_a + cols_b) + j;
                        if (j < (size_t)cols_a) {
                            void* src = (char*)tensora->data + (i * cols_a + j) * get_dtype_size(tensora->dtype);
                            dst[dst_idx] = nnl2_convert_to_float64(src, tensora->dtype);
                        } else {
                            size_t src_j = j - cols_a;
                            void* src = (char*)tensorb->data + (i * cols_b + src_j) * get_dtype_size(tensorb->dtype);
                            dst[dst_idx] = nnl2_convert_to_float64(src, tensorb->dtype);
                        }
                    }
                }
				
                break;
            }
			
            case FLOAT32: {
                float* dst = (float*)result->data;
                for (size_t i = 0; i < (size_t)rows; i++) {
                    for (size_t j = 0; j < (size_t)(cols_a + cols_b); j++) {
                        size_t dst_idx = i * (cols_a + cols_b) + j;
                        if (j < (size_t)cols_a) {
                            void* src = (char*)tensora->data + (i * cols_a + j) * get_dtype_size(tensora->dtype);
                            dst[dst_idx] = nnl2_convert_to_float32(src, tensora->dtype);
                        } else {
                            size_t src_j = j - cols_a;
                            void* src = (char*)tensorb->data + (i * cols_b + src_j) * get_dtype_size(tensorb->dtype);
                            dst[dst_idx] = nnl2_convert_to_float32(src, tensorb->dtype);
                        }
                    }
                }
				
                break;
            }
			
            case INT32: {
                int32_t* dst = (int32_t*)result->data;
                for (size_t i = 0; i < (size_t)rows; i++) {
                    for (size_t j = 0; j < (size_t)(cols_a + cols_b); j++) {
                        size_t dst_idx = i * (cols_a + cols_b) + j;
                        if (j < (size_t)cols_a) {
                            void* src = (char*)tensora->data + (i * cols_a + j) * get_dtype_size(tensora->dtype);
                            dst[dst_idx] = nnl2_convert_to_int32(src, tensora->dtype);
                        } else {
                            size_t src_j = j - cols_a;
                            void* src = (char*)tensorb->data + (i * cols_b + src_j) * get_dtype_size(tensorb->dtype);
                            dst[dst_idx] = nnl2_convert_to_int32(src, tensorb->dtype);
                        }
                    }
                }
				
                break;
            }
			
            default: {
                nnl2_free_tensor(result);
                return NULL;
			}
        }
    }
    
    return result;
}

// Worker function implementations

/** @brief
 * Parallel concat worker for same data types
 * 
 ** @see nnl2_own_pconcat_same_type
 **/
void* nnl2_own_pconcat_same_type(void* arg) {
    concat_ptask* task = (concat_ptask*)arg;
    char* src_a = (char*)task->src_a;
    char* src_b = (char*)task->src_b;
    char* dst = (char*)task->dst;
    
    size_t* indices = malloc(task->rank * sizeof(size_t));
    size_t* source_indices = malloc(task->rank * sizeof(size_t));
    
    if (indices == NULL || source_indices == NULL) {
        free(indices);
        free(source_indices);
        return NULL;
    }
    
    for (size_t linear_idx = task->start_idx; linear_idx < task->end_idx; linear_idx++) {
        // Convert linear index to multi-dimensional indices
        size_t temp = linear_idx;
        for (int i = task->rank - 1; i >= 0; i--) {
            indices[i] = temp % task->result_shape[i];
            temp /= task->result_shape[i];
        }
        
        // Calculate source tensor and adjusted indices
        char* source_ptr;
        memcpy(source_indices, indices, task->rank * sizeof(size_t));
        
        if (indices[task->axis] < task->a_axis_size) {
            size_t source_offset = 0;
            for (int i = 0; i < task->rank; i++) {
                source_offset += source_indices[i] * task->a_strides[i];
            }
            source_ptr = src_a + source_offset * task->item_size;
        } else {
            source_indices[task->axis] = indices[task->axis] - task->a_axis_size;
            size_t source_offset = 0;
            for (int i = 0; i < task->rank; i++) {
                source_offset += source_indices[i] * task->b_strides[i];
            }
			
            source_ptr = src_b + source_offset * task->item_size;
        }
        
        // Calculate destination offset
        size_t dest_offset = 0;
        for (int i = 0; i < task->rank; i++) {
            dest_offset += indices[i] * task->result_strides[i];
        }
        
        // Copy data
        char* dest_ptr = dst + dest_offset * task->item_size;
        memcpy(dest_ptr, source_ptr, task->item_size);
    }
    
    free(indices);
    free(source_indices);
    return NULL;
}

/** @brief
 * Parallel concat worker for type conversion
 * 
 ** @see nnl2_own_pconcat_convert
 **/
void* nnl2_own_pconcat_convert(void* arg) {
    concat_ptask* task = (concat_ptask*)arg;
    
    size_t* indices = malloc(task->rank * sizeof(size_t));
    size_t* source_indices = malloc(task->rank * sizeof(size_t));
    
    if (indices == NULL || source_indices == NULL) {
        free(indices);
        free(source_indices);
        return NULL;
    }
    
    for (size_t linear_idx = task->start_idx; linear_idx < task->end_idx; linear_idx++) {
        // Convert linear index to multi-dimensional indices
        size_t temp = linear_idx;
        for (int i = task->rank - 1; i >= 0; i--) {
            indices[i] = temp % task->result_shape[i];
            temp /= task->result_shape[i];
        }
        
        // Calculate destination offset
        size_t dest_offset = 0;
        for (int i = 0; i < task->rank; i++) {
            dest_offset += indices[i] * task->result_strides[i];
        }
        
        // Determine source and convert
        if (indices[task->axis] < task->a_axis_size) {
            size_t source_offset = 0;
            for (int i = 0; i < task->rank; i++) {
                source_offset += indices[i] * task->a_strides[i];
            }
			
            void* src = (char*)task->src_a + source_offset * get_dtype_size(task->type_a);
            
            switch(task->result_type) {
                case FLOAT64: {
                    double* dst = (double*)task->dst;
                    dst[dest_offset] = nnl2_convert_to_float64(src, task->type_a);
                    break;
                }
				
                case FLOAT32: {
                    float* dst = (float*)task->dst;
                    dst[dest_offset] = nnl2_convert_to_float32(src, task->type_a);
                    break;
                }
				
                case INT32: {
                    int32_t* dst = (int32_t*)task->dst;
                    dst[dest_offset] = nnl2_convert_to_int32(src, task->type_a);
                    break;
                }
				
                default: break;
            }
        } else {
            memcpy(source_indices, indices, task->rank * sizeof(size_t));
            source_indices[task->axis] = indices[task->axis] - task->a_axis_size;
            
            size_t source_offset = 0;
            for (int i = 0; i < task->rank; i++) {
                source_offset += source_indices[i] * task->b_strides[i];
            }
			
            void* src = (char*)task->src_b + source_offset * get_dtype_size(task->type_b);
            
            switch(task->result_type) {
                case FLOAT64: {
                    double* dst = (double*)task->dst;
                    dst[dest_offset] = nnl2_convert_to_float64(src, task->type_b);
                    break;
                }
				
                case FLOAT32: {
                    float* dst = (float*)task->dst;
                    dst[dest_offset] = nnl2_convert_to_float32(src, task->type_b);
                    break;
                }
				
                case INT32: {
                    int32_t* dst = (int32_t*)task->dst;
                    dst[dest_offset] = nnl2_convert_to_int32(src, task->type_b);
                    break;
                }
				
                default: break;
            }
        }
    }
    
    free(indices);
    free(source_indices);
    return NULL;
}

#endif

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
	
	#ifdef NNL2_PTHREAD_AVAILABLE
		REGISTER_BACKEND(nnl2_own_concat, nnl2_own, NNL2_OWN_NAME),
	#endif
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
