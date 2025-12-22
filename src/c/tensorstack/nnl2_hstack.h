#ifndef NNL2_HSTACK_H
#define NNL2_HSTACK_H

/** @brief 
 * Performs horizontal stacking of two tensors (naive implementation)
 *
 ** @param tensora
 * Pointer to the first input tensor
 *
 ** @param tensorb
 * Pointer to the second input tensor
 *
 ** @return
 * Pointer to a new tensor containing the horizontally stacked result
 *
 ** @note
 * Tensors must have the same rank and compatible shapes (all dimensions
 * except axis=1 must match)
 *
 ** @see nnl2_empty
 ** @see get_dtype_size()
 ** @see nnl2_convert_to_float64()
 ** @see nnl2_convert_to_float32()
 ** @see nnl2_convert_to_int32()
 **/
Tensor* naive_hstack(Tensor* tensora, Tensor* tensorb) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    if (tensora == NULL || tensorb == NULL) {
        return NULL;
    }
    
    TensorType typea = tensora->dtype;
    TensorType typeb = tensorb->dtype;
    int ranka = tensora->rank;
    int rankb = tensorb->rank;

    // Safety checks
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(ranka != rankb) {
            NNL2_ERROR("Tensors dimensions are different");
            return NULL;
        }

        // Check if all dimensions except the stacking axis are equal
        int stacking_axis = (ranka == 1) ? 0 : 1;
        for(int i = 0; i < ranka; i++) {
            if(i != stacking_axis && tensora->shape[i] != tensorb->shape[i]) {
                NNL2_ERROR("Tensors shapes are incompatible for hstack");
                return NULL;
            }
        }
    #endif

    TensorType winner_type = MAX(typea, typeb);

    // Allocate memory for result shape
    int* shapec = malloc(ranka * sizeof(int));
    if (shapec == NULL) {
        NNL2_ERROR("Memory allocation failed");
        return NULL; 
    }
    
    // Calculate result shape
    if (ranka == 1) {
        // For 1D tensors stack along axis 0
        shapec[0] = tensora->shape[0] + tensorb->shape[0];
    } else {
        // For multi-dimensional tensors stack along axis 1
        for(int i = 0; i < ranka; i++) {
            if(i == 1) {
                shapec[i] = tensora->shape[i] + tensorb->shape[i];
            } else {
                shapec[i] = tensora->shape[i];
            }
        }
    }
    
    Tensor* result = nnl2_empty(shapec, ranka, winner_type);
    free(shapec);
    
    if(result == NULL) {
        NNL2_ERROR("Failed to create result tensor in hstack");
        return NULL;
    }
    
    size_t total_elements = nnl2_product(result->shape, result->rank);
    if(total_elements == 0) {
        return result;
    }

    // Handle 1D tensors
    if(ranka == 1) {
        if(typea == typeb && typea == winner_type) {
            size_t item_size = get_dtype_size(winner_type);
            memcpy(result->data, tensora->data, tensora->shape[0] * item_size);
            memcpy((char*)result->data + tensora->shape[0] * item_size, 
                   tensorb->data, tensorb->shape[0] * item_size);
        } else {
            // Type conversion needed
            switch(winner_type) {
                case FLOAT64: {
                    volatile double* dst = (double*)result->data;

                    // Convert and copy first tensor
                    for(size_t i = 0; i < (size_t)tensora->shape[0]; i++) {
                        void* elem = (char*)tensora->data + i * get_dtype_size(typea);
                        dst[i] = nnl2_convert_to_float64(elem, typea);
                    }
                
                    // Convert and copy second tensor
                    for(size_t i = 0; i < (size_t)tensorb->shape[0]; i++) {
                        void* elem = (char*)tensorb->data + i * get_dtype_size(typeb);
                        dst[tensora->shape[0] + i] = nnl2_convert_to_float64(elem, typeb);
                    }
                    break;
                }
                
                case FLOAT32: {
                    volatile float* dst = (float*)result->data;
                    
                    // Convert and copy first tensor
                    for(size_t i = 0; i < (size_t)tensora->shape[0]; i++) {
                        void* elem = (char*)tensora->data + i * get_dtype_size(typea);
                        dst[i] = nnl2_convert_to_float32(elem, typea);
                    }
                    
                    // Convert and copy second tensor
                    for(size_t i = 0; i < (size_t)tensorb->shape[0]; i++) {
                        void* elem = (char*)tensorb->data + i * get_dtype_size(typeb);
                        dst[tensora->shape[0] + i] = nnl2_convert_to_float32(elem, typeb);
                    }
                    break;
                }
                
                case INT32: {
                    volatile int32_t* dst = (int32_t*)result->data;
                    
                    // Convert and copy first tensor
                    for(size_t i = 0; i < (size_t)tensora->shape[0]; i++) {
                        void* elem = (char*)tensora->data + i * get_dtype_size(typea);
                        dst[i] = nnl2_convert_to_int32(elem, typea);
                    }
                    
                    // Convert and copy second tensor
                    for(size_t i = 0; i < (size_t)tensorb->shape[0]; i++) {
                        void* elem = (char*)tensorb->data + i * get_dtype_size(typeb);
                        dst[tensora->shape[0] + i] = nnl2_convert_to_int32(elem, typeb);
                    }
                    break;
                }
                
                default: {
                    NNL2_TYPE_ERROR(winner_type);
                    nnl2_free_tensor(result);
                    return NULL;
                }
            }
        }
    } else {
        // Handle multi-dimensional tensors
        size_t outer_dim = (size_t)tensora->shape[0];
        size_t inner_dim_a = (size_t)tensora->shape[1];
        size_t inner_dim_b = (size_t)tensorb->shape[1];
        size_t inner_elements_rest = 1;
        
        // Calculate remaining elements (for dimensions > 2)
        if(ranka > 2) {
            inner_elements_rest = nnl2_product(tensora->shape + 2, ranka - 2);
        }
        
        size_t elements_per_row_a = inner_dim_a * inner_elements_rest;
        size_t elements_per_row_b = inner_dim_b * inner_elements_rest;
        size_t elements_per_row_result = elements_per_row_a + elements_per_row_b;
        
        if(typea == typeb && typea == winner_type) {
            size_t item_size = get_dtype_size(winner_type);
            size_t row_size_a = elements_per_row_a * item_size;
            size_t row_size_b = elements_per_row_b * item_size;
            
            char* src_a = tensora->data;
            char* src_b = tensorb->data;
            char* dst = result->data;
            
            // Process each outer dimension
            for(size_t i = 0; i < outer_dim; i++) {
                // Copy slice from first tensor
                memcpy(dst, src_a, row_size_a);
                dst += row_size_a;
                src_a += row_size_a;
                
                // Copy slice from second tensor
                memcpy(dst, src_b, row_size_b);
                dst += row_size_b;
                src_b += row_size_b;
            }
        } else {
            // Type conversion needed for multi-dimensional case
            switch(winner_type) {
                case FLOAT64: {
                    volatile double* dst = (double*)result->data;
                    
                    // Process each outer dimension
                    for(size_t i = 0; i < outer_dim; i++) {
                        size_t base_idx = i * elements_per_row_result;
                        
                        // Convert and copy slice from first tensor
                        for(size_t j = 0; j < elements_per_row_a; j++) {
                            size_t src_idx = i * elements_per_row_a + j;
                            size_t dst_idx = base_idx + j;
                            
                            void* elem = (char*)tensora->data + src_idx * get_dtype_size(typea);
                            dst[dst_idx] = nnl2_convert_to_float64(elem, typea);
                        }
                        
                        // Convert and copy slice from second tensor
                        for(size_t j = 0; j < elements_per_row_b; j++) {
                            size_t src_idx = i * elements_per_row_b + j;
                            size_t dst_idx = base_idx + elements_per_row_a + j;
                            
                            void* elem = (char*)tensorb->data + src_idx * get_dtype_size(typeb);
                            dst[dst_idx] = nnl2_convert_to_float64(elem, typeb);
                        }
                    }
                    break;
                }
                
                case FLOAT32: {
                    volatile float* dst = (float*)result->data;
                    
                    // Process each outer dimension
                    for(size_t i = 0; i < outer_dim; i++) {
                        size_t base_idx = i * elements_per_row_result;
                        
                        // Convert and copy slice from first tensor
                        for(size_t j = 0; j < elements_per_row_a; j++) {
                            size_t src_idx = i * elements_per_row_a + j;
                            size_t dst_idx = base_idx + j;
                            
                            void* elem = (char*)tensora->data + src_idx * get_dtype_size(typea);
                            dst[dst_idx] = nnl2_convert_to_float32(elem, typea);
                        }
                        
                        // Convert and copy slice from second tensor
                        for(size_t j = 0; j < elements_per_row_b; j++) {
                            size_t src_idx = i * elements_per_row_b + j;
                            size_t dst_idx = base_idx + elements_per_row_a + j;
                            
                            void* elem = (char*)tensorb->data + src_idx * get_dtype_size(typeb);
                            dst[dst_idx] = nnl2_convert_to_float32(elem, typeb);
                        }
                    }
                    break;
                }
                
                case INT32: {
                    volatile int32_t* dst = (int32_t*)result->data;
                    
                    // Process each outer dimension
                    for(size_t i = 0; i < outer_dim; i++) {
                        size_t base_idx = i * elements_per_row_result;
                        
                        // Convert and copy slice from first tensor
                        for(size_t j = 0; j < elements_per_row_a; j++) {
                            size_t src_idx = i * elements_per_row_a + j;
                            size_t dst_idx = base_idx + j;
                            
                            void* elem = (char*)tensora->data + src_idx * get_dtype_size(typea);
                            dst[dst_idx] = nnl2_convert_to_int32(elem, typea);
                        }
                        
                        // Convert and copy slice from second tensor
                        for(size_t j = 0; j < elements_per_row_b; j++) {
                            size_t src_idx = i * elements_per_row_b + j;
                            size_t dst_idx = base_idx + elements_per_row_a + j;
                            
                            void* elem = (char*)tensorb->data + src_idx * get_dtype_size(typeb);
                            dst[dst_idx] = nnl2_convert_to_int32(elem, typeb);
                        }
                    }
                    break;
                }
                
                default: {
                    NNL2_TYPE_ERROR(winner_type);
                    nnl2_free_tensor(result);
                    return NULL;
                }
            }
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return result;
}

#ifdef NNL2_PTHREAD_AVAILABLE

/** @brief
 * Threshold for enabling parallel execution of hstack operation
 */
#define NNL2_HSTACK_PARALLEL_THRESHOLD 100000

/** @brief
 * Worker function for parallel hstack with same data types
 * 
 ** @param arg 
 * Pointer to hstack_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 */
void* nnl2_own_phstack_same_type(void* arg);

/** @brief
 * Worker function for parallel hstack with type conversion
 * 
 ** @param arg 
 * Pointer to hstack_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 */
void* nnl2_own_phstack_convert(void* arg);

/** @brief
 * High-performance parallel implementation of horizontal tensor stacking
 * 
 ** @param tensora 
 * Pointer to first source tensor
 *
 ** @param tensorb
 * Pointer to second source tensor
 * 
 ** @return
 * Pointer to new stacked tensor, or NULL if error occurs
 *
 ** @details
 * Uses multi-threading with pthread and AVX256 vectorization for optimal
 * performance on modern CPU architectures. Automatically selects between
 * same-type copying and type conversion paths.
 * 
 ** @note
 * Uses NNL2_NUM_THREADS for parallelization configuration
 * Falls back to naive implementation for small tensors or complex cases
 * 
 ** @warning
 * Requires pthread support
 */
Tensor* nnl2_own_hstack(Tensor* tensora, Tensor* tensorb) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks at the maximum safety level
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensora, "First tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensorb, "Second tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensora->data, "First tensor data is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensorb->data, "Second tensor data is NULL", NULL);
        
        int32_t ranka = tensora->rank;
        int32_t rankb = tensorb->rank;
        if(ranka != rankb) {
            NNL2_ERROR("Tensors dimensions are different");
            return NULL;
        }

        // Check if all dimensions except the stacking axis are equal
        int stacking_axis = (ranka == 1) ? 0 : 1;
        for(int i = 0; i < ranka; i++) {
            if(i != stacking_axis && tensora->shape[i] != tensorb->shape[i]) {
                NNL2_ERROR("Tensors shapes are incompatible for hstack");
                return NULL;
            }
        }
    #endif
    
    TensorType typea = tensora->dtype;
    TensorType typeb = tensorb->dtype;

    TensorType winner_type = MAX(typea, typeb);
    bool same_type = (typea == typeb && typea == winner_type);

    // Allocate memory for result shape
    int* shapec = malloc(ranka * sizeof(int));
    if (shapec == NULL) {
        NNL2_ERROR("Memory allocation failed");
        return NULL; 
    }
    
    // Calculate result shape
    if (ranka == 1) {
        shapec[0] = tensora->shape[0] + tensorb->shape[0];
    } else {
        for(int i = 0; i < ranka; i++) {
            if(i == 1) {
                shapec[i] = tensora->shape[i] + tensorb->shape[i];
            } else {
                shapec[i] = tensora->shape[i];
            }
        }
    }
    
    Tensor* result = nnl2_empty(shapec, ranka, winner_type);
    free(shapec);
    
    if(result == NULL) {
        NNL2_ERROR("Failed to create result tensor in hstack");
        return NULL;
    }
    
    size_t total_elements = nnl2_product(result->shape, result->rank);
    if(total_elements == 0) {
        return result;
    }

    // Fallback to naive implementation for small tensors or 1D with type conversion
    if (total_elements < NNL2_HSTACK_PARALLEL_THRESHOLD || 
        (ranka == 1 && !same_type)) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return naive_hstack(tensora, tensorb);
    }

    bool src_a_aligned = NNL2_IS_ALIGNED(tensora->data, NNL2_TENSOR_ALIGNMENT_32);
    bool src_b_aligned = NNL2_IS_ALIGNED(tensorb->data, NNL2_TENSOR_ALIGNMENT_32);
    bool dst_aligned = NNL2_IS_ALIGNED(result->data, NNL2_TENSOR_ALIGNMENT_32);
    bool is_aligned = src_a_aligned && src_b_aligned && dst_aligned;
    
    // Warning for unaligned memory in safety modes
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!is_aligned) {
            NNL2_WARN("In nnl2_own_hstack, tensor memory is not aligned to 32 bytes. Performance may be suboptimal");
        }
    #endif
    
    size_t num_threads = NNL2_NUM_THREADS;
    pthread_t threads[num_threads];
    hstack_ptask tasks[num_threads];
    
    // Configure common task parameters
    for (size_t i = 0; i < num_threads; i++) {
        tasks[i].src_a = tensora->data;
        tasks[i].src_b = tensorb->data;
        tasks[i].dst = result->data;
        tasks[i].type_a = typea;
        tasks[i].type_b = typeb;
        tasks[i].result_type = winner_type;
        tasks[i].aligned = is_aligned;
        tasks[i].same_type = same_type;
    }
    
    if (ranka == 1) {
        // Parallelize by splitting both tensors
        size_t total_len = (size_t)(tensora->shape[0] + tensorb->shape[0]);
        size_t chunk = total_len / num_threads;
        size_t remainder = total_len % num_threads;
        
        size_t current_start = 0;
        for (size_t i = 0; i < num_threads; i++) {
            size_t current_chunk = chunk + (i < remainder ? 1 : 0);
            
            tasks[i].start_idx = current_start;
            tasks[i].end_idx = current_start + current_chunk;
            tasks[i].elements_per_row_a = (size_t)tensora->shape[0];
            tasks[i].elements_per_row_b = (size_t)tensorb->shape[0];
            tasks[i].elements_per_row_result = total_len;
            
            current_start += current_chunk;
        }
    } else {
        // Parallelize by outer dimension
        size_t outer_dim = (size_t)tensora->shape[0];
        size_t inner_dim_a = (size_t)tensora->shape[1];
        size_t inner_dim_b = (size_t)tensorb->shape[1];
        size_t inner_elements_rest = 1;
        
        if(ranka > 2) {
            inner_elements_rest = nnl2_product(tensora->shape + 2, ranka - 2);
        }
        
        size_t elements_per_row_a = inner_dim_a * inner_elements_rest;
        size_t elements_per_row_b = inner_dim_b * inner_elements_rest;
        size_t elements_per_row_result = elements_per_row_a + elements_per_row_b;
        
        size_t chunk = outer_dim / num_threads;
        size_t remainder = outer_dim % num_threads;
        
        size_t current_start = 0;
        for (size_t i = 0; i < num_threads; i++) {
            size_t current_chunk = chunk + (i < remainder ? 1 : 0);
            
            tasks[i].start_idx = current_start;
            tasks[i].end_idx = current_start + current_chunk;
            tasks[i].elements_per_row_a = elements_per_row_a;
            tasks[i].elements_per_row_b = elements_per_row_b;
            tasks[i].elements_per_row_result = elements_per_row_result;
            
            current_start += current_chunk;
        }
    }
    
    // Create and run threads
    for (size_t i = 0; i < num_threads; i++) {
        void* (*worker_func)(void*) = tasks[i].same_type ? 
            nnl2_own_phstack_same_type : nnl2_own_phstack_convert;
        
        int status = pthread_create(&threads[i], NULL, worker_func, &tasks[i]);
        if(status != 0) {
            NNL2_THREAD_CREATE_ERROR(status, "nnl2_own_hstack");
            num_threads = i;
            break;
        }
    }
    
    // Wait for all threads to complete
    for (size_t i = 0; i < num_threads; i++) {
        int join_status = pthread_join(threads[i], NULL);
        if(join_status != 0) {
            NNL2_THREAD_JOIN_ERROR(join_status, "nnl2_own_hstack");
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return result;
}

// Worker function implementations

/** @brief
 * Parallel hstack worker for same data types
 * 
 ** @see nnl2_own_phstack_same_type
 **/
void* nnl2_own_phstack_same_type(void* arg) {
    hstack_ptask* task = (hstack_ptask*)arg;
    char* src_a = (char*)task->src_a;
    char* src_b = (char*)task->src_b;
    char* dst = (char*)task->dst;
    
    size_t item_size = get_dtype_size(task->result_type);
    size_t row_size_a = task->elements_per_row_a * item_size;
    size_t row_size_b = task->elements_per_row_b * item_size;
    size_t row_size_result = task->elements_per_row_result * item_size;
    
    // Check if we're processing 1D or multi-dimensional
    if (task->elements_per_row_result == task->end_idx - task->start_idx) {
        // Direct memory copy
        size_t start_byte = task->start_idx * item_size;
        size_t end_byte = task->end_idx * item_size;
        size_t split_byte = task->elements_per_row_a * item_size;
        
        if (end_byte <= split_byte) {
            // Entire chunk in first tensor
            memcpy(dst + start_byte, src_a + start_byte, end_byte - start_byte);
        } else if (start_byte >= split_byte) {
            // Entire chunk in second tensor
            size_t src_start = start_byte - split_byte;
            size_t src_end = end_byte - split_byte;
            memcpy(dst + start_byte, src_b + src_start, src_end - src_start);
        } else {
            // Chunk spans both tensors
            size_t first_part = split_byte - start_byte;
            memcpy(dst + start_byte, src_a + start_byte, first_part);
            memcpy(dst + split_byte, src_b, end_byte - split_byte);
        }
    } else {
        // Process by outer dimension
        for (size_t i = task->start_idx; i < task->end_idx; i++) {
            size_t dst_offset = i * row_size_result;
            size_t src_a_offset = i * row_size_a;
            size_t src_b_offset = i * row_size_b;
            
            memcpy(dst + dst_offset, src_a + src_a_offset, row_size_a);
            memcpy(dst + dst_offset + row_size_a, src_b + src_b_offset, row_size_b);
        }
    }
    
    return NULL;
}

/** @brief
 * Parallel hstack worker for type conversion
 * 
 ** @see nnl2_own_phstack_convert
 **/
void* nnl2_own_phstack_convert(void* arg) {
    hstack_ptask* task = (hstack_ptask*)arg;
    
    // Multi-dimensional case with type conversion
    for (size_t i = task->start_idx; i < task->end_idx; i++) {
        size_t base_idx = i * task->elements_per_row_result;
        
        // Convert and copy slice from first tensor
        for (size_t j = 0; j < task->elements_per_row_a; j++) {
            size_t src_idx = i * task->elements_per_row_a + j;
            size_t dst_idx = base_idx + j;
            
            void* elem = (char*)task->src_a + src_idx * get_dtype_size(task->type_a);
            
            switch(task->result_type) {
                case FLOAT64: {
                    double* dst = (double*)task->dst;
                    dst[dst_idx] = nnl2_convert_to_float64(elem, task->type_a);
                    break;
                }
				
                case FLOAT32: {
                    float* dst = (float*)task->dst;
                    dst[dst_idx] = nnl2_convert_to_float32(elem, task->type_a);
                    break;
                }
				
                case INT32: {
                    int32_t* dst = (int32_t*)task->dst;
                    dst[dst_idx] = nnl2_convert_to_int32(elem, task->type_a);
                    break;
                }
				
                default: break;
            }
        }
        
        // Convert and copy slice from second tensor
        for (size_t j = 0; j < task->elements_per_row_b; j++) {
            size_t src_idx = i * task->elements_per_row_b + j;
            size_t dst_idx = base_idx + task->elements_per_row_a + j;
            
            void* elem = (char*)task->src_b + src_idx * get_dtype_size(task->type_b);
            
            switch(task->result_type) {
                case FLOAT64: {
                    double* dst = (double*)task->dst;
                    dst[dst_idx] = nnl2_convert_to_float64(elem, task->type_b);
                    break;
                }
				
                case FLOAT32: {
                    float* dst = (float*)task->dst;
                    dst[dst_idx] = nnl2_convert_to_float32(elem, task->type_b);
                    break;
                }
				
                case INT32: {
                    int32_t* dst = (int32_t*)task->dst;
                    dst[dst_idx] = nnl2_convert_to_int32(elem, task->type_b);
                    break;
                }
				
                default:
                    break;
            }
        }
    }
    
    return NULL;
}

#endif

/**
 * @ingroup backend_system
 * @brief Backend implementations for hstack operation
 * @details
 * Array follows the common backend registration pattern for horizontal stacking operations.
 * Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for horizontal tensor concatenation
 * 
 * @see nnl2_naive
 * @see naive_hstack
 */
Implementation hstack_backends[] = {
    REGISTER_BACKEND(naive_hstack, nnl2_naive, NAIVE_BACKEND_NAME),
    
    #ifdef NNL2_PTHREAD_AVAILABLE
        REGISTER_BACKEND(nnl2_own_hstack, nnl2_own, NNL2_OWN_NAME),
    #endif
};    

/**
 * @brief Function pointer for hstack operation
 * @ingroup backend_system 
 */
hstackfn hstack;

/** 
 * @brief Makes the hstack backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(hstack);

/** 
 * @brief Sets the backend for hstack operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for horizontal stacking
 * @see ESET_BACKEND_BY_NAME
 */
void set_hstack_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(hstack_backends, hstack, backend_name, current_backend(hstack));
}

/** 
 * @brief Gets the name of the active backend for hstack operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_hstack_backend() {
    return current_backend(hstack);
}

/** 
 * @brief Function declaration for getting all available hstack backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(hstack);

/**
 * @brief Function declaration for getting the number of available hstack backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 * @details
 * Returns the total number of registered horizontal stacking backend implementations.
 * Useful for iterating through available backends.
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(hstack);

#endif /** NNL2_HSTACK_H **/
