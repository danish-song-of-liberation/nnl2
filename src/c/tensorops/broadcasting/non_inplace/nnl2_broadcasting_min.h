#ifndef NNL2_BROADCASTING_MIN_H
#define NNL2_BROADCASTING_MIN_H

/** @brief
 * Performs element-wise minimum with broadcasting support
 *
 ** @param x
 * First tensor
 *
 ** @param y
 * Second tensor
 * 
 ** @return
 * New tensor containing the result of minimum operation
 *
 ** @note
 * Contains type conversion
 */
Tensor* naive_min_broadcasting(Tensor* x, Tensor* y) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks for max safety level 
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(x, "X tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(y, "Y tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(x->shape, "X shape is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(y->shape, "Y shape is NULL", NULL);
    #endif
 
    size_t numel_x = product(x->shape, x->rank);
    size_t numel_y = product(y->shape, y->rank);
    
    // Getting the tensor data types
    TensorType x_dtype = x->dtype;
    TensorType y_dtype = y->dtype;
    
    TensorType winner_in_the_type_hierarchy = MAX(x_dtype, y_dtype);
    
    // Сreating a resultant tensor
    Tensor* result = nnl2_empty(x->shape, x->rank, winner_in_the_type_hierarchy);

    if((numel_x % numel_y) == 0) {
        if(x_dtype == y_dtype) {
            // Handling the case where the data types match (more efficiently)
            switch(x_dtype) {
                case FLOAT64: {
                    double* cast_x_data = (double*)x->data;
                    double* cast_y_data = (double*)y->data;
                    double* cast_result_data = (double*)result->data;

                    for(size_t i = 0; i < (numel_x / numel_y); i++) {
                        for(size_t j = 0; j < numel_y; j++) {
                            cast_result_data[i * numel_y + j] = MIN(cast_x_data[i * numel_y + j], cast_y_data[j]);
                        }
                    }

                    break;
                }

                case FLOAT32: {
                    float* cast_x_data = (float*)x->data;
                    float* cast_y_data = (float*)y->data;
                    float* cast_result_data = (float*)result->data;

                    for(size_t i = 0; i < (numel_x / numel_y); i++) {
                        for(size_t j = 0; j < numel_y; j++) {
                            cast_result_data[i * numel_y + j] = MIN(cast_x_data[i * numel_y + j], cast_y_data[j]);
                        }
                    }

                    break;
                }

                case INT32: {
                    int32_t* cast_x_data = (int32_t*)x->data;
                    int32_t* cast_y_data = (int32_t*)y->data;
                    int32_t* cast_result_data = (int32_t*)result->data;

                    for(size_t i = 0; i < (numel_x / numel_y); i++) {
                        for(size_t j = 0; j < numel_y; j++) {
                            cast_result_data[i * numel_y + j] = MIN(cast_x_data[i * numel_y + j], cast_y_data[j]);
                        }
                    }

                    break;
                }

                default: {
                    NNL2_TYPE_ERROR(x_dtype);
                    return NULL;
                }
            }
            
            #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                NNL2_FUNC_EXIT();
            #endif
            
            return result;
        } else {
            // Handling a case with different data types (conversion required)
            size_t x_step = get_dtype_size(x_dtype);
            size_t y_step = get_dtype_size(y_dtype);
            
            switch(winner_in_the_type_hierarchy) {
                case FLOAT64: {
                    double* cast_data_result = (double*)result->data;
                    
                    char* cast_x_data = (char*)x->data;
                    char* cast_y_data =  (char*)y->data;
                    
                    for(size_t i = 0; i < (numel_x / numel_y); i++) {
                        for(size_t j = 0; j < numel_y; j++) {
                            void* elem_x = cast_x_data + (i * numel_y + j) * x_step;
                            void* elem_y = cast_y_data + j * y_step; 
                            
                            double x_val = nnl2_convert_to_float64(elem_x, x_dtype);
                            double y_val = nnl2_convert_to_float64(elem_y, y_dtype);
                            cast_data_result[i * numel_y + j] = MIN(x_val, y_val);
                        }
                    }
                    
                    break;
                }
                
                case FLOAT32: {
                    float* cast_data_result = (float*)result->data;
                    
                    char* cast_x_data = (char*)x->data;
                    char* cast_y_data =  (char*)y->data;
                    
                    for(size_t i = 0; i < (numel_x / numel_y); i++) {
                        for(size_t j = 0; j < numel_y; j++) {
                            void* elem_x = cast_x_data + (i * numel_y + j) * x_step;
                            void* elem_y = cast_y_data + j * y_step;
                            
                            float x_val = nnl2_convert_to_float32(elem_x, x_dtype);
                            float y_val = nnl2_convert_to_float32(elem_y, y_dtype);
                            cast_data_result[i * numel_y + j] = MIN(x_val, y_val);
                        }
                    }
                    
                    break;
                }
                
                case INT32: {
                    int32_t* cast_data_result = (int32_t*)result->data;
                    
                    char* cast_x_data = (char*)x->data;
                    char* cast_y_data =  (char*)y->data;
                    
                    for(size_t i = 0; i < (numel_x / numel_y); i++) {                    
                        for(size_t j = 0; j < numel_y; j++) {
                            void* elem_x = cast_x_data + (i * numel_y + j) * x_step;
                            void* elem_y = cast_y_data + j * y_step;
                        
                            int32_t x_val = nnl2_convert_to_int32(elem_x, x_dtype);
                            int32_t y_val = nnl2_convert_to_int32(elem_y, y_dtype);
                            cast_data_result[i * numel_y + j] = MIN(x_val, y_val);
                        }
                    }
                    
                    break;
                }
                
                default: {
                    NNL2_TYPE_ERROR(winner_in_the_type_hierarchy);
                    nnl2_free_tensor(result);
                    return NULL;
                }
            }
            
            #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                NNL2_FUNC_EXIT();
            #endif
            
            return result;
        }
    } else {
        NNL2_ERROR("Cannot broadcast y tensor");
        return NULL;
    }
}

#ifdef NNL2_PTHREAD_AVAILABLE

/** @brief
 * Threshold for enabling parallel execution of broadcasting minimum operation
 */
#define NNL2_MIN_BROADCASTING_PARALLEL_THRESHOLD 100000

/** @brief
 * Worker function for parallel double precision broadcasting minimum operation
 * 
 ** @param arg 
 * Pointer to min_broadcasting_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 * 
 ** @details
 * Processes FLOAT64 data using AVX256 instructions with cache prefetching
 * for optimal memory access patterns in broadcasting operation
 */
void* nnl2_own_pmin_broadcasting_float64(void* arg);

/** @brief
 * Worker function for parallel single precision broadcasting minimum operation
 * 
 ** @param arg 
 * Pointer to min_broadcasting_ptask structure containing thread parameters  
 * 
 ** @return 
 * NULL (for pthread API compatibility)
 * 
 ** @see nnl2_own_pmin_broadcasting_float64
 **/
void* nnl2_own_pmin_broadcasting_float32(void* arg);

/** @brief
 * Worker function for parallel integer broadcasting minimum operation
 * 
 ** @param arg 
 * Pointer to min_broadcasting_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 * 
 ** @see nnl2_own_pmin_broadcasting_float64
 **/
void* nnl2_own_pmin_broadcasting_int32(void* arg);

/** @brief
 * High-performance parallel implementation of broadcasting element-wise minimum operation
 * 
 ** @param x 
 * Pointer to the first input tensor
 *
 ** @param y 
 * Pointer to the second input tensor (broadcasted)
 * 
 ** @return
 * Pointer to new tensor containing element-wise minimum values with broadcasting
 *
 ** @details
 * Combines AVX256 vectorization, multi-threading with pthread, and cache
 * prefetching for maximum performance on modern CPU architectures.
 * Optimized for broadcasting patterns with efficient memory access.
 * 
 ** @note
 * Uses NNL2_NUM_THREADS for parallelization configuration
 * Falls back to naive implementation for small tensors, mixed types, or complex broadcasting
 * 
 ** @warning
 * Requires pthread support and AVX256 capable CPU
 * Only supports simple broadcasting patterns where numel_x % numel_y == 0
 */
Tensor* nnl2_own_min_broadcasting(const Tensor* x, const Tensor* y) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks for max safety level 
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(x, "X tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(y, "Y tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(x->shape, "X shape is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(y->shape, "Y shape is NULL", NULL);
    #endif
 
    size_t numel_x = product(x->shape, x->rank);
    size_t numel_y = product(y->shape, y->rank);
    
    // Check broadcasting compatibility
    if((numel_x % numel_y) != 0) {
        NNL2_ERROR("Cannot broadcast y tensor");
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return NULL;
    }
    
    TensorType x_dtype = x->dtype;
    TensorType y_dtype = y->dtype;
    TensorType result_dtype = MAX(x_dtype, y_dtype);
    
    // Create output tensor
    Tensor* result = nnl2_empty(x->shape, x->rank, result_dtype);
    if (result == NULL) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return NULL;
    }
    
    // Fallback to naive implementation for small tensors, mixed types, or complex cases
    if(numel_x < NNL2_MIN_BROADCASTING_PARALLEL_THRESHOLD || 
       x_dtype != y_dtype || 
       numel_y == 0) {
        Tensor* naive_result = naive_min_broadcasting((Tensor*)x, (Tensor*)y);
        if (naive_result != NULL) {
            // Copy data from naive result to our result tensor
            memcpy(result->data, naive_result->data, numel_x * get_dtype_size(result_dtype));
            nnl2_free_tensor(naive_result);
        }
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return result;
    }
    
    bool is_aligned = NNL2_IS_ALIGNED(x->data, NNL2_TENSOR_ALIGNMENT_32) &&
                      NNL2_IS_ALIGNED(y->data, NNL2_TENSOR_ALIGNMENT_32) &&
                      NNL2_IS_ALIGNED(result->data, NNL2_TENSOR_ALIGNMENT_32);
    
    // Warning for unaligned memory in safety modes
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!is_aligned) {
            NNL2_WARN("In nnl2_own_min_broadcasting, tensor memory is not aligned to 32 bytes. Performance may be suboptimal");
        }
    #endif
    
    size_t num_blocks = numel_x / numel_y;
    size_t num_threads = NNL2_NUM_THREADS;
    
    // Limit threads based on number of blocks
    num_threads = MIN(num_threads, num_blocks);
    
    pthread_t threads[num_threads];
    min_broadcasting_ptask tasks[num_threads];
    
    // Calculate optimal block distribution with load balancing
    size_t blocks_per_thread = num_blocks / num_threads;
    size_t remainder_blocks = num_blocks % num_threads;
    
    // Configure common task parameters
    for (size_t i = 0; i < num_threads; i++) {
        tasks[i].x = x;
        tasks[i].y = y;
        tasks[i].result = result;
        tasks[i].x_dtype = x_dtype;
        tasks[i].y_dtype = y_dtype;
        tasks[i].result_dtype = result_dtype;
        tasks[i].block_size = numel_y;
        tasks[i].aligned = is_aligned;
    }
    
    size_t current_block_start = 0;
    for (size_t i = 0; i < num_threads; i++) {
        size_t current_blocks = blocks_per_thread + (i < remainder_blocks ? 1 : 0);
        
        // Configure task for this thread
        tasks[i].start_block = current_block_start;
        tasks[i].end_block = current_block_start + current_blocks;
        
        // Select appropriate worker function based on data type
        void* (*worker_func)(void*) = NULL;
        switch(x_dtype) {
            case FLOAT64: worker_func = nnl2_own_pmin_broadcasting_float64; break;
            case FLOAT32: worker_func = nnl2_own_pmin_broadcasting_float32; break;
            case INT32:   worker_func = nnl2_own_pmin_broadcasting_int32;   break;
            default: {
                NNL2_TYPE_ERROR(x_dtype);
                nnl2_free_tensor(result);
                #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                    NNL2_FUNC_EXIT();
                #endif
                return NULL;
            }
        }
        
        // Create thread to process the assigned blocks
        int status = pthread_create(&threads[i], NULL, worker_func, &tasks[i]);
        if(status != 0) {
            NNL2_THREAD_CREATE_ERROR(status, "nnl2_own_min_broadcasting");
            num_threads = i;
            break;
        }
        
        current_block_start += current_blocks;
    }
    
    // Wait for all threads to complete
    for (size_t i = 0; i < num_threads; i++) {
        int join_status = pthread_join(threads[i], NULL);
        if(join_status != 0) {
            NNL2_THREAD_JOIN_ERROR(join_status, "nnl2_own_min_broadcasting");
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return result;
}

// Worker function implementations with AVX256 and prefetching

/** @brief
 * See documentation at declaration
 * 
 ** @see nnl2_own_pmin_broadcasting_float64
 **/
void* nnl2_own_pmin_broadcasting_float64(void* arg) {
    min_broadcasting_ptask* task = (min_broadcasting_ptask*)arg;
    const double* x_data = (const double*)task->x->data;
    const double* y_data = (const double*)task->y->data;
    double* result_data = (double*)task->result->data;
    size_t start_block = task->start_block;
    size_t end_block = task->end_block;
    size_t block_size = task->block_size;
    
    // Process each block assigned to this thread
    for(size_t block = start_block; block < end_block; block++) {
        size_t block_offset = block * block_size;
        size_t i = 0;
        
        // AVX256 processing with prefetching within the block
        if(task->aligned) {
            for(; i + 3 < block_size; i += 4) {
                // Prefetch next cache lines
                _mm_prefetch((char*)&x_data[block_offset + i + 16], _MM_HINT_T0);
                _mm_prefetch((char*)&y_data[i + 16], _MM_HINT_T0);
                
                __m256d v_x_data = _mm256_load_pd(&x_data[block_offset + i]);
                __m256d v_y_data = _mm256_load_pd(&y_data[i]);
                __m256d v_result = _mm256_min_pd(v_x_data, v_y_data);
                _mm256_store_pd(&result_data[block_offset + i], v_result);
            }
        } else {
            for(; i + 3 < block_size; i += 4) {
                _mm_prefetch((char*)&x_data[block_offset + i + 16], _MM_HINT_T0);
                _mm_prefetch((char*)&y_data[i + 16], _MM_HINT_T0);
                
                __m256d v_x_data = _mm256_loadu_pd(&x_data[block_offset + i]);
                __m256d v_y_data = _mm256_loadu_pd(&y_data[i]);
                __m256d v_result = _mm256_min_pd(v_x_data, v_y_data);
                _mm256_storeu_pd(&result_data[block_offset + i], v_result);
            }
        }
        
        // Scalar processing for remainder in the block
        for(; i < block_size; i++) {
            result_data[block_offset + i] = MIN(x_data[block_offset + i], y_data[i]);
        }
    }
    
    return NULL;
}

/** @brief
 * See documentation at declaration
 * 
 ** @see nnl2_own_pmin_broadcasting_float32
 **/
void* nnl2_own_pmin_broadcasting_float32(void* arg) {
    min_broadcasting_ptask* task = (min_broadcasting_ptask*)arg;
    const float* x_data = (const float*)task->x->data;
    const float* y_data = (const float*)task->y->data;
    float* result_data = (float*)task->result->data;
    size_t start_block = task->start_block;
    size_t end_block = task->end_block;
    size_t block_size = task->block_size;
    
    // Process each block assigned to this thread
    for(size_t block = start_block; block < end_block; block++) {
        size_t block_offset = block * block_size;
        size_t i = 0;
        
        // AVX256 processing with prefetching (8 elements per iteration)
        if(task->aligned) {
            for(; i + 7 < block_size; i += 8) {
                _mm_prefetch((char*)&x_data[block_offset + i + 32], _MM_HINT_T0);
                _mm_prefetch((char*)&y_data[i + 32], _MM_HINT_T0);
                
                __m256 v_x_data = _mm256_load_ps(&x_data[block_offset + i]);
                __m256 v_y_data = _mm256_load_ps(&y_data[i]);
                __m256 v_result = _mm256_min_ps(v_x_data, v_y_data);
                _mm256_store_ps(&result_data[block_offset + i], v_result);
            }
        } else {
            for(; i + 7 < block_size; i += 8) {
                _mm_prefetch((char*)&x_data[block_offset + i + 32], _MM_HINT_T0);
                _mm_prefetch((char*)&y_data[i + 32], _MM_HINT_T0);
                
                __m256 v_x_data = _mm256_loadu_ps(&x_data[block_offset + i]);
                __m256 v_y_data = _mm256_loadu_ps(&y_data[i]);
                __m256 v_result = _mm256_min_ps(v_x_data, v_y_data);
                _mm256_storeu_ps(&result_data[block_offset + i], v_result);
            }
        }
        
        // Scalar processing for remainder in the block
        for(; i < block_size; i++) {
            result_data[block_offset + i] = MIN(x_data[block_offset + i], y_data[i]);
        }
    }
    
    return NULL;
}

/** @brief
 * See documentation at declaration
 * 
 ** @see nnl2_own_pmin_broadcasting_int32
 **/
void* nnl2_own_pmin_broadcasting_int32(void* arg) {
    min_broadcasting_ptask* task = (min_broadcasting_ptask*)arg;
    const int32_t* x_data = (const int32_t*)task->x->data;
    const int32_t* y_data = (const int32_t*)task->y->data;
    int32_t* result_data = (int32_t*)task->result->data;
    size_t start_block = task->start_block;
    size_t end_block = task->end_block;
    size_t block_size = task->block_size;
    
    // Process each block assigned to this thread
    for(size_t block = start_block; block < end_block; block++) {
        size_t block_offset = block * block_size;
        size_t i = 0;
        
        // AVX256 processing with prefetching (8 elements per iteration)
        if(task->aligned) {
            for(; i + 7 < block_size; i += 8) {
                _mm_prefetch((char*)&x_data[block_offset + i + 32], _MM_HINT_T0);
                _mm_prefetch((char*)&y_data[i + 32], _MM_HINT_T0);
                
                __m256i v_x_data = _mm256_load_si256((__m256i*)&x_data[block_offset + i]);
                __m256i v_y_data = _mm256_load_si256((__m256i*)&y_data[i]);
                
                // For integers, we need to compare and select minimum
                __m256i v_compare = _mm256_cmpgt_epi32(v_x_data, v_y_data);
                __m256i v_result = _mm256_blendv_epi8(v_x_data, v_y_data, v_compare);
                
                _mm256_store_si256((__m256i*)&result_data[block_offset + i], v_result);
            }
        } else {
            for(; i + 7 < block_size; i += 8) {
                _mm_prefetch((char*)&x_data[block_offset + i + 32], _MM_HINT_T0);
                _mm_prefetch((char*)&y_data[i + 32], _MM_HINT_T0);
                
                __m256i v_x_data = _mm256_loadu_si256((__m256i*)&x_data[block_offset + i]);
                __m256i v_y_data = _mm256_loadu_si256((__m256i*)&y_data[i]);
                
                __m256i v_compare = _mm256_cmpgt_epi32(v_x_data, v_y_data);
                __m256i v_result = _mm256_blendv_epi8(v_x_data, v_y_data, v_compare);
                
                _mm256_storeu_si256((__m256i*)&result_data[block_offset + i], v_result);
            }
        }
        
        // Scalar processing for remainder in the block
        for(; i < block_size; i++) {
            result_data[block_offset + i] = MIN(x_data[block_offset + i], y_data[i]);
        }
    }
    
    return NULL;
}

#endif

/**
 * @ingroup backend_system
 * @brief Backend implementations for minimum operation with broadcasting
 */
Implementation min_broadcasting_backends[] = {
    REGISTER_BACKEND(naive_min_broadcasting, nnl2_naive, NAIVE_BACKEND_NAME),
    
    #if defined(NNL2_AVX256_AVAILABLE) && defined(NNL2_PTHREAD_AVAILABLE)
        REGISTER_BACKEND(nnl2_own_min_broadcasting, nnl2_own, NNL2_OWN_NAME),
    #endif
};  

/**
 * @brief Function pointer for minimum operation with broadcasting
 * @ingroup backend_system
 */
minbroadcastingfn min_broadcasting;

/**
 * @brief Sets the backend for minimum operation with broadcasting
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for minimum operation with broadcasting
 */
void set_min_broadcasting_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(min_broadcasting_backends, min_broadcasting, backend_name);
}

#endif /** NNL2_BROADCASTING_MIN_H **/
