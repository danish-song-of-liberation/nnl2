#ifndef NNL2_BROADCASTING_AXPY_INPLACE_H
#define NNL2_BROADCASTING_AXPY_INPLACE_H

/** @brief
 * Performs element-wise AXPY operation with broadcasting (in place)
 * Computes: summand = summand + alpha * sumend
 *
 ** @param summand
 * Pointer to summand tensor (will be modified in place)
 *
 ** @param sumend
 * Pointer to sumend tensor
 *
 ** @param alpha
 * Scalar multiplier
 */
void naive_axpy_broadcasting_inplace(nnl2_tensor* summand, const nnl2_tensor* sumend, float alpha) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks for max safety level 
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(summand, "Summand tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(sumend, "Sumend tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(summand->shape, "Summand shape is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(sumend->shape, "Sumend shape is NULL");
    #endif
    
    size_t numel_summand = product(summand->shape, summand->rank);
    size_t numel_sumend = product(sumend->shape, sumend->rank);
    
    // Getting the tensor data types
    nnl2_tensor_type summand_dtype = summand->dtype;
    nnl2_tensor_type sumend_dtype = sumend->dtype;

    if((numel_summand % numel_sumend) == 0) {
        // Handling the case where the data types match (more efficiently)
        if(summand_dtype == sumend_dtype) {
            switch(summand_dtype) {
                case FLOAT64: {
                    double* cast_summand_data = (double*)summand->data;
                    double* cast_sumend_data = (double*)sumend->data;
                    double alpha_double = (double)alpha;

                    for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
                        for(size_t j = 0; j < numel_sumend; j++) {
                            cast_summand_data[i * numel_sumend + j] += cast_sumend_data[j] * alpha_double;
                        }
                    }

                    break;
                }

                case FLOAT32: {
                    float* cast_summand_data = (float*)summand->data;
                    float* cast_sumend_data = (float*)sumend->data;

                    for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
                        for(size_t j = 0; j < numel_sumend; j++) {
                            cast_summand_data[i * numel_sumend + j] += cast_sumend_data[j] * alpha;
                        }
                    }

                    break;
                }

                case INT32: {
                    int32_t* cast_summand_data = (int32_t*)summand->data;
                    int32_t* cast_sumend_data = (int32_t*)sumend->data;
                    int32_t alpha_int = (int32_t)alpha;

                    for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
                        for(size_t j = 0; j < numel_sumend; j++) {
                            cast_summand_data[i * numel_sumend + j] += cast_sumend_data[j] * alpha_int;
                        }
                    }

                    break;
                }

                default: {
                    NNL2_TYPE_ERROR(summand_dtype);
                    return;
                }
            }
        } else {
            // Handling a case with different data types (conversion required)
            size_t sumend_step = get_dtype_size(sumend_dtype);
            char* sumend_data = (char*)sumend->data;
            
            switch(summand_dtype) {
                case FLOAT64: {
                    double* data_summand = (double*)summand->data;
                    double alpha_double = (double)alpha;
                
                    for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
                        for(size_t j = 0; j < numel_sumend; j++) {
                            void* sumend_elem = sumend_data + j * sumend_step;
                            data_summand[i * numel_sumend + j] += nnl2_convert_to_float64(sumend_elem, sumend_dtype) * alpha_double;
                        }
                    }
                
                    break; 
                }
                
                case FLOAT32: {
                    float* data_summand = (float*)summand->data;
                
                    for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
                        for(size_t j = 0; j < numel_sumend; j++) {
                            void* sumend_elem = sumend_data + j * sumend_step;
                            data_summand[i * numel_sumend + j] += nnl2_convert_to_float32(sumend_elem, sumend_dtype) * alpha;
                        }
                    }
                
                    break; 
                }
                
                case INT32: {
                    int32_t* data_summand = (int32_t*)summand->data;
                    int32_t alpha_int = (int32_t)alpha;
                
                    for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
                        for(size_t j = 0; j < numel_sumend; j++) {
                            void* sumend_elem = sumend_data + j * sumend_step;
                            data_summand[i * numel_sumend + j] += nnl2_convert_to_int32(sumend_elem, sumend_dtype) * alpha_int;
                        }
                    }
                
                    break; 
                }
                
                default: {
                    NNL2_TYPE_ERROR(summand_dtype);
                    return;
                }
            }
        }
    } else {
        NNL2_ERROR("Cannot broadcast sumend tensor");
        return;
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

#ifdef NNL2_PTHREAD_AVAILABLE

/** @brief
 * Threshold for enabling parallel execution of AXPY broadcasting in-place operation
 */
#define NNL2_AXPY_BROADCASTING_INPLACE_PARALLEL_THRESHOLD 100000

/** @brief
 * Worker function for parallel double precision AXPY broadcasting in-place operation
 * 
 ** @param arg 
 * Pointer to axpy_broadcasting_inplace_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 * 
 ** @details
 * Processes FLOAT64 data using AVX256 instructions with cache prefetching
 * for optimal memory access patterns in broadcasting AXPY in-place operation
 */
void* nnl2_own_paxpy_broadcasting_inplace_float64(void* arg);

/** @brief
 * Worker function for parallel single precision AXPY broadcasting in-place operation
 * 
 ** @param arg 
 * Pointer to axpy_broadcasting_inplace_ptask structure containing thread parameters  
 * 
 ** @return 
 * NULL (for pthread API compatibility)
 * 
 ** @see nnl2_own_paxpy_broadcasting_inplace_float64
 **/
void* nnl2_own_paxpy_broadcasting_inplace_float32(void* arg);

/** @brief
 * Worker function for parallel integer AXPY broadcasting in-place operation
 * 
 ** @param arg 
 * Pointer to axpy_broadcasting_inplace_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 * 
 ** @see nnl2_own_paxpy_broadcasting_inplace_float64
 **/
void* nnl2_own_paxpy_broadcasting_inplace_int32(void* arg);

/** @brief
 * High-performance parallel implementation of AXPY broadcasting in-place operation
 * 
 ** @param summand 
 * Pointer to the summand tensor (will be modified in-place)
 *
 ** @param sumend 
 * Pointer to the sumend tensor (broadcasted)
 *
 ** @param alpha
 * Scalar multiplier for the sumend tensor values
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
 * Modifies the summand tensor directly
 * Only supports simple broadcasting patterns where numel_summand % numel_sumend == 0
 */
void nnl2_own_axpy_broadcasting_inplace(nnl2_tensor* summand, const nnl2_tensor* sumend, float alpha) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks for max safety level 
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(summand, "Summand tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(sumend, "Sumend tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(summand->shape, "Summand shape is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(sumend->shape, "Sumend shape is NULL");
    #endif
    
    size_t numel_summand = product(summand->shape, summand->rank);
    size_t numel_sumend = product(sumend->shape, sumend->rank);
    
    // Check broadcasting compatibility
    if((numel_summand % numel_sumend) != 0) {
        NNL2_ERROR("Cannot broadcast sumend tensor");
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return;
    }
    
    nnl2_tensor_type summand_dtype = summand->dtype;
    nnl2_tensor_type sumend_dtype = sumend->dtype;
    
    // Fallback to naive implementation for small tensors, mixed types, or complex cases
    if(numel_summand < NNL2_AXPY_BROADCASTING_INPLACE_PARALLEL_THRESHOLD || 
       summand_dtype != sumend_dtype || 
       numel_sumend == 0) {
        naive_axpy_broadcasting_inplace(summand, sumend, alpha);
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return;
    }
    
    bool is_aligned = NNL2_IS_ALIGNED(summand->data, NNL2_TENSOR_ALIGNMENT_32) &&
                      NNL2_IS_ALIGNED(sumend->data, NNL2_TENSOR_ALIGNMENT_32);
    
    // Warning for unaligned memory in safety modes
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!is_aligned) {
            NNL2_WARN("In nnl2_own_axpy_broadcasting_inplace, tensor memory is not aligned to 32 bytes. Performance may be suboptimal");
        }
    #endif
    
    size_t num_blocks = numel_summand / numel_sumend;
    size_t num_threads = NNL2_NUM_THREADS;
    
    // Limit threads based on number of blocks
    num_threads = MIN(num_threads, num_blocks);
    
    pthread_t threads[num_threads];
    axpy_broadcasting_inplace_ptask tasks[num_threads];
    
    // Calculate optimal block distribution with load balancing
    size_t blocks_per_thread = num_blocks / num_threads;
    size_t remainder_blocks = num_blocks % num_threads;
    
    // Configure common task parameters
    for (size_t i = 0; i < num_threads; i++) {
        tasks[i].summand = summand;
        tasks[i].sumend = sumend;
        tasks[i].summand_dtype = summand_dtype;
        tasks[i].sumend_dtype = sumend_dtype;
        tasks[i].block_size = numel_sumend;
        tasks[i].aligned = is_aligned;
        
        switch(summand_dtype) {
            case FLOAT64: tasks[i].alpha.float64_alpha = (double)alpha; break;
            case FLOAT32: tasks[i].alpha.float32_alpha = alpha; break;
            case INT32:   tasks[i].alpha.int32_alpha = (int32_t)alpha; break;
            default: {
                NNL2_TYPE_ERROR(summand_dtype);
                #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                    NNL2_FUNC_EXIT();
                #endif
                return;
            }
        }
    }
    
    size_t current_block_start = 0;
    for (size_t i = 0; i < num_threads; i++) {
        size_t current_blocks = blocks_per_thread + (i < remainder_blocks ? 1 : 0);
        
        // Configure task for this thread
        tasks[i].start_block = current_block_start;
        tasks[i].end_block = current_block_start + current_blocks;
        
        // Select appropriate worker function based on data type
        void* (*worker_func)(void*) = NULL;
        switch(summand_dtype) {
            case FLOAT64: worker_func = nnl2_own_paxpy_broadcasting_inplace_float64; break;
            case FLOAT32: worker_func = nnl2_own_paxpy_broadcasting_inplace_float32; break;
            case INT32:   worker_func = nnl2_own_paxpy_broadcasting_inplace_int32;   break;
            default: {
                NNL2_TYPE_ERROR(summand_dtype);
                #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                    NNL2_FUNC_EXIT();
                #endif
                return;
            }
        }
        
        // Create thread to process the assigned blocks
        int status = pthread_create(&threads[i], NULL, worker_func, &tasks[i]);
        if(status != 0) {
            NNL2_THREAD_CREATE_ERROR(status, "nnl2_own_axpy_broadcasting_inplace");
            num_threads = i;
            break;
        }
        
        current_block_start += current_blocks;
    }
    
    // Wait for all threads to complete
    for (size_t i = 0; i < num_threads; i++) {
        int join_status = pthread_join(threads[i], NULL);
        if(join_status != 0) {
            NNL2_THREAD_JOIN_ERROR(join_status, "nnl2_own_axpy_broadcasting_inplace");
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

// Worker function implementations with AVX256 and prefetching

/** @brief
 * See documentation at declaration
 * 
 ** @see nnl2_own_paxpy_broadcasting_inplace_float64
 **/
void* nnl2_own_paxpy_broadcasting_inplace_float64(void* arg) {
    axpy_broadcasting_inplace_ptask* task = (axpy_broadcasting_inplace_ptask*)arg;
    double* summand_data = (double*)task->summand->data;
    const double* sumend_data = (const double*)task->sumend->data;
    size_t start_block = task->start_block;
    size_t end_block = task->end_block;
    size_t block_size = task->block_size;
    double alpha = task->alpha.float64_alpha;
    
    // Create AVX256 vector with alpha value repeated
    __m256d v_alpha = _mm256_set1_pd(alpha);
    
    // Process each block assigned to this thread
    for(size_t block = start_block; block < end_block; block++) {
        size_t block_offset = block * block_size;
        size_t i = 0;
        
        // AVX256 processing with prefetching within the block
        if(task->aligned) {
            for(; i + 3 < block_size; i += 4) {
                // Prefetch next cache lines
                _mm_prefetch((char*)&summand_data[block_offset + i + 16], _MM_HINT_T0);
                _mm_prefetch((char*)&sumend_data[i + 16], _MM_HINT_T0);
                
                __m256d v_summand = _mm256_load_pd(&summand_data[block_offset + i]);
                __m256d v_sumend = _mm256_load_pd(&sumend_data[i]);
                __m256d v_scaled_sumend = _mm256_mul_pd(v_sumend, v_alpha);
                __m256d v_result = _mm256_add_pd(v_summand, v_scaled_sumend);
                _mm256_store_pd(&summand_data[block_offset + i], v_result);
            }
        } else {
            for(; i + 3 < block_size; i += 4) {
                _mm_prefetch((char*)&summand_data[block_offset + i + 16], _MM_HINT_T0);
                _mm_prefetch((char*)&sumend_data[i + 16], _MM_HINT_T0);
                
                __m256d v_summand = _mm256_loadu_pd(&summand_data[block_offset + i]);
                __m256d v_sumend = _mm256_loadu_pd(&sumend_data[i]);
                __m256d v_scaled_sumend = _mm256_mul_pd(v_sumend, v_alpha);
                __m256d v_result = _mm256_add_pd(v_summand, v_scaled_sumend);
                _mm256_storeu_pd(&summand_data[block_offset + i], v_result);
            }
        }
        
        // Scalar processing for remainder in the block
        for(; i < block_size; i++) {
            summand_data[block_offset + i] += sumend_data[i] * alpha;
        }
    }
    
    return NULL;
}

/** @brief
 * See documentation at declaration
 * 
 ** @see nnl2_own_paxpy_broadcasting_inplace_float32
 **/
void* nnl2_own_paxpy_broadcasting_inplace_float32(void* arg) {
    axpy_broadcasting_inplace_ptask* task = (axpy_broadcasting_inplace_ptask*)arg;
    float* summand_data = (float*)task->summand->data;
    const float* sumend_data = (const float*)task->sumend->data;
    size_t start_block = task->start_block;
    size_t end_block = task->end_block;
    size_t block_size = task->block_size;
    float alpha = task->alpha.float32_alpha;
    
    // Create AVX256 vector with alpha value repeated
    __m256 v_alpha = _mm256_set1_ps(alpha);
    
    // Process each block assigned to this thread
    for(size_t block = start_block; block < end_block; block++) {
        size_t block_offset = block * block_size;
        size_t i = 0;
        
        // AVX256 processing with prefetching (8 elements per iteration)
        if(task->aligned) {
            for(; i + 7 < block_size; i += 8) {
                _mm_prefetch((char*)&summand_data[block_offset + i + 32], _MM_HINT_T0);
                _mm_prefetch((char*)&sumend_data[i + 32], _MM_HINT_T0);
                
                __m256 v_summand = _mm256_load_ps(&summand_data[block_offset + i]);
                __m256 v_sumend = _mm256_load_ps(&sumend_data[i]);
                __m256 v_scaled_sumend = _mm256_mul_ps(v_sumend, v_alpha);
                __m256 v_result = _mm256_add_ps(v_summand, v_scaled_sumend);
                _mm256_store_ps(&summand_data[block_offset + i], v_result);
            }
        } else {
            for(; i + 7 < block_size; i += 8) {
                _mm_prefetch((char*)&summand_data[block_offset + i + 32], _MM_HINT_T0);
                _mm_prefetch((char*)&sumend_data[i + 32], _MM_HINT_T0);
                
                __m256 v_summand = _mm256_loadu_ps(&summand_data[block_offset + i]);
                __m256 v_sumend = _mm256_loadu_ps(&sumend_data[i]);
                __m256 v_scaled_sumend = _mm256_mul_ps(v_sumend, v_alpha);
                __m256 v_result = _mm256_add_ps(v_summand, v_scaled_sumend);
                _mm256_storeu_ps(&summand_data[block_offset + i], v_result);
            }
        }
        
        // Scalar processing for remainder in the block
        for(; i < block_size; i++) {
            summand_data[block_offset + i] += sumend_data[i] * alpha;
        }
    }
    
    return NULL;
}

/** @brief
 * See documentation at declaration
 * 
 ** @see nnl2_own_paxpy_broadcasting_inplace_int32
 **/
void* nnl2_own_paxpy_broadcasting_inplace_int32(void* arg) {
    axpy_broadcasting_inplace_ptask* task = (axpy_broadcasting_inplace_ptask*)arg;
    int32_t* summand_data = (int32_t*)task->summand->data;
    const int32_t* sumend_data = (const int32_t*)task->sumend->data;
    size_t start_block = task->start_block;
    size_t end_block = task->end_block;
    size_t block_size = task->block_size;
    int32_t alpha = task->alpha.int32_alpha;
    
    // Create AVX256 vector with alpha value repeated
    __m256i v_alpha = _mm256_set1_epi32(alpha);
    
    // Process each block assigned to this thread
    for(size_t block = start_block; block < end_block; block++) {
        size_t block_offset = block * block_size;
        size_t i = 0;
        
        // AVX256 processing with prefetching (8 elements per iteration)
        if(task->aligned) {
            for(; i + 7 < block_size; i += 8) {
                _mm_prefetch((char*)&summand_data[block_offset + i + 32], _MM_HINT_T0);
                _mm_prefetch((char*)&sumend_data[i + 32], _MM_HINT_T0);
                
                __m256i v_summand = _mm256_load_si256((__m256i*)&summand_data[block_offset + i]);
                __m256i v_sumend = _mm256_load_si256((__m256i*)&sumend_data[i]);
                __m256i v_scaled_sumend = _mm256_mullo_epi32(v_sumend, v_alpha);
                __m256i v_result = _mm256_add_epi32(v_summand, v_scaled_sumend);
                _mm256_store_si256((__m256i*)&summand_data[block_offset + i], v_result);
            }
        } else {
            for(; i + 7 < block_size; i += 8) {
                _mm_prefetch((char*)&summand_data[block_offset + i + 32], _MM_HINT_T0);
                _mm_prefetch((char*)&sumend_data[i + 32], _MM_HINT_T0);
                
                __m256i v_summand = _mm256_loadu_si256((__m256i*)&summand_data[block_offset + i]);
                __m256i v_sumend = _mm256_loadu_si256((__m256i*)&sumend_data[i]);
                __m256i v_scaled_sumend = _mm256_mullo_epi32(v_sumend, v_alpha);
                __m256i v_result = _mm256_add_epi32(v_summand, v_scaled_sumend);
                _mm256_storeu_si256((__m256i*)&summand_data[block_offset + i], v_result);
            }
        }
        
        // Scalar processing for remainder in the block
        for(; i < block_size; i++) {
            summand_data[block_offset + i] += sumend_data[i] * alpha;
        }
    }
    
    return NULL;
}

#endif

/**
 * @ingroup backend_system
 * @brief Backend implementations for AXPY operation with broadcasting (in place)
 */
nnl2_runtime_implementation axpy_broadcasting_inplace_backends[] = {
    REGISTER_BACKEND(naive_axpy_broadcasting_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
    
    #if defined(NNL2_AVX256_AVAILABLE) && defined(NNL2_PTHREAD_AVAILABLE)
        REGISTER_BACKEND(nnl2_own_axpy_broadcasting_inplace, nnl2_own, NNL2_OWN_NAME),
    #endif
};  

/**
 * @brief Function pointer for AXPY operation with broadcasting (in place)
 * @ingroup backend_system
 */
axpybroadcastinginplacefn axpy_broadcasting_inplace;

/**
 * @brief Sets the backend for AXPY operation with broadcasting (in place)
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for AXPY operation with broadcasting
 */
void set_axpy_broadcasting_inplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(axpy_broadcasting_inplace_backends, axpy_broadcasting_inplace, backend_name);
}

#endif /** NNL2_BROADCASTING_AXPY_INPLACE_H **/
