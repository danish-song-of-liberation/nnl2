#ifndef NNL2_AXPY_H
#define NNL2_AXPY_H

/** @brief
 * Performs element-wise AXPY operation (naive implementation)
 * Computes: result = summand + alpha * sumend
 *
 ** @details
 * The function creates a new tensor containing the result of the AXPY operation
 * on the two input tensors. It supports various data types with automatic
 * casting to a higher type in the hierarchy
 *
 ** @param summand
 * Pointer to the summand tensor
 *
 ** @param sumend
 * Pointer to the sumend tensor to be scaled and added
 *
 ** @param alpha
 * Scalar multiplier for the sumend tensor
 *
 ** @return 
 * Pointer to a new tensor with the AXPY operation result
 *
 ** @note
 * Uses volatile pointers to prevent compiler optimizations
 *
 ** @note
 * Returns NULL in case of failure
 *
 ** @see nnl2_empty
 ** @see get_dtype_size()
 ** @see nnl2_convert_to_float64()
 ** @see nnl2_convert_to_float32()
 ** @see nnl2_convert_to_int32()
 **/
nnl2_tensor* naive_axpy(nnl2_tensor* summand, nnl2_tensor* sumend, float alpha) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks at the maximum safety level
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(summand, "Summand tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(summand->data, "Summand tensor's data is NULL", NULL);
        
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(sumend, "Sumend tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(sumend->data, "Sumend tensor's data is NULL", NULL);
    #endif
    
    // Calculate the total number of elements in the tensors
    size_t len = nnl2_product(summand->shape, summand->rank);
    
    nnl2_tensor_type dtype_summand = summand->dtype;
    nnl2_tensor_type dtype_sumend = sumend->dtype;
    
    // Selecting the winning type (higher in the hierarchy)
    nnl2_tensor_type winner_in_the_type_hierarchy = MAX(dtype_summand, dtype_sumend);

    // Create an output tensor with the same shape and winning data type
    nnl2_tensor* result = nnl2_empty(summand->shape, summand->rank, winner_in_the_type_hierarchy);
    
    if(len == 0) return result;
    
    if(dtype_summand == dtype_sumend) {
        // Handling the case if the data types match
        
        switch(dtype_summand) {
            case FLOAT64: {
                volatile double* data_summand = (double*)summand->data;
                volatile double* data_sumend = (double*)sumend->data;
                volatile double* data_result = (double*)result->data;
                double alpha_double = (double)alpha;
            
                // Element-wise AXPY operation
                for(size_t i = 0; i < len; i++) {
                    data_result[i] = data_summand[i] + (data_sumend[i] * alpha_double);
                }
                
                break;
            }
            
            case FLOAT32: {
                volatile float* data_summand = (float*)summand->data;
                volatile float* data_sumend = (float*)sumend->data;
                volatile float* data_result = (float*)result->data;
        
                // Element-wise AXPY operation
                for(size_t i = 0; i < len; i++) {
                    data_result[i] = data_summand[i] + (data_sumend[i] * alpha);
                }
                
                break;
            }
			
			case INT64: {
                volatile int64_t* data_summand = (int64_t*)summand->data;
                volatile int64_t* data_sumend = (int64_t*)sumend->data;
                volatile int64_t* data_result = (int64_t*)result->data;
                int64_t alpha_int64 = (int64_t)alpha;
        
                // Element-wise AXPY operation
                for(size_t i = 0; i < len; i++) {
                    data_result[i] = data_summand[i] + (data_sumend[i] * alpha_int64);
                }
                
                break;
            }
            
            case INT32: {
                volatile int32_t* data_summand = (int32_t*)summand->data;
                volatile int32_t* data_sumend = (int32_t*)sumend->data;
                volatile int32_t* data_result = (int32_t*)result->data;
                int32_t alpha_int = (int32_t)alpha;
        
                // Element-wise AXPY operation
                for(size_t i = 0; i < len; i++) {
                    data_result[i] = data_summand[i] + (data_sumend[i] * alpha_int);
                }
                
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(dtype_summand);
                nnl2_free_tensor(result);
                return NULL;
            }
        }
    } else {
        // Handling the case if the data types are NOT match
        switch(winner_in_the_type_hierarchy) {
            case FLOAT64: {
                volatile double* data_result = (double*)result->data;
                double alpha_double = (double)alpha;
                
                for(size_t i = 0; i < len; i++) {
                    // Calculate the pointers to the current elements, taking into account the size of the type
                    void* elem_summand = (char*)summand->data + i * get_dtype_size(dtype_summand);
                    void* elem_sumend = (char*)sumend->data + i * get_dtype_size(dtype_sumend);
                    
                    data_result[i] = nnl2_convert_to_float64(elem_summand, dtype_summand) + (nnl2_convert_to_float64(elem_sumend, dtype_sumend) * alpha_double);
                }
                
                break;
            }
            
            case FLOAT32: {
                volatile float* data_result = (float*)result->data;
                
                for(size_t i = 0; i < len; i++) {
                    // Calculate the pointers to the current elements, taking into account the size of the type
                    void* elem_summand = (char*)summand->data + i * get_dtype_size(dtype_summand);
                    void* elem_sumend = (char*)sumend->data + i * get_dtype_size(dtype_sumend);
                    
                    data_result[i] = nnl2_convert_to_float32(elem_summand, dtype_summand) + (nnl2_convert_to_float32(elem_sumend, dtype_sumend) * alpha);
                }
                
                break;
            }
			
			case INT64: {
                volatile int64_t* data_result = (int64_t*)result->data;
                int64_t alpha_int64 = (int64_t)alpha;
                
                if(alpha != (float)((int64_t)alpha)) {
                    NNL2_WARN("Alpha value will be truncated when converting to INT64");
                }
                
                for(size_t i = 0; i < len; i++) {
                    // Calculate the pointers to the current elements, taking into account the size of the type
                    void* elem_summand = (char*)summand->data + i * get_dtype_size(dtype_summand);
                    void* elem_sumend = (char*)sumend->data + i * get_dtype_size(dtype_sumend);
                    
                    data_result[i] = nnl2_convert_to_int64(elem_summand, dtype_summand) + 
                                    (nnl2_convert_to_int64(elem_sumend, dtype_sumend) * alpha_int64);
                }
                
                break;
            }
        
            case INT32: {
                volatile int32_t* data_result = (int32_t*)result->data;
                int32_t alpha_int = (int32_t)alpha;
                
                for(size_t i = 0; i < len; i++) {
                    // Calculate the pointers to the current elements, taking into account the size of the type
                    void* elem_summand = (char*)summand->data + i * get_dtype_size(dtype_summand);
                    void* elem_sumend = (char*)sumend->data + i * get_dtype_size(dtype_sumend);
                    
                    data_result[i] = nnl2_convert_to_int32(elem_summand, dtype_summand) + 
                                    (nnl2_convert_to_int32(elem_sumend, dtype_sumend) * alpha_int);
                }
                
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(winner_in_the_type_hierarchy);
                nnl2_free_tensor(result);
                return NULL;
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
 * Threshold for enabling parallel execution of AXPY operation
 */
#define NNL2_AXPY_PARALLEL_THRESHOLD 1000000

/** @brief
 * Worker function for parallel double precision AXPY operation
 * 
 ** @param arg 
 * Pointer to axpy_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 * 
 ** @details
 * Processes FLOAT64 data using AVX256 instructions with cache prefetching
 * for optimal memory access patterns in AXPY operation
 */
void* nnl2_own_paxpy_float64(void* arg);

/** @brief
 * Worker function for parallel single precision AXPY operation
 * 
 ** @param arg 
 * Pointer to axpy_ptask structure containing thread parameters  
 * 
 ** @return 
 * NULL (for pthread API compatibility)
 * 
 ** @see nnl2_own_paxpy_float64
 **/
void* nnl2_own_paxpy_float32(void* arg);

/** @brief
 * Worker function for parallel 64-bit integer AXPY operation
 * 
 ** @param arg 
 * Pointer to axpy_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 * 
 ** @see nnl2_own_paxpy_int32
 **/
void* nnl2_own_paxpy_int64(void* arg);

/** @brief
 * Worker function for parallel integer AXPY operation
 * 
 ** @param arg 
 * Pointer to axpy_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 * 
 ** @see nnl2_own_paxpy_float64
 **/
void* nnl2_own_paxpy_int32(void* arg);

/** @brief
 * High-performance parallel implementation of AXPY operation
 * 
 ** @param summand 
 * Pointer to the summand tensor
 *
 ** @param sumend 
 * Pointer to the sumend tensor
 *
 ** @param alpha
 * Scalar multiplier for the sumend tensor values
 * 
 ** @return
 * Pointer to new tensor containing AXPY operation result
 *
 ** @details
 * Combines AVX256 vectorization, multi-threading with pthread, and cache
 * prefetching for maximum performance on modern CPU architectures.
 * Automatically selects optimal thread count and chunk sizes.
 * 
 ** @note
 * Uses NNL2_NUM_THREADS for parallelization configuration
 * Falls back to naive implementation for small tensors or mixed types
 * 
 ** @warning
 * Requires pthread support and AVX256 capable CPU
 * Creates a new tensor for the result
 */
nnl2_tensor* nnl2_own_axpy(const nnl2_tensor* summand, const nnl2_tensor* sumend, float alpha) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks at the maximum safety level
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(summand, "Summand tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(summand->data, "Summand tensor data is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(sumend, "Sumend tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(sumend->data, "Sumend tensor data is NULL", NULL);
    #endif
    
    size_t total_elems = nnl2_product(summand->shape, summand->rank);
    
    nnl2_tensor_type dtype_summand = summand->dtype;
    nnl2_tensor_type dtype_sumend = sumend->dtype;
    nnl2_tensor_type result_dtype = MAX(dtype_summand, dtype_sumend);
    
    // Create output tensor
    nnl2_tensor* result = nnl2_empty(summand->shape, summand->rank, result_dtype);
    if (result == NULL) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return NULL;
    }
    
    if(total_elems == 0) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return result;
    }
    
    // Fallback to naive implementation for small tensors or mixed types
    if(total_elems < NNL2_AXPY_PARALLEL_THRESHOLD || dtype_summand != dtype_sumend) {
        nnl2_tensor* naive_result = naive_axpy((nnl2_tensor*)summand, (nnl2_tensor*)sumend, alpha);
        if (naive_result != NULL) {
            // Copy data from naive result to our result tensor
            memcpy(result->data, naive_result->data, total_elems * get_dtype_size(result_dtype));
            nnl2_free_tensor(naive_result);
        }
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return result;
    }
    
    bool is_aligned = NNL2_IS_ALIGNED(summand->data, NNL2_TENSOR_ALIGNMENT_32) &&
                      NNL2_IS_ALIGNED(sumend->data, NNL2_TENSOR_ALIGNMENT_32) &&
                      NNL2_IS_ALIGNED(result->data, NNL2_TENSOR_ALIGNMENT_32);
    
    // Warning for unaligned memory in safety modes
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!is_aligned) {
            NNL2_WARN("In nnl2_own_axpy, tensor memory is not aligned to 32 bytes. Performance may be suboptimal");
        }
    #endif
    
    size_t num_threads = NNL2_NUM_THREADS;
    pthread_t threads[num_threads];
    axpy_ptask tasks[num_threads];
    
    // Calculate optimal chunk sizes with load balancing
    size_t chunk = total_elems / num_threads;
    size_t remainder = total_elems % num_threads;
    
    // Configure alpha value based on data type
    for (size_t i = 0; i < num_threads; i++) {
        tasks[i].summand = summand;
        tasks[i].sumend = sumend;
        tasks[i].result = result;
        tasks[i].dtype_summand = dtype_summand;
        tasks[i].dtype_sumend = dtype_sumend;
        tasks[i].result_dtype = result_dtype;
        tasks[i].aligned = is_aligned;
        
        switch(dtype_summand) {
            case FLOAT64: tasks[i].alpha.float64_alpha = (double)alpha; break;
            case FLOAT32: tasks[i].alpha.float32_alpha = alpha; break;
            case INT32:   tasks[i].alpha.int32_alpha = (int32_t)alpha; break;
			case INT64:   tasks[i].alpha.int64_alpha = (int64_t)alpha; break;
			
            default: {
                NNL2_TYPE_ERROR(dtype_summand);
                nnl2_free_tensor(result);
				
                #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                    NNL2_FUNC_EXIT();
                #endif
				
                return NULL;
            }
        }
    }
    
    size_t current_start = 0;
    for (size_t i = 0; i < num_threads; i++) {
        size_t current_chunk = chunk + (i < remainder ? 1 : 0);
        
        // Configure task for this thread
        tasks[i].start = current_start;
        tasks[i].end = current_start + current_chunk;
        
        // Select appropriate worker function based on data type
        void* (*worker_func)(void*) = NULL;
        switch(dtype_summand) {
            case FLOAT64: worker_func = nnl2_own_paxpy_float64; break;
            case FLOAT32: worker_func = nnl2_own_paxpy_float32; break;
            case INT32:   worker_func = nnl2_own_paxpy_int32;   break;
			case INT64:   worker_func = nnl2_own_paxpy_int64;   break;

            default: {
                NNL2_TYPE_ERROR(dtype_summand);
                nnl2_free_tensor(result);
                #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                    NNL2_FUNC_EXIT();
                #endif
                return NULL;
            }
        }
        
        // Create thread to process the assigned chunk
        int status = pthread_create(&threads[i], NULL, worker_func, &tasks[i]);
        if(status != 0) {
            NNL2_THREAD_CREATE_ERROR(status, "nnl2_own_axpy");
            num_threads = i;
            break;
        }
        
        current_start += current_chunk;
    }
    
    // Wait for all threads to complete
    for (size_t i = 0; i < num_threads; i++) {
        int join_status = pthread_join(threads[i], NULL);
        if(join_status != 0) {
            NNL2_THREAD_JOIN_ERROR(join_status, "nnl2_own_axpy");
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
 ** @see nnl2_own_paxpy_float64
 **/
void* nnl2_own_paxpy_float64(void* arg) {
    axpy_ptask* task = (axpy_ptask*)arg;
    const double* summand_data = (const double*)task->summand->data;
    const double* sumend_data = (const double*)task->sumend->data;
    double* result_data = (double*)task->result->data;
    size_t start = task->start;
    size_t end = task->end;
    double alpha = task->alpha.float64_alpha;
    
    // Create AVX256 vector with alpha value repeated
    __m256d v_alpha = _mm256_set1_pd(alpha);
    
    size_t i = start;
    
    // AVX256 processing with prefetching
    if(task->aligned) {
        for(; i + 3 < end; i += 4) {
            // Prefetch next cache lines
            _mm_prefetch((char*)&summand_data[i + 16], _MM_HINT_T0);
            _mm_prefetch((char*)&sumend_data[i + 16], _MM_HINT_T0);
            
            __m256d v_summand = _mm256_load_pd(&summand_data[i]);
            __m256d v_sumend = _mm256_load_pd(&sumend_data[i]);
            __m256d v_scaled_sumend = _mm256_mul_pd(v_sumend, v_alpha);
            __m256d v_result = _mm256_add_pd(v_summand, v_scaled_sumend);
            _mm256_store_pd(&result_data[i], v_result);
        }
    } else {
        for(; i + 3 < end; i += 4) {
            _mm_prefetch((char*)&summand_data[i + 16], _MM_HINT_T0);
            _mm_prefetch((char*)&sumend_data[i + 16], _MM_HINT_T0);
            
            __m256d v_summand = _mm256_loadu_pd(&summand_data[i]);
            __m256d v_sumend = _mm256_loadu_pd(&sumend_data[i]);
            __m256d v_scaled_sumend = _mm256_mul_pd(v_sumend, v_alpha);
            __m256d v_result = _mm256_add_pd(v_summand, v_scaled_sumend);
            _mm256_storeu_pd(&result_data[i], v_result);
        }
    }
    
    // Scalar processing for remainder
    for(; i < end; i++) {
        result_data[i] = summand_data[i] + (sumend_data[i] * alpha);
    }
    
    return NULL;
}

/** @brief
 * See documentation at declaration
 * 
 ** @see nnl2_own_paxpy_float32
 **/
void* nnl2_own_paxpy_float32(void* arg) {
    axpy_ptask* task = (axpy_ptask*)arg;
    const float* summand_data = (const float*)task->summand->data;
    const float* sumend_data = (const float*)task->sumend->data;
    float* result_data = (float*)task->result->data;
    size_t start = task->start;
    size_t end = task->end;
    float alpha = task->alpha.float32_alpha;
    
    // Create AVX256 vector with alpha value repeated
    __m256 v_alpha = _mm256_set1_ps(alpha);
    
    size_t i = start;
    
    // AVX256 processing with prefetching (8 elements per iteration)
    if(task->aligned) {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&summand_data[i + 32], _MM_HINT_T0);
            _mm_prefetch((char*)&sumend_data[i + 32], _MM_HINT_T0);
            
            __m256 v_summand = _mm256_load_ps(&summand_data[i]);
            __m256 v_sumend = _mm256_load_ps(&sumend_data[i]);
            __m256 v_scaled_sumend = _mm256_mul_ps(v_sumend, v_alpha);
            __m256 v_result = _mm256_add_ps(v_summand, v_scaled_sumend);
            _mm256_store_ps(&result_data[i], v_result);
        }
    } else {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&summand_data[i + 32], _MM_HINT_T0);
            _mm_prefetch((char*)&sumend_data[i + 32], _MM_HINT_T0);
            
            __m256 v_summand = _mm256_loadu_ps(&summand_data[i]);
            __m256 v_sumend = _mm256_loadu_ps(&sumend_data[i]);
            __m256 v_scaled_sumend = _mm256_mul_ps(v_sumend, v_alpha);
            __m256 v_result = _mm256_add_ps(v_summand, v_scaled_sumend);
            _mm256_storeu_ps(&result_data[i], v_result);
        }
    }
    
    // Scalar processing for remainder
    for(; i < end; i++) {
        result_data[i] = summand_data[i] + (sumend_data[i] * alpha);
    }
    
    return NULL;
}

/** @brief
 * See documentation at declaration
 * 
 ** @see nnl2_own_paxpy_int64
 **/
void* nnl2_own_paxpy_int64(void* arg) {
    axpy_ptask* task = (axpy_ptask*)arg;
    const int64_t* summand_data = (const int64_t*)task->summand->data;
    const int64_t* sumend_data = (const int64_t*)task->sumend->data;
    int64_t* result_data = (int64_t*)task->result->data;
    size_t start = task->start;
    size_t end = task->end;
    int64_t alpha = task->alpha.int64_alpha;
    
    size_t i = start;
    
    for(; i + 3 < end; i += 4) {
        // Prefetch next cache lines
        _mm_prefetch((char*)&summand_data[i + 16], _MM_HINT_T0);
        _mm_prefetch((char*)&sumend_data[i + 16], _MM_HINT_T0);
        _mm_prefetch((char*)&result_data[i + 16], _MM_HINT_T1);
        
        result_data[i] = summand_data[i] + (sumend_data[i] * alpha);
        result_data[i+1] = summand_data[i+1] + (sumend_data[i+1] * alpha);
        result_data[i+2] = summand_data[i+2] + (sumend_data[i+2] * alpha);
        result_data[i+3] = summand_data[i+3] + (sumend_data[i+3] * alpha);
    }
    
    for(; i < end; i++) {
        result_data[i] = summand_data[i] + (sumend_data[i] * alpha);
    }
    
    return NULL;
}

/** @brief
 * See documentation at declaration
 * 
 ** @see nnl2_own_paxpy_int32
 **/
void* nnl2_own_paxpy_int32(void* arg) {
    axpy_ptask* task = (axpy_ptask*)arg;
    const int32_t* summand_data = (const int32_t*)task->summand->data;
    const int32_t* sumend_data = (const int32_t*)task->sumend->data;
    int32_t* result_data = (int32_t*)task->result->data;
    size_t start = task->start;
    size_t end = task->end;
    int32_t alpha = task->alpha.int32_alpha;
    
    // Create AVX256 vector with alpha value repeated
    __m256i v_alpha = _mm256_set1_epi32(alpha);
    
    size_t i = start;
    
    // AVX256 processing with prefetching (8 elements per iteration)
    if(task->aligned) {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&summand_data[i + 32], _MM_HINT_T0);
            _mm_prefetch((char*)&sumend_data[i + 32], _MM_HINT_T0);
            
            __m256i v_summand = _mm256_load_si256((__m256i*)&summand_data[i]);
            __m256i v_sumend = _mm256_load_si256((__m256i*)&sumend_data[i]);
            __m256i v_scaled_sumend = _mm256_mullo_epi32(v_sumend, v_alpha);
            __m256i v_result = _mm256_add_epi32(v_summand, v_scaled_sumend);
            _mm256_store_si256((__m256i*)&result_data[i], v_result);
        }
    } else {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&summand_data[i + 32], _MM_HINT_T0);
            _mm_prefetch((char*)&sumend_data[i + 32], _MM_HINT_T0);
            
            __m256i v_summand = _mm256_loadu_si256((__m256i*)&summand_data[i]);
            __m256i v_sumend = _mm256_loadu_si256((__m256i*)&sumend_data[i]);
            __m256i v_scaled_sumend = _mm256_mullo_epi32(v_sumend, v_alpha);
            __m256i v_result = _mm256_add_epi32(v_summand, v_scaled_sumend);
            _mm256_storeu_si256((__m256i*)&result_data[i], v_result);
        }
    }
    
    // Scalar processing for remainder
    for(; i < end; i++) {
        result_data[i] = summand_data[i] + (sumend_data[i] * alpha);
    }
    
    return NULL;
}

#endif

/**
 * @ingroup backend_system
 * @brief Backend implementations for AXPY operation
 */
nnl2_runtime_implementation axpy_backends[] = {
    REGISTER_BACKEND(naive_axpy, nnl2_naive, NAIVE_BACKEND_NAME),
    
    #if defined(NNL2_AVX256_AVAILABLE) && defined(NNL2_PTHREAD_AVAILABLE)
        REGISTER_BACKEND(nnl2_own_axpy, nnl2_own, NNL2_OWN_NAME),
    #endif
};  

/**
 * @brief Function pointer for AXPY operation
 * @ingroup backend_system
 */
axpyfn axpy;
MAKE_CURRENT_BACKEND(axpy);

/**
 * @brief Sets the backend for AXPY operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for AXPY operation
 */
void set_axpy_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(axpy_backends, axpy, backend_name, CURRENT_BACKEND(axpy));
}

/**
 * @brief Gets the name of the current backend for AXPY operation
 * @ingroup backend_system
 * @return const char* Name of the current backend
 */
const char* get_axpy_backend() {
    return CURRENT_BACKEND(axpy);
}

/**
 * @brief Gets the list of available backends for AXPY operation
 * @ingroup backend_system
 * @return const char** Array of backend names
 */
DEFINE_GET_BACKENDS_FUNCTION(axpy);

/**
 * @brief Gets the number of available backends for AXPY operation
 * @ingroup backend_system
 * @return size_t Number of available backends
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(axpy);

#endif /** NNL2_AXPY_H **/
