#ifndef NNL2_CORRESPONDENCE_AXPY_H
#define NNL2_CORRESPONDENCE_AXPY_H

/** @brief
 * Performs element-wise AXPF operation (scalar AXPY)
 * Computes: result = summand + alpha * sumend (where sumend is a scalar)
 *
 ** @param summand
 * Pointer to the input tensor
 *
 ** @param sumend
 * Pointer to the scalar value to be scaled and added
 *
 ** @param alpha
 * Scalar multiplier for the sumend value
 *
 ** @return
 * Pointer to a new tensor containing the result of the AXPF operation 
 * (or NULL in case of fail)
 */
nnl2_tensor* naive_axpf(nnl2_tensor* summand, void* sumend, float alpha) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    nnl2_tensor* result = nnl2_empty(summand->shape, summand->rank, summand->dtype);
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(result == NULL) {
            NNL2_ERROR("Failed to allocate new tensor");
        }
    #endif
    
    size_t total_elems = nnl2_product(summand->shape, summand->rank);
    if(total_elems == 0) return result;
    
    switch(summand->dtype) {
        case FLOAT64: {
            double* cast_data_original = (double*)summand->data; 
            double* cast_data_result = (double*)result->data;
            double cast_sumend = *((double*)sumend);
            double alpha_double = (double)alpha;
            for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = cast_data_original[i] + (cast_sumend * alpha_double);
            break;
        }
        
        case FLOAT32: {
            float* cast_data_original = (float*)summand->data;
            float* cast_data_result = (float*)result->data;
            float cast_sumend = *((float*)sumend);
            for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = cast_data_original[i] + (cast_sumend * alpha);
            break;
        }
        
        case INT32: {
            int32_t* cast_data_original = (int32_t*)summand->data;
            int32_t* cast_data_result = (int32_t*)result->data;
            int32_t cast_sumend = *((int32_t*)sumend);
            int32_t alpha_int = (int32_t)alpha;
            for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = cast_data_original[i] + (cast_sumend * alpha_int);
            break;
        }
        
        default: {
            NNL2_TYPE_ERROR(summand->dtype);
            nnl2_free_tensor(result);
            return NULL;
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return result;
}

#ifdef NNL2_PTHREAD_AVAILABLE

/** @brief
 * Threshold for enabling parallel execution of AXPF operation
 */
#define NNL2_AXPF_PARALLEL_THRESHOLD 1000000

/** @brief
 * Worker function for parallel double precision AXPF operation
 * 
 ** @param arg 
 * Pointer to axpf_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 * 
 ** @details
 * Processes FLOAT64 data using AVX256 instructions with cache prefetching
 * for optimal memory access patterns in AXPF operation
 */
void* nnl2_own_paxpf_float64(void* arg);

/** @brief
 * Worker function for parallel single precision AXPF operation
 * 
 ** @param arg 
 * Pointer to axpf_ptask structure containing thread parameters  
 * 
 ** @return 
 * NULL (for pthread API compatibility)
 * 
 ** @see nnl2_own_paxpf_float64
 **/
void* nnl2_own_paxpf_float32(void* arg);

/** @brief
 * Worker function for parallel integer AXPF operation
 * 
 ** @param arg 
 * Pointer to axpf_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 * 
 ** @see nnl2_own_paxpf_float64
 **/
void* nnl2_own_paxpf_int32(void* arg);

/** @brief
 * High-performance parallel implementation of AXPF operation (scalar AXPY)
 * 
 ** @param summand 
 * Pointer to the summand tensor
 *
 ** @param sumend 
 * Pointer to scalar value to be scaled and added
 *
 ** @param alpha
 * Scalar multiplier for the sumend value
 * 
 ** @return
 * Pointer to new tensor containing AXPF operation result
 *
 ** @details
 * Combines AVX256 vectorization, multi-threading with pthread, and cache
 * prefetching for maximum performance on modern CPU architectures.
 * Automatically selects optimal thread count and chunk sizes.
 * 
 ** @note
 * Uses NNL2_NUM_THREADS for parallelization configuration
 * Falls back to naive implementation for small tensors
 * 
 ** @warning
 * Requires pthread support and AVX256 capable CPU
 * Creates a new tensor for the result
 */
nnl2_tensor* nnl2_own_axpf(const nnl2_tensor* summand, void* sumend, float alpha) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks at the maximum safety level
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(summand, "Summand tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(summand->data, "Summand tensor data is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(sumend, "Sumend pointer is NULL", NULL);
    #endif
    
    size_t total_elems = nnl2_product(summand->shape, summand->rank);
    
    // Create output tensor
    nnl2_tensor* result = nnl2_empty(summand->shape, summand->rank, summand->dtype);
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
    
    // Fallback to naive implementation for small tensors
    if(total_elems < NNL2_AXPF_PARALLEL_THRESHOLD) {
        nnl2_tensor* naive_result = naive_axpf((nnl2_tensor*)summand, sumend, alpha);
        if (naive_result != NULL) {
            // Copy data from naive result to our result tensor
            memcpy(result->data, naive_result->data, total_elems * get_dtype_size(summand->dtype));
            nnl2_free_tensor(naive_result);
        }
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return result;
    }
    
    nnl2_tensor_type dtype = summand->dtype;
    bool is_aligned = NNL2_IS_ALIGNED(summand->data, NNL2_TENSOR_ALIGNMENT_32) &&
                      NNL2_IS_ALIGNED(result->data, NNL2_TENSOR_ALIGNMENT_32);
    
    // Warning for unaligned memory in safety modes
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!is_aligned) {
            NNL2_WARN("In nnl2_own_axpf, tensor memory is not aligned to 32 bytes. Performance may be suboptimal");
        }
    #endif
    
    size_t num_threads = NNL2_NUM_THREADS;
    pthread_t threads[num_threads];
    axpf_ptask tasks[num_threads];
    
    // Calculate optimal chunk sizes with load balancing
    size_t chunk = total_elems / num_threads;
    size_t remainder = total_elems % num_threads;
    
    // Configure sumend and alpha values based on data type
    for (size_t i = 0; i < num_threads; i++) {
        tasks[i].summand = summand;
        tasks[i].result = result;
        tasks[i].dtype = dtype;
        tasks[i].aligned = is_aligned;
        
        switch(dtype) {
            case FLOAT64: 
                tasks[i].sumend.float64_sumend = *((double*)sumend);
                tasks[i].alpha.float64_alpha = (double)alpha;
                break;
            case FLOAT32: 
                tasks[i].sumend.float32_sumend = *((float*)sumend);
                tasks[i].alpha.float32_alpha = alpha;
                break;
            case INT32: 
                tasks[i].sumend.int32_sumend = *((int32_t*)sumend);
                tasks[i].alpha.int32_alpha = (int32_t)alpha;
                break;
            default: {
                NNL2_TYPE_ERROR(dtype);
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
        switch(dtype) {
            case FLOAT64: worker_func = nnl2_own_paxpf_float64; break;
            case FLOAT32: worker_func = nnl2_own_paxpf_float32; break;
            case INT32:   worker_func = nnl2_own_paxpf_int32;   break;
            default: {
                NNL2_TYPE_ERROR(dtype);
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
            NNL2_THREAD_CREATE_ERROR(status, "nnl2_own_axpf");
            num_threads = i;
            break;
        }
        
        current_start += current_chunk;
    }
    
    // Wait for all threads to complete
    for (size_t i = 0; i < num_threads; i++) {
        int join_status = pthread_join(threads[i], NULL);
        if(join_status != 0) {
            NNL2_THREAD_JOIN_ERROR(join_status, "nnl2_own_axpf");
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
 ** @see nnl2_own_paxpf_float64
 **/
void* nnl2_own_paxpf_float64(void* arg) {
    axpf_ptask* task = (axpf_ptask*)arg;
    const double* summand_data = (const double*)task->summand->data;
    double* result_data = (double*)task->result->data;
    size_t start = task->start;
    size_t end = task->end;
    double sumend = task->sumend.float64_sumend;
    double alpha = task->alpha.float64_alpha;
    
    // Precompute scaled sumend value
    double scaled_sumend = sumend * alpha;
    
    // Create AVX256 vector with scaled sumend value repeated
    __m256d v_scaled_sumend = _mm256_set1_pd(scaled_sumend);
    
    size_t i = start;
    
    // AVX256 processing with prefetching
    if(task->aligned) {
        for(; i + 3 < end; i += 4) {
            // Prefetch next cache line
            _mm_prefetch((char*)&summand_data[i + 16], _MM_HINT_T0);
            
            __m256d v_summand = _mm256_load_pd(&summand_data[i]);
            __m256d v_result = _mm256_add_pd(v_summand, v_scaled_sumend);
            _mm256_store_pd(&result_data[i], v_result);
        }
    } else {
        for(; i + 3 < end; i += 4) {
            _mm_prefetch((char*)&summand_data[i + 16], _MM_HINT_T0);
            
            __m256d v_summand = _mm256_loadu_pd(&summand_data[i]);
            __m256d v_result = _mm256_add_pd(v_summand, v_scaled_sumend);
            _mm256_storeu_pd(&result_data[i], v_result);
        }
    }
    
    // Scalar processing for remainder
    for(; i < end; i++) {
        result_data[i] = summand_data[i] + scaled_sumend;
    }
    
    return NULL;
}

/** @brief
 * See documentation at declaration
 * 
 ** @see nnl2_own_paxpf_float32
 **/
void* nnl2_own_paxpf_float32(void* arg) {
    axpf_ptask* task = (axpf_ptask*)arg;
    const float* summand_data = (const float*)task->summand->data;
    float* result_data = (float*)task->result->data;
    size_t start = task->start;
    size_t end = task->end;
    float sumend = task->sumend.float32_sumend;
    float alpha = task->alpha.float32_alpha;
    
    // Precompute scaled sumend value
    float scaled_sumend = sumend * alpha;
    
    // Create AVX256 vector with scaled sumend value repeated
    __m256 v_scaled_sumend = _mm256_set1_ps(scaled_sumend);
    
    size_t i = start;
    
    // AVX256 processing with prefetching (8 elements per iteration)
    if(task->aligned) {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&summand_data[i + 32], _MM_HINT_T0);
            
            __m256 v_summand = _mm256_load_ps(&summand_data[i]);
            __m256 v_result = _mm256_add_ps(v_summand, v_scaled_sumend);
            _mm256_store_ps(&result_data[i], v_result);
        }
    } else {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&summand_data[i + 32], _MM_HINT_T0);
            
            __m256 v_summand = _mm256_loadu_ps(&summand_data[i]);
            __m256 v_result = _mm256_add_ps(v_summand, v_scaled_sumend);
            _mm256_storeu_ps(&result_data[i], v_result);
        }
    }
    
    // Scalar processing for remainder
    for(; i < end; i++) {
        result_data[i] = summand_data[i] + scaled_sumend;
    }
    
    return NULL;
}

/** @brief
 * See documentation at declaration
 * 
 ** @see nnl2_own_paxpf_int32
 **/
void* nnl2_own_paxpf_int32(void* arg) {
    axpf_ptask* task = (axpf_ptask*)arg;
    const int32_t* summand_data = (const int32_t*)task->summand->data;
    int32_t* result_data = (int32_t*)task->result->data;
    size_t start = task->start;
    size_t end = task->end;
    int32_t sumend = task->sumend.int32_sumend;
    int32_t alpha = task->alpha.int32_alpha;
    
    // Precompute scaled sumend value
    int32_t scaled_sumend = sumend * alpha;
    
    // Create AVX256 vector with scaled sumend value repeated
    __m256i v_scaled_sumend = _mm256_set1_epi32(scaled_sumend);
    
    size_t i = start;
    
    // AVX256 processing with prefetching (8 elements per iteration)
    if(task->aligned) {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&summand_data[i + 32], _MM_HINT_T0);
            
            __m256i v_summand = _mm256_load_si256((__m256i*)&summand_data[i]);
            __m256i v_result = _mm256_add_epi32(v_summand, v_scaled_sumend);
            _mm256_store_si256((__m256i*)&result_data[i], v_result);
        }
    } else {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&summand_data[i + 32], _MM_HINT_T0);
            
            __m256i v_summand = _mm256_loadu_si256((__m256i*)&summand_data[i]);
            __m256i v_result = _mm256_add_epi32(v_summand, v_scaled_sumend);
            _mm256_storeu_si256((__m256i*)&result_data[i], v_result);
        }
    }
    
    // Scalar processing for remainder
    for(; i < end; i++) {
        result_data[i] = summand_data[i] + scaled_sumend;
    }
    
    return NULL;
}

#endif

/**
 * @ingroup backend_system
 * @brief Backend implementations for AXPF operation
 */
nnl2_runtime_implementation axpf_backends[] = {
    REGISTER_BACKEND(naive_axpf, nnl2_naive, NAIVE_BACKEND_NAME),
    
    #if defined(NNL2_AVX256_AVAILABLE) && defined(NNL2_PTHREAD_AVAILABLE)
        REGISTER_BACKEND(nnl2_own_axpf, nnl2_own, NNL2_OWN_NAME),
    #endif
};  

/**
 * @brief Function pointer for AXPF operation
 * @ingroup backend_system
 */
axpffn axpf;

/**
 * @brief Sets the backend for AXPF operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for AXPF operation
 */
void set_axpf_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(axpf_backends, axpf, backend_name);
}

#endif /** NNL2_CORRESPONDENCE_AXPY_H **/
