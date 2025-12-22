#ifndef NNL2_CORRESPONDENCE_AXPY_INPLACE_H
#define NNL2_CORRESPONDENCE_AXPY_INPLACE_H

/** @brief 
 * Performs element-wise AXPF operation (scalar AXPY) in place
 * Computes: summand = summand + alpha * sumend (where sumend is a scalar)
 * 
 ** @param summand 
 * Pointer to the tensor that will be modified in place
 * 
 ** @param sumend 
 * Pointer to the scalar value to be scaled and added
 * 
 ** @param alpha
 * Scalar multiplier for the sumend value
 */
void naive_axpf_inplace(nnl2_tensor* summand, void* sumend, float alpha) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks for max safety level 
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(summand, "Summand tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(sumend, "Sumend scalar is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(summand->shape, "Summand shape is NULL");
    #endif
    
    size_t total_elems = nnl2_product(summand->shape, summand->rank);
    if(total_elems == 0) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return;
    }
    
    switch(summand->dtype) {
        case FLOAT64: {
            double* cast_summand = (double*)summand->data;
            double cast_sumend = *((double*)sumend);
            double alpha_double = (double)alpha;
            for(size_t i = 0; i < total_elems; i++) cast_summand[i] += cast_sumend * alpha_double;
            break;
        }
        
        case FLOAT32: {
            float* cast_summand = (float*)summand->data;
            float cast_sumend = *((float*)sumend);
            for(size_t i = 0; i < total_elems; i++) cast_summand[i] += cast_sumend * alpha;
            break;
        }
        
        case INT32: {
            int32_t* cast_summand = (int32_t*)summand->data;
            int32_t cast_sumend = *((int32_t*)sumend);
            int32_t alpha_int = (int32_t)alpha;
            for(size_t i = 0; i < total_elems; i++) cast_summand[i] += cast_sumend * alpha_int;
            break;
        }
        
        default: {
            NNL2_TYPE_ERROR(summand->dtype);
            return;
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

#ifdef NNL2_PTHREAD_AVAILABLE

/** @brief
 * Threshold for enabling parallel execution of AXPF in-place operation
 */
#define NNL2_AXPF_INPLACE_PARALLEL_THRESHOLD 100000

/** @brief
 * Worker function for parallel double precision AXPF in-place operation
 * 
 ** @param arg 
 * Pointer to axpf_inplace_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 * 
 ** @details
 * Processes FLOAT64 data using AVX256 instructions with cache prefetching
 * for optimal memory access patterns in scalar AXPY in-place operation
 */
void* nnl2_own_paxpf_inplace_float64(void* arg);

/** @brief
 * Worker function for parallel single precision AXPF in-place operation
 * 
 ** @param arg 
 * Pointer to axpf_inplace_ptask structure containing thread parameters  
 * 
 ** @return 
 * NULL (for pthread API compatibility)
 * 
 ** @see nnl2_own_paxpf_inplace_float64
 **/
void* nnl2_own_paxpf_inplace_float32(void* arg);

/** @brief
 * Worker function for parallel integer AXPF in-place operation
 * 
 ** @param arg 
 * Pointer to axpf_inplace_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 * 
 ** @see nnl2_own_paxpf_inplace_float64
 **/
void* nnl2_own_paxpf_inplace_int32(void* arg);

/** @brief
 * High-performance parallel implementation of AXPF in-place operation
 * 
 ** @param summand 
 * Pointer to the summand tensor (will be modified in-place)
 *
 ** @param sumend 
 * Pointer to the scalar sumend value
 *
 ** @param alpha
 * Scalar multiplier for the sumend value
 * 
 ** @details
 * Combines AVX256 vectorization, multi-threading with pthread, and cache
 * prefetching for maximum performance on modern CPU architectures.
 * Optimized for scalar operations with efficient memory access patterns.
 * 
 ** @note
 * Uses NNL2_NUM_THREADS for parallelization configuration
 * Falls back to naive implementation for small tensors
 * 
 ** @warning
 * Requires pthread support and AVX256 capable CPU
 * Modifies the summand tensor directly
 * Only supports FLOAT64, FLOAT32, and INT32 data types
 */
void nnl2_own_axpf_inplace(nnl2_tensor* summand, void* sumend, float alpha) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks for max safety level 
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(summand, "Summand tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(sumend, "Sumend scalar is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(summand->shape, "Summand shape is NULL");
    #endif
    
    size_t total_elems = nnl2_product(summand->shape, summand->rank);
    if(total_elems == 0) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return;
    }
    
    nnl2_tensor_type summand_dtype = summand->dtype;
    
    // Fallback to naive implementation for small tensors or unsupported types
    if(total_elems < NNL2_AXPF_INPLACE_PARALLEL_THRESHOLD) {
        naive_axpf_inplace(summand, sumend, alpha);
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return;
    }
    
    bool is_aligned = NNL2_IS_ALIGNED(summand->data, NNL2_TENSOR_ALIGNMENT_32);
    
    // Warning for unaligned memory in safety modes
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!is_aligned) {
            NNL2_WARN("In nnl2_own_axpf_inplace, tensor memory is not aligned to 32 bytes. Performance may be suboptimal");
        }
    #endif
    
    size_t num_threads = NNL2_NUM_THREADS;
    
    // Limit threads based on number of elements
    num_threads = MIN(num_threads, total_elems);
    
    pthread_t threads[num_threads];
    axpf_inplace_ptask tasks[num_threads];
    
    // Calculate optimal element distribution with load balancing
    size_t elems_per_thread = total_elems / num_threads;
    size_t remainder_elems = total_elems % num_threads;
    
    // Configure common task parameters
    for (size_t i = 0; i < num_threads; i++) {
        tasks[i].summand = summand;
        tasks[i].sumend = sumend;
        tasks[i].summand_dtype = summand_dtype;
        tasks[i].aligned = is_aligned;
        
        // Precompute alpha and sumend values for each data type
        switch(summand_dtype) {
            case FLOAT64: 
                tasks[i].alpha.float64_alpha = (double)alpha;
                tasks[i].sumend_val.float64_sumend = *((double*)sumend);
                break;
            case FLOAT32: 
                tasks[i].alpha.float32_alpha = alpha;
                tasks[i].sumend_val.float32_sumend = *((float*)sumend);
                break;
            case INT32:   
                tasks[i].alpha.int32_alpha = (int32_t)alpha;
                tasks[i].sumend_val.int32_sumend = *((int32_t*)sumend);
                break;
            default: {
                NNL2_TYPE_ERROR(summand_dtype);
                #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                    NNL2_FUNC_EXIT();
                #endif
                return;
            }
        }
    }
    
    size_t current_index_start = 0;
    for (size_t i = 0; i < num_threads; i++) {
        size_t current_elems = elems_per_thread + (i < remainder_elems ? 1 : 0);
        
        // Configure task for this thread
        tasks[i].start_index = current_index_start;
        tasks[i].end_index = current_index_start + current_elems;
        
        // Select appropriate worker function based on data type
        void* (*worker_func)(void*) = NULL;
        switch(summand_dtype) {
            case FLOAT64: worker_func = nnl2_own_paxpf_inplace_float64; break;
            case FLOAT32: worker_func = nnl2_own_paxpf_inplace_float32; break;
            case INT32:   worker_func = nnl2_own_paxpf_inplace_int32;   break;
            default: {
                NNL2_TYPE_ERROR(summand_dtype);
                #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                    NNL2_FUNC_EXIT();
                #endif
                return;
            }
        }
        
        // Create thread to process the assigned elements
        int status = pthread_create(&threads[i], NULL, worker_func, &tasks[i]);
        if(status != 0) {
            NNL2_THREAD_CREATE_ERROR(status, "nnl2_own_axpf_inplace");
            num_threads = i;
            break;
        }
        
        current_index_start += current_elems;
    }
    
    // Wait for all threads to complete
    for (size_t i = 0; i < num_threads; i++) {
        int join_status = pthread_join(threads[i], NULL);
        if(join_status != 0) {
            NNL2_THREAD_JOIN_ERROR(join_status, "nnl2_own_axpf_inplace");
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
 ** @see nnl2_own_paxpf_inplace_float64
 **/
void* nnl2_own_paxpf_inplace_float64(void* arg) {
    axpf_inplace_ptask* task = (axpf_inplace_ptask*)arg;
    double* summand_data = (double*)task->summand->data;
    size_t start_index = task->start_index;
    size_t end_index = task->end_index;
    double alpha = task->alpha.float64_alpha;
    double sumend_val = task->sumend_val.float64_sumend;
    
    // Precompute scaled sumend value
    double scaled_sumend = sumend_val * alpha;
    
    // Create AVX256 vector with scaled sumend value repeated
    __m256d v_scaled_sumend = _mm256_set1_pd(scaled_sumend);
    
    size_t i = start_index;
    
    // AVX256 processing with prefetching
    if(task->aligned) {
        for(; i + 3 < end_index; i += 4) {
            // Prefetch next cache lines
            _mm_prefetch((char*)&summand_data[i + 16], _MM_HINT_T0);
            
            __m256d v_summand = _mm256_load_pd(&summand_data[i]);
            __m256d v_result = _mm256_add_pd(v_summand, v_scaled_sumend);
            _mm256_store_pd(&summand_data[i], v_result);
        }
    } else {
        for(; i + 3 < end_index; i += 4) {
            _mm_prefetch((char*)&summand_data[i + 16], _MM_HINT_T0);
            
            __m256d v_summand = _mm256_loadu_pd(&summand_data[i]);
            __m256d v_result = _mm256_add_pd(v_summand, v_scaled_sumend);
            _mm256_storeu_pd(&summand_data[i], v_result);
        }
    }
    
    // Scalar processing for remainder elements
    for(; i < end_index; i++) {
        summand_data[i] += scaled_sumend;
    }
    
    return NULL;
}

/** @brief
 * See documentation at declaration
 * 
 ** @see nnl2_own_paxpf_inplace_float32
 **/
void* nnl2_own_paxpf_inplace_float32(void* arg) {
    axpf_inplace_ptask* task = (axpf_inplace_ptask*)arg;
    float* summand_data = (float*)task->summand->data;
    size_t start_index = task->start_index;
    size_t end_index = task->end_index;
    float alpha = task->alpha.float32_alpha;
    float sumend_val = task->sumend_val.float32_sumend;
    
    // Precompute scaled sumend value
    float scaled_sumend = sumend_val * alpha;
    
    // Create AVX256 vector with scaled sumend value repeated
    __m256 v_scaled_sumend = _mm256_set1_ps(scaled_sumend);
    
    size_t i = start_index;
    
    // AVX256 processing with prefetching (8 elements per iteration)
    if(task->aligned) {
        for(; i + 7 < end_index; i += 8) {
            _mm_prefetch((char*)&summand_data[i + 32], _MM_HINT_T0);
            
            __m256 v_summand = _mm256_load_ps(&summand_data[i]);
            __m256 v_result = _mm256_add_ps(v_summand, v_scaled_sumend);
            _mm256_store_ps(&summand_data[i], v_result);
        }
    } else {
        for(; i + 7 < end_index; i += 8) {
            _mm_prefetch((char*)&summand_data[i + 32], _MM_HINT_T0);
            
            __m256 v_summand = _mm256_loadu_ps(&summand_data[i]);
            __m256 v_result = _mm256_add_ps(v_summand, v_scaled_sumend);
            _mm256_storeu_ps(&summand_data[i], v_result);
        }
    }
    
    // Scalar processing for remainder elements
    for(; i < end_index; i++) {
        summand_data[i] += scaled_sumend;
    }
    
    return NULL;
}

/** @brief
 * See documentation at declaration
 * 
 ** @see nnl2_own_paxpf_inplace_int32
 **/
void* nnl2_own_paxpf_inplace_int32(void* arg) {
    axpf_inplace_ptask* task = (axpf_inplace_ptask*)arg;
    int32_t* summand_data = (int32_t*)task->summand->data;
    size_t start_index = task->start_index;
    size_t end_index = task->end_index;
    int32_t alpha = task->alpha.int32_alpha;
    int32_t sumend_val = task->sumend_val.int32_sumend;
    
    // Precompute scaled sumend value
    int32_t scaled_sumend = sumend_val * alpha;
    
    // Create AVX256 vector with scaled sumend value repeated
    __m256i v_scaled_sumend = _mm256_set1_epi32(scaled_sumend);
    
    size_t i = start_index;
    
    // AVX256 processing with prefetching (8 elements per iteration)
    if(task->aligned) {
        for(; i + 7 < end_index; i += 8) {
            _mm_prefetch((char*)&summand_data[i + 32], _MM_HINT_T0);
            
            __m256i v_summand = _mm256_load_si256((__m256i*)&summand_data[i]);
            __m256i v_result = _mm256_add_epi32(v_summand, v_scaled_sumend);
            _mm256_store_si256((__m256i*)&summand_data[i], v_result);
        }
    } else {
        for(; i + 7 < end_index; i += 8) {
            _mm_prefetch((char*)&summand_data[i + 32], _MM_HINT_T0);
            
            __m256i v_summand = _mm256_loadu_si256((__m256i*)&summand_data[i]);
            __m256i v_result = _mm256_add_epi32(v_summand, v_scaled_sumend);
            _mm256_storeu_si256((__m256i*)&summand_data[i], v_result);
        }
    }
    
    // Scalar processing for remainder elements
    for(; i < end_index; i++) {
        summand_data[i] += scaled_sumend;
    }
    
    return NULL;
}

#endif

/**
 * @ingroup backend_system
 * @brief Backend implementations for AXPF in-place operation
 */
nnl2_runtime_implementation axpf_inplace_backends[] = {
    REGISTER_BACKEND(naive_axpf_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
    
    #if defined(NNL2_AVX256_AVAILABLE) && defined(NNL2_PTHREAD_AVAILABLE)
        REGISTER_BACKEND(nnl2_own_axpf_inplace, nnl2_own, NNL2_OWN_NAME),
    #endif
};  

/**
 * @brief Function pointer for AXPF in-place operation
 * @ingroup backend_system
 */
axpfinplacefn axpf_inplace;

/**
 * @brief Sets the backend for AXPF in-place operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for AXPF in-place operation
 */
void set_axpf_inplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(axpf_inplace_backends, axpf_inplace, backend_name);
}

#endif /** NNL2_CORRESPONDENCE_AXPY_INPLACE_H **/
