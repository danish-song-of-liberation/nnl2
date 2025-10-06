#ifndef NNL2_CORRESPONDENCE_MIN_H
#define NNL2_CORRESPONDENCE_MIN_H

/** @brief
 * Performs element-wise minimum operation between tensor elements and a scalar value
 *
 ** @param tensor
 * Pointer to the input tensor to be processed
 *
 ** @param threshold
 * Pointer to the scalar threshold value for minimum operation
 *
 ** @return
 * Pointer to a new tensor containing the result of the minimum operation 
 * (or NULL in case of failure)
 */
Tensor* naive_min_minf(const Tensor* tensor, void* threshold) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    Tensor* result = nnl2_empty(tensor->shape, tensor->rank, tensor->dtype);
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(result == NULL) {
            NNL2_ERROR("Failed to allocate new tensor");
            return NULL;
        }
    #endif
    
    size_t total_elems = product(tensor->shape, tensor->rank);
    if(total_elems == 0) return result;
    
    switch(tensor->dtype) {
        case FLOAT64: {
            double* cast_data_original = (double*)tensor->data;
            double* cast_data_result = (double*)result->data;
            double min_val = *((double*)threshold);
            for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = MIN(cast_data_original[i], min_val);
            break;
        }
        
        case FLOAT32: {
            float* cast_data_original = (float*)tensor->data;
            float* cast_data_result = (float*)result->data;
            float min_val = *((float*)threshold);
            for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = MIN(cast_data_original[i], min_val);
            break;
        }
        
        case INT32: {
            int32_t* cast_data_original = (int32_t*)tensor->data;
            int32_t* cast_data_result = (int32_t*)result->data;
            int32_t min_val = *((int32_t*)threshold);
            for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = MIN(cast_data_original[i], min_val);
            break;
        }
        
        default: {
            NNL2_TYPE_ERROR(tensor->dtype);
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
 * Threshold for enabling parallel execution of element-wise minimum operation
 */
#define NNL2_MIN_MINF_PARALLEL_THRESHOLD 1000000

/** @brief
 * Worker function for parallel double precision element-wise minimum operation
 * 
 ** @param arg 
 * Pointer to min_minf_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 * 
 ** @details
 * Processes FLOAT64 data using AVX256 instructions with cache prefetching
 * for optimal memory access patterns
 */
void* nnl2_own_pmin_minf_float64(void* arg);

/** @brief
 * Worker function for parallel single precision element-wise minimum operation
 * 
 ** @param arg 
 * Pointer to min_minf_ptask structure containing thread parameters  
 * 
 ** @return 
 * NULL (for pthread API compatibility)
 * 
 ** @see nnl2_own_pmin_minf_float64
 **/
void* nnl2_own_pmin_minf_float32(void* arg);

/** @brief
 * Worker function for parallel integer element-wise minimum operation
 * 
 ** @param arg 
 * Pointer to min_minf_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 * 
 ** @see nnl2_own_pmin_minf_float64
 **/
void* nnl2_own_pmin_minf_int32(void* arg);

/** @brief
 * High-performance parallel implementation of element-wise minimum operation
 * 
 ** @param tensor 
 * Pointer to the input tensor
 *
 ** @param threshold 
 * Pointer to scalar threshold value for minimum operation
 * 
 ** @return
 * Pointer to new tensor containing element-wise minimum values
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
 */
Tensor* nnl2_own_min_minf(const Tensor* tensor, void* threshold) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks at the maximum safety level
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor, "Tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor->data, "Tensor data is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(threshold, "Threshold pointer is NULL", NULL);
    #endif
    
    size_t total_elems = product(tensor->shape, tensor->rank);
    
    // Create output tensor
    Tensor* result = nnl2_empty(tensor->shape, tensor->rank, tensor->dtype);
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
    if(total_elems < NNL2_MIN_MINF_PARALLEL_THRESHOLD) {
        Tensor* naive_result = naive_min_minf(tensor, threshold);
        if (naive_result != NULL) {
            // Copy data from naive result to our result tensor
            memcpy(result->data, naive_result->data, total_elems * get_dtype_size(tensor->dtype));
            nnl2_free_tensor(naive_result);
        }
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return result;
    }
    
    TensorType dtype = tensor->dtype;
    bool is_aligned = NNL2_IS_ALIGNED(tensor->data, NNL2_TENSOR_ALIGNMENT_32) &&
                      NNL2_IS_ALIGNED(result->data, NNL2_TENSOR_ALIGNMENT_32);
    
    // Warning for unaligned memory in safety modes
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!is_aligned) {
            NNL2_WARN("In nnl2_own_min_minf, tensor memory is not aligned to 32 bytes. Performance may be suboptimal");
        }
    #endif
    
    size_t num_threads = NNL2_NUM_THREADS;
    pthread_t threads[num_threads];
    min_minf_ptask tasks[num_threads];
    
    // Calculate optimal chunk sizes with load balancing
    size_t chunk = total_elems / num_threads;
    size_t remainder = total_elems % num_threads;
    
    // Configure threshold value based on data type
    for (size_t i = 0; i < num_threads; i++) {
        tasks[i].tensor = tensor;
        tasks[i].result = result;
        tasks[i].dtype = dtype;
        tasks[i].aligned = is_aligned;
        
        switch(dtype) {
            case FLOAT64: tasks[i].threshold.float64_threshold = *((double*)threshold); break;
            case FLOAT32: tasks[i].threshold.float32_threshold = *((float*)threshold);  break;
            case INT32:   tasks[i].threshold.int32_threshold = *((int32_t*)threshold);  break;
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
            case FLOAT64: worker_func = nnl2_own_pmin_minf_float64; break;
            case FLOAT32: worker_func = nnl2_own_pmin_minf_float32; break;
            case INT32:   worker_func = nnl2_own_pmin_minf_int32;   break;
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
            NNL2_THREAD_CREATE_ERROR(status, "nnl2_own_min_minf");
            num_threads = i;
            break;
        }
        
        current_start += current_chunk;
    }
    
    // Wait for all threads to complete
    for (size_t i = 0; i < num_threads; i++) {
        int join_status = pthread_join(threads[i], NULL);
        if(join_status != 0) {
            NNL2_THREAD_JOIN_ERROR(join_status, "nnl2_own_min_minf");
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
 ** @see nnl2_own_pmin_minf_float64
 **/
void* nnl2_own_pmin_minf_float64(void* arg) {
    min_minf_ptask* task = (min_minf_ptask*)arg;
    const double* data = (const double*)task->tensor->data;
    double* result_data = (double*)task->result->data;
    size_t start = task->start;
    size_t end = task->end;
    double threshold = task->threshold.float64_threshold;
    
    // Create AVX256 vector with threshold value repeated
    __m256d v_threshold = _mm256_set1_pd(threshold);
    
    size_t i = start;
    
    // AVX256 processing with prefetching
    if(task->aligned) {
        for(; i + 3 < end; i += 4) {
            // Prefetch next cache line
            _mm_prefetch((char*)&data[i + 16], _MM_HINT_T0);
            
            __m256d v_data = _mm256_load_pd(&data[i]);
            __m256d v_result = _mm256_min_pd(v_data, v_threshold);
            _mm256_store_pd(&result_data[i], v_result);
        }
    } else {
        for(; i + 3 < end; i += 4) {
            _mm_prefetch((char*)&data[i + 16], _MM_HINT_T0);
            
            __m256d v_data = _mm256_loadu_pd(&data[i]);
            __m256d v_result = _mm256_min_pd(v_data, v_threshold);
            _mm256_storeu_pd(&result_data[i], v_result);
        }
    }
    
    // Scalar processing for remainder
    for(; i < end; i++) {
        result_data[i] = MIN(data[i], threshold);
    }
    
    return NULL;
}

/** @brief
 * See documentation at declaration
 * 
 ** @see nnl2_own_pmin_minf_float32
 **/
void* nnl2_own_pmin_minf_float32(void* arg) {
    min_minf_ptask* task = (min_minf_ptask*)arg;
    const float* data = (const float*)task->tensor->data;
    float* result_data = (float*)task->result->data;
    size_t start = task->start;
    size_t end = task->end;
    float threshold = task->threshold.float32_threshold;
    
    // Create AVX256 vector with threshold value repeated
    __m256 v_threshold = _mm256_set1_ps(threshold);
    
    size_t i = start;
    
    // AVX256 processing with prefetching (8 elements per iteration)
    if(task->aligned) {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&data[i + 32], _MM_HINT_T0);
            
            __m256 v_data = _mm256_load_ps(&data[i]);
            __m256 v_result = _mm256_min_ps(v_data, v_threshold);
            _mm256_store_ps(&result_data[i], v_result);
        }
    } else {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&data[i + 32], _MM_HINT_T0);
            
            __m256 v_data = _mm256_loadu_ps(&data[i]);
            __m256 v_result = _mm256_min_ps(v_data, v_threshold);
            _mm256_storeu_ps(&result_data[i], v_result);
        }
    }
    
    // Scalar processing for remainder
    for(; i < end; i++) {
        result_data[i] = MIN(data[i], threshold);
    }
    
    return NULL;
}

/** @brief
 * See documentation at declaration
 * 
 ** @see nnl2_own_pmin_minf_int32
 **/
void* nnl2_own_pmin_minf_int32(void* arg) {
    min_minf_ptask* task = (min_minf_ptask*)arg;
    const int32_t* data = (const int32_t*)task->tensor->data;
    int32_t* result_data = (int32_t*)task->result->data;
    size_t start = task->start;
    size_t end = task->end;
    int32_t threshold = task->threshold.int32_threshold;
    
    // Create AVX256 vector with threshold value repeated
    __m256i v_threshold = _mm256_set1_epi32(threshold);
    
    size_t i = start;
    
    // AVX256 processing with prefetching (8 elements per iteration)
    if(task->aligned) {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&data[i + 32], _MM_HINT_T0);
            
            __m256i v_data = _mm256_load_si256((__m256i*)&data[i]);
            
            // For integers, we need to compare and select minimum
            __m256i v_compare = _mm256_cmpgt_epi32(v_data, v_threshold);
            __m256i v_result = _mm256_blendv_epi8(v_data, v_threshold, v_compare);
            
            _mm256_store_si256((__m256i*)&result_data[i], v_result);
        }
    } else {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&data[i + 32], _MM_HINT_T0);
            
            __m256i v_data = _mm256_loadu_si256((__m256i*)&data[i]);
            
            __m256i v_compare = _mm256_cmpgt_epi32(v_data, v_threshold);
            __m256i v_result = _mm256_blendv_epi8(v_data, v_threshold, v_compare);
            
            _mm256_storeu_si256((__m256i*)&result_data[i], v_result);
        }
    }
    
    // Scalar processing for remainder
    for(; i < end; i++) {
        result_data[i] = MIN(data[i], threshold);
    }
    
    return NULL;
}

#endif

/**
 * @ingroup backend_system
 * @brief Backend implementations for element-wise minimum operation
 * @details
 * Array follows the common backend registration pattern for element-wise minimum
 * operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for element-wise minimum
 * 
 * @see nnl2_naive
 * @see naive_min_minf
 */
Implementation min_minf_backends[] = {
    REGISTER_BACKEND(naive_min_minf, nnl2_naive, NAIVE_BACKEND_NAME),
    
    #if defined(NNL2_AVX256_AVAILABLE) && defined(NNL2_PTHREAD_AVAILABLE)
        REGISTER_BACKEND(nnl2_own_min_minf, nnl2_own, NNL2_OWN_NAME),
    #endif
};    

/**
 * @brief Function pointer for element-wise minimum operation
 * @ingroup backend_system
 */
minminffn min_minf;

/** 
 * @brief Sets the backend for element-wise minimum operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for minimum operation
 * @see SET_BACKEND_BY_NAME
 * @see min_minf_backends
 */
void set_min_minf_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(min_minf_backends, min_minf, backend_name);
}

#endif /** NNL2_CORRESPONDENCE_MIN_H **/
