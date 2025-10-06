#ifndef NNL2_CORRESPONDENCE_SUB_INPLACE_H
#define NNL2_CORRESPONDENCE_SUB_INPLACE_H

/** @brief 
 * Subtracts a scalar value from each element of a tensor (in-place).
 * 
 ** @param tensor 
 * Pointer to the tensor from which the value will be subtracted
 * 
 ** @param dec 
 * Pointer to the scalar value to subtract
 */
void naive_sub_decf_inplace(Tensor* tensor, void* dec) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    size_t total_elems = product(tensor->shape, tensor->rank);
    if(total_elems == 0) return;
    
    switch(tensor->dtype) {
        case FLOAT64: {
            double* cast_data = (double*)tensor->data;  // Casting
            double decrement = *((double*)dec);
            for(size_t i = 0; i < total_elems; i++) cast_data[i] -= decrement;
            break;
        }
        
        case FLOAT32: {
            float* cast_data = (float*)tensor->data;
            float decrement = *((float*)dec);
            for(size_t i = 0; i < total_elems; i++) cast_data[i] -= decrement;
            break;
        }
        
        case INT32: {
            int32_t* cast_data = (int32_t*)tensor->data;
            int32_t decrement = *((int32_t*)dec);
            for(size_t i = 0; i < total_elems; i++) cast_data[i] -= decrement;
            break;
        }
        
        default: {
            NNL2_TYPE_ERROR(tensor->dtype);
            return;
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}    

#ifdef NNL2_PTHREAD_AVAILABLE

/** @brief
 * Threshold for enabling parallel execution of the
 * scalar subtraction in-place operation
 */
#define NNL2_SUB_DECF_INPLACE_PARALLEL_THRESHOLD 1000000

/** @brief 
 * Worker function for parallel double precision scalar subtraction
 * 
 ** @param arg 
 * Pointer to subdecfinplace_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 * 
 ** @details
 * Processes FLOAT64 data using AVX256 instructions with cache prefetching
 * for optimal memory access patterns
 */
void* nnl2_own_psub_decf_float64(void* arg);

/** @brief
 * Worker function for parallel single precision scalar subtraction
 * 
 ** @param arg 
 * Pointer to subdecfinplace_ptask structure containing thread parameters  
 * 
 ** @return 
 * NULL (for pthread API compatibility)
 * 
 ** @see nnl2_own_psub_decf_float64
 **/
void* nnl2_own_psub_decf_float32(void* arg);

/** @brief
 * Worker function for parallel integer scalar subtraction
 * 
 ** @param arg 
 * Pointer to subdecfinplace_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 * 
 ** @see nnl2_own_psub_decf_float64
 **/
void* nnl2_own_psub_decf_int32(void* arg);

/** @brief
 * High-performance parallel implementation of in-place scalar subtraction
 * 
 ** @param tensor 
 * Pointer to tensor that will be modified in-place
 *
 ** @param dec 
 * Pointer to scalar value to subtract from each element
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
void nnl2_own_sub_decf_inplace(Tensor* tensor, void* dec) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks at the maximum safety level
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensor, "Tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensor->data, "Tensor data is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(dec, "Decrement pointer is NULL");
    #endif
    
    size_t total_elems = product(tensor->shape, tensor->rank);
    if(total_elems == 0) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return;
    }
    
    // Fallback to naive implementation for small tensors
    if(total_elems < NNL2_SUB_DECF_INPLACE_PARALLEL_THRESHOLD) {
        naive_sub_decf_inplace(tensor, dec);
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return;
    }
    
    TensorType dtype = tensor->dtype;
    bool is_aligned = NNL2_IS_ALIGNED(tensor->data, NNL2_TENSOR_ALIGNMENT_32);
    
    // Warning for unaligned memory in safety modes
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!is_aligned) {
            NNL2_WARN("In nnl2_own_2 sub scalar in-place, tensor memory is not aligned to 32 bytes. Performance may be suboptimal");
        }
    #endif
    
    size_t num_threads = NNL2_NUM_THREADS;
    pthread_t threads[num_threads];
    subdecfinplace_ptask tasks[num_threads];
    
    // Calculate optimal chunk sizes with load balancing
    size_t chunk = total_elems / num_threads;
    size_t remainder = total_elems % num_threads;
    
    // Configure decrement value based on data type
    for (size_t i = 0; i < num_threads; i++) {
        tasks[i].dtype = dtype;
        tasks[i].aligned = is_aligned;
        
        switch(dtype) {
            case FLOAT64: tasks[i].decrement.float64_dec = *((double*)dec);  break;
            case FLOAT32: tasks[i].decrement.float32_dec = *((float*)dec);   break;
            case INT32:   tasks[i].decrement.int32_dec = *((int32_t*)dec);   break;
			
            default: {
                NNL2_TYPE_ERROR(dtype);
                return;
			}
        }
    }
    
    size_t current_start = 0;
    for (size_t i = 0; i < num_threads; i++) {
        size_t current_chunk = chunk + (i < remainder ? 1 : 0);
        
        // Configure task for this thread
        tasks[i].tensor_data = tensor->data;
        tasks[i].start = current_start;
        tasks[i].end = current_start + current_chunk;
        
        // Select appropriate worker function based on data type
        void* (*worker_func)(void*) = NULL;
        switch(dtype) {
            case FLOAT64: worker_func = nnl2_own_psub_decf_float64; break;
            case FLOAT32: worker_func = nnl2_own_psub_decf_float32; break;
            case INT32:   worker_func = nnl2_own_psub_decf_int32;   break;
            default: {
                NNL2_TYPE_ERROR(dtype);
                #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                    NNL2_FUNC_EXIT();
                #endif
                return;
            }
        }
        
        // Create thread to process the assigned chunk
        int status = pthread_create(&threads[i], NULL, worker_func, &tasks[i]);
        if(status != 0) {
            NNL2_THREAD_CREATE_ERROR(status, "nnl2_own_2_sub_decf_inplace");
            num_threads = i;
            break;
        }
        
        current_start += current_chunk;
    }
    
    // Wait for all threads to complete
    for (size_t i = 0; i < num_threads; i++) {
        int join_status = pthread_join(threads[i], NULL);
        if(join_status != 0) {
            NNL2_THREAD_JOIN_ERROR(join_status, "nnl2_own_2_sub_decf_inplace");
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
 ** @see nnl2_own_psub_decf_float64
 **/
void* nnl2_own_psub_decf_float64(void* arg) {
    subdecfinplace_ptask* task = (subdecfinplace_ptask*)arg;
    double* data = (double*)task->tensor_data;
    size_t start = task->start;
    size_t end = task->end;
    double decrement = task->decrement.float64_dec;
    
    // Create AVX256 vector with decrement value repeated
    __m256d v_decrement = _mm256_set1_pd(decrement);
    
    size_t i = start;
    
    // AVX256 processing with prefetching
    if(task->aligned) {
        for(; i + 3 < end; i += 4) {
            // Prefetch next cache line
            _mm_prefetch((char*)&data[i + 16], _MM_HINT_T0);
            
            __m256d v_data = _mm256_load_pd(&data[i]);
            __m256d v_result = _mm256_sub_pd(v_data, v_decrement);
            _mm256_store_pd(&data[i], v_result);
        }
    } else {
        for(; i + 3 < end; i += 4) {
            _mm_prefetch((char*)&data[i + 16], _MM_HINT_T0);
            
            __m256d v_data = _mm256_loadu_pd(&data[i]);
            __m256d v_result = _mm256_sub_pd(v_data, v_decrement);
            _mm256_storeu_pd(&data[i], v_result);
        }
    }
    
    // Scalar processing for remainder
    for(; i < end; i++) {
        data[i] -= decrement;
    }
    
    return NULL;
}

/** @brief
 * See documentation at declaration
 * 
 ** @see nnl2_own_psub_decf_float32
 **/
void* nnl2_own_psub_decf_float32(void* arg) {
    subdecfinplace_ptask* task = (subdecfinplace_ptask*)arg;
    float* data = (float*)task->tensor_data;
    size_t start = task->start;
    size_t end = task->end;
    float decrement = task->decrement.float32_dec;
    
    // Create AVX256 vector with decrement value repeated
    __m256 v_decrement = _mm256_set1_ps(decrement);
    
    size_t i = start;
    
    // AVX256 processing with prefetching (8 elements per iteration)
    if(task->aligned) {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&data[i + 32], _MM_HINT_T0);
            
            __m256 v_data = _mm256_load_ps(&data[i]);
            __m256 v_result = _mm256_sub_ps(v_data, v_decrement);
            _mm256_store_ps(&data[i], v_result);
        }
    } else {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&data[i + 32], _MM_HINT_T0);
            
            __m256 v_data = _mm256_loadu_ps(&data[i]);
            __m256 v_result = _mm256_sub_ps(v_data, v_decrement);
            _mm256_storeu_ps(&data[i], v_result);
        }
    }
    
    // Scalar processing for remainder
    for(; i < end; i++) {
        data[i] -= decrement;
    }
    
    return NULL;
}

/** @brief
 * See documentation at declaration
 * 
 ** @see nnl2_own_psub_decf_int32
 **/
void* nnl2_own_psub_decf_int32(void* arg) {
    subdecfinplace_ptask* task = (subdecfinplace_ptask*)arg;
    int32_t* data = (int32_t*)task->tensor_data;
    size_t start = task->start;
    size_t end = task->end;
    int32_t decrement = task->decrement.int32_dec;
    
    // Create AVX256 vector with decrement value repeated
    __m256i v_decrement = _mm256_set1_epi32(decrement);
    
    size_t i = start;
    
    // AVX256 processing with prefetching (8 elements per iteration)
    if(task->aligned) {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&data[i + 32], _MM_HINT_T0);
            
            __m256i v_data = _mm256_load_si256((__m256i*)&data[i]);
            __m256i v_result = _mm256_sub_epi32(v_data, v_decrement);
            _mm256_store_si256((__m256i*)&data[i], v_result);
        }
    } else {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&data[i + 32], _MM_HINT_T0);
            
            __m256i v_data = _mm256_loadu_si256((__m256i*)&data[i]);
            __m256i v_result = _mm256_sub_epi32(v_data, v_decrement);
            _mm256_storeu_si256((__m256i*)&data[i], v_result);
        }
    }
    
    // Scalar processing for remainder
    for(; i < end; i++) {
        data[i] -= decrement;
    }
    
    return NULL;
}

#endif

/**
 * @ingroup backend_system
 * @brief Backend implementations for in-place scalar subtraction operation
 * @details
 * Array follows the common backend registration pattern for in-place scalar subtraction
 * operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for scalar subtraction
 * 
 * @see nnl2_naive
 * @see naive_sub_decf_inplace
 */
Implementation sub_decf_inplace_backends[] = {
    REGISTER_BACKEND(naive_sub_decf_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
	
	#if defined(NNL2_PTHREAD_AVAILABLE) && defined(NNL2_AVX256_AVAILABLE)
        REGISTER_BACKEND(nnl2_own_sub_decf_inplace, nnl2_own, NNL2_OWN_NAME),
    #endif
};    

/**
 * @brief Function pointer for in-place scalar subtraction operation
 * @ingroup backend_system 
 */
subdecfinplacefn sub_decf_inplace;

/** 
 * @brief Sets the backend for in-place scalar subtraction operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for scalar subtraction
 * @see SET_BACKEND_BY_NAME
 */
void set_sub_decf_inplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(sub_decf_inplace_backends, sub_decf_inplace, backend_name);
}

#endif /** NNL2_CORRESPONDENCE_SUB_INPLACE_H **/
