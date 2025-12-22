#ifndef NNL2_CORRESPONDENCE_MUL_INPLACE_H
#define NNL2_CORRESPONDENCE_MUL_INPLACE_H

/** @brief 
 * Multiplies each element of a tensor by a scalar value (in-place).
 * 
 ** @param tensor 
 * Pointer to the tensor whose elements will be multiplied
 * 
 ** @param multiplier 
 * Pointer to the scalar value to multiply by
 */
void naive_mul_mulf_inplace(nnl2_tensor* tensor, void* multiplier) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    size_t total_elems = nnl2_product(tensor->shape, tensor->rank);
    if(total_elems == 0) return;
    
    switch(tensor->dtype) {
        case FLOAT64: {
            double* cast_data = (double*)tensor->data;
            double multiply = *((double*)multiplier);
            for(size_t i = 0; i < total_elems; i++) cast_data[i] *= multiply;
            break;
        }
        
        case FLOAT32: {
            float* cast_data = (float*)tensor->data;
            float multiply = *((float*)multiplier);
            for(size_t i = 0; i < total_elems; i++) cast_data[i] *= multiply;
            break;
        }
        
        case INT32: {
            int32_t* cast_data = (int32_t*)tensor->data;
            int32_t multiply = *((int32_t*)multiplier);
            for(size_t i = 0; i < total_elems; i++) cast_data[i] *= multiply;
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
 * scalar multiplication in-place operation
 */
#define NNL2_MUL_MULF_INPLACE_PARALLEL_THRESHOLD 1000000

/** @brief 
 * Worker function for parallel double precision scalar multiplication
 * 
 ** @param arg 
 * Pointer to mulmulfinplace_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 * 
 ** @details
 * Processes FLOAT64 data using AVX256 instructions with cache prefetching
 * for optimal memory access patterns
 */
void* nnl2_own_pmul_mulf_float64(void* arg);

/** @brief
 * Worker function for parallel single precision scalar multiplication
 * 
 ** @param arg 
 * Pointer to mulmulfinplace_ptask structure containing thread parameters  
 * 
 ** @return 
 * NULL (for pthread API compatibility)
 * 
 ** @see nnl2_own_pmul_mulf_float64
 **/
void* nnl2_own_pmul_mulf_float32(void* arg);

/** @brief
 * Worker function for parallel integer scalar multiplication
 * 
 ** @param arg 
 * Pointer to mulmulfinplace_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 * 
 ** @see nnl2_own_pmul_mulf_float64
 **/
void* nnl2_own_pmul_mulf_int32(void* arg);

/** @brief
 * High-performance parallel implementation of in-place scalar multiplication
 * 
 ** @param tensor 
 * Pointer to tensor that will be modified in-place
 *
 ** @param multiplier 
 * Pointer to scalar value to multiply each element by
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
void nnl2_own_2_mul_mulf_inplace(nnl2_tensor* tensor, void* multiplier) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks at the maximum safety level
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensor, "nnl2_tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensor->data, "nnl2_tensor data is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(multiplier, "Multiplier pointer is NULL");
    #endif
    
    size_t total_elems = nnl2_product(tensor->shape, tensor->rank);
    if(total_elems == 0) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return;
    }
    
    // Fallback to naive implementation for small tensors
    if(total_elems < NNL2_MUL_MULF_INPLACE_PARALLEL_THRESHOLD) {
        naive_mul_mulf_inplace(tensor, multiplier);
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return;
    }
    
    nnl2_tensor_type dtype = tensor->dtype;
    bool is_aligned = NNL2_IS_ALIGNED(tensor->data, NNL2_TENSOR_ALIGNMENT_32);
    
    // Warning for unaligned memory in safety modes
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!is_aligned) {
            NNL2_WARN("In nnl2_own_2 mul scalar in-place, tensor memory is not aligned to 32 bytes. Performance may be suboptimal");
        }
    #endif
    
    size_t num_threads = NNL2_NUM_THREADS;
    pthread_t threads[num_threads];
    mulmulfinplace_ptask tasks[num_threads];
    
    // Calculate optimal chunk sizes with load balancing
    size_t chunk = total_elems / num_threads;
    size_t remainder = total_elems % num_threads;
    
    // Configure multiplier value based on data type
    for (size_t i = 0; i < num_threads; i++) {
        tasks[i].dtype = dtype;
        tasks[i].aligned = is_aligned;
        
        switch(dtype) {
            case FLOAT64: tasks[i].multiplier.float64_mult = *((double*)multiplier); break;
            case FLOAT32: tasks[i].multiplier.float32_mult = *((float*)multiplier);  break;
            case INT32:   tasks[i].multiplier.int32_mult = *((int32_t*)multiplier);  break;
			
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
            case FLOAT64: worker_func = nnl2_own_pmul_mulf_float64; break;
            case FLOAT32: worker_func = nnl2_own_pmul_mulf_float32; break;
            case INT32:   worker_func = nnl2_own_pmul_mulf_int32;   break;
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
            NNL2_THREAD_CREATE_ERROR(status, "nnl2_own_mul_mulf_inplace");
            num_threads = i;
            break;
        }
        
        current_start += current_chunk;
    }
    
    // Wait for all threads to complete
    for (size_t i = 0; i < num_threads; i++) {
        int join_status = pthread_join(threads[i], NULL);
        if(join_status != 0) {
            NNL2_THREAD_JOIN_ERROR(join_status, "nnl2_own_mul_mulf_inplace");
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
 ** @see nnl2_own_pmul_mulf_float64
 **/
void* nnl2_own_pmul_mulf_float64(void* arg) {
    mulmulfinplace_ptask* task = (mulmulfinplace_ptask*)arg;
    double* data = (double*)task->tensor_data;
    size_t start = task->start;
    size_t end = task->end;
    double multiplier = task->multiplier.float64_mult;
    
    // Create AVX256 vector with multiplier value repeated
    __m256d v_multiplier = _mm256_set1_pd(multiplier);
    
    size_t i = start;
    
    // AVX256 processing with prefetching
    if(task->aligned) {
        for(; i + 3 < end; i += 4) {
            // Prefetch next cache line
            _mm_prefetch((char*)&data[i + 16], _MM_HINT_T0);
            
            __m256d v_data = _mm256_load_pd(&data[i]);
            __m256d v_result = _mm256_mul_pd(v_data, v_multiplier);
            _mm256_store_pd(&data[i], v_result);
        }
    } else {
        for(; i + 3 < end; i += 4) {
            _mm_prefetch((char*)&data[i + 16], _MM_HINT_T0);
            
            __m256d v_data = _mm256_loadu_pd(&data[i]);
            __m256d v_result = _mm256_mul_pd(v_data, v_multiplier);
            _mm256_storeu_pd(&data[i], v_result);
        }
    }
    
    // Scalar processing for remainder
    for(; i < end; i++) {
        data[i] *= multiplier;
    }
    
    return NULL;
}

/** @brief
 * See documentation at declaration
 * 
 ** @see nnl2_own_pmul_mulf_float32
 **/
void* nnl2_own_pmul_mulf_float32(void* arg) {
    mulmulfinplace_ptask* task = (mulmulfinplace_ptask*)arg;
    float* data = (float*)task->tensor_data;
    size_t start = task->start;
    size_t end = task->end;
    float multiplier = task->multiplier.float32_mult;
    
    // Create AVX256 vector with multiplier value repeated
    __m256 v_multiplier = _mm256_set1_ps(multiplier);
    
    size_t i = start;
    
    // AVX256 processing with prefetching (8 elements per iteration)
    if(task->aligned) {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&data[i + 32], _MM_HINT_T0);
            
            __m256 v_data = _mm256_load_ps(&data[i]);
            __m256 v_result = _mm256_mul_ps(v_data, v_multiplier);
            _mm256_store_ps(&data[i], v_result);
        }
    } else {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&data[i + 32], _MM_HINT_T0);
            
            __m256 v_data = _mm256_loadu_ps(&data[i]);
            __m256 v_result = _mm256_mul_ps(v_data, v_multiplier);
            _mm256_storeu_ps(&data[i], v_result);
        }
    }
    
    // Scalar processing for remainder
    for(; i < end; i++) {
        data[i] *= multiplier;
    }
    
    return NULL;
}

/** @brief
 * See documentation at declaration
 * 
 ** @see nnl2_own_pmul_mulf_int32
 **/
void* nnl2_own_pmul_mulf_int32(void* arg) {
    mulmulfinplace_ptask* task = (mulmulfinplace_ptask*)arg;
    int32_t* data = (int32_t*)task->tensor_data;
    size_t start = task->start;
    size_t end = task->end;
    int32_t multiplier = task->multiplier.int32_mult;
    
    // Create AVX256 vector with multiplier value repeated
    __m256i v_multiplier = _mm256_set1_epi32(multiplier);
    
    size_t i = start;
    
    // AVX256 processing with prefetching (8 elements per iteration)
    if(task->aligned) {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&data[i + 32], _MM_HINT_T0);
            
            __m256i v_data = _mm256_load_si256((__m256i*)&data[i]);
            __m256i v_result = _mm256_mullo_epi32(v_data, v_multiplier);
            _mm256_store_si256((__m256i*)&data[i], v_result);
        }
    } else {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&data[i + 32], _MM_HINT_T0);
            
            __m256i v_data = _mm256_loadu_si256((__m256i*)&data[i]);
            __m256i v_result = _mm256_mullo_epi32(v_data, v_multiplier);
            _mm256_storeu_si256((__m256i*)&data[i], v_result);
        }
    }
    
    // Scalar processing for remainder
    for(; i < end; i++) {
        data[i] *= multiplier;
    }
    
    return NULL;
}

#endif

/**
 * @ingroup backend_system
 * @brief Backend implementations for in-place scalar multiplication operation
 * @details
 * Array follows the common backend registration pattern for in-place scalar multiplication
 * operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for scalar multiplication
 *  - nnl2_own:   High-performance implementation with AVX256, pthread and prefetching
 * 
 * @see nnl2_naive
 * @see naive_mul_mulf_inplace
 * @see nnl2_own_2_mul_mulf_inplace
 */
nnl2_runtime_implementation mul_mulf_inplace_backends[] = {
    REGISTER_BACKEND(naive_mul_mulf_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
	
	#if defined(NNL2_PTHREAD_AVAILABLE) && defined(NNL2_AVX256_AVAILABLE) && TENSOR_MEM_ALIGNMENT == 32
        REGISTER_BACKEND(nnl2_own_2_mul_mulf_inplace, nnl2_own_2, "nnl2_own_2"),
    #endif
};    

/**
 * @brief Function pointer for in-place scalar multiplication operation
 * @ingroup backend_system 
 */
mulmulfinplacefn mul_mulf_inplace;

/** 
 * @brief Sets the backend for in-place scalar multiplication operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for scalar multiplication
 * @see SET_BACKEND_BY_NAME
 */
void set_mul_mulf_inplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(mul_mulf_inplace_backends, mul_mulf_inplace, backend_name);
}

#endif /** NNL2_CORRESPONDENCE_MUL_INPLACE_H **/
