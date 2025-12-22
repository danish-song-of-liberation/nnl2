#ifndef NNL2_MAX_INPLACE_H
#define NNL2_MAX_INPLACE_H

/** @brief 
 * Performs element-wise maximum of two tensors (naive implementation)
 * 
 * Compares elements of the first tensor with corresponding elements 
 * of the second tensor, storing the maximum value in the first tensor
 *
 ** @param tensora 
 * Pointer to the tensor that will be modified 
 *
 ** @param tensorb 
 * Pointer to the tensor whose values will be used for comparison
 *
 ** @note 
 * Supports different data types through automatic conversion
 * The second tensor's elements are converted to the first tensor's data type
 *
 ** @note 
 * In max safety mode, validates tensor pointers and data pointers before access
 *
 ** @warning 
 * This function modifies the first tensor directly
 *
 * @example
 * // Create two tensors with the same shape
 * nnl2_tensor* a = nnl2_tensor((float[]){2.0, 3.0, 4.0}, (int[]){3}, 1, FLOAT32);
 * nnl2_tensor* b = nnl2_tensor((float[]){3.0, 1.0, 5.0}, (int[]){3}, 1, FLOAT32);
 * 
 * // Store element-wise maximum in tensor a
 * naive_maxinplace(a, b);
 * 
 * // Now a contains [3.0, 3.0, 5.0]
 * nnl2_quick_print_tensor(a);
 * 
 * // Cleanup
 * nnl2_free_tensor(a);
 * nnl2_free_tensor(b);
 */
void naive_maxinplace(nnl2_tensor* tensora, const nnl2_tensor* tensorb) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks at the maximum safety level
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensora, "First tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensora->data, "First tensor's data is NULL");
        
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensorb, "Second tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensorb->data, "Second tensor's data is NULL");
    #endif
    
    // Calculating the total number of elements in the first tensor
    size_t total_elems = nnl2_product(tensora->shape, tensora->rank);
    
    // If the tensor is empty, exit the function
    if(total_elems == 0) return;
    
    nnl2_tensor_type typea = tensora->dtype;
    nnl2_tensor_type typeb = tensorb->dtype;
    
    if(typea == typeb) {
        // Handling case when the tensors have the same type
        
        switch(typea) {
            case FLOAT64: {
                volatile double* data_a = (double*)tensora->data;
                volatile double* data_b = (double*)tensorb->data;
                
                // Element-wise maximum calculation
                for(size_t i = 0; i < total_elems; i++) {
                    data_a[i] = MAX(data_a[i], data_b[i]);
                }
                break;
            }
            
            case FLOAT32: {
                volatile float* data_a = (float*)tensora->data;
                volatile float* data_b = (float*)tensorb->data;
                
                // Element-wise maximum calculation
                for(size_t i = 0; i < total_elems; i++) {
                    data_a[i] = MAX(data_a[i], data_b[i]);
                }
                break;
            }
            
            case INT32: {
                volatile int32_t* data_a = (int32_t*)tensora->data;
                volatile int32_t* data_b = (int32_t*)tensorb->data;
                
                // Element-wise maximum calculation
                for(size_t i = 0; i < total_elems; i++) {
                    data_a[i] = MAX(data_a[i], data_b[i]);
                }
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(typea);
                return;
            }
        }
    } else {
        // Handling the case when tensors have different data types
        // Calculating the step size for accessing second tensor elements
        size_t typeb_step = get_dtype_size(typeb);
        
        // Casting second tensor data to char* for byte access
        char* data_b = (char*)tensorb->data;
        
        switch(typea) {
            case FLOAT64: {
                volatile double* data_a = (double*)tensora->data;
                
                // For each element, convert the second tensor element to FLOAT64 and compare
                for(size_t i = 0; i < total_elems; i++) {
                    void* elem_b = data_b + i * typeb_step;
                    data_a[i] = MAX(data_a[i], nnl2_convert_to_float64(elem_b, typeb));
                }
                break;
            }
            
            case FLOAT32: {
                volatile float* data_a = (float*)tensora->data;
                
                // For each element, convert the second tensor element to FLOAT32 and compare
                for(size_t i = 0; i < total_elems; i++) {
                    void* elem_b = data_b + i * typeb_step;
                    data_a[i] = MAX(data_a[i], nnl2_convert_to_float32(elem_b, typeb));
                }
                break;
            }
            
            case INT32: {
                volatile int32_t* data_a = (int32_t*)tensora->data;
                
                // For each element, convert the second tensor element to INT32 and compare
                for(size_t i = 0; i < total_elems; i++) {
                    void* elem_b = data_b + i * typeb_step;
                    data_a[i] = MAX(data_a[i], nnl2_convert_to_int32(elem_b, typeb));
                }
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(typea);
                return;
            }
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

#ifdef NNL2_PTHREAD_AVAILABLE

/** @brief
 * Threshold for enabling parallel execution of in-place maximum operation
 */
#define NNL2_MAX_INPLACE_PARALLEL_THRESHOLD 1000000

/** @brief
 * Worker function for parallel double precision in-place maximum operation
 * 
 ** @param arg 
 * Pointer to maxinplace_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 * 
 ** @details
 * Processes FLOAT64 data using AVX256 instructions with cache prefetching
 * for optimal memory access patterns in in-place operation
 */
void* nnl2_own_pmaxinplace_float64(void* arg);

/** @brief
 * Worker function for parallel single precision in-place maximum operation
 * 
 ** @param arg 
 * Pointer to maxinplace_ptask structure containing thread parameters  
 * 
 ** @return 
 * NULL (for pthread API compatibility)
 * 
 ** @see nnl2_own_pmaxinplace_float64
 **/
void* nnl2_own_pmaxinplace_float32(void* arg);

/** @brief
 * Worker function for parallel integer in-place maximum operation
 * 
 ** @param arg 
 * Pointer to maxinplace_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 * 
 ** @see nnl2_own_pmaxinplace_float64
 **/
void* nnl2_own_pmaxinplace_int32(void* arg);

/** @brief
 * High-performance parallel implementation of in-place element-wise maximum operation
 * 
 ** @param tensora 
 * Pointer to the first input tensor (will be modified in-place)
 *
 ** @param tensorb 
 * Pointer to the second input tensor
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
 * Modifies the first tensor directly
 */
void nnl2_own_maxinplace(nnl2_tensor* tensora, const nnl2_tensor* tensorb) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks at the maximum safety level
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensora, "First tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensora->data, "First tensor data is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensorb, "Second tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensorb->data, "Second tensor data is NULL");
    #endif
    
    size_t total_elems = nnl2_product(tensora->shape, tensora->rank);
    if(total_elems == 0) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return;
    }
    
    nnl2_tensor_type dtype_a = tensora->dtype;
    nnl2_tensor_type dtype_b = tensorb->dtype;
    
    // Fallback to naive implementation for small tensors or mixed types
    if(total_elems < NNL2_MAX_INPLACE_PARALLEL_THRESHOLD || dtype_a != dtype_b) {
        naive_maxinplace(tensora, tensorb);
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return;
    }
    
    bool is_aligned = NNL2_IS_ALIGNED(tensora->data, NNL2_TENSOR_ALIGNMENT_32) &&
                      NNL2_IS_ALIGNED(tensorb->data, NNL2_TENSOR_ALIGNMENT_32);
    
    // Warning for unaligned memory in safety modes
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!is_aligned) {
            NNL2_WARN("In nnl2_own_maxinplace, tensor memory is not aligned to 32 bytes. Performance may be suboptimal");
        }
    #endif
    
    size_t num_threads = NNL2_NUM_THREADS;
    pthread_t threads[num_threads];
    maxinplace_ptask tasks[num_threads];
    
    // Calculate optimal chunk sizes with load balancing
    size_t chunk = total_elems / num_threads;
    size_t remainder = total_elems % num_threads;
    
    // Configure common task parameters
    for (size_t i = 0; i < num_threads; i++) {
        tasks[i].tensora = tensora;
        tasks[i].tensorb = tensorb;
        tasks[i].dtype_a = dtype_a;
        tasks[i].dtype_b = dtype_b;
        tasks[i].aligned = is_aligned;
    }
    
    size_t current_start = 0;
    for (size_t i = 0; i < num_threads; i++) {
        size_t current_chunk = chunk + (i < remainder ? 1 : 0);
        
        // Configure task for this thread
        tasks[i].start = current_start;
        tasks[i].end = current_start + current_chunk;
        
        // Select appropriate worker function based on data type
        void* (*worker_func)(void*) = NULL;
        switch(dtype_a) {
            case FLOAT64: worker_func = nnl2_own_pmaxinplace_float64; break;
            case FLOAT32: worker_func = nnl2_own_pmaxinplace_float32; break;
            case INT32:   worker_func = nnl2_own_pmaxinplace_int32;   break;
            default: {
                NNL2_TYPE_ERROR(dtype_a);
                #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                    NNL2_FUNC_EXIT();
                #endif
                return;
            }
        }
        
        // Create thread to process the assigned chunk
        int status = pthread_create(&threads[i], NULL, worker_func, &tasks[i]);
        if(status != 0) {
            NNL2_THREAD_CREATE_ERROR(status, "nnl2_own_maxinplace");
            num_threads = i;
            break;
        }
        
        current_start += current_chunk;
    }
    
    // Wait for all threads to complete
    for (size_t i = 0; i < num_threads; i++) {
        int join_status = pthread_join(threads[i], NULL);
        if(join_status != 0) {
            NNL2_THREAD_JOIN_ERROR(join_status, "nnl2_own_maxinplace");
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
 ** @see nnl2_own_pmaxinplace_float64
 **/
void* nnl2_own_pmaxinplace_float64(void* arg) {
    maxinplace_ptask* task = (maxinplace_ptask*)arg;
    double* data_a = (double*)task->tensora->data;
    const double* data_b = (const double*)task->tensorb->data;
    size_t start = task->start;
    size_t end = task->end;
    
    size_t i = start;
    
    // AVX256 processing with prefetching
    if(task->aligned) {
        for(; i + 3 < end; i += 4) {
            // Prefetch next cache lines
            _mm_prefetch((char*)&data_a[i + 16], _MM_HINT_T0);
            _mm_prefetch((char*)&data_b[i + 16], _MM_HINT_T0);
            
            __m256d v_data_a = _mm256_load_pd(&data_a[i]);
            __m256d v_data_b = _mm256_load_pd(&data_b[i]);
            __m256d v_result = _mm256_max_pd(v_data_a, v_data_b);
            _mm256_store_pd(&data_a[i], v_result);
        }
    } else {
        for(; i + 3 < end; i += 4) {
            _mm_prefetch((char*)&data_a[i + 16], _MM_HINT_T0);
            _mm_prefetch((char*)&data_b[i + 16], _MM_HINT_T0);
            
            __m256d v_data_a = _mm256_loadu_pd(&data_a[i]);
            __m256d v_data_b = _mm256_loadu_pd(&data_b[i]);
            __m256d v_result = _mm256_max_pd(v_data_a, v_data_b);
            _mm256_storeu_pd(&data_a[i], v_result);
        }
    }
    
    // Scalar processing for remainder
    for(; i < end; i++) {
        data_a[i] = MAX(data_a[i], data_b[i]);
    }
    
    return NULL;
}

/** @brief
 * See documentation at declaration
 * 
 ** @see nnl2_own_pmaxinplace_float32
 **/
void* nnl2_own_pmaxinplace_float32(void* arg) {
    maxinplace_ptask* task = (maxinplace_ptask*)arg;
    float* data_a = (float*)task->tensora->data;
    const float* data_b = (const float*)task->tensorb->data;
    size_t start = task->start;
    size_t end = task->end;
    
    size_t i = start;
    
    // AVX256 processing with prefetching (8 elements per iteration)
    if(task->aligned) {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&data_a[i + 32], _MM_HINT_T0);
            _mm_prefetch((char*)&data_b[i + 32], _MM_HINT_T0);
            
            __m256 v_data_a = _mm256_load_ps(&data_a[i]);
            __m256 v_data_b = _mm256_load_ps(&data_b[i]);
            __m256 v_result = _mm256_max_ps(v_data_a, v_data_b);
            _mm256_store_ps(&data_a[i], v_result);
        }
    } else {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&data_a[i + 32], _MM_HINT_T0);
            _mm_prefetch((char*)&data_b[i + 32], _MM_HINT_T0);
            
            __m256 v_data_a = _mm256_loadu_ps(&data_a[i]);
            __m256 v_data_b = _mm256_loadu_ps(&data_b[i]);
            __m256 v_result = _mm256_max_ps(v_data_a, v_data_b);
            _mm256_storeu_ps(&data_a[i], v_result);
        }
    }
    
    // Scalar processing for remainder
    for(; i < end; i++) {
        data_a[i] = MAX(data_a[i], data_b[i]);
    }
    
    return NULL;
}

/** @brief
 * See documentation at declaration
 * 
 ** @see nnl2_own_pmaxinplace_int32
 **/
void* nnl2_own_pmaxinplace_int32(void* arg) {
    maxinplace_ptask* task = (maxinplace_ptask*)arg;
    int32_t* data_a = (int32_t*)task->tensora->data;
    const int32_t* data_b = (const int32_t*)task->tensorb->data;
    size_t start = task->start;
    size_t end = task->end;
    
    size_t i = start;
    
    // AVX256 processing with prefetching (8 elements per iteration)
    if(task->aligned) {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&data_a[i + 32], _MM_HINT_T0);
            _mm_prefetch((char*)&data_b[i + 32], _MM_HINT_T0);
            
            __m256i v_data_a = _mm256_load_si256((__m256i*)&data_a[i]);
            __m256i v_data_b = _mm256_load_si256((__m256i*)&data_b[i]);
            
            // For integers, we need to compare and select maximum
            __m256i v_compare = _mm256_cmpgt_epi32(v_data_a, v_data_b);
            __m256i v_result = _mm256_blendv_epi8(v_data_b, v_data_a, v_compare);
            
            _mm256_store_si256((__m256i*)&data_a[i], v_result);
        }
    } else {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&data_a[i + 32], _MM_HINT_T0);
            _mm_prefetch((char*)&data_b[i + 32], _MM_HINT_T0);
            
            __m256i v_data_a = _mm256_loadu_si256((__m256i*)&data_a[i]);
            __m256i v_data_b = _mm256_loadu_si256((__m256i*)&data_b[i]);
            
            __m256i v_compare = _mm256_cmpgt_epi32(v_data_a, v_data_b);
            __m256i v_result = _mm256_blendv_epi8(v_data_b, v_data_a, v_compare);
            
            _mm256_storeu_si256((__m256i*)&data_a[i], v_result);
        }
    }
    
    // Scalar processing for remainder
    for(; i < end; i++) {
        data_a[i] = MAX(data_a[i], data_b[i]);
    }
    
    return NULL;
}

#endif

/**
 * @ingroup backend_system
 * @brief Backend implementations for maxinplace operation
 * @details
 * Array follows the common backend registration pattern for element-wise maximum operations.
 * Currently registered backends:
 *  - nnl2_naive: Basic reference implementation
 * 
 * @see nnl2_naive
 */
nnl2_runtime_implementation maxinplace_backends[] = {
	REGISTER_BACKEND(naive_maxinplace, nnl2_naive, NAIVE_BACKEND_NAME),
	
	#if defined(NNL2_AVX256_AVAILABLE) && defined(NNL2_PTHREAD_AVAILABLE)
		REGISTER_BACKEND(nnl2_own_maxinplace, nnl2_own, NNL2_OWN_NAME),
	#endif
};

/**
 * @brief Function pointer for maxinplace operation
 * @ingroup backend_system 
 */
maxinplacefn maxinplace;

/** 
 * @brief Makes the maxinplace backend current
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(maxinplace);

/** 
 * @brief Sets the backend for maxinplace operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see ESET_BACKEND_BY_NAME
 */
void set_maxinplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(maxinplace_backends, maxinplace, backend_name, CURRENT_BACKEND(maxinplace));
}

/** 
 * @brief Gets the name of the active backend for maxinplace operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_maxinplace_backend() {
	return CURRENT_BACKEND(maxinplace);
}

/** 
 * @brief Function declaration for getting all available maxinplace backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(maxinplace);

/**
 * @brief Function declaration for getting the number of available maxinplace backends
 * @ingroup backend_system
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(maxinplace);

#endif /** NNL2_MAX_INPLACE_H **/
