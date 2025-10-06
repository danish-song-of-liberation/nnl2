#ifndef NNL2_MAX_H
#define NNL2_MAX_H

/** @brief
 * Performs element-wise maximum of two tensors (naive implementation)
 *
 ** @details
 * The function creates a new tensor containing the result of taking the maximum
 * of each element from the first tensor and the corresponding element in the second tensor.
 * It supports various data types with automatic casting to a higher type in the hierarchy.
 *
 ** @param tensora
 * Pointer to the first tensor
 *
 ** @param tensorb
 * Pointer to the second tensor
 *
 ** @return 
 * Pointer to a new tensor with the maximum values
 *
 ** @note
 * Uses volatile pointers to prevent compiler optimizations
 *
 ** @note
 * Returns NULL in case of failure or unsupported data type
 *
 ** @note
 * The result tensor has the same shape as input tensors and the highest data type
 *
 ** @see nnl2_empty
 ** @see get_dtype_size()
 ** @see nnl2_convert_to_float64()
 ** @see nnl2_convert_to_float32()
 ** @see nnl2_convert_to_int32()
 **/
Tensor* naive_max(const Tensor* tensora, const Tensor* tensorb) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	// Additional checks on maximum safety level
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensora, "First tensor is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensora->data, "First tensor data is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensorb, "Second tensor is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensorb->data, "Second tensor data is NULL", NULL);
	#endif
	
    // Calculate the total number of elements in the tensors
    size_t len = product(tensora->shape, tensora->rank);
    
    TensorType dtype_a = tensora->dtype;
    TensorType dtype_b = tensorb->dtype;
    
    // Selecting the winning type (higher in the hierarchy)
    TensorType winner_in_the_type_hierarchy = MAX(dtype_a, dtype_b);

    // Create an output tensor with the same shape and winning data type
    Tensor* result = nnl2_empty(tensora->shape, tensora->rank, winner_in_the_type_hierarchy);
    
    if (result == NULL) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return NULL;
    }
    
    if (len == 0) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return result;
    }
    
    if (dtype_a == dtype_b) {
        // Handling the case if the data types match
        
        switch (dtype_a) {
            case FLOAT64: {
                volatile double* data_a = (double*)tensora->data;
                volatile double* data_b = (double*)tensorb->data;
                volatile double* data_result = (double*)result->data;
            
                // Element-wise maximum calculation
                for (size_t i = 0; i < len; i++) {
                    data_result[i] = MAX(data_a[i], data_b[i]);
                }
                
                break;
            }
            
            case FLOAT32: {
                volatile float* data_a = (float*)tensora->data;
                volatile float* data_b = (float*)tensorb->data;
                volatile float* data_result = (float*)result->data;
        
                // Element-wise maximum calculation
                for (size_t i = 0; i < len; i++) {
                    data_result[i] = MAX(data_a[i], data_b[i]);
                }
                
                break;
            }
            
            case INT32: {
                volatile int32_t* data_a = (int32_t*)tensora->data;
                volatile int32_t* data_b = (int32_t*)tensorb->data;
                volatile int32_t* data_result = (int32_t*)result->data;
        
                // Element-wise maximum calculation
                for (size_t i = 0; i < len; i++) {
                    data_result[i] = MAX(data_a[i], data_b[i]);
                }
                
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(dtype_a);
                return NULL;
            }
        }
    } else {
        // Handling the case if the data types are NOT match
        switch (winner_in_the_type_hierarchy) {
            case FLOAT64: {
                volatile double* data_result = (double*)result->data;
                
                for (size_t i = 0; i < len; i++) {
                    // Calculate the pointers to the current elements, taking into account the size of the type
                    void* elem_a = (char*)tensora->data + i * get_dtype_size(dtype_a);
                    void* elem_b = (char*)tensorb->data + i * get_dtype_size(dtype_b);
                    
                    data_result[i] = MAX(nnl2_convert_to_float64(elem_a, dtype_a), nnl2_convert_to_float64(elem_b, dtype_b));
                }
                
                break;
            }
            
            case FLOAT32: {
                volatile float* data_result = (float*)result->data;
                
                for (size_t i = 0; i < len; i++) {
                    // Calculate the pointers to the current elements, taking into account the size of the type
                    void* elem_a = (char*)tensora->data + i * get_dtype_size(dtype_a);
                    void* elem_b = (char*)tensorb->data + i * get_dtype_size(dtype_b);
                    
                    data_result[i] = MAX(nnl2_convert_to_float32(elem_a, dtype_a), nnl2_convert_to_float32(elem_b, dtype_b));
                }
                
                break;
            }
        
            case INT32: {
                volatile int32_t* data_result = (int32_t*)result->data;
                
                for (size_t i = 0; i < len; i++) {
                    // Calculate the pointers to the current elements, taking into account the size of the type
                    void* elem_a = (char*)tensora->data + i * get_dtype_size(dtype_a);
                    void* elem_b = (char*)tensorb->data + i * get_dtype_size(dtype_b);
                    
                    data_result[i] = MAX(nnl2_convert_to_int32(elem_a, dtype_a), nnl2_convert_to_int32(elem_b, dtype_b));
                }
                
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(winner_in_the_type_hierarchy);
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
 * Threshold for enabling parallel execution of maximum operation
 */
#define NNL2_MAX_PARALLEL_THRESHOLD 1000000

/** @brief
 * Worker function for parallel double precision maximum operation
 * 
 ** @param arg 
 * Pointer to max_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 * 
 ** @details
 * Processes FLOAT64 data using AVX256 instructions with cache prefetching
 * for optimal memory access patterns
 */
void* nnl2_own_pmax_float64(void* arg);

/** @brief
 * Worker function for parallel single precision maximum operation
 * 
 ** @param arg 
 * Pointer to max_ptask structure containing thread parameters  
 * 
 ** @return 
 * NULL (for pthread API compatibility)
 * 
 ** @see nnl2_own_pmax_float64
 **/
void* nnl2_own_pmax_float32(void* arg);

/** @brief
 * Worker function for parallel integer maximum operation
 * 
 ** @param arg 
 * Pointer to max_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 * 
 ** @see nnl2_own_pmax_float64
 **/
void* nnl2_own_pmax_int32(void* arg);

/** @brief
 * High-performance parallel implementation of element-wise maximum operation
 * 
 ** @param tensora 
 * Pointer to the first input tensor
 *
 ** @param tensorb 
 * Pointer to the second input tensor
 * 
 ** @return
 * Pointer to new tensor containing element-wise maximum values
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
Tensor* nnl2_own_max(const Tensor* tensora, const Tensor* tensorb) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks at the maximum safety level
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensora, "First tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensora->data, "First tensor data is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensorb, "Second tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensorb->data, "Second tensor data is NULL", NULL);
    #endif
    
    size_t total_elems = product(tensora->shape, tensora->rank);
    
    TensorType dtype_a = tensora->dtype;
    TensorType dtype_b = tensorb->dtype;
    TensorType result_dtype = MAX(dtype_a, dtype_b);
    
    // Create output tensor
    Tensor* result = nnl2_empty(tensora->shape, tensora->rank, result_dtype);
    if (result == NULL) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return NULL;
    }
    
    if (total_elems == 0) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return result;
    }
    
    // Fallback to naive implementation for small tensors or mixed types
    if (total_elems < NNL2_MAX_PARALLEL_THRESHOLD || dtype_a != dtype_b) {
        Tensor* naive_result = naive_max(tensora, tensorb);
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
    
    bool is_aligned = NNL2_IS_ALIGNED(tensora->data, NNL2_TENSOR_ALIGNMENT_32) &&
                      NNL2_IS_ALIGNED(tensorb->data, NNL2_TENSOR_ALIGNMENT_32) &&
                      NNL2_IS_ALIGNED(result->data, NNL2_TENSOR_ALIGNMENT_32);
    
    // Warning for unaligned memory in safety modes
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if (!is_aligned) {
            NNL2_WARN("In nnl2_own_max, tensor memory is not aligned to 32 bytes. Performance may be suboptimal");
        }
    #endif
    
    size_t num_threads = NNL2_NUM_THREADS;
    pthread_t threads[num_threads];
    max_ptask tasks[num_threads];
    
    // Calculate optimal chunk sizes with load balancing
    size_t chunk = total_elems / num_threads;
    size_t remainder = total_elems % num_threads;
    
    // Configure common task parameters
    for (size_t i = 0; i < num_threads; i++) {
        tasks[i].tensora = tensora;
        tasks[i].tensorb = tensorb;
        tasks[i].result = result;
        tasks[i].dtype_a = dtype_a;
        tasks[i].dtype_b = dtype_b;
        tasks[i].result_dtype = result_dtype;
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
        switch (dtype_a) {
            case FLOAT64: worker_func = nnl2_own_pmax_float64; break;
            case FLOAT32: worker_func = nnl2_own_pmax_float32; break;
            case INT32:   worker_func = nnl2_own_pmax_int32;   break;
            default: {
                NNL2_TYPE_ERROR(dtype_a);
                nnl2_free_tensor(result);
                #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                    NNL2_FUNC_EXIT();
                #endif
                return NULL;
            }
        }
        
        // Create thread to process the assigned chunk
        int status = pthread_create(&threads[i], NULL, worker_func, &tasks[i]);
        if (status != 0) {
            NNL2_THREAD_CREATE_ERROR(status, "nnl2_own_max");
            num_threads = i;
            break;
        }
        
        current_start += current_chunk;
    }
    
    // Wait for all threads to complete
    for (size_t i = 0; i < num_threads; i++) {
        int join_status = pthread_join(threads[i], NULL);
        if (join_status != 0) {
            NNL2_THREAD_JOIN_ERROR(join_status, "nnl2_own_max");
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
 ** @see nnl2_own_pmax_float64
 **/
void* nnl2_own_pmax_float64(void* arg) {
    max_ptask* task = (max_ptask*)arg;
    double* data_a = (double*)task->tensora->data;
    double* data_b = (double*)task->tensorb->data;
    double* data_result = (double*)task->result->data;
    size_t start = task->start;
    size_t end = task->end;
    
    size_t i = start;
    
    // AVX256 processing with prefetching
    if (task->aligned) {
        for (; i + 3 < end; i += 4) {
            // Prefetch next cache lines
            _mm_prefetch((char*)&data_a[i + 16], _MM_HINT_T0);
            _mm_prefetch((char*)&data_b[i + 16], _MM_HINT_T0);
            
            __m256d v_data_a = _mm256_load_pd(&data_a[i]);
            __m256d v_data_b = _mm256_load_pd(&data_b[i]);
            __m256d v_result = _mm256_max_pd(v_data_a, v_data_b);
            _mm256_store_pd(&data_result[i], v_result);
        }
    } else {
        for (; i + 3 < end; i += 4) {
            _mm_prefetch((char*)&data_a[i + 16], _MM_HINT_T0);
            _mm_prefetch((char*)&data_b[i + 16], _MM_HINT_T0);
            
            __m256d v_data_a = _mm256_loadu_pd(&data_a[i]);
            __m256d v_data_b = _mm256_loadu_pd(&data_b[i]);
            __m256d v_result = _mm256_max_pd(v_data_a, v_data_b);
            _mm256_storeu_pd(&data_result[i], v_result);
        }
    }
    
    // Scalar processing for remainder
    for (; i < end; i++) {
        data_result[i] = MAX(data_a[i], data_b[i]);
    }
    
    return NULL;
}

/** @brief
 * See documentation at declaration
 * 
 ** @see nnl2_own_pmax_float32
 **/
void* nnl2_own_pmax_float32(void* arg) {
    max_ptask* task = (max_ptask*)arg;
    float* data_a = (float*)task->tensora->data;
    float* data_b = (float*)task->tensorb->data;
    float* data_result = (float*)task->result->data;
    size_t start = task->start;
    size_t end = task->end;
    
    size_t i = start;
    
    // AVX256 processing with prefetching (8 elements per iteration)
    if (task->aligned) {
        for (; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&data_a[i + 32], _MM_HINT_T0);
            _mm_prefetch((char*)&data_b[i + 32], _MM_HINT_T0);
            
            __m256 v_data_a = _mm256_load_ps(&data_a[i]);
            __m256 v_data_b = _mm256_load_ps(&data_b[i]);
            __m256 v_result = _mm256_max_ps(v_data_a, v_data_b);
            _mm256_store_ps(&data_result[i], v_result);
        }
    } else {
        for (; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&data_a[i + 32], _MM_HINT_T0);
            _mm_prefetch((char*)&data_b[i + 32], _MM_HINT_T0);
            
            __m256 v_data_a = _mm256_loadu_ps(&data_a[i]);
            __m256 v_data_b = _mm256_loadu_ps(&data_b[i]);
            __m256 v_result = _mm256_max_ps(v_data_a, v_data_b);
            _mm256_storeu_ps(&data_result[i], v_result);
        }
    }
    
    // Scalar processing for remainder
    for (; i < end; i++) {
        data_result[i] = MAX(data_a[i], data_b[i]);
    }
    
    return NULL;
}

/** @brief
 * See documentation at declaration
 * 
 ** @see nnl2_own_pmax_int32
 **/
void* nnl2_own_pmax_int32(void* arg) {
    max_ptask* task = (max_ptask*)arg;
    int32_t* data_a = (int32_t*)task->tensora->data;
    int32_t* data_b = (int32_t*)task->tensorb->data;
    int32_t* data_result = (int32_t*)task->result->data;
    size_t start = task->start;
    size_t end = task->end;
    
    size_t i = start;
    
    // AVX256 processing with prefetching (8 elements per iteration)
    if (task->aligned) {
        for (; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&data_a[i + 32], _MM_HINT_T0);
            _mm_prefetch((char*)&data_b[i + 32], _MM_HINT_T0);
            
            __m256i v_data_a = _mm256_load_si256((__m256i*)&data_a[i]);
            __m256i v_data_b = _mm256_load_si256((__m256i*)&data_b[i]);
            
            // For integers, we need to compare and select maximum
            __m256i v_compare = _mm256_cmpgt_epi32(v_data_a, v_data_b);
            __m256i v_result = _mm256_blendv_epi8(v_data_b, v_data_a, v_compare);
            
            _mm256_store_si256((__m256i*)&data_result[i], v_result);
        }
    } else {
        for (; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&data_a[i + 32], _MM_HINT_T0);
            _mm_prefetch((char*)&data_b[i + 32], _MM_HINT_T0);
            
            __m256i v_data_a = _mm256_loadu_si256((__m256i*)&data_a[i]);
            __m256i v_data_b = _mm256_loadu_si256((__m256i*)&data_b[i]);
            
            __m256i v_compare = _mm256_cmpgt_epi32(v_data_a, v_data_b);
            __m256i v_result = _mm256_blendv_epi8(v_data_b, v_data_a, v_compare);
            
            _mm256_storeu_si256((__m256i*)&data_result[i], v_result);
        }
    }
    
    // Scalar processing for remainder
    for (; i < end; i++) {
        data_result[i] = MAX(data_a[i], data_b[i]);
    }
    
    return NULL;
}

#endif

/**
 * @ingroup backend_system
 * @brief Backend implementations for max operation
 * @details
 * Array follows the common backend registration pattern for element-wise maximum operations.
 * Currently registered backends:
 *  - nnl2_naive: Basic reference implementation
 * 
 * @see nnl2_naive
 */
Implementation max_backends[] = {
	REGISTER_BACKEND(naive_max, nnl2_naive, NAIVE_BACKEND_NAME),
	
	#if defined(NNL2_AVX256_AVAILABLE) && defined(NNL2_PTHREAD_AVAILABLE)
		REGISTER_BACKEND(nnl2_own_max, nnl2_own, NNL2_OWN_NAME),
	#endif
};

/**
 * @brief Function pointer for max operation
 * @ingroup backend_system 
 */
maxfn nnl2_max;

/** 
 * @brief Makes the max backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(max);

/** 
 * @brief Sets the backend for max operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see ESET_BACKEND_BY_NAME
 */
void set_max_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(max_backends, nnl2_max, backend_name, current_backend(max));
}

/** 
 * @brief Gets the name of the active backend for max operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_max_backend() {
	return current_backend(max);
}

/** 
 * @brief Function declaration for getting all available max backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(max);

/**
 * @brief Function declaration for getting the number of available max backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(max);

#endif /** NNL2_MAX_H **/
