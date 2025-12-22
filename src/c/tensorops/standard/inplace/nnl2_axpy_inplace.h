#ifndef NNL2_AXPY_INPLACE_H
#define NNL2_AXPY_INPLACE_H

/** @brief 
 * Performs element-wise AXPY operation (naive implementation)
 * 
 * Computes: summand = summand + alpha * sumend
 * Performs the scaled vector addition operation on two tensors,
 * modifying the summand tensor in place
 *
 ** @param summand 
 * Pointer to the tensor that will be modified (receives the AXPY result)
 *
 ** @param sumend 
 * Pointer to the tensor whose values will be scaled and added to the summand
 *
 ** @param alpha
 * Scalar multiplier for the sumend tensor values
 *
 ** @note 
 * Supports different data types through automatic conversion
 * The sumend elements are converted to the summand's data type
 *
 ** @note 
 * In max safety mode, validates tensor pointers and data pointers before access
 *
 ** @warning 
 * This function modifies the summand tensor directly
 * Both tensors must have the same shape
 *
 * @example
 * // Create two tensors with the same shape
 * nnl2_tensor* a = nnl2_ones((int[]){2, 3}, 2, FLOAT32);
 * nnl2_tensor* b = nnl2_ones((int[]){2, 3}, 2, FLOAT32);
 * 
 * // Compute a = a + 2.5 * b
 * naive_axpy_inplace(a, b, 2.5f);
 * 
 * // Now a contains 3.5 in all elements
 * nnl2_quick_print_tensor(a);
 * 
 * // Cleanup
 * nnl2_free_tensor(a);
 * nnl2_free_tensor(b);
 */
void naive_axpy_inplace(nnl2_tensor* summand, nnl2_tensor* sumend, float alpha) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks at the maximum safety level
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(summand, "Summand tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(summand->data, "Summand tensor's data is NULL");
        
        NNL2_CHECK_NULL_IF_ERR_RETURN(sumend, "Sumend tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(sumend->data, "Sumend tensor's data is NULL");
    #endif
    
    // Calculating the total number of elements in the summand tensor
    size_t total_elems = nnl2_product(summand->shape, summand->rank);
    
    // If the tensor is empty, exit the function
    if(total_elems == 0) return;
    
    nnl2_tensor_type dtype_summand = summand->dtype;
    nnl2_tensor_type dtype_sumend = sumend->dtype;
    
    if(dtype_summand == dtype_sumend) {
        // Handling case when the tensors have the same type
        
        switch(dtype_summand) {
            case FLOAT64: {
                volatile double* data_summand = (double*)summand->data;
                volatile double* data_sumend = (double*)sumend->data;
                double alpha_double = (double)alpha;
                
                // Element-wise AXPY operation
                for(size_t i = 0; i < total_elems; i++) {
                    data_summand[i] += data_sumend[i] * alpha_double;
                }
				
                break;
            }
            
            case FLOAT32: {
                volatile float* data_summand = (float*)summand->data;
                volatile float* data_sumend = (float*)sumend->data;
                
                // Element-wise AXPY operation
                for(size_t i = 0; i < total_elems; i++) {
                    data_summand[i] += data_sumend[i] * alpha;
                }    
				
                break;
            }
            
            case INT32: {
                volatile int32_t* data_summand = (int32_t*)summand->data;
                volatile int32_t* data_sumend = (int32_t*)sumend->data;
                int32_t alpha_int = (int32_t)alpha;
                
                // Element-wise AXPY operation
                for(size_t i = 0; i < total_elems; i++) {
                    data_summand[i] += data_sumend[i] * alpha_int;
                }        
				
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(dtype_summand);
                return;
            }
        }
    } else {
        // Handling the case when tensors have different data types
        // Calculating the step size for accessing sumend tensor elements
        size_t sumend_step = get_dtype_size(dtype_sumend);
        
        // Casting sumend data to char* for byte access
        char* sumend_data = (char*)sumend->data;
        
        switch(dtype_summand) {
            case FLOAT64: {
                volatile double* data_summand = (double*)summand->data;
                double alpha_double = (double)alpha;
                
                // For each element, convert the sumend element to FLOAT64 and perform AXPY
                for(size_t i = 0; i < total_elems; i++) {
                    void* sumend_elem = sumend_data + i * sumend_step;
                    data_summand[i] += nnl2_convert_to_float64(sumend_elem, dtype_sumend) * alpha_double;
                }
                
                break; 
            }
            
            case FLOAT32: {
                volatile float* data_summand = (float*)summand->data;
                
                // For each element, convert the sumend element to FLOAT32 and perform AXPY
                for(size_t i = 0; i < total_elems; i++) {
                    void* sumend_elem = sumend_data + i * sumend_step;
                    data_summand[i] += nnl2_convert_to_float32(sumend_elem, dtype_sumend) * alpha;
                }
                
                break; 
            }
            
            case INT32: {
                volatile int32_t* data_summand = (int32_t*)summand->data;
                int32_t alpha_int = (int32_t)alpha;
                
                // For each element, convert the sumend element to INT32 and perform AXPY
                for(size_t i = 0; i < total_elems; i++) {
                    void* sumend_elem = sumend_data + i * sumend_step;
                    data_summand[i] += nnl2_convert_to_int32(sumend_elem, dtype_sumend) * alpha_int;
                }
                
                break; 
            }
            
            default: {
                NNL2_TYPE_ERROR(dtype_summand);
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
 * Threshold for enabling parallel execution of AXPY in-place operation
 */
#define NNL2_AXPY_INPLACE_PARALLEL_THRESHOLD 1000000

/** @brief
 * Worker function for parallel double precision AXPY in-place operation
 * 
 ** @param arg 
 * Pointer to axpy_inplace_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 * 
 ** @details
 * Processes FLOAT64 data using AVX256 instructions with cache prefetching
 * for optimal memory access patterns in AXPY in-place operation
 */
void* nnl2_own_paxpy_inplace_float64(void* arg);

/** @brief
 * Worker function for parallel single precision AXPY in-place operation
 * 
 ** @param arg 
 * Pointer to axpy_inplace_ptask structure containing thread parameters  
 * 
 ** @return 
 * NULL (for pthread API compatibility)
 * 
 ** @see nnl2_own_paxpy_inplace_float64
 **/
void* nnl2_own_paxpy_inplace_float32(void* arg);

/** @brief
 * Worker function for parallel integer AXPY in-place operation
 * 
 ** @param arg 
 * Pointer to axpy_inplace_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 * 
 ** @see nnl2_own_paxpy_inplace_float64
 **/
void* nnl2_own_paxpy_inplace_int32(void* arg);

/** @brief
 * High-performance parallel implementation of AXPY in-place operation
 * 
 ** @param summand 
 * Pointer to the summand tensor (will be modified in-place)
 *
 ** @param sumend 
 * Pointer to the sumend tensor
 *
 ** @param alpha
 * Scalar multiplier for the sumend tensor values
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
 * Modifies the summand tensor directly
 * Both tensors must have the same shape
 */
void nnl2_own_axpy_inplace(nnl2_tensor* summand, const nnl2_tensor* sumend, float alpha) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks at the maximum safety level
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(summand, "Summand tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(summand->data, "Summand tensor data is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(sumend, "Sumend tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(sumend->data, "Sumend tensor data is NULL");
    #endif
    
    size_t total_elems = nnl2_product(summand->shape, summand->rank);
    if(total_elems == 0) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return;
    }
    
    nnl2_tensor_type dtype_summand = summand->dtype;
    nnl2_tensor_type dtype_sumend = sumend->dtype;
    
    // Fallback to naive implementation for small tensors or mixed types
    if(total_elems < NNL2_AXPY_INPLACE_PARALLEL_THRESHOLD || dtype_summand != dtype_sumend) {
        naive_axpy_inplace(summand, (nnl2_tensor*)sumend, alpha);
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
            NNL2_WARN("In nnl2_own_axpy_inplace, tensor memory is not aligned to 32 bytes. Performance may be suboptimal");
        }
    #endif
    
    size_t num_threads = NNL2_NUM_THREADS;
    pthread_t threads[num_threads];
    axpy_inplace_ptask tasks[num_threads];
    
    // Calculate optimal chunk sizes with load balancing
    size_t chunk = total_elems / num_threads;
    size_t remainder = total_elems % num_threads;
    
    // Configure alpha value based on data type
    for (size_t i = 0; i < num_threads; i++) {
        tasks[i].summand = summand;
        tasks[i].sumend = sumend;
        tasks[i].dtype_summand = dtype_summand;
        tasks[i].dtype_sumend = dtype_sumend;
        tasks[i].aligned = is_aligned;
        
        switch(dtype_summand) {
            case FLOAT64: tasks[i].alpha.float64_alpha = (double)alpha; break;
            case FLOAT32: tasks[i].alpha.float32_alpha = alpha; break;
            case INT32:   tasks[i].alpha.int32_alpha = (int32_t)alpha; break;
            default: {
                NNL2_TYPE_ERROR(dtype_summand);
                #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                    NNL2_FUNC_EXIT();
                #endif
                return;
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
            case FLOAT64: worker_func = nnl2_own_paxpy_inplace_float64; break;
            case FLOAT32: worker_func = nnl2_own_paxpy_inplace_float32; break;
            case INT32:   worker_func = nnl2_own_paxpy_inplace_int32;   break;
            default: {
                NNL2_TYPE_ERROR(dtype_summand);
                #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                    NNL2_FUNC_EXIT();
                #endif
                return;
            }
        }
        
        // Create thread to process the assigned chunk
        int status = pthread_create(&threads[i], NULL, worker_func, &tasks[i]);
        if(status != 0) {
            NNL2_THREAD_CREATE_ERROR(status, "nnl2_own_axpy_inplace");
            num_threads = i;
            break;
        }
        
        current_start += current_chunk;
    }
    
    // Wait for all threads to complete
    for (size_t i = 0; i < num_threads; i++) {
        int join_status = pthread_join(threads[i], NULL);
        if(join_status != 0) {
            NNL2_THREAD_JOIN_ERROR(join_status, "nnl2_own_axpy_inplace");
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
 ** @see nnl2_own_paxpy_inplace_float64
 **/
void* nnl2_own_paxpy_inplace_float64(void* arg) {
    axpy_inplace_ptask* task = (axpy_inplace_ptask*)arg;
    double* summand_data = (double*)task->summand->data;
    const double* sumend_data = (const double*)task->sumend->data;
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
            _mm256_store_pd(&summand_data[i], v_result);
        }
    } else {
        for(; i + 3 < end; i += 4) {
            _mm_prefetch((char*)&summand_data[i + 16], _MM_HINT_T0);
            _mm_prefetch((char*)&sumend_data[i + 16], _MM_HINT_T0);
            
            __m256d v_summand = _mm256_loadu_pd(&summand_data[i]);
            __m256d v_sumend = _mm256_loadu_pd(&sumend_data[i]);
            __m256d v_scaled_sumend = _mm256_mul_pd(v_sumend, v_alpha);
            __m256d v_result = _mm256_add_pd(v_summand, v_scaled_sumend);
            _mm256_storeu_pd(&summand_data[i], v_result);
        }
    }
    
    // Scalar processing for remainder
    for(; i < end; i++) {
        summand_data[i] += sumend_data[i] * alpha;
    }
    
    return NULL;
}

/** @brief
 * See documentation at declaration
 * 
 ** @see nnl2_own_paxpy_inplace_float32
 **/
void* nnl2_own_paxpy_inplace_float32(void* arg) {
    axpy_inplace_ptask* task = (axpy_inplace_ptask*)arg;
    float* summand_data = (float*)task->summand->data;
    const float* sumend_data = (const float*)task->sumend->data;
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
            _mm256_store_ps(&summand_data[i], v_result);
        }
    } else {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&summand_data[i + 32], _MM_HINT_T0);
            _mm_prefetch((char*)&sumend_data[i + 32], _MM_HINT_T0);
            
            __m256 v_summand = _mm256_loadu_ps(&summand_data[i]);
            __m256 v_sumend = _mm256_loadu_ps(&sumend_data[i]);
            __m256 v_scaled_sumend = _mm256_mul_ps(v_sumend, v_alpha);
            __m256 v_result = _mm256_add_ps(v_summand, v_scaled_sumend);
            _mm256_storeu_ps(&summand_data[i], v_result);
        }
    }
    
    // Scalar processing for remainder
    for(; i < end; i++) {
        summand_data[i] += sumend_data[i] * alpha;
    }
    
    return NULL;
}

/** @brief
 * See documentation at declaration
 * 
 ** @see nnl2_own_paxpy_inplace_int32
 **/
void* nnl2_own_paxpy_inplace_int32(void* arg) {
    axpy_inplace_ptask* task = (axpy_inplace_ptask*)arg;
    int32_t* summand_data = (int32_t*)task->summand->data;
    const int32_t* sumend_data = (const int32_t*)task->sumend->data;
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
            _mm256_store_si256((__m256i*)&summand_data[i], v_result);
        }
    } else {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&summand_data[i + 32], _MM_HINT_T0);
            _mm_prefetch((char*)&sumend_data[i + 32], _MM_HINT_T0);
            
            __m256i v_summand = _mm256_loadu_si256((__m256i*)&summand_data[i]);
            __m256i v_sumend = _mm256_loadu_si256((__m256i*)&sumend_data[i]);
            __m256i v_scaled_sumend = _mm256_mullo_epi32(v_sumend, v_alpha);
            __m256i v_result = _mm256_add_epi32(v_summand, v_scaled_sumend);
            _mm256_storeu_si256((__m256i*)&summand_data[i], v_result);
        }
    }
    
    // Scalar processing for remainder
    for(; i < end; i++) {
        summand_data[i] += sumend_data[i] * alpha;
    }
    
    return NULL;
}

#endif

/**
 * @ingroup backend_system
 * @brief Backend implementations for AXPY in-place operation
 */
nnl2_runtime_implementation axpy_inplace_backends[] = {
    REGISTER_BACKEND(naive_axpy_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
    
    #if defined(NNL2_AVX256_AVAILABLE) && defined(NNL2_PTHREAD_AVAILABLE)
        REGISTER_BACKEND(nnl2_own_axpy_inplace, nnl2_own, NNL2_OWN_NAME),
    #endif
};  

/**
 * @brief Function pointer for AXPY in-place operation
 * @ingroup backend_system
 */
axpyinplacefn axpy_inplace;
MAKE_CURRENT_BACKEND(axpy_inplace);

/**
 * @brief Sets the backend for AXPY in-place operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for AXPY in-place operation
 */
void set_axpy_inplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(axpy_inplace_backends, axpy_inplace, backend_name, CURRENT_BACKEND(axpy_inplace));
}

/**
 * @brief Gets the name of the current backend for AXPY in-place operation
 * @ingroup backend_system
 * @return const char* Name of the current backend
 */
const char* get_axpy_inplace_backend() {
    return CURRENT_BACKEND(axpy_inplace);
}

/**
 * @brief Gets the list of available backends for AXPY in-place operation
 * @ingroup backend_system
 * @return const char** Array of backend names
 */
DEFINE_GET_BACKENDS_FUNCTION(axpy_inplace);

/**
 * @brief Gets the number of available backends for AXPY in-place operation
 * @ingroup backend_system
 * @return size_t Number of available backends
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(axpy_inplace);

#endif /** NNL2_AXPY_INPLACE_H **/
