#ifndef NNL2_MUL_INPLACE_H
#define NNL2_MUL_INPLACE_H

/** @brief 
 * Performs element-wise multiplication of two tensors (naive implementation)
 * 
 * Multiplies the elements of the multiplicand tensor by the corresponding elements 
 * of the multiplier tensor, modifying the multiplicand tensor in place
 *
 ** @param multiplicand 
 * Pointer to the tensor that will be modified (receives the multiplication result)
 *
 ** @param multiplier 
 * Pointer to the tensor whose values will multiply the multiplicand
 *
 ** @note 
 * Supports different data types through automatic conversion
 * The multiplier elements are converted to the multiplicand's data type
 *
 ** @note 
 * In max safety mode, validates tensor pointers and data pointers before access
 *
 ** @warning 
 * This function modifies the multiplicand tensor directly
 *
 * @example
 * // Create two tensors with the same shape
 * Tensor* a = nnl2_ones((int[]){2, 3}, 2, FLOAT32);
 * Tensor* b = nnl2_ones((int[]){2, 3}, 2, FLOAT32);
 * 
 * // Multiply a by b (a becomes a * b)
 * naive_mulinplace(a, b);
 * 
 * // Now a contains 1.0 in all elements
 * nnl2_quick_print_tensor(a);
 * 
 * // Cleanup
 * nnl2_free_tensor(a);
 * nnl2_free_tensor(b);
 */
void nnl2_naive_mulinplace(Tensor* multiplicand, const Tensor* multiplier) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks at the maximum safety level
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX   
        NNL2_CHECK_NULL_IF_ERR_RETURN(multiplicand, "Multiplicand tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(multiplicand->data, "Multiplicand tensor's data is NULL");
        
        NNL2_CHECK_NULL_IF_ERR_RETURN(multiplier, "Multiplier tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(multiplier->data, "Multiplier tensor's data is NULL");
    #endif
    
    // Calculating the total number of elements in the multiplicand tensor
    size_t len_multiplicand = product(multiplicand->shape, multiplicand->rank);
    
    // If the tensor is empty, exit the function
    if(len_multiplicand == 0) return;
    
    TensorType dtype_multiplicand = multiplicand->dtype;
    TensorType dtype_multiplier = multiplier->dtype;
    
    if(dtype_multiplicand == dtype_multiplier) {
        // Handling case when the tensors have the same type
        
        switch(dtype_multiplicand) {
            case FLOAT64: {
                volatile double* data_multiplicand = (double*)multiplicand->data;
                volatile double* data_multiplier = (double*)multiplier->data;
                
                // Element-wise multiplication
                for(size_t i = 0; i < len_multiplicand; i++) data_multiplicand[i] *= data_multiplier[i];
                break;
            }
            
            case FLOAT32: {
                volatile float* data_multiplicand = (float*)multiplicand->data;
                volatile float* data_multiplier = (float*)multiplier->data;
                
                // Element-wise multiplication
                for(size_t i = 0; i < len_multiplicand; i++) data_multiplicand[i] *= data_multiplier[i];    
                break;
            }
            
            case INT32: {
                volatile int32_t* data_multiplicand = (int32_t*)multiplicand->data;
                volatile int32_t* data_multiplier = (int32_t*)multiplier->data;
                
                // Element-wise multiplication
                for(size_t i = 0; i < len_multiplicand; i++) data_multiplicand[i] *= data_multiplier[i];        
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(dtype_multiplicand);
                return;
            }
        }
    } else {
        // Handling the case when tensors have different data types
        // Calculating the step size for accessing multiplier tensor elements
        size_t multiplier_step = get_dtype_size(dtype_multiplier);
        
        // Casting multiplier data to char* for byte access
        char* multiplier_data = (char*)multiplier->data;
        
        switch(dtype_multiplicand) {
            case FLOAT64: {
                volatile double* data_multiplicand = (double*)multiplicand->data;
                
                // For each element, convert the multiplier element to FLOAT64 and multiply it
                for(size_t i = 0; i < len_multiplicand; i++) {
                    void* multiplier_elem = multiplier_data + i * multiplier_step;
                    data_multiplicand[i] *= nnl2_convert_to_float64(multiplier_elem, dtype_multiplier);
                }
                
                break; 
            }
            
            case FLOAT32: {
                volatile float* data_multiplicand = (float*)multiplicand->data;
                
                // For each element, convert the multiplier element to FLOAT32 and multiply it
                for(size_t i = 0; i < len_multiplicand; i++) {
                    void* multiplier_elem = multiplier_data + i * multiplier_step;
                    data_multiplicand[i] *= nnl2_convert_to_float32(multiplier_elem, dtype_multiplier);
                }
                
                break; 
            }
            
            case INT32: {
                volatile int32_t* data_multiplicand = (int32_t*)multiplicand->data;
                
                // For each element, convert the multiplier element to INT32 and multiply it
                for(size_t i = 0; i < len_multiplicand; i++) {
                    void* multiplier_elem = multiplier_data + i * multiplier_step;
                    data_multiplicand[i] *= nnl2_convert_to_int32(multiplier_elem, dtype_multiplier);
                }
                
                break; 
            }
            
            default: {
                NNL2_TYPE_ERROR(dtype_multiplicand);
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
 * Threshold for enabling parallel execution of the
 * multiplication in-place operation
 */
#define NNL2_MUL_INPLACE_PARALLEL_THRESHOLD 1000000

/** @brief 
 * Worker function for parallel double precision multiplication with same data types
 * 
 ** @param arg 
 * Pointer to mulinplace_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 * 
 ** @details
 * Processes FLOAT64 data using AVX256 instructions with cache prefetching
 * for optimal memory access patterns
 */
void* nnl2_own_pmulinplace_float64_same_type(void* arg);

/** @brief
 * Worker function for parallel single precision multiplication with same data types
 * 
 ** @param arg 
 * Pointer to mulinplace_ptask structure containing thread parameters  
 * 
 ** @return 
 * NULL (for pthread API compatibility)
 * 
 ** @see nnl2_own_pmulinplace_float64_same_type
 **/
void* nnl2_own_pmulinplace_float32_same_type(void* arg);

/** @brief
 * Worker function for parallel integer multiplication with same data types
 * 
 ** @param arg 
 * Pointer to mulinplace_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 * 
 ** @see nnl2_own_pmulinplace_float64_same_type
 **/
void* nnl2_own_pmulinplace_int32_same_type(void* arg);

/** @brief
 * Worker function for parallel double precision multiplication with type conversion
 * 
 ** @param arg 
 * Pointer to mulinplace_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 * 
 ** @details
 * Handles type conversion from multiplier to multiplicand data type with AVX256
 * optimizations and prefetching
 */
void* nnl2_own_pmulinplace_float64_diff_type(void* arg);

/** @brief
 * Worker function for parallel single precision multiplication with type conversion
 * 
 ** @param arg 
 * Pointer to mulinplace_ptask structure containing thread parameters
 *
 ** @return
 * NULL (for pthread API compatibility)
 * 
 ** @see nnl2_own_pmulinplace_float64_diff_type
 **/
void* nnl2_own_pmulinplace_float32_diff_type(void* arg);

/** @brief
 * Worker function for parallel integer multiplication with type conversion
 * 
 ** @param arg
 * Pointer to mulinplace_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 * 
 ** @see nnl2_own_pmulinplace_float64_diff_type
 **/
void* nnl2_own_pmulinplace_int32_diff_type(void* arg);

/** @brief
 * High-performance parallel implementation of in-place multiplication
 * 
 ** @param multiplicand 
 * Pointer to tensor that will be modified in-place
 *
 ** @param multiplier 
 * Pointer to tensor whose values will multiply the multiplicand
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
void nnl2_own_mulinplace(Tensor* multiplicand, const Tensor* multiplier) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks at the maximum safety level
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(multiplicand, "Multiplicand tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(multiplicand->data, "Multiplicand tensor's data is NULL");
        
        NNL2_CHECK_NULL_IF_ERR_RETURN(multiplier, "Multiplier tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(multiplier->data, "Multiplier tensor's data is NULL");
    #endif
    
    // Calculating the total number of elements in the multiplicand tensor
    size_t len_multiplicand = product(multiplicand->shape, multiplicand->rank);
    
    // Fallback to naive implementation for small tensors
    if(len_multiplicand < NNL2_MUL_INPLACE_PARALLEL_THRESHOLD) {
        nnl2_naive_mulinplace(multiplicand, multiplier);
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return;
    }
    
    TensorType dtype_multiplicand = multiplicand->dtype;
    TensorType dtype_multiplier = multiplier->dtype;
    
    bool is_aligned_multiplicand = NNL2_IS_ALIGNED(multiplicand->data, NNL2_TENSOR_ALIGNMENT_32);
    bool is_aligned_multiplier = NNL2_IS_ALIGNED(multiplier->data, NNL2_TENSOR_ALIGNMENT_32);
    
    // Warning for unaligned memory in safety modes
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!is_aligned_multiplicand) {
            NNL2_WARN("In nnl2_own_2 mul in-place, multiplicand memory is not aligned to 32 bytes. Performance may be suboptimal");
        }
        
        if(!is_aligned_multiplier && dtype_multiplicand == dtype_multiplier) {
            NNL2_WARN("In nnl2_own_2 mul in-place, multiplier memory is not aligned to 32 bytes. Performance may be suboptimal");
        }
    #endif
    
    size_t num_threads = NNL2_NUM_THREADS;
    pthread_t threads[num_threads];
    mulinplace_ptask tasks[num_threads];
    
    // Calculate optimal chunk sizes with load balancing
    size_t chunk = len_multiplicand / num_threads;
    size_t remainder = len_multiplicand % num_threads;
    
    size_t current_start = 0;
    for (size_t i = 0; i < num_threads; i++) {
        size_t current_chunk = chunk + (i < remainder ? 1 : 0);
        
        // Configure task for this thread
        tasks[i].multiplicand_data = multiplicand->data;
        tasks[i].multiplier_data = multiplier->data;
        tasks[i].start = current_start;
        tasks[i].end = current_start + current_chunk;
        tasks[i].dtype_multiplicand = dtype_multiplicand;
        tasks[i].dtype_multiplier = dtype_multiplier;
        tasks[i].aligned_multiplicand = is_aligned_multiplicand;
        tasks[i].aligned_multiplier = is_aligned_multiplier;
        tasks[i].multiplier_step = (dtype_multiplicand == dtype_multiplier) ? get_dtype_size(dtype_multiplier) : 0;
        
        // Select appropriate worker function based on data types
        void* (*worker_func)(void*) = NULL;
        
        if(dtype_multiplicand == dtype_multiplier) {
            switch(dtype_multiplicand) {
                case FLOAT64: worker_func = nnl2_own_pmulinplace_float64_same_type; break;
                case FLOAT32: worker_func = nnl2_own_pmulinplace_float32_same_type; break;
                case INT32:   worker_func = nnl2_own_pmulinplace_int32_same_type;   break;
                default: {
                    NNL2_TYPE_ERROR(dtype_multiplicand);
                    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                        NNL2_FUNC_EXIT();
                    #endif
                    return;
                }
            }
        } else {
            tasks[i].multiplier_step = get_dtype_size(dtype_multiplier);
            switch(dtype_multiplicand) {
                case FLOAT64: worker_func = nnl2_own_pmulinplace_float64_diff_type; break;
                case FLOAT32: worker_func = nnl2_own_pmulinplace_float32_diff_type; break;
                case INT32:   worker_func = nnl2_own_pmulinplace_int32_diff_type;   break;
                default: {
                    NNL2_TYPE_ERROR(dtype_multiplicand);
                    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                        NNL2_FUNC_EXIT();
                    #endif
                    return;
                }
            }
        }
        
        // Create thread to process the assigned chunk
        int status = pthread_create(&threads[i], NULL, worker_func, &tasks[i]);
        if(status != 0) {
            NNL2_THREAD_CREATE_ERROR(status, "nnl2_own_2_mulinplace");
            num_threads = i;
            break;
        }
        
        current_start += current_chunk;
    }
    
    // Wait for all threads to complete
    for (size_t i = 0; i < num_threads; i++) {
        int join_status = pthread_join(threads[i], NULL);
        if(join_status != 0) {
            NNL2_THREAD_JOIN_ERROR(join_status, "nnl2_own_2_mulinplace");
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
 ** @see nnl2_own_pmulinplace_float64_same_type
 **/
void* nnl2_own_pmulinplace_float64_same_type(void* arg) {
    mulinplace_ptask* task = (mulinplace_ptask*)arg;
    double* multiplicand = (double*)task->multiplicand_data;
    double* multiplier = (double*)task->multiplier_data;
    size_t start = task->start;
    size_t end = task->end;
    
    size_t i = start;
    
    // AVX256 processing with prefetching
    if(task->aligned_multiplicand && task->aligned_multiplier) {
        for(; i + 3 < end; i += 4) {
            // Prefetch next cache lines
            _mm_prefetch((char*)&multiplicand[i + 16], _MM_HINT_T0);
            _mm_prefetch((char*)&multiplier[i + 16], _MM_HINT_T0);
            
            __m256d v_multiplicand = _mm256_load_pd(&multiplicand[i]);
            __m256d v_multiplier = _mm256_load_pd(&multiplier[i]);
            __m256d v_result = _mm256_mul_pd(v_multiplicand, v_multiplier);
            _mm256_store_pd(&multiplicand[i], v_result);
        }
    } else if(task->aligned_multiplicand) {
        for(; i + 3 < end; i += 4) {
            _mm_prefetch((char*)&multiplicand[i + 16], _MM_HINT_T0);
            _mm_prefetch((char*)&multiplier[i + 16], _MM_HINT_T0);
            
            __m256d v_multiplicand = _mm256_load_pd(&multiplicand[i]);
            __m256d v_multiplier = _mm256_loadu_pd(&multiplier[i]);
            __m256d v_result = _mm256_mul_pd(v_multiplicand, v_multiplier);
            _mm256_store_pd(&multiplicand[i], v_result);
        }
    } else if(task->aligned_multiplier) {
        for(; i + 3 < end; i += 4) {
            _mm_prefetch((char*)&multiplicand[i + 16], _MM_HINT_T0);
            _mm_prefetch((char*)&multiplier[i + 16], _MM_HINT_T0);
            
            __m256d v_multiplicand = _mm256_loadu_pd(&multiplicand[i]);
            __m256d v_multiplier = _mm256_load_pd(&multiplier[i]);
            __m256d v_result = _mm256_mul_pd(v_multiplicand, v_multiplier);
            _mm256_storeu_pd(&multiplicand[i], v_result);
        }
    } else {
        for(; i + 3 < end; i += 4) {
            _mm_prefetch((char*)&multiplicand[i + 16], _MM_HINT_T0);
            _mm_prefetch((char*)&multiplier[i + 16], _MM_HINT_T0);
            
            __m256d v_multiplicand = _mm256_loadu_pd(&multiplicand[i]);
            __m256d v_multiplier = _mm256_loadu_pd(&multiplier[i]);
            __m256d v_result = _mm256_mul_pd(v_multiplicand, v_multiplier);
            _mm256_storeu_pd(&multiplicand[i], v_result);
        }
    }
    
    // Scalar processing for remainder
    for(; i < end; i++) {
        multiplicand[i] *= multiplier[i];
    }
    
    return NULL;
}

/** @brief
 * See documentation at declaration
 * 
 ** @see nnl2_own_pmulinplace_float32_same_type
 **/
void* nnl2_own_pmulinplace_float32_same_type(void* arg) {
    mulinplace_ptask* task = (mulinplace_ptask*)arg;
    float* multiplicand = (float*)task->multiplicand_data;
    float* multiplier = (float*)task->multiplier_data;
    size_t start = task->start;
    size_t end = task->end;
    
    size_t i = start;
    
    // AVX256 processing with prefetching (8 elements per iteration)
    if(task->aligned_multiplicand && task->aligned_multiplier) {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&multiplicand[i + 32], _MM_HINT_T0);
            _mm_prefetch((char*)&multiplier[i + 32], _MM_HINT_T0);
            
            __m256 v_multiplicand = _mm256_load_ps(&multiplicand[i]);
            __m256 v_multiplier = _mm256_load_ps(&multiplier[i]);
            __m256 v_result = _mm256_mul_ps(v_multiplicand, v_multiplier);
            _mm256_store_ps(&multiplicand[i], v_result);
        }
    } else if(task->aligned_multiplicand) {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&multiplicand[i + 32], _MM_HINT_T0);
            _mm_prefetch((char*)&multiplier[i + 32], _MM_HINT_T0);
            
            __m256 v_multiplicand = _mm256_load_ps(&multiplicand[i]);
            __m256 v_multiplier = _mm256_loadu_ps(&multiplier[i]);
            __m256 v_result = _mm256_mul_ps(v_multiplicand, v_multiplier);
            _mm256_store_ps(&multiplicand[i], v_result);
        }
    } else if(task->aligned_multiplier) {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&multiplicand[i + 32], _MM_HINT_T0);
            _mm_prefetch((char*)&multiplier[i + 32], _MM_HINT_T0);
            
            __m256 v_multiplicand = _mm256_loadu_ps(&multiplicand[i]);
            __m256 v_multiplier = _mm256_load_ps(&multiplier[i]);
            __m256 v_result = _mm256_mul_ps(v_multiplicand, v_multiplier);
            _mm256_storeu_ps(&multiplicand[i], v_result);
        }
    } else {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&multiplicand[i + 32], _MM_HINT_T0);
            _mm_prefetch((char*)&multiplier[i + 32], _MM_HINT_T0);
            
            __m256 v_multiplicand = _mm256_loadu_ps(&multiplicand[i]);
            __m256 v_multiplier = _mm256_loadu_ps(&multiplier[i]);
            __m256 v_result = _mm256_mul_ps(v_multiplicand, v_multiplier);
            _mm256_storeu_ps(&multiplicand[i], v_result);
        }
    }
    
    // Scalar processing for remainder
    for(; i < end; i++) {
        multiplicand[i] *= multiplier[i];
    }
    
    return NULL;
}

/** @brief
 * See documentation at declaration
 * 
 ** @see nnl2_own_pmulinplace_int32_same_type
 **/
void* nnl2_own_pmulinplace_int32_same_type(void* arg) {
    mulinplace_ptask* task = (mulinplace_ptask*)arg;
    int32_t* multiplicand = (int32_t*)task->multiplicand_data;
    int32_t* multiplier = (int32_t*)task->multiplier_data;
    size_t start = task->start;
    size_t end = task->end;
    
    size_t i = start;
    
    // AVX256 processing with prefetching (8 elements per iteration)
    if(task->aligned_multiplicand && task->aligned_multiplier) {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&multiplicand[i + 32], _MM_HINT_T0);
            _mm_prefetch((char*)&multiplier[i + 32], _MM_HINT_T0);
            
            __m256i v_multiplicand = _mm256_load_si256((__m256i*)&multiplicand[i]);
            __m256i v_multiplier = _mm256_load_si256((__m256i*)&multiplier[i]);
            __m256i v_result = _mm256_mullo_epi32(v_multiplicand, v_multiplier);
            _mm256_store_si256((__m256i*)&multiplicand[i], v_result);
        }
    } else if(task->aligned_multiplicand) {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&multiplicand[i + 32], _MM_HINT_T0);
            _mm_prefetch((char*)&multiplier[i + 32], _MM_HINT_T0);
            
            __m256i v_multiplicand = _mm256_load_si256((__m256i*)&multiplicand[i]);
            __m256i v_multiplier = _mm256_loadu_si256((__m256i*)&multiplier[i]);
            __m256i v_result = _mm256_mullo_epi32(v_multiplicand, v_multiplier);
            _mm256_store_si256((__m256i*)&multiplicand[i], v_result);
        }
    } else if(task->aligned_multiplier) {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&multiplicand[i + 32], _MM_HINT_T0);
            _mm_prefetch((char*)&multiplier[i + 32], _MM_HINT_T0);
            
            __m256i v_multiplicand = _mm256_loadu_si256((__m256i*)&multiplicand[i]);
            __m256i v_multiplier = _mm256_load_si256((__m256i*)&multiplier[i]);
            __m256i v_result = _mm256_mullo_epi32(v_multiplicand, v_multiplier);
            _mm256_storeu_si256((__m256i*)&multiplicand[i], v_result);
        }
    } else {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&multiplicand[i + 32], _MM_HINT_T0);
            _mm_prefetch((char*)&multiplier[i + 32], _MM_HINT_T0);
            
            __m256i v_multiplicand = _mm256_loadu_si256((__m256i*)&multiplicand[i]);
            __m256i v_multiplier = _mm256_loadu_si256((__m256i*)&multiplier[i]);
            __m256i v_result = _mm256_mullo_epi32(v_multiplicand, v_multiplier);
            _mm256_storeu_si256((__m256i*)&multiplicand[i], v_result);
        }
    }
    
    // Scalar processing for remainder
    for(; i < end; i++) {
        multiplicand[i] *= multiplier[i];
    }
    
    return NULL;
}

// Different type worker functions with conversion and prefetching

/** @brief
 * See documentation at declaration
 * 
 ** @see nnl2_own_pmulinplace_float64_diff_type
 **/
void* nnl2_own_pmulinplace_float64_diff_type(void* arg) {
    mulinplace_ptask* task = (mulinplace_ptask*)arg;
    double* multiplicand = (double*)task->multiplicand_data;
    char* multiplier_data = (char*)task->multiplier_data;
    size_t start = task->start;
    size_t end = task->end;
    size_t multiplier_step = task->multiplier_step;
    
    size_t i = start;
    
    // AVX256 processing with type conversion and prefetching
    if(task->aligned_multiplicand) {
        for(; i + 3 < end; i += 4) {
            _mm_prefetch((char*)&multiplicand[i + 16], _MM_HINT_T0);
            
            __m256d v_multiplicand = _mm256_load_pd(&multiplicand[i]);
            __m256d v_multiplier = _mm256_set_pd(
                nnl2_convert_to_float64(multiplier_data + (i + 3) * multiplier_step, task->dtype_multiplier),
                nnl2_convert_to_float64(multiplier_data + (i + 2) * multiplier_step, task->dtype_multiplier),
                nnl2_convert_to_float64(multiplier_data + (i + 1) * multiplier_step, task->dtype_multiplier),
                nnl2_convert_to_float64(multiplier_data + (i + 0) * multiplier_step, task->dtype_multiplier)
            );
            
            __m256d v_result = _mm256_mul_pd(v_multiplicand, v_multiplier);
            _mm256_store_pd(&multiplicand[i], v_result);
        }
    } else {
        for(; i + 3 < end; i += 4) {
            _mm_prefetch((char*)&multiplicand[i + 16], _MM_HINT_T0);
            
            __m256d v_multiplicand = _mm256_loadu_pd(&multiplicand[i]);
            __m256d v_multiplier = _mm256_set_pd(
                nnl2_convert_to_float64(multiplier_data + (i + 3) * multiplier_step, task->dtype_multiplier),
                nnl2_convert_to_float64(multiplier_data + (i + 2) * multiplier_step, task->dtype_multiplier),
                nnl2_convert_to_float64(multiplier_data + (i + 1) * multiplier_step, task->dtype_multiplier),
                nnl2_convert_to_float64(multiplier_data + (i + 0) * multiplier_step, task->dtype_multiplier)
            );
            
            __m256d v_result = _mm256_mul_pd(v_multiplicand, v_multiplier);
            _mm256_storeu_pd(&multiplicand[i], v_result);
        }
    }
    
    // Scalar processing for remainder
    for(; i < end; i++) {
        void* multiplier_elem = multiplier_data + i * multiplier_step;
        multiplicand[i] *= nnl2_convert_to_float64(multiplier_elem, task->dtype_multiplier);
    }
    
    return NULL;
}

/** @brief
 * See documentation at declaration
 * 
 ** @see nnl2_own_pmulinplace_float32_diff_type
 **/
void* nnl2_own_pmulinplace_float32_diff_type(void* arg) {
    mulinplace_ptask* task = (mulinplace_ptask*)arg;
    float* multiplicand = (float*)task->multiplicand_data;
    char* multiplier_data = (char*)task->multiplier_data;
    size_t start = task->start;
    size_t end = task->end;
    size_t multiplier_step = task->multiplier_step;
    
    size_t i = start;
    
    // AVX256 processing with type conversion and prefetching
    if(task->aligned_multiplicand) {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&multiplicand[i + 32], _MM_HINT_T0);
            
            __m256 v_multiplicand = _mm256_load_ps(&multiplicand[i]);
            __m256 v_multiplier = _mm256_set_ps(
                nnl2_convert_to_float32(multiplier_data + (i + 7) * multiplier_step, task->dtype_multiplier),
                nnl2_convert_to_float32(multiplier_data + (i + 6) * multiplier_step, task->dtype_multiplier),
                nnl2_convert_to_float32(multiplier_data + (i + 5) * multiplier_step, task->dtype_multiplier),
                nnl2_convert_to_float32(multiplier_data + (i + 4) * multiplier_step, task->dtype_multiplier),
                nnl2_convert_to_float32(multiplier_data + (i + 3) * multiplier_step, task->dtype_multiplier),
                nnl2_convert_to_float32(multiplier_data + (i + 2) * multiplier_step, task->dtype_multiplier),
                nnl2_convert_to_float32(multiplier_data + (i + 1) * multiplier_step, task->dtype_multiplier),
                nnl2_convert_to_float32(multiplier_data + (i + 0) * multiplier_step, task->dtype_multiplier)
            );
            
            __m256 v_result = _mm256_mul_ps(v_multiplicand, v_multiplier);
            _mm256_store_ps(&multiplicand[i], v_result);
        }
    } else {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&multiplicand[i + 32], _MM_HINT_T0);
            
            __m256 v_multiplicand = _mm256_loadu_ps(&multiplicand[i]);
            __m256 v_multiplier = _mm256_set_ps(
                nnl2_convert_to_float32(multiplier_data + (i + 7) * multiplier_step, task->dtype_multiplier),
                nnl2_convert_to_float32(multiplier_data + (i + 6) * multiplier_step, task->dtype_multiplier),
                nnl2_convert_to_float32(multiplier_data + (i + 5) * multiplier_step, task->dtype_multiplier),
                nnl2_convert_to_float32(multiplier_data + (i + 4) * multiplier_step, task->dtype_multiplier),
                nnl2_convert_to_float32(multiplier_data + (i + 3) * multiplier_step, task->dtype_multiplier),
                nnl2_convert_to_float32(multiplier_data + (i + 2) * multiplier_step, task->dtype_multiplier),
                nnl2_convert_to_float32(multiplier_data + (i + 1) * multiplier_step, task->dtype_multiplier),
                nnl2_convert_to_float32(multiplier_data + (i + 0) * multiplier_step, task->dtype_multiplier)
            );
            
            __m256 v_result = _mm256_mul_ps(v_multiplicand, v_multiplier);
            _mm256_storeu_ps(&multiplicand[i], v_result);
        }
    }
    
    // Scalar processing for remainder
    for(; i < end; i++) {
        void* multiplier_elem = multiplier_data + i * multiplier_step;
        multiplicand[i] *= nnl2_convert_to_float32(multiplier_elem, task->dtype_multiplier);
    }
    
    return NULL;
}

/** @brief
 * See documentation at declaration
 * 
 ** @see nnl2_own_pmulinplace_int32_diff_type
 **/
void* nnl2_own_pmulinplace_int32_diff_type(void* arg) {
    mulinplace_ptask* task = (mulinplace_ptask*)arg;
    int32_t* multiplicand = (int32_t*)task->multiplicand_data;
    char* multiplier_data = (char*)task->multiplier_data;
    size_t start = task->start;
    size_t end = task->end;
    size_t multiplier_step = task->multiplier_step;
    
    size_t i = start;
    
    // AVX256 processing with type conversion and prefetching
    if(task->aligned_multiplicand) {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&multiplicand[i + 32], _MM_HINT_T0);
            
            __m256i v_multiplicand = _mm256_load_si256((__m256i*)&multiplicand[i]);
            __m256i v_multiplier = _mm256_set_epi32(
                nnl2_convert_to_int32(multiplier_data + (i + 7) * multiplier_step, task->dtype_multiplier),
                nnl2_convert_to_int32(multiplier_data + (i + 6) * multiplier_step, task->dtype_multiplier),
                nnl2_convert_to_int32(multiplier_data + (i + 5) * multiplier_step, task->dtype_multiplier),
                nnl2_convert_to_int32(multiplier_data + (i + 4) * multiplier_step, task->dtype_multiplier),
                nnl2_convert_to_int32(multiplier_data + (i + 3) * multiplier_step, task->dtype_multiplier),
                nnl2_convert_to_int32(multiplier_data + (i + 2) * multiplier_step, task->dtype_multiplier),
                nnl2_convert_to_int32(multiplier_data + (i + 1) * multiplier_step, task->dtype_multiplier),
                nnl2_convert_to_int32(multiplier_data + (i + 0) * multiplier_step, task->dtype_multiplier)
            );
            
            __m256i v_result = _mm256_mullo_epi32(v_multiplicand, v_multiplier);
            _mm256_store_si256((__m256i*)&multiplicand[i], v_result);
        }
    } else {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&multiplicand[i + 32], _MM_HINT_T0);
            
            __m256i v_multiplicand = _mm256_loadu_si256((__m256i*)&multiplicand[i]);
            __m256i v_multiplier = _mm256_set_epi32(
                nnl2_convert_to_int32(multiplier_data + (i + 7) * multiplier_step, task->dtype_multiplier),
                nnl2_convert_to_int32(multiplier_data + (i + 6) * multiplier_step, task->dtype_multiplier),
                nnl2_convert_to_int32(multiplier_data + (i + 5) * multiplier_step, task->dtype_multiplier),
                nnl2_convert_to_int32(multiplier_data + (i + 4) * multiplier_step, task->dtype_multiplier),
                nnl2_convert_to_int32(multiplier_data + (i + 3) * multiplier_step, task->dtype_multiplier),
                nnl2_convert_to_int32(multiplier_data + (i + 2) * multiplier_step, task->dtype_multiplier),
                nnl2_convert_to_int32(multiplier_data + (i + 1) * multiplier_step, task->dtype_multiplier),
                nnl2_convert_to_int32(multiplier_data + (i + 0) * multiplier_step, task->dtype_multiplier)
            );
            
            __m256i v_result = _mm256_mullo_epi32(v_multiplicand, v_multiplier);
            _mm256_storeu_si256((__m256i*)&multiplicand[i], v_result);
        }
    }
    
    // Scalar processing for remainder
    for(; i < end; i++) {
        void* multiplier_elem = multiplier_data + i * multiplier_step;
        multiplicand[i] *= nnl2_convert_to_int32(multiplier_elem, task->dtype_multiplier);
    }
    
    return NULL;
}

#endif

/** 
 * @ingroup backend_system
 * @brief Backend implementations for multiplication operation
 * @details
 * Array follows the common backend registration pattern.
 * Currently registered backends:
 *  - nnl2_naive_mulinplace: Basic reference implementation
 * 
 * @see nnl2_naive_mulinplace
 */
Implementation mulinplace_backends[] = {
	REGISTER_BACKEND(nnl2_naive_mulinplace, nnl2_naive, NAIVE_BACKEND_NAME),
	
	#if defined(NNL2_PTHREAD_AVAILABLE) && defined(NNL2_AVX256_AVAILABLE) 
		REGISTER_BACKEND(nnl2_own_mulinplace, nnl2_own_2, NNL2_OWN_NAME),
	#endif
};

/**
 * @brief Function pointer for multiplication operation
 * @ingroup backend_system 
 */
mulinplacefn mulinplace;

/** 
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
make_current_backend(mulinplace);

/** 
 * @brief Sets the backend for multiplication operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void set_mulinplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(mulinplace_backends, mulinplace, backend_name, CURRENT_BACKEND(mulinplace));
}

/** 
 * @brief Gets the name of the active backend for multiplication operation
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* get_mulinplace_backend() {
	return CURRENT_BACKEND(mulinplace);
}

/** 
 * @brief Function declaration for getting all `mulinplace` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(mulinplace);

/**
 * @brief Function declaration for getting the number of all `mulinplace` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(mulinplace);

#endif /** NNL2_MUL_INPLACE_H **/
