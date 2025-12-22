#ifndef NNL2_DIV_INPLACE_H
#define NNL2_DIV_INPLACE_H

/** @brief 
 * Performs element-wise division of two tensors (naive implementation)
 * 
 * Divides the elements of the dividend tensor by the corresponding elements 
 * of the divisor tensor, modifying the dividend tensor in place
 *
 ** @param dividend 
 * Pointer to the tensor that will be modified (receives the division result)
 *
 ** @param divisor 
 * Pointer to the tensor whose values will divide the dividend
 *
 ** @note 
 * Supports different data types through automatic conversion
 * The divisor elements are converted to the dividend's data type
 *
 ** @note 
 * In max safety mode, validates tensor pointers and data pointers before access
 *
 ** @warning 
 * This function modifies the dividend tensor directly
 * Division by zero may result in undefined behavior depending on data type
 *
 * @example
 * // Create two tensors with the same shape
 * nnl2_tensor* a = nnl2_ones((int[]){2, 3}, 2, FLOAT32);
 * nnl2_tensor* b = nnl2_ones((int[]){2, 3}, 2, FLOAT32);
 * 
 * // Divide a by b (a becomes a / b)
 * nnl2_naive_divinplace(a, b);
 * 
 * // Now a contains 1.0 in all elements
 * nnl2_quick_print_tensor(a);
 * 
 * // Cleanup
 * nnl2_free_tensor(a);
 * nnl2_free_tensor(b);
 */
void nnl2_naive_divinplace(nnl2_tensor* dividend, const nnl2_tensor* divisor) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks at the maximum safety level
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(dividend, "Dividend tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(dividend->data, "Dividend tensor's data is NULL");
        
        NNL2_CHECK_NULL_IF_ERR_RETURN(divisor, "Divisor tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(divisor->data, "Divisor tensor's data is NULL");
    #endif
    
    // Calculating the total number of elements in the dividend tensor
    size_t len_dividend = product(dividend->shape, dividend->rank);
    
    // If the tensor is empty, exit the function
    if(len_dividend == 0) return;
    
    nnl2_tensor_type dtype_dividend = dividend->dtype;
    nnl2_tensor_type dtype_divisor = divisor->dtype;
    
    if(dtype_dividend == dtype_divisor) {
        // Handling case when the tensors have the same type
        
        switch(dtype_dividend) {
            case FLOAT64: {
                volatile double* data_dividend = (double*)dividend->data;
                volatile double* data_divisor = (double*)divisor->data;
                
                // Element-wise division
                for(size_t i = 0; i < len_dividend; i++) data_dividend[i] /= data_divisor[i];
                break;
            }
            
            case FLOAT32: {
                volatile float* data_dividend = (float*)dividend->data;
                volatile float* data_divisor = (float*)divisor->data;
                
                // Element-wise division
                for(size_t i = 0; i < len_dividend; i++) data_dividend[i] /= data_divisor[i];    
                break;
            }
            
            case INT32: {
                volatile int32_t* data_dividend = (int32_t*)dividend->data;
                volatile int32_t* data_divisor = (int32_t*)divisor->data;
                
                // Element-wise division
                for(size_t i = 0; i < len_dividend; i++) data_dividend[i] /= data_divisor[i];        
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(dtype_dividend);
                return;
            }
        }
    } else {
        // Handling the case when tensors have different data types
        // Calculating the step size for accessing divisor tensor elements
        size_t divisor_step = get_dtype_size(dtype_divisor);
        
        // Casting divisor data to char* for byte access
        char* divisor_data = (char*)divisor->data;
        
        switch(dtype_dividend) {
            case FLOAT64: {
                volatile double* data_dividend = (double*)dividend->data;
                
                // For each element, convert the divisor element to float64 and divide by it
                for(size_t i = 0; i < len_dividend; i++) {
                    void* divisor_elem = divisor_data + i * divisor_step;
                    data_dividend[i] /= nnl2_convert_to_float64(divisor_elem, dtype_divisor);
                }
                
                break; 
            }
            
            case FLOAT32: {
                volatile float* data_dividend = (float*)dividend->data;
                
                // For each element, convert the divisor element to float32 and divide by it
                for(size_t i = 0; i < len_dividend; i++) {
                    void* divisor_elem = divisor_data + i * divisor_step;
                    data_dividend[i] /= nnl2_convert_to_float32(divisor_elem, dtype_divisor);
                }
                
                break; 
            }
            
            case INT32: {
                volatile int32_t* data_dividend = (int32_t*)dividend->data;
                
                // For each element, convert the divisor element to int32 and divide by it
                for(size_t i = 0; i < len_dividend; i++) {
                    void* divisor_elem = divisor_data + i * divisor_step;
                    data_dividend[i] /= nnl2_convert_to_int32(divisor_elem, dtype_divisor);
                }
                
                break; 
            }
            
            default: {
                NNL2_TYPE_ERROR(dtype_dividend);
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
 * division in-place operation
 */
#define NNL2_DIV_INPLACE_PARALLEL_THRESHOLD 1000000

/** @brief 
 * Worker function for parallel double precision division with same data types
 * 
 ** @param arg 
 * Pointer to divinplace_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 * 
 ** @details
 * Processes FLOAT64 data using AVX256 instructions with cache prefetching
 * for optimal memory access patterns
 */
void* nnl2_own_pdiv_float64_same_type(void* arg);

/** @brief
 * Worker function for parallel single precision division with same data types
 * 
 ** @param arg 
 * Pointer to divinplace_ptask structure containing thread parameters  
 * 
 ** @return 
 * NULL (for pthread API compatibility)
 * 
 ** @see nnl2_own_pdiv_float64_same_type
 **/
void* nnl2_own_pdiv_float32_same_type(void* arg);

/** @brief
 * Worker function for parallel integer division with same data types
 * 
 ** @param arg 
 * Pointer to divinplace_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 * 
 ** @see nnl2_own_pdiv_float64_same_type
 **/
void* nnl2_own_pdiv_int32_same_type(void* arg);

/** @brief
 * Worker function for parallel double precision division with type conversion
 * 
 ** @param arg 
 * Pointer to divinplace_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 * 
 ** @details
 * Handles type conversion from divisor to dividend data type with AVX256
 * optimizations and prefetching
 */
void* nnl2_own_pdiv_float64_diff_type(void* arg);

/** @brief
 * Worker function for parallel single precision division with type conversion
 * 
 ** @param arg 
 * Pointer to divinplace_ptask structure containing thread parameters
 *
 ** @return
 * NULL (for pthread API compatibility)
 * 
 ** @see nnl2_own_pdiv_float64_diff_type
 **/
void* nnl2_own_pdiv_float32_diff_type(void* arg);

/** @brief
 * Worker function for parallel integer division with type conversion
 * 
 ** @param arg
 * Pointer to divinplace_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 * 
 ** @see nnl2_own_pdiv_float64_diff_type
 **/
void* nnl2_own_pdiv_int32_diff_type(void* arg);

/** @brief
 * High-performance parallel implementation of in-place division
 * 
 ** @param dividend 
 * Pointer to tensor that will be modified in-place
 *
 ** @param divisor 
 * Pointer to tensor whose values will divide the dividend
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
 * Division by zero may result in undefined behavior
 */
void nnl2_own_divinplace(nnl2_tensor* dividend, const nnl2_tensor* divisor) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks at the maximum safety level
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(dividend, "Dividend tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(dividend->data, "Dividend tensor's data is NULL");
        
        NNL2_CHECK_NULL_IF_ERR_RETURN(divisor, "Divisor tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(divisor->data, "Divisor tensor's data is NULL");
    #endif
    
    // Calculating the total number of elements in the dividend tensor
    size_t len_dividend = product(dividend->shape, dividend->rank);
    
    // Fallback to naive implementation for small tensors
    if(len_dividend < NNL2_DIV_INPLACE_PARALLEL_THRESHOLD) {
        nnl2_naive_divinplace(dividend, divisor);
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return;
    }
    
    nnl2_tensor_type dtype_dividend = dividend->dtype;
    nnl2_tensor_type dtype_divisor = divisor->dtype;
    
    bool is_aligned_dividend = NNL2_IS_ALIGNED(dividend->data, NNL2_TENSOR_ALIGNMENT_32);
    bool is_aligned_divisor = NNL2_IS_ALIGNED(divisor->data, NNL2_TENSOR_ALIGNMENT_32);
    
    // Warning for unaligned memory in safety modes
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!is_aligned_dividend) {
            NNL2_WARN("In nnl2_own_2 div in-place, dividend memory is not aligned to 32 bytes. Performance may be suboptimal");
        }
        
        if(!is_aligned_divisor && dtype_dividend == dtype_divisor) {
            NNL2_WARN("In nnl2_own_2 div in-place, divisor memory is not aligned to 32 bytes. Performance may be suboptimal");
        }
    #endif
    
    size_t num_threads = NNL2_NUM_THREADS;
    pthread_t threads[num_threads];
    divinplace_ptask tasks[num_threads];
    
    // Calculate optimal chunk sizes with load balancing
    size_t chunk = len_dividend / num_threads;
    size_t remainder = len_dividend % num_threads;
    
    size_t current_start = 0;
    for (size_t i = 0; i < num_threads; i++) {
        size_t current_chunk = chunk + (i < remainder ? 1 : 0);
        
        // Configure task for this thread
        tasks[i].dividend_data = dividend->data;
        tasks[i].divisor_data = divisor->data;
        tasks[i].start = current_start;
        tasks[i].end = current_start + current_chunk;
        tasks[i].dtype_dividend = dtype_dividend;
        tasks[i].dtype_divisor = dtype_divisor;
        tasks[i].aligned_dividend = is_aligned_dividend;
        tasks[i].aligned_divisor = is_aligned_divisor;
        tasks[i].divisor_step = (dtype_dividend == dtype_divisor) ? get_dtype_size(dtype_divisor) : 0;
        
        // Select appropriate worker function based on data types
        void* (*worker_func)(void*) = NULL;
        
        if(dtype_dividend == dtype_divisor) {
            switch(dtype_dividend) {
                case FLOAT64: worker_func = nnl2_own_pdiv_float64_same_type; break;
                case FLOAT32: worker_func = nnl2_own_pdiv_float32_same_type; break;
                case INT32:   worker_func = nnl2_own_pdiv_int32_same_type;   break;
                default: {
                    NNL2_TYPE_ERROR(dtype_dividend);
                    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                        NNL2_FUNC_EXIT();
                    #endif
                    return;
                }
            }
        } else {
            tasks[i].divisor_step = get_dtype_size(dtype_divisor);
            switch(dtype_dividend) {
                case FLOAT64: worker_func = nnl2_own_pdiv_float64_diff_type; break;
                case FLOAT32: worker_func = nnl2_own_pdiv_float32_diff_type; break;
                case INT32:   worker_func = nnl2_own_pdiv_int32_diff_type;   break;
                default: {
                    NNL2_TYPE_ERROR(dtype_dividend);
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
            NNL2_THREAD_CREATE_ERROR(status, "nnl2_own_2_divinplace");
            num_threads = i;
            break;
        }
        
        current_start += current_chunk;
    }
    
    // Wait for all threads to complete
    for (size_t i = 0; i < num_threads; i++) {
        int join_status = pthread_join(threads[i], NULL);
        if(join_status != 0) {
            NNL2_THREAD_JOIN_ERROR(join_status, "nnl2_own_2_divinplace");
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
 ** @see nnl2_own_pdiv_float64_same_type
 **/
void* nnl2_own_pdiv_float64_same_type(void* arg) {
    divinplace_ptask* task = (divinplace_ptask*)arg;
    double* dividend = (double*)task->dividend_data;
    double* divisor = (double*)task->divisor_data;
    size_t start = task->start;
    size_t end = task->end;
    
    size_t i = start;
    
    // AVX256 processing with prefetching
    if(task->aligned_dividend && task->aligned_divisor) {
        for(; i + 3 < end; i += 4) {
            // Prefetch next cache lines
            _mm_prefetch((char*)&dividend[i + 16], _MM_HINT_T0);
            _mm_prefetch((char*)&divisor[i + 16], _MM_HINT_T0);
            
            __m256d v_dividend = _mm256_load_pd(&dividend[i]);
            __m256d v_divisor = _mm256_load_pd(&divisor[i]);
            __m256d v_result = _mm256_div_pd(v_dividend, v_divisor);
            _mm256_store_pd(&dividend[i], v_result);
        }
    } else if(task->aligned_dividend) {
        for(; i + 3 < end; i += 4) {
            _mm_prefetch((char*)&dividend[i + 16], _MM_HINT_T0);
            _mm_prefetch((char*)&divisor[i + 16], _MM_HINT_T0);
            
            __m256d v_dividend = _mm256_load_pd(&dividend[i]);
            __m256d v_divisor = _mm256_loadu_pd(&divisor[i]);
            __m256d v_result = _mm256_div_pd(v_dividend, v_divisor);
            _mm256_store_pd(&dividend[i], v_result);
        }
    } else if(task->aligned_divisor) {
        for(; i + 3 < end; i += 4) {
            _mm_prefetch((char*)&dividend[i + 16], _MM_HINT_T0);
            _mm_prefetch((char*)&divisor[i + 16], _MM_HINT_T0);
            
            __m256d v_dividend = _mm256_loadu_pd(&dividend[i]);
            __m256d v_divisor = _mm256_load_pd(&divisor[i]);
            __m256d v_result = _mm256_div_pd(v_dividend, v_divisor);
            _mm256_storeu_pd(&dividend[i], v_result);
        }
    } else {
        for(; i + 3 < end; i += 4) {
            _mm_prefetch((char*)&dividend[i + 16], _MM_HINT_T0);
            _mm_prefetch((char*)&divisor[i + 16], _MM_HINT_T0);
            
            __m256d v_dividend = _mm256_loadu_pd(&dividend[i]);
            __m256d v_divisor = _mm256_loadu_pd(&divisor[i]);
            __m256d v_result = _mm256_div_pd(v_dividend, v_divisor);
            _mm256_storeu_pd(&dividend[i], v_result);
        }
    }
    
    // Scalar processing for remainder
    for(; i < end; i++) {
        dividend[i] /= divisor[i];
    }
    
    return NULL;
}

/** @brief
 * See documentation at declaration
 * 
 ** @see nnl2_own_pdiv_float32_same_type
 **/
void* nnl2_own_pdiv_float32_same_type(void* arg) {
    divinplace_ptask* task = (divinplace_ptask*)arg;
    float* dividend = (float*)task->dividend_data;
    float* divisor = (float*)task->divisor_data;
    size_t start = task->start;
    size_t end = task->end;
    
    size_t i = start;
    
    // AVX256 processing with prefetching (8 elements per iteration)
    if(task->aligned_dividend && task->aligned_divisor) {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&dividend[i + 32], _MM_HINT_T0);
            _mm_prefetch((char*)&divisor[i + 32], _MM_HINT_T0);
            
            __m256 v_dividend = _mm256_load_ps(&dividend[i]);
            __m256 v_divisor = _mm256_load_ps(&divisor[i]);
            __m256 v_result = _mm256_div_ps(v_dividend, v_divisor);
            _mm256_store_ps(&dividend[i], v_result);
        }
    } else if(task->aligned_dividend) {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&dividend[i + 32], _MM_HINT_T0);
            _mm_prefetch((char*)&divisor[i + 32], _MM_HINT_T0);
            
            __m256 v_dividend = _mm256_load_ps(&dividend[i]);
            __m256 v_divisor = _mm256_loadu_ps(&divisor[i]);
            __m256 v_result = _mm256_div_ps(v_dividend, v_divisor);
            _mm256_store_ps(&dividend[i], v_result);
        }
    } else if(task->aligned_divisor) {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&dividend[i + 32], _MM_HINT_T0);
            _mm_prefetch((char*)&divisor[i + 32], _MM_HINT_T0);
            
            __m256 v_dividend = _mm256_loadu_ps(&dividend[i]);
            __m256 v_divisor = _mm256_load_ps(&divisor[i]);
            __m256 v_result = _mm256_div_ps(v_dividend, v_divisor);
            _mm256_storeu_ps(&dividend[i], v_result);
        }
    } else {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&dividend[i + 32], _MM_HINT_T0);
            _mm_prefetch((char*)&divisor[i + 32], _MM_HINT_T0);
            
            __m256 v_dividend = _mm256_loadu_ps(&dividend[i]);
            __m256 v_divisor = _mm256_loadu_ps(&divisor[i]);
            __m256 v_result = _mm256_div_ps(v_dividend, v_divisor);
            _mm256_storeu_ps(&dividend[i], v_result);
        }
    }
    
    // Scalar processing for remainder
    for(; i < end; i++) {
        dividend[i] /= divisor[i];
    }
    
    return NULL;
}

/** @brief
 * See documentation at declaration
 * 
 ** @see nnl2_own_pdiv_int32_same_type
 **/
void* nnl2_own_pdiv_int32_same_type(void* arg) {
    divinplace_ptask* task = (divinplace_ptask*)arg;
    int32_t* dividend = (int32_t*)task->dividend_data;
    int32_t* divisor = (int32_t*)task->divisor_data;
    size_t start = task->start;
    size_t end = task->end;
    
    size_t i = start;
    
    // AVX256 processing with prefetching (8 elements per iteration)
    // Note: Integer division uses scalar operations as AVX256 doesn't have integer division
    for(; i < end; i++) {
        dividend[i] /= divisor[i];
    }
    
    return NULL;
}

// Different type worker functions with conversion and prefetching

/** @brief
 * See documentation at declaration
 * 
 ** @see nnl2_own_pdiv_float64_diff_type
 **/
void* nnl2_own_pdiv_float64_diff_type(void* arg) {
    divinplace_ptask* task = (divinplace_ptask*)arg;
    double* dividend = (double*)task->dividend_data;
    char* divisor_data = (char*)task->divisor_data;
    size_t start = task->start;
    size_t end = task->end;
    size_t divisor_step = task->divisor_step;
    
    size_t i = start;
    
    // AVX256 processing with type conversion and prefetching
    if(task->aligned_dividend) {
        for(; i + 3 < end; i += 4) {
            _mm_prefetch((char*)&dividend[i + 16], _MM_HINT_T0);
            
            __m256d v_dividend = _mm256_load_pd(&dividend[i]);
            __m256d v_divisor = _mm256_set_pd(
                nnl2_convert_to_float64(divisor_data + (i + 3) * divisor_step, task->dtype_divisor),
                nnl2_convert_to_float64(divisor_data + (i + 2) * divisor_step, task->dtype_divisor),
                nnl2_convert_to_float64(divisor_data + (i + 1) * divisor_step, task->dtype_divisor),
                nnl2_convert_to_float64(divisor_data + (i + 0) * divisor_step, task->dtype_divisor)
            );
            
            __m256d v_result = _mm256_div_pd(v_dividend, v_divisor);
            _mm256_store_pd(&dividend[i], v_result);
        }
    } else {
        for(; i + 3 < end; i += 4) {
            _mm_prefetch((char*)&dividend[i + 16], _MM_HINT_T0);
            
            __m256d v_dividend = _mm256_loadu_pd(&dividend[i]);
            __m256d v_divisor = _mm256_set_pd(
                nnl2_convert_to_float64(divisor_data + (i + 3) * divisor_step, task->dtype_divisor),
                nnl2_convert_to_float64(divisor_data + (i + 2) * divisor_step, task->dtype_divisor),
                nnl2_convert_to_float64(divisor_data + (i + 1) * divisor_step, task->dtype_divisor),
                nnl2_convert_to_float64(divisor_data + (i + 0) * divisor_step, task->dtype_divisor)
            );
            
            __m256d v_result = _mm256_div_pd(v_dividend, v_divisor);
            _mm256_storeu_pd(&dividend[i], v_result);
        }
    }
    
    // Scalar processing for remainder
    for(; i < end; i++) {
        void* divisor_elem = divisor_data + i * divisor_step;
        dividend[i] /= nnl2_convert_to_float64(divisor_elem, task->dtype_divisor);
    }
    
    return NULL;
}

/** @brief
 * See documentation at declaration
 * 
 ** @see nnl2_own_pdiv_float32_diff_type
 **/
void* nnl2_own_pdiv_float32_diff_type(void* arg) {
    divinplace_ptask* task = (divinplace_ptask*)arg;
    float* dividend = (float*)task->dividend_data;
    char* divisor_data = (char*)task->divisor_data;
    size_t start = task->start;
    size_t end = task->end;
    size_t divisor_step = task->divisor_step;
    
    size_t i = start;
    
    // AVX256 processing with type conversion and prefetching
    if(task->aligned_dividend) {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&dividend[i + 32], _MM_HINT_T0);
            
            __m256 v_dividend = _mm256_load_ps(&dividend[i]);
            __m256 v_divisor = _mm256_set_ps(
                nnl2_convert_to_float32(divisor_data + (i + 7) * divisor_step, task->dtype_divisor),
                nnl2_convert_to_float32(divisor_data + (i + 6) * divisor_step, task->dtype_divisor),
                nnl2_convert_to_float32(divisor_data + (i + 5) * divisor_step, task->dtype_divisor),
                nnl2_convert_to_float32(divisor_data + (i + 4) * divisor_step, task->dtype_divisor),
                nnl2_convert_to_float32(divisor_data + (i + 3) * divisor_step, task->dtype_divisor),
                nnl2_convert_to_float32(divisor_data + (i + 2) * divisor_step, task->dtype_divisor),
                nnl2_convert_to_float32(divisor_data + (i + 1) * divisor_step, task->dtype_divisor),
                nnl2_convert_to_float32(divisor_data + (i + 0) * divisor_step, task->dtype_divisor)
            );
            
            __m256 v_result = _mm256_div_ps(v_dividend, v_divisor);
            _mm256_store_ps(&dividend[i], v_result);
        }
    } else {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&dividend[i + 32], _MM_HINT_T0);
            
            __m256 v_dividend = _mm256_loadu_ps(&dividend[i]);
            __m256 v_divisor = _mm256_set_ps(
                nnl2_convert_to_float32(divisor_data + (i + 7) * divisor_step, task->dtype_divisor),
                nnl2_convert_to_float32(divisor_data + (i + 6) * divisor_step, task->dtype_divisor),
                nnl2_convert_to_float32(divisor_data + (i + 5) * divisor_step, task->dtype_divisor),
                nnl2_convert_to_float32(divisor_data + (i + 4) * divisor_step, task->dtype_divisor),
                nnl2_convert_to_float32(divisor_data + (i + 3) * divisor_step, task->dtype_divisor),
                nnl2_convert_to_float32(divisor_data + (i + 2) * divisor_step, task->dtype_divisor),
                nnl2_convert_to_float32(divisor_data + (i + 1) * divisor_step, task->dtype_divisor),
                nnl2_convert_to_float32(divisor_data + (i + 0) * divisor_step, task->dtype_divisor)
            );
            
            __m256 v_result = _mm256_div_ps(v_dividend, v_divisor);
            _mm256_storeu_ps(&dividend[i], v_result);
        }
    }
    
    // Scalar processing for remainder
    for(; i < end; i++) {
        void* divisor_elem = divisor_data + i * divisor_step;
        dividend[i] /= nnl2_convert_to_float32(divisor_elem, task->dtype_divisor);
    }
    
    return NULL;
}

/** @brief
 * See documentation at declaration
 * 
 ** @see nnl2_own_pdiv_int32_diff_type
 **/
void* nnl2_own_pdiv_int32_diff_type(void* arg) {
    divinplace_ptask* task = (divinplace_ptask*)arg;
    int32_t* dividend = (int32_t*)task->dividend_data;
    char* divisor_data = (char*)task->divisor_data;
    size_t start = task->start;
    size_t end = task->end;
    size_t divisor_step = task->divisor_step;
    
    size_t i = start;
    
    // Integer division uses scalar operations
    for(; i < end; i++) {
        void* divisor_elem = divisor_data + i * divisor_step;
        dividend[i] /= nnl2_convert_to_int32(divisor_elem, task->dtype_divisor);
    }
    
    return NULL;
}

#endif

/** 
 * @ingroup backend_system
 * @brief Backend implementations for division operation
 * @details
 * Array follows the common backend registration pattern.
 * Currently registered backends:
 *  - nnl2_naive_divinplace: Basic reference implementation
 * 
 * @see nnl2_naive_divinplace
 */
nnl2_runtime_implementation divinplace_backends[] = {
	REGISTER_BACKEND(nnl2_naive_divinplace, nnl2_naive, NAIVE_BACKEND_NAME),
	
	#if defined(NNL2_PTHREAD_AVAILABLE) && NNL2_AVX256_AVAILABLE
		REGISTER_BACKEND(nnl2_own_divinplace, nnl2_own_2, NNL2_OWN_NAME),
	#endif
};

/**
 * @brief Function pointer for division operation
 * @ingroup backend_system 
 */
divinplacefn divinplace;

/** 
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(divinplace);

/** 
 * @brief Sets the backend for division operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void set_divinplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(divinplace_backends, divinplace, backend_name, CURRENT_BACKEND(divinplace));
}

/** 
 * @brief Gets the name of the active backend for division operation
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* get_divinplace_backend() {
	return CURRENT_BACKEND(divinplace);
}

/** 
 * @brief Function declaration for getting all `divinplace` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(divinplace);

/**
 * @brief Function declaration for getting the number of all `divinplace` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(divinplace);

#endif /** NNL2_DIV_INPLACE_H **/
