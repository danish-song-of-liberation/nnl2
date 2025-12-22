#ifndef NNL2_BROADCASTING_DIV_INPLACE_H
#define NNL2_BROADCASTING_DIV_INPLACE_H

/** @brief
 * Performs element-wise division with broadcasting (in place)
 *
 ** @param dividend
 * Pointer to dividend tensor (will be modified in place)
 *
 ** @param divisor
 * Pointer to divisor tensor
 */
void naive_div_broadcasting_inplace(nnl2_tensor* dividend, const nnl2_tensor* divisor) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks for max safety level 
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(dividend, "Dividend tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(divisor, "Divisor tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(dividend->shape, "Dividend shape is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(divisor->shape, "Divisor shape is NULL");
    #endif
    
    size_t numel_dividend = nnl2_product(dividend->shape, dividend->rank);
    size_t numel_divisor = nnl2_product(divisor->shape, divisor->rank);
    
    // Getting the tensor data types
    nnl2_tensor_type dividend_dtype = dividend->dtype;
    nnl2_tensor_type divisor_dtype = divisor->dtype;

    if((numel_dividend % numel_divisor) == 0) {
        // Handling the case where the data types match (more efficiently)
        if(dividend_dtype == divisor_dtype) {
            switch(dividend_dtype) {
                case FLOAT64: {
                    double* cast_dividend_data = (double*)dividend->data;
                    double* cast_divisor_data = (double*)divisor->data;

                    for(size_t i = 0; i < (numel_dividend / numel_divisor); i++) {
                        for(size_t j = 0; j < numel_divisor; j++) {
                            cast_dividend_data[i * numel_divisor + j] /= cast_divisor_data[j];
                        }
                    }

                    break;
                }

                case FLOAT32: {
                    float* cast_dividend_data = (float*)dividend->data;
                    float* cast_divisor_data = (float*)divisor->data;

                    for(size_t i = 0; i < (numel_dividend / numel_divisor); i++) {
                        for(size_t j = 0; j < numel_divisor; j++) {
                            cast_dividend_data[i * numel_divisor + j] /= cast_divisor_data[j];
                        }
                    }

                    break;
                }

                case INT32: {
                    int32_t* cast_dividend_data = (int32_t*)dividend->data;
                    int32_t* cast_divisor_data = (int32_t*)divisor->data;

                    for(size_t i = 0; i < (numel_dividend / numel_divisor); i++) {
                        for(size_t j = 0; j < numel_divisor; j++) {
                            cast_dividend_data[i * numel_divisor + j] /= cast_divisor_data[j];
                        }
                    }

                    break;
                }

                default: {
                    NNL2_TYPE_ERROR(dividend_dtype);
                    return;
                }
            }
        } else {
            // Handling a case with different data types (conversion required)
            size_t divisor_step = get_dtype_size(divisor_dtype);
            char* divisor_data = (char*)divisor->data;
            
            switch(dividend_dtype) {
                case FLOAT64: {
                    double* data_dividend = (double*)dividend->data;
                
                    for(size_t i = 0; i < (numel_dividend / numel_divisor); i++) {
                        for(size_t j = 0; j < numel_divisor; j++) {
                            void* divisor_elem = divisor_data + j * divisor_step;
                            data_dividend[i * numel_divisor + j] /= nnl2_convert_to_float64(divisor_elem, divisor_dtype);
                        }
                    }
                
                    break; 
                }
                
                case FLOAT32: {
                    float* data_dividend = (float*)dividend->data;
                
                    for(size_t i = 0; i < (numel_dividend / numel_divisor); i++) {
                        for(size_t j = 0; j < numel_divisor; j++) {
                            void* divisor_elem = divisor_data + j * divisor_step;
                            data_dividend[i * numel_divisor + j] /= nnl2_convert_to_float32(divisor_elem, divisor_dtype);
                        }
                    }
                
                    break; 
                }
                
                case INT32: {
                    int32_t* data_dividend = (int32_t*)dividend->data;
                
                    for(size_t i = 0; i < (numel_dividend / numel_divisor); i++) {
                        for(size_t j = 0; j < numel_divisor; j++) {
                            void* divisor_elem = divisor_data + j * divisor_step;
                            data_dividend[i * numel_divisor + j] /= nnl2_convert_to_int32(divisor_elem, divisor_dtype);
                        }
                    }
                
                    break; 
                }
                
                default: {
                    NNL2_TYPE_ERROR(dividend_dtype);
                    return;
                }
            }
        }
    } else {
        NNL2_ERROR("Cannot broadcast divisor tensor");
        return;
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

#if defined(NNL2_PTHREAD_AVAILABLE) && defined(NNL2_AVX256_AVAILABLE) && TENSOR_MEM_ALIGNMENT == 32

/** @brief
 * Threshold for enabling parallel execution of the
 * in-place broadcasting division operation
 */
#define NNL2_DIV_BROADCASTING_INPLACE_PARALLEL_THRESHOLD 500000

/** @brief 
 * Worker function for parallel double precision in-place broadcasting division
 * 
 ** @param arg 
 * Pointer to divbroadcasting_inplace_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 */
void* nnl2_own_pdiv_broadcasting_inplace_float64(void* arg);

/** @brief
 * Worker function for parallel single precision in-place broadcasting division
 * 
 ** @param arg 
 * Pointer to divbroadcasting_inplace_ptask structure containing thread parameters  
 * 
 ** @return 
 * NULL (for pthread API compatibility)
 */
void* nnl2_own_pdiv_broadcasting_inplace_float32(void* arg);

/** @brief
 * Worker function for parallel integer in-place broadcasting division
 * 
 ** @param arg 
 * Pointer to divbroadcasting_inplace_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 */
void* nnl2_own_pdiv_broadcasting_inplace_int32(void* arg);

/** @brief
 * High-performance parallel implementation of in-place broadcasting division
 * 
 ** @param dividend
 * Pointer to dividend tensor (will be modified in-place)
 *
 ** @param divisor
 * Pointer to divisor tensor (broadcasted)
 */
void nnl2_own_div_broadcasting_inplace(nnl2_tensor* dividend, const nnl2_tensor* divisor) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(dividend, "Dividend tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(divisor, "Divisor tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(dividend->shape, "Dividend shape is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(divisor->shape, "Divisor shape is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(dividend->data, "Dividend data is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(divisor->data, "Divisor data is NULL");
    #endif
    
    size_t numel_dividend = nnl2_product(dividend->shape, dividend->rank);
    size_t numel_divisor = nnl2_product(divisor->shape, divisor->rank);
    
    // Check broadcasting compatibility
    if((numel_dividend % numel_divisor) != 0) {
        NNL2_ERROR("Cannot broadcast divisor tensor");
        return;
    }
    
    if(numel_dividend == 0 || numel_divisor == 0) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return;
    }
    
    // Fall back to naive implementation for small tensors or different dtypes
    if(numel_dividend < NNL2_DIV_BROADCASTING_INPLACE_PARALLEL_THRESHOLD || 
       dividend->dtype != divisor->dtype) {
        naive_div_broadcasting_inplace(dividend, divisor);
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return;
    }
    
    nnl2_tensor_type dtype = dividend->dtype;
    size_t broadcast_ratio = numel_dividend / numel_divisor;
    
    bool is_aligned_dividend = NNL2_IS_ALIGNED(dividend->data, NNL2_TENSOR_ALIGNMENT_32);
    bool is_aligned_divisor = NNL2_IS_ALIGNED(divisor->data, NNL2_TENSOR_ALIGNMENT_32);
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!is_aligned_dividend) {
            NNL2_WARN("In nnl2_own div broadcasting in-place, dividend memory is not aligned to 32 bytes. Performance may be suboptimal");
        }
        
        if(!is_aligned_divisor) {
            NNL2_WARN("In nnl2_own div broadcasting in-place, divisor memory is not aligned to 32 bytes. Performance may be suboptimal");
        }
    #endif
    
    size_t num_threads = NNL2_NUM_THREADS;
    pthread_t threads[num_threads];
    divbroadcasting_inplace_ptask tasks[num_threads];
    
    // Calculate chunk size per thread for the outer loop
    size_t chunk = broadcast_ratio / num_threads;
    size_t remainder = broadcast_ratio % num_threads;
    
    // Initialize common task parameters
    for(size_t i = 0; i < num_threads; i++) {
        tasks[i].dtype = dtype;
        tasks[i].aligned_dividend = is_aligned_dividend;
        tasks[i].aligned_divisor = is_aligned_divisor;
        tasks[i].dividend_data = dividend->data;
        tasks[i].divisor_data = divisor->data;
        tasks[i].numel_divisor = numel_divisor;
        tasks[i].broadcast_ratio = broadcast_ratio;
    }
    
    size_t current_start = 0;
    for(size_t i = 0; i < num_threads; i++) {
        size_t current_chunk = chunk + (i < remainder ? 1 : 0);
        
        tasks[i].start = current_start;
        tasks[i].end = current_start + current_chunk;
        
        void* (*worker_func)(void*) = NULL;
        switch(dtype) {
            case FLOAT64: worker_func = nnl2_own_pdiv_broadcasting_inplace_float64; break;
            case FLOAT32: worker_func = nnl2_own_pdiv_broadcasting_inplace_float32; break;
            case INT32:   worker_func = nnl2_own_pdiv_broadcasting_inplace_int32;   break;
            
            default: {
                NNL2_TYPE_ERROR(dtype);
                return;
            }
        }
        
        int status = pthread_create(&threads[i], NULL, worker_func, &tasks[i]);
        if(status != 0) {
            NNL2_THREAD_CREATE_ERROR(status, "nnl2_own_div_broadcasting_inplace");
            num_threads = i;
            break;
        }
        
        current_start += current_chunk;
    }
    
    for(size_t i = 0; i < num_threads; i++) {
        int join_status = pthread_join(threads[i], NULL);
        if(join_status != 0) {
            NNL2_THREAD_JOIN_ERROR(join_status, "nnl2_own_div_broadcasting_inplace");
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

void* nnl2_own_pdiv_broadcasting_inplace_float64(void* arg) {
    divbroadcasting_inplace_ptask* task = (divbroadcasting_inplace_ptask*)arg;
    double* dividend_data = (double*)task->dividend_data;
    const double* divisor_data = (const double*)task->divisor_data;
    size_t start = task->start;
    size_t end = task->end;
    size_t numel_divisor = task->numel_divisor;
    
    for(size_t block = start; block < end; block++) {
        size_t base_idx = block * numel_divisor;
        
        __m256d v_divisor, v_dividend, v_result;
        size_t j = 0;
        
        if(task->aligned_dividend && task->aligned_divisor) {
            for(; j + 3 < numel_divisor; j += 4) {
                _mm_prefetch((char*)&divisor_data[j + 16], _MM_HINT_T0);
                _mm_prefetch((char*)&dividend_data[base_idx + j + 16], _MM_HINT_T0);
                
                v_divisor = _mm256_load_pd(&divisor_data[j]);
                v_dividend = _mm256_load_pd(&dividend_data[base_idx + j]);
                v_result = _mm256_div_pd(v_dividend, v_divisor);
                _mm256_store_pd(&dividend_data[base_idx + j], v_result);
            }
        } else if(task->aligned_dividend) {
            for(; j + 3 < numel_divisor; j += 4) {
                _mm_prefetch((char*)&divisor_data[j + 16], _MM_HINT_T0);
                _mm_prefetch((char*)&dividend_data[base_idx + j + 16], _MM_HINT_T0);
                
                v_divisor = _mm256_loadu_pd(&divisor_data[j]);
                v_dividend = _mm256_load_pd(&dividend_data[base_idx + j]);
                v_result = _mm256_div_pd(v_dividend, v_divisor);
                _mm256_store_pd(&dividend_data[base_idx + j], v_result);
            }
        } else if(task->aligned_divisor) {
            for(; j + 3 < numel_divisor; j += 4) {
                _mm_prefetch((char*)&divisor_data[j + 16], _MM_HINT_T0);
                _mm_prefetch((char*)&dividend_data[base_idx + j + 16], _MM_HINT_T0);
                
                v_divisor = _mm256_load_pd(&divisor_data[j]);
                v_dividend = _mm256_loadu_pd(&dividend_data[base_idx + j]);
                v_result = _mm256_div_pd(v_dividend, v_divisor);
                _mm256_storeu_pd(&dividend_data[base_idx + j], v_result);
            }
        } else {
            for(; j + 3 < numel_divisor; j += 4) {
                _mm_prefetch((char*)&divisor_data[j + 16], _MM_HINT_T0);
                _mm_prefetch((char*)&dividend_data[base_idx + j + 16], _MM_HINT_T0);
                
                v_divisor = _mm256_loadu_pd(&divisor_data[j]);
                v_dividend = _mm256_loadu_pd(&dividend_data[base_idx + j]);
                v_result = _mm256_div_pd(v_dividend, v_divisor);
                _mm256_storeu_pd(&dividend_data[base_idx + j], v_result);
            }
        }
        
        // Handle remaining elements
        for(; j < numel_divisor; j++) {
            dividend_data[base_idx + j] /= divisor_data[j];
        }
    }
    
    return NULL;
}

void* nnl2_own_pdiv_broadcasting_inplace_float32(void* arg) {
    divbroadcasting_inplace_ptask* task = (divbroadcasting_inplace_ptask*)arg;
    float* dividend_data = (float*)task->dividend_data;
    const float* divisor_data = (const float*)task->divisor_data;
    size_t start = task->start;
    size_t end = task->end;
    size_t numel_divisor = task->numel_divisor;
    
    for(size_t block = start; block < end; block++) {
        size_t base_idx = block * numel_divisor;
        
        __m256 v_divisor, v_dividend, v_result;
        size_t j = 0;
        
        if(task->aligned_dividend && task->aligned_divisor) {
            for(; j + 7 < numel_divisor; j += 8) {
                _mm_prefetch((char*)&divisor_data[j + 32], _MM_HINT_T0);
                _mm_prefetch((char*)&dividend_data[base_idx + j + 32], _MM_HINT_T0);
                
                v_divisor = _mm256_load_ps(&divisor_data[j]);
                v_dividend = _mm256_load_ps(&dividend_data[base_idx + j]);
                v_result = _mm256_div_ps(v_dividend, v_divisor);
                _mm256_store_ps(&dividend_data[base_idx + j], v_result);
            }
        } else if(task->aligned_dividend) {
            for(; j + 7 < numel_divisor; j += 8) {
                _mm_prefetch((char*)&divisor_data[j + 32], _MM_HINT_T0);
                _mm_prefetch((char*)&dividend_data[base_idx + j + 32], _MM_HINT_T0);
                
                v_divisor = _mm256_loadu_ps(&divisor_data[j]);
                v_dividend = _mm256_load_ps(&dividend_data[base_idx + j]);
                v_result = _mm256_div_ps(v_dividend, v_divisor);
                _mm256_store_ps(&dividend_data[base_idx + j], v_result);
            }
        } else if(task->aligned_divisor) {
            for(; j + 7 < numel_divisor; j += 8) {
                _mm_prefetch((char*)&divisor_data[j + 32], _MM_HINT_T0);
                _mm_prefetch((char*)&dividend_data[base_idx + j + 32], _MM_HINT_T0);
                
                v_divisor = _mm256_load_ps(&divisor_data[j]);
                v_dividend = _mm256_loadu_ps(&dividend_data[base_idx + j]);
                v_result = _mm256_div_ps(v_dividend, v_divisor);
                _mm256_storeu_ps(&dividend_data[base_idx + j], v_result);
            }
        } else {
            for(; j + 7 < numel_divisor; j += 8) {
                _mm_prefetch((char*)&divisor_data[j + 32], _MM_HINT_T0);
                _mm_prefetch((char*)&dividend_data[base_idx + j + 32], _MM_HINT_T0);
                
                v_divisor = _mm256_loadu_ps(&divisor_data[j]);
                v_dividend = _mm256_loadu_ps(&dividend_data[base_idx + j]);
                v_result = _mm256_div_ps(v_dividend, v_divisor);
                _mm256_storeu_ps(&dividend_data[base_idx + j], v_result);
            }
        }
        
        // Handle remaining elements
        for(; j < numel_divisor; j++) {
            dividend_data[base_idx + j] /= divisor_data[j];
        }
    }
    
    return NULL;
}

void* nnl2_own_pdiv_broadcasting_inplace_int32(void* arg) {
    divbroadcasting_inplace_ptask* task = (divbroadcasting_inplace_ptask*)arg;
    int32_t* dividend_data = (int32_t*)task->dividend_data;
    const int32_t* divisor_data = (const int32_t*)task->divisor_data;
    size_t start = task->start;
    size_t end = task->end;
    size_t numel_divisor = task->numel_divisor;
    
    // Integer division uses scalar operations for precise arithmetic semantics
    for(size_t block = start; block < end; block++) {
        size_t base_idx = block * numel_divisor;
        
        for(size_t j = 0; j < numel_divisor; j++) {
            dividend_data[base_idx + j] /= divisor_data[j];
        }
    }
    
    return NULL;
}

#endif

/**
 * @ingroup backend_system
 * @brief Backend implementations for division with broadcasting (in place)
 * @details
 * Array follows the common backend registration pattern for in-place division
 * with broadcasting operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for in-place division with broadcasting
 *  - nnl2_own: High-performance implementation with AVX256, pthread and prefetching
 * 
 * @see nnl2_naive
 * @see naive_div_broadcasting_inplace
 * @see nnl2_own_div_broadcasting_inplace
 */
nnl2_runtime_implementation div_broadcasting_inplace_backends[] = {
    REGISTER_BACKEND(naive_div_broadcasting_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
    
    #if defined(NNL2_PTHREAD_AVAILABLE) && defined(NNL2_AVX256_AVAILABLE) && TENSOR_MEM_ALIGNMENT == 32
        REGISTER_BACKEND(nnl2_own_div_broadcasting_inplace, nnl2_own, NNL2_OWN_NAME),
    #endif
};  

/**
 * @brief Function pointer for division with broadcasting operation (in place)
 * @ingroup backend_system
 */
divbroadcastinginplacefn div_broadcasting_inplace;

/**
 * @brief Sets the backend for division with broadcasting operation (in place)
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for division with broadcasting
 */
void set_div_broadcasting_inplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(div_broadcasting_inplace_backends, div_broadcasting_inplace, backend_name);
}

#endif /** NNL2_BROADCASTING_DIV_INPLACE_H **/
