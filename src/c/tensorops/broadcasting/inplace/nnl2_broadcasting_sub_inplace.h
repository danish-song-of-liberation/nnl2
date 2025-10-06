#ifndef NNL2_BROADCASTING_SUB_INPLACE_H
#define NNL2_BROADCASTING_SUB_INPLACE_H

/** @brief
 * Performs element-wise subtraction with broadcasting (in place)
 *
 ** @details
 * Subtracts subtrahend tensor from minuend tensor with broadcasting support
 *
 ** @param minuend
 * Pointer to minuend tensor (will be modified in place)
 *
 ** @param subtrahend
 * Pointer to subtrahend tensor
 */
void naive_sub_broadcasting_inplace(Tensor* minuend, const Tensor* subtrahend) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks for max safety level 
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX     
        NNL2_CHECK_NULL_IF_ERR_RETURN(minuend, "Minuend tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(subtrahend, "Subtrahend tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(minuend->shape, "Minuend shape is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(subtrahend->shape, "Subtrahend shape is NULL");
    #endif
    
    // Calculate the total number of elements in each tensor
    size_t numel_minuend = product(minuend->shape, minuend->rank);
    size_t numel_subtrahend = product(subtrahend->shape, subtrahend->rank);
    
    // Getting the tensor data types
    TensorType minuend_dtype = minuend->dtype;
    TensorType subtrahend_dtype = subtrahend->dtype;
    
    // Checking the possibility of broadcasting (numel_minuend must be a multiple of numel_subtrahend)
    if((numel_minuend % numel_subtrahend) == 0) {
        // Handling the case where the data types match (more efficiently)
        if(minuend_dtype == subtrahend_dtype) {
            switch(minuend_dtype) {
                case FLOAT64: {
                    double* cast_minuend_data = (double*)minuend->data;
                    double* cast_subtrahend_data = (double*)subtrahend->data;
                    
                    for(size_t i = 0; i < (numel_minuend / numel_subtrahend); i++) {
                        for(size_t j = 0; j < numel_subtrahend; j++) {
                            cast_minuend_data[i * numel_subtrahend + j] -= cast_subtrahend_data[j];
                        }
                    }
                    
                    break;
                }
                
                case FLOAT32: {
                    float* cast_minuend_data = (float*)minuend->data;
                    float* cast_subtrahend_data = (float*)subtrahend->data;
                    
                    for(size_t i = 0; i < (numel_minuend / numel_subtrahend); i++) {
                        for(size_t j = 0; j < numel_subtrahend; j++) {
                            cast_minuend_data[i * numel_subtrahend + j] -= cast_subtrahend_data[j];
                        }
                    }
                    
                    break;
                }
                
                case INT32: {
                    int32_t* cast_minuend_data = (int32_t*)minuend->data;
                    int32_t* cast_subtrahend_data = (int32_t*)subtrahend->data;
                    
                    for(size_t i = 0; i < (numel_minuend / numel_subtrahend); i++) {
                        for(size_t j = 0; j < numel_subtrahend; j++) {
                            cast_minuend_data[i * numel_subtrahend + j] -= cast_subtrahend_data[j];
                        }
                    }
                    
                    break;
                }
                
                default: {
                    NNL2_TYPE_ERROR(minuend_dtype);
                    return;
                }
            }    
        } else {
            // Handling a case with different data types (conversion required)
            size_t subtrahend_step = get_dtype_size(subtrahend_dtype); // The size of the element in bytes
            char* subtrahend_data = (char*)subtrahend->data; // Byte pointer for accessing data
            
            switch(minuend_dtype) {
                case FLOAT64: {
                    double* data_minuend = (double*)minuend->data;
                
                    for(size_t i = 0; i < (numel_minuend / numel_subtrahend); i++) {
                        for(size_t j = 0; j < numel_subtrahend; j++) {
                            // Get a pointer to the subtrahend element and convert its type
                            void* subtrahend_elem = subtrahend_data + j * subtrahend_step;
                            data_minuend[i * numel_subtrahend + j] -= nnl2_convert_to_float64(subtrahend_elem, subtrahend_dtype);
                        }
                    }
                
                    break; 
                }
                
                case FLOAT32: {
                    float* data_minuend = (float*)minuend->data;
                
                    for(size_t i = 0; i < (numel_minuend / numel_subtrahend); i++) {
                        for(size_t j = 0; j < numel_subtrahend; j++) {
                            void* subtrahend_elem = subtrahend_data + j * subtrahend_step;
                            data_minuend[i * numel_subtrahend + j] -= nnl2_convert_to_float32(subtrahend_elem, subtrahend_dtype);
                        }
                    }
                
                    break; 
                }
                
                case INT32: {
                    int32_t* data_minuend = (int32_t*)minuend->data;
                
                    for(size_t i = 0; i < (numel_minuend / numel_subtrahend); i++) {
                        for(size_t j = 0; j < numel_subtrahend; j++) {
                            void* subtrahend_elem = subtrahend_data + j * subtrahend_step;
                            data_minuend[i * numel_subtrahend + j] -= nnl2_convert_to_int32(subtrahend_elem, subtrahend_dtype);
                        }
                    }
                
                    break; 
                }
                
                default: {
                    NNL2_TYPE_ERROR(minuend_dtype);
                    return;
                }
            }
        }
    } 
    
    else {
        NNL2_ERROR("Cannot broadcast subtrahend tensor");
        return;
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

#if defined(NNL2_PTHREAD_AVAILABLE) && defined(NNL2_AVX256_AVAILABLE) && TENSOR_MEM_ALIGNMENT == 32

/** @brief
 * Threshold for enabling parallel execution of the
 * in-place broadcasting subtraction operation
 */
#define NNL2_SUB_BROADCASTING_INPLACE_PARALLEL_THRESHOLD 500000

/** @brief 
 * Worker function for parallel double precision in-place broadcasting subtraction
 * 
 ** @param arg 
 * Pointer to subbroadcasting_inplace_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 */
void* nnl2_own_psub_broadcasting_inplace_float64(void* arg);

/** @brief
 * Worker function for parallel single precision in-place broadcasting subtraction
 * 
 ** @param arg 
 * Pointer to subbroadcasting_inplace_ptask structure containing thread parameters  
 * 
 ** @return 
 * NULL (for pthread API compatibility)
 */
void* nnl2_own_psub_broadcasting_inplace_float32(void* arg);

/** @brief
 * Worker function for parallel integer in-place broadcasting subtraction
 * 
 ** @param arg 
 * Pointer to subbroadcasting_inplace_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 */
void* nnl2_own_psub_broadcasting_inplace_int32(void* arg);

/** @brief
 * High-performance parallel implementation of in-place broadcasting subtraction
 * 
 ** @param minuend
 * Pointer to minuend tensor (will be modified in-place)
 *
 ** @param subtrahend
 * Pointer to subtrahend tensor (broadcasted)
 */
void nnl2_own_sub_broadcasting_inplace(Tensor* minuend, const Tensor* subtrahend) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(minuend, "Minuend tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(subtrahend, "Subtrahend tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(minuend->shape, "Minuend shape is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(subtrahend->shape, "Subtrahend shape is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(minuend->data, "Minuend data is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(subtrahend->data, "Subtrahend data is NULL");
    #endif
    
    size_t numel_minuend = product(minuend->shape, minuend->rank);
    size_t numel_subtrahend = product(subtrahend->shape, subtrahend->rank);
    
    // Check broadcasting compatibility
    if((numel_minuend % numel_subtrahend) != 0) {
        NNL2_ERROR("Cannot broadcast subtrahend tensor");
        return;
    }
    
    if(numel_minuend == 0 || numel_subtrahend == 0) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return;
    }
    
    // Fall back to naive implementation for small tensors or different dtypes
    if(numel_minuend < NNL2_SUB_BROADCASTING_INPLACE_PARALLEL_THRESHOLD || 
       minuend->dtype != subtrahend->dtype) {
        naive_sub_broadcasting_inplace(minuend, subtrahend);
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return;
    }
    
    TensorType dtype = minuend->dtype;
    size_t broadcast_ratio = numel_minuend / numel_subtrahend;
    
    bool is_aligned_minuend = NNL2_IS_ALIGNED(minuend->data, NNL2_TENSOR_ALIGNMENT_32);
    bool is_aligned_subtrahend = NNL2_IS_ALIGNED(subtrahend->data, NNL2_TENSOR_ALIGNMENT_32);
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!is_aligned_minuend) {
            NNL2_WARN("In nnl2_own sub broadcasting in-place, minuend memory is not aligned to 32 bytes. Performance may be suboptimal");
        }
        
        if(!is_aligned_subtrahend) {
            NNL2_WARN("In nnl2_own sub broadcasting in-place, subtrahend memory is not aligned to 32 bytes. Performance may be suboptimal");
        }
    #endif
    
    size_t num_threads = NNL2_NUM_THREADS;
    pthread_t threads[num_threads];
    subbroadcasting_inplace_ptask tasks[num_threads];
    
    // Calculate chunk size per thread for the outer loop
    size_t chunk = broadcast_ratio / num_threads;
    size_t remainder = broadcast_ratio % num_threads;
    
    // Initialize common task parameters
    for(size_t i = 0; i < num_threads; i++) {
        tasks[i].dtype = dtype;
        tasks[i].aligned_minuend = is_aligned_minuend;
        tasks[i].aligned_subtrahend = is_aligned_subtrahend;
        tasks[i].minuend_data = minuend->data;
        tasks[i].subtrahend_data = subtrahend->data;
        tasks[i].numel_subtrahend = numel_subtrahend;
        tasks[i].broadcast_ratio = broadcast_ratio;
    }
    
    size_t current_start = 0;
    for(size_t i = 0; i < num_threads; i++) {
        size_t current_chunk = chunk + (i < remainder ? 1 : 0);
        
        tasks[i].start = current_start;
        tasks[i].end = current_start + current_chunk;
        
        void* (*worker_func)(void*) = NULL;
        switch(dtype) {
            case FLOAT64: worker_func = nnl2_own_psub_broadcasting_inplace_float64; break;
            case FLOAT32: worker_func = nnl2_own_psub_broadcasting_inplace_float32; break;
            case INT32:   worker_func = nnl2_own_psub_broadcasting_inplace_int32;   break;
            
            default: {
                NNL2_TYPE_ERROR(dtype);
                return;
            }
        }
        
        int status = pthread_create(&threads[i], NULL, worker_func, &tasks[i]);
        if(status != 0) {
            NNL2_THREAD_CREATE_ERROR(status, "nnl2_own_sub_broadcasting_inplace");
            num_threads = i;
            break;
        }
        
        current_start += current_chunk;
    }
    
    for(size_t i = 0; i < num_threads; i++) {
        int join_status = pthread_join(threads[i], NULL);
        if(join_status != 0) {
            NNL2_THREAD_JOIN_ERROR(join_status, "nnl2_own_sub_broadcasting_inplace");
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

void* nnl2_own_psub_broadcasting_inplace_float64(void* arg) {
    subbroadcasting_inplace_ptask* task = (subbroadcasting_inplace_ptask*)arg;
    double* minuend_data = (double*)task->minuend_data;
    const double* subtrahend_data = (const double*)task->subtrahend_data;
    size_t start = task->start;
    size_t end = task->end;
    size_t numel_subtrahend = task->numel_subtrahend;
    
    for(size_t block = start; block < end; block++) {
        size_t base_idx = block * numel_subtrahend;
        
        __m256d v_subtrahend, v_minuend, v_result;
        size_t j = 0;
        
        if(task->aligned_minuend && task->aligned_subtrahend) {
            for(; j + 3 < numel_subtrahend; j += 4) {
                _mm_prefetch((char*)&subtrahend_data[j + 16], _MM_HINT_T0);
                _mm_prefetch((char*)&minuend_data[base_idx + j + 16], _MM_HINT_T0);
                
                v_subtrahend = _mm256_load_pd(&subtrahend_data[j]);
                v_minuend = _mm256_load_pd(&minuend_data[base_idx + j]);
                v_result = _mm256_sub_pd(v_minuend, v_subtrahend);
                _mm256_store_pd(&minuend_data[base_idx + j], v_result);
            }
        } else if(task->aligned_minuend) {
            for(; j + 3 < numel_subtrahend; j += 4) {
                _mm_prefetch((char*)&subtrahend_data[j + 16], _MM_HINT_T0);
                _mm_prefetch((char*)&minuend_data[base_idx + j + 16], _MM_HINT_T0);
                
                v_subtrahend = _mm256_loadu_pd(&subtrahend_data[j]);
                v_minuend = _mm256_load_pd(&minuend_data[base_idx + j]);
                v_result = _mm256_sub_pd(v_minuend, v_subtrahend);
                _mm256_store_pd(&minuend_data[base_idx + j], v_result);
            }
        } else if(task->aligned_subtrahend) {
            for(; j + 3 < numel_subtrahend; j += 4) {
                _mm_prefetch((char*)&subtrahend_data[j + 16], _MM_HINT_T0);
                _mm_prefetch((char*)&minuend_data[base_idx + j + 16], _MM_HINT_T0);
                
                v_subtrahend = _mm256_load_pd(&subtrahend_data[j]);
                v_minuend = _mm256_loadu_pd(&minuend_data[base_idx + j]);
                v_result = _mm256_sub_pd(v_minuend, v_subtrahend);
                _mm256_storeu_pd(&minuend_data[base_idx + j], v_result);
            }
        } else {
            for(; j + 3 < numel_subtrahend; j += 4) {
                _mm_prefetch((char*)&subtrahend_data[j + 16], _MM_HINT_T0);
                _mm_prefetch((char*)&minuend_data[base_idx + j + 16], _MM_HINT_T0);
                
                v_subtrahend = _mm256_loadu_pd(&subtrahend_data[j]);
                v_minuend = _mm256_loadu_pd(&minuend_data[base_idx + j]);
                v_result = _mm256_sub_pd(v_minuend, v_subtrahend);
                _mm256_storeu_pd(&minuend_data[base_idx + j], v_result);
            }
        }
        
        // Handle remaining elements
        for(; j < numel_subtrahend; j++) {
            minuend_data[base_idx + j] -= subtrahend_data[j];
        }
    }
    
    return NULL;
}

void* nnl2_own_psub_broadcasting_inplace_float32(void* arg) {
    subbroadcasting_inplace_ptask* task = (subbroadcasting_inplace_ptask*)arg;
    float* minuend_data = (float*)task->minuend_data;
    const float* subtrahend_data = (const float*)task->subtrahend_data;
    size_t start = task->start;
    size_t end = task->end;
    size_t numel_subtrahend = task->numel_subtrahend;
    
    for(size_t block = start; block < end; block++) {
        size_t base_idx = block * numel_subtrahend;
        
        __m256 v_subtrahend, v_minuend, v_result;
        size_t j = 0;
        
        if(task->aligned_minuend && task->aligned_subtrahend) {
            for(; j + 7 < numel_subtrahend; j += 8) {
                _mm_prefetch((char*)&subtrahend_data[j + 32], _MM_HINT_T0);
                _mm_prefetch((char*)&minuend_data[base_idx + j + 32], _MM_HINT_T0);
                
                v_subtrahend = _mm256_load_ps(&subtrahend_data[j]);
                v_minuend = _mm256_load_ps(&minuend_data[base_idx + j]);
                v_result = _mm256_sub_ps(v_minuend, v_subtrahend);
                _mm256_store_ps(&minuend_data[base_idx + j], v_result);
            }
        } else if(task->aligned_minuend) {
            for(; j + 7 < numel_subtrahend; j += 8) {
                _mm_prefetch((char*)&subtrahend_data[j + 32], _MM_HINT_T0);
                _mm_prefetch((char*)&minuend_data[base_idx + j + 32], _MM_HINT_T0);
                
                v_subtrahend = _mm256_loadu_ps(&subtrahend_data[j]);
                v_minuend = _mm256_load_ps(&minuend_data[base_idx + j]);
                v_result = _mm256_sub_ps(v_minuend, v_subtrahend);
                _mm256_store_ps(&minuend_data[base_idx + j], v_result);
            }
        } else if(task->aligned_subtrahend) {
            for(; j + 7 < numel_subtrahend; j += 8) {
                _mm_prefetch((char*)&subtrahend_data[j + 32], _MM_HINT_T0);
                _mm_prefetch((char*)&minuend_data[base_idx + j + 32], _MM_HINT_T0);
                
                v_subtrahend = _mm256_load_ps(&subtrahend_data[j]);
                v_minuend = _mm256_loadu_ps(&minuend_data[base_idx + j]);
                v_result = _mm256_sub_ps(v_minuend, v_subtrahend);
                _mm256_storeu_ps(&minuend_data[base_idx + j], v_result);
            }
        } else {
            for(; j + 7 < numel_subtrahend; j += 8) {
                _mm_prefetch((char*)&subtrahend_data[j + 32], _MM_HINT_T0);
                _mm_prefetch((char*)&minuend_data[base_idx + j + 32], _MM_HINT_T0);
                
                v_subtrahend = _mm256_loadu_ps(&subtrahend_data[j]);
                v_minuend = _mm256_loadu_ps(&minuend_data[base_idx + j]);
                v_result = _mm256_sub_ps(v_minuend, v_subtrahend);
                _mm256_storeu_ps(&minuend_data[base_idx + j], v_result);
            }
        }
        
        // Handle remaining elements
        for(; j < numel_subtrahend; j++) {
            minuend_data[base_idx + j] -= subtrahend_data[j];
        }
    }
    
    return NULL;
}

void* nnl2_own_psub_broadcasting_inplace_int32(void* arg) {
    subbroadcasting_inplace_ptask* task = (subbroadcasting_inplace_ptask*)arg;
    int32_t* minuend_data = (int32_t*)task->minuend_data;
    const int32_t* subtrahend_data = (const int32_t*)task->subtrahend_data;
    size_t start = task->start;
    size_t end = task->end;
    size_t numel_subtrahend = task->numel_subtrahend;
    
    // Integer subtraction uses scalar operations for precise arithmetic semantics
    for(size_t block = start; block < end; block++) {
        size_t base_idx = block * numel_subtrahend;
        
        for(size_t j = 0; j < numel_subtrahend; j++) {
            minuend_data[base_idx + j] -= subtrahend_data[j];
        }
    }
    
    return NULL;
}

#endif

/**
 * @ingroup backend_system
 * @brief Backend implementations for subtraction with broadcasting (in place)
 * @details
 * Array follows the common backend registration pattern for subtraction
 * with broadcasting operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for subtraction with broadcasting
 *  - nnl2_own: High-performance implementation with AVX256, pthread and prefetching
 * 
 * @see nnl2_naive
 * @see naive_sub_broadcasting_inplace
 * @see nnl2_own_sub_broadcasting_inplace
 */
Implementation sub_broadcasting_inplace_backends[] = {
    REGISTER_BACKEND(naive_sub_broadcasting_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
    
    #if defined(NNL2_PTHREAD_AVAILABLE) && defined(NNL2_AVX256_AVAILABLE) && TENSOR_MEM_ALIGNMENT == 32
        REGISTER_BACKEND(nnl2_own_sub_broadcasting_inplace, nnl2_own, NNL2_OWN_NAME),
    #endif
};

/**
 * @brief Function pointer for subtraction with broadcasting operation (in place)
 * @ingroup backend_system
 */
subbroadcastinginplacefn sub_broadcasting_inplace;

/**
 * @brief Sets the backend for subtraction with broadcasting operation (in place)
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for subtraction with broadcasting
 * @see SET_BACKEND_BY_NAME
 * @see sub_broadcasting_inplace_backends
 */
void set_sub_broadcasting_inplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(sub_broadcasting_inplace_backends, sub_broadcasting_inplace, backend_name);
}

#endif /** NNL2_BROADCASTING_SUB_INPLACE_H **/
