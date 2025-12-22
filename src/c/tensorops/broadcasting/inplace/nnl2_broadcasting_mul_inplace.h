#ifndef NNL2_BROADCASTING_MUL_INPLACE_H
#define NNL2_BROADCASTING_MUL_INPLACE_H

/** @brief
 * Performs element-wise multiplication with broadcasting (in place)
 *
 ** @param multiplicand
 * Pointer to multiplicand tensor (will be modified in place)
 *
 ** @param multiplier
 * Pointer to multiplier tensor
 */
void naive_mul_broadcasting_inplace(nnl2_tensor* multiplicand, const nnl2_tensor* multiplier) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks for max safety level 
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(multiplicand, "Multiplicand tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(multiplier, "Multiplier tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(multiplicand->shape, "Multiplicand shape is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(multiplier->shape, "Multiplier shape is NULL");
    #endif
    
    size_t numel_multiplicand = product(multiplicand->shape, multiplicand->rank);
    size_t numel_multiplier = product(multiplier->shape, multiplier->rank);
    
    // Getting the tensor data types
    nnl2_tensor_type multiplicand_dtype = multiplicand->dtype;
    nnl2_tensor_type multiplier_dtype = multiplier->dtype;

    if((numel_multiplicand % numel_multiplier) == 0) {
        // Handling the case where the data types match (more efficiently)
        if(multiplicand_dtype == multiplier_dtype) {
            switch(multiplicand_dtype) {
                case FLOAT64: {
                    double* cast_multiplicand_data = (double*)multiplicand->data;
                    double* cast_multiplier_data = (double*)multiplier->data;

                    for(size_t i = 0; i < (numel_multiplicand / numel_multiplier); i++) {
                        for(size_t j = 0; j < numel_multiplier; j++) {
                            cast_multiplicand_data[i * numel_multiplier + j] *= cast_multiplier_data[j];
                        }
                    }

                    break;
                }

                case FLOAT32: {
                    float* cast_multiplicand_data = (float*)multiplicand->data;
                    float* cast_multiplier_data = (float*)multiplier->data;

                    for(size_t i = 0; i < (numel_multiplicand / numel_multiplier); i++) {
                        for(size_t j = 0; j < numel_multiplier; j++) {
                            cast_multiplicand_data[i * numel_multiplier + j] *= cast_multiplier_data[j];
                        }
                    }

                    break;
                }

                case INT32: {
                    int32_t* cast_multiplicand_data = (int32_t*)multiplicand->data;
                    int32_t* cast_multiplier_data = (int32_t*)multiplier->data;

                    for(size_t i = 0; i < (numel_multiplicand / numel_multiplier); i++) {
                        for(size_t j = 0; j < numel_multiplier; j++) {
                            cast_multiplicand_data[i * numel_multiplier + j] *= cast_multiplier_data[j];
                        }
                    }

                    break;
                }

                default: {
                    NNL2_TYPE_ERROR(multiplicand_dtype);
                    return;
                }
            }
        } else {
            // Handling a case with different data types (conversion required)
            size_t multiplier_step = get_dtype_size(multiplier_dtype);
            char* multiplier_data = (char*)multiplier->data;
            
            switch(multiplicand_dtype) {
                case FLOAT64: {
                    double* data_multiplicand = (double*)multiplicand->data;
                
                    for(size_t i = 0; i < (numel_multiplicand / numel_multiplier); i++) {
                        for(size_t j = 0; j < numel_multiplier; j++) {
                            void* multiplier_elem = multiplier_data + j * multiplier_step;
                            data_multiplicand[i * numel_multiplier + j] *= nnl2_convert_to_float64(multiplier_elem, multiplier_dtype);
                        }
                    }
                
                    break; 
                }
                
                case FLOAT32: {
                    float* data_multiplicand = (float*)multiplicand->data;
                
                    for(size_t i = 0; i < (numel_multiplicand / numel_multiplier); i++) {
                        for(size_t j = 0; j < numel_multiplier; j++) {
                            void* multiplier_elem = multiplier_data + j * multiplier_step;
                            data_multiplicand[i * numel_multiplier + j] *= nnl2_convert_to_float32(multiplier_elem, multiplier_dtype);
                        }
                    }
                
                    break; 
                }
                
                case INT32: {
                    int32_t* data_multiplicand = (int32_t*)multiplicand->data;
                
                    for(size_t i = 0; i < (numel_multiplicand / numel_multiplier); i++) {
                        for(size_t j = 0; j < numel_multiplier; j++) {
                            void* multiplier_elem = multiplier_data + j * multiplier_step;
                            data_multiplicand[i * numel_multiplier + j] *= nnl2_convert_to_int32(multiplier_elem, multiplier_dtype);
                        }
                    }
                
                    break; 
                }
                
                default: {
                    NNL2_TYPE_ERROR(multiplicand_dtype);
                    return;
                }
            }
        }
    } else {
        NNL2_ERROR("Cannot broadcast multiplier tensor");
        return;
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

#if defined(NNL2_PTHREAD_AVAILABLE) && defined(NNL2_AVX256_AVAILABLE) && TENSOR_MEM_ALIGNMENT == 32

/** @brief
 * Threshold for enabling parallel execution of the
 * in-place broadcasting multiplication operation
 */
#define NNL2_MUL_BROADCASTING_INPLACE_PARALLEL_THRESHOLD 500000

/** @brief 
 * Worker function for parallel double precision in-place broadcasting multiplication
 * 
 ** @param arg 
 * Pointer to mulbroadcasting_inplace_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 */
void* nnl2_own_pmul_broadcasting_inplace_float64(void* arg);

/** @brief
 * Worker function for parallel single precision in-place broadcasting multiplication
 * 
 ** @param arg 
 * Pointer to mulbroadcasting_inplace_ptask structure containing thread parameters  
 * 
 ** @return 
 * NULL (for pthread API compatibility)
 */
void* nnl2_own_pmul_broadcasting_inplace_float32(void* arg);

/** @brief
 * Worker function for parallel integer in-place broadcasting multiplication
 * 
 ** @param arg 
 * Pointer to mulbroadcasting_inplace_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 */
void* nnl2_own_pmul_broadcasting_inplace_int32(void* arg);

/** @brief
 * High-performance parallel implementation of in-place broadcasting multiplication
 * 
 ** @param multiplicand
 * Pointer to multiplicand tensor (will be modified in-place)
 *
 ** @param multiplier
 * Pointer to multiplier tensor (broadcasted)
 */
void nnl2_own_mul_broadcasting_inplace(nnl2_tensor* multiplicand, const nnl2_tensor* multiplier) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(multiplicand, "Multiplicand tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(multiplier, "Multiplier tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(multiplicand->shape, "Multiplicand shape is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(multiplier->shape, "Multiplier shape is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(multiplicand->data, "Multiplicand data is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(multiplier->data, "Multiplier data is NULL");
    #endif
    
    size_t numel_multiplicand = product(multiplicand->shape, multiplicand->rank);
    size_t numel_multiplier = product(multiplier->shape, multiplier->rank);
    
    // Check broadcasting compatibility
    if((numel_multiplicand % numel_multiplier) != 0) {
        NNL2_ERROR("Cannot broadcast multiplier tensor");
        return;
    }
    
    if(numel_multiplicand == 0 || numel_multiplier == 0) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return;
    }
    
    // Fall back to naive implementation for small tensors or different dtypes
    if(numel_multiplicand < NNL2_MUL_BROADCASTING_INPLACE_PARALLEL_THRESHOLD || 
       multiplicand->dtype != multiplier->dtype) {
        naive_mul_broadcasting_inplace(multiplicand, multiplier);
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return;
    }
    
    nnl2_tensor_type dtype = multiplicand->dtype;
    size_t broadcast_ratio = numel_multiplicand / numel_multiplier;
    
    bool is_aligned_multiplicand = NNL2_IS_ALIGNED(multiplicand->data, NNL2_TENSOR_ALIGNMENT_32);
    bool is_aligned_multiplier = NNL2_IS_ALIGNED(multiplier->data, NNL2_TENSOR_ALIGNMENT_32);
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!is_aligned_multiplicand) {
            NNL2_WARN("In nnl2_own mul broadcasting in-place, multiplicand memory is not aligned to 32 bytes. Performance may be suboptimal");
        }
        
        if(!is_aligned_multiplier) {
            NNL2_WARN("In nnl2_own mul broadcasting in-place, multiplier memory is not aligned to 32 bytes. Performance may be suboptimal");
        }
    #endif
    
    size_t num_threads = NNL2_NUM_THREADS;
    pthread_t threads[num_threads];
    mulbroadcasting_inplace_ptask tasks[num_threads];
    
    // Calculate chunk size per thread for the outer loop
    size_t chunk = broadcast_ratio / num_threads;
    size_t remainder = broadcast_ratio % num_threads;
    
    // Initialize common task parameters
    for(size_t i = 0; i < num_threads; i++) {
        tasks[i].dtype = dtype;
        tasks[i].aligned_multiplicand = is_aligned_multiplicand;
        tasks[i].aligned_multiplier = is_aligned_multiplier;
        tasks[i].multiplicand_data = multiplicand->data;
        tasks[i].multiplier_data = multiplier->data;
        tasks[i].numel_multiplier = numel_multiplier;
        tasks[i].broadcast_ratio = broadcast_ratio;
    }
    
    size_t current_start = 0;
    for(size_t i = 0; i < num_threads; i++) {
        size_t current_chunk = chunk + (i < remainder ? 1 : 0);
        
        tasks[i].start = current_start;
        tasks[i].end = current_start + current_chunk;
        
        void* (*worker_func)(void*) = NULL;
        switch(dtype) {
            case FLOAT64: worker_func = nnl2_own_pmul_broadcasting_inplace_float64; break;
            case FLOAT32: worker_func = nnl2_own_pmul_broadcasting_inplace_float32; break;
            case INT32:   worker_func = nnl2_own_pmul_broadcasting_inplace_int32;   break;
            
            default: {
                NNL2_TYPE_ERROR(dtype);
                return;
            }
        }
        
        int status = pthread_create(&threads[i], NULL, worker_func, &tasks[i]);
        if(status != 0) {
            NNL2_THREAD_CREATE_ERROR(status, "nnl2_own_mul_broadcasting_inplace");
            num_threads = i;
            break;
        }
        
        current_start += current_chunk;
    }
    
    for(size_t i = 0; i < num_threads; i++) {
        int join_status = pthread_join(threads[i], NULL);
        if(join_status != 0) {
            NNL2_THREAD_JOIN_ERROR(join_status, "nnl2_own_mul_broadcasting_inplace");
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

void* nnl2_own_pmul_broadcasting_inplace_float64(void* arg) {
    mulbroadcasting_inplace_ptask* task = (mulbroadcasting_inplace_ptask*)arg;
    double* multiplicand_data = (double*)task->multiplicand_data;
    const double* multiplier_data = (const double*)task->multiplier_data;
    size_t start = task->start;
    size_t end = task->end;
    size_t numel_multiplier = task->numel_multiplier;
    
    for(size_t block = start; block < end; block++) {
        size_t base_idx = block * numel_multiplier;
        
        __m256d v_multiplier, v_multiplicand, v_result;
        size_t j = 0;
        
        if(task->aligned_multiplicand && task->aligned_multiplier) {
            for(; j + 3 < numel_multiplier; j += 4) {
                _mm_prefetch((char*)&multiplier_data[j + 16], _MM_HINT_T0);
                _mm_prefetch((char*)&multiplicand_data[base_idx + j + 16], _MM_HINT_T0);
                
                v_multiplier = _mm256_load_pd(&multiplier_data[j]);
                v_multiplicand = _mm256_load_pd(&multiplicand_data[base_idx + j]);
                v_result = _mm256_mul_pd(v_multiplicand, v_multiplier);
                _mm256_store_pd(&multiplicand_data[base_idx + j], v_result);
            }
        } else if(task->aligned_multiplicand) {
            for(; j + 3 < numel_multiplier; j += 4) {
                _mm_prefetch((char*)&multiplier_data[j + 16], _MM_HINT_T0);
                _mm_prefetch((char*)&multiplicand_data[base_idx + j + 16], _MM_HINT_T0);
                
                v_multiplier = _mm256_loadu_pd(&multiplier_data[j]);
                v_multiplicand = _mm256_load_pd(&multiplicand_data[base_idx + j]);
                v_result = _mm256_mul_pd(v_multiplicand, v_multiplier);
                _mm256_store_pd(&multiplicand_data[base_idx + j], v_result);
            }
        } else if(task->aligned_multiplier) {
            for(; j + 3 < numel_multiplier; j += 4) {
                _mm_prefetch((char*)&multiplier_data[j + 16], _MM_HINT_T0);
                _mm_prefetch((char*)&multiplicand_data[base_idx + j + 16], _MM_HINT_T0);
                
                v_multiplier = _mm256_load_pd(&multiplier_data[j]);
                v_multiplicand = _mm256_loadu_pd(&multiplicand_data[base_idx + j]);
                v_result = _mm256_mul_pd(v_multiplicand, v_multiplier);
                _mm256_storeu_pd(&multiplicand_data[base_idx + j], v_result);
            }
        } else {
            for(; j + 3 < numel_multiplier; j += 4) {
                _mm_prefetch((char*)&multiplier_data[j + 16], _MM_HINT_T0);
                _mm_prefetch((char*)&multiplicand_data[base_idx + j + 16], _MM_HINT_T0);
                
                v_multiplier = _mm256_loadu_pd(&multiplier_data[j]);
                v_multiplicand = _mm256_loadu_pd(&multiplicand_data[base_idx + j]);
                v_result = _mm256_mul_pd(v_multiplicand, v_multiplier);
                _mm256_storeu_pd(&multiplicand_data[base_idx + j], v_result);
            }
        }
        
        // Handle remaining elements
        for(; j < numel_multiplier; j++) {
            multiplicand_data[base_idx + j] *= multiplier_data[j];
        }
    }
    
    return NULL;
}

void* nnl2_own_pmul_broadcasting_inplace_float32(void* arg) {
    mulbroadcasting_inplace_ptask* task = (mulbroadcasting_inplace_ptask*)arg;
    float* multiplicand_data = (float*)task->multiplicand_data;
    const float* multiplier_data = (const float*)task->multiplier_data;
    size_t start = task->start;
    size_t end = task->end;
    size_t numel_multiplier = task->numel_multiplier;
    
    for(size_t block = start; block < end; block++) {
        size_t base_idx = block * numel_multiplier;
        
        __m256 v_multiplier, v_multiplicand, v_result;
        size_t j = 0;
        
        if(task->aligned_multiplicand && task->aligned_multiplier) {
            for(; j + 7 < numel_multiplier; j += 8) {
                _mm_prefetch((char*)&multiplier_data[j + 32], _MM_HINT_T0);
                _mm_prefetch((char*)&multiplicand_data[base_idx + j + 32], _MM_HINT_T0);
                
                v_multiplier = _mm256_load_ps(&multiplier_data[j]);
                v_multiplicand = _mm256_load_ps(&multiplicand_data[base_idx + j]);
                v_result = _mm256_mul_ps(v_multiplicand, v_multiplier);
                _mm256_store_ps(&multiplicand_data[base_idx + j], v_result);
            }
        } else if(task->aligned_multiplicand) {
            for(; j + 7 < numel_multiplier; j += 8) {
                _mm_prefetch((char*)&multiplier_data[j + 32], _MM_HINT_T0);
                _mm_prefetch((char*)&multiplicand_data[base_idx + j + 32], _MM_HINT_T0);
                
                v_multiplier = _mm256_loadu_ps(&multiplier_data[j]);
                v_multiplicand = _mm256_load_ps(&multiplicand_data[base_idx + j]);
                v_result = _mm256_mul_ps(v_multiplicand, v_multiplier);
                _mm256_store_ps(&multiplicand_data[base_idx + j], v_result);
            }
        } else if(task->aligned_multiplier) {
            for(; j + 7 < numel_multiplier; j += 8) {
                _mm_prefetch((char*)&multiplier_data[j + 32], _MM_HINT_T0);
                _mm_prefetch((char*)&multiplicand_data[base_idx + j + 32], _MM_HINT_T0);
                
                v_multiplier = _mm256_load_ps(&multiplier_data[j]);
                v_multiplicand = _mm256_loadu_ps(&multiplicand_data[base_idx + j]);
                v_result = _mm256_mul_ps(v_multiplicand, v_multiplier);
                _mm256_storeu_ps(&multiplicand_data[base_idx + j], v_result);
            }
        } else {
            for(; j + 7 < numel_multiplier; j += 8) {
                _mm_prefetch((char*)&multiplier_data[j + 32], _MM_HINT_T0);
                _mm_prefetch((char*)&multiplicand_data[base_idx + j + 32], _MM_HINT_T0);
                
                v_multiplier = _mm256_loadu_ps(&multiplier_data[j]);
                v_multiplicand = _mm256_loadu_ps(&multiplicand_data[base_idx + j]);
                v_result = _mm256_mul_ps(v_multiplicand, v_multiplier);
                _mm256_storeu_ps(&multiplicand_data[base_idx + j], v_result);
            }
        }
        
        // Handle remaining elements
        for(; j < numel_multiplier; j++) {
            multiplicand_data[base_idx + j] *= multiplier_data[j];
        }
    }
    
    return NULL;
}

void* nnl2_own_pmul_broadcasting_inplace_int32(void* arg) {
    mulbroadcasting_inplace_ptask* task = (mulbroadcasting_inplace_ptask*)arg;
    int32_t* multiplicand_data = (int32_t*)task->multiplicand_data;
    const int32_t* multiplier_data = (const int32_t*)task->multiplier_data;
    size_t start = task->start;
    size_t end = task->end;
    size_t numel_multiplier = task->numel_multiplier;
    
    // Integer multiplication uses scalar operations for precise arithmetic semantics
    for(size_t block = start; block < end; block++) {
        size_t base_idx = block * numel_multiplier;
        
        for(size_t j = 0; j < numel_multiplier; j++) {
            multiplicand_data[base_idx + j] *= multiplier_data[j];
        }
    }
    
    return NULL;
}

#endif

/**
 * @ingroup backend_system
 * @brief Backend implementations for multiplication with broadcasting (in place)
 * @details
 * Array follows the common backend registration pattern for in-place multiplication
 * with broadcasting operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for in-place multiplication with broadcasting
 *  - nnl2_own: High-performance implementation with AVX256, pthread and prefetching
 * 
 * @see nnl2_naive
 * @see naive_mul_broadcasting_inplace
 * @see nnl2_own_mul_broadcasting_inplace
 */
nnl2_runtime_implementation mul_broadcasting_inplace_backends[] = {
    REGISTER_BACKEND(naive_mul_broadcasting_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
    
    #if defined(NNL2_PTHREAD_AVAILABLE) && defined(NNL2_AVX256_AVAILABLE) && TENSOR_MEM_ALIGNMENT == 32
        REGISTER_BACKEND(nnl2_own_mul_broadcasting_inplace, nnl2_own, NNL2_OWN_NAME),
    #endif
};  

/**
 * @brief Function pointer for multiplication with broadcasting operation (in place)
 * @ingroup backend_system
 */
mulbroadcastinginplacefn mul_broadcasting_inplace;

/**
 * @brief Sets the backend for multiplication with broadcasting operation (in place)
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for multiplication with broadcasting
 */
void set_mul_broadcasting_inplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(mul_broadcasting_inplace_backends, mul_broadcasting_inplace, backend_name);
}

#endif /** NNL2_BROADCASTING_MUL_INPLACE_H **/
