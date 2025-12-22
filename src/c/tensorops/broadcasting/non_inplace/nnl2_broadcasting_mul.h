#ifndef NNL2_BROADCASTING_MUL_H
#define NNL2_BROADCASTING_MUL_H

/** @brief
 * Performs element-wise multiplication with broadcasting support
 *
 ** @param multiplicand
 * First tensor to multiply
 *
 ** @param multiplier
 * Second tensor to multiply
 * 
 ** @return
 * New tensor containing the result of multiplication
 *
 ** @note
 * Contains type conversion
 */
nnl2_tensor* naive_mul_broadcasting(nnl2_tensor* multiplicand, nnl2_tensor* multiplier) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks for max safety level 
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(multiplicand, "Multiplicand tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(multiplier, "Multiplier tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(multiplicand->shape, "Multiplicand shape is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(multiplier->shape, "Multiplier shape is NULL", NULL);
    #endif
 
    size_t numel_multiplicand = nnl2_product(multiplicand->shape, multiplicand->rank);
    size_t numel_multiplier = nnl2_product(multiplier->shape, multiplier->rank);
    
    // Getting the tensor data types
    nnl2_tensor_type multiplicand_dtype = multiplicand->dtype;
    nnl2_tensor_type multiplier_dtype = multiplier->dtype;
    
    nnl2_tensor_type winner_in_the_type_hierarchy = MAX(multiplicand_dtype, multiplier_dtype);
    
    // Сreating a resultant tensor
    nnl2_tensor* result = nnl2_empty(multiplicand->shape, multiplicand->rank, winner_in_the_type_hierarchy);

    if((numel_multiplicand % numel_multiplier) == 0) {
        if(multiplicand_dtype == multiplier_dtype) {
            // Handling the case where the data types match (more efficiently)
            switch(multiplicand_dtype) {
                case FLOAT64: {
                    double* cast_multiplicand_data = (double*)multiplicand->data;
                    double* cast_multiplier_data = (double*)multiplier->data;
                    double* cast_result_data = (double*)result->data;

                    for(size_t i = 0; i < (numel_multiplicand / numel_multiplier); i++) {
                        for(size_t j = 0; j < numel_multiplier; j++) {
                            cast_result_data[i * numel_multiplier + j] = cast_multiplicand_data[i * numel_multiplier + j] * cast_multiplier_data[j];
                        }
                    }

                    break;
                }

                case FLOAT32: {
                    float* cast_multiplicand_data = (float*)multiplicand->data;
                    float* cast_multiplier_data = (float*)multiplier->data;
                    float* cast_result_data = (float*)result->data;

                    for(size_t i = 0; i < (numel_multiplicand / numel_multiplier); i++) {
                        for(size_t j = 0; j < numel_multiplier; j++) {
                            cast_result_data[i * numel_multiplier + j] = cast_multiplicand_data[i * numel_multiplier + j] * cast_multiplier_data[j];
                        }
                    }

                    break;
                }

                case INT32: {
                    int32_t* cast_multiplicand_data = (int32_t*)multiplicand->data;
                    int32_t* cast_multiplier_data = (int32_t*)multiplier->data;
                    int32_t* cast_result_data = (int32_t*)result->data;

                    for(size_t i = 0; i < (numel_multiplicand / numel_multiplier); i++) {
                        for(size_t j = 0; j < numel_multiplier; j++) {
                            cast_result_data[i * numel_multiplier + j] = cast_multiplicand_data[i * numel_multiplier + j] * cast_multiplier_data[j];
                        }
                    }

                    break;
                }

                default: {
                    NNL2_TYPE_ERROR(multiplicand_dtype);
                    return NULL;
                }
            }
            
            #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                NNL2_FUNC_EXIT();
            #endif
            
            return result;
        } else {
            // Handling a case with different data types (conversion required)
            size_t multiplicand_step = get_dtype_size(multiplicand_dtype);
            size_t multiplier_step = get_dtype_size(multiplier_dtype);
            
            switch(winner_in_the_type_hierarchy) {
                case FLOAT64: {
                    double* cast_data_result = (double*)result->data;
                    
                    char* cast_multiplicand_data = (char*)multiplicand->data;
                    char* cast_multiplier_data =  (char*)multiplier->data;
                    
                    for(size_t i = 0; i < (numel_multiplicand / numel_multiplier); i++) {
                        for(size_t j = 0; j < numel_multiplier; j++) {
                            void* elem_multiplicand = cast_multiplicand_data + (i * numel_multiplier + j) * multiplicand_step;
                            void* elem_multiplier = cast_multiplier_data + j * multiplier_step; 
                            
                            cast_data_result[i * numel_multiplier + j] = nnl2_convert_to_float64(elem_multiplicand, multiplicand_dtype) * nnl2_convert_to_float64(elem_multiplier, multiplier_dtype);
                        }
                    }
                    
                    break;
                }
                
                case FLOAT32: {
                    float* cast_data_result = (float*)result->data;
                    
                    char* cast_multiplicand_data = (char*)multiplicand->data;
                    char* cast_multiplier_data =  (char*)multiplier->data;
                    
                    for(size_t i = 0; i < (numel_multiplicand / numel_multiplier); i++) {
                        for(size_t j = 0; j < numel_multiplier; j++) {
                            void* elem_multiplicand = cast_multiplicand_data + (i * numel_multiplier + j) * multiplicand_step;
                            void* elem_multiplier = cast_multiplier_data + j * multiplier_step;
                            
                            cast_data_result[i * numel_multiplier + j] = nnl2_convert_to_float32(elem_multiplicand, multiplicand_dtype) * nnl2_convert_to_float32(elem_multiplier, multiplier_dtype);
                        }
                    }
                    
                    break;
                }
                
                case INT32: {
                    int32_t* cast_data_result = (int32_t*)result->data;
                    
                    char* cast_multiplicand_data = (char*)multiplicand->data;
                    char* cast_multiplier_data =  (char*)multiplier->data;
                    
                    for(size_t i = 0; i < (numel_multiplicand / numel_multiplier); i++) {                    
                        for(size_t j = 0; j < numel_multiplier; j++) {
                            void* elem_multiplicand = cast_multiplicand_data + (i * numel_multiplier + j) * multiplicand_step;
                            void* elem_multiplier = cast_multiplier_data + j * multiplier_step;
                        
                            cast_data_result[i * numel_multiplier + j] = nnl2_convert_to_int32(elem_multiplicand, multiplicand_dtype) * nnl2_convert_to_int32(elem_multiplier, multiplier_dtype);
                        }
                    }
                    
                    break;
                }
                
                default: {
                    NNL2_TYPE_ERROR(winner_in_the_type_hierarchy);
                    nnl2_free_tensor(result);
                    return NULL;
                }
            }
            
            #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                NNL2_FUNC_EXIT();
            #endif
            
            return result;
        }
    } else {
        NNL2_ERROR("Cannot broadcast multiplier tensor");
        return NULL;
    }
}

#if defined(NNL2_PTHREAD_AVAILABLE) && defined(NNL2_AVX256_AVAILABLE) && TENSOR_MEM_ALIGNMENT == 32

/** @brief
 * Threshold for enabling parallel execution of the
 * broadcasting multiplication operation
 */
#define NNL2_MUL_BROADCASTING_PARALLEL_THRESHOLD 500000

/** @brief 
 * Worker function for parallel double precision broadcasting multiplication
 * 
 ** @param arg 
 * Pointer to mulbroadcasting_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 */
void* nnl2_own_pmul_broadcasting_float64(void* arg);

/** @brief
 * Worker function for parallel single precision broadcasting multiplication
 * 
 ** @param arg 
 * Pointer to mulbroadcasting_ptask structure containing thread parameters  
 * 
 ** @return 
 * NULL (for pthread API compatibility)
 */
void* nnl2_own_pmul_broadcasting_float32(void* arg);

/** @brief
 * Worker function for parallel integer broadcasting multiplication
 * 
 ** @param arg 
 * Pointer to mulbroadcasting_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 */
void* nnl2_own_pmul_broadcasting_int32(void* arg);

/** @brief
 * High-performance parallel implementation of broadcasting multiplication
 * 
 ** @param multiplicand
 * Pointer to multiplicand tensor
 *
 ** @param multiplier
 * Pointer to multiplier tensor (broadcasted)
 *
 ** @return
 * Pointer to a new tensor containing the result of the multiplication operation
 */
nnl2_tensor* nnl2_own_mul_broadcasting(nnl2_tensor* multiplicand, nnl2_tensor* multiplier) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(multiplicand, "Multiplicand tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(multiplier, "Multiplier tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(multiplicand->shape, "Multiplicand shape is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(multiplier->shape, "Multiplier shape is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(multiplicand->data, "Multiplicand data is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(multiplier->data, "Multiplier data is NULL", NULL);
    #endif
    
    size_t numel_multiplicand = nnl2_product(multiplicand->shape, multiplicand->rank);
    size_t numel_multiplier = nnl2_product(multiplier->shape, multiplier->rank);
    
    // Check broadcasting compatibility
    if((numel_multiplicand % numel_multiplier) != 0) {
        NNL2_ERROR("Cannot broadcast multiplier tensor");
        return NULL;
    }
    
    // Determine result data type
    nnl2_tensor_type result_dtype = MAX(multiplicand->dtype, multiplier->dtype);
    
    // Create result tensor
    nnl2_tensor* result = nnl2_empty(multiplicand->shape, multiplicand->rank, result_dtype);
    if(result == NULL) {
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
            NNL2_ERROR("Failed to allocate result tensor");
        #endif
        return NULL;
    }
    
    if(numel_multiplicand == 0 || numel_multiplier == 0) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return result;
    }
    
    // Fall back to naive implementation for small tensors or different dtypes
    if(numel_multiplicand < NNL2_MUL_BROADCASTING_PARALLEL_THRESHOLD || 
       multiplicand->dtype != multiplier->dtype) {
        nnl2_tensor* naive_result = naive_mul_broadcasting(multiplicand, multiplier);
        if(naive_result == NULL) {
            nnl2_free_tensor(result);
            return NULL;
        }
        nnl2_free_tensor(result);
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return naive_result;
    }
    
    size_t broadcast_ratio = numel_multiplicand / numel_multiplier;
    
    bool is_aligned_multiplicand = NNL2_IS_ALIGNED(multiplicand->data, NNL2_TENSOR_ALIGNMENT_32);
    bool is_aligned_multiplier = NNL2_IS_ALIGNED(multiplier->data, NNL2_TENSOR_ALIGNMENT_32);
    bool is_aligned_result = NNL2_IS_ALIGNED(result->data, NNL2_TENSOR_ALIGNMENT_32);
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!is_aligned_multiplicand) {
            NNL2_WARN("In nnl2_own mul broadcasting, multiplicand memory is not aligned to 32 bytes. Performance may be suboptimal");
        }
        
        if(!is_aligned_multiplier) {
            NNL2_WARN("In nnl2_own mul broadcasting, multiplier memory is not aligned to 32 bytes. Performance may be suboptimal");
        }
        
        if(!is_aligned_result) {
            NNL2_WARN("In nnl2_own mul broadcasting, result memory is not aligned to 32 bytes. Performance may be suboptimal");
        }
    #endif
    
    size_t num_threads = NNL2_NUM_THREADS;
    pthread_t threads[num_threads];
    mulbroadcasting_ptask tasks[num_threads];
    
    // Calculate chunk size per thread for the outer loop
    size_t chunk = broadcast_ratio / num_threads;
    size_t remainder = broadcast_ratio % num_threads;
    
    // Initialize common task parameters
    for(size_t i = 0; i < num_threads; i++) {
        tasks[i].dtype = result_dtype;
        tasks[i].aligned_multiplicand = is_aligned_multiplicand;
        tasks[i].aligned_multiplier = is_aligned_multiplier;
        tasks[i].aligned_result = is_aligned_result;
        tasks[i].multiplicand_data = multiplicand->data;
        tasks[i].multiplier_data = multiplier->data;
        tasks[i].result_data = result->data;
        tasks[i].numel_multiplier = numel_multiplier;
        tasks[i].broadcast_ratio = broadcast_ratio;
    }
    
    size_t current_start = 0;
    for(size_t i = 0; i < num_threads; i++) {
        size_t current_chunk = chunk + (i < remainder ? 1 : 0);
        
        tasks[i].start = current_start;
        tasks[i].end = current_start + current_chunk;
        
        void* (*worker_func)(void*) = NULL;
        switch(result_dtype) {
            case FLOAT64: worker_func = nnl2_own_pmul_broadcasting_float64; break;
            case FLOAT32: worker_func = nnl2_own_pmul_broadcasting_float32; break;
            case INT32:   worker_func = nnl2_own_pmul_broadcasting_int32;   break;
            
            default: {
                NNL2_TYPE_ERROR(result_dtype);
                nnl2_free_tensor(result);
                return NULL;
            }
        }
        
        int status = pthread_create(&threads[i], NULL, worker_func, &tasks[i]);
        if(status != 0) {
            NNL2_THREAD_CREATE_ERROR(status, "nnl2_own_mul_broadcasting");
            num_threads = i;
            break;
        }
        
        current_start += current_chunk;
    }
    
    for(size_t i = 0; i < num_threads; i++) {
        int join_status = pthread_join(threads[i], NULL);
        if(join_status != 0) {
            NNL2_THREAD_JOIN_ERROR(join_status, "nnl2_own_mul_broadcasting");
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return result;
}

void* nnl2_own_pmul_broadcasting_float64(void* arg) {
    mulbroadcasting_ptask* task = (mulbroadcasting_ptask*)arg;
    const double* multiplicand_data = (const double*)task->multiplicand_data;
    const double* multiplier_data = (const double*)task->multiplier_data;
    double* result_data = (double*)task->result_data;
    size_t start = task->start;
    size_t end = task->end;
    size_t numel_multiplier = task->numel_multiplier;
    
    for(size_t block = start; block < end; block++) {
        size_t base_idx = block * numel_multiplier;
        
        __m256d v_multiplier, v_multiplicand, v_result;
        size_t j = 0;
        
        if(task->aligned_multiplicand && task->aligned_multiplier && task->aligned_result) {
            for(; j + 3 < numel_multiplier; j += 4) {
                _mm_prefetch((char*)&multiplier_data[j + 16], _MM_HINT_T0);
                _mm_prefetch((char*)&multiplicand_data[base_idx + j + 16], _MM_HINT_T0);
                
                v_multiplier = _mm256_load_pd(&multiplier_data[j]);
                v_multiplicand = _mm256_load_pd(&multiplicand_data[base_idx + j]);
                v_result = _mm256_mul_pd(v_multiplicand, v_multiplier);
                _mm256_store_pd(&result_data[base_idx + j], v_result);
            }
        } else if(task->aligned_multiplicand && task->aligned_multiplier) {
            for(; j + 3 < numel_multiplier; j += 4) {
                _mm_prefetch((char*)&multiplier_data[j + 16], _MM_HINT_T0);
                _mm_prefetch((char*)&multiplicand_data[base_idx + j + 16], _MM_HINT_T0);
                
                v_multiplier = _mm256_load_pd(&multiplier_data[j]);
                v_multiplicand = _mm256_load_pd(&multiplicand_data[base_idx + j]);
                v_result = _mm256_mul_pd(v_multiplicand, v_multiplier);
                _mm256_storeu_pd(&result_data[base_idx + j], v_result);
            }
        } else if(task->aligned_multiplicand && task->aligned_result) {
            for(; j + 3 < numel_multiplier; j += 4) {
                _mm_prefetch((char*)&multiplier_data[j + 16], _MM_HINT_T0);
                _mm_prefetch((char*)&multiplicand_data[base_idx + j + 16], _MM_HINT_T0);
                
                v_multiplier = _mm256_loadu_pd(&multiplier_data[j]);
                v_multiplicand = _mm256_load_pd(&multiplicand_data[base_idx + j]);
                v_result = _mm256_mul_pd(v_multiplicand, v_multiplier);
                _mm256_store_pd(&result_data[base_idx + j], v_result);
            }
        } else if(task->aligned_multiplier && task->aligned_result) {
            for(; j + 3 < numel_multiplier; j += 4) {
                _mm_prefetch((char*)&multiplier_data[j + 16], _MM_HINT_T0);
                _mm_prefetch((char*)&multiplicand_data[base_idx + j + 16], _MM_HINT_T0);
                
                v_multiplier = _mm256_load_pd(&multiplier_data[j]);
                v_multiplicand = _mm256_loadu_pd(&multiplicand_data[base_idx + j]);
                v_result = _mm256_mul_pd(v_multiplicand, v_multiplier);
                _mm256_store_pd(&result_data[base_idx + j], v_result);
            }
        } else if(task->aligned_multiplicand) {
            for(; j + 3 < numel_multiplier; j += 4) {
                _mm_prefetch((char*)&multiplier_data[j + 16], _MM_HINT_T0);
                _mm_prefetch((char*)&multiplicand_data[base_idx + j + 16], _MM_HINT_T0);
                
                v_multiplier = _mm256_loadu_pd(&multiplier_data[j]);
                v_multiplicand = _mm256_load_pd(&multiplicand_data[base_idx + j]);
                v_result = _mm256_mul_pd(v_multiplicand, v_multiplier);
                _mm256_storeu_pd(&result_data[base_idx + j], v_result);
            }
        } else if(task->aligned_multiplier) {
            for(; j + 3 < numel_multiplier; j += 4) {
                _mm_prefetch((char*)&multiplier_data[j + 16], _MM_HINT_T0);
                _mm_prefetch((char*)&multiplicand_data[base_idx + j + 16], _MM_HINT_T0);
                
                v_multiplier = _mm256_load_pd(&multiplier_data[j]);
                v_multiplicand = _mm256_loadu_pd(&multiplicand_data[base_idx + j]);
                v_result = _mm256_mul_pd(v_multiplicand, v_multiplier);
                _mm256_storeu_pd(&result_data[base_idx + j], v_result);
            }
        } else if(task->aligned_result) {
            for(; j + 3 < numel_multiplier; j += 4) {
                _mm_prefetch((char*)&multiplier_data[j + 16], _MM_HINT_T0);
                _mm_prefetch((char*)&multiplicand_data[base_idx + j + 16], _MM_HINT_T0);
                
                v_multiplier = _mm256_loadu_pd(&multiplier_data[j]);
                v_multiplicand = _mm256_loadu_pd(&multiplicand_data[base_idx + j]);
                v_result = _mm256_mul_pd(v_multiplicand, v_multiplier);
                _mm256_store_pd(&result_data[base_idx + j], v_result);
            }
        } else {
            for(; j + 3 < numel_multiplier; j += 4) {
                _mm_prefetch((char*)&multiplier_data[j + 16], _MM_HINT_T0);
                _mm_prefetch((char*)&multiplicand_data[base_idx + j + 16], _MM_HINT_T0);
                
                v_multiplier = _mm256_loadu_pd(&multiplier_data[j]);
                v_multiplicand = _mm256_loadu_pd(&multiplicand_data[base_idx + j]);
                v_result = _mm256_mul_pd(v_multiplicand, v_multiplier);
                _mm256_storeu_pd(&result_data[base_idx + j], v_result);
            }
        }
        
        // Handle remaining elements
        for(; j < numel_multiplier; j++) {
            result_data[base_idx + j] = multiplicand_data[base_idx + j] * multiplier_data[j];
        }
    }
    
    return NULL;
}

void* nnl2_own_pmul_broadcasting_float32(void* arg) {
    mulbroadcasting_ptask* task = (mulbroadcasting_ptask*)arg;
    const float* multiplicand_data = (const float*)task->multiplicand_data;
    const float* multiplier_data = (const float*)task->multiplier_data;
    float* result_data = (float*)task->result_data;
    size_t start = task->start;
    size_t end = task->end;
    size_t numel_multiplier = task->numel_multiplier;
    
    for(size_t block = start; block < end; block++) {
        size_t base_idx = block * numel_multiplier;
        
        __m256 v_multiplier, v_multiplicand, v_result;
        size_t j = 0;
        
        if(task->aligned_multiplicand && task->aligned_multiplier && task->aligned_result) {
            for(; j + 7 < numel_multiplier; j += 8) {
                _mm_prefetch((char*)&multiplier_data[j + 32], _MM_HINT_T0);
                _mm_prefetch((char*)&multiplicand_data[base_idx + j + 32], _MM_HINT_T0);
                
                v_multiplier = _mm256_load_ps(&multiplier_data[j]);
                v_multiplicand = _mm256_load_ps(&multiplicand_data[base_idx + j]);
                v_result = _mm256_mul_ps(v_multiplicand, v_multiplier);
                _mm256_store_ps(&result_data[base_idx + j], v_result);
            }
        } else if(task->aligned_multiplicand && task->aligned_multiplier) {
            for(; j + 7 < numel_multiplier; j += 8) {
                _mm_prefetch((char*)&multiplier_data[j + 32], _MM_HINT_T0);
                _mm_prefetch((char*)&multiplicand_data[base_idx + j + 32], _MM_HINT_T0);
                
                v_multiplier = _mm256_load_ps(&multiplier_data[j]);
                v_multiplicand = _mm256_load_ps(&multiplicand_data[base_idx + j]);
                v_result = _mm256_mul_ps(v_multiplicand, v_multiplier);
                _mm256_storeu_ps(&result_data[base_idx + j], v_result);
            }
        } else if(task->aligned_multiplicand && task->aligned_result) {
            for(; j + 7 < numel_multiplier; j += 8) {
                _mm_prefetch((char*)&multiplier_data[j + 32], _MM_HINT_T0);
                _mm_prefetch((char*)&multiplicand_data[base_idx + j + 32], _MM_HINT_T0);
                
                v_multiplier = _mm256_loadu_ps(&multiplier_data[j]);
                v_multiplicand = _mm256_load_ps(&multiplicand_data[base_idx + j]);
                v_result = _mm256_mul_ps(v_multiplicand, v_multiplier);
                _mm256_store_ps(&result_data[base_idx + j], v_result);
            }
        } else if(task->aligned_multiplier && task->aligned_result) {
            for(; j + 7 < numel_multiplier; j += 8) {
                _mm_prefetch((char*)&multiplier_data[j + 32], _MM_HINT_T0);
                _mm_prefetch((char*)&multiplicand_data[base_idx + j + 32], _MM_HINT_T0);
                
                v_multiplier = _mm256_load_ps(&multiplier_data[j]);
                v_multiplicand = _mm256_loadu_ps(&multiplicand_data[base_idx + j]);
                v_result = _mm256_mul_ps(v_multiplicand, v_multiplier);
                _mm256_store_ps(&result_data[base_idx + j], v_result);
            }
        } else if(task->aligned_multiplicand) {
            for(; j + 7 < numel_multiplier; j += 8) {
                _mm_prefetch((char*)&multiplier_data[j + 32], _MM_HINT_T0);
                _mm_prefetch((char*)&multiplicand_data[base_idx + j + 32], _MM_HINT_T0);
                
                v_multiplier = _mm256_loadu_ps(&multiplier_data[j]);
                v_multiplicand = _mm256_load_ps(&multiplicand_data[base_idx + j]);
                v_result = _mm256_mul_ps(v_multiplicand, v_multiplier);
                _mm256_storeu_ps(&result_data[base_idx + j], v_result);
            }
        } else if(task->aligned_multiplier) {
            for(; j + 7 < numel_multiplier; j += 8) {
                _mm_prefetch((char*)&multiplier_data[j + 32], _MM_HINT_T0);
                _mm_prefetch((char*)&multiplicand_data[base_idx + j + 32], _MM_HINT_T0);
                
                v_multiplier = _mm256_load_ps(&multiplier_data[j]);
                v_multiplicand = _mm256_loadu_ps(&multiplicand_data[base_idx + j]);
                v_result = _mm256_mul_ps(v_multiplicand, v_multiplier);
                _mm256_storeu_ps(&result_data[base_idx + j], v_result);
            }
        } else if(task->aligned_result) {
            for(; j + 7 < numel_multiplier; j += 8) {
                _mm_prefetch((char*)&multiplier_data[j + 32], _MM_HINT_T0);
                _mm_prefetch((char*)&multiplicand_data[base_idx + j + 32], _MM_HINT_T0);
                
                v_multiplier = _mm256_loadu_ps(&multiplier_data[j]);
                v_multiplicand = _mm256_loadu_ps(&multiplicand_data[base_idx + j]);
                v_result = _mm256_mul_ps(v_multiplicand, v_multiplier);
                _mm256_store_ps(&result_data[base_idx + j], v_result);
            }
        } else {
            for(; j + 7 < numel_multiplier; j += 8) {
                _mm_prefetch((char*)&multiplier_data[j + 32], _MM_HINT_T0);
                _mm_prefetch((char*)&multiplicand_data[base_idx + j + 32], _MM_HINT_T0);
                
                v_multiplier = _mm256_loadu_ps(&multiplier_data[j]);
                v_multiplicand = _mm256_loadu_ps(&multiplicand_data[base_idx + j]);
                v_result = _mm256_mul_ps(v_multiplicand, v_multiplier);
                _mm256_storeu_ps(&result_data[base_idx + j], v_result);
            }
        }
        
        // Handle remaining elements
        for(; j < numel_multiplier; j++) {
            result_data[base_idx + j] = multiplicand_data[base_idx + j] * multiplier_data[j];
        }
    }
    
    return NULL;
}

void* nnl2_own_pmul_broadcasting_int32(void* arg) {
    mulbroadcasting_ptask* task = (mulbroadcasting_ptask*)arg;
    const int32_t* multiplicand_data = (const int32_t*)task->multiplicand_data;
    const int32_t* multiplier_data = (const int32_t*)task->multiplier_data;
    int32_t* result_data = (int32_t*)task->result_data;
    size_t start = task->start;
    size_t end = task->end;
    size_t numel_multiplier = task->numel_multiplier;
    
    // Integer multiplication uses scalar operations for precise arithmetic semantics
    for(size_t block = start; block < end; block++) {
        size_t base_idx = block * numel_multiplier;
        
        for(size_t j = 0; j < numel_multiplier; j++) {
            result_data[base_idx + j] = multiplicand_data[base_idx + j] * multiplier_data[j];
        }
    }
    
    return NULL;
}

#endif

/**
 * @ingroup backend_system
 * @brief Backend implementations for multiplication with broadcasting
 * @details
 * Array follows the common backend registration pattern for multiplication
 * with broadcasting operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for multiplication with broadcasting
 *  - nnl2_own: High-performance implementation with AVX256, pthread and prefetching
 * 
 * @see nnl2_naive
 * @see naive_mul_broadcasting
 * @see nnl2_own_mul_broadcasting
 */
nnl2_runtime_implementation mul_broadcasting_backends[] = {
    REGISTER_BACKEND(naive_mul_broadcasting, nnl2_naive, NAIVE_BACKEND_NAME),
    
    #if defined(NNL2_PTHREAD_AVAILABLE) && defined(NNL2_AVX256_AVAILABLE) && TENSOR_MEM_ALIGNMENT == 32
        REGISTER_BACKEND(nnl2_own_mul_broadcasting, nnl2_own, NNL2_OWN_NAME),
    #endif
};  

/**
 * @brief Function pointer for multiplication with broadcasting operation
 * @ingroup backend_system
 */
mulbroadcastingfn mul_broadcasting;

/**
 * @brief Sets the backend for multiplication with broadcasting operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for multiplication with broadcasting
 */
void set_mul_broadcasting_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(mul_broadcasting_backends, mul_broadcasting, backend_name);
}

#endif /** NNL2_BROADCASTING_MUL_H **/
