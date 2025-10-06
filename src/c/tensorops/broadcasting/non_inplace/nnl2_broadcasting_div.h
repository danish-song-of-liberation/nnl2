#ifndef NNL2_BROADCASTING_DIV_H
#define NNL2_BROADCASTING_DIV_H

/** @brief
 * Performs element-wise division with broadcasting support
 *
 ** @param dividend
 * First tensor to divide
 *
 ** @param divisor
 * Second tensor to divide by
 * 
 ** @return
 * New tensor containing the result of division
 *
 ** @note
 * Contains type conversion
 */
Tensor* naive_div_broadcasting(Tensor* dividend, Tensor* divisor) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks for max safety level 
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(dividend, "Dividend tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(divisor, "Divisor tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(dividend->shape, "Dividend shape is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(divisor->shape, "Divisor shape is NULL", NULL);
    #endif
 
    size_t numel_dividend = product(dividend->shape, dividend->rank);
    size_t numel_divisor = product(divisor->shape, divisor->rank);
    
    // Getting the tensor data types
    TensorType dividend_dtype = dividend->dtype;
    TensorType divisor_dtype = divisor->dtype;
    
    TensorType winner_in_the_type_hierarchy = MAX(dividend_dtype, divisor_dtype);
    
    // Сreating a resultant tensor
    Tensor* result = nnl2_empty(dividend->shape, dividend->rank, winner_in_the_type_hierarchy);

    if((numel_dividend % numel_divisor) == 0) {
        if(dividend_dtype == divisor_dtype) {
            // Handling the case where the data types match (more efficiently)
            switch(dividend_dtype) {
                case FLOAT64: {
                    double* cast_dividend_data = (double*)dividend->data;
                    double* cast_divisor_data = (double*)divisor->data;
                    double* cast_result_data = (double*)result->data;

                    for(size_t i = 0; i < (numel_dividend / numel_divisor); i++) {
                        for(size_t j = 0; j < numel_divisor; j++) {
                            cast_result_data[i * numel_divisor + j] = cast_dividend_data[i * numel_divisor + j] / cast_divisor_data[j];
                        }
                    }

                    break;
                }

                case FLOAT32: {
                    float* cast_dividend_data = (float*)dividend->data;
                    float* cast_divisor_data = (float*)divisor->data;
                    float* cast_result_data = (float*)result->data;

                    for(size_t i = 0; i < (numel_dividend / numel_divisor); i++) {
                        for(size_t j = 0; j < numel_divisor; j++) {
                            cast_result_data[i * numel_divisor + j] = cast_dividend_data[i * numel_divisor + j] / cast_divisor_data[j];
                        }
                    }

                    break;
                }

                case INT32: {
                    int32_t* cast_dividend_data = (int32_t*)dividend->data;
                    int32_t* cast_divisor_data = (int32_t*)divisor->data;
                    int32_t* cast_result_data = (int32_t*)result->data;

                    for(size_t i = 0; i < (numel_dividend / numel_divisor); i++) {
                        for(size_t j = 0; j < numel_divisor; j++) {
                            cast_result_data[i * numel_divisor + j] = cast_dividend_data[i * numel_divisor + j] / cast_divisor_data[j];
                        }
                    }

                    break;
                }

                default: {
                    NNL2_TYPE_ERROR(dividend_dtype);
                    return NULL;
                }
            }
            
            #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                NNL2_FUNC_EXIT();
            #endif
            
            return result;
        } else {
            // Handling a case with different data types (conversion required)
            size_t dividend_step = get_dtype_size(dividend_dtype);
            size_t divisor_step = get_dtype_size(divisor_dtype);
            
            switch(winner_in_the_type_hierarchy) {
                case FLOAT64: {
                    double* cast_data_result = (double*)result->data;
                    
                    char* cast_dividend_data = (char*)dividend->data;
                    char* cast_divisor_data =  (char*)divisor->data;
                    
                    for(size_t i = 0; i < (numel_dividend / numel_divisor); i++) {
                        for(size_t j = 0; j < numel_divisor; j++) {
                            void* elem_dividend = cast_dividend_data + (i * numel_divisor + j) * dividend_step;
                            void* elem_divisor = cast_divisor_data + j * divisor_step; 
                            
                            cast_data_result[i * numel_divisor + j] = nnl2_convert_to_float64(elem_dividend, dividend_dtype) / nnl2_convert_to_float64(elem_divisor, divisor_dtype);
                        }
                    }
                    
                    break;
                }
                
                case FLOAT32: {
                    float* cast_data_result = (float*)result->data;
                    
                    char* cast_dividend_data = (char*)dividend->data;
                    char* cast_divisor_data =  (char*)divisor->data;
                    
                    for(size_t i = 0; i < (numel_dividend / numel_divisor); i++) {
                        for(size_t j = 0; j < numel_divisor; j++) {
                            void* elem_dividend = cast_dividend_data + (i * numel_divisor + j) * dividend_step;
                            void* elem_divisor = cast_divisor_data + j * divisor_step;
                            
                            cast_data_result[i * numel_divisor + j] = nnl2_convert_to_float32(elem_dividend, dividend_dtype) / nnl2_convert_to_float32(elem_divisor, divisor_dtype);
                        }
                    }
                    
                    break;
                }
                
                case INT32: {
                    int32_t* cast_data_result = (int32_t*)result->data;
                    
                    char* cast_dividend_data = (char*)dividend->data;
                    char* cast_divisor_data =  (char*)divisor->data;
                    
                    for(size_t i = 0; i < (numel_dividend / numel_divisor); i++) {                    
                        for(size_t j = 0; j < numel_divisor; j++) {
                            void* elem_dividend = cast_dividend_data + (i * numel_divisor + j) * dividend_step;
                            void* elem_divisor = cast_divisor_data + j * divisor_step;
                        
                            cast_data_result[i * numel_divisor + j] = nnl2_convert_to_int32(elem_dividend, dividend_dtype) / nnl2_convert_to_int32(elem_divisor, divisor_dtype);
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
        NNL2_ERROR("Cannot broadcast divisor tensor");
        return NULL;
    }
}

#if defined(NNL2_PTHREAD_AVAILABLE) && defined(NNL2_AVX256_AVAILABLE) && TENSOR_MEM_ALIGNMENT == 32

/** @brief
 * Threshold for enabling parallel execution of the
 * broadcasting division operation
 */
#define NNL2_DIV_BROADCASTING_PARALLEL_THRESHOLD 500000

/** @brief 
 * Worker function for parallel double precision broadcasting division
 * 
 ** @param arg 
 * Pointer to divbroadcasting_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 */
void* nnl2_own_pdiv_broadcasting_float64(void* arg);

/** @brief
 * Worker function for parallel single precision broadcasting division
 * 
 ** @param arg 
 * Pointer to divbroadcasting_ptask structure containing thread parameters  
 * 
 ** @return 
 * NULL (for pthread API compatibility)
 */
void* nnl2_own_pdiv_broadcasting_float32(void* arg);

/** @brief
 * Worker function for parallel integer broadcasting division
 * 
 ** @param arg 
 * Pointer to divbroadcasting_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 */
void* nnl2_own_pdiv_broadcasting_int32(void* arg);

/** @brief
 * High-performance parallel implementation of broadcasting division
 * 
 ** @param dividend
 * Pointer to dividend tensor
 *
 ** @param divisor
 * Pointer to divisor tensor (broadcasted)
 *
 ** @return
 * Pointer to a new tensor containing the result of the division operation
 */
Tensor* nnl2_own_div_broadcasting(Tensor* dividend, Tensor* divisor) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(dividend, "Dividend tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(divisor, "Divisor tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(dividend->shape, "Dividend shape is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(divisor->shape, "Divisor shape is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(dividend->data, "Dividend data is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(divisor->data, "Divisor data is NULL", NULL);
    #endif
    
    size_t numel_dividend = product(dividend->shape, dividend->rank);
    size_t numel_divisor = product(divisor->shape, divisor->rank);
    
    // Check broadcasting compatibility
    if((numel_dividend % numel_divisor) != 0) {
        NNL2_ERROR("Cannot broadcast divisor tensor");
        return NULL;
    }
    
    // Determine result data type
    TensorType result_dtype = MAX(dividend->dtype, divisor->dtype);
    
    // Create result tensor
    Tensor* result = nnl2_empty(dividend->shape, dividend->rank, result_dtype);
    if(result == NULL) {
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
            NNL2_ERROR("Failed to allocate result tensor");
        #endif
        return NULL;
    }
    
    if(numel_dividend == 0 || numel_divisor == 0) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return result;
    }
    
    // Fall back to naive implementation for small tensors or different dtypes
    if(numel_dividend < NNL2_DIV_BROADCASTING_PARALLEL_THRESHOLD || 
       dividend->dtype != divisor->dtype) {
        Tensor* naive_result = naive_div_broadcasting(dividend, divisor);
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
    
    size_t broadcast_ratio = numel_dividend / numel_divisor;
    
    bool is_aligned_dividend = NNL2_IS_ALIGNED(dividend->data, NNL2_TENSOR_ALIGNMENT_32);
    bool is_aligned_divisor = NNL2_IS_ALIGNED(divisor->data, NNL2_TENSOR_ALIGNMENT_32);
    bool is_aligned_result = NNL2_IS_ALIGNED(result->data, NNL2_TENSOR_ALIGNMENT_32);
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!is_aligned_dividend) {
            NNL2_WARN("In nnl2_own div broadcasting, dividend memory is not aligned to 32 bytes. Performance may be suboptimal");
        }
        
        if(!is_aligned_divisor) {
            NNL2_WARN("In nnl2_own div broadcasting, divisor memory is not aligned to 32 bytes. Performance may be suboptimal");
        }
        
        if(!is_aligned_result) {
            NNL2_WARN("In nnl2_own div broadcasting, result memory is not aligned to 32 bytes. Performance may be suboptimal");
        }
    #endif
    
    size_t num_threads = NNL2_NUM_THREADS;
    pthread_t threads[num_threads];
    divbroadcasting_ptask tasks[num_threads];
    
    // Calculate chunk size per thread for the outer loop
    size_t chunk = broadcast_ratio / num_threads;
    size_t remainder = broadcast_ratio % num_threads;
    
    // Initialize common task parameters
    for(size_t i = 0; i < num_threads; i++) {
        tasks[i].dtype = result_dtype;
        tasks[i].aligned_dividend = is_aligned_dividend;
        tasks[i].aligned_divisor = is_aligned_divisor;
        tasks[i].aligned_result = is_aligned_result;
        tasks[i].dividend_data = dividend->data;
        tasks[i].divisor_data = divisor->data;
        tasks[i].result_data = result->data;
        tasks[i].numel_divisor = numel_divisor;
        tasks[i].broadcast_ratio = broadcast_ratio;
    }
    
    size_t current_start = 0;
    for(size_t i = 0; i < num_threads; i++) {
        size_t current_chunk = chunk + (i < remainder ? 1 : 0);
        
        tasks[i].start = current_start;
        tasks[i].end = current_start + current_chunk;
        
        void* (*worker_func)(void*) = NULL;
        switch(result_dtype) {
            case FLOAT64: worker_func = nnl2_own_pdiv_broadcasting_float64; break;
            case FLOAT32: worker_func = nnl2_own_pdiv_broadcasting_float32; break;
            case INT32:   worker_func = nnl2_own_pdiv_broadcasting_int32;   break;
            
            default: {
                NNL2_TYPE_ERROR(result_dtype);
                nnl2_free_tensor(result);
                return NULL;
            }
        }
        
        int status = pthread_create(&threads[i], NULL, worker_func, &tasks[i]);
        if(status != 0) {
            NNL2_THREAD_CREATE_ERROR(status, "nnl2_own_div_broadcasting");
            num_threads = i;
            break;
        }
        
        current_start += current_chunk;
    }
    
    for(size_t i = 0; i < num_threads; i++) {
        int join_status = pthread_join(threads[i], NULL);
        if(join_status != 0) {
            NNL2_THREAD_JOIN_ERROR(join_status, "nnl2_own_div_broadcasting");
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return result;
}

void* nnl2_own_pdiv_broadcasting_float64(void* arg) {
    divbroadcasting_ptask* task = (divbroadcasting_ptask*)arg;
    const double* dividend_data = (const double*)task->dividend_data;
    const double* divisor_data = (const double*)task->divisor_data;
    double* result_data = (double*)task->result_data;
    size_t start = task->start;
    size_t end = task->end;
    size_t numel_divisor = task->numel_divisor;
    
    for(size_t block = start; block < end; block++) {
        size_t base_idx = block * numel_divisor;
        
        __m256d v_divisor, v_dividend, v_result;
        size_t j = 0;
        
        if(task->aligned_dividend && task->aligned_divisor && task->aligned_result) {
            for(; j + 3 < numel_divisor; j += 4) {
                _mm_prefetch((char*)&divisor_data[j + 16], _MM_HINT_T0);
                _mm_prefetch((char*)&dividend_data[base_idx + j + 16], _MM_HINT_T0);
                
                v_divisor = _mm256_load_pd(&divisor_data[j]);
                v_dividend = _mm256_load_pd(&dividend_data[base_idx + j]);
                v_result = _mm256_div_pd(v_dividend, v_divisor);
                _mm256_store_pd(&result_data[base_idx + j], v_result);
            }
        } else if(task->aligned_dividend && task->aligned_divisor) {
            for(; j + 3 < numel_divisor; j += 4) {
                _mm_prefetch((char*)&divisor_data[j + 16], _MM_HINT_T0);
                _mm_prefetch((char*)&dividend_data[base_idx + j + 16], _MM_HINT_T0);
                
                v_divisor = _mm256_load_pd(&divisor_data[j]);
                v_dividend = _mm256_load_pd(&dividend_data[base_idx + j]);
                v_result = _mm256_div_pd(v_dividend, v_divisor);
                _mm256_storeu_pd(&result_data[base_idx + j], v_result);
            }
        } else if(task->aligned_dividend && task->aligned_result) {
            for(; j + 3 < numel_divisor; j += 4) {
                _mm_prefetch((char*)&divisor_data[j + 16], _MM_HINT_T0);
                _mm_prefetch((char*)&dividend_data[base_idx + j + 16], _MM_HINT_T0);
                
                v_divisor = _mm256_loadu_pd(&divisor_data[j]);
                v_dividend = _mm256_load_pd(&dividend_data[base_idx + j]);
                v_result = _mm256_div_pd(v_dividend, v_divisor);
                _mm256_store_pd(&result_data[base_idx + j], v_result);
            }
        } else if(task->aligned_divisor && task->aligned_result) {
            for(; j + 3 < numel_divisor; j += 4) {
                _mm_prefetch((char*)&divisor_data[j + 16], _MM_HINT_T0);
                _mm_prefetch((char*)&dividend_data[base_idx + j + 16], _MM_HINT_T0);
                
                v_divisor = _mm256_load_pd(&divisor_data[j]);
                v_dividend = _mm256_loadu_pd(&dividend_data[base_idx + j]);
                v_result = _mm256_div_pd(v_dividend, v_divisor);
                _mm256_store_pd(&result_data[base_idx + j], v_result);
            }
        } else if(task->aligned_dividend) {
            for(; j + 3 < numel_divisor; j += 4) {
                _mm_prefetch((char*)&divisor_data[j + 16], _MM_HINT_T0);
                _mm_prefetch((char*)&dividend_data[base_idx + j + 16], _MM_HINT_T0);
                
                v_divisor = _mm256_loadu_pd(&divisor_data[j]);
                v_dividend = _mm256_load_pd(&dividend_data[base_idx + j]);
                v_result = _mm256_div_pd(v_dividend, v_divisor);
                _mm256_storeu_pd(&result_data[base_idx + j], v_result);
            }
        } else if(task->aligned_divisor) {
            for(; j + 3 < numel_divisor; j += 4) {
                _mm_prefetch((char*)&divisor_data[j + 16], _MM_HINT_T0);
                _mm_prefetch((char*)&dividend_data[base_idx + j + 16], _MM_HINT_T0);
                
                v_divisor = _mm256_load_pd(&divisor_data[j]);
                v_dividend = _mm256_loadu_pd(&dividend_data[base_idx + j]);
                v_result = _mm256_div_pd(v_dividend, v_divisor);
                _mm256_storeu_pd(&result_data[base_idx + j], v_result);
            }
        } else if(task->aligned_result) {
            for(; j + 3 < numel_divisor; j += 4) {
                _mm_prefetch((char*)&divisor_data[j + 16], _MM_HINT_T0);
                _mm_prefetch((char*)&dividend_data[base_idx + j + 16], _MM_HINT_T0);
                
                v_divisor = _mm256_loadu_pd(&divisor_data[j]);
                v_dividend = _mm256_loadu_pd(&dividend_data[base_idx + j]);
                v_result = _mm256_div_pd(v_dividend, v_divisor);
                _mm256_store_pd(&result_data[base_idx + j], v_result);
            }
        } else {
            for(; j + 3 < numel_divisor; j += 4) {
                _mm_prefetch((char*)&divisor_data[j + 16], _MM_HINT_T0);
                _mm_prefetch((char*)&dividend_data[base_idx + j + 16], _MM_HINT_T0);
                
                v_divisor = _mm256_loadu_pd(&divisor_data[j]);
                v_dividend = _mm256_loadu_pd(&dividend_data[base_idx + j]);
                v_result = _mm256_div_pd(v_dividend, v_divisor);
                _mm256_storeu_pd(&result_data[base_idx + j], v_result);
            }
        }
        
        // Handle remaining elements
        for(; j < numel_divisor; j++) {
            result_data[base_idx + j] = dividend_data[base_idx + j] / divisor_data[j];
        }
    }
    
    return NULL;
}

void* nnl2_own_pdiv_broadcasting_float32(void* arg) {
    divbroadcasting_ptask* task = (divbroadcasting_ptask*)arg;
    const float* dividend_data = (const float*)task->dividend_data;
    const float* divisor_data = (const float*)task->divisor_data;
    float* result_data = (float*)task->result_data;
    size_t start = task->start;
    size_t end = task->end;
    size_t numel_divisor = task->numel_divisor;
    
    for(size_t block = start; block < end; block++) {
        size_t base_idx = block * numel_divisor;
        
        __m256 v_divisor, v_dividend, v_result;
        size_t j = 0;
        
        if(task->aligned_dividend && task->aligned_divisor && task->aligned_result) {
            for(; j + 7 < numel_divisor; j += 8) {
                _mm_prefetch((char*)&divisor_data[j + 32], _MM_HINT_T0);
                _mm_prefetch((char*)&dividend_data[base_idx + j + 32], _MM_HINT_T0);
                
                v_divisor = _mm256_load_ps(&divisor_data[j]);
                v_dividend = _mm256_load_ps(&dividend_data[base_idx + j]);
                v_result = _mm256_div_ps(v_dividend, v_divisor);
                _mm256_store_ps(&result_data[base_idx + j], v_result);
            }
        } else if(task->aligned_dividend && task->aligned_divisor) {
            for(; j + 7 < numel_divisor; j += 8) {
                _mm_prefetch((char*)&divisor_data[j + 32], _MM_HINT_T0);
                _mm_prefetch((char*)&dividend_data[base_idx + j + 32], _MM_HINT_T0);
                
                v_divisor = _mm256_load_ps(&divisor_data[j]);
                v_dividend = _mm256_load_ps(&dividend_data[base_idx + j]);
                v_result = _mm256_div_ps(v_dividend, v_divisor);
                _mm256_storeu_ps(&result_data[base_idx + j], v_result);
            }
        } else if(task->aligned_dividend && task->aligned_result) {
            for(; j + 7 < numel_divisor; j += 8) {
                _mm_prefetch((char*)&divisor_data[j + 32], _MM_HINT_T0);
                _mm_prefetch((char*)&dividend_data[base_idx + j + 32], _MM_HINT_T0);
                
                v_divisor = _mm256_loadu_ps(&divisor_data[j]);
                v_dividend = _mm256_load_ps(&dividend_data[base_idx + j]);
                v_result = _mm256_div_ps(v_dividend, v_divisor);
                _mm256_store_ps(&result_data[base_idx + j], v_result);
            }
        } else if(task->aligned_divisor && task->aligned_result) {
            for(; j + 7 < numel_divisor; j += 8) {
                _mm_prefetch((char*)&divisor_data[j + 32], _MM_HINT_T0);
                _mm_prefetch((char*)&dividend_data[base_idx + j + 32], _MM_HINT_T0);
                
                v_divisor = _mm256_load_ps(&divisor_data[j]);
                v_dividend = _mm256_loadu_ps(&dividend_data[base_idx + j]);
                v_result = _mm256_div_ps(v_dividend, v_divisor);
                _mm256_store_ps(&result_data[base_idx + j], v_result);
            }
        } else if(task->aligned_dividend) {
            for(; j + 7 < numel_divisor; j += 8) {
                _mm_prefetch((char*)&divisor_data[j + 32], _MM_HINT_T0);
                _mm_prefetch((char*)&dividend_data[base_idx + j + 32], _MM_HINT_T0);
                
                v_divisor = _mm256_loadu_ps(&divisor_data[j]);
                v_dividend = _mm256_load_ps(&dividend_data[base_idx + j]);
                v_result = _mm256_div_ps(v_dividend, v_divisor);
                _mm256_storeu_ps(&result_data[base_idx + j], v_result);
            }
        } else if(task->aligned_divisor) {
            for(; j + 7 < numel_divisor; j += 8) {
                _mm_prefetch((char*)&divisor_data[j + 32], _MM_HINT_T0);
                _mm_prefetch((char*)&dividend_data[base_idx + j + 32], _MM_HINT_T0);
                
                v_divisor = _mm256_load_ps(&divisor_data[j]);
                v_dividend = _mm256_loadu_ps(&dividend_data[base_idx + j]);
                v_result = _mm256_div_ps(v_dividend, v_divisor);
                _mm256_storeu_ps(&result_data[base_idx + j], v_result);
            }
        } else if(task->aligned_result) {
            for(; j + 7 < numel_divisor; j += 8) {
                _mm_prefetch((char*)&divisor_data[j + 32], _MM_HINT_T0);
                _mm_prefetch((char*)&dividend_data[base_idx + j + 32], _MM_HINT_T0);
                
                v_divisor = _mm256_loadu_ps(&divisor_data[j]);
                v_dividend = _mm256_loadu_ps(&dividend_data[base_idx + j]);
                v_result = _mm256_div_ps(v_dividend, v_divisor);
                _mm256_store_ps(&result_data[base_idx + j], v_result);
            }
        } else {
            for(; j + 7 < numel_divisor; j += 8) {
                _mm_prefetch((char*)&divisor_data[j + 32], _MM_HINT_T0);
                _mm_prefetch((char*)&dividend_data[base_idx + j + 32], _MM_HINT_T0);
                
                v_divisor = _mm256_loadu_ps(&divisor_data[j]);
                v_dividend = _mm256_loadu_ps(&dividend_data[base_idx + j]);
                v_result = _mm256_div_ps(v_dividend, v_divisor);
                _mm256_storeu_ps(&result_data[base_idx + j], v_result);
            }
        }
        
        // Handle remaining elements
        for(; j < numel_divisor; j++) {
            result_data[base_idx + j] = dividend_data[base_idx + j] / divisor_data[j];
        }
    }
    
    return NULL;
}

void* nnl2_own_pdiv_broadcasting_int32(void* arg) {
    divbroadcasting_ptask* task = (divbroadcasting_ptask*)arg;
    const int32_t* dividend_data = (const int32_t*)task->dividend_data;
    const int32_t* divisor_data = (const int32_t*)task->divisor_data;
    int32_t* result_data = (int32_t*)task->result_data;
    size_t start = task->start;
    size_t end = task->end;
    size_t numel_divisor = task->numel_divisor;
    
    // Integer division uses scalar operations for precise arithmetic semantics
    for(size_t block = start; block < end; block++) {
        size_t base_idx = block * numel_divisor;
        
        for(size_t j = 0; j < numel_divisor; j++) {
            result_data[base_idx + j] = dividend_data[base_idx + j] / divisor_data[j];
        }
    }
    
    return NULL;
}

#endif

/**
 * @ingroup backend_system
 * @brief Backend implementations for division with broadcasting
 * @details
 * Array follows the common backend registration pattern for division
 * with broadcasting operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for division with broadcasting
 *  - nnl2_own: High-performance implementation with AVX256, pthread and prefetching
 * 
 * @see nnl2_naive
 * @see naive_div_broadcasting
 * @see nnl2_own_div_broadcasting
 */
Implementation div_broadcasting_backends[] = {
    REGISTER_BACKEND(naive_div_broadcasting, nnl2_naive, NAIVE_BACKEND_NAME),
    
    #if defined(NNL2_PTHREAD_AVAILABLE) && defined(NNL2_AVX256_AVAILABLE) && TENSOR_MEM_ALIGNMENT == 32
        REGISTER_BACKEND(nnl2_own_div_broadcasting, nnl2_own, NNL2_OWN_NAME),
    #endif
};  

/**
 * @brief Function pointer for division with broadcasting operation
 * @ingroup backend_system
 */
divbroadcastingfn div_broadcasting;

/**
 * @brief Sets the backend for division with broadcasting operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for division with broadcasting
 */
void set_div_broadcasting_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(div_broadcasting_backends, div_broadcasting, backend_name);
}

#endif /** NNL2_BROADCASTING_DIV_H **/
