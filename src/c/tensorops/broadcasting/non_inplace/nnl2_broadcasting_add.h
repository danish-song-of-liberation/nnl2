#ifndef NNL2_BROADCASTING_ADD_H
#define NNL2_BROADCASTING_ADD_H

/** @brief
 * Performs element-wise addition with broadcasting support
 *
 ** @param summand
 * First tensor to add
 *
 ** @param sumend
 * Second tensor to add
 * 
 ** @return
 * New tensor containing the result of addition
 *
 ** @note
 * Contains type conversion
 */
nnl2_tensor* naive_add_broadcasting(nnl2_tensor* summand, nnl2_tensor* sumend) {	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	// Additional checks for max safety level 
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(summand, "Summand tensor is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(sumend, "Sumend tensor is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(summand->shape, "Summand shape is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(sumend->shape, "Sumend shape is NULL", NULL);
	#endif
 
	// Calculate the total number of elements in each tensor
	size_t numel_summand = nnl2_product(summand->shape, summand->rank);
	size_t numel_sumend = nnl2_product(sumend->shape, sumend->rank);
	
	// Getting the tensor data types
	nnl2_tensor_type summand_dtype = summand->dtype;
	nnl2_tensor_type sumend_dtype = sumend->dtype;
	
	nnl2_tensor_type winner_in_the_type_hierarchy = MAX(summand_dtype, sumend_dtype);
	
	// Сreating a resultant tensor
	nnl2_tensor* result = nnl2_empty(summand->shape, summand->rank, winner_in_the_type_hierarchy);
	
	// Checking the possibility of broadcasting (numel_summand must be a multiple of numel_sumend)
	if((numel_summand % numel_sumend) == 0) {
		if(summand_dtype == sumend_dtype) {
			// Handling the case where the data types match (more efficiently)
			switch(summand_dtype) {
				case FLOAT64: {
					double* cast_summand_data = (double*)summand->data;
					double* cast_sumend_data = (double*)sumend->data;
					double* cast_result_data = (double*)result->data;
					
					for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
						for(size_t j = 0; j < numel_sumend; j++) {
							cast_result_data[i * numel_sumend + j] = cast_summand_data[i * numel_sumend + j] + cast_sumend_data[j];
						}
					}
					
					break;
				}
				
				case FLOAT32: {
					float* cast_summand_data = (float*)summand->data;
					float* cast_sumend_data = (float*)sumend->data;
					float* cast_result_data = (float*)result->data;
					
					for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
						for(size_t j = 0; j < numel_sumend; j++) {
							cast_result_data[i * numel_sumend + j] = cast_summand_data[i * numel_sumend + j] + cast_sumend_data[j];
						}
					}
					
					break;
				}
				
				case INT64: {
					int64_t* cast_summand_data = (int64_t*)summand->data;
					int64_t* cast_sumend_data = (int64_t*)sumend->data;
					int64_t* cast_result_data = (int64_t*)result->data;
					
					for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
						for(size_t j = 0; j < numel_sumend; j++) {
							cast_result_data[i * numel_sumend + j] = cast_summand_data[i * numel_sumend + j] + cast_sumend_data[j];
						}
					}
					
					break;
				}
				
				case INT32: {
					int32_t* cast_summand_data = (int32_t*)summand->data;
					int32_t* cast_sumend_data = (int32_t*)sumend->data;
					int32_t* cast_result_data = (int32_t*)result->data;
					
					for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
						for(size_t j = 0; j < numel_sumend; j++) {
							cast_result_data[i * numel_sumend + j] = cast_summand_data[i * numel_sumend + j] + cast_sumend_data[j];
						}
					}
					
					break;
				}
				
				default: {
					NNL2_TYPE_ERROR(summand_dtype);
					return NULL;
				}
			}
			
			#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
				NNL2_FUNC_EXIT();
			#endif
			
			return result;
		} 
		
		else {
			// Handling a case with different data types (conversion required)
			size_t summand_step = get_dtype_size(summand_dtype);
			size_t sumend_step = get_dtype_size(sumend_dtype);
			
			switch(winner_in_the_type_hierarchy) {
				case FLOAT64: {
					double* cast_data_result = (double*)result->data;
					
					char* cast_summand_data = (char*)summand->data;
					char* cast_sumend_data =  (char*)sumend->data;
					
					for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
						for(size_t j = 0; j < numel_sumend; j++) {
							// Get a pointer to summand element, sumend element and convert its type
							void* elem_summand = cast_summand_data + (i * numel_sumend + j) * summand_step;
							void* elem_sumend = cast_sumend_data + j * sumend_step; 
							
							cast_data_result[i * numel_sumend + j] = nnl2_convert_to_float64(elem_summand, summand_dtype) + nnl2_convert_to_float64(elem_sumend, sumend_dtype);
						}
					}
					
					break;
				}
				
				case FLOAT32: {
					float* cast_data_result = (float*)result->data;
					
					char* cast_summand_data = (char*)summand->data;
					char* cast_sumend_data =  (char*)sumend->data;
					
					for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
						for(size_t j = 0; j < numel_sumend; j++) {
							void* elem_summand = cast_summand_data + (i * numel_sumend + j) * summand_step;
							void* elem_sumend =  cast_sumend_data + j * sumend_step;
							
							cast_data_result[i * numel_sumend + j] = nnl2_convert_to_float32(elem_summand, summand_dtype) + nnl2_convert_to_float32(elem_sumend, sumend_dtype);
						}
					}
					
					break;
				}
				
				case INT64: {
					int64_t* cast_data_result = (int64_t*)result->data;
					
					char* cast_summand_data = (char*)summand->data;
					char* cast_sumend_data =  (char*)sumend->data;
					
					for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {					
						for(size_t j = 0; j < numel_sumend; j++) {
							void* elem_summand = cast_summand_data + (i * numel_sumend + j) * summand_step;
							void* elem_sumend =  cast_sumend_data + j * sumend_step;
						
							cast_data_result[i * numel_sumend + j] = nnl2_convert_to_int64(elem_summand, summand_dtype) + nnl2_convert_to_int64(elem_sumend, sumend_dtype);
						}
					}
					
					break;
				}
				
				case INT32: {
					int32_t* cast_data_result = (int32_t*)result->data;
					
					char* cast_summand_data = (char*)summand->data;
					char* cast_sumend_data =  (char*)sumend->data;
					
					for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {					
						for(size_t j = 0; j < numel_sumend; j++) {
							void* elem_summand = cast_summand_data + (i * numel_sumend + j) * summand_step;
							void* elem_sumend =  cast_sumend_data + j * sumend_step;
						
							cast_data_result[i * numel_sumend + j] = nnl2_convert_to_int32(elem_summand, summand_dtype) + nnl2_convert_to_int32(elem_sumend, sumend_dtype);
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
	} 
	
	else {
		NNL2_ERROR("Cannot broadcast sumend tensor");
		return NULL;
	}
}

#if defined(NNL2_PTHREAD_AVAILABLE) && defined(NNL2_AVX256_AVAILABLE) && TENSOR_MEM_ALIGNMENT == 32

/** @brief
 * Threshold for enabling parallel execution of the
 * broadcasting addition operation
 */
#define NNL2_ADD_BROADCASTING_PARALLEL_THRESHOLD 500000

/** @brief 
 * Worker function for parallel double precision broadcasting addition
 * 
 ** @param arg 
 * Pointer to addbroadcasting_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 */
void* nnl2_own_padd_broadcasting_float64(void* arg);

/** @brief
 * Worker function for parallel single precision broadcasting addition
 * 
 ** @param arg 
 * Pointer to addbroadcasting_ptask structure containing thread parameters  
 * 
 ** @return 
 * NULL (for pthread API compatibility)
 */
void* nnl2_own_padd_broadcasting_float32(void* arg);

/** @brief
 * Worker function for parallel int64 broadcasting addition
 * 
 ** @param arg 
 * Pointer to addbroadcasting_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 */
void* nnl2_own_padd_broadcasting_int64(void* arg);

/** @brief
 * Worker function for parallel integer broadcasting addition
 * 
 ** @param arg 
 * Pointer to addbroadcasting_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 */
void* nnl2_own_padd_broadcasting_int32(void* arg);

/** @brief
 * High-performance parallel implementation of broadcasting addition
 * 
 ** @param summand
 * Pointer to summand tensor
 *
 ** @param sumend
 * Pointer to sumend tensor (broadcasted)
 *
 ** @return
 * Pointer to a new tensor containing the result of the addition operation
 */
nnl2_tensor* nnl2_own_add_broadcasting(nnl2_tensor* summand, nnl2_tensor* sumend) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(summand, "Summand tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(sumend, "Sumend tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(summand->shape, "Summand shape is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(sumend->shape, "Sumend shape is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(summand->data, "Summand data is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(sumend->data, "Sumend data is NULL", NULL);
    #endif
    
    size_t numel_summand = nnl2_product(summand->shape, summand->rank);
    size_t numel_sumend = nnl2_product(sumend->shape, sumend->rank);
    
    // Check broadcasting compatibility
    if((numel_summand % numel_sumend) != 0) {
        NNL2_ERROR("Cannot broadcast sumend tensor");
        return NULL;
    }
    
    // Determine result data type
    nnl2_tensor_type result_dtype = MAX(summand->dtype, sumend->dtype);
    
    // Create result tensor
    nnl2_tensor* result = nnl2_empty(summand->shape, summand->rank, result_dtype);
    if(result == NULL) {
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
            NNL2_ERROR("Failed to allocate result tensor");
        #endif
        return NULL;
    }
    
    if(numel_summand == 0 || numel_sumend == 0) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return result;
    }
    
    // Fall back to naive implementation for small tensors or different dtypes
    if(numel_summand < NNL2_ADD_BROADCASTING_PARALLEL_THRESHOLD || 
       summand->dtype != sumend->dtype) {
        nnl2_tensor* naive_result = naive_add_broadcasting(summand, sumend);
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
    
    size_t broadcast_ratio = numel_summand / numel_sumend;
    
    bool is_aligned_summand = NNL2_IS_ALIGNED(summand->data, NNL2_TENSOR_ALIGNMENT_32);
    bool is_aligned_sumend = NNL2_IS_ALIGNED(sumend->data, NNL2_TENSOR_ALIGNMENT_32);
    bool is_aligned_result = NNL2_IS_ALIGNED(result->data, NNL2_TENSOR_ALIGNMENT_32);
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!is_aligned_summand) {
            NNL2_WARN("In nnl2_own add broadcasting, summand memory is not aligned to 32 bytes. Performance may be suboptimal");
        }
        
        if(!is_aligned_sumend) {
            NNL2_WARN("In nnl2_own add broadcasting, sumend memory is not aligned to 32 bytes. Performance may be suboptimal");
        }
        
        if(!is_aligned_result) {
            NNL2_WARN("In nnl2_own add broadcasting, result memory is not aligned to 32 bytes. Performance may be suboptimal");
        }
    #endif
    
    size_t num_threads = NNL2_NUM_THREADS;
    pthread_t threads[num_threads];
    addbroadcasting_ptask tasks[num_threads];
    
    // Calculate chunk size per thread for the outer loop
    size_t chunk = broadcast_ratio / num_threads;
    size_t remainder = broadcast_ratio % num_threads;
    
    // Initialize common task parameters
    for(size_t i = 0; i < num_threads; i++) {
        tasks[i].dtype = result_dtype;
        tasks[i].aligned_summand = is_aligned_summand;
        tasks[i].aligned_sumend = is_aligned_sumend;
        tasks[i].aligned_result = is_aligned_result;
        tasks[i].summand_data = summand->data;
        tasks[i].sumend_data = sumend->data;
        tasks[i].result_data = result->data;
        tasks[i].numel_sumend = numel_sumend;
        tasks[i].broadcast_ratio = broadcast_ratio;
    }
    
    size_t current_start = 0;
    for(size_t i = 0; i < num_threads; i++) {
        size_t current_chunk = chunk + (i < remainder ? 1 : 0);
        
        tasks[i].start = current_start;
        tasks[i].end = current_start + current_chunk;
        
        void* (*worker_func)(void*) = NULL;
        switch(result_dtype) {
            case FLOAT64: worker_func = nnl2_own_padd_broadcasting_float64; break;
            case FLOAT32: worker_func = nnl2_own_padd_broadcasting_float32; break;
            case INT32:   worker_func = nnl2_own_padd_broadcasting_int32;   break;
            case INT64:   worker_func = nnl2_own_padd_broadcasting_int64;   break;
            
            default: {
                NNL2_TYPE_ERROR(result_dtype);
                nnl2_free_tensor(result);
                return NULL;
            }
        }
        
        int status = pthread_create(&threads[i], NULL, worker_func, &tasks[i]);
        if(status != 0) {
            NNL2_THREAD_CREATE_ERROR(status, "nnl2_own_add_broadcasting");
            num_threads = i;
            break;
        }
        
        current_start += current_chunk;
    }
    
    for(size_t i = 0; i < num_threads; i++) {
        int join_status = pthread_join(threads[i], NULL);
        if(join_status != 0) {
            NNL2_THREAD_JOIN_ERROR(join_status, "nnl2_own_add_broadcasting");
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return result;
}

void* nnl2_own_padd_broadcasting_float64(void* arg) {
    addbroadcasting_ptask* task = (addbroadcasting_ptask*)arg;
    const double* summand_data = (const double*)task->summand_data;
    const double* sumend_data = (const double*)task->sumend_data;
    double* result_data = (double*)task->result_data;
    size_t start = task->start;
    size_t end = task->end;
    size_t numel_sumend = task->numel_sumend;
    
    for(size_t block = start; block < end; block++) {
        size_t base_idx = block * numel_sumend;
        
        __m256d v_sumend, v_summand, v_result;
        size_t j = 0;
        
        if(task->aligned_summand && task->aligned_sumend && task->aligned_result) {
            for(; j + 3 < numel_sumend; j += 4) {
                _mm_prefetch((char*)&sumend_data[j + 16], _MM_HINT_T0);
                _mm_prefetch((char*)&summand_data[base_idx + j + 16], _MM_HINT_T0);
                
                v_sumend = _mm256_load_pd(&sumend_data[j]);
                v_summand = _mm256_load_pd(&summand_data[base_idx + j]);
                v_result = _mm256_add_pd(v_summand, v_sumend);
                _mm256_store_pd(&result_data[base_idx + j], v_result);
            }
        } else if(task->aligned_summand && task->aligned_sumend) {
            for(; j + 3 < numel_sumend; j += 4) {
                _mm_prefetch((char*)&sumend_data[j + 16], _MM_HINT_T0);
                _mm_prefetch((char*)&summand_data[base_idx + j + 16], _MM_HINT_T0);
                
                v_sumend = _mm256_load_pd(&sumend_data[j]);
                v_summand = _mm256_load_pd(&summand_data[base_idx + j]);
                v_result = _mm256_add_pd(v_summand, v_sumend);
                _mm256_storeu_pd(&result_data[base_idx + j], v_result);
            }
        } else if(task->aligned_summand && task->aligned_result) {
            for(; j + 3 < numel_sumend; j += 4) {
                _mm_prefetch((char*)&sumend_data[j + 16], _MM_HINT_T0);
                _mm_prefetch((char*)&summand_data[base_idx + j + 16], _MM_HINT_T0);
                
                v_sumend = _mm256_loadu_pd(&sumend_data[j]);
                v_summand = _mm256_load_pd(&summand_data[base_idx + j]);
                v_result = _mm256_add_pd(v_summand, v_sumend);
                _mm256_store_pd(&result_data[base_idx + j], v_result);
            }
        } else if(task->aligned_sumend && task->aligned_result) {
            for(; j + 3 < numel_sumend; j += 4) {
                _mm_prefetch((char*)&sumend_data[j + 16], _MM_HINT_T0);
                _mm_prefetch((char*)&summand_data[base_idx + j + 16], _MM_HINT_T0);
                
                v_sumend = _mm256_load_pd(&sumend_data[j]);
                v_summand = _mm256_loadu_pd(&summand_data[base_idx + j]);
                v_result = _mm256_add_pd(v_summand, v_sumend);
                _mm256_store_pd(&result_data[base_idx + j], v_result);
            }
        } else if(task->aligned_summand) {
            for(; j + 3 < numel_sumend; j += 4) {
                _mm_prefetch((char*)&sumend_data[j + 16], _MM_HINT_T0);
                _mm_prefetch((char*)&summand_data[base_idx + j + 16], _MM_HINT_T0);
                
                v_sumend = _mm256_loadu_pd(&sumend_data[j]);
                v_summand = _mm256_load_pd(&summand_data[base_idx + j]);
                v_result = _mm256_add_pd(v_summand, v_sumend);
                _mm256_storeu_pd(&result_data[base_idx + j], v_result);
            }
        } else if(task->aligned_sumend) {
            for(; j + 3 < numel_sumend; j += 4) {
                _mm_prefetch((char*)&sumend_data[j + 16], _MM_HINT_T0);
                _mm_prefetch((char*)&summand_data[base_idx + j + 16], _MM_HINT_T0);
                
                v_sumend = _mm256_load_pd(&sumend_data[j]);
                v_summand = _mm256_loadu_pd(&summand_data[base_idx + j]);
                v_result = _mm256_add_pd(v_summand, v_sumend);
                _mm256_storeu_pd(&result_data[base_idx + j], v_result);
            }
        } else if(task->aligned_result) {
            for(; j + 3 < numel_sumend; j += 4) {
                _mm_prefetch((char*)&sumend_data[j + 16], _MM_HINT_T0);
                _mm_prefetch((char*)&summand_data[base_idx + j + 16], _MM_HINT_T0);
                
                v_sumend = _mm256_loadu_pd(&sumend_data[j]);
                v_summand = _mm256_loadu_pd(&summand_data[base_idx + j]);
                v_result = _mm256_add_pd(v_summand, v_sumend);
                _mm256_store_pd(&result_data[base_idx + j], v_result);
            }
        } else {
            for(; j + 3 < numel_sumend; j += 4) {
                _mm_prefetch((char*)&sumend_data[j + 16], _MM_HINT_T0);
                _mm_prefetch((char*)&summand_data[base_idx + j + 16], _MM_HINT_T0);
                
                v_sumend = _mm256_loadu_pd(&sumend_data[j]);
                v_summand = _mm256_loadu_pd(&summand_data[base_idx + j]);
                v_result = _mm256_add_pd(v_summand, v_sumend);
                _mm256_storeu_pd(&result_data[base_idx + j], v_result);
            }
        }
        
        // Handle remaining elements
        for(; j < numel_sumend; j++) {
            result_data[base_idx + j] = summand_data[base_idx + j] + sumend_data[j];
        }
    }
    
    return NULL;
}

void* nnl2_own_padd_broadcasting_float32(void* arg) {
    addbroadcasting_ptask* task = (addbroadcasting_ptask*)arg;
    const float* summand_data = (const float*)task->summand_data;
    const float* sumend_data = (const float*)task->sumend_data;
    float* result_data = (float*)task->result_data;
    size_t start = task->start;
    size_t end = task->end;
    size_t numel_sumend = task->numel_sumend;
    
    for(size_t block = start; block < end; block++) {
        size_t base_idx = block * numel_sumend;
        
        __m256 v_sumend, v_summand, v_result;
        size_t j = 0;
        
        if(task->aligned_summand && task->aligned_sumend && task->aligned_result) {
            for(; j + 7 < numel_sumend; j += 8) {
                _mm_prefetch((char*)&sumend_data[j + 32], _MM_HINT_T0);
                _mm_prefetch((char*)&summand_data[base_idx + j + 32], _MM_HINT_T0);
                
                v_sumend = _mm256_load_ps(&sumend_data[j]);
                v_summand = _mm256_load_ps(&summand_data[base_idx + j]);
                v_result = _mm256_add_ps(v_summand, v_sumend);
                _mm256_store_ps(&result_data[base_idx + j], v_result);
            }
        } else if(task->aligned_summand && task->aligned_sumend) {
            for(; j + 7 < numel_sumend; j += 8) {
                _mm_prefetch((char*)&sumend_data[j + 32], _MM_HINT_T0);
                _mm_prefetch((char*)&summand_data[base_idx + j + 32], _MM_HINT_T0);
                
                v_sumend = _mm256_load_ps(&sumend_data[j]);
                v_summand = _mm256_load_ps(&summand_data[base_idx + j]);
                v_result = _mm256_add_ps(v_summand, v_sumend);
                _mm256_storeu_ps(&result_data[base_idx + j], v_result);
            }
        } else if(task->aligned_summand && task->aligned_result) {
            for(; j + 7 < numel_sumend; j += 8) {
                _mm_prefetch((char*)&sumend_data[j + 32], _MM_HINT_T0);
                _mm_prefetch((char*)&summand_data[base_idx + j + 32], _MM_HINT_T0);
                
                v_sumend = _mm256_loadu_ps(&sumend_data[j]);
                v_summand = _mm256_load_ps(&summand_data[base_idx + j]);
                v_result = _mm256_add_ps(v_summand, v_sumend);
                _mm256_store_ps(&result_data[base_idx + j], v_result);
            }
        } else if(task->aligned_sumend && task->aligned_result) {
            for(; j + 7 < numel_sumend; j += 8) {
                _mm_prefetch((char*)&sumend_data[j + 32], _MM_HINT_T0);
                _mm_prefetch((char*)&summand_data[base_idx + j + 32], _MM_HINT_T0);
                
                v_sumend = _mm256_load_ps(&sumend_data[j]);
                v_summand = _mm256_loadu_ps(&summand_data[base_idx + j]);
                v_result = _mm256_add_ps(v_summand, v_sumend);
                _mm256_store_ps(&result_data[base_idx + j], v_result);
            }
        } else if(task->aligned_summand) {
            for(; j + 7 < numel_sumend; j += 8) {
                _mm_prefetch((char*)&sumend_data[j + 32], _MM_HINT_T0);
                _mm_prefetch((char*)&summand_data[base_idx + j + 32], _MM_HINT_T0);
                
                v_sumend = _mm256_loadu_ps(&sumend_data[j]);
                v_summand = _mm256_load_ps(&summand_data[base_idx + j]);
                v_result = _mm256_add_ps(v_summand, v_sumend);
                _mm256_storeu_ps(&result_data[base_idx + j], v_result);
            }
        } else if(task->aligned_sumend) {
            for(; j + 7 < numel_sumend; j += 8) {
                _mm_prefetch((char*)&sumend_data[j + 32], _MM_HINT_T0);
                _mm_prefetch((char*)&summand_data[base_idx + j + 32], _MM_HINT_T0);
                
                v_sumend = _mm256_load_ps(&sumend_data[j]);
                v_summand = _mm256_loadu_ps(&summand_data[base_idx + j]);
                v_result = _mm256_add_ps(v_summand, v_sumend);
                _mm256_storeu_ps(&result_data[base_idx + j], v_result);
            }
        } else if(task->aligned_result) {
            for(; j + 7 < numel_sumend; j += 8) {
                _mm_prefetch((char*)&sumend_data[j + 32], _MM_HINT_T0);
                _mm_prefetch((char*)&summand_data[base_idx + j + 32], _MM_HINT_T0);
                
                v_sumend = _mm256_loadu_ps(&sumend_data[j]);
                v_summand = _mm256_loadu_ps(&summand_data[base_idx + j]);
                v_result = _mm256_add_ps(v_summand, v_sumend);
                _mm256_store_ps(&result_data[base_idx + j], v_result);
            }
        } else {
            for(; j + 7 < numel_sumend; j += 8) {
                _mm_prefetch((char*)&sumend_data[j + 32], _MM_HINT_T0);
                _mm_prefetch((char*)&summand_data[base_idx + j + 32], _MM_HINT_T0);
                
                v_sumend = _mm256_loadu_ps(&sumend_data[j]);
                v_summand = _mm256_loadu_ps(&summand_data[base_idx + j]);
                v_result = _mm256_add_ps(v_summand, v_sumend);
                _mm256_storeu_ps(&result_data[base_idx + j], v_result);
            }
        }
        
        // Handle remaining elements
        for(; j < numel_sumend; j++) {
            result_data[base_idx + j] = summand_data[base_idx + j] + sumend_data[j];
        }
    }
    
    return NULL;
}

void* nnl2_own_padd_broadcasting_int64(void* arg) {
    addbroadcasting_ptask* task = (addbroadcasting_ptask*)arg;
    const int64_t* summand_data = (const int64_t*)task->summand_data;
    const int64_t* sumend_data = (const int64_t*)task->sumend_data;
    int64_t* result_data = (int64_t*)task->result_data;
    size_t start = task->start;
    size_t end = task->end;
    size_t numel_sumend = task->numel_sumend;
    
    for(size_t block = start; block < end; block++) {
        size_t base_idx = block * numel_sumend;
        
        for(size_t j = 0; j < numel_sumend; j++) {
            result_data[base_idx + j] = summand_data[base_idx + j] + sumend_data[j];
        }
    }
    
    return NULL;
}

void* nnl2_own_padd_broadcasting_int32(void* arg) {
    addbroadcasting_ptask* task = (addbroadcasting_ptask*)arg;
    const int32_t* summand_data = (const int32_t*)task->summand_data;
    const int32_t* sumend_data = (const int32_t*)task->sumend_data;
    int32_t* result_data = (int32_t*)task->result_data;
    size_t start = task->start;
    size_t end = task->end;
    size_t numel_sumend = task->numel_sumend;
    
    // Integer addition uses scalar operations for precise arithmetic semantics
    for(size_t block = start; block < end; block++) {
        size_t base_idx = block * numel_sumend;
        
        for(size_t j = 0; j < numel_sumend; j++) {
            result_data[base_idx + j] = summand_data[base_idx + j] + sumend_data[j];
        }
    }
    
    return NULL;
}

#endif

/**
 * @ingroup backend_system
 * @brief Backend implementations for addition with broadcasting
 * @details
 * Array follows the common backend registration pattern for addition
 * with broadcasting operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for addition with broadcasting
 *  - nnl2_own: High-performance implementation with AVX256, pthread and prefetching
 * 
 * @see nnl2_naive
 * @see naive_add_broadcasting
 * @see nnl2_own_add_broadcasting
 */
nnl2_runtime_implementation add_broadcasting_backends[] = {
    REGISTER_BACKEND(naive_add_broadcasting, nnl2_naive, NAIVE_BACKEND_NAME),
    
    #if defined(NNL2_PTHREAD_AVAILABLE) && defined(NNL2_AVX256_AVAILABLE) && TENSOR_MEM_ALIGNMENT == 32
        REGISTER_BACKEND(nnl2_own_add_broadcasting, nnl2_own, NNL2_OWN_NAME),
    #endif
};

/**
 * @brief Function pointer for addition with broadcasting operation
 * @ingroup backend_system
 */
addbroadcastingfn add_broadcasting;

/**
 * @brief Sets the backend for addition with broadcasting operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for addition with broadcasting
 * @see SET_BACKEND_BY_NAME
 * @see add_broadcasting_backends
 */
void set_add_broadcasting_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(add_broadcasting_backends, add_broadcasting, backend_name);
}

#endif /** NNL2_BROADCASTING_ADD_H **/
