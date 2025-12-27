#ifndef NNL2_BROADCASTING_SUB_H
#define NNL2_BROADCASTING_SUB_H

/** @brief
 * Performs element-wise subtraction with broadcasting support
 *
 ** @param minuend
 * First tensor to subtract from
 *
 ** @param subtrahend
 * Second tensor to subtract
 * 
 ** @return
 * New tensor containing the result of subtraction
 *
 ** @note
 * Contains type conversion
 */
nnl2_tensor* naive_sub_broadcasting(nnl2_tensor* minuend, nnl2_tensor* subtrahend) {	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	// Additional checks for max safety level 
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX	
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(minuend, "Minuend tensor is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(subtrahend, "Subtrahend tensor is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(minuend->shape, "Minuend shape is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(subtrahend->shape, "Subtrahend shape is NULL", NULL);
	#endif
 
	// Calculate the total number of elements in each tensor
	size_t numel_minuend = nnl2_product(minuend->shape, minuend->rank);
	size_t numel_subtrahend = nnl2_product(subtrahend->shape, subtrahend->rank);
	
	// Getting the tensor data types
	nnl2_tensor_type minuend_dtype = minuend->dtype;
	nnl2_tensor_type subtrahend_dtype = subtrahend->dtype;
	
	nnl2_tensor_type winner_in_the_type_hierarchy = MAX(minuend_dtype, subtrahend_dtype);
	
	// Сreating a resultant tensor
	nnl2_tensor* result = nnl2_empty(minuend->shape, minuend->rank, winner_in_the_type_hierarchy);
	
	// Checking the possibility of broadcasting (numel_minuend must be a multiple of numel_subtrahend)
	if((numel_minuend % numel_subtrahend) == 0) {
		if(minuend_dtype == subtrahend_dtype) {
			// Handling the case where the data types match (more efficiently)
			switch(minuend_dtype) {
				case FLOAT64: {
					double* cast_minuend_data = (double*)minuend->data;
					double* cast_subtrahend_data = (double*)subtrahend->data;
					double* cast_result_data = (double*)result->data;
					
					for(size_t i = 0; i < (numel_minuend / numel_subtrahend); i++) {
						for(size_t j = 0; j < numel_subtrahend; j++) {
							cast_result_data[i * numel_subtrahend + j] = cast_minuend_data[i * numel_subtrahend + j] - cast_subtrahend_data[j];
						}
					}
					
					break;
				}
				
				case FLOAT32: {
					float* cast_minuend_data = (float*)minuend->data;
					float* cast_subtrahend_data = (float*)subtrahend->data;
					float* cast_result_data = (float*)result->data;
					
					for(size_t i = 0; i < (numel_minuend / numel_subtrahend); i++) {
						for(size_t j = 0; j < numel_subtrahend; j++) {
							cast_result_data[i * numel_subtrahend + j] = cast_minuend_data[i * numel_subtrahend + j] - cast_subtrahend_data[j];
						}
					}
					
					break;
				}
				
				case INT64: {
					int64_t* cast_minuend_data = (int64_t*)minuend->data;
					int64_t* cast_subtrahend_data = (int64_t*)subtrahend->data;
					int64_t* cast_result_data = (int64_t*)result->data;
					
					for(size_t i = 0; i < (numel_minuend / numel_subtrahend); i++) {
						for(size_t j = 0; j < numel_subtrahend; j++) {
							cast_result_data[i * numel_subtrahend + j] = cast_minuend_data[i * numel_subtrahend + j] - cast_subtrahend_data[j];
						}
					}
					
					break;
				}
				
				case INT32: {
					int32_t* cast_minuend_data = (int32_t*)minuend->data;
					int32_t* cast_subtrahend_data = (int32_t*)subtrahend->data;
					int32_t* cast_result_data = (int32_t*)result->data;
					
					for(size_t i = 0; i < (numel_minuend / numel_subtrahend); i++) {
						for(size_t j = 0; j < numel_subtrahend; j++) {
							cast_result_data[i * numel_subtrahend + j] = cast_minuend_data[i * numel_subtrahend + j] - cast_subtrahend_data[j];
						}
					}
					
					break;
				}
				
				default: {
					NNL2_TYPE_ERROR(minuend_dtype);
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
			size_t minuend_step = get_dtype_size(minuend_dtype);
			size_t subtrahend_step = get_dtype_size(subtrahend_dtype);
			
			switch(winner_in_the_type_hierarchy) {
				case FLOAT64: {
					double* cast_data_result = (double*)result->data;
					
					char* cast_minuend_data = (char*)minuend->data;
					char* cast_subtrahend_data =  (char*)subtrahend->data;
					
					for(size_t i = 0; i < (numel_minuend / numel_subtrahend); i++) {
						for(size_t j = 0; j < numel_subtrahend; j++) {
							// Get a pointer to minuend element, subtrahend element and convert its type
							void* elem_minuend = cast_minuend_data + (i * numel_subtrahend + j) * minuend_step;
							void* elem_subtrahend = cast_subtrahend_data + j * subtrahend_step; 
							
							cast_data_result[i * numel_subtrahend + j] = nnl2_convert_to_float64(elem_minuend, minuend_dtype) - nnl2_convert_to_float64(elem_subtrahend, subtrahend_dtype);
						}
					}
					
					break;
				}
				
				case FLOAT32: {
					float* cast_data_result = (float*)result->data;
					
					char* cast_minuend_data = (char*)minuend->data;
					char* cast_subtrahend_data =  (char*)subtrahend->data;
					
					for(size_t i = 0; i < (numel_minuend / numel_subtrahend); i++) {
						for(size_t j = 0; j < numel_subtrahend; j++) {
							void* elem_minuend = cast_minuend_data + (i * numel_subtrahend + j) * minuend_step;
							void* elem_subtrahend =  cast_subtrahend_data + j * subtrahend_step;
							
							cast_data_result[i * numel_subtrahend + j] = nnl2_convert_to_float32(elem_minuend, minuend_dtype) - nnl2_convert_to_float32(elem_subtrahend, subtrahend_dtype);
						}
					}
					
					break;
				}
				
				case INT64: {
					int64_t* cast_data_result = (int64_t*)result->data;
					
					char* cast_minuend_data = (char*)minuend->data;
					char* cast_subtrahend_data =  (char*)subtrahend->data;
					
					for(size_t i = 0; i < (numel_minuend / numel_subtrahend); i++) {					
						for(size_t j = 0; j < numel_subtrahend; j++) {
							void* elem_minuend = cast_minuend_data + (i * numel_subtrahend + j) * minuend_step;
							void* elem_subtrahend =  cast_subtrahend_data + j * subtrahend_step;
						
							cast_data_result[i * numel_subtrahend + j] = nnl2_convert_to_int64(elem_minuend, minuend_dtype) - nnl2_convert_to_int64(elem_subtrahend, subtrahend_dtype);
						}
					}
					
					break;
				}
				
				case INT32: {
					int32_t* cast_data_result = (int32_t*)result->data;
					
					char* cast_minuend_data = (char*)minuend->data;
					char* cast_subtrahend_data =  (char*)subtrahend->data;
					
					for(size_t i = 0; i < (numel_minuend / numel_subtrahend); i++) {					
						for(size_t j = 0; j < numel_subtrahend; j++) {
							void* elem_minuend = cast_minuend_data + (i * numel_subtrahend + j) * minuend_step;
							void* elem_subtrahend =  cast_subtrahend_data + j * subtrahend_step;
						
							cast_data_result[i * numel_subtrahend + j] = nnl2_convert_to_int32(elem_minuend, minuend_dtype) - nnl2_convert_to_int32(elem_subtrahend, subtrahend_dtype);
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
		NNL2_ERROR("Cannot broadcast subtrahend tensor");
		return NULL;
	}
}

#if defined(NNL2_PTHREAD_AVAILABLE) && defined(NNL2_AVX256_AVAILABLE) && TENSOR_MEM_ALIGNMENT == 32

/** @brief
 * Threshold for enabling parallel execution of the
 * broadcasting subtraction operation
 */
#define NNL2_SUB_BROADCASTING_PARALLEL_THRESHOLD 500000

///@{ [subbroadcasting_ptask]

typedef struct {
    nnl2_tensor_type dtype;         ///< Data type of result tensor
    bool aligned_minuend;           ///< Flag indicating if minuend data is 32-byte aligned
    bool aligned_subtrahend;        ///< Flag indicating if subtrahend data is 32-byte aligned  
    bool aligned_result;            ///< Flag indicating if result data is 32-byte aligned
    const void* minuend_data;       ///< Pointer to minuend tensor data (read-only)
    const void* subtrahend_data;    ///< Pointer to subtrahend tensor data (read-only, broadcasted)
    void* result_data;              ///< Pointer to result tensor data (mutable)
    size_t start;                   ///< Start block index for this thread's chunk
    size_t end;                     ///< End block index for this thread's chunk
    size_t numel_subtrahend;        ///< Number of elements in subtrahend tensor
    size_t broadcast_ratio;         ///< Broadcast ratio (numel_minuend / numel_subtrahend)
} subbroadcasting_ptask;

///@} [subbroadcasting_ptask]

/** @brief 
 * Worker function for parallel double precision broadcasting subtraction
 * 
 ** @param arg 
 * Pointer to subbroadcasting_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 */
void* nnl2_own_psub_broadcasting_float64(void* arg);

/** @brief
 * Worker function for parallel single precision broadcasting subtraction
 * 
 ** @param arg 
 * Pointer to subbroadcasting_ptask structure containing thread parameters  
 * 
 ** @return 
 * NULL (for pthread API compatibility)
 */
void* nnl2_own_psub_broadcasting_float32(void* arg);

/** @brief
 * Worker function for parallel int64 broadcasting subtraction
 * 
 ** @param arg 
 * Pointer to subbroadcasting_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 */
void* nnl2_own_psub_broadcasting_int64(void* arg);

/** @brief
 * Worker function for parallel integer broadcasting subtraction
 * 
 ** @param arg 
 * Pointer to subbroadcasting_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 */
void* nnl2_own_psub_broadcasting_int32(void* arg);

/** @brief
 * High-performance parallel implementation of broadcasting subtraction
 * 
 ** @param minuend
 * Pointer to minuend tensor
 *
 ** @param subtrahend
 * Pointer to subtrahend tensor (broadcasted)
 *
 ** @return
 * Pointer to a new tensor containing the result of the subtraction operation
 */
nnl2_tensor* nnl2_own_sub_broadcasting(nnl2_tensor* minuend, nnl2_tensor* subtrahend) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(minuend, "Minuend tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(subtrahend, "Subtrahend tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(minuend->shape, "Minuend shape is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(subtrahend->shape, "Subtrahend shape is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(minuend->data, "Minuend data is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(subtrahend->data, "Subtrahend data is NULL", NULL);
    #endif
    
    size_t numel_minuend = nnl2_product(minuend->shape, minuend->rank);
    size_t numel_subtrahend = nnl2_product(subtrahend->shape, subtrahend->rank);
    
    // Check broadcasting compatibility
    if((numel_minuend % numel_subtrahend) != 0) {
        NNL2_ERROR("Cannot broadcast subtrahend tensor");
        return NULL;
    }
    
    // Determine result data type
    nnl2_tensor_type result_dtype = MAX(minuend->dtype, subtrahend->dtype);
    
    // Create result tensor
    nnl2_tensor* result = nnl2_empty(minuend->shape, minuend->rank, result_dtype);
    if(result == NULL) {
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
            NNL2_ERROR("Failed to allocate result tensor");
        #endif
        return NULL;
    }
    
    if(numel_minuend == 0 || numel_subtrahend == 0) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return result;
    }
    
    // Fall back to naive implementation for small tensors or different dtypes
    if(numel_minuend < NNL2_SUB_BROADCASTING_PARALLEL_THRESHOLD || 
       minuend->dtype != subtrahend->dtype) {
        nnl2_tensor* naive_result = naive_sub_broadcasting(minuend, subtrahend);
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
    
    size_t broadcast_ratio = numel_minuend / numel_subtrahend;
    
    bool is_aligned_minuend = NNL2_IS_ALIGNED(minuend->data, NNL2_TENSOR_ALIGNMENT_32);
    bool is_aligned_subtrahend = NNL2_IS_ALIGNED(subtrahend->data, NNL2_TENSOR_ALIGNMENT_32);
    bool is_aligned_result = NNL2_IS_ALIGNED(result->data, NNL2_TENSOR_ALIGNMENT_32);
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!is_aligned_minuend) {
            NNL2_WARN("In nnl2_own sub broadcasting, minuend memory is not aligned to 32 bytes. Performance may be suboptimal");
        }
        
        if(!is_aligned_subtrahend) {
            NNL2_WARN("In nnl2_own sub broadcasting, subtrahend memory is not aligned to 32 bytes. Performance may be suboptimal");
        }
        
        if(!is_aligned_result) {
            NNL2_WARN("In nnl2_own sub broadcasting, result memory is not aligned to 32 bytes. Performance may be suboptimal");
        }
    #endif
    
    size_t num_threads = NNL2_NUM_THREADS;
    pthread_t threads[num_threads];
    subbroadcasting_ptask tasks[num_threads];
    
    // Calculate chunk size per thread for the outer loop
    size_t chunk = broadcast_ratio / num_threads;
    size_t remainder = broadcast_ratio % num_threads;
    
    // Initialize common task parameters
    for(size_t i = 0; i < num_threads; i++) {
        tasks[i].dtype = result_dtype;
        tasks[i].aligned_minuend = is_aligned_minuend;
        tasks[i].aligned_subtrahend = is_aligned_subtrahend;
        tasks[i].aligned_result = is_aligned_result;
        tasks[i].minuend_data = minuend->data;
        tasks[i].subtrahend_data = subtrahend->data;
        tasks[i].result_data = result->data;
        tasks[i].numel_subtrahend = numel_subtrahend;
        tasks[i].broadcast_ratio = broadcast_ratio;
    }
    
    size_t current_start = 0;
    for(size_t i = 0; i < num_threads; i++) {
        size_t current_chunk = chunk + (i < remainder ? 1 : 0);
        
        tasks[i].start = current_start;
        tasks[i].end = current_start + current_chunk;
        
        void* (*worker_func)(void*) = NULL;
        switch(result_dtype) {
            case FLOAT64: worker_func = nnl2_own_psub_broadcasting_float64; break;
            case FLOAT32: worker_func = nnl2_own_psub_broadcasting_float32; break;
            case INT32:   worker_func = nnl2_own_psub_broadcasting_int32;   break;
            case INT64:   worker_func = nnl2_own_psub_broadcasting_int64;   break;
            
            default: {
                NNL2_TYPE_ERROR(result_dtype);
                nnl2_free_tensor(result);
                return NULL;
            }
        }
        
        int status = pthread_create(&threads[i], NULL, worker_func, &tasks[i]);
        if(status != 0) {
            NNL2_THREAD_CREATE_ERROR(status, "nnl2_own_sub_broadcasting");
            num_threads = i;
            break;
        }
        
        current_start += current_chunk;
    }
    
    for(size_t i = 0; i < num_threads; i++) {
        int join_status = pthread_join(threads[i], NULL);
        if(join_status != 0) {
            NNL2_THREAD_JOIN_ERROR(join_status, "nnl2_own_sub_broadcasting");
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return result;
}

void* nnl2_own_psub_broadcasting_float64(void* arg) {
    subbroadcasting_ptask* task = (subbroadcasting_ptask*)arg;
    const double* minuend_data = (const double*)task->minuend_data;
    const double* subtrahend_data = (const double*)task->subtrahend_data;
    double* result_data = (double*)task->result_data;
    size_t start = task->start;
    size_t end = task->end;
    size_t numel_subtrahend = task->numel_subtrahend;
    
    for(size_t block = start; block < end; block++) {
        size_t base_idx = block * numel_subtrahend;
        
        __m256d v_subtrahend, v_minuend, v_result;
        size_t j = 0;
        
        if(task->aligned_minuend && task->aligned_subtrahend && task->aligned_result) {
            for(; j + 3 < numel_subtrahend; j += 4) {
                _mm_prefetch((char*)&subtrahend_data[j + 16], _MM_HINT_T0);
                _mm_prefetch((char*)&minuend_data[base_idx + j + 16], _MM_HINT_T0);
                
                v_subtrahend = _mm256_load_pd(&subtrahend_data[j]);
                v_minuend = _mm256_load_pd(&minuend_data[base_idx + j]);
                v_result = _mm256_sub_pd(v_minuend, v_subtrahend);
                _mm256_store_pd(&result_data[base_idx + j], v_result);
            }
        } else if(task->aligned_minuend && task->aligned_subtrahend) {
            for(; j + 3 < numel_subtrahend; j += 4) {
                _mm_prefetch((char*)&subtrahend_data[j + 16], _MM_HINT_T0);
                _mm_prefetch((char*)&minuend_data[base_idx + j + 16], _MM_HINT_T0);
                
                v_subtrahend = _mm256_load_pd(&subtrahend_data[j]);
                v_minuend = _mm256_load_pd(&minuend_data[base_idx + j]);
                v_result = _mm256_sub_pd(v_minuend, v_subtrahend);
                _mm256_storeu_pd(&result_data[base_idx + j], v_result);
            }
        } else if(task->aligned_minuend && task->aligned_result) {
            for(; j + 3 < numel_subtrahend; j += 4) {
                _mm_prefetch((char*)&subtrahend_data[j + 16], _MM_HINT_T0);
                _mm_prefetch((char*)&minuend_data[base_idx + j + 16], _MM_HINT_T0);
                
                v_subtrahend = _mm256_loadu_pd(&subtrahend_data[j]);
                v_minuend = _mm256_load_pd(&minuend_data[base_idx + j]);
                v_result = _mm256_sub_pd(v_minuend, v_subtrahend);
                _mm256_store_pd(&result_data[base_idx + j], v_result);
            }
        } else if(task->aligned_subtrahend && task->aligned_result) {
            for(; j + 3 < numel_subtrahend; j += 4) {
                _mm_prefetch((char*)&subtrahend_data[j + 16], _MM_HINT_T0);
                _mm_prefetch((char*)&minuend_data[base_idx + j + 16], _MM_HINT_T0);
                
                v_subtrahend = _mm256_load_pd(&subtrahend_data[j]);
                v_minuend = _mm256_loadu_pd(&minuend_data[base_idx + j]);
                v_result = _mm256_sub_pd(v_minuend, v_subtrahend);
                _mm256_store_pd(&result_data[base_idx + j], v_result);
            }
        } else if(task->aligned_minuend) {
            for(; j + 3 < numel_subtrahend; j += 4) {
                _mm_prefetch((char*)&subtrahend_data[j + 16], _MM_HINT_T0);
                _mm_prefetch((char*)&minuend_data[base_idx + j + 16], _MM_HINT_T0);
                
                v_subtrahend = _mm256_loadu_pd(&subtrahend_data[j]);
                v_minuend = _mm256_load_pd(&minuend_data[base_idx + j]);
                v_result = _mm256_sub_pd(v_minuend, v_subtrahend);
                _mm256_storeu_pd(&result_data[base_idx + j], v_result);
            }
        } else if(task->aligned_subtrahend) {
            for(; j + 3 < numel_subtrahend; j += 4) {
                _mm_prefetch((char*)&subtrahend_data[j + 16], _MM_HINT_T0);
                _mm_prefetch((char*)&minuend_data[base_idx + j + 16], _MM_HINT_T0);
                
                v_subtrahend = _mm256_load_pd(&subtrahend_data[j]);
                v_minuend = _mm256_loadu_pd(&minuend_data[base_idx + j]);
                v_result = _mm256_sub_pd(v_minuend, v_subtrahend);
                _mm256_storeu_pd(&result_data[base_idx + j], v_result);
            }
        } else if(task->aligned_result) {
            for(; j + 3 < numel_subtrahend; j += 4) {
                _mm_prefetch((char*)&subtrahend_data[j + 16], _MM_HINT_T0);
                _mm_prefetch((char*)&minuend_data[base_idx + j + 16], _MM_HINT_T0);
                
                v_subtrahend = _mm256_loadu_pd(&subtrahend_data[j]);
                v_minuend = _mm256_loadu_pd(&minuend_data[base_idx + j]);
                v_result = _mm256_sub_pd(v_minuend, v_subtrahend);
                _mm256_store_pd(&result_data[base_idx + j], v_result);
            }
        } else {
            for(; j + 3 < numel_subtrahend; j += 4) {
                _mm_prefetch((char*)&subtrahend_data[j + 16], _MM_HINT_T0);
                _mm_prefetch((char*)&minuend_data[base_idx + j + 16], _MM_HINT_T0);
                
                v_subtrahend = _mm256_loadu_pd(&subtrahend_data[j]);
                v_minuend = _mm256_loadu_pd(&minuend_data[base_idx + j]);
                v_result = _mm256_sub_pd(v_minuend, v_subtrahend);
                _mm256_storeu_pd(&result_data[base_idx + j], v_result);
            }
        }
        
        // Handle remaining elements
        for(; j < numel_subtrahend; j++) {
            result_data[base_idx + j] = minuend_data[base_idx + j] - subtrahend_data[j];
        }
    }
    
    return NULL;
}

void* nnl2_own_psub_broadcasting_float32(void* arg) {
    subbroadcasting_ptask* task = (subbroadcasting_ptask*)arg;
    const float* minuend_data = (const float*)task->minuend_data;
    const float* subtrahend_data = (const float*)task->subtrahend_data;
    float* result_data = (float*)task->result_data;
    size_t start = task->start;
    size_t end = task->end;
    size_t numel_subtrahend = task->numel_subtrahend;
    
    for(size_t block = start; block < end; block++) {
        size_t base_idx = block * numel_subtrahend;
        
        __m256 v_subtrahend, v_minuend, v_result;
        size_t j = 0;
        
        if(task->aligned_minuend && task->aligned_subtrahend && task->aligned_result) {
            for(; j + 7 < numel_subtrahend; j += 8) {
                _mm_prefetch((char*)&subtrahend_data[j + 32], _MM_HINT_T0);
                _mm_prefetch((char*)&minuend_data[base_idx + j + 32], _MM_HINT_T0);
                
                v_subtrahend = _mm256_load_ps(&subtrahend_data[j]);
                v_minuend = _mm256_load_ps(&minuend_data[base_idx + j]);
                v_result = _mm256_sub_ps(v_minuend, v_subtrahend);
                _mm256_store_ps(&result_data[base_idx + j], v_result);
            }
        } else if(task->aligned_minuend && task->aligned_subtrahend) {
            for(; j + 7 < numel_subtrahend; j += 8) {
                _mm_prefetch((char*)&subtrahend_data[j + 32], _MM_HINT_T0);
                _mm_prefetch((char*)&minuend_data[base_idx + j + 32], _MM_HINT_T0);
                
                v_subtrahend = _mm256_load_ps(&subtrahend_data[j]);
                v_minuend = _mm256_load_ps(&minuend_data[base_idx + j]);
                v_result = _mm256_sub_ps(v_minuend, v_subtrahend);
                _mm256_storeu_ps(&result_data[base_idx + j], v_result);
            }
        } else if(task->aligned_minuend && task->aligned_result) {
            for(; j + 7 < numel_subtrahend; j += 8) {
                _mm_prefetch((char*)&subtrahend_data[j + 32], _MM_HINT_T0);
                _mm_prefetch((char*)&minuend_data[base_idx + j + 32], _MM_HINT_T0);
                
                v_subtrahend = _mm256_loadu_ps(&subtrahend_data[j]);
                v_minuend = _mm256_load_ps(&minuend_data[base_idx + j]);
                v_result = _mm256_sub_ps(v_minuend, v_subtrahend);
                _mm256_store_ps(&result_data[base_idx + j], v_result);
            }
        } else if(task->aligned_subtrahend && task->aligned_result) {
            for(; j + 7 < numel_subtrahend; j += 8) {
                _mm_prefetch((char*)&subtrahend_data[j + 32], _MM_HINT_T0);
                _mm_prefetch((char*)&minuend_data[base_idx + j + 32], _MM_HINT_T0);
                
                v_subtrahend = _mm256_load_ps(&subtrahend_data[j]);
                v_minuend = _mm256_loadu_ps(&minuend_data[base_idx + j]);
                v_result = _mm256_sub_ps(v_minuend, v_subtrahend);
                _mm256_store_ps(&result_data[base_idx + j], v_result);
            }
        } else if(task->aligned_minuend) {
            for(; j + 7 < numel_subtrahend; j += 8) {
                _mm_prefetch((char*)&subtrahend_data[j + 32], _MM_HINT_T0);
                _mm_prefetch((char*)&minuend_data[base_idx + j + 32], _MM_HINT_T0);
                
                v_subtrahend = _mm256_loadu_ps(&subtrahend_data[j]);
                v_minuend = _mm256_load_ps(&minuend_data[base_idx + j]);
                v_result = _mm256_sub_ps(v_minuend, v_subtrahend);
                _mm256_storeu_ps(&result_data[base_idx + j], v_result);
            }
        } else if(task->aligned_subtrahend) {
            for(; j + 7 < numel_subtrahend; j += 8) {
                _mm_prefetch((char*)&subtrahend_data[j + 32], _MM_HINT_T0);
                _mm_prefetch((char*)&minuend_data[base_idx + j + 32], _MM_HINT_T0);
                
                v_subtrahend = _mm256_load_ps(&subtrahend_data[j]);
                v_minuend = _mm256_loadu_ps(&minuend_data[base_idx + j]);
                v_result = _mm256_sub_ps(v_minuend, v_subtrahend);
                _mm256_storeu_ps(&result_data[base_idx + j], v_result);
            }
        } else if(task->aligned_result) {
            for(; j + 7 < numel_subtrahend; j += 8) {
                _mm_prefetch((char*)&subtrahend_data[j + 32], _MM_HINT_T0);
                _mm_prefetch((char*)&minuend_data[base_idx + j + 32], _MM_HINT_T0);
                
                v_subtrahend = _mm256_loadu_ps(&subtrahend_data[j]);
                v_minuend = _mm256_loadu_ps(&minuend_data[base_idx + j]);
                v_result = _mm256_sub_ps(v_minuend, v_subtrahend);
                _mm256_store_ps(&result_data[base_idx + j], v_result);
            }
        } else {
            for(; j + 7 < numel_subtrahend; j += 8) {
                _mm_prefetch((char*)&subtrahend_data[j + 32], _MM_HINT_T0);
                _mm_prefetch((char*)&minuend_data[base_idx + j + 32], _MM_HINT_T0);
                
                v_subtrahend = _mm256_loadu_ps(&subtrahend_data[j]);
                v_minuend = _mm256_loadu_ps(&minuend_data[base_idx + j]);
                v_result = _mm256_sub_ps(v_minuend, v_subtrahend);
                _mm256_storeu_ps(&result_data[base_idx + j], v_result);
            }
        }
        
        // Handle remaining elements
        for(; j < numel_subtrahend; j++) {
            result_data[base_idx + j] = minuend_data[base_idx + j] - subtrahend_data[j];
        }
    }
    
    return NULL;
}

void* nnl2_own_psub_broadcasting_int64(void* arg) {
    subbroadcasting_ptask* task = (subbroadcasting_ptask*)arg;
    const int64_t* minuend_data = (const int64_t*)task->minuend_data;
    const int64_t* subtrahend_data = (const int64_t*)task->subtrahend_data;
    int64_t* result_data = (int64_t*)task->result_data;
    size_t start = task->start;
    size_t end = task->end;
    size_t numel_subtrahend = task->numel_subtrahend;
    
    for(size_t block = start; block < end; block++) {
        size_t base_idx = block * numel_subtrahend;
        
        for(size_t j = 0; j < numel_subtrahend; j++) {
            result_data[base_idx + j] = minuend_data[base_idx + j] - subtrahend_data[j];
        }
    }
    
    return NULL;
}

void* nnl2_own_psub_broadcasting_int32(void* arg) {
    subbroadcasting_ptask* task = (subbroadcasting_ptask*)arg;
    const int32_t* minuend_data = (const int32_t*)task->minuend_data;
    const int32_t* subtrahend_data = (const int32_t*)task->subtrahend_data;
    int32_t* result_data = (int32_t*)task->result_data;
    size_t start = task->start;
    size_t end = task->end;
    size_t numel_subtrahend = task->numel_subtrahend;
    
    // Integer subtraction uses scalar operations for precise arithmetic semantics
    for(size_t block = start; block < end; block++) {
        size_t base_idx = block * numel_subtrahend;
        
        for(size_t j = 0; j < numel_subtrahend; j++) {
            result_data[base_idx + j] = minuend_data[base_idx + j] - subtrahend_data[j];
        }
    }
    
    return NULL;
}

#endif

/**
 * @ingroup backend_system
 * @brief Backend implementations for subtraction with broadcasting
 * @details
 * Array follows the common backend registration pattern for subtraction
 * with broadcasting operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for subtraction with broadcasting
 *  - nnl2_own: High-performance implementation with AVX256, pthread and prefetching
 * 
 * @see nnl2_naive
 * @see naive_sub_broadcasting
 * @see nnl2_own_sub_broadcasting
 */
nnl2_runtime_implementation sub_broadcasting_backends[] = {
    REGISTER_BACKEND(naive_sub_broadcasting, nnl2_naive, NAIVE_BACKEND_NAME),
    
    #if defined(NNL2_PTHREAD_AVAILABLE) && defined(NNL2_AVX256_AVAILABLE) && TENSOR_MEM_ALIGNMENT == 32
        REGISTER_BACKEND(nnl2_own_sub_broadcasting, nnl2_own, NNL2_OWN_NAME),
    #endif
};

/**
 * @brief Function pointer for subtraction with broadcasting operation
 * @ingroup backend_system
 */
subbroadcastingfn sub_broadcasting;

/**
 * @brief Sets the backend for subtraction with broadcasting operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for subtraction with broadcasting
 */
void set_sub_broadcasting_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(sub_broadcasting_backends, sub_broadcasting, backend_name);
}

#endif /** NNL2_BROADCASTING_SUB_H **/
