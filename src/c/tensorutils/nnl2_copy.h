#ifndef NNL2_COPY_H
#define NNL2_COPY_H

/** @brief
 * Creates a copy of a tensor with possible data type conversion
 *
 ** @param tensor
 * Pointer to the source tensor for copying
 *
 ** @param copy_type
 * The target data type for the copy
 *
 ** @return 
 * A pointer to a new tensor copy, or NULL if an error occurs
 *
 ** @note
 * Can perform additional checks depending on the safety level
 *
 ** @see nnl2_empty
 ** @see nnl2_convert_to_float64
 ** @see nnl2_convert_to_float32
 ** @see nnl2_convert_to_int32
 **/
Tensor* naive_copy(Tensor* tensor, TensorType copy_type) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Additional checks depending on the safety level
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		// NULL checks
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor, "Passed tensor is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor->data, "Tensor data is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor->shape, "Tensor shape is NULL", NULL);
	#endif
	
	TensorType dtype = tensor->dtype;
	
	size_t total_elems = product(tensor->shape, tensor->rank);
	Tensor* result;
	
	if(dtype == copy_type) {
		result = nnl2_empty(tensor->shape, tensor->rank, dtype);
		
		// Element-by-element copying based on data type
		switch(dtype) {
			case FLOAT64: {
				double* cast_data_original = (double*)tensor->data;
				double* cast_data_copy = (double*)result->data;	
				for(size_t it = 0; it < total_elems; it++) cast_data_copy[it] = cast_data_original[it];
				break;
			}
			
			case FLOAT32: {
				float* cast_data_original = (float*)tensor->data;
				float* cast_data_copy = (float*)result->data;	
				for(size_t it = 0; it < total_elems; it++) cast_data_copy[it] = cast_data_original[it];
				break;
			}
			
			case INT32: {
				int32_t* cast_data_original = (int32_t*)tensor->data;
				int32_t* cast_data_copy = (int32_t*)result->data;	
				for(size_t it = 0; it < total_elems; it++) cast_data_copy[it] = cast_data_original[it];
				break;
			} 
			
			default: {
				NNL2_TYPE_ERROR(dtype);
				return NULL;
			}
		}
	} else {
		// Create a tensor with the target data type
		result = nnl2_empty(tensor->shape, tensor->rank, copy_type);
		
		// Data conversion and copying
		switch(copy_type) {
			case FLOAT64: {
				double* cast_data_copy = (double*)result->data;
				
				for(size_t it = 0; it < total_elems; it++) {
					// Getting a pointer to the current element of the source tensor
					void* original_elem = (char*)tensor->data + it * get_dtype_size(dtype);
					
					// Convert and copy the element
					cast_data_copy[it] = nnl2_convert_to_float64(original_elem, dtype);
				}
				
				break;
			}
			
			case FLOAT32: {
				float* cast_data_copy = (float*)result->data;
				
				for(size_t it = 0; it < total_elems; it++) {
					// Getting a pointer to the current element of the source tensor
					void* original_elem = (char*)tensor->data + it * get_dtype_size(dtype);
					
					// Convert and copy the element
					cast_data_copy[it] = nnl2_convert_to_float32(original_elem, dtype);
				}
				
				break;
			}
			
			case INT32: {
				int32_t* cast_data_copy = (int32_t*)result->data;
				
				for(size_t it = 0; it < total_elems; it++) {
					// Getting a pointer to the current element of the source tensor
					void* original_elem = (char*)tensor->data + it * get_dtype_size(dtype);
					
					// Convert and copy the element
					cast_data_copy[it] = nnl2_convert_to_int32(original_elem, dtype);
				}
				
				break;
			}
			
			default: {
				NNL2_TYPE_ERROR(copy_type);
				return NULL;
			}
		}
	}

	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif

	return result;
}

#ifdef NNL2_PTHREAD_AVAILABLE

/** @brief
 * Threshold for enabling parallel execution of copy operation
 */
#define NNL2_COPY_PARALLEL_THRESHOLD 1000000

/** @brief
 * Worker function for parallel double precision copy (same type)
 * 
 ** @param arg 
 * Pointer to copy_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 */
void* nnl2_own_pcopy_float64(void* arg);

/** @brief
 * Worker function for parallel single precision copy (same type)
 * 
 ** @param arg 
 * Pointer to copy_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 */
void* nnl2_own_pcopy_float32(void* arg);

/** @brief
 * Worker function for parallel integer copy (same type)
 * 
 ** @param arg 
 * Pointer to copy_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 */
void* nnl2_own_pcopy_int32(void* arg);

/** @brief
 * Worker function for parallel copy with type conversion
 * 
 ** @param arg 
 * Pointer to copy_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 */
void* nnl2_own_pcopy_convert(void* arg);

/** @brief
 * High-performance parallel implementation of tensor copy
 * 
 ** @param tensor 
 * Pointer to source tensor
 *
 ** @param copy_type
 * Target data type for the copy
 * 
 ** @return
 * Pointer to new tensor copy, or NULL if error occurs
 *
 ** @details
 * Combines AVX256 vectorization, multi-threading with pthread, and cache
 * prefetching for maximum performance on modern CPU architectures.
 * Automatically selects optimal thread count and chunk sizes.
 * 
 ** @note
 * Uses NNL2_NUM_THREADS for parallelization configuration
 * Falls back to naive implementation for small tensors or type conversion
 * 
 ** @warning
 * Requires pthread support and AVX256 capable CPU
 */
Tensor* nnl2_own_copy(Tensor* tensor, TensorType copy_type) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks at the maximum safety level
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor, "Tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor->data, "Tensor data is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor->shape, "Tensor shape is NULL", NULL);
    #endif
    
    size_t total_elems = product(tensor->shape, tensor->rank);
    if(total_elems == 0) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return nnl2_empty(tensor->shape, tensor->rank, copy_type);
    }
    
    TensorType dtype = tensor->dtype;
    Tensor* result = nnl2_empty(tensor->shape, tensor->rank, copy_type);
    
    if(result == NULL) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return NULL;
    }
    
    // Fallback to naive implementation for small tensors or type conversion
    if(total_elems < NNL2_COPY_PARALLEL_THRESHOLD || dtype != copy_type) {
        // For type conversion or small tensors, use optimized sequential approach
        if(dtype == copy_type) {
            // Same type - use memcpy for small tensors
            size_t bytes = total_elems * get_dtype_size(dtype);
            memcpy(result->data, tensor->data, bytes);
        } else {
            // Type conversion - fallback to naive
            naive_copy(tensor, copy_type);
        }
        
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return result;
    }
    
    bool src_aligned = NNL2_IS_ALIGNED(tensor->data, NNL2_TENSOR_ALIGNMENT_32);
    bool dst_aligned = NNL2_IS_ALIGNED(result->data, NNL2_TENSOR_ALIGNMENT_32);
    bool is_aligned = src_aligned && dst_aligned;
    
    // Warning for unaligned memory in safety modes
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!is_aligned) {
            NNL2_WARN("In nnl2_own_copy, tensor memory is not aligned to 32 bytes. Performance may be suboptimal");
        }
    #endif
    
    size_t num_threads = NNL2_NUM_THREADS;
    pthread_t threads[num_threads];
    copy_ptask tasks[num_threads];
    
    // Calculate optimal chunk sizes with load balancing
    size_t chunk = total_elems / num_threads;
    size_t remainder = total_elems % num_threads;
    
    // Configure tasks
    for (size_t i = 0; i < num_threads; i++) {
        tasks[i].dtype = dtype;
        tasks[i].target_dtype = copy_type;
        tasks[i].aligned = is_aligned;
        tasks[i].src_data = tensor->data;
        tasks[i].dst_data = result->data;
    }
    
    size_t current_start = 0;
    for (size_t i = 0; i < num_threads; i++) {
        size_t current_chunk = chunk + (i < remainder ? 1 : 0);
        
        // Configure task for this thread
        tasks[i].start = current_start;
        tasks[i].end = current_start + current_chunk;
        
        // Select appropriate worker function based on data type
        void* (*worker_func)(void*) = NULL;
        switch(dtype) {
            case FLOAT64: worker_func = nnl2_own_pcopy_float64; break;
            case FLOAT32: worker_func = nnl2_own_pcopy_float32; break;
            case INT32:   worker_func = nnl2_own_pcopy_int32;   break;
			
            default: {
                NNL2_TYPE_ERROR(dtype);
                nnl2_free_tensor(result);
                #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                    NNL2_FUNC_EXIT();
                #endif
                return NULL;
            }
        }
        
        // Create thread to process the assigned chunk
        int status = pthread_create(&threads[i], NULL, worker_func, &tasks[i]);
        if(status != 0) {
            NNL2_THREAD_CREATE_ERROR(status, "nnl2_own_copy");
            num_threads = i;
            break;
        }
        
        current_start += current_chunk;
    }
    
    // Wait for all threads to complete
    for (size_t i = 0; i < num_threads; i++) {
        int join_status = pthread_join(threads[i], NULL);
        if(join_status != 0) {
            NNL2_THREAD_JOIN_ERROR(join_status, "nnl2_own_copy");
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return result;
}

// Worker function implementations with AVX256 and prefetching

/** @brief
 * See documentation at declaration
 * 
 ** @see nnl2_own_pcopy_float64
 **/
void* nnl2_own_pcopy_float64(void* arg) {
    copy_ptask* task = (copy_ptask*)arg;
    double* src_data = (double*)task->src_data;
    double* dst_data = (double*)task->dst_data;
    size_t start = task->start;
    size_t end = task->end;
    
    size_t i = start;
    
    // AVX256 processing with prefetching
    if(task->aligned) {
        for(; i + 3 < end; i += 4) {
            // Prefetch next cache lines for both source and destination
            _mm_prefetch((char*)&src_data[i + 16], _MM_HINT_T0);
            _mm_prefetch((char*)&dst_data[i + 16], _MM_HINT_T1);
            
            __m256d v_data = _mm256_load_pd(&src_data[i]);
            _mm256_store_pd(&dst_data[i], v_data);
        }
    } else {
        for(; i + 3 < end; i += 4) {
            _mm_prefetch((char*)&src_data[i + 16], _MM_HINT_T0);
            _mm_prefetch((char*)&dst_data[i + 16], _MM_HINT_T1);
            
            __m256d v_data = _mm256_loadu_pd(&src_data[i]);
            _mm256_storeu_pd(&dst_data[i], v_data);
        }
    }
    
    // Scalar processing for remainder
    for(; i < end; i++) {
        dst_data[i] = src_data[i];
    }
    
    return NULL;
}

/** @brief
 * See documentation at declaration
 * 
 ** @see nnl2_own_pcopy_float32
 **/
void* nnl2_own_pcopy_float32(void* arg) {
    copy_ptask* task = (copy_ptask*)arg;
    float* src_data = (float*)task->src_data;
    float* dst_data = (float*)task->dst_data;
    size_t start = task->start;
    size_t end = task->end;
    
    size_t i = start;
    
    // AVX256 processing with prefetching (8 elements per iteration)
    if(task->aligned) {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&src_data[i + 32], _MM_HINT_T0);
            _mm_prefetch((char*)&dst_data[i + 32], _MM_HINT_T1);
            
            __m256 v_data = _mm256_load_ps(&src_data[i]);
            _mm256_store_ps(&dst_data[i], v_data);
        }
    } else {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&src_data[i + 32], _MM_HINT_T0);
            _mm_prefetch((char*)&dst_data[i + 32], _MM_HINT_T1);
            
            __m256 v_data = _mm256_loadu_ps(&src_data[i]);
            _mm256_storeu_ps(&dst_data[i], v_data);
        }
    }
    
    // Scalar processing for remainder
    for(; i < end; i++) {
        dst_data[i] = src_data[i];
    }
    
    return NULL;
}

/** @brief
 * See documentation at declaration
 * 
 ** @see nnl2_own_pcopy_int32
 **/
void* nnl2_own_pcopy_int32(void* arg) {
    copy_ptask* task = (copy_ptask*)arg;
    int32_t* src_data = (int32_t*)task->src_data;
    int32_t* dst_data = (int32_t*)task->dst_data;
    size_t start = task->start;
    size_t end = task->end;
    
    size_t i = start;
    
    // AVX256 processing with prefetching (8 elements per iteration)
    if(task->aligned) {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&src_data[i + 32], _MM_HINT_T0);
            _mm_prefetch((char*)&dst_data[i + 32], _MM_HINT_T1);
            
            __m256i v_data = _mm256_load_si256((__m256i*)&src_data[i]);
            _mm256_store_si256((__m256i*)&dst_data[i], v_data);
        }
    } else {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&src_data[i + 32], _MM_HINT_T0);
            _mm_prefetch((char*)&dst_data[i + 32], _MM_HINT_T1);
            
            __m256i v_data = _mm256_loadu_si256((__m256i*)&src_data[i]);
            _mm256_storeu_si256((__m256i*)&dst_data[i], v_data);
        }
    }
    
    // Scalar processing for remainder
    for(; i < end; i++) {
        dst_data[i] = src_data[i];
    }
    
    return NULL;
}

#endif

/**
 * @ingroup backend_system
 * @brief Backend implementations for copy operation
 * @details
 * Array follows the common backend registration pattern for copy operations.
 * Currently registered backends:
 *  - nnl2_naive: Basic reference implementation
 * 
 * @see nnl2_naive
 */
Implementation copy_backends[] = {
	REGISTER_BACKEND(naive_copy, nnl2_naive, NAIVE_BACKEND_NAME),
	
	#if defined(NNL2_AVX256_AVAILABLE) && defined(NNL2_PTHREAD_AVAILABLE)
		REGISTER_BACKEND(nnl2_own_copy, nnl2_own, NNL2_OWN_NAME),
	#endif
};

/**
 * @brief Function pointer for copy operation
 * @ingroup backend_system 
 */
copyfn nnl2_copy;

/** 
 * @brief Makes the copy backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(copy);

/** 
 * @brief Sets the backend for copy operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see ESET_BACKEND_BY_NAME
 */
void set_copy_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(copy_backends, nnl2_copy, backend_name, current_backend(copy));
}

/** 
 * @brief Gets the name of the active backend for copy operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_copy_backend() {
	return current_backend(copy);
}

/** 
 * @brief Function declaration for getting all available copy backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(copy);

/**
 * @brief Function declaration for getting the number of available copy backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(copy);

#endif /** NNL2_COPY_H **/
