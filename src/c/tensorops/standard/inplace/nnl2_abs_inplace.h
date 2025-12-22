#ifndef NNL2_ABS_INPLACE_H
#define NNL2_ABS_INPLACE_H

/** @brief 
 * Calculates the absolute values of the tensor elements in place
 *
 ** @param tensor
 * A pointer to a tensor that will be modified
 *
 ** @see fabs
 ** @see fabsf
 ** @see abs
 **/
void naive_absinplace(nnl2_tensor* tensor) {	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif

	size_t total_elems = product(tensor->shape, tensor->rank);
	if(total_elems == 0) return; // If tensor is empty then return
	
	void* data = tensor->data;
	
	switch(tensor->dtype) {
		case FLOAT64: {
			double* cast_data = (double*)data;	
			for(size_t i = 0; i < total_elems; i++) cast_data[i] = fabs(cast_data[i]);
			break;
		}
		
		case FLOAT32: {
			float* cast_data = (float*)data;	
			for(size_t i = 0; i < total_elems; i++) cast_data[i] = fabsf(cast_data[i]);
			break;
		}
		
		case INT32: {
			int32_t* cast_data = (int32_t*)data;	
			for(size_t i = 0; i < total_elems; i++) cast_data[i] = abs(cast_data[i]);
			break;
		}
		
		default: {
			NNL2_TYPE_ERROR(tensor->dtype);
			return;
		}
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

#ifdef NNL2_PTHREAD_AVAILABLE

/** @brief
 * Threshold for enabling parallel execution of the
 * in-place absolute value operation
 */
#define NNL2_ABS_INPLACE_PARALLEL_THREASHOLD 50000

/** @brief 
 * Worker function for parallel in-place absolute value operation
 * 
 * @param arg 
 * Pointer to abs_inplace_ptask structure containing task parameters
 *
 * @return NULL (for pthread api)
 */
void* nnl2_own_pabs_inplace(void* arg);

#ifdef NNL2_AVX256_AVAILABLE

/** @brief 
 * SIMD worker function for parallel in-place absolute value for float32
 * 
 * @param arg 
 * Pointer to abs_inplace_ptask structure containing task parameters
 *
 * @return NULL (for pthread api)
 */
void* nnl2_own_pabs_inplace_simd_float32(void* arg);

/** @brief 
 * SIMD worker function for parallel in-place absolute value for float64
 * 
 * @param arg 
 * Pointer to abs_inplace_ptask structure containing task parameters
 *
 * @return NULL (for pthread api)
 */
void* nnl2_own_pabs_inplace_simd_float64(void* arg);

/** @brief 
 * SIMD worker function for parallel in-place absolute value for int32
 * 
 * @param arg 
 * Pointer to abs_inplace_ptask structure containing task parameters
 *
 * @return NULL (for pthread api)
 */
void* nnl2_own_pabs_inplace_simd_int32(void* arg);

#endif

/** @brief
 * Optimized parallel implementation of in-place absolute value using pthreads + SIMD
 *
 ** @param tensor
 * Pointer to the input tensor (will be modified in-place)
 */
void nnl2_own_abs_inplace(nnl2_tensor* tensor) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Calculate the total number of elements in the tensor
    size_t total_elems = product(tensor->shape, tensor->rank);
    if(total_elems == 0) return;
    
    // For very small tensors use inline optimization
    if(total_elems < 256) {
        void* data = tensor->data;
        
        switch(tensor->dtype) {
            case FLOAT64: {
                double* cast_data = (double*)data;
                // Unrolled loop for small sizes
                size_t i = 0;
                for(; i + 3 < total_elems; i += 4) {
                    cast_data[i] = fabs(cast_data[i]);
                    cast_data[i+1] = fabs(cast_data[i+1]);
                    cast_data[i+2] = fabs(cast_data[i+2]);
                    cast_data[i+3] = fabs(cast_data[i+3]);
                }
                for(; i < total_elems; i++) {
                    cast_data[i] = fabs(cast_data[i]);
                }
                break;
            }
            
            case FLOAT32: {
                float* cast_data = (float*)data;
                // Unrolled loop for small sizes
                size_t i = 0;
                for(; i + 7 < total_elems; i += 8) {
                    cast_data[i] = fabsf(cast_data[i]);
                    cast_data[i+1] = fabsf(cast_data[i+1]);
                    cast_data[i+2] = fabsf(cast_data[i+2]);
                    cast_data[i+3] = fabsf(cast_data[i+3]);
                    cast_data[i+4] = fabsf(cast_data[i+4]);
                    cast_data[i+5] = fabsf(cast_data[i+5]);
                    cast_data[i+6] = fabsf(cast_data[i+6]);
                    cast_data[i+7] = fabsf(cast_data[i+7]);
                }
                for(; i < total_elems; i++) {
                    cast_data[i] = fabsf(cast_data[i]);
                }
                break;
            }
            
            case INT32: {
                int32_t* cast_data = (int32_t*)data;
                // Unrolled loop for small sizes
                size_t i = 0;
                for(; i + 7 < total_elems; i += 8) {
                    cast_data[i] = abs(cast_data[i]);
                    cast_data[i+1] = abs(cast_data[i+1]);
                    cast_data[i+2] = abs(cast_data[i+2]);
                    cast_data[i+3] = abs(cast_data[i+3]);
                    cast_data[i+4] = abs(cast_data[i+4]);
                    cast_data[i+5] = abs(cast_data[i+5]);
                    cast_data[i+6] = abs(cast_data[i+6]);
                    cast_data[i+7] = abs(cast_data[i+7]);
                }
                for(; i < total_elems; i++) {
                    cast_data[i] = abs(cast_data[i]);
                }
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(tensor->dtype);
                return;
            }
        }
        
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return;
    }
    
    // Use more threads for large tensors
    size_t num_threads = NNL2_NUM_THREADS;
    if(total_elems > 1000000) {
        num_threads = NNL2_NUM_THREADS * 2;
    }
    
    // Limit maximum number of threads
    if(num_threads > 16) num_threads = 16;
    
    // Allocate arrays for thread handles and task descriptors
    pthread_t threads[num_threads];
    abs_inplace_ptask tasks[num_threads];
    
    // Calculate base chunk size and remainder for balanced distribution
    size_t chunk = total_elems / num_threads;
    size_t remainder = total_elems % num_threads;
    
    bool use_simd = false;
    
    #ifdef NNL2_AVX256_AVAILABLE
		bool aligned_data = NNL2_IS_ALIGNED(tensor->data, NNL2_TENSOR_ALIGNMENT_32);
		use_simd = aligned_data;
    #endif
    
    // Distribute work among threads with load balancing
    size_t current_start = 0;
    for (size_t i = 0; i < num_threads; i++) {
        size_t current_chunk = chunk + (i < remainder ? 1 : 0);
        
        // Configure task for this thread
        tasks[i].data = tensor->data;
        tasks[i].start = current_start;
        tasks[i].end = current_start + current_chunk;
        tasks[i].dtype = tensor->dtype;
        
        // Create thread to process the assigned chunk
        int status;
        
        #ifdef NNL2_AVX256_AVAILABLE
            if(use_simd) {
                switch(tensor->dtype) {
                    case FLOAT64: status = pthread_create(&threads[i], NULL, nnl2_own_pabs_inplace_simd_float64, &tasks[i]); break;
                    case FLOAT32: status = pthread_create(&threads[i], NULL, nnl2_own_pabs_inplace_simd_float32, &tasks[i]); break;
                    case INT32:   status = pthread_create(&threads[i], NULL, nnl2_own_pabs_inplace_simd_int32, &tasks[i]);   break;
                    
                    default: {
                        status = pthread_create(&threads[i], NULL, nnl2_own_pabs_inplace, &tasks[i]);
                        break;
                    }
                }
            } else 
        #endif
        {
            status = pthread_create(&threads[i], NULL, nnl2_own_pabs_inplace, &tasks[i]);
        }
        
        if(status != 0) {
            NNL2_THREAD_CREATE_ERROR(status, "nnl2_own_abs_inplace");
            // Clean up already created threads
            for(size_t j = 0; j < i; j++) {
                pthread_join(threads[j], NULL);
            }
            return;
        }
        
        current_start += current_chunk;
    }
    
    // Wait for all threads to complete their work
    for (size_t i = 0; i < num_threads; i++) {
        int join_status = pthread_join(threads[i], NULL);
        if(join_status != 0) {
            NNL2_THREAD_JOIN_ERROR(join_status, "nnl2_own_abs_inplace");
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

/** @brief
 * See docs at declaration
 *
 ** @see nnl2_own_pabs_inplace
 **/
void* nnl2_own_pabs_inplace(void* arg) {
    abs_inplace_ptask* task = (abs_inplace_ptask*)arg;
    
    void* data = task->data;
    
    switch(task->dtype) {
        case FLOAT64: {
            double* cast_data = (double*)data;
            size_t i = task->start;
            
            // Optimized loop with unrolling
            for(; i + 7 < task->end; i += 8) {
                cast_data[i] = fabs(cast_data[i]);
                cast_data[i+1] = fabs(cast_data[i+1]);
                cast_data[i+2] = fabs(cast_data[i+2]);
                cast_data[i+3] = fabs(cast_data[i+3]);
                cast_data[i+4] = fabs(cast_data[i+4]);
                cast_data[i+5] = fabs(cast_data[i+5]);
                cast_data[i+6] = fabs(cast_data[i+6]);
                cast_data[i+7] = fabs(cast_data[i+7]);
            }
			
            for(; i < task->end; i++) {
                cast_data[i] = fabs(cast_data[i]);
            }
			
            break;
        }
        
        case FLOAT32: {
            float* cast_data = (float*)data;
            size_t i = task->start;
            
            // Optimized loop with unrolling
            for(; i + 15 < task->end; i += 16) {
                cast_data[i] = fabsf(cast_data[i]);
                cast_data[i+1] = fabsf(cast_data[i+1]);
                cast_data[i+2] = fabsf(cast_data[i+2]);
                cast_data[i+3] = fabsf(cast_data[i+3]);
                cast_data[i+4] = fabsf(cast_data[i+4]);
                cast_data[i+5] = fabsf(cast_data[i+5]);
                cast_data[i+6] = fabsf(cast_data[i+6]);
                cast_data[i+7] = fabsf(cast_data[i+7]);
                cast_data[i+8] = fabsf(cast_data[i+8]);
                cast_data[i+9] = fabsf(cast_data[i+9]);
                cast_data[i+10] = fabsf(cast_data[i+10]);
                cast_data[i+11] = fabsf(cast_data[i+11]);
                cast_data[i+12] = fabsf(cast_data[i+12]);
                cast_data[i+13] = fabsf(cast_data[i+13]);
                cast_data[i+14] = fabsf(cast_data[i+14]);
                cast_data[i+15] = fabsf(cast_data[i+15]);
            }
			
            for(; i < task->end; i++) {
                cast_data[i] = fabsf(cast_data[i]);
            }
			
            break;
        }
        
        case INT32: {
            int32_t* cast_data = (int32_t*)data;
            size_t i = task->start;
            
            // Optimized loop with unrolling
            for(; i + 15 < task->end; i += 16) {
                cast_data[i] = abs(cast_data[i]);
                cast_data[i+1] = abs(cast_data[i+1]);
                cast_data[i+2] = abs(cast_data[i+2]);
                cast_data[i+3] = abs(cast_data[i+3]);
                cast_data[i+4] = abs(cast_data[i+4]);
                cast_data[i+5] = abs(cast_data[i+5]);
                cast_data[i+6] = abs(cast_data[i+6]);
                cast_data[i+7] = abs(cast_data[i+7]);
                cast_data[i+8] = abs(cast_data[i+8]);
                cast_data[i+9] = abs(cast_data[i+9]);
                cast_data[i+10] = abs(cast_data[i+10]);
                cast_data[i+11] = abs(cast_data[i+11]);
                cast_data[i+12] = abs(cast_data[i+12]);
                cast_data[i+13] = abs(cast_data[i+13]);
                cast_data[i+14] = abs(cast_data[i+14]);
                cast_data[i+15] = abs(cast_data[i+15]);
            }
			
            for(; i < task->end; i++) {
                cast_data[i] = abs(cast_data[i]);
            }
			
            break;
        }
        
        default: {
            break;
        }
    }
    
    return NULL;
}

#ifdef NNL2_AVX256_AVAILABLE

/** @brief
 * SIMD worker function for float32 in-place absolute value
 *
 ** @see nnl2_own_pabs_inplace_simd_float32
 **/
void* nnl2_own_pabs_inplace_simd_float32(void* arg) {
    abs_inplace_ptask* task = (abs_inplace_ptask*)arg;
    
    float* data = (float*)task->data;
    
    // Create mask for absolute value (clear sign bit)
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    
    size_t i = task->start;
    
    // Process in large blocks of 64 elements for better cache locality
    for(; i + 63 < task->end; i += 64) {
        // Prefetch next block
        _mm_prefetch((char*)&data[i + 64], _MM_HINT_T0);
        
        // Process 64 elements in 8 iterations
        __m256 v0 = _mm256_load_ps(&data[i]);
        __m256 v1 = _mm256_load_ps(&data[i + 8]);
        __m256 v2 = _mm256_load_ps(&data[i + 16]);
        __m256 v3 = _mm256_load_ps(&data[i + 24]);
        __m256 v4 = _mm256_load_ps(&data[i + 32]);
        __m256 v5 = _mm256_load_ps(&data[i + 40]);
        __m256 v6 = _mm256_load_ps(&data[i + 48]);
        __m256 v7 = _mm256_load_ps(&data[i + 56]);
        
        _mm256_store_ps(&data[i], _mm256_andnot_ps(sign_mask, v0));
        _mm256_store_ps(&data[i + 8], _mm256_andnot_ps(sign_mask, v1));
        _mm256_store_ps(&data[i + 16], _mm256_andnot_ps(sign_mask, v2));
        _mm256_store_ps(&data[i + 24], _mm256_andnot_ps(sign_mask, v3));
        _mm256_store_ps(&data[i + 32], _mm256_andnot_ps(sign_mask, v4));
        _mm256_store_ps(&data[i + 40], _mm256_andnot_ps(sign_mask, v5));
        _mm256_store_ps(&data[i + 48], _mm256_andnot_ps(sign_mask, v6));
        _mm256_store_ps(&data[i + 56], _mm256_andnot_ps(sign_mask, v7));
    }
    
    // Process remaining 32 elements
    for(; i + 31 < task->end; i += 32) {
        __m256 v0 = _mm256_load_ps(&data[i]);
        __m256 v1 = _mm256_load_ps(&data[i + 8]);
        __m256 v2 = _mm256_load_ps(&data[i + 16]);
        __m256 v3 = _mm256_load_ps(&data[i + 24]);
        
        _mm256_store_ps(&data[i], _mm256_andnot_ps(sign_mask, v0));
        _mm256_store_ps(&data[i + 8], _mm256_andnot_ps(sign_mask, v1));
        _mm256_store_ps(&data[i + 16], _mm256_andnot_ps(sign_mask, v2));
        _mm256_store_ps(&data[i + 24], _mm256_andnot_ps(sign_mask, v3));
    }
    
    // Process remaining 16 elements
    for(; i + 15 < task->end; i += 16) {
        __m256 v0 = _mm256_load_ps(&data[i]);
        __m256 v1 = _mm256_load_ps(&data[i + 8]);
        
        _mm256_store_ps(&data[i], _mm256_andnot_ps(sign_mask, v0));
        _mm256_store_ps(&data[i + 8], _mm256_andnot_ps(sign_mask, v1));
    }
    
    // Process remaining 8 elements
    for(; i + 7 < task->end; i += 8) {
        __m256 v = _mm256_load_ps(&data[i]);
        _mm256_store_ps(&data[i], _mm256_andnot_ps(sign_mask, v));
    }
    
    // Process tail
    for(; i < task->end; i++) {
        data[i] = fabsf(data[i]);
    }
    
    return NULL;
}

/** @brief
 * SIMD worker function for float64 in-place absolute value
 *
 ** @see nnl2_own_pabs_inplace_simd_float64
 **/
void* nnl2_own_pabs_inplace_simd_float64(void* arg) {
    abs_inplace_ptask* task = (abs_inplace_ptask*)arg;
    
    double* data = (double*)task->data;
    
    const __m256d sign_mask = _mm256_set1_pd(-0.0);
    
    size_t i = task->start;
    
    // Process in large blocks of 32 elements
    for(; i + 31 < task->end; i += 32) {
        // Prefetch next block
        _mm_prefetch((char*)&data[i + 32], _MM_HINT_T0);
        
        // Process 32 elements in 8 iterations
        __m256d v0 = _mm256_load_pd(&data[i]);
        __m256d v1 = _mm256_load_pd(&data[i + 4]);
        __m256d v2 = _mm256_load_pd(&data[i + 8]);
        __m256d v3 = _mm256_load_pd(&data[i + 12]);
        __m256d v4 = _mm256_load_pd(&data[i + 16]);
        __m256d v5 = _mm256_load_pd(&data[i + 20]);
        __m256d v6 = _mm256_load_pd(&data[i + 24]);
        __m256d v7 = _mm256_load_pd(&data[i + 28]);
        
        _mm256_store_pd(&data[i], _mm256_andnot_pd(sign_mask, v0));
        _mm256_store_pd(&data[i + 4], _mm256_andnot_pd(sign_mask, v1));
        _mm256_store_pd(&data[i + 8], _mm256_andnot_pd(sign_mask, v2));
        _mm256_store_pd(&data[i + 12], _mm256_andnot_pd(sign_mask, v3));
        _mm256_store_pd(&data[i + 16], _mm256_andnot_pd(sign_mask, v4));
        _mm256_store_pd(&data[i + 20], _mm256_andnot_pd(sign_mask, v5));
        _mm256_store_pd(&data[i + 24], _mm256_andnot_pd(sign_mask, v6));
        _mm256_store_pd(&data[i + 28], _mm256_andnot_pd(sign_mask, v7));
    }
    
    // Process tail
    for(; i + 3 < task->end; i += 4) {
        __m256d v = _mm256_load_pd(&data[i]);
        _mm256_store_pd(&data[i], _mm256_andnot_pd(sign_mask, v));
    }
		
    for(; i < task->end; i++) {
        data[i] = fabs(data[i]);
    }
    
    return NULL;
}

/** @brief
 * SIMD worker function for int32 in-place absolute value
 *
 ** @see nnl2_own_pabs_inplace_simd_int32
 **/
void* nnl2_own_pabs_inplace_simd_int32(void* arg) {
    abs_inplace_ptask* task = (abs_inplace_ptask*)arg;
    
    int32_t* data = (int32_t*)task->data;
    
    size_t i = task->start;
    
    // Process in large blocks of 64 elements
    for(; i + 63 < task->end; i += 64) {
        // Prefetch next block
        _mm_prefetch((char*)&data[i + 64], _MM_HINT_T0);
        
        // Process 64 elements in 8 iterations
        for(size_t j = 0; j < 64; j += 8) {
            __m256i v_input = _mm256_load_si256((__m256i*)&data[i + j]);
            __m256i v_mask = _mm256_srai_epi32(v_input, 31);
            __m256i v_result = _mm256_sub_epi32(_mm256_xor_si256(v_input, v_mask), v_mask);
            _mm256_store_si256((__m256i*)&data[i + j], v_result);
        }
    }
    
    // Process tail
    for(; i + 7 < task->end; i += 8) {
        __m256i v_input = _mm256_load_si256((__m256i*)&data[i]);
        __m256i v_mask = _mm256_srai_epi32(v_input, 31);
        __m256i v_result = _mm256_sub_epi32(_mm256_xor_si256(v_input, v_mask), v_mask);
        _mm256_store_si256((__m256i*)&data[i], v_result);
    }
	
    for(; i < task->end; i++) {
        data[i] = abs(data[i]);
    }
    
    return NULL;
}

#endif

#endif

/**
 * @ingroup backend_system
 * @brief Backend implementations for absinplace operation
 * @details
 * Array follows the common backend registration pattern for in-place absolute value operations.
 * Currently registered backends:
 *  - nnl2_naive: Basic reference implementation
 *  - nnl2_own: Optimized pthread + SIMD implementation
 * 
 * @see nnl2_naive
 * @see nnl2_own_abs_inplace
 */
nnl2_runtime_implementation absinplace_backends[] = {
	REGISTER_BACKEND(naive_absinplace, nnl2_naive, NAIVE_BACKEND_NAME),
	
	#ifdef NNL2_PTHREAD_AVAILABLE
		REGISTER_BACKEND(nnl2_own_abs_inplace, nnl2_own, NNL2_OWN_NAME),
	#endif
};	

/**
 * @brief Function pointer for absinplace operation
 * @ingroup backend_system 
 */
absinplacefn absinplace;

/** 
 * @brief Makes the absinplace backend current
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(absinplace);

/** 
 * @brief Sets the backend for absinplace operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see ESET_BACKEND_BY_NAME
 */
void set_absinplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(absinplace_backends, absinplace, backend_name, CURRENT_BACKEND(absinplace));
}

/** 
 * @brief Gets the name of the active backend for absinplace operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_absinplace_backend() {
	return CURRENT_BACKEND(absinplace);
}

/** 
 * @brief Function declaration for getting all available absinplace backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(absinplace);

/**
 * @brief Function declaration for getting the number of available absinplace backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(absinplace);

#endif /** NNL2_ABS_INPLACE_H **/
