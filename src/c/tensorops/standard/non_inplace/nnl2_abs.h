#ifndef NNL2_ABS_H
#define NNL2_ABS_H

/** @brief
 * Naive implementation of absolute value operation
 *
 ** @param tensor 
 * Input tensor
 *
 ** @return 
 * New tensor with absolute values of input elements
 */
nnl2_tensor* naive_abs(nnl2_tensor* tensor) {	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif

	size_t total_elems = nnl2_product(tensor->shape, tensor->rank);	
	
	nnl2_tensor* result = nnl2_empty(tensor->shape, tensor->rank, tensor->dtype);
	if(total_elems == 0) return result;
	
	void* data_t = tensor->data;
	void* data_r = result->data;
	
	switch(tensor->dtype) {
		case FLOAT64: {
			double* cast_data_t = (double*)data_t;	
			double* cast_data_r = (double*)data_r;
			for(size_t i = 0; i < total_elems; i++) cast_data_r[i] = fabs(cast_data_t[i]);
			break;
		}
		
		case FLOAT32: {
			float* cast_data_t = (float*)data_t;	
			float* cast_data_r = (float*)data_r;
			for(size_t i = 0; i < total_elems; i++) cast_data_r[i] = fabsf(cast_data_t[i]);
			break;
		}
		
		case INT32: {
			int32_t* cast_data_t = (int32_t*)data_t;	
			int32_t* cast_data_r = (int32_t*)data_r;
			for(size_t i = 0; i < total_elems; i++) cast_data_r[i] = abs(cast_data_t[i]);
			break;
		}
		
		default: {
			NNL2_TYPE_ERROR(tensor->dtype);
			nnl2_free_tensor(result);
			return NULL;
		}
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
	
	return result;
}

#ifdef NNL2_PTHREAD_AVAILABLE

/** @brief
 * Threshold for enabling parallel execution of the
 * absolute value operation
 */
#define NNL2_ABS_PARALLEL_THREASHOLD 50000 

/** @brief
 * Threshold for enabling prefetching in abs operation
 */
#define NNL2_ABS_PREFETCH_THREASHOLD 4096   

/** @brief 
 * Worker function for parallel absolute value operation
 * 
 * @param arg 
 * Pointer to abs_ptask structure containing task parameters
 *
 * @return NULL (for pthread api)
 */
void* nnl2_own_pabs(void* arg);

#ifdef NNL2_AVX256_AVAILABLE

/** @brief 
 * Ultra-optimized SIMD worker function for parallel absolute value for float32
 * 
 * @param arg 
 * Pointer to abs_ptask structure containing task parameters
 *
 * @return NULL (for pthread api)
 */
void* nnl2_own_pabs_ultra_float32(void* arg);

/** @brief 
 * Ultra-optimized SIMD worker function for parallel absolute value for float64
 * 
 * @param arg 
 * Pointer to abs_ptask structure containing task parameters
 *
 * @return NULL (for pthread api)
 */
void* nnl2_own_pabs_ultra_float64(void* arg);

/** @brief 
 * Ultra-optimized SIMD worker function for parallel absolute value for int32
 * 
 * @param arg 
 * Pointer to abs_ptask structure containing task parameters
 *
 * @return NULL (for pthread api)
 */
void* nnl2_own_pabs_ultra_int32(void* arg);

#endif

/** @brief
 * Extreme-optimized parallel implementation of absolute value using pthreads + SIMD
 *
 ** @param tensor
 * Pointer to the input tensor
 *
 ** @return 
 * Pointer to a new tensor with absolute values
 */
nnl2_tensor* nnl2_own_abs(const nnl2_tensor* tensor) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Calculate the total number of elements in the tensor
    size_t total_elems = nnl2_product(tensor->shape, tensor->rank);
    
    // Create an output tensor with the same shape and data type
    nnl2_tensor* result = nnl2_empty(tensor->shape, tensor->rank, tensor->dtype);
    
    if(total_elems == 0) return result;
    
    if(total_elems < 256) {
        void* data_t = tensor->data;
        void* data_r = result->data;
        
        switch(tensor->dtype) {
            case FLOAT64: {
                double* cast_data_t = (double*)data_t;    
                double* cast_data_r = (double*)data_r;

                size_t i = 0;
				
                for(; i + 3 < total_elems; i += 4) {
                    cast_data_r[i] = fabs(cast_data_t[i]);
                    cast_data_r[i+1] = fabs(cast_data_t[i+1]);
                    cast_data_r[i+2] = fabs(cast_data_t[i+2]);
                    cast_data_r[i+3] = fabs(cast_data_t[i+3]);
                }
				
                for(; i < total_elems; i++) {
                    cast_data_r[i] = fabs(cast_data_t[i]);
                }
				
                break;
            }
            
            case FLOAT32: {
                float* cast_data_t = (float*)data_t;    
                float* cast_data_r = (float*)data_r;

                size_t i = 0;
				
                for(; i + 7 < total_elems; i += 8) {
                    cast_data_r[i] = fabsf(cast_data_t[i]);
                    cast_data_r[i+1] = fabsf(cast_data_t[i+1]);
                    cast_data_r[i+2] = fabsf(cast_data_t[i+2]);
                    cast_data_r[i+3] = fabsf(cast_data_t[i+3]);
                    cast_data_r[i+4] = fabsf(cast_data_t[i+4]);
                    cast_data_r[i+5] = fabsf(cast_data_t[i+5]);
                    cast_data_r[i+6] = fabsf(cast_data_t[i+6]);
                    cast_data_r[i+7] = fabsf(cast_data_t[i+7]);
                }
				
                for(; i < total_elems; i++) {
                    cast_data_r[i] = fabsf(cast_data_t[i]);
                }
				
                break;
            }
            
            case INT32: {
                int32_t* cast_data_t = (int32_t*)data_t;    
                int32_t* cast_data_r = (int32_t*)data_r;
				
                size_t i = 0;
				
                for(; i + 7 < total_elems; i += 8) {
                    cast_data_r[i] = abs(cast_data_t[i]);
                    cast_data_r[i+1] = abs(cast_data_t[i+1]);
                    cast_data_r[i+2] = abs(cast_data_t[i+2]);
                    cast_data_r[i+3] = abs(cast_data_t[i+3]);
                    cast_data_r[i+4] = abs(cast_data_t[i+4]);
                    cast_data_r[i+5] = abs(cast_data_t[i+5]);
                    cast_data_r[i+6] = abs(cast_data_t[i+6]);
                    cast_data_r[i+7] = abs(cast_data_t[i+7]);
                }
				
                for(; i < total_elems; i++) {
                    cast_data_r[i] = abs(cast_data_t[i]);
                }
				
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(tensor->dtype);
                nnl2_free_tensor(result);
                return NULL;
            }
        }
        
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        
        return result;
    }
   
    size_t num_threads = NNL2_NUM_THREADS;
    if(total_elems > 1000000) {
        num_threads = NNL2_NUM_THREADS * 2;
    }
    
    if(num_threads > 16) num_threads = 16;
    
    // Allocate arrays for thread handles and task descriptors
    pthread_t threads[num_threads];
    abs_ptask tasks[num_threads];
    
    // Calculate base chunk size and remainder for balanced distribution
    size_t chunk = total_elems / num_threads;
    size_t remainder = total_elems % num_threads;
    
    bool use_ultra_simd = false;
    
    #ifdef NNL2_AVX256_AVAILABLE
		bool aligned_input = NNL2_IS_ALIGNED(tensor->data, NNL2_TENSOR_ALIGNMENT_32);
		bool aligned_result = NNL2_IS_ALIGNED(result->data, NNL2_TENSOR_ALIGNMENT_32);
		use_ultra_simd = aligned_input && aligned_result;
    #endif
    
    // Distribute work among threads with load balancing
    size_t current_start = 0;
    for (size_t i = 0; i < num_threads; i++) {
        size_t current_chunk = chunk + (i < remainder ? 1 : 0);
        
        // Configure task for this thread
        tasks[i].input_data = tensor->data;
        tasks[i].result_data = result->data;
        tasks[i].start = current_start;
        tasks[i].end = current_start + current_chunk;
        tasks[i].dtype = tensor->dtype;
        
        // Create thread to process the assigned chunk
        int status;
        
        #ifdef NNL2_AVX256_AVAILABLE
            if(use_ultra_simd) {
                switch(tensor->dtype) {
                    case FLOAT64: status = pthread_create(&threads[i], NULL, nnl2_own_pabs_ultra_float64, &tasks[i]); break;
                    case FLOAT32: status = pthread_create(&threads[i], NULL, nnl2_own_pabs_ultra_float32, &tasks[i]); break;
                    case INT32:   status = pthread_create(&threads[i], NULL, nnl2_own_pabs_ultra_int32, &tasks[i]);   break;
                    
                    default: {
                        status = pthread_create(&threads[i], NULL, nnl2_own_pabs, &tasks[i]);
                        break;
                    }
                }
            } else 
        #endif
        {
            status = pthread_create(&threads[i], NULL, nnl2_own_pabs, &tasks[i]);
        }
        
        if(status != 0) {
            NNL2_THREAD_CREATE_ERROR(status, "nnl2_hyper_abs");
            // Clean up already created threads
            for(size_t j = 0; j < i; j++) {
                pthread_join(threads[j], NULL);
            }
            
            nnl2_free_tensor(result);
            return NULL;
        }
        
        current_start += current_chunk;
    }
    
    // Wait for all threads to complete their work
    for (size_t i = 0; i < num_threads; i++) {
        int join_status = pthread_join(threads[i], NULL);
        if(join_status != 0) {
            NNL2_THREAD_JOIN_ERROR(join_status, "nnl2_hyper_abs");
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return result;
}

/** @brief
 * See docs at declaration
 *
 ** @see nnl2_own_pabs
 **/
void* nnl2_own_pabs(void* arg) {
    abs_ptask* task = (abs_ptask*)arg;
    
    void* data_input = (void*)task->input_data;
    void* data_result = (void*)task->result_data;
    
    switch(task->dtype) {
        case FLOAT64: {
            double* input = (double*)data_input;
            double* result = (double*)data_result;
            size_t i = task->start;
            
            for(; i + 7 < task->end; i += 8) {
                result[i] = fabs(input[i]);
                result[i+1] = fabs(input[i+1]);
                result[i+2] = fabs(input[i+2]);
                result[i+3] = fabs(input[i+3]);
                result[i+4] = fabs(input[i+4]);
                result[i+5] = fabs(input[i+5]);
                result[i+6] = fabs(input[i+6]);
                result[i+7] = fabs(input[i+7]);
            }
			
            for(; i < task->end; i++) {
                result[i] = fabs(input[i]);
            }
			
            break;
        }
        
        case FLOAT32: {
            float* input = (float*)data_input;
            float* result = (float*)data_result;
            size_t i = task->start;

            for(; i + 15 < task->end; i += 16) {
                result[i] = fabsf(input[i]);
                result[i+1] = fabsf(input[i+1]);
                result[i+2] = fabsf(input[i+2]);
                result[i+3] = fabsf(input[i+3]);
                result[i+4] = fabsf(input[i+4]);
                result[i+5] = fabsf(input[i+5]);
                result[i+6] = fabsf(input[i+6]);
                result[i+7] = fabsf(input[i+7]);
                result[i+8] = fabsf(input[i+8]);
                result[i+9] = fabsf(input[i+9]);
                result[i+10] = fabsf(input[i+10]);
                result[i+11] = fabsf(input[i+11]);
                result[i+12] = fabsf(input[i+12]);
                result[i+13] = fabsf(input[i+13]);
                result[i+14] = fabsf(input[i+14]);
                result[i+15] = fabsf(input[i+15]);
            }
			
            for(; i < task->end; i++) {
                result[i] = fabsf(input[i]);
            }
			
            break;
        }
        
        case INT32: {
            int32_t* input = (int32_t*)data_input;
            int32_t* result = (int32_t*)data_result;
            size_t i = task->start;

            for(; i + 15 < task->end; i += 16) {
                result[i] = abs(input[i]);
                result[i+1] = abs(input[i+1]);
                result[i+2] = abs(input[i+2]);
                result[i+3] = abs(input[i+3]);
                result[i+4] = abs(input[i+4]);
                result[i+5] = abs(input[i+5]);
                result[i+6] = abs(input[i+6]);
                result[i+7] = abs(input[i+7]);
                result[i+8] = abs(input[i+8]);
                result[i+9] = abs(input[i+9]);
                result[i+10] = abs(input[i+10]);
                result[i+11] = abs(input[i+11]);
                result[i+12] = abs(input[i+12]);
                result[i+13] = abs(input[i+13]);
                result[i+14] = abs(input[i+14]);
                result[i+15] = abs(input[i+15]);
            }
			
            for(; i < task->end; i++) {
                result[i] = abs(input[i]);
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
 * Ultra-optimized SIMD worker function for float32 absolute value
 *
 ** @see nnl2_own_pabs_ultra_float32
 **/
void* nnl2_own_pabs_ultra_float32(void* arg) {
    abs_ptask* task = (abs_ptask*)arg;
    
    float* data_input = (float*)task->input_data;
    float* data_result = (float*)task->result_data;
    
    // Create mask for absolute value (clear sign bit)
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    
    size_t i = task->start;
    
    for(; i + 63 < task->end; i += 64) {
        // Prefetch next block
        _mm_prefetch((char*)&data_input[i + 64], _MM_HINT_T0);
        _mm_prefetch((char*)&data_result[i + 64], _MM_HINT_T1);
        
        __m256 v0 = _mm256_load_ps(&data_input[i]);
        __m256 v1 = _mm256_load_ps(&data_input[i + 8]);
        __m256 v2 = _mm256_load_ps(&data_input[i + 16]);
        __m256 v3 = _mm256_load_ps(&data_input[i + 24]);
        __m256 v4 = _mm256_load_ps(&data_input[i + 32]);
        __m256 v5 = _mm256_load_ps(&data_input[i + 40]);
        __m256 v6 = _mm256_load_ps(&data_input[i + 48]);
        __m256 v7 = _mm256_load_ps(&data_input[i + 56]);
        
        _mm256_store_ps(&data_result[i], _mm256_andnot_ps(sign_mask, v0));
        _mm256_store_ps(&data_result[i + 8], _mm256_andnot_ps(sign_mask, v1));
        _mm256_store_ps(&data_result[i + 16], _mm256_andnot_ps(sign_mask, v2));
        _mm256_store_ps(&data_result[i + 24], _mm256_andnot_ps(sign_mask, v3));
        _mm256_store_ps(&data_result[i + 32], _mm256_andnot_ps(sign_mask, v4));
        _mm256_store_ps(&data_result[i + 40], _mm256_andnot_ps(sign_mask, v5));
        _mm256_store_ps(&data_result[i + 48], _mm256_andnot_ps(sign_mask, v6));
        _mm256_store_ps(&data_result[i + 56], _mm256_andnot_ps(sign_mask, v7));
    }
    
    for(; i + 31 < task->end; i += 32) {
        __m256 v0 = _mm256_load_ps(&data_input[i]);
        __m256 v1 = _mm256_load_ps(&data_input[i + 8]);
        __m256 v2 = _mm256_load_ps(&data_input[i + 16]);
        __m256 v3 = _mm256_load_ps(&data_input[i + 24]);
        
        _mm256_store_ps(&data_result[i], _mm256_andnot_ps(sign_mask, v0));
        _mm256_store_ps(&data_result[i + 8], _mm256_andnot_ps(sign_mask, v1));
        _mm256_store_ps(&data_result[i + 16], _mm256_andnot_ps(sign_mask, v2));
        _mm256_store_ps(&data_result[i + 24], _mm256_andnot_ps(sign_mask, v3));
    }
    
    for(; i + 15 < task->end; i += 16) {
        __m256 v0 = _mm256_load_ps(&data_input[i]);
        __m256 v1 = _mm256_load_ps(&data_input[i + 8]);
        
        _mm256_store_ps(&data_result[i], _mm256_andnot_ps(sign_mask, v0));
        _mm256_store_ps(&data_result[i + 8], _mm256_andnot_ps(sign_mask, v1));
    }
	
    for(; i + 7 < task->end; i += 8) {
        __m256 v = _mm256_load_ps(&data_input[i]);
        _mm256_store_ps(&data_result[i], _mm256_andnot_ps(sign_mask, v));
    }

    for(; i < task->end; i++) {
        data_result[i] = fabsf(data_input[i]);
    }
    
    return NULL;
}

/** @brief
 * Ultra-optimized SIMD worker function for float64 absolute value
 *
 ** @see nnl2_own_pabs_ultra_float64
 **/
void* nnl2_own_pabs_ultra_float64(void* arg) {
    abs_ptask* task = (abs_ptask*)arg;
    
    double* data_input = (double*)task->input_data;
    double* data_result = (double*)task->result_data;
    
    const __m256d sign_mask = _mm256_set1_pd(-0.0);
    
    size_t i = task->start;

    for(; i + 31 < task->end; i += 32) {
        // Prefetch next block
        _mm_prefetch((char*)&data_input[i + 32], _MM_HINT_T0);
        _mm_prefetch((char*)&data_result[i + 32], _MM_HINT_T1);

        __m256d v0 = _mm256_load_pd(&data_input[i]);
        __m256d v1 = _mm256_load_pd(&data_input[i + 4]);
        __m256d v2 = _mm256_load_pd(&data_input[i + 8]);
        __m256d v3 = _mm256_load_pd(&data_input[i + 12]);
        __m256d v4 = _mm256_load_pd(&data_input[i + 16]);
        __m256d v5 = _mm256_load_pd(&data_input[i + 20]);
        __m256d v6 = _mm256_load_pd(&data_input[i + 24]);
        __m256d v7 = _mm256_load_pd(&data_input[i + 28]);
        
        _mm256_store_pd(&data_result[i], _mm256_andnot_pd(sign_mask, v0));
        _mm256_store_pd(&data_result[i + 4], _mm256_andnot_pd(sign_mask, v1));
        _mm256_store_pd(&data_result[i + 8], _mm256_andnot_pd(sign_mask, v2));
        _mm256_store_pd(&data_result[i + 12], _mm256_andnot_pd(sign_mask, v3));
        _mm256_store_pd(&data_result[i + 16], _mm256_andnot_pd(sign_mask, v4));
        _mm256_store_pd(&data_result[i + 20], _mm256_andnot_pd(sign_mask, v5));
        _mm256_store_pd(&data_result[i + 24], _mm256_andnot_pd(sign_mask, v6));
        _mm256_store_pd(&data_result[i + 28], _mm256_andnot_pd(sign_mask, v7));
    }

    for(; i + 3 < task->end; i += 4) {
        __m256d v = _mm256_load_pd(&data_input[i]);
        _mm256_store_pd(&data_result[i], _mm256_andnot_pd(sign_mask, v));
    }
    for(; i < task->end; i++) {
        data_result[i] = fabs(data_input[i]);
    }
    
    return NULL;
}

/** @brief
 * Ultra-optimized SIMD worker function for int32 absolute value
 *
 ** @see nnl2_own_pabs_ultra_int32
 **/
void* nnl2_own_pabs_ultra_int32(void* arg) {
    abs_ptask* task = (abs_ptask*)arg;
    
    int32_t* data_input = (int32_t*)task->input_data;
    int32_t* data_result = (int32_t*)task->result_data;
    
    size_t i = task->start;

    for(; i + 63 < task->end; i += 64) {
        // Prefetch next block
        _mm_prefetch((char*)&data_input[i + 64], _MM_HINT_T0);
        _mm_prefetch((char*)&data_result[i + 64], _MM_HINT_T1);

        for(size_t j = 0; j < 64; j += 8) {
            __m256i v_input = _mm256_load_si256((__m256i*)&data_input[i + j]);
            __m256i v_mask = _mm256_srai_epi32(v_input, 31);
            __m256i v_result = _mm256_sub_epi32(_mm256_xor_si256(v_input, v_mask), v_mask);
            _mm256_store_si256((__m256i*)&data_result[i + j], v_result);
        }
    }
    
    for(; i + 7 < task->end; i += 8) {
        __m256i v_input = _mm256_load_si256((__m256i*)&data_input[i]);
        __m256i v_mask = _mm256_srai_epi32(v_input, 31);
        __m256i v_result = _mm256_sub_epi32(_mm256_xor_si256(v_input, v_mask), v_mask);
        _mm256_store_si256((__m256i*)&data_result[i], v_result);
    }
    for(; i < task->end; i++) {
        data_result[i] = abs(data_input[i]);
    }
    
    return NULL;
}

#endif

#endif

/**
 * @ingroup backend_system
 * @brief Backend implementations for abs operation
 * @details
 * Array follows the common backend registration pattern for absolute value operations.
 * Currently registered backends:
 *  - nnl2_naive: Basic reference implementation
 *  - nnl2_hyper: Hyper-optimized pthread + SIMD implementation
 * 
 * @see nnl2_naive
 * @see nnl2_hyper_abs
 */
nnl2_runtime_implementation abs_backends[] = {
	REGISTER_BACKEND(naive_abs, nnl2_naive, NAIVE_BACKEND_NAME),
	
	#ifdef NNL2_PTHREAD_AVAILABLE
		REGISTER_BACKEND(nnl2_own_abs, nnl2_own, NNL2_OWN_NAME),
	#endif
};	

/**
 * @brief Function pointer for abs operation
 * @ingroup backend_system 
 */
absfn nnl2_abs;

/** 
 * @brief Makes the abs backend current
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(abs);

/** 
 * @brief Sets the backend for abs operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see ESET_BACKEND_BY_NAME
 */
void set_abs_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(abs_backends, nnl2_abs, backend_name, CURRENT_BACKEND(abs));
}

/** 
 * @brief Gets the name of the active backend for abs operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_abs_backend() {
	return CURRENT_BACKEND(abs);
}

/** 
 * @brief Function declaration for getting all available abs backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(abs);

/**
 * @brief Function declaration for getting the number of available abs backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(abs);

#endif /** NNL2_ABS_H **/
