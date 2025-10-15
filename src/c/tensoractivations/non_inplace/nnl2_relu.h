#ifndef NNL2_RELU_H
#define NNL2_RELU_H

/** @brief
 * Applies ReLU (ReLU(x) = max(x, 0)) activation function to a tensor, returning a new tensor (naive implementation)
 *
 ** @param tensor
 * Pointer to the input tensor
 *
 ** @return
 * Pointer to a new tensor containing the ReLU-activated values
 * Returns NULL in case of failure
 *
 ** @see nnl2_relu_float64
 ** @see nnl2_relu_float32
 ** @see nnl2_relu_int32
 ** @see nnl2_empty
 ** @see product
 **/
Tensor* naive_relu(Tensor* tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		if(tensor == NULL) {
			NNL2_ERROR("Passed tensor is NULL");
		}
	#endif

	int total_elems = product(tensor->shape, tensor->rank);	
	Tensor* result = nnl2_empty(tensor->shape, tensor->rank, tensor->dtype);
	
	if(total_elems == 0) return result; // If tensor is empty return tensor with 0 elements
	
	void* data_t = tensor->data;
	void* data_r = result->data;
	
	switch(tensor->dtype) {
		case FLOAT64: {
			double* cast_data_t = (double*)data_t;	
			double* cast_data_r = (double*)data_r;
			for(int i = 0; i < total_elems; i++) cast_data_r[i] = nnl2_relu_float64(cast_data_t[i]);
			break;
		}
		
		case FLOAT32: {
			float* cast_data_t = (float*)data_t;	
			float* cast_data_r = (float*)data_r;
			for(int i = 0; i < total_elems; i++) cast_data_r[i] = nnl2_relu_float32(cast_data_t[i]);
			break;
		}
		
		case INT32: {
			int32_t* cast_data_t = (int32_t*)data_t;	
			int32_t* cast_data_r = (int32_t*)data_r;
			for(int i = 0; i < total_elems; i++) cast_data_r[i] = nnl2_relu_int32(cast_data_t[i]);
			break;
		}
		
		default: {
			NNL2_TYPE_ERROR(tensor->dtype);
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
 * Threshold for enabling parallel execution of ReLU operation
 */
#define NNL2_RELU_PARALLEL_THRESHOLD 100000

/** @brief
 * Worker function for parallel double precision ReLU
 */
void* nnl2_own_prelu_float64(void* arg);

/** @brief
 * Worker function for parallel single precision ReLU
 */
void* nnl2_own_prelu_float32(void* arg);

/** @brief
 * Worker function for parallel integer ReLU
 */
void* nnl2_own_prelu_int32(void* arg);

/** @brief
 * Ultra-optimized ReLU with non-temporal stores and aggressive vectorization
 */
void* nnl2_own_prelu_float64_nt(void* arg);

void nnl2_own_relu_float64_non_inplace(Tensor* tensor, Tensor* result);

/** @brief
 * High-performance parallel implementation of ReLU activation function
 * 
 ** @param tensor
 * Pointer to the input tensor
 *
 ** @return
 * Pointer to a new tensor containing the ReLU-activated values
 *
 ** @details
 * Uses multi-threading with pthread and AVX256 vectorization for
 * maximum performance on modern CPU architectures. Optimized for
 * both out-of-place and in-place operations.
 */
Tensor* nnl2_own_relu(Tensor* tensor) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        if(tensor == NULL) {
            NNL2_ERROR("Passed tensor is NULL");
            return NULL;
        }
    #endif

    size_t total_elems = product(tensor->shape, tensor->rank);	
    Tensor* result = nnl2_empty(tensor->shape, tensor->rank, tensor->dtype);
    
    if(total_elems == 0) return result;
    
    // For very large tensors, use extreme optimizations
    if(total_elems > 10000000) { // 10M+ elements
        bool src_aligned = NNL2_IS_ALIGNED(tensor->data, NNL2_TENSOR_ALIGNMENT_64);
        bool dst_aligned = NNL2_IS_ALIGNED(result->data, NNL2_TENSOR_ALIGNMENT_64);
        
        if(src_aligned && dst_aligned && tensor->dtype == FLOAT64) {
            // Use extreme optimization for large aligned float64 tensors
            nnl2_own_relu_float64_non_inplace(tensor, result);
            #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                NNL2_FUNC_EXIT();
            #endif
            return result;
        }
    }
    
    // Fallback to naive implementation for small tensors
    if(total_elems < NNL2_RELU_PARALLEL_THRESHOLD) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return naive_relu(tensor);
    }
    
    bool src_aligned = NNL2_IS_ALIGNED(tensor->data, NNL2_TENSOR_ALIGNMENT_32);
    bool dst_aligned = NNL2_IS_ALIGNED(result->data, NNL2_TENSOR_ALIGNMENT_32);
    bool is_aligned = src_aligned && dst_aligned;
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!is_aligned) {
            NNL2_WARN("In nnl2_own_relu, tensor memory is not aligned to 32 bytes. Performance may be suboptimal");
        }
    #endif
    
    size_t num_threads = NNL2_NUM_THREADS;
    
    pthread_t threads[num_threads];
    relu_ptask tasks[num_threads];
    
    // Calculate optimal chunk sizes with load balancing
    size_t chunk = total_elems / num_threads;
    size_t remainder = total_elems % num_threads;
    
    // Configure tasks
    for (size_t i = 0; i < num_threads; i++) {
        tasks[i].dtype = tensor->dtype;
        tasks[i].aligned = is_aligned;
        tasks[i].src_data = tensor->data;
        tasks[i].dst_data = result->data;
        tasks[i].inplace = false;
    }
    
    size_t current_start = 0;
    for (size_t i = 0; i < num_threads; i++) {
        size_t current_chunk = chunk + (i < remainder ? 1 : 0);
        
        tasks[i].start_idx = current_start;
        tasks[i].end_idx = current_start + current_chunk;
        
        // Select appropriate worker function
        void* (*worker_func)(void*) = NULL;
        switch(tensor->dtype) {
            case FLOAT64: {
                worker_func = (total_elems > 10000000 && is_aligned) ? nnl2_own_prelu_float64_nt : nnl2_own_prelu_float64; 
                break;
			}
			
            case FLOAT32: worker_func = nnl2_own_prelu_float32; break;
            case INT32:   worker_func = nnl2_own_prelu_int32;   break;
			
            default: {
                NNL2_TYPE_ERROR(tensor->dtype);
                nnl2_free_tensor(result);
                return NULL;
            }
        }
        
        int status = pthread_create(&threads[i], NULL, worker_func, &tasks[i]);
        if(status != 0) {
            NNL2_THREAD_CREATE_ERROR(status, "nnl2_own_relu");
            num_threads = i;
            break;
        }
        
        current_start += current_chunk;
    }
    
    // Wait for all threads to complete
    for (size_t i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return result;
}

/** @brief
 * Extreme optimization for large float64 tensors
 */
void nnl2_own_relu_float64_non_inplace(Tensor* tensor, Tensor* result) {
    double* src_data = (double*)tensor->data;
    double* dst_data = (double*)result->data;
    size_t total_elems = product(tensor->shape, tensor->rank);
    
    #if defined(NNL2_AVX512_AVAILABLE)
		// AVX-512 version (8 elements per iteration)
		size_t i = 0;
		__m512d v_zero = _mm512_setzero_pd();
		
		for(; i + 7 < total_elems; i += 8) {
			__m512d v_data = _mm512_load_pd(&src_data[i]);
			__m512d v_result = _mm512_max_pd(v_data, v_zero);
			_mm512_stream_pd(&dst_data[i], v_result); // Non-temporal store
		}
		
		// AVX256 for remainder
		if(i + 3 < total_elems) {
			__m256d v_data = _mm256_load_pd(&src_data[i]);
			__m256d v_result = _mm256_max_pd(v_data, _mm256_setzero_pd());
			_mm256_stream_pd(&dst_data[i], v_result); // Non-temporal store
			i += 4;
		}
		
		// Scalar for final elements
		for(; i < total_elems; i++) {
			dst_data[i] = src_data[i] > 0.0 ? src_data[i] : 0.0;
		}
    #elif defined(NNL2_AVX256_AVAILABLE)
		// AVX256 version with prefetching and non-temporal stores
		size_t i = 0;
		__m256d v_zero = _mm256_setzero_pd();
		
		// Main loop with prefetching
		for(; i + 31 < total_elems; i += 32) {
			// Prefetch next cache lines
			_mm_prefetch((char*)&src_data[i + 64], _MM_HINT_T0);
			_mm_prefetch((char*)&dst_data[i + 64], _MM_HINT_T1);
			
			// Process 32 elements (8 AVX operations)
			for(int j = 0; j < 8; j++) {
				__m256d v_data = _mm256_load_pd(&src_data[i + j * 4]);
				__m256d v_result = _mm256_max_pd(v_data, v_zero);
				_mm256_stream_pd(&dst_data[i + j * 4], v_result); // Non-temporal store
			}
		}
		
		// Process remaining elements
		for(; i + 3 < total_elems; i += 4) {
			__m256d v_data = _mm256_load_pd(&src_data[i]);
			__m256d v_result = _mm256_max_pd(v_data, v_zero);
			_mm256_store_pd(&dst_data[i], v_result);
		}
		
		// Scalar for final elements
		for(; i < total_elems; i++) {
			dst_data[i] = src_data[i] > 0.0 ? src_data[i] : 0.0;
		}
    #endif
}

// Worker function implementations with AVX256

void* nnl2_own_prelu_float64(void* arg) {
    relu_ptask* task = (relu_ptask*)arg;
    double* src_data = (double*)task->src_data;
    double* dst_data = (double*)task->dst_data;
    size_t start = task->start_idx;
    size_t end = task->end_idx;
    
    #if defined(NNL2_AVX256_AVAILABLE)
		if(task->aligned && (end - start) >= 4) {
			__m256d v_zero = _mm256_setzero_pd();
			
			// AVX256 processing (4 elements per iteration)
			size_t i = start;
			for(; i + 3 < end; i += 4) {
				__m256d v_data = _mm256_load_pd(&src_data[i]);
				__m256d v_result = _mm256_max_pd(v_data, v_zero);
				_mm256_store_pd(&dst_data[i], v_result);
			}
			
			// Scalar processing for remainder
			for(; i < end; i++) {
				dst_data[i] = src_data[i] > 0.0 ? src_data[i] : 0.0;
			}
		} else {
    #endif
			// Scalar processing for unaligned memory or small chunks
			for(size_t i = start; i < end; i++) {
				dst_data[i] = src_data[i] > 0.0 ? src_data[i] : 0.0;
			}
    #if defined(NNL2_AVX256_AVAILABLE)
		}
    #endif
    
    return NULL;
}

/** @brief
 * Extreme optimization with non-temporal stores
 */
void* nnl2_own_prelu_float64_nt(void* arg) {
    relu_ptask* task = (relu_ptask*)arg;
    double* src_data = (double*)task->src_data;
    double* dst_data = (double*)task->dst_data;
    size_t start = task->start_idx;
    size_t end = task->end_idx;
    
    #if defined(NNL2_AVX256_AVAILABLE)
    __m256d v_zero = _mm256_setzero_pd();
    
    size_t i = start;
    
    // Main loop with non-temporal stores
    for(; i + 31 < end; i += 32) {
        // Process 32 elements at once with prefetching
        for(int j = 0; j < 8; j++) {
            __m256d v_data = _mm256_load_pd(&src_data[i + j * 4]);
            __m256d v_result = _mm256_max_pd(v_data, v_zero);
            _mm256_stream_pd(&dst_data[i + j * 4], v_result); // Non-temporal
        }
    }
    
    // Regular AVX for remainder
    for(; i + 3 < end; i += 4) {
        __m256d v_data = _mm256_load_pd(&src_data[i]);
        __m256d v_result = _mm256_max_pd(v_data, v_zero);
        _mm256_store_pd(&dst_data[i], v_result);
    }
    #endif
    
    // Scalar for final elements
    for(; i < end; i++) {
        dst_data[i] = src_data[i] > 0.0 ? src_data[i] : 0.0;
    }
    
    return NULL;
}

void* nnl2_own_prelu_float32(void* arg) {
    relu_ptask* task = (relu_ptask*)arg;
    float* src_data = (float*)task->src_data;
    float* dst_data = (float*)task->dst_data;
    size_t start = task->start_idx;
    size_t end = task->end_idx;
    
    #if defined(NNL2_AVX256_AVAILABLE)
		if(task->aligned && (end - start) >= 8) {
			__m256 v_zero = _mm256_setzero_ps();
			
			// AVX256 processing (8 elements per iteration)
			size_t i = start;
			for(; i + 7 < end; i += 8) {
				__m256 v_data = _mm256_load_ps(&src_data[i]);
				__m256 v_result = _mm256_max_ps(v_data, v_zero);
				_mm256_store_ps(&dst_data[i], v_result);
			}
			
			// Scalar processing for remainder
			for(; i < end; i++) {
				dst_data[i] = src_data[i] > 0.0f ? src_data[i] : 0.0f;
			}
		} else {
    #endif
			// Scalar processing for unaligned memory or small chunks
			for(size_t i = start; i < end; i++) {
				dst_data[i] = src_data[i] > 0.0f ? src_data[i] : 0.0f;
			}
    #if defined(NNL2_AVX256_AVAILABLE)
		}
    #endif
    
    return NULL;
}

void* nnl2_own_prelu_int32(void* arg) {
    relu_ptask* task = (relu_ptask*)arg;
    int32_t* src_data = (int32_t*)task->src_data;
    int32_t* dst_data = (int32_t*)task->dst_data;
    size_t start = task->start_idx;
    size_t end = task->end_idx;
    
    #if defined(NNL2_AVX256_AVAILABLE)
		if(task->aligned && (end - start) >= 8) {
			__m256i v_zero = _mm256_setzero_si256();
			
			// AVX256 processing (8 elements per iteration)
			size_t i = start;
			for(; i + 7 < end; i += 8) {
				__m256i v_data = _mm256_load_si256((__m256i*)&src_data[i]);
				__m256i v_result = _mm256_max_epi32(v_data, v_zero);
				_mm256_store_si256((__m256i*)&dst_data[i], v_result);
			}
			
			// Scalar processing for remainder
			for(; i < end; i++) {
				dst_data[i] = src_data[i] > 0 ? src_data[i] : 0;
			}
		} else {
    #endif
			// Scalar processing for unaligned memory or small chunks
			for(size_t i = start; i < end; i++) {
				dst_data[i] = src_data[i] > 0 ? src_data[i] : 0;
			}
    #if defined(NNL2_AVX256_AVAILABLE)
		}
    #endif
    
    return NULL;
}


/** @brief
 * Extreme in-place optimization for large float64 tensors
 */
void nnl2_own_relu_inplace_float64_extreme(Tensor* tensor) {
    double* data = (double*)tensor->data;
    size_t total_elems = product(tensor->shape, tensor->rank);
    
    #if defined(NNL2_AVX512_AVAILABLE)
		// AVX-512 version
		size_t i = 0;
		__m512d v_zero = _mm512_setzero_pd();
		
		for(; i + 7 < total_elems; i += 8) {
			__m512d v_data = _mm512_load_pd(&data[i]);
			__m512d v_result = _mm512_max_pd(v_data, v_zero);
			_mm512_store_pd(&data[i], v_result);
		}	
    #elif defined(NNL2_AVX256_AVAILABLE)
		// AVX256 version with aggressive unrolling
		size_t i = 0;
		__m256d v_zero = _mm256_setzero_pd();
		
		// Process 64 elements at a time (16 AVX operations)
		for(; i + 63 < total_elems; i += 64) {
			// Unroll 16 times
			for(int j = 0; j < 16; j++) {
				__m256d v_data = _mm256_load_pd(&data[i + j * 4]);
				__m256d v_result = _mm256_max_pd(v_data, v_zero);
				_mm256_store_pd(&data[i + j * 4], v_result);
			}
		}
		
		// Process remaining elements
		for(; i + 3 < total_elems; i += 4) {
			__m256d v_data = _mm256_load_pd(&data[i]);
			__m256d v_result = _mm256_max_pd(v_data, v_zero);
			_mm256_store_pd(&data[i], v_result);
		}
    #endif
    
    // Scalar for final elements
    for(; i < total_elems; i++) {
        data[i] = data[i] > 0.0 ? data[i] : 0.0;
    }
}

#endif

/**
 * @ingroup backend_system
 * @brief Backend implementations for ReLU operation
 * @details
 * Array follows the common backend registration pattern for ReLU operations.
 * Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for ReLU activation function
 * 
 * @see nnl2_naive
 * @see naive_relu
 */
Implementation relu_backends[] = {
	REGISTER_BACKEND(naive_relu, nnl2_naive, NAIVE_BACKEND_NAME),
	
	#ifdef NNL2_PTHREAD_AVAILABLE
	    REGISTER_BACKEND(nnl2_own_relu, nnl2_own, NNL2_OWN_NAME),
	#endif
};	

/**
 * @brief Function pointer for ReLU operation
 * @ingroup backend_system 
 */
relufn relu;

/** 
 * @brief Makes the relu backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(relu);

/** 
 * @brief Sets the backend for ReLU operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for ReLU
 * @see ESET_BACKEND_BY_NAME
 */
void set_relu_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(relu_backends, relu, backend_name, current_backend(relu));
}

/** 
 * @brief Gets the name of the active backend for ReLU operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_relu_backend() {
	return current_backend(relu);
}

/** 
 * @brief Function declaration for getting all available ReLU backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(relu);

/**
 * @brief Function declaration for getting the number of available ReLU backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(relu);

#endif /** NNL2_RELU_H **/
