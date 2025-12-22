#ifndef NNL2_LEAKY_RELU_INPLACE_H
#define NNL2_LEAKY_RELU_INPLACE_H

/** @brief
 * Threshold for enabling parallel execution of the
 * Leaky ReLU operation during in-place calculations
 */
#define NNL2_LEAKY_RELU_INPLACE_PARALLEL_THREASHOLD 1000000

/** @brief
 * Applies Leaky ReLU (max(alpha * x, x)) function to an in-place tensor
 *
 * Example 1: leakyrelu(1) -> 1
 * Example 2: leakyrelu(-1, alpha=0.01) -> -0.01
 *	
 ** @param tensor
 * A pointer to a tensor for modification
 *
 ** @param alpha
 * Slope coefficient for negative values (usually a small positive number)
 */
void naive_leakyreluinplace(Tensor* tensor, float alpha) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	size_t total_elems = nnl2_product(tensor->shape, tensor->rank);	
	if(total_elems == 0) return; // If tensor is empty return
	
	void* data = tensor->data;
	
	switch(tensor->dtype) {
		case FLOAT64: {
			double* cast_data = (double*)data;	
			for(size_t i = 0; i < total_elems; i++) nnl2_leaky_relu_float64_inplace(&cast_data[i], alpha);
			break;
		}
		
		case FLOAT32: {
			float* cast_data = (float*)data;	
			for(size_t i = 0; i < total_elems; i++) nnl2_leaky_relu_float32_inplace(&cast_data[i], alpha);
			break;
		}
		
		case INT32: {
			int32_t* cast_data = (int32_t*)data;	
			for(size_t i = 0; i < total_elems; i++) nnl2_leaky_relu_int32_inplace(&cast_data[i], alpha);
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

#if defined(NNL2_PTHREAD_AVAILABLE) && defined(NNL2_AVX256_AVAILABLE)

///@{ [lreluavx]

/** @brief
 * Applies Leaky ReLU to a double (float64) array using AVX256 (unaligned memory)
 * 
 ** @param data 
 * Pointer to the array of double data
 *
 ** @param size 
 * Size of the array
 *
 ** @param alpha 
 * Slope coefficient for negative values (typically 0.01)
 */
void nnl2_leaky_relu_avx_f64(double* data, size_t size, double alpha);

/** @brief 
 * Applies Leaky ReLU to a float (float32) array using AVX256 (unaligned memory)
 * 
 ** @param data 
 * Pointer to the array of float data
 *
 ** @param size 
 * Size of the array
 *
 ** @param alpha 
 * Slope coefficient for negative values (typically 0.01)
 */
void nnl2_leaky_relu_avx256_f32(float* data, size_t size, float alpha);

/** @brief 
 * Applies Leaky ReLU to a double (float64) array using AVX (aligned memory)
 * 
 ** @param data 
 * Pointer to the aligned array of double data (must be 32-byte aligned)
 *
 ** @param size 
 * Size of the array
 *
 ** @param alpha 
 * Slope coefficient for negative values (typically 0.01)
 * 
 ** @warning 
 * Requires aligned memory
 */
void nnl2_leaky_relu_avx_f64_align(double* data, size_t size, double alpha);

/** @brief 
 * Applies Leaky ReLU to a float (float32) array using AVX (aligned memory)
 * 
 ** @param data 
 * Pointer to the aligned array of float data (must be 32-byte aligned)
 *
 ** @param size 
 * Size of the array
 *
 ** @param alpha 
 * Slope coefficient for negative values (typically 0.01)
 * 
 * @warning 
 * Requires aligned memory
 */
void nnl2_leaky_relu_avx256_f32_align(float* data, size_t size, float alpha);

///@} [lreluavx]

/** @brief
 * See docs at declaration
 *
 ** @see nnl2_leaky_relu_avx_f64 (declaration)
 **/
void nnl2_leaky_relu_avx_f64(double* data, size_t size, double alpha) {
    // Broadcast alpha value to all elements of AVX vector
    __m256d alpha_vec = _mm256_set1_pd(alpha);
    // Create zero vector for comparison
    __m256d zero = _mm256_setzero_pd();
    
    size_t i = 0;
	
    // Process 4 double elements per iteration using AVX
    for (; i + 3 < size; i += 4) {
        // Load 4 unaligned double values from memory
        __m256d x = _mm256_loadu_pd(data + i);
        // 0xFF for elements > 0, 0x00 otherwise
        __m256d mask = _mm256_cmp_pd(x, zero, _CMP_GT_OS);
        // For positive elements use x, for negative use alpha*x
        __m256d result = _mm256_blendv_pd(_mm256_mul_pd(alpha_vec, x), x, mask);
        // Store result back to memory (unaligned)
        _mm256_storeu_pd(data + i, result);
    }
   
    for (; i < size; i++) {
        nnl2_leaky_relu_float64_inplace(&data[i], alpha);
    }
}

/** @brief
 * See docs at declaration
 *
 ** @see nnl2_leaky_relu_avx_f32 (declaration)
 **/
void nnl2_leaky_relu_avx256_f32(float* data, size_t size, float alpha) {
    // Broadcast alpha value to all elements of AVX vector
    __m256 alpha_vec = _mm256_set1_ps(alpha);
    // Create zero vector for comparison
    __m256 zero = _mm256_setzero_ps();
    
    size_t i = 0;

    // Process 8 float elements per iteration using AVX
    for (; i + 7 < size; i += 8) {
        // Load 8 unaligned float values from memory
        __m256 x = _mm256_loadu_ps(data + i);
        // 0xFF for elements > 0, 0x00 otherwise
        __m256 mask = _mm256_cmp_ps(x, zero, _CMP_GT_OS);
        // for positive elements use x, for negative use alpha*x
        __m256 result = _mm256_blendv_ps(_mm256_mul_ps(alpha_vec, x), x, mask);
        // Store result back to memory (unaligned)
        _mm256_storeu_ps(data + i, result);
    }
    
    for (; i < size; i++) {
        nnl2_leaky_relu_float32_inplace(&data[i], alpha);
    }
}

/** @brief
 * See docs at declaration
 *
 ** @see nnl2_leaky_relu_avx_f64_align (declaration)
 **/
void nnl2_leaky_relu_avx_f64_align(double* data, size_t size, double alpha) {
    // Broadcast alpha value to all elements of AVX vector
    __m256d alpha_vec = _mm256_set1_pd(alpha);
    // Create zero vector for comparison
    __m256d zero = _mm256_setzero_pd();
    
    size_t i = 0;
	
    // Process 4 double elements per iteration using AVX (aligned memory)
    for (; i + 3 < size; i += 4) {
        // Load 4 aligned double values from memory (requires 32-byte alignment)
        __m256d x = _mm256_load_pd(data + i);
        // 0xFF for elements > 0, 0x00 otherwise
        __m256d mask = _mm256_cmp_pd(x, zero, _CMP_GT_OS);
        // for positive elements use x, for negative use alpha*x
        __m256d result = _mm256_blendv_pd(_mm256_mul_pd(alpha_vec, x), x, mask);
        // Store result back to memory (aligned)
        _mm256_store_pd(data + i, result);
    }
    
    for (; i < size; i++) {
        nnl2_leaky_relu_float64_inplace(&data[i], alpha);
    }
}

/** @brief
 * See docs at declaration
 *
 ** @see nnl2_leaky_relu_avx_f32_align (declaration)
 **/
void nnl2_leaky_relu_avx256_f32_align(float* data, size_t size, float alpha) {
    // Broadcast alpha value to all elements of AVX vector
    __m256 alpha_vec = _mm256_set1_ps(alpha);
    // Create zero vector for comparison
    __m256 zero = _mm256_setzero_ps();
    
    size_t i = 0;

    // Process 8 float elements per iteration using AVX (aligned memory)
    for (; i + 7 < size; i += 8) {
        // Load 8 aligned float values from memory (requires 32-byte alignment)
        __m256 x = _mm256_load_ps(data + i);
        // 0xFF for elements > 0, 0x00 otherwise
        __m256 mask = _mm256_cmp_ps(x, zero, _CMP_GT_OS);
        // for positive elements use x, for negative use alpha*x
        __m256 result = _mm256_blendv_ps(_mm256_mul_ps(alpha_vec, x), x, mask);
        // Store result back to memory (aligned)
        _mm256_store_ps(data + i, result);
    }
    
    for (; i < size; i++) {
        nnl2_leaky_relu_float32_inplace(&data[i], alpha);
    }
}
	
/** @brief 
 * Multithreaded in-place Leaky ReLU for double (float64) precision floating-point arrays
 * 
 * @param data 
 * Pointer to double array
 *
 * @param total_size 
 * Total number of elements in the array
 *
 * @param alpha
 * Negative slope coefficient
 *
 * @param nthreads 
 * Number of threads for parallelization
 *
 * @return NULL (for pthread api)
 */
void* nnl2_own_leaky_relu_inplace_float64(double* data, size_t total_size, float alpha, size_t nthreads, bool aligned);

/** @brief 
 * Similarly nnl2_own_leaky_relu_inplace_float64 but for float32
 *
 ** @see nnl2_own_leaky_relu_inplace_float64
 **/
void* nnl2_own_leaky_relu_inplace_float32(float* data, size_t total_size, float alpha, size_t nthreads, bool aligned);

/** @brief 
 * Similarly nnl2_own_leaky_relu_inplace_float64 but for int32
 *
 ** @see nnl2_own_leaky_relu_inplace_float64
 **/
void* nnl2_own_leaky_relu_inplace_int32(int32_t* data, size_t total_size, float alpha, size_t nthreads);

/** @brief 
 * Worker function wrapper for parallel Leaky ReLU execution on float64 arrays
 * 
 * @param arg 
 * Pointer to single_arr_ptask structure containing task parameters
 *
 * @return NULL (for pthread api)
 */
void* nnl2_own_pleaky_relu_inplace_float64(void* arg);

/** @brief 
 * Worker function wrapper for parallel Leaky ReLU execution on aligned float64 arrays
 * 
 * @param arg 
 * Pointer to single_arr_ptask structure containing task parameters
 *
 * @return NULL (for pthread api)
 */
void* nnl2_own_pleaky_relu_inplace_float64_align(void* arg);

/** @brief 
 * Similarly nnl2_own_pleaky_relu_inplace_float64 but for float32
 *
 ** @see nnl2_own_pleaky_relu_inplace_float64
 **/
void* nnl2_own_pleaky_relu_inplace_float32(void* arg);

/** @brief 
 * Similarly nnl2_own_pleaky_relu_inplace_float64_align but for float32
 *
 ** @see nnl2_own_pleaky_relu_inplace_float64_align
 **/
void* nnl2_own_pleaky_relu_inplace_float32_align(void* arg);

/** @brief 
 * Similarly nnl2_own_pleaky_relu_inplace_float64 but for int32
 *
 ** @see nnl2_own_pleaky_relu_inplace_float64
 **/
void* nnl2_own_pleaky_relu_inplace_int32(void* arg);

/** @brief
 * Main function for applying Leaky ReLU activation to tensor in-place
 * 
 ** @param tensor 
 * Pointer to tensor to be modified
 *
 ** @param alpha
 * Slope coefficient for negative values
 */
void nnl2_own_leaky_relu_inplace(Tensor* tensor, float alpha) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	size_t total_elems = nnl2_product(tensor->shape, tensor->rank);	
	bool tensor_aligned = NNL2_IS_ALIGNED(tensor->data, NNL2_TENSOR_ALIGNMENT_32);
	
	// Warning for unaligned memory in safety modes (performance impact)
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if((!tensor_aligned) && (tensor->dtype != INT32)) {
			NNL2_WARN("In the own nnl2 implementation of Leaky-ReLU, tensor memory is not aligned to 32 bytes. Calculations may be slightly slower");
		}
	#endif
	
	if(total_elems == 0) {
		return; // If tensor is empty return
	} else if (total_elems < NNL2_LEAKY_RELU_INPLACE_PARALLEL_THREASHOLD) {
		naive_leakyreluinplace(tensor, alpha);
		#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
			NNL2_FUNC_EXIT();
		#endif
		return;
	}
	
	void* data = tensor->data;
	
	switch(tensor->dtype) {
		case FLOAT64: nnl2_own_leaky_relu_inplace_float64((double*)data, total_elems, alpha, NNL2_NUM_THREADS, tensor_aligned);  break;
		case FLOAT32: nnl2_own_leaky_relu_inplace_float32((float*)data, total_elems, alpha, NNL2_NUM_THREADS, tensor_aligned);   break;
		case INT32:   nnl2_own_leaky_relu_inplace_int32((int32_t*)data, total_elems, alpha, NNL2_NUM_THREADS);   				 break;
		
		default: {
			NNL2_TYPE_ERROR(tensor->dtype);
			return;
		}
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

/** @brief
 * See docs at declaration
 *
 ** @see nnl2_own_leaky_relu_inplace_float64
 **/
void* nnl2_own_leaky_relu_inplace_float64(double* data, size_t total_size, float alpha, size_t num_threads, bool aligned) {
    // Allocate arrays for thread handles and task descriptors
    pthread_t threads[num_threads];
    leaky_relu_single_arr_ptask tasks[num_threads];
    
    // Calculate base chunk size and remainder for balanced distribution
    size_t chunk = total_size / num_threads;
    size_t remainder = total_size % num_threads;
    
    // Distribute work among threads with load balancing
    size_t current_start = 0;
    for (size_t i = 0; i < num_threads; i++) {
        size_t current_chunk = chunk + (i < remainder ? 1 : 0);
        
        // Configure task for this thread
        tasks[i].data = data;
        tasks[i].start = current_start;
        tasks[i].end = current_start + current_chunk;
        tasks[i].alpha = alpha;
        
        // Create thread to process the assigned chunk
		int status;
		if(aligned) {
		    status = pthread_create(&threads[i], NULL, nnl2_own_pleaky_relu_inplace_float64_align, &tasks[i]);
		} else {
			status = pthread_create(&threads[i], NULL, nnl2_own_pleaky_relu_inplace_float64, &tasks[i]);
		}
		
        if(status != 0) {
			NNL2_THREAD_CREATE_ERROR(status, "nnl2_own_leaky_relu_inplace_float64");
			num_threads = i;
			break;
		}
		
        current_start += current_chunk;
    }
    
    // Wait for all threads to complete their work
    for (size_t i = 0; i < num_threads; i++) {
		int join_status = pthread_join(threads[i], NULL);
        if(join_status != 0) {
			NNL2_THREAD_JOIN_ERROR(join_status, "nnl2_own_leaky_relu_inplace_float64");
		}
    }
    
    return NULL;
}

/** @brief
 * See docs at declaration
 *
 ** @see nnl2_own_leaky_relu_inplace_float32
 **/
void* nnl2_own_leaky_relu_inplace_float32(float* data, size_t total_size, float alpha, size_t num_threads, bool aligned) {
    // Allocate arrays for thread handles and task descriptors
    pthread_t threads[num_threads];
    leaky_relu_single_arr_ptask tasks[num_threads];
    
    // Calculate base chunk size and remainder for balanced distribution
    size_t chunk = total_size / num_threads;
    size_t remainder = total_size % num_threads;
    
    // Distribute work among threads with load balancing
    size_t current_start = 0;
    for (size_t i = 0; i < num_threads; i++) {
        size_t current_chunk = chunk + (i < remainder ? 1 : 0);
        
        // Configure task for this thread
        tasks[i].data = data;
        tasks[i].start = current_start;
        tasks[i].end = current_start + current_chunk;
        tasks[i].alpha = alpha;
        
        // Create thread to process the assigned chunk
        int status;
		if(aligned) {
			status = pthread_create(&threads[i], NULL, nnl2_own_pleaky_relu_inplace_float32_align, &tasks[i]);
		} else {
			status = pthread_create(&threads[i], NULL, nnl2_own_pleaky_relu_inplace_float32, &tasks[i]);
		}
		
        if(status != 0) {
            NNL2_THREAD_CREATE_ERROR(status, "nnl2_own_leaky_relu_inplace_float32");
            num_threads = i;
            break;
        }
        
        current_start += current_chunk;
    }
    
    // Wait for all threads to complete their work
    for (size_t i = 0; i < num_threads; i++) {
        int join_status = pthread_join(threads[i], NULL);
        if(join_status != 0) {
            NNL2_THREAD_JOIN_ERROR(join_status, "nnl2_own_leaky_relu_inplace_float32");
        }
    }
    
    return NULL;
}

/** @brief
 * See docs at declaration
 *
 ** @see nnl2_own_leaky_relu_inplace_int32
 **/    
void* nnl2_own_leaky_relu_inplace_int32(int32_t* data, size_t total_size, float alpha, size_t num_threads) {
    // Allocate arrays for thread handles and task descriptors
    pthread_t threads[num_threads];
    leaky_relu_single_arr_ptask tasks[num_threads];
    
    // Calculate base chunk size and remainder for balanced distribution
    size_t chunk = total_size / num_threads;
    size_t remainder = total_size % num_threads;
    
    // Distribute work among threads with load balancing
    size_t current_start = 0;
    for (size_t i = 0; i < num_threads; i++) {
        size_t current_chunk = chunk + (i < remainder ? 1 : 0);
        
        // Configure task for this thread
        tasks[i].data = data;
        tasks[i].start = current_start;
        tasks[i].end = current_start + current_chunk;
        tasks[i].alpha = alpha;
        
        // Create thread to process the assigned chunk
        int status = pthread_create(&threads[i], NULL, nnl2_own_pleaky_relu_inplace_int32, &tasks[i]);
        if(status != 0) {
            NNL2_THREAD_CREATE_ERROR(status, "nnl2_own_leaky_relu_inplace_int32");
            num_threads = i;
            break;
        }
        
        current_start += current_chunk;
    }
    
    // Wait for all threads to complete their work
    for (size_t i = 0; i < num_threads; i++) {
        int join_status = pthread_join(threads[i], NULL);
        if(join_status != 0) {
            NNL2_THREAD_JOIN_ERROR(join_status, "nnl2_own_leaky_relu_inplace_int32");
        }
    }
    
    return NULL;
}

/** @brief
 * See docs at declaration
 *
 ** @see nnl2_own_pleaky_relu_inplace_float64
 **/
void* nnl2_own_pleaky_relu_inplace_float64(void* arg) {
    leaky_relu_single_arr_ptask* task = (leaky_relu_single_arr_ptask*)arg;
    double* input = (double*)task->data;
    size_t start = task->start;
    size_t end = task->end;
    float alpha = task->alpha;
    
    size_t chunk_size = end - start;
    
    #ifdef NNL2_AVX256_AVAILABLE
		if(chunk_size >= 4) {
			nnl2_leaky_relu_avx_f64(input + start, chunk_size, (double)alpha);
		} else {
			for (size_t i = start; i < end; i++) {
				nnl2_leaky_relu_float64_inplace(&input[i], alpha);
			}
		}
    #else
		for (size_t i = start; i < end; i++) {
			nnl2_leaky_relu_float64_inplace(&input[i], alpha);
		}
    #endif
    
    return NULL;
}

/** @brief
 * See docs at declaration
 *
 ** @see nnl2_own_pleaky_relu_inplace_float64
 **/
void* nnl2_own_pleaky_relu_inplace_float64_align(void* arg) {
    leaky_relu_single_arr_ptask* task = (leaky_relu_single_arr_ptask*)arg;
    double* input = (double*)task->data;
    size_t start = task->start;
    size_t end = task->end;
    float alpha = task->alpha;
    
    size_t chunk_size = end - start;
    
    #ifdef NNL2_AVX256_AVAILABLE
		if(chunk_size >= 4) {
			nnl2_leaky_relu_avx_f64_align(input + start, chunk_size, (double)alpha);
		} else {
			for (size_t i = start; i < end; i++) {
				nnl2_leaky_relu_float64_inplace(&input[i], alpha);
			}
		}
    #else
		for (size_t i = start; i < end; i++) {
			nnl2_leaky_relu_float64_inplace(&input[i], alpha);
		}
    #endif
    
    return NULL;
}

/** @brief
 * See docs at declaration
 *
 ** @see nnl2_own_pleaky_relu_inplace_float32
 **/
void* nnl2_own_pleaky_relu_inplace_float32(void* arg) {
    leaky_relu_single_arr_ptask* task = (leaky_relu_single_arr_ptask*)arg;
    float* input = (float*)task->data;
    size_t start = task->start;
    size_t end = task->end;
    float alpha = task->alpha;
    
    size_t chunk_size = end - start;
    
    #ifdef NNL2_AVX256_AVAILABLE
		if(chunk_size >= 8) {
			nnl2_leaky_relu_avx256_f32(input + start, chunk_size, alpha);
		} else {
			for (size_t i = start; i < end; i++) {
				nnl2_leaky_relu_float32_inplace(&input[i], alpha);
			}
		}
    #else
		for (size_t i = start; i < end; i++) {
			nnl2_leaky_relu_float32_inplace(&input[i], alpha);
		}
    #endif
    
    return NULL;
}

/** @brief
 * See docs at declaration
 *
 ** @see nnl2_own_pleaky_relu_inplace_float32
 **/
void* nnl2_own_pleaky_relu_inplace_float32_align(void* arg) {
    leaky_relu_single_arr_ptask* task = (leaky_relu_single_arr_ptask*)arg;
    float* input = (float*)task->data;
    size_t start = task->start;
    size_t end = task->end;
    float alpha = task->alpha;
    
    size_t chunk_size = end - start;
    
    #ifdef NNL2_AVX256_AVAILABLE
		if(chunk_size >= 8) {
			nnl2_leaky_relu_avx256_f32_align(input + start, chunk_size, alpha);
		} else {
			for (size_t i = start; i < end; i++) {
				nnl2_leaky_relu_float32_inplace(&input[i], alpha);
			}
		}
    #else
		for (size_t i = start; i < end; i++) {
			nnl2_leaky_relu_float32_inplace(&input[i], alpha);
		}
    #endif
    
    return NULL;
}

/** @brief
 * See docs at declaration
 *
 ** @see nnl2_own_pleaky_relu_inplace_int32
 **/
void* nnl2_own_pleaky_relu_inplace_int32(void* arg) {
    // Extract task parameters from argument
    leaky_relu_single_arr_ptask* task = (leaky_relu_single_arr_ptask*)arg;
    int32_t* input = (int32_t*)task->data;
    
    // Apply Leaky ReLU activation to each element in the assigned range
    for (size_t i = task->start; i < task->end; i++) {
        nnl2_leaky_relu_int32_inplace(&input[i], task->alpha);
    }
    
    return NULL;
}

#endif

/**
 * @ingroup backend_system
 * @brief Backend implementations for LeakyReLU in-place operation
 * @details
 * Array follows the common backend registration pattern for LeakyReLU in-place operations.
 * Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for LeakyReLU activation function
 * 
 * @see nnl2_naive
 * @see naive_leakyreluinplace
 */
Implementation leakyreluinplace_backends[] = {
	REGISTER_BACKEND(naive_leakyreluinplace, nnl2_naive, NAIVE_BACKEND_NAME),
	
	#ifdef NNL2_PTHREAD_AVAILABLE
		REGISTER_BACKEND(nnl2_own_leaky_relu_inplace, nnl2_own, NNL2_OWN_NAME),
	#endif
};	

/**
 * @brief Function pointer for LeakyReLU in-place operation
 * @ingroup backend_system 
 */
leakyreluinplacefn leakyreluinplace;

/** 
 * @brief Makes the leakyreluinplace backend current
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(leakyreluinplace);

/** 
 * @brief Sets the backend for LeakyReLU in-place operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for LeakyReLU in-place
 * @see ESET_BACKEND_BY_NAME
 */
void set_leakyreluinplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(leakyreluinplace_backends, leakyreluinplace, backend_name, CURRENT_BACKEND(leakyreluinplace));
}

/** 
 * @brief Gets the name of the active backend for LeakyReLU in-place operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_leakyreluinplace_backend() {
	return CURRENT_BACKEND(leakyreluinplace);
}

/** 
 * @brief Function declaration for getting all available LeakyReLU in-place backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(leakyreluinplace);

/**
 * @brief Function declaration for getting the number of available LeakyReLU in-place backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(leakyreluinplace);

#endif /** NNL2_LEAKY_RELU_INPLACE_H **/
