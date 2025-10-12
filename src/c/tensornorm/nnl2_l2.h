#ifndef NNL2_L2_H
#define NNL2_L2_H

/** @brief
 * Computes the L2 norm (Euclidean norm) of a tensor
 *
 ** @param tensor
 * Pointer to the input tensor
 *
 ** @param axes
 * Array of axes for reduction
 *
 ** @param num_axes
 * Number of axes for reduction
 *
 ** @param result
 * Pointer to memory for storing the result
 *
 ** @see sqrt
 ** @see sqrtf
 **/
void naive_l2norm(Tensor* tensor, int* axes, int num_axes, void* result) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	bool reduce_all = false;
	
	if(num_axes == 1 && axes[0] == 0) {
		reduce_all = true;
	} 
	
	if(reduce_all) {
		size_t total_elems = product(tensor->shape, tensor->rank);
		
		switch(tensor->dtype) {
			case FLOAT64: {
                double* cast_data = (double*)tensor->data;
                double acc = 0.0;
                for (size_t it = 0; it < total_elems; it++) {
                    double val = cast_data[it];
                    acc += val * val;
                }
                *((double*)result) = sqrt(acc); // Store result in output pointer
                break;
            }
    
            case FLOAT32: {
                float* cast_data = (float*)tensor->data;
                float acc = 0.0f;
                for (size_t it = 0; it < total_elems; it++) {
                    float val = cast_data[it];
                    acc += val * val;
                }
                *((float*)result) = sqrtf(acc); 
                break;
            }
            
            case INT32: {
                int32_t* cast_data = (int32_t*)tensor->data;
                int32_t acc = 0;
                for (size_t it = 0; it < total_elems; it++) {
                    int32_t val = cast_data[it];
                    acc += val * val; 
                }
                *((int32_t*)result) = (int32_t)sqrt(acc);  
                break;
            }
			
			default: {
				NNL2_TYPE_ERROR(tensor->dtype);
				return;
			}
		}
	} else {
		NNL2_ERROR("Norm axes in development");
		
		// I don't plan to add support for axes 
		// in the near future because I don't need it
		
		return;
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

#ifdef NNL2_PTHREAD_AVAILABLE

/** @brief
 * Threshold for enabling parallel execution of L2 norm operation
 */
#define NNL2_L2NORM_PARALLEL_THRESHOLD 100000

/** @brief
 * Worker function for parallel double precision L2 norm
 */
void* nnl2_own_pl2norm_float64(void* arg);

/** @brief
 * Worker function for parallel single precision L2 norm
 */
void* nnl2_own_pl2norm_float32(void* arg);

/** @brief
 * Worker function for parallel integer L2 norm
 */
void* nnl2_own_pl2norm_int32(void* arg);

/** @brief
 * High-performance parallel implementation of L2 norm computation
 * 
 ** @param tensor
 * Pointer to the input tensor
 *
 ** @param axes
 * Array of axes for reduction
 *
 ** @param num_axes
 * Number of axes for reduction
 *
 ** @param result
 * Pointer to memory for storing the result
 *
 ** @details
 * Uses multi-threading with pthread and AVX256 vectorization for
 * maximum performance on modern CPU architectures. Implements
 * numerical optimizations for improved accuracy and speed.
 */
void nnl2_own_l2norm(Tensor* tensor, int* axes, int num_axes, void* result) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    bool reduce_all = false;
    
    if(num_axes == 1 && axes[0] == 0) {
        reduce_all = true;
    } 
    
    if(!reduce_all) {
        NNL2_ERROR("Norm axes in development");
        return;
    }
    
    size_t total_elems = product(tensor->shape, tensor->rank);
    if(total_elems == 0) {
        switch(tensor->dtype) {
            case FLOAT64: *((double*)result) = 0.0; break;
            case FLOAT32: *((float*)result) = 0.0f; break;
            case INT32: *((int32_t*)result) = 0; break;
            default: break;
        }
        return;
    }
    
    // Fallback to naive implementation for small tensors
    if(total_elems < NNL2_L2NORM_PARALLEL_THRESHOLD) {
        naive_l2norm(tensor, axes, num_axes, result);
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return;
    }
    
    bool is_aligned = NNL2_IS_ALIGNED(tensor->data, NNL2_TENSOR_ALIGNMENT_32);
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!is_aligned) {
            NNL2_WARN("In nnl2_own_l2norm, tensor memory is not aligned to 32 bytes. Performance may be suboptimal");
        }
    #endif
    
    size_t num_threads = NNL2_NUM_THREADS;
    pthread_t threads[num_threads];
    l2norm_ptask tasks[num_threads];
    
    // Calculate optimal chunk sizes with load balancing
    size_t chunk = total_elems / num_threads;
    size_t remainder = total_elems % num_threads;
    
    // Configure tasks
    for (size_t i = 0; i < num_threads; i++) {
        tasks[i].dtype = tensor->dtype;
        tasks[i].aligned = is_aligned;
        tasks[i].src_data = tensor->data;
        
        // Initialize accumulators
        switch(tensor->dtype) {
            case FLOAT64: tasks[i].accumulator.float64_acc = 0.0; break;
            case FLOAT32: tasks[i].accumulator.float32_acc = 0.0f; break;
            case INT32:   tasks[i].accumulator.int32_acc = 0; break;
            default: break;
        }
    }
    
    size_t current_start = 0;
    for (size_t i = 0; i < num_threads; i++) {
        size_t current_chunk = chunk + (i < remainder ? 1 : 0);
        
        tasks[i].start_idx = current_start;
        tasks[i].end_idx = current_start + current_chunk;
        
        // Select appropriate worker function
        void* (*worker_func)(void*) = NULL;
        switch(tensor->dtype) {
            case FLOAT64: worker_func = nnl2_own_pl2norm_float64; break;
            case FLOAT32: worker_func = nnl2_own_pl2norm_float32; break;
            case INT32:   worker_func = nnl2_own_pl2norm_int32;   break;
            default: {
                NNL2_TYPE_ERROR(tensor->dtype);
                #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                    NNL2_FUNC_EXIT();
                #endif
                return;
            }
        }
        
        int status = pthread_create(&threads[i], NULL, worker_func, &tasks[i]);
        if(status != 0) {
            NNL2_THREAD_CREATE_ERROR(status, "nnl2_own_l2norm");
            num_threads = i;
            break;
        }
        
        current_start += current_chunk;
    }
    
    // Wait for all threads to complete and accumulate squared sums
    switch(tensor->dtype) {
        case FLOAT64: {
            double total_squared = 0.0;
            for (size_t i = 0; i < num_threads; i++) {
                pthread_join(threads[i], NULL);
                total_squared += tasks[i].accumulator.float64_acc;
            }
			
            *((double*)result) = sqrt(total_squared);
            break;
        }
		
        case FLOAT32: {
            float total_squared = 0.0f;
            for (size_t i = 0; i < num_threads; i++) {
                pthread_join(threads[i], NULL);
                total_squared += tasks[i].accumulator.float32_acc;
            }
			
            *((float*)result) = sqrtf(total_squared);
            break;
        }
		
        case INT32: {
            int32_t total_squared = 0;
            for (size_t i = 0; i < num_threads; i++) {
                pthread_join(threads[i], NULL);
                total_squared += tasks[i].accumulator.int32_acc;
            }
			
            *((int32_t*)result) = (int32_t)sqrt(total_squared);
            break;
        }
		
        default: break;
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

// Worker function implementations with AVX256 and numerical optimizations

void* nnl2_own_pl2norm_float64(void* arg) {
    l2norm_ptask* task = (l2norm_ptask*)arg;
    double* data = (double*)task->src_data;
    size_t start = task->start_idx;
    size_t end = task->end_idx;
    
    double sum_squared = 0.0;
    size_t i = start;
    
    #if defined(NNL2_AVX256_AVAILABLE)
		if(task->aligned && (end - start) >= 4) {
			__m256d v_sum = _mm256_setzero_pd();
			
			// AVX256 processing (4 elements per iteration)
			for(; i + 3 < end; i += 4) {
				__m256d v_data = _mm256_load_pd(&data[i]);
				__m256d v_squared = _mm256_mul_pd(v_data, v_data);
				v_sum = _mm256_add_pd(v_sum, v_squared);
			}
			
			// Horizontal sum of AVX vector
			double temp[4] __attribute__((aligned(32)));
			_mm256_store_pd(temp, v_sum);
			sum_squared = temp[0] + temp[1] + temp[2] + temp[3];
		}
    #endif
    
    // Scalar processing for remainder with Kahan summation for better accuracy
    double compensation = 0.0;
    for(; i < end; i++) {
        double val = data[i];
        double squared = val * val;
        
        // Kahan summation for improved numerical accuracy
        double y = squared - compensation;
        double t = sum_squared + y;
        compensation = (t - sum_squared) - y;
        sum_squared = t;
    }
    
    task->accumulator.float64_acc = sum_squared;
    return NULL;
}

void* nnl2_own_pl2norm_float32(void* arg) {
    l2norm_ptask* task = (l2norm_ptask*)arg;
    float* data = (float*)task->src_data;
    size_t start = task->start_idx;
    size_t end = task->end_idx;
    
    float sum_squared = 0.0f;
    size_t i = start;
    
    #if defined(NNL2_AVX256_AVAILABLE)
		if(task->aligned && (end - start) >= 8) {
			__m256 v_sum = _mm256_setzero_ps();
			
			// AVX256 processing (8 elements per iteration)
			for(; i + 7 < end; i += 8) {
				__m256 v_data = _mm256_load_ps(&data[i]);
				__m256 v_squared = _mm256_mul_ps(v_data, v_data);
				v_sum = _mm256_add_ps(v_sum, v_squared);
			}
			
			// Horizontal sum of AVX vector
			float temp[8] __attribute__((aligned(32)));
			_mm256_store_ps(temp, v_sum);
			sum_squared = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
		}
    #endif
    
    float compensation = 0.0f;
    for(; i < end; i++) {
        float val = data[i];
        float squared = val * val;
		
        float y = squared - compensation;
        float t = sum_squared + y;
        compensation = (t - sum_squared) - y;
        sum_squared = t;
    }
    
    task->accumulator.float32_acc = sum_squared;
    return NULL;
}

void* nnl2_own_pl2norm_int32(void* arg) {
    l2norm_ptask* task = (l2norm_ptask*)arg;
    int32_t* data = (int32_t*)task->src_data;
    size_t start = task->start_idx;
    size_t end = task->end_idx;
    
    int64_t sum_squared = 0;  // Use int64_t to avoid overflow
    size_t i = start;
    
    #if defined(NNL2_AVX256_AVAILABLE)
		if(task->aligned && (end - start) >= 8) {
			// For integers, we process in smaller chunks to avoid overflow
			const size_t CHUNK_SIZE = 256;  // Process in chunks to prevent overflow
			
			for(; i + CHUNK_SIZE <= end; i += CHUNK_SIZE) {
				__m256i v_sum0 = _mm256_setzero_si256();
				__m256i v_sum1 = _mm256_setzero_si256();
				__m256i v_sum2 = _mm256_setzero_si256();
				__m256i v_sum3 = _mm256_setzero_si256();
				
				// Process 32 elements per iteration (4 AVX vectors)
				for(size_t j = 0; j < CHUNK_SIZE && (i + j + 31) < end; j += 32) {
					__m256i v_data0 = _mm256_load_si256((__m256i*)&data[i + j]);
					__m256i v_data1 = _mm256_load_si256((__m256i*)&data[i + j + 8]);
					__m256i v_data2 = _mm256_load_si256((__m256i*)&data[i + j + 16]);
					__m256i v_data3 = _mm256_load_si256((__m256i*)&data[i + j + 24]);
					
					// Square the values (using 32-bit multiplication)
					__m256i v_sq0 = _mm256_mullo_epi32(v_data0, v_data0);
					__m256i v_sq1 = _mm256_mullo_epi32(v_data1, v_data1);
					__m256i v_sq2 = _mm256_mullo_epi32(v_data2, v_data2);
					__m256i v_sq3 = _mm256_mullo_epi32(v_data3, v_data3);
					
					// Accumulate
					v_sum0 = _mm256_add_epi32(v_sum0, v_sq0);
					v_sum1 = _mm256_add_epi32(v_sum1, v_sq1);
					v_sum2 = _mm256_add_epi32(v_sum2, v_sq2);
					v_sum3 = _mm256_add_epi32(v_sum3, v_sq3);
				}
				
				// Horizontal sum of all vectors
				int32_t temp0[8], temp1[8], temp2[8], temp3[8];
				_mm256_store_si256((__m256i*)temp0, v_sum0);
				_mm256_store_si256((__m256i*)temp1, v_sum1);
				_mm256_store_si256((__m256i*)temp2, v_sum2);
				_mm256_store_si256((__m256i*)temp3, v_sum3);
				
				for(int k = 0; k < 8; k++) {
					sum_squared += (int64_t)temp0[k] + (int64_t)temp1[k] + (int64_t)temp2[k] + (int64_t)temp3[k];
				}
			}
		}
    #endif
    
    // Scalar processing for remainder
    for(; i < end; i++) {
        int32_t val = data[i];
        sum_squared += (int64_t)val * (int64_t)val;
    }
    
    task->accumulator.int32_acc = (int32_t)sum_squared;  
    return NULL;
}

#endif

/**
 * @ingroup backend_system
 * @brief Backend implementations for L2 normalization operation
 * @details
 * Array follows the common backend registration pattern for L2 normalization
 * operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for L2 normalization
 * 
 * @see nnl2_naive
 * @see naive_l2norm
 */
Implementation l2norm_backends[] = {
	REGISTER_BACKEND(naive_l2norm, nnl2_naive, NAIVE_BACKEND_NAME),
	
	#ifdef NNL2_PTHREAD_AVAILABLE
	    REGISTER_BACKEND(nnl2_own_l2norm, nnl2_own, NNL2_OWN_NAME),
	#endif
};	

/**
 * @brief Function pointer for L2 normalization operation
 * @ingroup backend_system 
 */
l2normfn l2norm;

/** 
 * @brief Makes the L2 normalization backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
MAKE_CURRENT_BACKEND(l2norm);

/** 
 * @brief Sets the backend for L2 normalization operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for L2 normalization
 * @see ESET_BACKEND_BY_NAME
 */
void set_l2norm_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(l2norm_backends, l2norm, backend_name, CURRENT_BACKEND(l2norm));
}

/** 
 * @brief Gets the name of the active backend for L2 normalization operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_l2norm_backend() {
	return CURRENT_BACKEND(l2norm);
}

/** 
 * @brief Function declaration for getting all available L2 normalization backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(l2norm);

/**
 * @brief Function declaration for getting the number of available L2 normalization backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(l2norm);

#endif /** NNL2_L2_H **/
