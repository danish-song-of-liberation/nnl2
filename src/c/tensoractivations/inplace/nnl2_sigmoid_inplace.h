#ifndef NNL2_SIGMOID_INPLACE_H
#define NNL2_SIGMOID_INPLACE_H

#include <stdbool.h>

///@{
	
/** @brief
 * Threshold for enabling parallel execution of the
 * Sigmoid operation during in-place calculations
 */
#define NNL2_SIGMOID_INPLACE_PARALLEL_THREASHOLD 1000000

///@}

/** @brief
 * Applies sigmoid activation function to a tensor in-place
 *
 ** @param tensor
 * Pointer to the input tensor to be modified in-place
 *
 ** @param approx
 * Whether to use approximation for faster computation
 * - true: uses fast approximation sigmoid(x) ≈ 0.5 + 0.5 * x / (1 + |x|)
 * - false: uses exact calculation sigmoid(x) = 1 / (1 + exp(-x))
 *
 ** @details
 * The sigmoid function maps input values to the range (0, 1).
 * For integer tensors (INT32), this function will trigger a fatal error.
 *
 ** @see nnl2_product
 ** @see nnl2_sigmoid_float64_inplace
 ** @see nnl2_sigmoid_float32_inplace
 **/
void naive_sigmoidinplace(Tensor* tensor, bool approx) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Calculate total number of elements
	size_t total_elems = nnl2_product(tensor->shape, tensor->rank);	
	if(total_elems == 0) return; // If 0 elems then return
	
	void* data = tensor->data;
	
	switch(tensor->dtype) {
		case FLOAT64: {
			double* cast_data = (double*)data;
			if (approx) {
				for(size_t i = 0; i < total_elems; i++) {
					double x = cast_data[i];
					double abs_x = fabs(x);
					cast_data[i] = 0.5 + 0.5 * x / (1.0 + abs_x);
				}
			} else {
				for(size_t i = 0; i < total_elems; i++) nnl2_sigmoid_float64_inplace(&cast_data[i]);
			}
			break;
		}
		
		case FLOAT32: {
			float* cast_data = (float*)data;
			if (approx) {
				for(size_t i = 0; i < total_elems; i++) {
					float x = cast_data[i];
					if (x < -8.0f) cast_data[i] = 0.0f;
					else if (x > 8.0f) cast_data[i] = 1.0f;
					else {
						float abs_x = fabsf(x);
						cast_data[i] = 0.5f + 0.5f * x / (1.0f + abs_x);
					}
				}
			} else {
				for(size_t i = 0; i < total_elems; i++) nnl2_sigmoid_float32_inplace(&cast_data[i]);
			}
			break;
		}
		
		case INT32: {
			NNL2_FATAL("Sigmoid in-place cannot be applied to the provided tensor");
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
 * Optimized vectorized implementation of approximate sigmoid for float64 data
 *
 ** @param data
 * Pointer to the double-precision floating point data array
 *
 ** @param size
 * Number of elements in the data array
 *
 ** @details
 * Uses AVX256 SIMD instructions to process 4 elements simultaneously.
 * Implements the approximation: sigmoid(x) ≈ 0.5 + 0.5 * x / (1 + |x|)
 * Automatically handles both aligned and unaligned memory access.
 *
 ** @see nnl2_own_psigmoid_inplace_float64_superapprox
 **/
static inline void nnl2_sigmoid_vector_float64_superapprox(double* data, size_t size) {  
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d half = _mm256_set1_pd(0.5);
    
    // Check alignment
    bool data_aligned = NNL2_IS_ALIGNED(data, NNL2_TENSOR_ALIGNMENT_32);
    
    size_t i = 0;
    
    // Process aligned blocks
    if (data_aligned) {
        for (; i + 4 <= size; i += 4) {
            __m256d x = _mm256_load_pd(&data[i]);
            
            __m256d abs_x = _mm256_andnot_pd(_mm256_set1_pd(-0.0), x);
            __m256d denom = _mm256_add_pd(one, abs_x);
            __m256d ratio = _mm256_div_pd(x, denom);
            __m256d result = _mm256_add_pd(half, _mm256_mul_pd(half, ratio));
            
            _mm256_store_pd(&data[i], result);
        }
    } else {
        // Process unaligned
        for (; i + 4 <= size; i += 4) {
            __m256d x = _mm256_loadu_pd(&data[i]);
            
            __m256d abs_x = _mm256_andnot_pd(_mm256_set1_pd(-0.0), x);
            __m256d denom = _mm256_add_pd(one, abs_x);
            __m256d ratio = _mm256_div_pd(x, denom);
            __m256d result = _mm256_add_pd(half, _mm256_mul_pd(half, ratio));
            
            _mm256_storeu_pd(&data[i], result);
        }
    }
    
    // Process remaining elements
    for (; i < size; i++) {
        double x = data[i];
        double abs_x = fabs(x);
        data[i] = 0.5 + 0.5 * x / (1.0 + abs_x);
    }
}

/** @brief
 * Optimized vectorized implementation of approximate sigmoid for float32 data
 *
 ** @param data
 * Pointer to the single-precision floating point data array
 *
 ** @param size
 * Number of elements in the data array
 *
 ** @details
 * Uses AVX256 SIMD instructions to process 8 elements simultaneously.
 * Implements clamped approximation with thresholds at ±8.0 for numerical stability.
 * For x < -8.0: returns 0.0, for x > 8.0: returns 1.0
 * Otherwise uses: 0.5 + 0.5 * x / (1 + |x|)
 *
 ** @see nnl2_own_psigmoid_inplace_float32_superapprox
 **/
static inline void nnl2_sigmoid_vector_float32_superapprox(float* data, size_t size) {
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 threshold = _mm256_set1_ps(8.0f);
    
    // Check alignment
    bool data_aligned = NNL2_IS_ALIGNED(data, NNL2_TENSOR_ALIGNMENT_32);
    
    size_t i = 0;
    
    // Process aligned blocks
    if (data_aligned) {
        for (; i + 8 <= size; i += 8) {
            __m256 x = _mm256_load_ps(&data[i]);
            
            __m256 x_clamped = _mm256_min_ps(threshold, _mm256_max_ps(_mm256_sub_ps(_mm256_setzero_ps(), threshold), x));
            
            __m256 abs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), x_clamped);
            __m256 denom = _mm256_add_ps(one, abs_x);
            __m256 ratio = _mm256_div_ps(x_clamped, denom);
            __m256 result = _mm256_add_ps(half, _mm256_mul_ps(half, ratio));
            
            _mm256_store_ps(&data[i], result);
        }
    } else {
        // Process unaligned
        for (; i + 8 <= size; i += 8) {
            __m256 x = _mm256_loadu_ps(&data[i]);
            
            __m256 x_clamped = _mm256_min_ps(threshold, _mm256_max_ps(_mm256_sub_ps(_mm256_setzero_ps(), threshold), x));
            
            __m256 abs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), x_clamped);
            __m256 denom = _mm256_add_ps(one, abs_x);
            __m256 ratio = _mm256_div_ps(x_clamped, denom);
            __m256 result = _mm256_add_ps(half, _mm256_mul_ps(half, ratio));
            
            _mm256_storeu_ps(&data[i], result);
        }
    }
    
    // Process remaining elements
    for (; i < size; i++) {
        float x = data[i];
        if (x < -8.0f) data[i] = 0.0f;
        else if (x > 8.0f) data[i] = 1.0f;
        else {
            float abs_x = fabsf(x);
            data[i] = 0.5f + 0.5f * x / (1.0f + abs_x);
        }
    }
}

/** @brief
 * Optimized vectorized implementation of exact sigmoid for float64 data
 *
 ** @param data
 * Pointer to the double-precision floating point data array
 *
 ** @param size
 * Number of elements in the data array
 *
 ** @details
 * Uses AVX256 for loading/storing but computes exact sigmoid using exp().
 * Implements: sigmoid(x) = 1 / (1 + exp(-x))
 * Handles both aligned and unaligned memory access patterns.
 *
 ** @see nnl2_own_psigmoid_inplace_float64_full
 **/
static inline void nnl2_sigmoid_vector_float64_full(double* data, size_t size) {
    // Check alignment
    bool data_aligned = NNL2_IS_ALIGNED(data, NNL2_TENSOR_ALIGNMENT_32);
    
    size_t i = 0;
    
    // Process aligned blocks
    if (data_aligned) {
        for (; i + 4 <= size; i += 4) {
            __m256d x = _mm256_load_pd(&data[i]);
            __m256d neg_x = _mm256_sub_pd(_mm256_setzero_pd(), x);
            
            double temp[4] __attribute__((aligned(32)));
            _mm256_store_pd(temp, neg_x);
            
            double result[4] __attribute__((aligned(32)));
            for (int j = 0; j < 4; j++) {
                result[j] = 1.0 / (1.0 + exp(temp[j]));
            }
            
            _mm256_store_pd(&data[i], _mm256_load_pd(result));
        }
    } else {
        // Process unaligned
        for (; i + 4 <= size; i += 4) {
            __m256d x = _mm256_loadu_pd(&data[i]);
            __m256d neg_x = _mm256_sub_pd(_mm256_setzero_pd(), x);
            
            double temp[4];
            _mm256_storeu_pd(temp, neg_x);
            
            double result[4];
            for (int j = 0; j < 4; j++) {
                result[j] = 1.0 / (1.0 + exp(temp[j]));
            }
            
            _mm256_storeu_pd(&data[i], _mm256_loadu_pd(result));
        }
    }
    
    // Process remaining elements
    for (; i < size; i++) {
        data[i] = 1.0 / (1.0 + exp(-data[i]));
    }
}

/** @brief
 * Optimized vectorized implementation of exact sigmoid for float32 data
 *
 ** @param data
 * Pointer to the single-precision floating point data array
 *
 ** @param size
 * Number of elements in the data array
 *
 ** @details
 * Uses AVX256 SIMD instructions to process 8 elements simultaneously.
 * Implements exact sigmoid calculation: 1 / (1 + expf(-x))
 * Falls back to scalar processing for remaining elements.
 *
 ** @see nnl2_own_psigmoid_inplace_float32_full
 **/
static inline void nnl2_sigmoid_vector_float32_full(float* data, size_t size) {
    // Check alignment
    bool data_aligned = NNL2_IS_ALIGNED(data, NNL2_TENSOR_ALIGNMENT_32);
    
    size_t i = 0;
    
    // Process aligned blocks with SIMD where possible
    if (data_aligned) {
        // Use SIMD for aligned data
        for (; i + 8 <= size; i += 8) {
            __m256 x = _mm256_load_ps(&data[i]);
            __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
            
            float temp[8] __attribute__((aligned(32)));
            _mm256_store_ps(temp, neg_x);
            
            float result[8] __attribute__((aligned(32)));
            for (int j = 0; j < 8; j++) {
                result[j] = 1.0f / (1.0f + expf(temp[j]));
            }
            
            _mm256_store_ps(&data[i], _mm256_load_ps(result));
        }
    } else {
        // Use SIMD for unaligned data
        for (; i + 8 <= size; i += 8) {
            __m256 x = _mm256_loadu_ps(&data[i]);
            __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
            
            float temp[8];
            _mm256_storeu_ps(temp, neg_x);
            
            float result[8];
            for (int j = 0; j < 8; j++) {
                result[j] = 1.0f / (1.0f + expf(temp[j]));
            }
            
            _mm256_storeu_ps(&data[i], _mm256_loadu_ps(result));
        }
    }
    
    // Process remaining elements
    for (; i < size; i++) {
        data[i] = 1.0f / (1.0f + expf(-data[i]));
    }
}

/** @brief
 * Worker function for parallel approximate sigmoid on float64 data
 *
 ** @param arg
 * Pointer to single_arr_ptask structure containing work parameters
 *
 ** @return
 * Always returns NULL (required by pthread interface)
 *
 ** @details
 * Processes a chunk of double-precision data using the super-approximation method.
 * Designed to be executed in parallel by multiple threads.
 *
 ** @see single_arr_ptask
 ** @see nnl2_sigmoid_vector_float64_superapprox
 **/
void* nnl2_own_psigmoid_inplace_float64_superapprox(void* arg) {
    single_arr_ptask* task = (single_arr_ptask*)arg;
    double* input = (double*)task->data;
    size_t start = task->start;
    size_t end = task->end;
    size_t size = end - start;
    
    nnl2_sigmoid_vector_float64_superapprox(input + start, size);
    return NULL;
}

/** @brief
 * Worker function for parallel exact sigmoid on float64 data
 *
 ** @param arg
 * Pointer to single_arr_ptask structure containing work parameters
 *
 ** @return
 * Always returns NULL (required by pthread interface)
 *
 ** @details
 * Processes a chunk of double-precision data using the exact sigmoid calculation.
 * Designed for parallel execution with proper chunk boundaries.
 *
 ** @see single_arr_ptask
 ** @see nnl2_sigmoid_vector_float64_full
 **/
void* nnl2_own_psigmoid_inplace_float64_full(void* arg) {
    single_arr_ptask* task = (single_arr_ptask*)arg;
    double* input = (double*)task->data;
    size_t start = task->start;
    size_t end = task->end;
    size_t size = end - start;
    
    nnl2_sigmoid_vector_float64_full(input + start, size);
    return NULL;
}

/** @brief
 * Worker function for parallel approximate sigmoid on float32 data
 *
 ** @param arg
 * Pointer to single_arr_ptask structure containing work parameters
 *
 ** @return
 * Always returns NULL (required by pthread interface)
 *
 ** @details
 * Processes a chunk of single-precision data using the super-approximation method.
 * Uses AVX256 vectorization for optimal performance on large data chunks.
 *
 ** @see single_arr_ptask
 ** @see nnl2_sigmoid_vector_float32_superapprox
 **/
void* nnl2_own_psigmoid_inplace_float32_superapprox(void* arg) {
    single_arr_ptask* task = (single_arr_ptask*)arg;
    float* input = (float*)task->data;
    size_t start = task->start;
    size_t end = task->end;
    size_t size = end - start;
    
    nnl2_sigmoid_vector_float32_superapprox(input + start, size);
    return NULL;
}

/** @brief
 * Worker function for parallel exact sigmoid on float32 data
 *
 ** @param arg
 * Pointer to single_arr_ptask structure containing work parameters
 *
 ** @return
 * Always returns NULL (required by pthread interface)
 *
 ** @details
 * Processes a chunk of single-precision data using the exact sigmoid calculation.
 * Optimized for parallel processing with proper memory alignment handling.
 *
 ** @see single_arr_ptask
 ** @see nnl2_sigmoid_vector_float32_full
 **/
void* nnl2_own_psigmoid_inplace_float32_full(void* arg) {
    single_arr_ptask* task = (single_arr_ptask*)arg;
    float* input = (float*)task->data;
    size_t start = task->start;
    size_t end = task->end;
    size_t size = end - start;
    
    nnl2_sigmoid_vector_float32_full(input + start, size);
    return NULL;
}

/** @brief
 * Parallel in-place sigmoid implementation for float64 tensors
 *
 ** @param data
 * Pointer to double-precision data array
 *
 ** @param total_size
 * Total number of elements in the data array
 *
 ** @param num_threads
 * Number of threads to use for parallel processing
 *
 ** @param approx
 * Whether to use approximation (true) or exact calculation (false)
 *
 ** @return
 * Always returns NULL
 *
 ** @details
 * Automatically switches between single-threaded and multi-threaded execution
 * based on data size. Uses optimized chunk sizing for load balancing.
 *
 ** @see nnl2_own_psigmoid_inplace_float64_superapprox
 ** @see nnl2_own_psigmoid_inplace_float64_full
 **/
void* nnl2_own_sigmoid_inplace_float64(double* data, size_t total_size, size_t num_threads, bool approx) {
    if (total_size < 10000) {
        if (approx) {
            nnl2_sigmoid_vector_float64_superapprox(data, total_size);
        } else {
            nnl2_sigmoid_vector_float64_full(data, total_size);
        }
        return NULL;
    }
    
    pthread_t threads[num_threads];
    single_arr_ptask tasks[num_threads];
    
    size_t chunk = total_size / num_threads;
    chunk = (chunk + 3) & ~3; 
    
    size_t current_start = 0;
    size_t actual_threads = 0;
    
    void* (*worker_func)(void*) = NULL;
    if (approx) {
        worker_func = nnl2_own_psigmoid_inplace_float64_superapprox;
    } else {
        worker_func = nnl2_own_psigmoid_inplace_float64_full;
    }
    
    for (size_t i = 0; i < num_threads && current_start < total_size; i++) {
        size_t current_end = current_start + chunk;
        if (current_end > total_size) current_end = total_size;
        if (current_start >= current_end) break;
        
        tasks[i].data = data;
        tasks[i].start = current_start;
        tasks[i].end = current_end;
        
        int status = pthread_create(&threads[i], NULL, worker_func, &tasks[i]);
        if(status != 0) {
            NNL2_THREAD_CREATE_ERROR(status, "nnl2_own_sigmoid_inplace_float64");
            break;
        }
        
        actual_threads++;
        current_start = current_end;
    }
    
    for (size_t i = 0; i < actual_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    return NULL;
}

/** @brief
 * Parallel in-place sigmoid implementation for float32 tensors
 *
 ** @param data
 * Pointer to single-precision data array
 *
 ** @param total_size
 * Total number of elements in the data array
 *
 ** @param num_threads
 * Number of threads to use for parallel processing
 *
 ** @param approx
 * Whether to use approximation (true) or exact calculation (false)
 *
 ** @return
 * Always returns NULL
 *
 ** @details
 * Optimized for float32 data with proper SIMD alignment and chunk sizing.
 * Uses 8-element vectorization for optimal AVX256 performance.
 *
 ** @see nnl2_own_psigmoid_inplace_float32_superapprox
 ** @see nnl2_own_psigmoid_inplace_float32_full
 **/
void* nnl2_own_sigmoid_inplace_float32(float* data, size_t total_size, size_t num_threads, bool approx) {
    if (total_size < 20000) {
        if (approx) {
            nnl2_sigmoid_vector_float32_superapprox(data, total_size);
        } else {
            nnl2_sigmoid_vector_float32_full(data, total_size);
        }
        return NULL;
    }
    
    pthread_t threads[num_threads];
    single_arr_ptask tasks[num_threads];
    
    size_t chunk = total_size / num_threads;
    chunk = (chunk + 7) & ~7; 
    
    size_t current_start = 0;
    size_t actual_threads = 0;
    
    void* (*worker_func)(void*) = NULL;
    if (approx) {
        worker_func = nnl2_own_psigmoid_inplace_float32_superapprox;
    } else {
        worker_func = nnl2_own_psigmoid_inplace_float32_full;
    }
    
    for (size_t i = 0; i < num_threads && current_start < total_size; i++) {
        size_t current_end = current_start + chunk;
        if (current_end > total_size) current_end = total_size;
        if (current_start >= current_end) break;
        
        tasks[i].data = data;
        tasks[i].start = current_start;
        tasks[i].end = current_end;
        
        int status = pthread_create(&threads[i], NULL, worker_func, &tasks[i]);
        if(status != 0) {
            NNL2_THREAD_CREATE_ERROR(status, "nnl2_own_sigmoid_inplace_float32");
            break;
        }
        
        actual_threads++;
        current_start = current_end;
    }
    
    for (size_t i = 0; i < actual_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    return NULL;
}

/** @brief
 * High-performance parallel in-place sigmoid activation function
 *
 ** @param tensor
 * Pointer to the tensor to be modified in-place
 *
 ** @param approx
 * Whether to use approximation for faster computation
 *
 ** @details
 * Uses multi-threading and SIMD vectorization for optimal performance on large tensors.
 * Automatically selects between naive and optimized implementations based on tensor size.
 * Supports both float32 and float64 data types with proper error handling.
 *
 ** @see naive_sigmoidinplace
 ** @see nnl2_own_sigmoid_inplace_float64
 ** @see nnl2_own_sigmoid_inplace_float32
 **/
void nnl2_own_sigmoid_inplace(Tensor* tensor, bool approx) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    size_t total_elems = nnl2_product(tensor->shape, tensor->rank);    
    
    if(total_elems == 0) {
        return;
    }

    if(total_elems < NNL2_SIGMOID_INPLACE_PARALLEL_THREASHOLD) {
        naive_sigmoidinplace(tensor, approx);
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return;
    }
    
    void* data = tensor->data;
    
    // Check tensor data alignment for potential optimization
    bool tensor_aligned = NNL2_IS_ALIGNED(data, NNL2_TENSOR_ALIGNMENT_32);
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_DEBUG
        if (!tensor_aligned) {
            NNL2_DEBUG("Tensor data is not aligned, using unaligned SIMD operations");
        }
    #endif
    
    switch(tensor->dtype) {
        case FLOAT64: nnl2_own_sigmoid_inplace_float64((double*)data, total_elems, NNL2_NUM_THREADS, approx);  break;
        case FLOAT32: nnl2_own_sigmoid_inplace_float32((float*)data, total_elems, NNL2_NUM_THREADS, approx);   break;
		
        case INT32: {
            NNL2_ERROR("Sigmoid in-place cannot be applied to the provided tensor");
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

#endif

/**
 * @ingroup backend_system
 * @brief Backend implementations for sigmoidinplace operation
 * @details
 * Array follows the common backend registration pattern for sigmoidinplace operations.
 * Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for in-place sigmoid activation function
 *  - nnl2_own: Own nnl2 implementation for in-place sigmoid activation
 * 
 * @see nnl2_naive
 * @see naive_sigmoidinplace
 */
Implementation sigmoidinplace_backends[] = {
	REGISTER_BACKEND(naive_sigmoidinplace, nnl2_naive, NAIVE_BACKEND_NAME),
	
	#ifdef NNL2_PTHREAD_AVAILABLE
		REGISTER_BACKEND(nnl2_own_sigmoid_inplace, nnl2_own, NNL2_OWN_NAME),
	#endif
};	

/**
 * @brief Function pointer for sigmoidinplace operation
 * @ingroup backend_system 
 */
sigmoidinplacefn sigmoidinplace;

/** 
 * @brief Makes the sigmoidinplace backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(sigmoidinplace);

/** 
 * @brief Sets the backend for sigmoidinplace operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for sigmoidinplace
 * @see ESET_BACKEND_BY_NAME
 */
void set_sigmoidinplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(sigmoidinplace_backends, sigmoidinplace, backend_name, current_backend(sigmoidinplace));
}

/** 
 * @brief Gets the name of the active backend for sigmoidinplace operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_sigmoidinplace_backend() {
	return current_backend(sigmoidinplace);
}

/** 
 * @brief Function declaration for getting all available sigmoidinplace backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(sigmoidinplace);

/**
 * @brief Function declaration for getting the number of available sigmoidinplace backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(sigmoidinplace);

#endif /** NNL2_SIGMOID_INPLACE_H **/
