#ifndef NNL2_SIGMOID_H
#define NNL2_SIGMOID_H

#include <stdbool.h>

///@{
	
/** @brief
 * Threshold for enabling parallel execution of Sigmoid operation
 */
#define NNL2_SIGMOID_PARALLEL_THRESHOLD 1000000

///@}

/** @brief
 * Applies sigmoid activation function to a tensor, returning a new tensor
 *
 ** @param tensor
 * Pointer to the input tensor
 *
 ** @param approx
 * Whether to use approximation for faster computation
 * true: uses fast approximation sigmoid(x) ~= 0.5 + 0.5 * x / (1 + |x|)
 * false: uses exact calculation sigmoid(x) = 1 / (1 + exp(-x))
 *
 ** @return
 * Pointer to a new tensor containing the sigmoid-activated values
 * Returns NULL in case of failure
 *
 ** @details
 * The sigmoid function maps input values to the range (0, 1).
 * For integer tensors (INT32), the result tensor will have FLOAT64 dtype.
 * Creates a new tensor with the same shape as input, but may change dtype.
 *
 ** @see nnl2_product
 ** @see nnl2_sigmoid_float64
 ** @see nnl2_sigmoid_float32
 ** @see nnl2_empty
 **/
Tensor* naive_sigmoid(Tensor* tensor, bool approx) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        if(tensor == NULL) {
            NNL2_ERROR("Passed tensor is NULL");
            return NULL;
        }
    #endif

    size_t total_elems = nnl2_product(tensor->shape, tensor->rank);    
    Tensor* result = NULL;
    
    // For INT32 input, create FLOAT64 output tensor
    if (tensor->dtype == INT32) {
        result = nnl2_empty(tensor->shape, tensor->rank, FLOAT64);
    } else {
        result = nnl2_empty(tensor->shape, tensor->rank, tensor->dtype);
    }
    
    if(total_elems == 0) return result;
    
    void* data_t = tensor->data;
    void* data_r = result->data;
    
    switch(tensor->dtype) {
        case FLOAT64: {
            double* cast_data_t = (double*)data_t;    
            double* cast_data_r = (double*)data_r;
            if (approx) {
                for(size_t i = 0; i < total_elems; i++) {
                    double x = cast_data_t[i];
                    double abs_x = fabs(x);
                    cast_data_r[i] = 0.5 + 0.5 * x / (1.0 + abs_x);
                }
            } else {
                for(size_t i = 0; i < total_elems; i++) {
                    cast_data_r[i] = nnl2_sigmoid_float64(cast_data_t[i]);
                }
            }
            break;
        }
        
        case FLOAT32: {
            float* cast_data_t = (float*)data_t;    
            float* cast_data_r = (float*)data_r;
            if (approx) {
                for(size_t i = 0; i < total_elems; i++) {
                    float x = cast_data_t[i];
                    if (x < -8.0f) cast_data_r[i] = 0.0f;
                    else if (x > 8.0f) cast_data_r[i] = 1.0f;
                    else {
                        float abs_x = fabsf(x);
                        cast_data_r[i] = 0.5f + 0.5f * x / (1.0f + abs_x);
                    }
                }
            } else {
                for(size_t i = 0; i < total_elems; i++) {
                    cast_data_r[i] = nnl2_sigmoid_float32(cast_data_t[i]);
                }
            }
            break;
        }
        
        case INT32: {
            // Convert INT32 to FLOAT64 and apply sigmoid
            int32_t* cast_data_t = (int32_t*)data_t;    
            double* cast_data_r = (double*)data_r;
            if (approx) {
                for(size_t i = 0; i < total_elems; i++) {
                    double x = (double)cast_data_t[i];
                    double abs_x = fabs(x);
                    cast_data_r[i] = 0.5 + 0.5 * x / (1.0 + abs_x);
                }
            } else {
                for(size_t i = 0; i < total_elems; i++) {
                    double x = (double)cast_data_t[i];
                    cast_data_r[i] = nnl2_sigmoid_float64(x);
                }
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

#ifdef NNL2_PTHREAD_AVAILABLE

/** @brief
 * Optimized vectorized implementation of approximate sigmoid for float64 data (out-of-place)
 *
 ** @param src_data
 * Pointer to the source double-precision floating point data array
 *
 ** @param dst_data
 * Pointer to the destination double-precision floating point data array
 *
 ** @param size
 * Number of elements in the data arrays
 *
 ** @details
 * Uses AVX256 SIMD instructions to process 4 elements simultaneously.
 * Implements the approximation: sigmoid(x) ≈ 0.5 + 0.5 * x / (1 + |x|)
 * Automatically handles both aligned and unaligned memory access.
 *
 ** @see nnl2_own_psigmoid_float64_superapprox
 **/
static inline void nnl2_sigmoid_vector_float64_superapprox_out(double* src_data, double* dst_data, size_t size) {  
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d half = _mm256_set1_pd(0.5);
    
    // Check alignment
    bool src_aligned = NNL2_IS_ALIGNED(src_data, NNL2_TENSOR_ALIGNMENT_32);
    bool dst_aligned = NNL2_IS_ALIGNED(dst_data, NNL2_TENSOR_ALIGNMENT_32);
    
    size_t i = 0;
    
    // Process aligned blocks
    if (src_aligned && dst_aligned) {
        for (; i + 4 <= size; i += 4) {
            __m256d x = _mm256_load_pd(&src_data[i]);
            
            __m256d abs_x = _mm256_andnot_pd(_mm256_set1_pd(-0.0), x);
            __m256d denom = _mm256_add_pd(one, abs_x);
            __m256d ratio = _mm256_div_pd(x, denom);
            __m256d result = _mm256_add_pd(half, _mm256_mul_pd(half, ratio));
            
            _mm256_store_pd(&dst_data[i], result);
        }
    } else {
        // Process unaligned
        for (; i + 4 <= size; i += 4) {
            __m256d x = _mm256_loadu_pd(&src_data[i]);
            
            __m256d abs_x = _mm256_andnot_pd(_mm256_set1_pd(-0.0), x);
            __m256d denom = _mm256_add_pd(one, abs_x);
            __m256d ratio = _mm256_div_pd(x, denom);
            __m256d result = _mm256_add_pd(half, _mm256_mul_pd(half, ratio));
            
            _mm256_storeu_pd(&dst_data[i], result);
        }
    }
    
    // Process remaining elements
    for (; i < size; i++) {
        double x = src_data[i];
        double abs_x = fabs(x);
        dst_data[i] = 0.5 + 0.5 * x / (1.0 + abs_x);
    }
}

/** @brief
 * Optimized vectorized implementation of approximate sigmoid for float32 data (out-of-place)
 *
 ** @param src_data
 * Pointer to the source single-precision floating point data array
 *
 ** @param dst_data
 * Pointer to the destination single-precision floating point data array
 *
 ** @param size
 * Number of elements in the data arrays
 *
 ** @details
 * Uses AVX256 SIMD instructions to process 8 elements simultaneously.
 * Implements clamped approximation with thresholds at ±8.0 for numerical stability.
 * For x < -8.0: returns 0.0, for x > 8.0: returns 1.0
 * Otherwise uses: 0.5 + 0.5 * x / (1 + |x|)
 *
 ** @see nnl2_own_psigmoid_float32_superapprox
 **/
static inline void nnl2_sigmoid_vector_float32_superapprox_out(float* src_data, float* dst_data, size_t size) {
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 threshold = _mm256_set1_ps(8.0f);
    
    // Check alignment
    bool src_aligned = NNL2_IS_ALIGNED(src_data, NNL2_TENSOR_ALIGNMENT_32);
    bool dst_aligned = NNL2_IS_ALIGNED(dst_data, NNL2_TENSOR_ALIGNMENT_32);
    
    size_t i = 0;
    
    // Process aligned blocks
    if (src_aligned && dst_aligned) {
        for (; i + 8 <= size; i += 8) {
            __m256 x = _mm256_load_ps(&src_data[i]);
            
            __m256 x_clamped = _mm256_min_ps(threshold, _mm256_max_ps(_mm256_sub_ps(_mm256_setzero_ps(), threshold), x));
            
            __m256 abs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), x_clamped);
            __m256 denom = _mm256_add_ps(one, abs_x);
            __m256 ratio = _mm256_div_ps(x_clamped, denom);
            __m256 result = _mm256_add_ps(half, _mm256_mul_ps(half, ratio));
            
            _mm256_store_ps(&dst_data[i], result);
        }
    } else {
        // Process unaligned
        for (; i + 8 <= size; i += 8) {
            __m256 x = _mm256_loadu_ps(&src_data[i]);
            
            __m256 x_clamped = _mm256_min_ps(threshold, _mm256_max_ps(_mm256_sub_ps(_mm256_setzero_ps(), threshold), x));
            
            __m256 abs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), x_clamped);
            __m256 denom = _mm256_add_ps(one, abs_x);
            __m256 ratio = _mm256_div_ps(x_clamped, denom);
            __m256 result = _mm256_add_ps(half, _mm256_mul_ps(half, ratio));
            
            _mm256_storeu_ps(&dst_data[i], result);
        }
    }
    
    // Process remaining elements
    for (; i < size; i++) {
        float x = src_data[i];
        if (x < -8.0f) dst_data[i] = 0.0f;
        else if (x > 8.0f) dst_data[i] = 1.0f;
        else {
            float abs_x = fabsf(x);
            dst_data[i] = 0.5f + 0.5f * x / (1.0f + abs_x);
        }
    }
}

/** @brief
 * Optimized vectorized implementation of exact sigmoid for float64 data (out-of-place)
 *
 ** @param src_data
 * Pointer to the source double-precision floating point data array
 *
 ** @param dst_data
 * Pointer to the destination double-precision floating point data array
 *
 ** @param size
 * Number of elements in the data arrays
 *
 ** @details
 * Uses AVX256 for loading/storing but computes exact sigmoid using exp().
 * Implements: sigmoid(x) = 1 / (1 + exp(-x))
 * Handles both aligned and unaligned memory access patterns.
 *
 ** @see nnl2_own_psigmoid_float64_full
 **/
static inline void nnl2_sigmoid_vector_float64_full_out(double* src_data, double* dst_data, size_t size) {
    // Check alignment
    bool src_aligned = NNL2_IS_ALIGNED(src_data, NNL2_TENSOR_ALIGNMENT_32);
    bool dst_aligned = NNL2_IS_ALIGNED(dst_data, NNL2_TENSOR_ALIGNMENT_32);
    
    size_t i = 0;
    
    // Process aligned blocks
    if (src_aligned && dst_aligned) {
        for (; i + 4 <= size; i += 4) {
            __m256d x = _mm256_load_pd(&src_data[i]);
            __m256d neg_x = _mm256_sub_pd(_mm256_setzero_pd(), x);
            
            double temp[4] __attribute__((aligned(32)));
            _mm256_store_pd(temp, neg_x);
            
            double result[4] __attribute__((aligned(32)));
            for (int j = 0; j < 4; j++) {
                result[j] = 1.0 / (1.0 + exp(temp[j]));
            }
            
            _mm256_store_pd(&dst_data[i], _mm256_load_pd(result));
        }
    } else {
        // Process unaligned
        for (; i + 4 <= size; i += 4) {
            __m256d x = _mm256_loadu_pd(&src_data[i]);
            __m256d neg_x = _mm256_sub_pd(_mm256_setzero_pd(), x);
            
            double temp[4];
            _mm256_storeu_pd(temp, neg_x);
            
            double result[4];
            for (int j = 0; j < 4; j++) {
                result[j] = 1.0 / (1.0 + exp(temp[j]));
            }
            
            _mm256_storeu_pd(&dst_data[i], _mm256_loadu_pd(result));
        }
    }
    
    // Process remaining elements
    for (; i < size; i++) {
        dst_data[i] = 1.0 / (1.0 + exp(-src_data[i]));
    }
}

/** @brief
 * Optimized vectorized implementation of exact sigmoid for float32 data (out-of-place)
 *
 ** @param src_data
 * Pointer to the source single-precision floating point data array
 *
 ** @param dst_data
 * Pointer to the destination single-precision floating point data array
 *
 ** @param size
 * Number of elements in the data arrays
 *
 ** @details
 * Uses AVX256 SIMD instructions to process 8 elements simultaneously.
 * Implements exact sigmoid calculation: 1 / (1 + expf(-x))
 * Falls back to scalar processing for remaining elements.
 *
 ** @see nnl2_own_psigmoid_float32_full
 **/
static inline void nnl2_sigmoid_vector_float32_full_out(float* src_data, float* dst_data, size_t size) {
    // Check alignment
    bool src_aligned = NNL2_IS_ALIGNED(src_data, NNL2_TENSOR_ALIGNMENT_32);
    bool dst_aligned = NNL2_IS_ALIGNED(dst_data, NNL2_TENSOR_ALIGNMENT_32);
    
    size_t i = 0;
    
    // Process aligned blocks with SIMD where possible
    if (src_aligned && dst_aligned) {
        // Use SIMD for aligned data
        for (; i + 8 <= size; i += 8) {
            __m256 x = _mm256_load_ps(&src_data[i]);
            __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
            
            float temp[8] __attribute__((aligned(32)));
            _mm256_store_ps(temp, neg_x);
            
            float result[8] __attribute__((aligned(32)));
            for (int j = 0; j < 8; j++) {
                result[j] = 1.0f / (1.0f + expf(temp[j]));
            }
            
            _mm256_store_ps(&dst_data[i], _mm256_load_ps(result));
        }
    } else {
        // Use SIMD for unaligned data
        for (; i + 8 <= size; i += 8) {
            __m256 x = _mm256_loadu_ps(&src_data[i]);
            __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
            
            float temp[8];
            _mm256_storeu_ps(temp, neg_x);
            
            float result[8];
            for (int j = 0; j < 8; j++) {
                result[j] = 1.0f / (1.0f + expf(temp[j]));
            }
            
            _mm256_storeu_ps(&dst_data[i], _mm256_loadu_ps(result));
        }
    }
    
    // Process remaining elements
    for (; i < size; i++) {
        dst_data[i] = 1.0f / (1.0f + expf(-src_data[i]));
    }
}

/** @brief
 * Optimized vectorized implementation of sigmoid for INT32 to FLOAT64 conversion (out-of-place)
 *
 ** @param src_data
 * Pointer to the source int32 data array
 *
 ** @param dst_data
 * Pointer to the destination double-precision floating point data array
 *
 ** @param size
 * Number of elements in the data arrays
 *
 ** @param approx
 * Whether to use approximation for faster computation
 *
 ** @details
 * Converts INT32 values to FLOAT64 and applies sigmoid function.
 * Uses AVX256 for efficient conversion and computation.
 */
static inline void nnl2_sigmoid_vector_int32_to_float64_out(int32_t* src_data, double* dst_data, size_t size, bool approx) {
    size_t i = 0;
    
    if (approx) {
        // Process with approximation using SIMD where possible
        for (; i + 4 <= size; i += 4) {
            // Load 4 int32 values
            __m128i x_int = _mm_loadu_si128((__m128i*)(src_data + i));
            
            // Convert to double (4 values)
            __m256d x = _mm256_cvtepi32_pd(x_int);
            
            // Compute absolute value
            __m256d abs_x = _mm256_andnot_pd(_mm256_set1_pd(-0.0), x);
            
            // Compute sigmoid approximation: 0.5 + 0.5 * x / (1 + |x|)
            __m256d denom = _mm256_add_pd(_mm256_set1_pd(1.0), abs_x);
            __m256d ratio = _mm256_div_pd(x, denom);
            __m256d result = _mm256_add_pd(_mm256_set1_pd(0.5), _mm256_mul_pd(_mm256_set1_pd(0.5), ratio));
            
            _mm256_storeu_pd(dst_data + i, result);
        }
    } else {
        // Process with exact sigmoid using SIMD where possible
        for (; i + 4 <= size; i += 4) {
            // Load 4 int32 values
            __m128i x_int = _mm_loadu_si128((__m128i*)(src_data + i));
            
            // Convert to double (4 values)
            __m256d x = _mm256_cvtepi32_pd(x_int);
            
            // Compute -x
            __m256d neg_x = _mm256_sub_pd(_mm256_setzero_pd(), x);
            
            // Store to temporary array and compute exp
            double temp[4];
            _mm256_storeu_pd(temp, neg_x);
            
            double result[4];
            for (int j = 0; j < 4; j++) {
                result[j] = 1.0 / (1.0 + exp(temp[j]));
            }
            
            _mm256_storeu_pd(dst_data + i, _mm256_loadu_pd(result));
        }
    }
    
    // Process remaining elements
    for (; i < size; i++) {
        double x = (double)src_data[i];
        if (approx) {
            double abs_x = fabs(x);
            dst_data[i] = 0.5 + 0.5 * x / (1.0 + abs_x);
        } else {
            dst_data[i] = 1.0 / (1.0 + exp(-x));
        }
    }
}

/** @brief
 * Worker function for parallel approximate sigmoid on float64 data (out-of-place)
 *
 ** @param arg
 * Pointer to sigmoid_ptask structure containing work parameters
 *
 ** @return
 * Always returns NULL (required by pthread interface)
 *
 ** @details
 * Processes a chunk of double-precision data using the super-approximation method.
 * Designed to be executed in parallel by multiple threads.
 *
 ** @see sigmoid_ptask
 ** @see nnl2_sigmoid_vector_float64_superapprox_out
 **/
void* nnl2_own_psigmoid_float64_superapprox(void* arg) {
    sigmoid_ptask* task = (sigmoid_ptask*)arg;
    double* src_data = (double*)task->src_data;
    double* dst_data = (double*)task->dst_data;
    size_t start = task->start_idx;
    size_t end = task->end_idx;
    size_t size = end - start;
    
    nnl2_sigmoid_vector_float64_superapprox_out(src_data + start, dst_data + start, size);
    return NULL;
}

/** @brief
 * Worker function for parallel exact sigmoid on float64 data (out-of-place)
 *
 ** @param arg
 * Pointer to sigmoid_ptask structure containing work parameters
 *
 ** @return
 * Always returns NULL (required by pthread interface)
 *
 ** @details
 * Processes a chunk of double-precision data using the exact sigmoid calculation.
 * Designed for parallel execution with proper chunk boundaries.
 *
 ** @see sigmoid_ptask
 ** @see nnl2_sigmoid_vector_float64_full_out
 **/
void* nnl2_own_psigmoid_float64_full(void* arg) {
    sigmoid_ptask* task = (sigmoid_ptask*)arg;
    double* src_data = (double*)task->src_data;
    double* dst_data = (double*)task->dst_data;
    size_t start = task->start_idx;
    size_t end = task->end_idx;
    size_t size = end - start;
    
    nnl2_sigmoid_vector_float64_full_out(src_data + start, dst_data + start, size);
    return NULL;
}

/** @brief
 * Worker function for parallel approximate sigmoid on float32 data (out-of-place)
 *
 ** @param arg
 * Pointer to sigmoid_ptask structure containing work parameters
 *
 ** @return
 * Always returns NULL (required by pthread interface)
 *
 ** @details
 * Processes a chunk of single-precision data using the super-approximation method.
 * Uses AVX256 vectorization for optimal performance on large data chunks.
 *
 ** @see sigmoid_ptask
 ** @see nnl2_sigmoid_vector_float32_superapprox_out
 **/
void* nnl2_own_psigmoid_float32_superapprox(void* arg) {
    sigmoid_ptask* task = (sigmoid_ptask*)arg;
    float* src_data = (float*)task->src_data;
    float* dst_data = (float*)task->dst_data;
    size_t start = task->start_idx;
    size_t end = task->end_idx;
    size_t size = end - start;
    
    nnl2_sigmoid_vector_float32_superapprox_out(src_data + start, dst_data + start, size);
    return NULL;
}

/** @brief
 * Worker function for parallel exact sigmoid on float32 data (out-of-place)
 *
 ** @param arg
 * Pointer to sigmoid_ptask structure containing work parameters
 *
 ** @return
 * Always returns NULL (required by pthread interface)
 *
 ** @details
 * Processes a chunk of single-precision data using the exact sigmoid calculation.
 * Optimized for parallel processing with proper memory alignment handling.
 *
 ** @see sigmoid_ptask
 ** @see nnl2_sigmoid_vector_float32_full_out
 **/
void* nnl2_own_psigmoid_float32_full(void* arg) {
    sigmoid_ptask* task = (sigmoid_ptask*)arg;
    float* src_data = (float*)task->src_data;
    float* dst_data = (float*)task->dst_data;
    size_t start = task->start_idx;
    size_t end = task->end_idx;
    size_t size = end - start;
    
    nnl2_sigmoid_vector_float32_full_out(src_data + start, dst_data + start, size);
    return NULL;
}

/** @brief
 * Worker function for parallel sigmoid on INT32 data with FLOAT64 output
 *
 ** @param arg
 * Pointer to sigmoid_ptask structure containing work parameters
 *
 ** @return
 * Always returns NULL (required by pthread interface)
 *
 ** @details
 * Processes a chunk of INT32 data, converts to FLOAT64 and applies sigmoid.
 * Supports both approximate and exact sigmoid calculations.
 */
void* nnl2_own_psigmoid_int32_to_float64(void* arg) {
    sigmoid_ptask* task = (sigmoid_ptask*)arg;
    int32_t* src_data = (int32_t*)task->src_data;
    double* dst_data = (double*)task->dst_data;
    size_t start = task->start_idx;
    size_t end = task->end_idx;
    size_t size = end - start;
    
    nnl2_sigmoid_vector_int32_to_float64_out(src_data + start, dst_data + start, size, task->approx);
    return NULL;
}

/** @brief
 * High-performance parallel implementation of sigmoid activation function
 *
 ** @param tensor
 * Pointer to the input tensor
 *
 ** @param approx
 * Whether to use approximation for faster computation
 *
 ** @return
 * Pointer to a new tensor containing the sigmoid-activated values
 * Returns NULL in case of failure
 *
 ** @details
 * Uses multi-threading and SIMD vectorization for optimal performance on large tensors.
 * Automatically selects between naive and optimized implementations based on tensor size.
 * Supports both float32 and float64 data types with proper error handling.
 * For INT32 input, returns FLOAT64 output tensor.
 * For very large tensors, uses extreme optimizations with non-temporal stores.
 *
 ** @see naive_sigmoid
 ** @see nnl2_empty
 ** @see nnl2_sigmoid_vector_float64_superapprox_out
 ** @see nnl2_sigmoid_vector_float32_superapprox_out
 **/
Tensor* nnl2_own_sigmoid(Tensor* tensor, bool approx) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        if(tensor == NULL) {
            NNL2_ERROR("Passed tensor is NULL");
            return NULL;
        }
    #endif

    size_t total_elems = nnl2_product(tensor->shape, tensor->rank);    
    Tensor* result = NULL;
    
    // For INT32 input, create FLOAT64 output tensor
    if (tensor->dtype == INT32) {
        result = nnl2_empty(tensor->shape, tensor->rank, FLOAT64);
    } else {
        result = nnl2_empty(tensor->shape, tensor->rank, tensor->dtype);
    }
    
    if(total_elems == 0) return result;

    // Fallback to naive implementation for small tensors
    if(total_elems < NNL2_SIGMOID_PARALLEL_THRESHOLD) {
        Tensor* naive_result = naive_sigmoid(tensor, approx);
        if (naive_result) {
            // Copy data from naive_result to result
            size_t data_size = total_elems * get_dtype_size(result->dtype);
            memcpy(result->data, naive_result->data, data_size);
            nnl2_free_tensor(naive_result);
        }
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return result;
    }
    
    // For INT32 input with parallel processing, handle specially
    if (tensor->dtype == INT32) {
        size_t num_threads = NNL2_NUM_THREADS;
        pthread_t threads[num_threads];
        sigmoid_ptask tasks[num_threads];
        
        // Calculate optimal chunk sizes with load balancing
        size_t chunk = total_elems / num_threads;
        size_t remainder = total_elems % num_threads;
        
        // Configure tasks for INT32 to FLOAT64 conversion
        size_t current_start = 0;
        for (size_t i = 0; i < num_threads; i++) {
            size_t current_chunk = chunk + (i < remainder ? 1 : 0);
            
            tasks[i].dtype = tensor->dtype;
            tasks[i].aligned = false; // INT32 to FLOAT64 conversion doesn't benefit from alignment as much
            tasks[i].approx = approx;
            tasks[i].start_idx = current_start;
            tasks[i].end_idx = current_start + current_chunk;
            tasks[i].src_data = tensor->data;
            tasks[i].dst_data = result->data;
            
            int status = pthread_create(&threads[i], NULL, nnl2_own_psigmoid_int32_to_float64, &tasks[i]);
            if(status != 0) {
                NNL2_THREAD_CREATE_ERROR(status, "nnl2_own_sigmoid");
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
    
    // Original parallel processing for FLOAT32 and FLOAT64
    bool src_aligned = NNL2_IS_ALIGNED(tensor->data, NNL2_TENSOR_ALIGNMENT_32);
    bool dst_aligned = NNL2_IS_ALIGNED(result->data, NNL2_TENSOR_ALIGNMENT_32);
    bool is_aligned = src_aligned && dst_aligned;
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!is_aligned) {
            NNL2_WARN("In nnl2_own_sigmoid, tensor memory is not aligned to 32 bytes. Performance may be suboptimal");
        }
    #endif
    
    size_t num_threads = NNL2_NUM_THREADS;
    pthread_t threads[num_threads];
    sigmoid_ptask tasks[num_threads];
    
    // Calculate optimal chunk sizes with load balancing
    size_t chunk = total_elems / num_threads;
    size_t remainder = total_elems % num_threads;
    
    // Configure tasks
    for (size_t i = 0; i < num_threads; i++) {
        tasks[i].dtype = tensor->dtype;
        tasks[i].aligned = is_aligned;
        tasks[i].approx = approx;
        tasks[i].src_data = tensor->data;
        tasks[i].dst_data = result->data;
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
                worker_func = approx ? nnl2_own_psigmoid_float64_superapprox : nnl2_own_psigmoid_float64_full;
                break;
            }
			
            case FLOAT32: {
                worker_func = approx ? nnl2_own_psigmoid_float32_superapprox : nnl2_own_psigmoid_float32_full;
                break;
            }
			
            case INT32: {
                // This case should be handled above, but included for completeness
                worker_func = nnl2_own_psigmoid_int32_to_float64;
                break;
            }
			
            default: {
                NNL2_TYPE_ERROR(tensor->dtype);
                nnl2_free_tensor(result);
                return NULL;
            }
        }
        
        int status = pthread_create(&threads[i], NULL, worker_func, &tasks[i]);
        if(status != 0) {
            NNL2_THREAD_CREATE_ERROR(status, "nnl2_own_sigmoid");
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

#endif

/**
 * @ingroup backend_system
 * @brief Backend implementations for sigmoid operation
 * @details
 * Array follows the common backend registration pattern for sigmoid operations.
 * Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for sigmoid activation function
 *  - nnl2_own: Own nnl2 implementation for sigmoid activation
 * 
 * @see nnl2_naive
 * @see naive_sigmoid
 */
Implementation sigmoid_backends[] = {
    REGISTER_BACKEND(naive_sigmoid, nnl2_naive, NAIVE_BACKEND_NAME),
    
    #ifdef NNL2_PTHREAD_AVAILABLE
        REGISTER_BACKEND(nnl2_own_sigmoid, nnl2_own, NNL2_OWN_NAME),
    #endif
};    

/**
 * @brief Function pointer for sigmoid operation
 * @ingroup backend_system 
 */
sigmoidfn sigmoid;

/** 
 * @brief Makes the sigmoid backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(sigmoid);

/** 
 * @brief Sets the backend for sigmoid operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for sigmoid
 * @see ESET_BACKEND_BY_NAME
 */
void set_sigmoid_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(sigmoid_backends, sigmoid, backend_name, current_backend(sigmoid));
}

/** 
 * @brief Gets the name of the active backend for sigmoid operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_sigmoid_backend() {
    return current_backend(sigmoid);
}

/** 
 * @brief Function declaration for getting all available sigmoid backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(sigmoid);

/**
 * @brief Function declaration for getting the number of available sigmoid backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(sigmoid);

#endif /** NNL2_SIGMOID_H **/
