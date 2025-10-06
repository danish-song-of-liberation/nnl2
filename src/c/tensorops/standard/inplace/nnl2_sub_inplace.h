#ifndef NNL2_SUB_INPLACE_H
#define NNL2_SUB_INPLACE_H

/** @brief 
 * Performs element-wise subtraction of two tensors (naive implementation)
 * 
 * Subtracts the elements of the subtrahend tensor from the corresponding elements 
 * of the minuend tensor, modifying the minuend tensor in place
 *
 ** @param minuend 
 * Pointer to the tensor that will be modified (receives the subtraction result)
 *
 ** @param subtrahend 
 * Pointer to the tensor whose values will be subtracted from the minuend
 *
 ** @note 
 * Supports different data types through automatic conversion
 * The subtrahend elements are converted to the minuend's data type
 *
 ** @note 
 * In max safety mode, validates tensor pointers and data pointers before access
 *
 ** @warning 
 * This function modifies the minuend tensor directly
 *
 * @example
 * // Create two tensors with the same shape
 * Tensor* a = nnl2_ones((int[]){2, 3}, 2, FLOAT32);
 * Tensor* b = nnl2_ones((int[]){2, 3}, 2, FLOAT32);
 * 
 * // Subtract b from a (a becomes a - b)
 * naive_subinplace(a, b);
 * 
 * // Now a contains 0.0 in all elements
 * nnl2_quick_print_tensor(a);
 * 
 * // Cleanup
 * nnl2_free_tensor(a);
 * nnl2_free_tensor(b);
 */
void nnl2_naive_subinplace(Tensor* minuend, const Tensor* subtrahend) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks at the maximum safety level
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(minuend, "Minuend tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(minuend->data, "Minuend tensor's data is NULL");
        
        NNL2_CHECK_NULL_IF_ERR_RETURN(subtrahend, "Subtrahend tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(subtrahend->data, "Subtrahend tensor's data is NULL");
    #endif
    
    // Calculating the total number of elements in the minuend tensor
    size_t len_minuend = product(minuend->shape, minuend->rank);
    
    // If the tensor is empty, exit the function
    if(len_minuend == 0) return;
    
    TensorType dtype_minuend = minuend->dtype;
    TensorType dtype_subtrahend = subtrahend->dtype;
    
    if(dtype_minuend == dtype_subtrahend) {
        // Handling case when the tensors have the same type
        
        switch(dtype_minuend) {
            case FLOAT64: {
                volatile double* data_minuend = (double*)minuend->data;
                volatile double* data_subtrahend = (double*)subtrahend->data;
                
                // Element-wise subtraction
                for(size_t i = 0; i < len_minuend; i++) data_minuend[i] -= data_subtrahend[i];
                break;
            }
            
            case FLOAT32: {
                volatile float* data_minuend = (float*)minuend->data;
                volatile float* data_subtrahend = (float*)subtrahend->data;
                
                // Element-wise subtraction
                for(size_t i = 0; i < len_minuend; i++) data_minuend[i] -= data_subtrahend[i];    
                break;
            }
            
            case INT32: {
                volatile int32_t* data_minuend = (int32_t*)minuend->data;
                volatile int32_t* data_subtrahend = (int32_t*)subtrahend->data;
                
                // Element-wise subtraction
                for(size_t i = 0; i < len_minuend; i++) data_minuend[i] -= data_subtrahend[i];        
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(dtype_minuend);
                return;
            }
        }
    } else {
        // Handling the case when tensors have different data types
        // Calculating the step size for accessing subtrahend tensor elements
        size_t subtrahend_step = get_dtype_size(dtype_subtrahend);
        
        // Casting subtrahend data to char* for byte access
        char* subtrahend_data = (char*)subtrahend->data;
        
        switch(dtype_minuend) {
            case FLOAT64: {
                volatile double* data_minuend = (double*)minuend->data;
                
                // For each element, convert the subtrahend element to FLOAT64 and subtract it
                for(size_t i = 0; i < len_minuend; i++) {
                    void* subtrahend_elem = subtrahend_data + i * subtrahend_step;
                    data_minuend[i] -= nnl2_convert_to_float64(subtrahend_elem, dtype_subtrahend);
                }
                
                break; 
            }
            
            case FLOAT32: {
                volatile float* data_minuend = (float*)minuend->data;
                
                // For each element, convert the subtrahend element to FLOAT32 and subtract it
                for(size_t i = 0; i < len_minuend; i++) {
                    void* subtrahend_elem = subtrahend_data + i * subtrahend_step;
                    data_minuend[i] -= nnl2_convert_to_float32(subtrahend_elem, dtype_subtrahend);
                }
                
                break; 
            }
            
            case INT32: {
                volatile int32_t* data_minuend = (int32_t*)minuend->data;
                
                // For each element, convert the subtrahend element to INT32 and subtract it
                for(size_t i = 0; i < len_minuend; i++) {
                    void* subtrahend_elem = subtrahend_data + i * subtrahend_step;
                    data_minuend[i] -= nnl2_convert_to_int32(subtrahend_elem, dtype_subtrahend);
                }
                
                break; 
            }
            
            default: {
                NNL2_TYPE_ERROR(dtype_minuend);
                return;
            }
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}



#ifdef NNL2_AVX256_AVAILABLE

// Declarations

/** @brief 
 * AVX256 optimized subtraction for double with the same types
 *
 ** @param minuend 
 * Pointer to the minuend data (mutable)
 *
 ** @param subtrahend 
 * Pointer to the subtrahend data
 *
 ** @param len 
 * Number of elements to process
 *
 ** @param aligned_minuend 
 * Flag for aligning the minuend data
 *
 ** @param aligned_subtrahend 
 * Flag for aligning the subtrahend data
 */
static inline void nnl2_avx_sub_float64_same_type(double* minuend, double* subtrahend, size_t len, bool aligned_minuend, bool aligned_subtrahend);

/** @brief
 * AVX256 optimized subtraction for float with the same types
 * Documentation is identical to that of nnl2_avx_sub_float64_same_type
 *
 ** @see nnl2_avx_sub_float64_same_type
 **/
static inline void nnl2_avx_sub_float32_same_type(float* minuend, float* subtrahend, size_t len, bool aligned_minuend, bool aligned_subtrahend);

/** @brief
 * AVX256 optimized subtraction for int32 with the same types
 * Documentation is identical to that of nnl2_avx_sub_float64_same_type
 *
 ** @see nnl2_avx_sub_float64_same_type
 **/
static inline void nnl2_avx_sub_int32_same_type(int32_t* minuend, int32_t* subtrahend, size_t len, bool aligned_minuend, bool aligned_subtrahend);

/** @brief
 * AVX256 optimized subtraction for double with different types
 *
 ** @param minuend 
 * Pointer to the minuend tensor data (mutable)
 * 
 ** @param subtrahend 
 * Pointer to the subtrahend tensor data (may be of a different type)
 *
 ** @param len 
 * Number of elements to process
 *
 ** @param aligned_minuend 
 * Flag for aligning the minuend data
 */
static inline void nnl2_avx_sub_float64_diff_type(double* minuend, const Tensor* subtrahend, size_t len, bool aligned_minuend);

/** @brief
 * AVX256 optimized subtraction for float with different types
 * Documentation is identical to that of nnl2_avx_sub_float64_diff_type
 *
 ** @see nnl2_avx_sub_float64_diff_type
 **/
static inline void nnl2_avx_sub_float32_diff_type(float* minuend, const Tensor* subtrahend, size_t len, bool aligned_minuend);

/** @brief
 * AVX256 optimized subtraction for int32 with different types
 * Documentation is identical to that of nnl2_avx_sub_float64_diff_type
 *
 ** @see nnl2_avx_sub_float64_diff_type
 **/
static inline void nnl2_avx_sub_int32_diff_type(int32_t* minuend, const Tensor* subtrahend, size_t len, bool aligned_minuend);

// Main function

/** @brief
 * AVX256-optimized in-place subtraction operation 
 *
 ** @param minuend
 * A tensor from which values are subtracted (mutable)
 *
 ** @param subtrahend
 * The tensor whose values are being subtracted
 *
 ** @note
 * Additional checks may be performed depending on the safety level
 *
 ** @note
 * Supports type conversion
 *
 ** @note
 * Tensors can be either memory-aligned or non-memory-aligned
 *
 ** @warning
 * if the tensors are not memory-aligned, the calculations may be slightly slower
 *
 ** @see nnl2_avx_sub_float64_same_type
 ** @see nnl2_avx_sub_float32_same_type
 ** @see nnl2_avx_sub_int32_same_type
 **
 ** @see nnl2_avx_sub_float64_diff_type
 ** @see nnl2_avx_sub_float32_diff_type
 ** @see nnl2_avx_sub_int32_diff_type
 **/
void nnl2_avx256_subinplace(Tensor* minuend, const Tensor* subtrahend) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks at the maximum safety level
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(minuend, "Minuend tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(minuend->data, "Minuend tensor's data is NULL");
        
        NNL2_CHECK_NULL_IF_ERR_RETURN(subtrahend, "Subtrahend tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(subtrahend->data, "Subtrahend tensor's data is NULL");
    #endif
    
    // Calculating the total number of elements in the minuend tensor
    size_t len_minuend = product(minuend->shape, minuend->rank);    
    
    // If the tensor is empty, exit the function
    if(len_minuend == 0) return;
    
    TensorType dtype_minuend = minuend->dtype;
    TensorType dtype_subtrahend = subtrahend->dtype;
    
    bool is_aligned_minuend = NNL2_IS_ALIGNED(minuend->data, NNL2_TENSOR_ALIGNMENT_32);
    bool is_aligned_subtrahend = NNL2_IS_ALIGNED(subtrahend->data, NNL2_TENSOR_ALIGNMENT_32);
    
    // Warning for unaligned memory in safety modes (performance impact)
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MINI
        if(!is_aligned_minuend) {
            NNL2_WARN("In the avx256 implementation of sub in-place, minuend memory is not aligned to 32 bytes. Calculations may be slightly slower");
        }
        
        if(!is_aligned_subtrahend && dtype_minuend == dtype_subtrahend) {
            NNL2_WARN("In the avx256 implementation of sub in-place, subtrahend memory is not aligned to 32 bytes. Calculations may be slightly slower");
        }
    #endif
    
    if(dtype_minuend == dtype_subtrahend) {
        // Handling case when the tensors have the same type
        
        switch (dtype_minuend) {
            case FLOAT64: nnl2_avx_sub_float64_same_type((double*)minuend->data, (double*)subtrahend->data, len_minuend, is_aligned_minuend, is_aligned_subtrahend);  break;
            case FLOAT32: nnl2_avx_sub_float32_same_type((float*)minuend->data, (float*)subtrahend->data, len_minuend, is_aligned_minuend, is_aligned_subtrahend);    break;    
            case INT32:   nnl2_avx_sub_int32_same_type((int32_t*)minuend->data, (int32_t*)subtrahend->data, len_minuend, is_aligned_minuend, is_aligned_subtrahend);  break;
            
            default: {
                NNL2_TYPE_ERROR(dtype_minuend);
                return;
            }
        }
    } else {
        // Handling the case when tensors have different data types

        switch(dtype_minuend) {
            case FLOAT64: nnl2_avx_sub_float64_diff_type((double*)minuend->data, subtrahend, len_minuend, is_aligned_minuend);  break;
            case FLOAT32: nnl2_avx_sub_float32_diff_type((float*)minuend->data, subtrahend, len_minuend, is_aligned_minuend);   break;
            case INT32:   nnl2_avx_sub_int32_diff_type((int32_t*)minuend->data, subtrahend, len_minuend, is_aligned_minuend);   break;
            
            default: {
                NNL2_TYPE_ERROR(dtype_minuend);
                return;
            }
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

// Implementations of auxiliary functions for the same type

/** @brief 
 * Implementation of double subtraction with the same types
 *
 ** @details 
 * Handles 4 combinations of memory alignment:
 * - The minuend and subtrahend tensors are aligned in memory
 * - The minuend tensor is aligned in memory, but the subtrahend is not
 * - The subtrahend tensor is aligned in memory, but the minuend is not
 * - The minuend and subtrahend tensors are not aligned in memory
 *
 ** @see nnl2_avx256_subinplace
 **/
static inline void nnl2_avx_sub_float64_same_type(double* minuend, double* subtrahend, size_t len, bool aligned_minuend, bool aligned_subtrahend) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
    
    size_t i = 0;
    
    // Case 1: Both tensors are aligned 
    if(aligned_minuend && aligned_subtrahend) {
        for(; i + 3 < len; i += 4) {
            __m256d v_minuend = _mm256_load_pd(&minuend[i]);        // Fast loading of aligned data
            __m256d v_subtrahend = _mm256_load_pd(&subtrahend[i]);  // Fast loading of aligned data
            __m256d v_result = _mm256_sub_pd(v_minuend, v_subtrahend);  // Vector subtraction
            _mm256_store_pd(&minuend[i], v_result);                 // Fast saving to aligned memory
        }
    } 
    
    // Case 2: Only the minuend is aligned
    else if(aligned_minuend) {
        for(; i + 3 < len; i += 4) {
            __m256d v_minuend = _mm256_load_pd(&minuend[i]);            // Fast loading of aligned data
            __m256d v_subtrahend = _mm256_loadu_pd(&subtrahend[i]);     // Slow loading of unaligned data
            __m256d v_result = _mm256_sub_pd(v_minuend, v_subtrahend);  // Vector subtraction
            _mm256_store_pd(&minuend[i], v_result);                     // Fast saving to aligned memory
        }
    } 
    
    // Case 3: Only the subtrahend is aligned
    else if(aligned_subtrahend) {
        for(; i + 3 < len; i += 4) {
            __m256d v_minuend = _mm256_loadu_pd(&minuend[i]);           // Slow loading of unaligned data
            __m256d v_subtrahend = _mm256_load_pd(&subtrahend[i]);      // Fast loading of aligned data
            __m256d v_result = _mm256_sub_pd(v_minuend, v_subtrahend);  // Vector subtraction
            _mm256_storeu_pd(&minuend[i], v_result);                    // Slow saving to unaligned memory
        }
    } 
    
    // Case 4: Both tensors are not aligned
    else {
        for(; i + 3 < len; i += 4) {
            __m256d v_minuend = _mm256_loadu_pd(&minuend[i]);           // Slow loading of unaligned data
            __m256d v_subtrahend = _mm256_loadu_pd(&subtrahend[i]);     // Slow loading of unaligned data
            __m256d v_result = _mm256_sub_pd(v_minuend, v_subtrahend);  // Vector subtraction
            _mm256_storeu_pd(&minuend[i], v_result);                    // Slow saving to unaligned memory
        }
    }
    
    // Processing the remainder
    for(; i < len; i++) minuend[i] -= subtrahend[i];
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
}

/** @brief 
 * Implementation of float subtraction with the same types
 * Similar to double, but processes 8 elements per iteration
 *
 ** @see nnl2_avx256_subinplace
 ** @see nnl2_avx_sub_float64_same_type
 **/
static inline void nnl2_avx_sub_float32_same_type(float* minuend, float* subtrahend, size_t len, bool aligned_minuend, bool aligned_subtrahend) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
    
    size_t i = 0;
    
    // Case 1: Both tensors are aligned 
    if(aligned_minuend && aligned_subtrahend) {
        for(; i + 7 < len; i += 8) {
            __m256 v_minuend = _mm256_load_ps(&minuend[i]);         // Fast loading of aligned data
            __m256 v_subtrahend = _mm256_load_ps(&subtrahend[i]);   // Fast loading of aligned data
            __m256 v_result = _mm256_sub_ps(v_minuend, v_subtrahend);  // Vector subtraction
            _mm256_store_ps(&minuend[i], v_result);                 // Fast saving to aligned memory
        }
    } 
    
    // Case 2: Only the minuend is aligned
    else if(aligned_minuend) {
        for(; i + 7 < len; i += 8) {
            __m256 v_minuend = _mm256_load_ps(&minuend[i]);             // Fast loading of aligned data
            __m256 v_subtrahend = _mm256_loadu_ps(&subtrahend[i]);      // Slow loading of unaligned data
            __m256 v_result = _mm256_sub_ps(v_minuend, v_subtrahend);   // Vector subtraction
            _mm256_store_ps(&minuend[i], v_result);                     // Fast saving to aligned memory
        }
    } 
    
    // Case 3: Only the subtrahend is aligned
    else if(aligned_subtrahend) {
        for(; i + 7 < len; i += 8) {
            __m256 v_minuend = _mm256_loadu_ps(&minuend[i]);            // Slow loading of unaligned data
            __m256 v_subtrahend = _mm256_load_ps(&subtrahend[i]);       // Fast loading of aligned data
            __m256 v_result = _mm256_sub_ps(v_minuend, v_subtrahend);   // Vector subtraction
            _mm256_storeu_ps(&minuend[i], v_result);                    // Slow saving to unaligned memory
        }
    } 
    
    // Case 4: Both tensors are not aligned
    else {
        for(; i + 7 < len; i += 8) {    
            __m256 v_minuend = _mm256_loadu_ps(&minuend[i]);            // Slow loading of unaligned data
            __m256 v_subtrahend = _mm256_loadu_ps(&subtrahend[i]);      // Slow loading of unaligned data
            __m256 v_result = _mm256_sub_ps(v_minuend, v_subtrahend);   // Vector subtraction
            _mm256_storeu_ps(&minuend[i], v_result);                    // Slow saving to unaligned memory
        }
    }
    
    // Processing the remainder
    for(; i < len; i++) minuend[i] -= subtrahend[i];
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
}

/** @brief 
 * Implementation of int32 subtraction with the same types
 *
 ** @see nnl2_avx256_subinplace
 ** @see nnl2_avx_sub_float64_same_type
 ** @see nnl2_avx_sub_float32_same_type
 **/
static inline void nnl2_avx_sub_int32_same_type(int32_t* minuend, int32_t* subtrahend, size_t len, bool aligned_minuend, bool aligned_subtrahend) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
    
    size_t i = 0;
    
    // Case 1: Both tensors are aligned 
    if(aligned_minuend && aligned_subtrahend) {
        for(; i + 7 < len; i += 8) {
            __m256i v_minuend = _mm256_load_si256((__m256i*)&minuend[i]);       // Fast loading of aligned data
            __m256i v_subtrahend = _mm256_load_si256((__m256i*)&subtrahend[i]); // Fast loading of aligned data
            __m256i v_result = _mm256_sub_epi32(v_minuend, v_subtrahend);       // Vector subtraction
            _mm256_store_si256((__m256i*)&minuend[i], v_result);                // Fast saving to aligned memory
        }
    } 
    
    // Case 2: Only the minuend is aligned
    else if(aligned_minuend) {
        for(; i + 7 < len; i += 8) {
            __m256i v_minuend = _mm256_load_si256((__m256i*)&minuend[i]);           // Fast loading of aligned data
            __m256i v_subtrahend = _mm256_loadu_si256((__m256i*)&subtrahend[i]);    // Slow loading of unaligned data
            __m256i v_result = _mm256_sub_epi32(v_minuend, v_subtrahend);           // Vector subtraction
            _mm256_store_si256((__m256i*)&minuend[i], v_result);                    // Fast saving to aligned memory
        }
    } 
    
    // Case 3: Only the subtrahend is aligned
    else if(aligned_subtrahend) {
        for(; i + 7 < len; i += 8) {
            __m256i v_minuend = _mm256_loadu_si256((__m256i*)&minuend[i]);          // Slow loading of unaligned data
            __m256i v_subtrahend = _mm256_load_si256((__m256i*)&subtrahend[i]);     // Fast loading of aligned data
            __m256i v_result = _mm256_sub_epi32(v_minuend, v_subtrahend);           // Vector subtraction
            _mm256_storeu_si256((__m256i*)&minuend[i], v_result);                   // Slow saving to unaligned memory
        }
    } 
    
    // Case 4: Both tensors are not aligned
    else {
        for(; i + 7 < len; i += 8) {
            __m256i v_minuend = _mm256_loadu_si256((__m256i*)&minuend[i]);          // Slow loading of unaligned data
            __m256i v_subtrahend = _mm256_loadu_si256((__m256i*)&subtrahend[i]);    // Slow loading of unaligned data
            __m256i v_result = _mm256_sub_epi32(v_minuend, v_subtrahend);           // Vector subtraction
            _mm256_storeu_si256((__m256i*)&minuend[i], v_result);                   // Slow saving to unaligned memory
        }
    }
    
    // Processing the remainder
    for(; i < len; i++) minuend[i] -= subtrahend[i];
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
}

// implementations of auxiliary functions for different types

/** @brief 
 * Implementation of double subtraction with conversion from other types
 *
 ** @details 
 * Converts subtrahend elements to double before subtraction
 *
 ** @see nnl2_avx256_subinplace
 **/
static inline void nnl2_avx_sub_float64_diff_type(double* minuend, const Tensor* subtrahend, size_t len, bool aligned_minuend) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
    
    // Calculating the step between elements in bytes (for accessing raw data)
    size_t subtrahend_step = get_dtype_size(subtrahend->dtype);
    char* subtrahend_data = (char*)subtrahend->data;
    
    size_t i = 0;
    
    // Vector processing of 4 elements per iteration
    if(aligned_minuend) {
        for(; i + 3 < len; i += 4) {
            // Loading 4 double from minuend
            __m256d v_minuend = _mm256_load_pd(&minuend[i]);
            
            // Conversion and creation of a vector of 4 doubles
            // _mm256_set_pd fills the vector in reverse order (from oldest to youngest)
            __m256d v_subtrahend = _mm256_set_pd(
                nnl2_convert_to_float64(subtrahend_data + (i + 3) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_float64(subtrahend_data + (i + 2) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_float64(subtrahend_data + (i + 1) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_float64(subtrahend_data + (i + 0) * subtrahend_step, subtrahend->dtype)
            );
            
            // Vector subtraction and saving the result
            __m256d v_result = _mm256_sub_pd(v_minuend, v_subtrahend);
            _mm256_store_pd(&minuend[i], v_result);
        }
    } else {
        // Similarly, but with unaligned memory
        
        for(; i + 3 < len; i += 4) {
            __m256d v_minuend = _mm256_loadu_pd(&minuend[i]);
            __m256d v_subtrahend = _mm256_set_pd(
                nnl2_convert_to_float64(subtrahend_data + (i + 3) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_float64(subtrahend_data + (i + 2) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_float64(subtrahend_data + (i + 1) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_float64(subtrahend_data + (i + 0) * subtrahend_step, subtrahend->dtype)
            );
            
            __m256d v_result = _mm256_sub_pd(v_minuend, v_subtrahend);
            _mm256_storeu_pd(&minuend[i], v_result);
        }
    }
    
    // Scalar processing of the remaining elements
    for(; i < len; i++) {
        void* subtrahend_elem = subtrahend_data + i * subtrahend_step;
        minuend[i] -= nnl2_convert_to_float64(subtrahend_elem, subtrahend->dtype);
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
}

/** @brief 
 * Implementation of float subtraction with conversion from other types
 *
 ** @details 
 * Converts subtrahend elements to float before subtraction
 *
 ** @see nnl2_avx256_subinplace
 **/
static inline void nnl2_avx_sub_float32_diff_type(float* minuend, const Tensor* subtrahend, size_t len, bool aligned_minuend) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
    
    size_t subtrahend_step = get_dtype_size(subtrahend->dtype);
    char* subtrahend_data = (char*)subtrahend->data;
    
    size_t i = 0;
    
    // Vector processing of 8 elements per iteration
    if(aligned_minuend) {
        for(; i + 7 < len; i += 8) {
            __m256 v_minuend = _mm256_load_ps(&minuend[i]);
            
            // Creating a vector of 8 floats with conversion
            // _mm256_set_ps fills in reverse order
            __m256 v_subtrahend = _mm256_set_ps(
                nnl2_convert_to_float32(subtrahend_data + (i + 7) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_float32(subtrahend_data + (i + 6) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_float32(subtrahend_data + (i + 5) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_float32(subtrahend_data + (i + 4) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_float32(subtrahend_data + (i + 3) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_float32(subtrahend_data + (i + 2) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_float32(subtrahend_data + (i + 1) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_float32(subtrahend_data + (i + 0) * subtrahend_step, subtrahend->dtype)
            );
            
            __m256 v_result = _mm256_sub_ps(v_minuend, v_subtrahend);
            _mm256_store_ps(&minuend[i], v_result);
        }
    } else {
        // Similarly for unaligned memory
        for(; i + 7 < len; i += 8) {
            __m256 v_minuend = _mm256_loadu_ps(&minuend[i]);
            __m256 v_subtrahend = _mm256_set_ps(
                nnl2_convert_to_float32(subtrahend_data + (i + 7) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_float32(subtrahend_data + (i + 6) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_float32(subtrahend_data + (i + 5) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_float32(subtrahend_data + (i + 4) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_float32(subtrahend_data + (i + 3) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_float32(subtrahend_data + (i + 2) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_float32(subtrahend_data + (i + 1) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_float32(subtrahend_data + (i + 0) * subtrahend_step, subtrahend->dtype)
            );
            
            __m256 v_result = _mm256_sub_ps(v_minuend, v_subtrahend);
            _mm256_storeu_ps(&minuend[i], v_result);
        }
    }
    
    // Scalar processing of the remainder
    for(; i < len; i++) {
        void* subtrahend_elem = subtrahend_data + i * subtrahend_step;
        minuend[i] -= nnl2_convert_to_float32(subtrahend_elem, subtrahend->dtype);
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
}    

/** @brief 
 * Implementation of int32 subtraction with conversion from other types
 *
 ** @details 
 * Converts subtrahend elements to int32 before subtraction
 *
 ** @see nnl2_avx256_subinplace
 **/
static inline void nnl2_avx_sub_int32_diff_type(int32_t* minuend, const Tensor* subtrahend, size_t len, bool aligned_minuend) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
    
    size_t subtrahend_step = get_dtype_size(subtrahend->dtype);
    char* subtrahend_data = (char*)subtrahend->data;
    
    size_t i = 0;
    
    // Vector processing of 8 elements per iteration
    if(aligned_minuend) {
        for(; i + 7 < len; i += 8) {
            __m256i v_minuend = _mm256_load_si256((__m256i*)&minuend[i]);
            
            // Creating a vector of 8 int32s with conversion
            __m256i v_subtrahend = _mm256_set_epi32(
                nnl2_convert_to_int32(subtrahend_data + (i + 7) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_int32(subtrahend_data + (i + 6) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_int32(subtrahend_data + (i + 5) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_int32(subtrahend_data + (i + 4) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_int32(subtrahend_data + (i + 3) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_int32(subtrahend_data + (i + 2) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_int32(subtrahend_data + (i + 1) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_int32(subtrahend_data + (i + 0) * subtrahend_step, subtrahend->dtype)
            );
            
            __m256i v_result = _mm256_sub_epi32(v_minuend, v_subtrahend);
            _mm256_store_si256((__m256i*)&minuend[i], v_result);
        }
    } else {
        // Similarly for unaligned memory
        for(; i + 7 < len; i += 8) {
            __m256i v_minuend = _mm256_loadu_si256((__m256i*)&minuend[i]);
            __m256i v_subtrahend = _mm256_set_epi32(
                nnl2_convert_to_int32(subtrahend_data + (i + 7) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_int32(subtrahend_data + (i + 6) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_int32(subtrahend_data + (i + 5) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_int32(subtrahend_data + (i + 4) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_int32(subtrahend_data + (i + 3) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_int32(subtrahend_data + (i + 2) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_int32(subtrahend_data + (i + 1) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_int32(subtrahend_data + (i + 0) * subtrahend_step, subtrahend->dtype)
            );
            
            __m256i v_result = _mm256_sub_epi32(v_minuend, v_subtrahend);
            _mm256_storeu_si256((__m256i*)&minuend[i], v_result);
        }
    }
    
    // Scalar processing of the remainder
    for(; i < len; i++) {
        void* subtrahend_elem = subtrahend_data + i * subtrahend_step;
        minuend[i] -= nnl2_convert_to_int32(subtrahend_elem, subtrahend->dtype);
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
}

#endif

#ifdef NNL2_PTHREAD_AVAILABLE

/** @brief
 * Threshold for enabling parallel execution of the
 * subtraction operation during in-place calculations
 */
#define NNL2_SUB_INPLACE_PARALLEL_THRESHOLD 1000000

/** @brief 
 * Worker function for parallel double precision subtraction with same data types
 * 
 ** @param arg 
 * Pointer to subinplace_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 * 
 ** @details
 * Processes FLOAT64 data using AVX256 instructions with cache prefetching
 * for optimal memory access patterns
 */
void* nnl2_own_psub_float64_same_type(void* arg);

/** @brief
 * Worker function for parallel single precision subtraction with same data types
 * 
 ** @param arg 
 * Pointer to subinplace_ptask structure containing thread parameters  
 * 
 ** @return 
 * NULL (for pthread API compatibility)
 * 
 ** @see nnl2_own_psub_float64_same_type
 **/
void* nnl2_own_psub_float32_same_type(void* arg);

/** @brief
 * Worker function for parallel integer subtraction with same data types
 * 
 ** @param arg 
 * Pointer to subinplace_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 * 
 ** @see nnl2_own_psub_float64_same_type
 **/
void* nnl2_own_psub_int32_same_type(void* arg);

/** @brief
 * Worker function for parallel double precision subtraction with type conversion
 * 
 ** @param arg 
 * Pointer to subinplace_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 * 
 ** @details
 * Handles type conversion from subtrahend to minuend data type with AVX256
 * optimizations and prefetching
 */
void* nnl2_own_psub_float64_diff_type(void* arg);

/** @brief
 * Worker function for parallel single precision subtraction with type conversion
 * 
 ** @param arg 
 * Pointer to subinplace_ptask structure containing thread parameters
 *
 ** @return
 * NULL (for pthread API compatibility)
 * 
 ** @see nnl2_own_psub_float64_diff_type
 **/
void* nnl2_own_psub_float32_diff_type(void* arg);

/** @brief
 * Worker function for parallel integer subtraction with type conversion
 * 
 ** @param arg
 * Pointer to subinplace_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 * 
 ** @see nnl2_own_psub_float64_diff_type
 **/
void* nnl2_own_psub_int32_diff_type(void* arg);

/** @brief
 * High-performance parallel implementation of in-place subtraction
 * 
 ** @param minuend 
 * Pointer to tensor that will be modified in-place
 *
 ** @param subtrahend 
 * Pointer to tensor whose values will be subtracted from minuend
 * 
 ** @details
 * Combines AVX256 vectorization, multi-threading with pthread, and cache
 * prefetching for maximum performance on modern CPU architectures.
 * Automatically selects optimal thread count and chunk sizes.
 * 
 ** @note
 * Uses NNL2_NUM_THREADS for parallelization configuration
 * Falls back to naive implementation for small tensors
 */
void nnl2_own_sub_inplace(Tensor* minuend, const Tensor* subtrahend) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks at the maximum safety level
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(minuend, "Minuend tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(minuend->data, "Minuend tensor's data is NULL");
        
        NNL2_CHECK_NULL_IF_ERR_RETURN(subtrahend, "Subtrahend tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(subtrahend->data, "Subtrahend tensor's data is NULL");
    #endif
    
    // Calculating the total number of elements in the minuend tensor
    size_t len_minuend = product(minuend->shape, minuend->rank);
    
    // Fallback to naive implementation for small tensors
    if(len_minuend < NNL2_SUB_INPLACE_PARALLEL_THRESHOLD) {
        nnl2_naive_subinplace(minuend, subtrahend);
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return;
    }
    
    TensorType dtype_minuend = minuend->dtype;
    TensorType dtype_subtrahend = subtrahend->dtype;
    
    bool is_aligned_minuend = NNL2_IS_ALIGNED(minuend->data, NNL2_TENSOR_ALIGNMENT_32);
    bool is_aligned_subtrahend = NNL2_IS_ALIGNED(subtrahend->data, NNL2_TENSOR_ALIGNMENT_32);
    
    // Warning for unaligned memory in safety modes
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!is_aligned_minuend) {
            NNL2_WARN("In nnl2_own_2 sub in-place, minuend memory is not aligned to 32 bytes. Performance may be suboptimal");
        }
        
        if(!is_aligned_subtrahend && dtype_minuend == dtype_subtrahend) {
            NNL2_WARN("In nnl2_own_2 sub in-place, subtrahend memory is not aligned to 32 bytes. Performance may be suboptimal");
        }
    #endif
    
    size_t num_threads = NNL2_NUM_THREADS;
    pthread_t threads[num_threads];
    subinplace_ptask tasks[num_threads];
    
    // Calculate optimal chunk sizes with load balancing
    size_t chunk = len_minuend / num_threads;
    size_t remainder = len_minuend % num_threads;
    
    size_t current_start = 0;
    for (size_t i = 0; i < num_threads; i++) {
        size_t current_chunk = chunk + (i < remainder ? 1 : 0);
        
        // Configure task for this thread
        tasks[i].minuend_data = minuend->data;
        tasks[i].subtrahend_data = subtrahend->data;
        tasks[i].start = current_start;
        tasks[i].end = current_start + current_chunk;
        tasks[i].dtype_minuend = dtype_minuend;
        tasks[i].dtype_subtrahend = dtype_subtrahend;
        tasks[i].aligned_minuend = is_aligned_minuend;
        tasks[i].aligned_subtrahend = is_aligned_subtrahend;
        tasks[i].subtrahend_step = (dtype_minuend == dtype_subtrahend) ? get_dtype_size(dtype_subtrahend) : 0;
        
        // Select appropriate worker function based on data types
        void* (*worker_func)(void*) = NULL;
        
        if(dtype_minuend == dtype_subtrahend) {
            switch(dtype_minuend) {
                case FLOAT64: worker_func = nnl2_own_psub_float64_same_type; break;
                case FLOAT32: worker_func = nnl2_own_psub_float32_same_type; break;
                case INT32:   worker_func = nnl2_own_psub_int32_same_type;   break;
                default: {
                    NNL2_TYPE_ERROR(dtype_minuend);
                    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                        NNL2_FUNC_EXIT();
                    #endif
                    return;
                }
            }
        } else {
            tasks[i].subtrahend_step = get_dtype_size(dtype_subtrahend);
            switch(dtype_minuend) {
                case FLOAT64: worker_func = nnl2_own_psub_float64_diff_type; break;
                case FLOAT32: worker_func = nnl2_own_psub_float32_diff_type; break;
                case INT32:   worker_func = nnl2_own_psub_int32_diff_type;   break;
                default: {
                    NNL2_TYPE_ERROR(dtype_minuend);
                    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                        NNL2_FUNC_EXIT();
                    #endif
                    return;
                }
            }
        }
        
        // Create thread to process the assigned chunk
        int status = pthread_create(&threads[i], NULL, worker_func, &tasks[i]);
        if(status != 0) {
            NNL2_THREAD_CREATE_ERROR(status, "nnl2_own_2_subinplace");
            num_threads = i;
            break;
        }
        
        current_start += current_chunk;
    }
    
    // Wait for all threads to complete
    for (size_t i = 0; i < num_threads; i++) {
        int join_status = pthread_join(threads[i], NULL);
        if(join_status != 0) {
            NNL2_THREAD_JOIN_ERROR(join_status, "nnl2_own_2_subinplace");
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

// Worker function implementations with AVX256 and prefetching

/** @brief
 * See documentation at declaration
 * 
 ** @see nnl2_own_psub_float64_same_type
 **/
void* nnl2_own_psub_float64_same_type(void* arg) {
    subinplace_ptask* task = (subinplace_ptask*)arg;
    double* minuend = (double*)task->minuend_data;
    double* subtrahend = (double*)task->subtrahend_data;
    size_t start = task->start;
    size_t end = task->end;
    
    size_t i = start;
    
    // AVX256 processing with prefetching
    if(task->aligned_minuend && task->aligned_subtrahend) {
        for(; i + 3 < end; i += 4) {
            // Prefetch next cache lines
            _mm_prefetch((char*)&minuend[i + 16], _MM_HINT_T0);
            _mm_prefetch((char*)&subtrahend[i + 16], _MM_HINT_T0);
            
            __m256d v_minuend = _mm256_load_pd(&minuend[i]);
            __m256d v_subtrahend = _mm256_load_pd(&subtrahend[i]);
            __m256d v_result = _mm256_sub_pd(v_minuend, v_subtrahend);
            _mm256_store_pd(&minuend[i], v_result);
        }
    } else if(task->aligned_minuend) {
        for(; i + 3 < end; i += 4) {
            _mm_prefetch((char*)&minuend[i + 16], _MM_HINT_T0);
            _mm_prefetch((char*)&subtrahend[i + 16], _MM_HINT_T0);
            
            __m256d v_minuend = _mm256_load_pd(&minuend[i]);
            __m256d v_subtrahend = _mm256_loadu_pd(&subtrahend[i]);
            __m256d v_result = _mm256_sub_pd(v_minuend, v_subtrahend);
            _mm256_store_pd(&minuend[i], v_result);
        }
    } else if(task->aligned_subtrahend) {
        for(; i + 3 < end; i += 4) {
            _mm_prefetch((char*)&minuend[i + 16], _MM_HINT_T0);
            _mm_prefetch((char*)&subtrahend[i + 16], _MM_HINT_T0);
            
            __m256d v_minuend = _mm256_loadu_pd(&minuend[i]);
            __m256d v_subtrahend = _mm256_load_pd(&subtrahend[i]);
            __m256d v_result = _mm256_sub_pd(v_minuend, v_subtrahend);
            _mm256_storeu_pd(&minuend[i], v_result);
        }
    } else {
        for(; i + 3 < end; i += 4) {
            _mm_prefetch((char*)&minuend[i + 16], _MM_HINT_T0);
            _mm_prefetch((char*)&subtrahend[i + 16], _MM_HINT_T0);
            
            __m256d v_minuend = _mm256_loadu_pd(&minuend[i]);
            __m256d v_subtrahend = _mm256_loadu_pd(&subtrahend[i]);
            __m256d v_result = _mm256_sub_pd(v_minuend, v_subtrahend);
            _mm256_storeu_pd(&minuend[i], v_result);
        }
    }
    
    // Scalar processing for remainder
    for(; i < end; i++) {
        minuend[i] -= subtrahend[i];
    }
    
    return NULL;
}

/** @brief
 * See documentation at declaration
 * 
 ** @see nnl2_own_psub_float32_same_type
 **/
void* nnl2_own_psub_float32_same_type(void* arg) {
    subinplace_ptask* task = (subinplace_ptask*)arg;
    float* minuend = (float*)task->minuend_data;
    float* subtrahend = (float*)task->subtrahend_data;
    size_t start = task->start;
    size_t end = task->end;
    
    size_t i = start;
    
    // AVX256 processing with prefetching (8 elements per iteration)
    if(task->aligned_minuend && task->aligned_subtrahend) {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&minuend[i + 32], _MM_HINT_T0);
            _mm_prefetch((char*)&subtrahend[i + 32], _MM_HINT_T0);
            
            __m256 v_minuend = _mm256_load_ps(&minuend[i]);
            __m256 v_subtrahend = _mm256_load_ps(&subtrahend[i]);
            __m256 v_result = _mm256_sub_ps(v_minuend, v_subtrahend);
            _mm256_store_ps(&minuend[i], v_result);
        }
    } else if(task->aligned_minuend) {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&minuend[i + 32], _MM_HINT_T0);
            _mm_prefetch((char*)&subtrahend[i + 32], _MM_HINT_T0);
            
            __m256 v_minuend = _mm256_load_ps(&minuend[i]);
            __m256 v_subtrahend = _mm256_loadu_ps(&subtrahend[i]);
            __m256 v_result = _mm256_sub_ps(v_minuend, v_subtrahend);
            _mm256_store_ps(&minuend[i], v_result);
        }
    } else if(task->aligned_subtrahend) {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&minuend[i + 32], _MM_HINT_T0);
            _mm_prefetch((char*)&subtrahend[i + 32], _MM_HINT_T0);
            
            __m256 v_minuend = _mm256_loadu_ps(&minuend[i]);
            __m256 v_subtrahend = _mm256_load_ps(&subtrahend[i]);
            __m256 v_result = _mm256_sub_ps(v_minuend, v_subtrahend);
            _mm256_storeu_ps(&minuend[i], v_result);
        }
    } else {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&minuend[i + 32], _MM_HINT_T0);
            _mm_prefetch((char*)&subtrahend[i + 32], _MM_HINT_T0);
            
            __m256 v_minuend = _mm256_loadu_ps(&minuend[i]);
            __m256 v_subtrahend = _mm256_loadu_ps(&subtrahend[i]);
            __m256 v_result = _mm256_sub_ps(v_minuend, v_subtrahend);
            _mm256_storeu_ps(&minuend[i], v_result);
        }
    }
    
    // Scalar processing for remainder
    for(; i < end; i++) {
        minuend[i] -= subtrahend[i];
    }
    
    return NULL;
}

/** @brief
 * See documentation at declaration
 * 
 ** @see nnl2_own_psub_int32_same_type
 **/
void* nnl2_own_psub_int32_same_type(void* arg) {
    subinplace_ptask* task = (subinplace_ptask*)arg;
    int32_t* minuend = (int32_t*)task->minuend_data;
    int32_t* subtrahend = (int32_t*)task->subtrahend_data;
    size_t start = task->start;
    size_t end = task->end;
    
    size_t i = start;
    
    // AVX256 processing with prefetching (8 elements per iteration)
    if(task->aligned_minuend && task->aligned_subtrahend) {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&minuend[i + 32], _MM_HINT_T0);
            _mm_prefetch((char*)&subtrahend[i + 32], _MM_HINT_T0);
            
            __m256i v_minuend = _mm256_load_si256((__m256i*)&minuend[i]);
            __m256i v_subtrahend = _mm256_load_si256((__m256i*)&subtrahend[i]);
            __m256i v_result = _mm256_sub_epi32(v_minuend, v_subtrahend);
            _mm256_store_si256((__m256i*)&minuend[i], v_result);
        }
    } else if(task->aligned_minuend) {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&minuend[i + 32], _MM_HINT_T0);
            _mm_prefetch((char*)&subtrahend[i + 32], _MM_HINT_T0);
            
            __m256i v_minuend = _mm256_load_si256((__m256i*)&minuend[i]);
            __m256i v_subtrahend = _mm256_loadu_si256((__m256i*)&subtrahend[i]);
            __m256i v_result = _mm256_sub_epi32(v_minuend, v_subtrahend);
            _mm256_store_si256((__m256i*)&minuend[i], v_result);
        }
    } else if(task->aligned_subtrahend) {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&minuend[i + 32], _MM_HINT_T0);
            _mm_prefetch((char*)&subtrahend[i + 32], _MM_HINT_T0);
            
            __m256i v_minuend = _mm256_loadu_si256((__m256i*)&minuend[i]);
            __m256i v_subtrahend = _mm256_load_si256((__m256i*)&subtrahend[i]);
            __m256i v_result = _mm256_sub_epi32(v_minuend, v_subtrahend);
            _mm256_storeu_si256((__m256i*)&minuend[i], v_result);
        }
    } else {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&minuend[i + 32], _MM_HINT_T0);
            _mm_prefetch((char*)&subtrahend[i + 32], _MM_HINT_T0);
            
            __m256i v_minuend = _mm256_loadu_si256((__m256i*)&minuend[i]);
            __m256i v_subtrahend = _mm256_loadu_si256((__m256i*)&subtrahend[i]);
            __m256i v_result = _mm256_sub_epi32(v_minuend, v_subtrahend);
            _mm256_storeu_si256((__m256i*)&minuend[i], v_result);
        }
    }
    
    // Scalar processing for remainder
    for(; i < end; i++) {
        minuend[i] -= subtrahend[i];
    }
    
    return NULL;
}

// Different type worker functions with conversion and prefetching

/** @brief
 * See documentation at declaration
 * 
 ** @see nnl2_own_psub_float64_diff_type
 **/
void* nnl2_own_psub_float64_diff_type(void* arg) {
    subinplace_ptask* task = (subinplace_ptask*)arg;
    double* minuend = (double*)task->minuend_data;
    char* subtrahend_data = (char*)task->subtrahend_data;
    size_t start = task->start;
    size_t end = task->end;
    size_t subtrahend_step = task->subtrahend_step;
    
    size_t i = start;
    
    // AVX256 processing with type conversion and prefetching
    if(task->aligned_minuend) {
        for(; i + 3 < end; i += 4) {
            _mm_prefetch((char*)&minuend[i + 16], _MM_HINT_T0);
            
            __m256d v_minuend = _mm256_load_pd(&minuend[i]);
            __m256d v_subtrahend = _mm256_set_pd(
                nnl2_convert_to_float64(subtrahend_data + (i + 3) * subtrahend_step, task->dtype_subtrahend),
                nnl2_convert_to_float64(subtrahend_data + (i + 2) * subtrahend_step, task->dtype_subtrahend),
                nnl2_convert_to_float64(subtrahend_data + (i + 1) * subtrahend_step, task->dtype_subtrahend),
                nnl2_convert_to_float64(subtrahend_data + (i + 0) * subtrahend_step, task->dtype_subtrahend)
            );
            
            __m256d v_result = _mm256_sub_pd(v_minuend, v_subtrahend);
            _mm256_store_pd(&minuend[i], v_result);
        }
    } else {
        for(; i + 3 < end; i += 4) {
            _mm_prefetch((char*)&minuend[i + 16], _MM_HINT_T0);
            
            __m256d v_minuend = _mm256_loadu_pd(&minuend[i]);
            __m256d v_subtrahend = _mm256_set_pd(
                nnl2_convert_to_float64(subtrahend_data + (i + 3) * subtrahend_step, task->dtype_subtrahend),
                nnl2_convert_to_float64(subtrahend_data + (i + 2) * subtrahend_step, task->dtype_subtrahend),
                nnl2_convert_to_float64(subtrahend_data + (i + 1) * subtrahend_step, task->dtype_subtrahend),
                nnl2_convert_to_float64(subtrahend_data + (i + 0) * subtrahend_step, task->dtype_subtrahend)
            );
            
            __m256d v_result = _mm256_sub_pd(v_minuend, v_subtrahend);
            _mm256_storeu_pd(&minuend[i], v_result);
        }
    }
    
    // Scalar processing for remainder
    for(; i < end; i++) {
        void* subtrahend_elem = subtrahend_data + i * subtrahend_step;
        minuend[i] -= nnl2_convert_to_float64(subtrahend_elem, task->dtype_subtrahend);
    }
    
    return NULL;
}

/** @brief
 * See documentation at declaration
 * 
 ** @see nnl2_own_psub_float32_diff_type
 **/
void* nnl2_own_psub_float32_diff_type(void* arg) {
    subinplace_ptask* task = (subinplace_ptask*)arg;
    float* minuend = (float*)task->minuend_data;
    char* subtrahend_data = (char*)task->subtrahend_data;
    size_t start = task->start;
    size_t end = task->end;
    size_t subtrahend_step = task->subtrahend_step;
    
    size_t i = start;
    
    // AVX256 processing with type conversion and prefetching
    if(task->aligned_minuend) {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&minuend[i + 32], _MM_HINT_T0);
            
            __m256 v_minuend = _mm256_load_ps(&minuend[i]);
            __m256 v_subtrahend = _mm256_set_ps(
                nnl2_convert_to_float32(subtrahend_data + (i + 7) * subtrahend_step, task->dtype_subtrahend),
                nnl2_convert_to_float32(subtrahend_data + (i + 6) * subtrahend_step, task->dtype_subtrahend),
                nnl2_convert_to_float32(subtrahend_data + (i + 5) * subtrahend_step, task->dtype_subtrahend),
                nnl2_convert_to_float32(subtrahend_data + (i + 4) * subtrahend_step, task->dtype_subtrahend),
                nnl2_convert_to_float32(subtrahend_data + (i + 3) * subtrahend_step, task->dtype_subtrahend),
                nnl2_convert_to_float32(subtrahend_data + (i + 2) * subtrahend_step, task->dtype_subtrahend),
                nnl2_convert_to_float32(subtrahend_data + (i + 1) * subtrahend_step, task->dtype_subtrahend),
                nnl2_convert_to_float32(subtrahend_data + (i + 0) * subtrahend_step, task->dtype_subtrahend)
            );
            
            __m256 v_result = _mm256_sub_ps(v_minuend, v_subtrahend);
            _mm256_store_ps(&minuend[i], v_result);
        }
    } else {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&minuend[i + 32], _MM_HINT_T0);
            
            __m256 v_minuend = _mm256_loadu_ps(&minuend[i]);
            __m256 v_subtrahend = _mm256_set_ps(
                nnl2_convert_to_float32(subtrahend_data + (i + 7) * subtrahend_step, task->dtype_subtrahend),
                nnl2_convert_to_float32(subtrahend_data + (i + 6) * subtrahend_step, task->dtype_subtrahend),
                nnl2_convert_to_float32(subtrahend_data + (i + 5) * subtrahend_step, task->dtype_subtrahend),
                nnl2_convert_to_float32(subtrahend_data + (i + 4) * subtrahend_step, task->dtype_subtrahend),
                nnl2_convert_to_float32(subtrahend_data + (i + 3) * subtrahend_step, task->dtype_subtrahend),
                nnl2_convert_to_float32(subtrahend_data + (i + 2) * subtrahend_step, task->dtype_subtrahend),
                nnl2_convert_to_float32(subtrahend_data + (i + 1) * subtrahend_step, task->dtype_subtrahend),
                nnl2_convert_to_float32(subtrahend_data + (i + 0) * subtrahend_step, task->dtype_subtrahend)
            );
            
            __m256 v_result = _mm256_sub_ps(v_minuend, v_subtrahend);
            _mm256_storeu_ps(&minuend[i], v_result);
        }
    }
    
    // Scalar processing for remainder
    for(; i < end; i++) {
        void* subtrahend_elem = subtrahend_data + i * subtrahend_step;
        minuend[i] -= nnl2_convert_to_float32(subtrahend_elem, task->dtype_subtrahend);
    }
    
    return NULL;
}

/** @brief
 * See documentation at declaration
 * 
 ** @see nnl2_own_psub_int32_diff_type
 **/
void* nnl2_own_psub_int32_diff_type(void* arg) {
    subinplace_ptask* task = (subinplace_ptask*)arg;
    int32_t* minuend = (int32_t*)task->minuend_data;
    char* subtrahend_data = (char*)task->subtrahend_data;
    size_t start = task->start;
    size_t end = task->end;
    size_t subtrahend_step = task->subtrahend_step;
    
    size_t i = start;
    
    // AVX256 processing with type conversion and prefetching
    if(task->aligned_minuend) {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&minuend[i + 32], _MM_HINT_T0);
            
            __m256i v_minuend = _mm256_load_si256((__m256i*)&minuend[i]);
            __m256i v_subtrahend = _mm256_set_epi32(
                nnl2_convert_to_int32(subtrahend_data + (i + 7) * subtrahend_step, task->dtype_subtrahend),
                nnl2_convert_to_int32(subtrahend_data + (i + 6) * subtrahend_step, task->dtype_subtrahend),
                nnl2_convert_to_int32(subtrahend_data + (i + 5) * subtrahend_step, task->dtype_subtrahend),
                nnl2_convert_to_int32(subtrahend_data + (i + 4) * subtrahend_step, task->dtype_subtrahend),
                nnl2_convert_to_int32(subtrahend_data + (i + 3) * subtrahend_step, task->dtype_subtrahend),
                nnl2_convert_to_int32(subtrahend_data + (i + 2) * subtrahend_step, task->dtype_subtrahend),
                nnl2_convert_to_int32(subtrahend_data + (i + 1) * subtrahend_step, task->dtype_subtrahend),
                nnl2_convert_to_int32(subtrahend_data + (i + 0) * subtrahend_step, task->dtype_subtrahend)
            );
            
            __m256i v_result = _mm256_sub_epi32(v_minuend, v_subtrahend);
            _mm256_store_si256((__m256i*)&minuend[i], v_result);
        }
    } else {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&minuend[i + 32], _MM_HINT_T0);
            
            __m256i v_minuend = _mm256_loadu_si256((__m256i*)&minuend[i]);
            __m256i v_subtrahend = _mm256_set_epi32(
                nnl2_convert_to_int32(subtrahend_data + (i + 7) * subtrahend_step, task->dtype_subtrahend),
                nnl2_convert_to_int32(subtrahend_data + (i + 6) * subtrahend_step, task->dtype_subtrahend),
                nnl2_convert_to_int32(subtrahend_data + (i + 5) * subtrahend_step, task->dtype_subtrahend),
                nnl2_convert_to_int32(subtrahend_data + (i + 4) * subtrahend_step, task->dtype_subtrahend),
                nnl2_convert_to_int32(subtrahend_data + (i + 3) * subtrahend_step, task->dtype_subtrahend),
                nnl2_convert_to_int32(subtrahend_data + (i + 2) * subtrahend_step, task->dtype_subtrahend),
                nnl2_convert_to_int32(subtrahend_data + (i + 1) * subtrahend_step, task->dtype_subtrahend),
                nnl2_convert_to_int32(subtrahend_data + (i + 0) * subtrahend_step, task->dtype_subtrahend)
            );
            
            __m256i v_result = _mm256_sub_epi32(v_minuend, v_subtrahend);
            _mm256_storeu_si256((__m256i*)&minuend[i], v_result);
        }
    }
    
    // Scalar processing for remainder
    for(; i < end; i++) {
        void* subtrahend_elem = subtrahend_data + i * subtrahend_step;
        minuend[i] -= nnl2_convert_to_int32(subtrahend_elem, task->dtype_subtrahend);
    }
    
    return NULL;
}

#endif

/** @ingroup backend_system
 ** @brief Backend implementations for subtract in-place
 ** @details
 * Array follows the common backend registration pattern
 * Currently register backends:
 *  - nnl2_naive_subinplace: Basic reference implementation
 *  - nnl2_avx256_subinplace: AVX256 implementation 
 *
 ** @see nnl2_naive_subinplace
 ** @see nnl2_avx256_subinplace
 **/
Implementation subinplace_backends[] = {
	REGISTER_BACKEND(nnl2_naive_subinplace, nnl2_naive, NAIVE_BACKEND_NAME),
	
	#if defined(NNL2_AVX256_AVAILABLE) && TENSOR_MEM_ALIGNMENT == 32
		REGISTER_BACKEND(nnl2_avx256_subinplace, nnl2_avx256, AVX256_BACKEND_NAME),
	#endif
	
	#if defined(NNL2_AVX256_AVAILABLE) && defined(NNL2_PTHREAD_AVAILABLE)
		REGISTER_BACKEND(nnl2_own_sub_inplace, nnl2_own_2, NNL2_OWN_NAME),
	#endif
};

/**
 * @brief Function pointer for subtract in-place
 * @ingroup backend_system 
 */
subinplacefn subinplace;

/** 
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(subinplace);

/** 
 * @brief Sets the backend for subtract in-place
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void set_subinplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(subinplace_backends, subinplace, backend_name, current_backend(subinplace));
}

/** 
 * @brief Gets the name of the active backend for subtract in-place
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* get_subinplace_backend() {
	return current_backend(subinplace);
}

/** 
 * @brief Function declaration for getting all `subinplace` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(subinplace);

/**
 * @brief Function declaration for getting the number of all `subinplace` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(subinplace);

#endif /** NNL2_SUB_INPLACE_H **/
