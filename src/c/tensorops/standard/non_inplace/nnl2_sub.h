#ifndef NNL2_SUB_H
#define NNL2_SUB_H

/** @brief
 * Performs element-wise subtraction of two tensors (naive implementation)
 *
 ** @details
 * The function creates a new tensor containing the difference of the corresponding elements
 * of the two input tensors. It supports various data types with automatic
 * casting to a higher type in the hierarchy
 *
 ** @param minuend
 * Pointer to the minuend tensor (number from which to subtract)
 *
 ** @param subtrahend
 * Pointer to the subtrahend tensor (number to subtract)
 *
 ** @return 
 * Pointer to a new tensor with the subtraction result
 *
 ** @note
 * Uses volatile pointers to prevent compiler optimizations (sometimes, with -O2/-O3, naive loops cause errors without volatile. based on experience, adding volatile does not affect speed)
 *
 ** @note
 * Returns NULL in case of failure
 *
 ** @see nnl2_empty
 ** @see get_dtype_size()
 ** @see nnl2_convert_to_float64()
 ** @see nnl2_convert_to_float32()
 ** @see nnl2_convert_to_int32()
 **/
Tensor* nnl2_naive_sub(const Tensor* minuend, const Tensor* subtrahend) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Calculate the total number of elements in the tensors
	size_t len = product(minuend->shape, minuend->rank);
	
	TensorType dtype_minuend = minuend->dtype;
	TensorType dtype_subtrahend = subtrahend->dtype;
	
	// Selecting the winning type (higher in the hierarchy)
	TensorType winner_in_the_type_hierarchy = MAX(dtype_minuend, dtype_subtrahend);

	// Create an output tensor with the same shape and data type
	Tensor* difference = nnl2_empty(minuend->shape, minuend->rank, winner_in_the_type_hierarchy);
	
	if(len == 0) return difference;
	
	if(dtype_minuend == dtype_subtrahend) {
		// Handling the case if the data types match
		
		switch(dtype_minuend) {
			case FLOAT64: {
				volatile double* data_minuend = (double*)minuend->data;
				volatile double* data_subtrahend = (double*)subtrahend->data;
				volatile double* data_difference = (double*)difference->data;
			
				// Element-wise subtraction
				for(size_t i = 0; i < len; i++) {
					data_difference[i] = data_minuend[i] - data_subtrahend[i];
				}
				
				break;
			}
			
			case FLOAT32: {
				volatile float* data_minuend = (float*)minuend->data;
				volatile float* data_subtrahend = (float*)subtrahend->data;
				volatile float* data_difference = (float*)difference->data;
		
				// Element-wise subtraction
				for(size_t i = 0; i < len; i++) {
					data_difference[i] = data_minuend[i] - data_subtrahend[i];
				}
				
				break;
			}
			
			case INT32: {
				volatile int32_t* data_minuend = (int32_t*)minuend->data;
				volatile int32_t* data_subtrahend = (int32_t*)subtrahend->data;
				volatile int32_t* data_difference = (int32_t*)difference->data;
		
				// Element-wise subtraction
				for(size_t i = 0; i < len; i++) {
					data_difference[i] = data_minuend[i] - data_subtrahend[i];
				}
				
				break;
			}
			
			default: {
				NNL2_TYPE_ERROR(dtype_minuend);
				return NULL;
			}
		}
	} else {
		// Handling the case if the data types are NOT match
		switch(winner_in_the_type_hierarchy) {
			case FLOAT64: {
				volatile double* data_difference = (double*)difference->data;
				
				for(size_t i = 0; i < len; i++) {
					// Calculate the pointers to the current elements, taking into account the size of the type
					void* elem_minuend = (char*)minuend->data + i * get_dtype_size(dtype_minuend);
					void* elem_subtrahend = (char*)subtrahend->data + i * get_dtype_size(dtype_subtrahend);
					
					data_difference[i] = nnl2_convert_to_float64(elem_minuend, dtype_minuend) - nnl2_convert_to_float64(elem_subtrahend, dtype_subtrahend);
				}
				
				break;
			}
			
			case FLOAT32: {
				volatile float* data_difference = (float*)difference->data;
				
				for(size_t i = 0; i < len; i++) {
					// Calculate the pointers to the current elements, taking into account the size of the type
					void* elem_minuend = (char*)minuend->data + i * get_dtype_size(dtype_minuend);
					void* elem_subtrahend = (char*)subtrahend->data + i * get_dtype_size(dtype_subtrahend);
					
					data_difference[i] = nnl2_convert_to_float32(elem_minuend, dtype_minuend) - nnl2_convert_to_float32(elem_subtrahend, dtype_subtrahend);
				}
				
				break;
			}
        
			case INT32: {
				volatile int32_t* data_difference = (int32_t*)difference->data;
				
				for(size_t i = 0; i < len; i++) {
					// Calculate the pointers to the current elements, taking into account the size of the type
					void* elem_minuend = (char*)minuend->data + i * get_dtype_size(dtype_minuend);
					void* elem_subtrahend = (char*)subtrahend->data + i * get_dtype_size(dtype_subtrahend);
					
					data_difference[i] = nnl2_convert_to_int32(elem_minuend, dtype_minuend) - nnl2_convert_to_int32(elem_subtrahend, dtype_subtrahend);
				}
				
				break;
			}
			
			default: {
				NNL2_TYPE_ERROR(winner_in_the_type_hierarchy);
				return NULL;
			}
		}
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
	
	return difference;
}

#ifdef __AVX__

/** @brief 
 * AVX256 optimized element-wise subtraction for int32 tensors (non-in-place)
 *
 ** @details
 * Performs vectorized subtraction of two int32 tensors using AVX256 instructions
 * Handles four different alignment scenarios for optimal performance
 *
 ** @param a 
 * Pointer to destination tensor data (will store result)
 *
 ** @param b 
 * Pointer to source tensor data (will not be modified)
 *
 ** @param len 
 * Number of elements to process
 *
 ** @param aligned_a 
 * Whether tensor a is aligned to 32-byte boundary
 *
 ** @param aligned_b 
 * Whether tensor b is aligned to 32-byte boundary
 *
 ** @see nnl2_avx256_sub
 **/
static inline void nnl2_avx_sub_non_in_place_int32_same_type(int32_t* a, const int32_t* b, size_t len, bool aligned_a, bool aligned_b);

/** @brief 
 * AVX256 optimized element-wise subtraction for float32 tensors (non-in-place)
 *
 * Вocumentation is identical to the documentation of the 
 * nnl2_avx_sub_non_in_place_int32_same_type declaration
 *
 ** @see nnl2_avx256_sub
 ** @see nnl2_avx_sub_non_in_place_int32_same_type
 **/
static inline void nnl2_avx_sub_non_in_place_float32_same_type(float* a, const float* b, size_t len, bool aligned_a, bool aligned_b);

/** @brief 
 * AVX256 optimized element-wise subtraction for float64 tensors (non-in-place)
 *
 * Вocumentation is identical to the documentation of the 
 * nnl2_avx_sub_non_in_place_int32_same_type declaration
 *
 ** @see nnl2_avx256_sub
 ** @see nnl2_avx_sub_non_in_place_float32_same_type
 ** @see nnl2_avx_sub_non_in_place_int32_same_type
 **/
static inline void nnl2_avx_sub_non_in_place_float64_same_type(double* a, const double* b, size_t len, bool aligned_a, bool aligned_b);

/** @brief 
 * Performs element-wise subtraction of two tensors using AVX256 instructions
 * 
 ** @details
 * The function creates a new tensor containing the difference of corresponding elements
 * from two input tensors. It supports various data types with automatic type
 * promotion to the highest type in the hierarchy. For same data types, it uses
 * optimized AVX256 vector instructions. For mixed types, it falls back to scalar
 * operations with type conversion
 * 
 ** @param minuend 
 * Pointer to the minuend tensor (number from which to subtract)
 *
 ** @param subtrahend 
 * Pointer to the subtrahend tensor (number to subtract)
 * 
 ** @return 
 * Pointer to a new tensor containing the element-wise difference
 * 
 ** @note 
 * For mixed types, scalar operations are used due to AVX limitations
 * in handling type conversions within vector instructions
 *
 ** @note  
 * Includes proper handling of empty tensors (len == 0)
 * 
 ** @see nnl2_empty()
 ** @see get_dtype_size()
 ** @see nnl2_convert_to_float64()
 ** @see nnl2_convert_to_float32()  
 ** @see nnl2_convert_to_int32()
 **/
Tensor* nnl2_avx256_sub(const Tensor* minuend, const Tensor* subtrahend) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
    size_t len = product(minuend->shape, minuend->rank);
    
    TensorType dtype_minuend = minuend->dtype;
    TensorType dtype_subtrahend = subtrahend->dtype;
	
	// Selecting the winning type (higher in the hierarchy)
	TensorType winner_in_the_type_hierarchy = MAX(dtype_minuend, dtype_subtrahend);
    
    Tensor* difference = nnl2_empty(minuend->shape, minuend->rank, winner_in_the_type_hierarchy);
	
	if(len == 0) return difference; 
    
	if(dtype_minuend == dtype_subtrahend) {
	    // Check alignment for both tensors
		bool aligned_minuend = NNL2_IS_ALIGNED(minuend->data, NNL2_TENSOR_ALIGNMENT_32);
		bool aligned_subtrahend = NNL2_IS_ALIGNED(subtrahend->data, NNL2_TENSOR_ALIGNMENT_32);

		// Handling the case when the data types are the same
		switch(dtype_minuend) {
			case FLOAT64: {
                double* data_minuend = (double*)minuend->data;
                double* data_subtrahend = (double*)subtrahend->data;
                double* data_difference = (double*)difference->data;
                
                // Copy data from minuend to result first
                memcpy(data_difference, data_minuend, len * sizeof(double));
                
                // Use optimized subtraction
                nnl2_avx_sub_non_in_place_float64_same_type(data_difference, data_subtrahend, len, aligned_minuend, aligned_subtrahend);
                break;
            }
            
            case FLOAT32: {
                float* data_minuend = (float*)minuend->data;
                float* data_subtrahend = (float*)subtrahend->data;
                float* data_difference = (float*)difference->data;
                
                // Copy data from minuend to result first
                memcpy(data_difference, data_minuend, len * sizeof(float));
                
                // Use optimized subtraction
                nnl2_avx_sub_non_in_place_float32_same_type(data_difference, data_subtrahend, len, aligned_minuend, aligned_subtrahend);
                break;
            }
            
            case INT32: {
                int32_t* data_minuend = (int32_t*)minuend->data;
                int32_t* data_subtrahend = (int32_t*)subtrahend->data;
                int32_t* data_difference = (int32_t*)difference->data;
                
                // Copy data from minuend to result first
                memcpy(data_difference, data_minuend, len * sizeof(int32_t));
                
                // Use optimized subtraction
                nnl2_avx_sub_non_in_place_int32_same_type(data_difference, data_subtrahend, len, aligned_minuend, aligned_subtrahend);
                break;
            }
			
			default: {
				NNL2_TYPE_ERROR(dtype_minuend);
				return NULL;
			}
		} 
	} else {
		// Handling the case when the data types are NOT the same
		// For mixed types, using scalar operations since AVX doesn't easily handle
        // type conversions within the same instruction
		
		switch(winner_in_the_type_hierarchy) {
			case FLOAT64: {
                double* data_difference = (double*)difference->data;
                
				// Element-wise subtraction
                for(size_t i = 0; i < len; i++) {
                    void* elem_minuend = (char*)minuend->data + i * get_dtype_size(dtype_minuend);
                    void* elem_subtrahend = (char*)subtrahend->data + i * get_dtype_size(dtype_subtrahend);
                    
                    data_difference[i] = nnl2_convert_to_float64(elem_minuend, dtype_minuend) - nnl2_convert_to_float64(elem_subtrahend, dtype_subtrahend);
                }
				
                break;
            }
            
            case FLOAT32: {
                float* data_difference = (float*)difference->data;
				
				// Element-wise subtraction
                for(size_t i = 0; i < len; i++) {
                    void* elem_minuend = (char*)minuend->data + i * get_dtype_size(dtype_minuend);
                    void* elem_subtrahend = (char*)subtrahend->data + i * get_dtype_size(dtype_subtrahend);
                    
                    data_difference[i] = nnl2_convert_to_float32(elem_minuend, dtype_minuend) - nnl2_convert_to_float32(elem_subtrahend, dtype_subtrahend);
                }
                
                break;
            }
            
            case INT32: {
                int32_t* data_difference = (int32_t*)difference->data;

				// Element-wise subtraction
                for(size_t i = 0; i < len; i++) {
                    void* elem_minuend = (char*)minuend->data + i * get_dtype_size(dtype_minuend);
                    void* elem_subtrahend = (char*)subtrahend->data + i * get_dtype_size(dtype_subtrahend);
                    
                    data_difference[i] = nnl2_convert_to_int32(elem_minuend, dtype_minuend) - nnl2_convert_to_int32(elem_subtrahend, dtype_subtrahend);
                }
                
                break;
            }
			
			default: {
				NNL2_TYPE_ERROR(winner_in_the_type_hierarchy);
				return NULL;
			}
		}
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
    
    return difference;
}

/** @brief 
 * AVX-optimized element-wise subtraction for int32 tensors (non-in-place)
 *
 ** @details
 * See docs at declaration
 *
 ** @see nnl2_avx_sub_non_in_place_int32_same_type (declaration)
 **/
static inline void nnl2_avx_sub_non_in_place_int32_same_type(int32_t* a, const int32_t* b, size_t len, bool aligned_a, bool aligned_b) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
    
    size_t i = 0;
    
    // Case 1: Both tensors are aligned 
    if(aligned_a && aligned_b) {
        for(; i + 7 < len; i += 8) {
            __m256i v_a = _mm256_load_si256((__m256i*)&a[i]);       // Fast loading of aligned data
            __m256i v_b = _mm256_load_si256((__m256i*)&b[i]);       // Fast loading of aligned data
            __m256i v_result = _mm256_sub_epi32(v_a, v_b);          // Vector subtraction
            _mm256_store_si256((__m256i*)&a[i], v_result);          // Fast saving to aligned memory
        }
    } 
    
    // Case 2: Only tensor a is aligned
    else if(aligned_a) {
        for(; i + 7 < len; i += 8) {
            __m256i v_a = _mm256_load_si256((__m256i*)&a[i]);       // Fast loading of aligned data
            __m256i v_b = _mm256_loadu_si256((__m256i*)&b[i]);      // Slow loading of unaligned data
            __m256i v_result = _mm256_sub_epi32(v_a, v_b);          // Vector subtraction
            _mm256_store_si256((__m256i*)&a[i], v_result);          // Fast saving to aligned memory
        }
    } 
    
    // Case 3: Only tensor b is aligned
    else if(aligned_b) {
        for(; i + 7 < len; i += 8) {
            __m256i v_a = _mm256_loadu_si256((__m256i*)&a[i]);      // Slow loading of unaligned data
            __m256i v_b = _mm256_load_si256((__m256i*)&b[i]);       // Fast loading of aligned data
            __m256i v_result = _mm256_sub_epi32(v_a, v_b);          // Vector subtraction
            _mm256_storeu_si256((__m256i*)&a[i], v_result);         // Slow saving to unaligned memory
        }
    } 
    
    // Case 4: Both tensors are not aligned
    else {
        for(; i + 7 < len; i += 8) {
            __m256i v_a = _mm256_loadu_si256((__m256i*)&a[i]);      // Slow loading of unaligned data
            __m256i v_b = _mm256_loadu_si256((__m256i*)&b[i]);      // Slow loading of unaligned data
            __m256i v_result = _mm256_sub_epi32(v_a, v_b);          // Vector subtraction
            _mm256_storeu_si256((__m256i*)&a[i], v_result);         // Slow saving to unaligned memory
        }
    }
    
    // Processing the remainder
    for(; i < len; i++) a[i] -= b[i];
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
}

/** @brief 
 * AVX-optimized element-wise subtraction for float32 tensors (non-in-place)
 *
 ** @details
 * See docs at declaration
 *
 ** @see nnl2_avx_sub_non_in_place_float32_same_type (declaration)
 **/
static inline void nnl2_avx_sub_non_in_place_float32_same_type(float* a, const float* b, size_t len, bool aligned_a, bool aligned_b) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
    
    size_t i = 0;
    
    // Case 1: Both tensors are aligned 
    if(aligned_a && aligned_b) {
        for(; i + 7 < len; i += 8) {
            __m256 v_a = _mm256_load_ps(&a[i]);        // Fast loading of aligned data
            __m256 v_b = _mm256_load_ps(&b[i]);        // Fast loading of aligned data
            __m256 v_result = _mm256_sub_ps(v_a, v_b); // Vector subtraction
            _mm256_store_ps(&a[i], v_result);          // Fast saving to aligned memory
        }
    } 
    
    // Case 2: Only tensor a is aligned
    else if(aligned_a) {
        for(; i + 7 < len; i += 8) {
            __m256 v_a = _mm256_load_ps(&a[i]);        // Fast loading of aligned data
            __m256 v_b = _mm256_loadu_ps(&b[i]);       // Slow loading of unaligned data
            __m256 v_result = _mm256_sub_ps(v_a, v_b); // Vector subtraction
            _mm256_store_ps(&a[i], v_result);          // Fast saving to aligned memory
        }
    } 
    
    // Case 3: Only tensor b is aligned
    else if(aligned_b) {
        for(; i + 7 < len; i += 8) {
            __m256 v_a = _mm256_loadu_ps(&a[i]);       // Slow loading of unaligned data
            __m256 v_b = _mm256_load_ps(&b[i]);        // Fast loading of aligned data
            __m256 v_result = _mm256_sub_ps(v_a, v_b); // Vector subtraction
            _mm256_storeu_ps(&a[i], v_result);         // Slow saving to unaligned memory
        }
    } 
    
    // Case 4: Both tensors are not aligned
    else {
        for(; i + 7 < len; i += 8) {
            __m256 v_a = _mm256_loadu_ps(&a[i]);       // Slow loading of unaligned data
            __m256 v_b = _mm256_loadu_ps(&b[i]);       // Slow loading of unaligned data
            __m256 v_result = _mm256_sub_ps(v_a, v_b); // Vector subtraction
            _mm256_storeu_ps(&a[i], v_result);         // Slow saving to unaligned memory
        }
    }
    
    // Processing the remainder
    for(; i < len; i++) a[i] -= b[i];
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
}

/** @brief 
 * AVX-optimized element-wise subtraction for float64 tensors (non-in-place)
 *
 ** @details
 * See docs at declaration
 *
 ** @see nnl2_avx_sub_non_in_place_float64_same_type (declaration)
 **/
static inline void nnl2_avx_sub_non_in_place_float64_same_type(double* a, const double* b, size_t len, bool aligned_a, bool aligned_b) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
    
    size_t i = 0;
    
    // Case 1: Both tensors are aligned 
    if(aligned_a && aligned_b) {
        for(; i + 3 < len; i += 4) {
            __m256d v_a = _mm256_load_pd(&a[i]);        // Fast loading of aligned data
            __m256d v_b = _mm256_load_pd(&b[i]);        // Fast loading of aligned data
            __m256d v_result = _mm256_sub_pd(v_a, v_b); // Vector subtraction
            _mm256_store_pd(&a[i], v_result);           // Fast saving to aligned memory
        }
    } 
    
    // Case 2: Only tensor a is aligned
    else if(aligned_a) {
        for(; i + 3 < len; i += 4) {
            __m256d v_a = _mm256_load_pd(&a[i]);        // Fast loading of aligned data
            __m256d v_b = _mm256_loadu_pd(&b[i]);       // Slow loading of unaligned data
            __m256d v_result = _mm256_sub_pd(v_a, v_b); // Vector subtraction
            _mm256_store_pd(&a[i], v_result);           // Fast saving to aligned memory
        }
    } 
    
    // Case 3: Only tensor b is aligned
    else if(aligned_b) {
        for(; i + 3 < len; i += 4) {
            __m256d v_a = _mm256_loadu_pd(&a[i]);       // Slow loading of unaligned data
            __m256d v_b = _mm256_load_pd(&b[i]);        // Fast loading of aligned data
            __m256d v_result = _mm256_sub_pd(v_a, v_b); // Vector subtraction
            _mm256_storeu_pd(&a[i], v_result);          // Slow saving to unaligned memory
        }
    } 
    
    // Case 4: Both tensors are not aligned
    else {
        for(; i + 3 < len; i += 4) {
            __m256d v_a = _mm256_loadu_pd(&a[i]);       // Slow loading of unaligned data
            __m256d v_b = _mm256_loadu_pd(&b[i]);       // Slow loading of unaligned data
            __m256d v_result = _mm256_sub_pd(v_a, v_b); // Vector subtraction
            _mm256_storeu_pd(&a[i], v_result);          // Slow saving to unaligned memory
        }
    }
    
    // Processing the remainder
    for(; i < len; i++) a[i] -= b[i];
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
}

#endif



#ifdef NNL2_PTHREAD_AVAILABLE

/** @brief
 * Threshold for enabling parallel execution of the
 * subtraction operation
 */
#define NNL2_SUB_PARALLEL_THREASHOLD 1000000

/** @brief 
 * Worker function for parallel subtraction for same data types
 * 
 * @param arg 
 * Pointer to sub_ptask structure containing task parameters
 *
 * @return NULL (for pthread api)
 */
void* nnl2_own_psub_same_type(void* arg);

/** @brief 
 * Worker function for parallel subtraction for mixed data types
 * 
 * @param arg 
 * Pointer to sub_ptask structure containing task parameters
 *
 * @return NULL (for pthread api)
 */
void* nnl2_own_psub_mixed_types(void* arg);

#ifdef NNL2_AVX256_AVAILABLE

/** @brief 
 * SIMD-optimized worker function for parallel subtraction for same float64 data types
 * 
 * @param arg 
 * Pointer to sub_ptask structure containing task parameters
 *
 * @return NULL (for pthread api)
 */
void* nnl2_own_psub_simd_float64(void* arg);

/** @brief 
 * SIMD-optimized worker function for parallel subtraction for same float32 data types
 * 
 * @param arg 
 * Pointer to sub_ptask structure containing task parameters
 *
 * @return NULL (for pthread api)
 */
void* nnl2_own_psub_simd_float32(void* arg);

/** @brief 
 * SIMD-optimized worker function for parallel subtraction for same int32 data types
 * 
 * @param arg 
 * Pointer to sub_ptask structure containing task parameters
 *
 * @return NULL (for pthread api)
 */
void* nnl2_own_psub_simd_int32(void* arg);

#endif

/** @brief
 * Parallel implementation of tensor subtraction using pthreads
 *
 ** @param minuend
 * Pointer to the minuend tensor
 *
 ** @param subtrahend
 * Pointer to the subtrahend tensor
 *
 ** @return 
 * Pointer to a new tensor with the subtraction result
 */
Tensor* nnl2_own_sub(const Tensor* minuend, const Tensor* subtrahend) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Calculate the total number of elements in the tensors
    size_t len = product(minuend->shape, minuend->rank);
    
    TensorType dtype_minuend = minuend->dtype;
    TensorType dtype_subtrahend = subtrahend->dtype;
    
    // Selecting the winning type (higher in the hierarchy)
    TensorType winner_in_the_type_hierarchy = MAX(dtype_minuend, dtype_subtrahend);

    // Create an output tensor with the same shape and data type
    Tensor* difference = nnl2_empty(minuend->shape, minuend->rank, winner_in_the_type_hierarchy);
    
    if(len == 0) return difference;
    
    // Use naive implementation for small tensors
    if(len < NNL2_SUB_PARALLEL_THREASHOLD) {
        difference = nnl2_naive_sub(minuend, subtrahend);
        if(difference == NULL) {
			NNL2_ERROR("Failed to subtract");
		}
		
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
		
        return difference;
    }
    
    // Allocate arrays for thread handles and task descriptors
    pthread_t threads[NNL2_NUM_THREADS];
    sub_ptask tasks[NNL2_NUM_THREADS];
    
    // Calculate base chunk size and remainder for balanced distribution
    size_t chunk = len / NNL2_NUM_THREADS;
    size_t remainder = len % NNL2_NUM_THREADS;
    
    bool use_simd = false;
	
    #ifdef NNL2_AVX256_AVAILABLE
    if(dtype_minuend == dtype_subtrahend) {
        bool aligned_minuend = NNL2_IS_ALIGNED(minuend->data, NNL2_TENSOR_ALIGNMENT_32);
        bool aligned_subtrahend = NNL2_IS_ALIGNED(subtrahend->data, NNL2_TENSOR_ALIGNMENT_32);
        bool aligned_result = NNL2_IS_ALIGNED(difference->data, NNL2_TENSOR_ALIGNMENT_32);
        use_simd = aligned_minuend && aligned_subtrahend && aligned_result;
    }
    #endif
    
    // Distribute work among threads with load balancing
    size_t current_start = 0;
    for (size_t i = 0; i < NNL2_NUM_THREADS; i++) {
        size_t current_chunk = chunk + (i < remainder ? 1 : 0);
        
        // Configure task for this thread
        tasks[i].minuend_data = minuend->data;
        tasks[i].subtrahend_data = subtrahend->data;
        tasks[i].result_data = difference->data;
        tasks[i].start = current_start;
        tasks[i].end = current_start + current_chunk;
        tasks[i].dtype_minuend = dtype_minuend;
        tasks[i].dtype_subtrahend = dtype_subtrahend;
        tasks[i].result_dtype = winner_in_the_type_hierarchy;
        
        // Create thread to process the assigned chunk
        int status;
        
        #ifdef NNL2_AVX256_AVAILABLE
			if(use_simd && dtype_minuend == dtype_subtrahend) {
				switch(dtype_minuend) {
					case FLOAT64: status = pthread_create(&threads[i], NULL, nnl2_own_psub_simd_float64, &tasks[i]); break;
					case FLOAT32: status = pthread_create(&threads[i], NULL, nnl2_own_psub_simd_float32, &tasks[i]); break;
					case INT32:   status = pthread_create(&threads[i], NULL, nnl2_own_psub_simd_int32, &tasks[i]);   break;
					
					default: {
						status = pthread_create(&threads[i], NULL, nnl2_own_psub_same_type, &tasks[i]);
						break;
					}
				}
			} else 
        #endif
        {
            if(dtype_minuend == dtype_subtrahend) {
                status = pthread_create(&threads[i], NULL, nnl2_own_psub_same_type, &tasks[i]);
            } else {
                status = pthread_create(&threads[i], NULL, nnl2_own_psub_mixed_types, &tasks[i]);
            }
        }
        
        if(status != 0) {
            NNL2_THREAD_CREATE_ERROR(status, "nnl2_own_sub");
            // Clean up already created threads
            for(size_t j = 0; j < i; j++) {
                pthread_join(threads[j], NULL);
            }
			
            nnl2_free_tensor(difference);
            return NULL;
        }
        
        current_start += current_chunk;
    }
    
    // Wait for all threads to complete their work
    for (size_t i = 0; i < NNL2_NUM_THREADS; i++) {
        int join_status = pthread_join(threads[i], NULL);
        if(join_status != 0) {
            NNL2_THREAD_JOIN_ERROR(join_status, "nnl2_own_sub");
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return difference;
}

/** @brief
 * See docs at declaration
 *
 ** @see nnl2_own_psub_same_type
 **/
void* nnl2_own_psub_same_type(void* arg) {
    sub_ptask* task = (sub_ptask*)arg;
    
    switch(task->dtype_minuend) {
        case FLOAT64: {
            volatile double* data_minuend = (double*)task->minuend_data;
            volatile double* data_subtrahend = (double*)task->subtrahend_data;
            volatile double* data_result = (double*)task->result_data;
            
            for(size_t i = task->start; i < task->end; i++) {
                data_result[i] = data_minuend[i] - data_subtrahend[i];
            }
			
            break;
        }
        
        case FLOAT32: {
            volatile float* data_minuend = (float*)task->minuend_data;
            volatile float* data_subtrahend = (float*)task->subtrahend_data;
            volatile float* data_result = (float*)task->result_data;
            
            for(size_t i = task->start; i < task->end; i++) {
                data_result[i] = data_minuend[i] - data_subtrahend[i];
            }
			
            break;
        }
        
        case INT32: {
            volatile int32_t* data_minuend = (int32_t*)task->minuend_data;
            volatile int32_t* data_subtrahend = (int32_t*)task->subtrahend_data;
            volatile int32_t* data_result = (int32_t*)task->result_data;
            
            for(size_t i = task->start; i < task->end; i++) {
                data_result[i] = data_minuend[i] - data_subtrahend[i];
            }
			
            break;
        }
        
        default: {
            // Type error should be handled in main function
            break;
        }
    }
    
    return NULL;
}

#ifdef NNL2_AVX256_AVAILABLE

/** @brief
 * SIMD-optimized worker function for float64 subtraction
 *
 ** @see nnl2_own_psub_simd_float64
 **/
void* nnl2_own_psub_simd_float64(void* arg) {
    sub_ptask* task = (sub_ptask*)arg;
    
    double* data_minuend = (double*)task->minuend_data;
    double* data_subtrahend = (double*)task->subtrahend_data;
    double* data_result = (double*)task->result_data;
    
    size_t i = task->start;
    
    // Process 4 elements at a time using AVX
    for(; i + 3 < task->end; i += 4) {
        __m256d v_minuend = _mm256_load_pd(&data_minuend[i]);        // Load 4 doubles
        __m256d v_subtrahend = _mm256_load_pd(&data_subtrahend[i]);  // Load 4 doubles
        __m256d v_result = _mm256_sub_pd(v_minuend, v_subtrahend);   // Vector subtraction
        _mm256_store_pd(&data_result[i], v_result);                  // Store result
    }
    
    // Process remainder elements
    for(; i < task->end; i++) {
        data_result[i] = data_minuend[i] - data_subtrahend[i];
    }
    
    return NULL;
}

/** @brief
 * SIMD-optimized worker function for float32 subtraction
 *
 ** @see nnl2_own_psub_simd_float32
 **/
void* nnl2_own_psub_simd_float32(void* arg) {
    sub_ptask* task = (sub_ptask*)arg;
    
    float* data_minuend = (float*)task->minuend_data;
    float* data_subtrahend = (float*)task->subtrahend_data;
    float* data_result = (float*)task->result_data;
    
    size_t i = task->start;
    
    // Process 8 elements at a time using AVX
    for(; i + 7 < task->end; i += 8) {
        __m256 v_minuend = _mm256_load_ps(&data_minuend[i]);        // Load 8 floats
        __m256 v_subtrahend = _mm256_load_ps(&data_subtrahend[i]);  // Load 8 floats
        __m256 v_result = _mm256_sub_ps(v_minuend, v_subtrahend);   // Vector subtraction
        _mm256_store_ps(&data_result[i], v_result);                 // Store result
    }
    
    // Process remainder elements
    for(; i < task->end; i++) {
        data_result[i] = data_minuend[i] - data_subtrahend[i];
    }
    
    return NULL;
}

/** @brief
 * SIMD-optimized worker function for int32 subtraction
 *
 ** @see nnl2_own_psub_simd_int32
 **/
void* nnl2_own_psub_simd_int32(void* arg) {
    sub_ptask* task = (sub_ptask*)arg;
    
    int32_t* data_minuend = (int32_t*)task->minuend_data;
    int32_t* data_subtrahend = (int32_t*)task->subtrahend_data;
    int32_t* data_result = (int32_t*)task->result_data;
    
    size_t i = task->start;
    
    // Process 8 elements at a time using AVX
    for(; i + 7 < task->end; i += 8) {
        __m256i v_minuend = _mm256_load_si256((__m256i*)&data_minuend[i]);      // Load 8 int32
        __m256i v_subtrahend = _mm256_load_si256((__m256i*)&data_subtrahend[i]);// Load 8 int32
        __m256i v_result = _mm256_sub_epi32(v_minuend, v_subtrahend);           // Vector subtraction
        _mm256_store_si256((__m256i*)&data_result[i], v_result);                // Store result
    }
    
    // Process remainder elements
    for(; i < task->end; i++) {
        data_result[i] = data_minuend[i] - data_subtrahend[i];
    }
    
    return NULL;
}

#endif

/** @brief
 * See docs at declaration
 *
 ** @see nnl2_own_psub_mixed_types
 **/
void* nnl2_own_psub_mixed_types(void* arg) {
    sub_ptask* task = (sub_ptask*)arg;
    
    switch(task->result_dtype) {
        case FLOAT64: {
            volatile double* data_result = (double*)task->result_data;
            
            for(size_t i = task->start; i < task->end; i++) {
                void* elem_minuend = (char*)task->minuend_data + i * get_dtype_size(task->dtype_minuend);
                void* elem_subtrahend = (char*)task->subtrahend_data + i * get_dtype_size(task->dtype_subtrahend);
                
                data_result[i] = nnl2_convert_to_float64(elem_minuend, task->dtype_minuend) - nnl2_convert_to_float64(elem_subtrahend, task->dtype_subtrahend);
            }
			
            break;
        }
        
        case FLOAT32: {
            volatile float* data_result = (float*)task->result_data;
            
            for(size_t i = task->start; i < task->end; i++) {
                void* elem_minuend = (char*)task->minuend_data + i * get_dtype_size(task->dtype_minuend);
                void* elem_subtrahend = (char*)task->subtrahend_data + i * get_dtype_size(task->dtype_subtrahend);
                
                data_result[i] = nnl2_convert_to_float32(elem_minuend, task->dtype_minuend) - nnl2_convert_to_float32(elem_subtrahend, task->dtype_subtrahend);
            }
			
            break;
        }
        
        case INT32: {
            volatile int32_t* data_result = (int32_t*)task->result_data;
            
            for(size_t i = task->start; i < task->end; i++) {
                void* elem_minuend = (char*)task->minuend_data + i * get_dtype_size(task->dtype_minuend);
                void* elem_subtrahend = (char*)task->subtrahend_data + i * get_dtype_size(task->dtype_subtrahend);
                
                data_result[i] = nnl2_convert_to_int32(elem_minuend, task->dtype_minuend) - nnl2_convert_to_int32(elem_subtrahend, task->dtype_subtrahend);
            }
			
            break;
        }
        
        default: {
            // Type error should be handled in main function
            break;
        }
    }
    
    return NULL;
}

#endif

/** 
 * @ingroup backend_system
 * @brief Backend implementations for subtraction operation
 * @details
 * Array follows the common backend registration pattern.
 * Currently registered backends:
 *  - nnl2_naive_sub: Basic reference implementation
 *  - nnl2_avx256_sub: AVX256 implementation (if available)
 * 
 * @see nnl2_naive_sub
 * @see nnl2_avx256_sub
 */
Implementation sub_backends[] = {
	REGISTER_BACKEND(nnl2_naive_sub, nnl2_naive, NAIVE_BACKEND_NAME),
	
	#ifdef NNL2_AVX256_AVAILABLE
		REGISTER_BACKEND(nnl2_avx256_sub, nnl2_avx256, AVX256_BACKEND_NAME),
	#endif
	
	#ifdef NNL2_PTHREAD_AVAILABLE
		REGISTER_BACKEND(nnl2_own_sub, nnl2_own, NNL2_OWN_NAME),
	#endif
};

/**
 * @brief Function pointer for subtraction operation
 * @ingroup backend_system 
 */
subfn sub;

/** 
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(sub);

/** 
 * @brief Sets the backend for subtraction operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void set_sub_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(sub_backends, sub, backend_name, CURRENT_BACKEND(sub));
}

/** 
 * @brief Gets the name of the active backend for subtraction operation
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* get_sub_backend() {
	return current_backend(sub);
}

/** 
 * @brief Function declaration for getting all `sub` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(sub);

/**
 * @brief Function declaration for getting the number of all `sub` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(sub);

#endif /** NNL2_SUB_H **/
