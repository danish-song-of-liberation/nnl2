#ifndef NNL2_ADD_H
#define NNL2_ADD_H

#define nnl2_add(addend, sumend) add(addend, sumend) // I'm too lazy to rename the function in all places

/** @brief
 * Threshold for enabling parallel execution of the
 * addition operation
 */
#define NNL2_ADD_PARALLEL_THREASHOLD 1000000

/** @brief
 * Performs element-wise addition of two tensors (naive implementation)
 *
 ** @details
 * The function creates a new tensor containing the sum of the corresponding elements
 * of the two input tensors. It supports various data types with automatic
 * casting to a higher type in the hierarchy
 *
 ** @param summand
 * Pointer to the summand tensor
 *
 ** @param addend
 * Pointer to the addend tensor
 *
 ** @return 
 * Pointer to a new tensor with the addition result
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
nnl2_tensor* nnl2_naive_add(nnl2_tensor* summand, nnl2_tensor* addend) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Calculate the total number of elements in the tensors
	size_t len = product(summand->shape, summand->rank);
	
	nnl2_tensor_type dtype_summand = summand->dtype;
	nnl2_tensor_type dtype_addend = addend->dtype;
	
	// Selecting the winning type (higher in the hierarchy)
	nnl2_tensor_type winner_in_the_type_hierarchy = MAX(dtype_summand, dtype_addend);

	// Create an output tensor with the same shape and data type
	nnl2_tensor* amount = nnl2_empty(summand->shape, summand->rank, winner_in_the_type_hierarchy);
	
	if(len == 0) return amount;
	
	if(dtype_summand == dtype_addend) {
		// Handling the case if the data types match
		
		switch(dtype_summand) {
			case FLOAT64: {
				volatile double* data_summand = (double*)summand->data;
				volatile double* data_addend = (double*)addend->data;
				volatile double* data_amount = (double*)amount->data;
			
				// Element-wise addition
				for(size_t i = 0; i < len; i++) {
					data_amount[i] = data_summand[i] + data_addend[i];
				}
				
				break;
			}
			
			case FLOAT32: {
				volatile float* data_summand = (float*)summand->data;
				volatile float* data_addend = (float*)addend->data;
				volatile float* data_amount = (float*)amount->data;
		
				// Element-wise addition
				for(size_t i = 0; i < len; i++) {
					data_amount[i] = data_summand[i] + data_addend[i];
				}
				
				break;
			}
			
			case INT32: {
				volatile int32_t* data_summand = (int32_t*)summand->data;
				volatile int32_t* data_addend = (int32_t*)addend->data;
				volatile int32_t* data_amount = (int32_t*)amount->data;
		
				// Element-wise addition
				for(size_t i = 0; i < len; i++) {
					data_amount[i] = data_summand[i] + data_addend[i];
				}
				
				break;
			}
			
			default: {
				NNL2_TYPE_ERROR(dtype_summand);
				return NULL;
			}
		}
	} else {
		// Handling the case if the data types are NOT match
		switch(winner_in_the_type_hierarchy) {
			case FLOAT64: {
				volatile double* data_amount = (double*)amount->data;
				
				for(size_t i = 0; i < len; i++) {
					// Calculate the pointers to the current elements, taking into account the size of the type
					void* elem_summand = (char*)summand->data + i * get_dtype_size(dtype_summand);
					void* elem_addend = (char*)addend->data + i * get_dtype_size(dtype_addend);
					
					data_amount[i] = nnl2_convert_to_float64(elem_summand, dtype_summand) + nnl2_convert_to_float64(elem_addend, dtype_addend);
				}
				
				break;
			}
			
			case FLOAT32: {
				volatile float* data_amount = (float*)amount->data;
				
				for(size_t i = 0; i < len; i++) {
					// Calculate the pointers to the current elements, taking into account the size of the type
					void* elem_summand = (char*)summand->data + i * get_dtype_size(dtype_summand);
					void* elem_addend = (char*)addend->data + i * get_dtype_size(dtype_addend);
					
					data_amount[i] = nnl2_convert_to_float32(elem_summand, dtype_summand) + nnl2_convert_to_float32(elem_addend, dtype_addend);
				}
				
				break;
			}
        
			case INT32: {
				volatile int32_t* data_amount = (int32_t*)amount->data;
				
				for(size_t i = 0; i < len; i++) {
					// Calculate the pointers to the current elements, taking into account the size of the type
					void* elem_summand = (char*)summand->data + i * get_dtype_size(dtype_summand);
					void* elem_addend = (char*)addend->data + i * get_dtype_size(dtype_addend);
					
					data_amount[i] = nnl2_convert_to_int32(elem_summand, dtype_summand) + nnl2_convert_to_int32(elem_addend, dtype_addend);
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
	
	return amount;
}

#ifdef NNL2_AVX256_AVAILABLE

/** @brief 
 * AVX256 optimized element-wise addition for int32 tensors (non-in-place)
 *
 ** @details
 * Performs vectorized addition of two int32 tensors using AVX256 instructions
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
 ** @see nnl2_avx256_add
 **/
static inline void nnl2_avx_add_non_in_place_int32_same_type(int32_t* a, const int32_t* b, size_t len, bool aligned_a, bool aligned_b);

/** @brief 
 * AVX256 optimized element-wise addition for float32 tensors (non-in-place)
 *
 * Вocumentation is identical to the documentation of the 
 * nnl2_avx_add_non_in_place_int32_same_type declaration
 *
 ** @see nnl2_avx256_add
 ** @see nnl2_avx_add_non_in_place_int32_same_type
 **/
static inline void nnl2_avx_add_non_in_place_float32_same_type(float* a, const float* b, size_t len, bool aligned_a, bool aligned_b);

/** @brief 
 * AVX256 optimized element-wise addition for float64 tensors (non-in-place)
 *
 * Вocumentation is identical to the documentation of the 
 * nnl2_avx_add_non_in_place_int32_same_type declaration
 *
 ** @see nnl2_avx256_add
 ** @see nnl2_avx_add_non_in_place_float32_same_type
 ** @see nnl2_avx_add_non_in_place_int32_same_type
 **/
static inline void nnl2_avx_add_non_in_place_float64_same_type(double* a, const double* b, size_t len, bool aligned_a, bool aligned_b);

/** @brief 
 * Performs element-wise addition of two tensors using AVX256 instructions
 * 
 ** @details
 * The function creates a new tensor containing the sum of corresponding elements
 * from two input tensors. It supports various data types with automatic type
 * promotion to the highest type in the hierarchy. For same data types, it uses
 * optimized AVX256 vector instructions. For mixed types, it falls back to scalar
 * operations with type conversion
 * 
 ** @param summand 
 * Pointer to the summand tensor
 *
 ** @param addend 
 * Pointer to the addend tensor
 * 
 ** @return 
 * Pointer to a new tensor containing the element-wise sum
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
nnl2_tensor* nnl2_avx256_add(const nnl2_tensor* summand, const nnl2_tensor* addend) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
    size_t len = product(summand->shape, summand->rank);
    
    nnl2_tensor_type dtype_summand = summand->dtype;
    nnl2_tensor_type dtype_addend = addend->dtype;
	
	// Selecting the winning type (higher in the hierarchy)
	nnl2_tensor_type winner_in_the_type_hierarchy = MAX(dtype_summand, dtype_addend);
    
    nnl2_tensor* sum = nnl2_empty(summand->shape, summand->rank, winner_in_the_type_hierarchy);
	
	if(len == 0) return sum; 
    
	if(dtype_summand == dtype_addend) {
	    // Check alignment for both tensors
		bool aligned_summand = NNL2_IS_ALIGNED(summand->data, NNL2_TENSOR_ALIGNMENT_32);
		bool aligned_addend = NNL2_IS_ALIGNED(addend->data, NNL2_TENSOR_ALIGNMENT_32);

		// Handling the case when the data types are the same
		switch(dtype_summand) {
			case FLOAT64: {
                double* data_summand = (double*)summand->data;
                double* data_addend = (double*)addend->data;
                double* data_sum = (double*)sum->data;
                
                // Copy data from summand to result first
                memcpy(data_sum, data_summand, len * sizeof(double));
                
                // Use optimized addition
                nnl2_avx_add_non_in_place_float64_same_type(data_sum, data_addend, len, aligned_summand, aligned_addend);
                break;
            }
            
            case FLOAT32: {
                float* data_summand = (float*)summand->data;
                float* data_addend = (float*)addend->data;
                float* data_sum = (float*)sum->data;
                
                // Copy data from summand to result first
                memcpy(data_sum, data_summand, len * sizeof(float));
                
                // Use optimized addition
                nnl2_avx_add_non_in_place_float32_same_type(data_sum, data_addend, len, aligned_summand, aligned_addend);
                break;
            }
            
            case INT32: {
                int32_t* data_summand = (int32_t*)summand->data;
                int32_t* data_addend = (int32_t*)addend->data;
                int32_t* data_sum = (int32_t*)sum->data;
                
                // Copy data from summand to result first
                memcpy(data_sum, data_summand, len * sizeof(int32_t));
                
                // Use optimized addition
                nnl2_avx_add_non_in_place_int32_same_type(data_sum, data_addend, len, aligned_summand, aligned_addend);
                break;
            }
			
			default: {
				NNL2_TYPE_ERROR(dtype_summand);
				return NULL;
			}
		} 
	} else {
		// Handling the case when the data types are NOT the same
		// For mixed types, using scalar operations since AVX doesn't easily handle
        // type conversions within the same instruction
		
		switch(winner_in_the_type_hierarchy) {
			case FLOAT64: {
                double* data_sum = (double*)sum->data;
                
				// Element-wise addition
                for(size_t i = 0; i < len; i++) {
                    void* elem_summand = (char*)summand->data + i * get_dtype_size(dtype_summand);
                    void* elem_addend = (char*)addend->data + i * get_dtype_size(dtype_addend);
                    
                    data_sum[i] = nnl2_convert_to_float64(elem_summand, dtype_summand) + nnl2_convert_to_float64(elem_addend, dtype_addend);
                }
				
                break;
            }
            
            case FLOAT32: {
                float* data_sum = (float*)sum->data;
				
				// Element-wise addition
                for(size_t i = 0; i < len; i++) {
                    void* elem_summand = (char*)summand->data + i * get_dtype_size(dtype_summand);
                    void* elem_addend = (char*)addend->data + i * get_dtype_size(dtype_addend);
                    
                    data_sum[i] = nnl2_convert_to_float32(elem_summand, dtype_summand) + nnl2_convert_to_float32(elem_addend, dtype_addend);
                }
                
                break;
            }
            
            case INT32: {
                int32_t* data_sum = (int32_t*)sum->data;

				// Element-wise addition
                for(size_t i = 0; i < len; i++) {
                    void* elem_summand = (char*)summand->data + i * get_dtype_size(dtype_summand);
                    void* elem_addend = (char*)addend->data + i * get_dtype_size(dtype_addend);
                    
                    data_sum[i] = nnl2_convert_to_int32(elem_summand, dtype_summand) + nnl2_convert_to_int32(elem_addend, dtype_addend);
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
    
    return sum;
}

/** @brief 
 * AVX-optimized element-wise addition for int32 tensors (non-in-place)
 *
 ** @details
 * See docs at declaration
 *
 ** @see nnl2_avx_add_non_in_place_int32_same_type (declaration)
 **/
static inline void nnl2_avx_add_non_in_place_int32_same_type(int32_t* a, const int32_t* b, size_t len, bool aligned_a, bool aligned_b) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
    
    size_t i = 0;
    
    // Case 1: Both tensors are aligned 
    if(aligned_a && aligned_b) {
        for(; i + 7 < len; i += 8) {
            __m256i v_a = _mm256_load_si256((__m256i*)&a[i]);       // Fast loading of aligned data
            __m256i v_b = _mm256_load_si256((__m256i*)&b[i]);       // Fast loading of aligned data
            __m256i v_result = _mm256_add_epi32(v_a, v_b);          // Vector addition
            _mm256_store_si256((__m256i*)&a[i], v_result);          // Fast saving to aligned memory
        }
    } 
    
    // Case 2: Only tensor a is aligned
    else if(aligned_a) {
        for(; i + 7 < len; i += 8) {
            __m256i v_a = _mm256_load_si256((__m256i*)&a[i]);       // Fast loading of aligned data
            __m256i v_b = _mm256_loadu_si256((__m256i*)&b[i]);      // Slow loading of unaligned data
            __m256i v_result = _mm256_add_epi32(v_a, v_b);          // Vector addition
            _mm256_store_si256((__m256i*)&a[i], v_result);          // Fast saving to aligned memory
        }
    } 
    
    // Case 3: Only tensor b is aligned
    else if(aligned_b) {
        for(; i + 7 < len; i += 8) {
            __m256i v_a = _mm256_loadu_si256((__m256i*)&a[i]);      // Slow loading of unaligned data
            __m256i v_b = _mm256_load_si256((__m256i*)&b[i]);       // Fast loading of aligned data
            __m256i v_result = _mm256_add_epi32(v_a, v_b);          // Vector addition
            _mm256_storeu_si256((__m256i*)&a[i], v_result);         // Slow saving to unaligned memory
        }
    } 
    
    // Case 4: Both tensors are not aligned
    else {
        for(; i + 7 < len; i += 8) {
            __m256i v_a = _mm256_loadu_si256((__m256i*)&a[i]);      // Slow loading of unaligned data
            __m256i v_b = _mm256_loadu_si256((__m256i*)&b[i]);      // Slow loading of unaligned data
            __m256i v_result = _mm256_add_epi32(v_a, v_b);          // Vector addition
            _mm256_storeu_si256((__m256i*)&a[i], v_result);         // Slow saving to unaligned memory
        }
    }
    
    // Processing the remainder
    for(; i < len; i++) a[i] += b[i];
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
}

/** @brief 
 * AVX-optimized element-wise addition for float32 tensors (non-in-place)
 *
 ** @details
 * See docs at declaration
 *
 ** @see nnl2_avx_add_non_in_place_float32_same_type (declaration)
 **/
static inline void nnl2_avx_add_non_in_place_float32_same_type(float* a, const float* b, size_t len, bool aligned_a, bool aligned_b) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
    
    size_t i = 0;
    
    // Case 1: Both tensors are aligned 
    if(aligned_a && aligned_b) {
        for(; i + 7 < len; i += 8) {
            __m256 v_a = _mm256_load_ps(&a[i]);        // Fast loading of aligned data
            __m256 v_b = _mm256_load_ps(&b[i]);        // Fast loading of aligned data
            __m256 v_result = _mm256_add_ps(v_a, v_b); // Vector addition
            _mm256_store_ps(&a[i], v_result);          // Fast saving to aligned memory
        }
    } 
    
    // Case 2: Only tensor a is aligned
    else if(aligned_a) {
        for(; i + 7 < len; i += 8) {
            __m256 v_a = _mm256_load_ps(&a[i]);        // Fast loading of aligned data
            __m256 v_b = _mm256_loadu_ps(&b[i]);       // Slow loading of unaligned data
            __m256 v_result = _mm256_add_ps(v_a, v_b); // Vector addition
            _mm256_store_ps(&a[i], v_result);          // Fast saving to aligned memory
        }
    } 
    
    // Case 3: Only tensor b is aligned
    else if(aligned_b) {
        for(; i + 7 < len; i += 8) {
            __m256 v_a = _mm256_loadu_ps(&a[i]);       // Slow loading of unaligned data
            __m256 v_b = _mm256_load_ps(&b[i]);        // Fast loading of aligned data
            __m256 v_result = _mm256_add_ps(v_a, v_b); // Vector addition
            _mm256_storeu_ps(&a[i], v_result);         // Slow saving to unaligned memory
        }
    } 
    
    // Case 4: Both tensors are not aligned
    else {
        for(; i + 7 < len; i += 8) {
            __m256 v_a = _mm256_loadu_ps(&a[i]);       // Slow loading of unaligned data
            __m256 v_b = _mm256_loadu_ps(&b[i]);       // Slow loading of unaligned data
            __m256 v_result = _mm256_add_ps(v_a, v_b); // Vector addition
            _mm256_storeu_ps(&a[i], v_result);         // Slow saving to unaligned memory
        }
    }
    
    // Processing the remainder
    for(; i < len; i++) a[i] += b[i];
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
}

/** @brief 
 * AVX-optimized element-wise addition for float64 tensors (non-in-place)
 *
 ** @details
 * See docs at declaration
 *
 ** @see nnl2_avx_add_non_in_place_float64_same_type (declaration)
 **/
static inline void nnl2_avx_add_non_in_place_float64_same_type(double* a, const double* b, size_t len, bool aligned_a, bool aligned_b) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
    
    size_t i = 0;
    
    // Case 1: Both tensors are aligned 
    if(aligned_a && aligned_b) {
        for(; i + 3 < len; i += 4) {
            __m256d v_a = _mm256_load_pd(&a[i]);        // Fast loading of aligned data
            __m256d v_b = _mm256_load_pd(&b[i]);        // Fast loading of aligned data
            __m256d v_result = _mm256_add_pd(v_a, v_b); // Vector addition
            _mm256_store_pd(&a[i], v_result);           // Fast saving to aligned memory
        }
    } 
    
    // Case 2: Only tensor a is aligned
    else if(aligned_a) {
        for(; i + 3 < len; i += 4) {
            __m256d v_a = _mm256_load_pd(&a[i]);        // Fast loading of aligned data
            __m256d v_b = _mm256_loadu_pd(&b[i]);       // Slow loading of unaligned data
            __m256d v_result = _mm256_add_pd(v_a, v_b); // Vector addition
            _mm256_store_pd(&a[i], v_result);           // Fast saving to aligned memory
        }
    } 
    
    // Case 3: Only tensor b is aligned
    else if(aligned_b) {
        for(; i + 3 < len; i += 4) {
            __m256d v_a = _mm256_loadu_pd(&a[i]);       // Slow loading of unaligned data
            __m256d v_b = _mm256_load_pd(&b[i]);        // Fast loading of aligned data
            __m256d v_result = _mm256_add_pd(v_a, v_b); // Vector addition
            _mm256_storeu_pd(&a[i], v_result);          // Slow saving to unaligned memory
        }
    } 
    
    // Case 4: Both tensors are not aligned
    else {
        for(; i + 3 < len; i += 4) {
            __m256d v_a = _mm256_loadu_pd(&a[i]);       // Slow loading of unaligned data
            __m256d v_b = _mm256_loadu_pd(&b[i]);       // Slow loading of unaligned data
            __m256d v_result = _mm256_add_pd(v_a, v_b); // Vector addition
            _mm256_storeu_pd(&a[i], v_result);          // Slow saving to unaligned memory
        }
    }
    
    // Processing the remainder
    for(; i < len; i++) a[i] += b[i];
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
}

#endif

#ifdef NNL2_PTHREAD_AVAILABLE

/** @brief 
 * Worker function for parallel addition for same data types
 * 
 * @param arg 
 * Pointer to add_ptask structure containing task parameters
 *
 * @return NULL (for pthread api)
 */
void* nnl2_own_padd_same_type(void* arg);

/** @brief 
 * Worker function for parallel addition for mixed data types
 * 
 * @param arg 
 * Pointer to add_ptask structure containing task parameters
 *
 * @return NULL (for pthread api)
 */
void* nnl2_own_padd_mixed_types(void* arg);

/** @brief 
 * SIMD-optimized worker function for parallel addition for same float64 data types
 * 
 * @param arg 
 * Pointer to add_ptask structure containing task parameters
 *
 * @return NULL (for pthread api)
 */
void* nnl2_own_padd_simd_float64(void* arg);

/** @brief 
 * SIMD-optimized worker function for parallel addition for same float32 data types
 * 
 * @param arg 
 * Pointer to add_ptask structure containing task parameters
 *
 * @return NULL (for pthread api)
 */
void* nnl2_own_padd_simd_float32(void* arg);

/** @brief 
 * SIMD-optimized worker function for parallel addition for same int32 data types
 * 
 * @param arg 
 * Pointer to add_ptask structure containing task parameters
 *
 * @return NULL (for pthread api)
 */
void* nnl2_own_padd_simd_int32(void* arg);

#ifdef NNL2_AVX2

/** @brief 
 * SIMD-optimized worker function for parallel addition for same float64 data types
 * 
 * @param arg 
 * Pointer to add_ptask structure containing task parameters
 *
 * @return NULL (for pthread api)
 */
void* nnl2_own_padd_simd_float64(void* arg);

/** @brief 
 * SIMD-optimized worker function for parallel addition for same float32 data types
 * 
 * @param arg 
 * Pointer to add_ptask structure containing task parameters
 *
 * @return NULL (for pthread api)
 */
void* nnl2_own_padd_simd_float32(void* arg);

/** @brief 
 * SIMD-optimized worker function for parallel addition for same int32 data types
 * 
 * @param arg 
 * Pointer to add_ptask structure containing task parameters
 *
 * @return NULL (for pthread api)
 */
void* nnl2_own_padd_simd_int32(void* arg);

#endif

/** @brief
 * Parallel implementation of tensor addition using pthreads
 *
 ** @param summand
 * Pointer to the summand tensor
 *
 ** @param addend
 * Pointer to the addend tensor
 *
 ** @return 
 * Pointer to a new tensor with the addition result
 */
nnl2_tensor* nnl2_own_add(nnl2_tensor* summand, nnl2_tensor* addend) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Calculate the total number of elements in the tensors
    size_t len = product(summand->shape, summand->rank);
    
    nnl2_tensor_type dtype_summand = summand->dtype;
    nnl2_tensor_type dtype_addend = addend->dtype;
    
    // Selecting the winning type (higher in the hierarchy)
    nnl2_tensor_type winner_in_the_type_hierarchy = MAX(dtype_summand, dtype_addend);

    // Create an output tensor with the same shape and data type
    nnl2_tensor* amount = nnl2_empty(summand->shape, summand->rank, winner_in_the_type_hierarchy);
    
    if(len == 0) return amount;
    
    // Use naive implementation for small tensors
    if(len < NNL2_ADD_PARALLEL_THREASHOLD) {
        amount = nnl2_naive_add(summand, addend);
        if(amount == NULL) {
			NNL2_ERROR("Failed to add");
		}
		
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
		
        return amount;
    }
    
    // Allocate arrays for thread handles and task descriptors
    pthread_t threads[NNL2_NUM_THREADS];
    add_ptask tasks[NNL2_NUM_THREADS];
    
    // Calculate base chunk size and remainder for balanced distribution
    size_t chunk = len / NNL2_NUM_THREADS;
    size_t remainder = len % NNL2_NUM_THREADS;
    
    bool use_simd = false;
	
    #ifdef NNL2_AVX256_AVAILABLE
    if(dtype_summand == dtype_addend) {
        bool aligned_summand = NNL2_IS_ALIGNED(summand->data, NNL2_TENSOR_ALIGNMENT_32);
        bool aligned_addend = NNL2_IS_ALIGNED(addend->data, NNL2_TENSOR_ALIGNMENT_32);
        bool aligned_result = NNL2_IS_ALIGNED(amount->data, NNL2_TENSOR_ALIGNMENT_32);
        use_simd = aligned_summand && aligned_addend && aligned_result;
    }
    #endif
    
    // Distribute work among threads with load balancing
    size_t current_start = 0;
    for (size_t i = 0; i < NNL2_NUM_THREADS; i++) {
        size_t current_chunk = chunk + (i < remainder ? 1 : 0);
        
        // Configure task for this thread
        tasks[i].summand_data = summand->data;
        tasks[i].addend_data = addend->data;
        tasks[i].result_data = amount->data;
        tasks[i].start = current_start;
        tasks[i].end = current_start + current_chunk;
        tasks[i].dtype_summand = dtype_summand;
        tasks[i].dtype_addend = dtype_addend;
        tasks[i].result_dtype = winner_in_the_type_hierarchy;
        
        // Create thread to process the assigned chunk
        int status;
        
        #ifdef NNL2_AVX256_AVAILABLE
			if(use_simd && dtype_summand == dtype_addend) {
				switch(dtype_summand) {
					case FLOAT64: status = pthread_create(&threads[i], NULL, nnl2_own_padd_simd_float64, &tasks[i]); break;
					case FLOAT32: status = pthread_create(&threads[i], NULL, nnl2_own_padd_simd_float32, &tasks[i]); break;
					case INT32:   status = pthread_create(&threads[i], NULL, nnl2_own_padd_simd_int32, &tasks[i]);   break;
					
					default: {
						status = pthread_create(&threads[i], NULL, nnl2_own_padd_same_type, &tasks[i]);
						break;
					}
				}
			} else 
        #endif
        {
            if(dtype_summand == dtype_addend) {
                status = pthread_create(&threads[i], NULL, nnl2_own_padd_same_type, &tasks[i]);
            } else {
                status = pthread_create(&threads[i], NULL, nnl2_own_padd_mixed_types, &tasks[i]);
            }
        }
        
        if(status != 0) {
            NNL2_THREAD_CREATE_ERROR(status, "nnl2_own_add");
            // Clean up already created threads
            for(size_t j = 0; j < i; j++) {
                pthread_join(threads[j], NULL);
            }
			
            nnl2_free_tensor(amount);
            return NULL;
        }
        
        current_start += current_chunk;
    }
    
    // Wait for all threads to complete their work
    for (size_t i = 0; i < NNL2_NUM_THREADS; i++) {
        int join_status = pthread_join(threads[i], NULL);
        if(join_status != 0) {
            NNL2_THREAD_JOIN_ERROR(join_status, "nnl2_own_add");
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return amount;
}

/** @brief
 * See docs at declaration
 *
 ** @see nnl2_own_padd_same_type
 **/
void* nnl2_own_padd_same_type(void* arg) {
    add_ptask* task = (add_ptask*)arg;
    
    switch(task->dtype_summand) {
        case FLOAT64: {
            volatile double* data_summand = (double*)task->summand_data;
            volatile double* data_addend = (double*)task->addend_data;
            volatile double* data_amount = (double*)task->result_data;
            
            for(size_t i = task->start; i < task->end; i++) {
                data_amount[i] = data_summand[i] + data_addend[i];
            }
			
            break;
        }
        
        case FLOAT32: {
            volatile float* data_summand = (float*)task->summand_data;
            volatile float* data_addend = (float*)task->addend_data;
            volatile float* data_amount = (float*)task->result_data;
            
            for(size_t i = task->start; i < task->end; i++) {
                data_amount[i] = data_summand[i] + data_addend[i];
            }
			
            break;
        }
        
        case INT32: {
            volatile int32_t* data_summand = (int32_t*)task->summand_data;
            volatile int32_t* data_addend = (int32_t*)task->addend_data;
            volatile int32_t* data_amount = (int32_t*)task->result_data;
            
            for(size_t i = task->start; i < task->end; i++) {
                data_amount[i] = data_summand[i] + data_addend[i];
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
 * SIMD-optimized worker function for float64 addition
 *
 ** @see nnl2_own_padd_simd_float64
 **/
void* nnl2_own_padd_simd_float64(void* arg) {
    add_ptask* task = (add_ptask*)arg;
    
    double* data_summand = (double*)task->summand_data;
    double* data_addend = (double*)task->addend_data;
    double* data_amount = (double*)task->result_data;
    
    size_t i = task->start;
    
    // Process 4 elements at a time using AVX
    for(; i + 3 < task->end; i += 4) {
        __m256d v_summand = _mm256_load_pd(&data_summand[i]);        // Load 4 doubles
        __m256d v_addend = _mm256_load_pd(&data_addend[i]);          // Load 4 doubles
        __m256d v_result = _mm256_add_pd(v_summand, v_addend);       // Vector addition
        _mm256_store_pd(&data_amount[i], v_result);                  // Store result
    }
    
    // Process remainder elements
    for(; i < task->end; i++) {
        data_amount[i] = data_summand[i] + data_addend[i];
    }
    
    return NULL;
}

/** @brief
 * SIMD-optimized worker function for float32 addition
 *
 ** @see nnl2_own_padd_simd_float32
 **/
void* nnl2_own_padd_simd_float32(void* arg) {
    add_ptask* task = (add_ptask*)arg;
    
    float* data_summand = (float*)task->summand_data;
    float* data_addend = (float*)task->addend_data;
    float* data_amount = (float*)task->result_data;
    
    size_t i = task->start;
    
    // Process 8 elements at a time using AVX
    for(; i + 7 < task->end; i += 8) {
        __m256 v_summand = _mm256_load_ps(&data_summand[i]);        // Load 8 floats
        __m256 v_addend = _mm256_load_ps(&data_addend[i]);          // Load 8 floats
        __m256 v_result = _mm256_add_ps(v_summand, v_addend);       // Vector addition
        _mm256_store_ps(&data_amount[i], v_result);                 // Store result
    }
    
    // Process remainder elements
    for(; i < task->end; i++) {
        data_amount[i] = data_summand[i] + data_addend[i];
    }
    
    return NULL;
}

/** @brief
 * SIMD-optimized worker function for int32 addition
 *
 ** @see nnl2_own_padd_simd_int32
 **/
void* nnl2_own_padd_simd_int32(void* arg) {
    add_ptask* task = (add_ptask*)arg;
    
    int32_t* data_summand = (int32_t*)task->summand_data;
    int32_t* data_addend = (int32_t*)task->addend_data;
    int32_t* data_amount = (int32_t*)task->result_data;
    
    size_t i = task->start;
    
    // Process 8 elements at a time using AVX
    for(; i + 7 < task->end; i += 8) {
        __m256i v_summand = _mm256_load_si256((__m256i*)&data_summand[i]);  // Load 8 int32
        __m256i v_addend = _mm256_load_si256((__m256i*)&data_addend[i]);    // Load 8 int32
        __m256i v_result = _mm256_add_epi32(v_summand, v_addend);           // Vector addition
        _mm256_store_si256((__m256i*)&data_amount[i], v_result);            // Store result
    }
    
    // Process remainder elements
    for(; i < task->end; i++) {
        data_amount[i] = data_summand[i] + data_addend[i];
    }
    
    return NULL;
}

#endif

/** @brief
 * See docs at declaration
 *
 ** @see nnl2_own_padd_mixed_types
 **/
void* nnl2_own_padd_mixed_types(void* arg) {
    add_ptask* task = (add_ptask*)arg;
    
    switch(task->result_dtype) {
        case FLOAT64: {
            volatile double* data_amount = (double*)task->result_data;
            
            for(size_t i = task->start; i < task->end; i++) {
                void* elem_summand = (char*)task->summand_data + i * get_dtype_size(task->dtype_summand);
                void* elem_addend = (char*)task->addend_data + i * get_dtype_size(task->dtype_addend);
                
                data_amount[i] = nnl2_convert_to_float64(elem_summand, task->dtype_summand) + nnl2_convert_to_float64(elem_addend, task->dtype_addend);
            }
			
            break;
        }
        
        case FLOAT32: {
            volatile float* data_amount = (float*)task->result_data;
            
            for(size_t i = task->start; i < task->end; i++) {
                void* elem_summand = (char*)task->summand_data + i * get_dtype_size(task->dtype_summand);
                void* elem_addend = (char*)task->addend_data + i * get_dtype_size(task->dtype_addend);
                
                data_amount[i] = nnl2_convert_to_float32(elem_summand, task->dtype_summand) + nnl2_convert_to_float32(elem_addend, task->dtype_addend);
            }
			
            break;
        }
        
        case INT32: {
            volatile int32_t* data_amount = (int32_t*)task->result_data;
            
            for(size_t i = task->start; i < task->end; i++) {
                void* elem_summand = (char*)task->summand_data + i * get_dtype_size(task->dtype_summand);
                void* elem_addend = (char*)task->addend_data + i * get_dtype_size(task->dtype_addend);
                
                data_amount[i] = nnl2_convert_to_int32(elem_summand, task->dtype_summand) + nnl2_convert_to_int32(elem_addend, task->dtype_addend);
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
 * @brief Backend implementations for addition operation
 * @details
 * Array follows the common backend registration pattern.
 * Currently registered backends:
 *  - nnl2_naive: Basic reference implementation
 *  - nnl2_own: Parallel pthread implementation
 *  - nnl2_avx256: AVX256 implementation (if available)
 * 
 * @see nnl2_naive_add
 * @see nnl2_avx256_add
 */
nnl2_runtime_implementation add_backends[] = {
	REGISTER_BACKEND(nnl2_naive_add, nnl2_naive, NAIVE_BACKEND_NAME),
	
	#if defined(__AVX__) && TENSOR_MEM_ALIGNMENT == 32
		REGISTER_BACKEND(nnl2_avx256_add, nnl2_avx256, AVX256_BACKEND_NAME),
	#endif
	
	#ifdef NNL2_PTHREAD_AVAILABLE
		REGISTER_BACKEND(nnl2_own_add, nnl2_own_2, NNL2_OWN_NAME),
	#endif
};

/**
 * @brief Function pointer for addition operation
 * @ingroup backend_system 
 */
addfn add;

/** 
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(add);

/** 
 * @brief Sets the backend for addition operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void set_add_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(add_backends, add, backend_name, CURRENT_BACKEND(add));
}

/** 
 * @brief Gets the name of the active backend for addition operation
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* get_add_backend() {
	return CURRENT_BACKEND(add);
}

/** 
 * @brief Function declaration for getting all `add` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(add);

/**
 * @brief Function declaration for getting the number of all `add` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(add);

#endif /** NNL2_ADD_H **/
