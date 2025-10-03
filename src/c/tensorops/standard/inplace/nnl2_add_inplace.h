#ifndef NNL2_ADD_INPLACE_H
#define NNL2_ADD_INPLACE_H

/** @brief 
 * Performs element-wise addition of two tensors (naive implementation)
 * 
 * Adds the elements of the addend tensor to the corresponding elements 
 * of the summand tensor, modifying the summand tensor in place
 *
 ** @param summand 
 * Pointer to the tensor that will be modified (receives the addition result)
 *
 ** @param addend 
 * Pointer to the tensor whose values will be added to the summand
 *
 ** @note 
 * Supports different data types through automatic conversion
 * The addend elements are converted to the summand's data type
 *
 ** @note 
 * In max safety mode, validates tensor pointers and data pointers before access
 *
 ** @warning 
 * This function modifies the summand tensor directly
 *
 * @example
 * // Create two tensors with the same shape
 * Tensor* a = nnl2_ones((int[]){2, 3}, 2, FLOAT32);
 * Tensor* b = nnl2_ones((int[]){2, 3}, 2, FLOAT32);
 * 
 * // Add b to a (a becomes a + b)
 * naive_addinplace(a, b);
 * 
 * // Now a contains 2.0 in all elements
 * nnl2_quick_print_tensor(a);
 * 
 * // Cleanup
 * nnl2_free_tensor(a);
 * nnl2_free_tensor(b);
 */
void nnl2_naive_addinplace(Tensor* summand, const Tensor* addend) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	// Additional checks at the maximum safety level
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(summand, "Summand tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(summand->data, "Summand tensor's data is NULL");
		
		NNL2_CHECK_NULL_IF_ERR_RETURN(addend, "Addend tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(addend->data, "Addend tensor's data is NULL");
	#endif
	
	// Calculating the total number of elements in the summand tensor
	size_t len_summand = product(summand->shape, summand->rank);
	
	// If the tensor is empty, exit the function
	if(len_summand == 0) return;
	
	TensorType dtype_summand = summand->dtype;
	TensorType dtype_addend = addend->dtype;
	
	if(dtype_summand == dtype_addend) {
		// Handling case when the tensors have the same type
		
		switch(dtype_summand) {
			case FLOAT64: {
				volatile double* data_summand = (double*)summand->data;
				volatile double* data_addend = (double*)addend->data;
				
				// Element-wise addition
				for(size_t i = 0; i < len_summand; i++) data_summand[i] += data_addend[i];
				break;
			}
			
			case FLOAT32: {
				volatile float* data_summand = (float*)summand->data;
				volatile float* data_addend = (float*)addend->data;
				
				// Element-wise addition
				for(size_t i = 0; i < len_summand; i++) data_summand[i] += data_addend[i];	
				break;
			}
			
			case INT32: {
				volatile int32_t* data_summand = (int32_t*)summand->data;
				volatile int32_t* data_addend = (int32_t*)addend->data;
				
				// Element-wise addition
				for(size_t i = 0; i < len_summand; i++) data_summand[i] += data_addend[i];		
				break;
			}
			
			default: {
				NNL2_TYPE_ERROR(dtype_summand);
				return;
			}
		}
	} else {
		// Handling the case when tensors have different data types
		// Calculating the step size for accessing addend tensor elements
		size_t addend_step = get_dtype_size(dtype_addend);
		
		// Casting addend data to char* for byte access
		char* addend_data = (char*)addend->data;
		
		switch(dtype_summand) {
			case FLOAT64: {
				volatile double* data_summand = (double*)summand->data;
				
				// For each element, convert the addend element to FLOAT64 and add it
				for(size_t i = 0; i < len_summand; i++) {
					void* addend_elem = addend_data + i * addend_step;
					data_summand[i] += nnl2_convert_to_float64(addend_elem, dtype_addend);
				}
				
				break; 
			}
			
			case FLOAT32: {
				volatile float* data_summand = (float*)summand->data;
				
				// For each element, convert the addend element to FLOAT32 and add it
				for(size_t i = 0; i < len_summand; i++) {
					void* addend_elem = addend_data + i * addend_step;
					data_summand[i] += nnl2_convert_to_float32(addend_elem, dtype_addend);
				}
				
				break; 
			}
			
			case INT32: {
				volatile int32_t* data_summand = (int32_t*)summand->data;
				
				// For each element, convert the addend element to INT32 and add it
				for(size_t i = 0; i < len_summand; i++) {
					void* addend_elem = addend_data + i * addend_step;
					data_summand[i] += nnl2_convert_to_int32(addend_elem, dtype_addend);
				}
				
				break; 
			}
			
			default: {
				NNL2_TYPE_ERROR(dtype_summand);
				return;
			}
		}
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

#ifdef __AVX__

// Declarations

/** @brief 
 * AVX256 optimized addition for double with the same types
 *
 ** @param summand 
 * Pointer to the summand data (mutable)
 *
 ** @param addend 
 * Pointer to the addend data
 *
 ** @param len 
 * Number of elements to process
 *
 ** @param aligned_summand 
 * Flag for aligning the summand data
 *
 ** @param aligned_addend 
 * Flag for aligning the addend data
 */
static inline void nnl2_avx_add_float64_same_type(double* summand, double* addend, size_t len, bool aligned_summand, bool aligned_addend);

/** @brief
 * AVX256 optimized addition for float with the same types
 * Documentation is identical to that of nnl2_avx_add_float64_same_type
 *
 ** @see nnl2_avx_add_float64_same_type
 **/
static inline void nnl2_avx_add_float32_same_type(float* summand, float* addend, size_t len, bool aligned_summand, bool aligned_addend);

/** @brief
 * AVX256 optimized addition for int32 with the same types
 * Documentation is identical to that of nnl2_avx_add_float64_same_type
 *
 ** @see nnl2_avx_add_float64_same_type
 **/
static inline void nnl2_avx_add_int32_same_type(int32_t* summand, int32_t* addend, size_t len, bool aligned_summand, bool aligned_addend);

/** @brief
 * AVX256 optimized addition for double with different types
 *
 ** @param summand 
 * Pointer to the summand tensor data (mutable)
 * 
 ** @param addend 
 * Pointer to the addend tensor data (may be of a different type)
 *
 ** @param len 
 * Number of elements to process
 *
 ** @param aligned_summand 
 * Flag for aligning the summand data
 */
static inline void nnl2_avx_add_float64_diff_type(double* summand, const Tensor* addend, size_t len, bool aligned_summand);

/** @brief
 * AVX256 optimized addition for float with different types
 * Documentation is identical to that of nnl2_avx_add_float64_diff_type
 *
 ** @see nnl2_avx_add_float64_diff_type
 **/
static inline void nnl2_avx_add_float32_diff_type(float* summand, const Tensor* addend, size_t len, bool aligned_summand);

/** @brief
 * AVX256 optimized addition for int32 with different types
 * Documentation is identical to that of nnl2_avx_add_float64_diff_type
 *
 ** @see nnl2_avx_add_float64_diff_type
 **/
static inline void nnl2_avx_add_int32_diff_type(int32_t* summand, const Tensor* addend, size_t len, bool aligned_summand);

// Main function

/** @brief
 * AVX256-optimized in-place addition operation 
 *
 ** @param summand
 * A tensor to which values are added (mutable)
 *
 ** @param addend
 * The tensor whose values are being added
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
 ** @see nnl2_avx_add_float64_same_type
 ** @see nnl2_avx_add_float32_same_type
 ** @see nnl2_avx_add_int32_same_type
 **
 ** @see nnl2_avx_add_float64_diff_type
 ** @see nnl2_avx_add_float32_diff_type
 ** @see nnl2_avx_add_int32_diff_type
 **/
void nnl2_avx256_addinplace(Tensor* summand, const Tensor* addend) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	// Additional checks at the maximum safety level
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX  
        NNL2_CHECK_NULL_IF_ERR_RETURN(summand, "Summand tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(summand->data, "Summand tensor's data is NULL");
        
        NNL2_CHECK_NULL_IF_ERR_RETURN(addend, "Addend tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(addend->data, "Addend tensor's data is NULL");
    #endif
	
	// Calculating the total number of elements in the summand tensor
    size_t len_summand = product(summand->shape, summand->rank);	
	
	// If the tensor is empty, exit the function
    if(len_summand == 0) return;
    
    TensorType dtype_summand = summand->dtype;
	TensorType dtype_addend = addend->dtype;
	
	bool is_aligned_summand = NNL2_IS_ALIGNED(summand->data, NNL2_TENSOR_ALIGNMENT_32);
	bool is_aligned_addend = NNL2_IS_ALIGNED(addend->data, NNL2_TENSOR_ALIGNMENT_32);
	
	// Warning for unaligned memory in safety modes (performance impact)
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(!is_aligned_summand) {
			NNL2_WARN("In the avx256 implementation of add in-place, summand memory is not aligned to 32 bytes. Calculations may be slightly slower");
		}
		
		if(!is_aligned_addend && dtype_summand == dtype_addend) {
            NNL2_WARN("In the avx256 implementation of add in-place, addend memory is not aligned to 32 bytes. Calculations may be slightly slower");
        }
	#endif
	
	if(dtype_summand == dtype_addend) {
		// Handling case when the tensors have the same type
		
		switch (dtype_summand) {
			case FLOAT64: nnl2_avx_add_float64_same_type((double*)summand->data, (double*)addend->data, len_summand, is_aligned_summand, is_aligned_addend);  break;
			case FLOAT32: nnl2_avx_add_float32_same_type((float*)summand->data, (float*)addend->data, len_summand, is_aligned_summand, is_aligned_addend);    break;	
			case INT32:   nnl2_avx_add_int32_same_type((int32_t*)summand->data, (int32_t*)addend->data, len_summand, is_aligned_summand, is_aligned_addend);  break;
			
			default: {
				NNL2_TYPE_ERROR(dtype_summand);
				return;
			}
		}
	} else {
		// Handling the case when tensors have different data types

        switch(dtype_summand) {
            case FLOAT64: nnl2_avx_add_float64_diff_type((double*)summand->data, addend, len_summand, is_aligned_summand);  break;
            case FLOAT32: nnl2_avx_add_float32_diff_type((float*)summand->data, addend, len_summand, is_aligned_summand);   break;
            case INT32:   nnl2_avx_add_int32_diff_type((int32_t*)summand->data, addend, len_summand, is_aligned_summand);   break;
            
            default: {
                NNL2_TYPE_ERROR(dtype_summand);
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
 * Implementation of double addition with the same types
 *
 ** @details 
 * Handles 4 combinations of memory alignment:
 * - The summand and addend tensors are aligned in memory
 * - The summand tensor is aligned in memory, but the addend is not
 * - The addend tensor is aligned in memory, but the summand is not
 * - The summand and addend tensors are not aligned in memory
 *
 ** @see nnl2_avx256_addinplace
 **/
static inline void nnl2_avx_add_float64_same_type(double* summand, double* addend, size_t len, bool aligned_summand, bool aligned_addend) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
	
    size_t i = 0;
    
	// Case 1: Both tensors are aligned 
    if(aligned_summand && aligned_addend) {
        for(; i + 3 < len; i += 4) {
            __m256d v_summand = _mm256_load_pd(&summand[i]); 	    // Fast loading of aligned data
            __m256d v_addend = _mm256_load_pd(&addend[i]);		    // Fast loading of aligned data
            __m256d v_result = _mm256_add_pd(v_summand, v_addend);  // Vector addition
            _mm256_store_pd(&summand[i], v_result);				    // Fast saving to aligned memory
        }
    } 
	
	// Case 2: Only the summand is aligned
	else if(aligned_summand) {
        for(; i + 3 < len; i += 4) {
            __m256d v_summand = _mm256_load_pd(&summand[i]);	    // Fast loading of aligned data
            __m256d v_addend = _mm256_loadu_pd(&addend[i]);			// Slow loading of unaligned data
            __m256d v_result = _mm256_add_pd(v_summand, v_addend);  // Vector addition
            _mm256_store_pd(&summand[i], v_result);					// Fast saving to aligned memory
        }
    } 
	
	// Case 3: Only the addend is aligned
	else if(aligned_addend) {
        for(; i + 3 < len; i += 4) {
            __m256d v_summand = _mm256_loadu_pd(&summand[i]);       // Slow loading of unaligned data
            __m256d v_addend = _mm256_load_pd(&addend[i]);			// Fast loading of aligned data
            __m256d v_result = _mm256_add_pd(v_summand, v_addend);  // Vector addition
            _mm256_storeu_pd(&summand[i], v_result);				// Slow saving to unaligned memoty
        }
    } 
	
	// Case 4: Both tensors are not aligned
	else {
        for(; i + 3 < len; i += 4) {
            __m256d v_summand = _mm256_loadu_pd(&summand[i]);	    // Slow loading of unaligned data
            __m256d v_addend = _mm256_loadu_pd(&addend[i]);		    // Slow loading of unaligned data
            __m256d v_result = _mm256_add_pd(v_summand, v_addend);  // Vector addition
            _mm256_storeu_pd(&summand[i], v_result);				// Slow saving to unaligned memoty
        }
    }
    
	// Processing the remainder
    for(; i < len; i++) summand[i] += addend[i];
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
}

/** @brief 
 * Implementation of float addition with the same types
 * Similar to double, but processes 8 elements per iteration
 *
 ** @see nnl2_avx256_addinplace
 ** @see nnl2_avx_add_float64_same_type
 **/
static inline void nnl2_avx_add_float32_same_type(float* summand, float* addend, size_t len, bool aligned_summand, bool aligned_addend) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
	
    size_t i = 0;
    
	// Case 1: Both tensors are aligned 
    if(aligned_summand && aligned_addend) {
        for(; i + 7 < len; i += 8) {
            __m256 v_summand = _mm256_load_ps(&summand[i]);        // Fast loading of aligned data
            __m256 v_addend = _mm256_load_ps(&addend[i]);		   // Fast loading of aligned data
            __m256 v_result = _mm256_add_ps(v_summand, v_addend);  // Vector addition
            _mm256_store_ps(&summand[i], v_result);				   // Fast saving to aligned memory
        }
    } 
	
	// Case 2: Only the summand is aligned
	else if(aligned_summand) {
        for(; i + 7 < len; i += 8) {
            __m256 v_summand = _mm256_load_ps(&summand[i]);		   // Fast loading of aligned data
            __m256 v_addend = _mm256_loadu_ps(&addend[i]);		   // Slow loading of unaligned data
            __m256 v_result = _mm256_add_ps(v_summand, v_addend);  // Vector addition
            _mm256_store_ps(&summand[i], v_result);			       // Fast saving to aligned memory
        }
    } 
	
	// Case 3: Only the addend is aligned
	else if(aligned_addend) {
        for(; i + 7 < len; i += 8) {
            __m256 v_summand = _mm256_loadu_ps(&summand[i]);	   // Slow loading of unaligned data
            __m256 v_addend = _mm256_load_ps(&addend[i]);		   // Fast loading of aligned data
            __m256 v_result = _mm256_add_ps(v_summand, v_addend);  // Vector addition
            _mm256_storeu_ps(&summand[i], v_result);			   // Slow saving to unaligned memoty
        }
    } 
	
	// Case 4: Both tensors are not aligned
	else {
        for(; i + 7 < len; i += 8) {	
            __m256 v_summand = _mm256_loadu_ps(&summand[i]);	   // Slow loading of unaligned data
            __m256 v_addend = _mm256_loadu_ps(&addend[i]);		   // Slow loading of unaligned data
            __m256 v_result = _mm256_add_ps(v_summand, v_addend);  // Vector addition
            _mm256_storeu_ps(&summand[i], v_result);			   // Slow saving to unaligned memoty
        }
    }
    
	// Processing the remainder
    for(; i < len; i++) summand[i] += addend[i];
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
}

/** @brief 
 * Implementation of int32 addition with the same types
 *
 ** @see nnl2_avx256_addinplace
 ** @see nnl2_avx_add_float64_same_type
 ** @see nnl2_avx_add_float32_same_type
 **/
static inline void nnl2_avx_add_int32_same_type(int32_t* summand, int32_t* addend, size_t len, bool aligned_summand, bool aligned_addend) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
	
    size_t i = 0;
    
	// Case 1: Both tensors are aligned 
    if(aligned_summand && aligned_addend) {
        for(; i + 7 < len; i += 8) {
            __m256i v_summand = _mm256_load_si256((__m256i*)&summand[i]);  // Fast loading of aligned data
            __m256i v_addend = _mm256_load_si256((__m256i*)&addend[i]);	   // Fast loading of aligned data
            __m256i v_result = _mm256_add_epi32(v_summand, v_addend);	   // Vector addition
            _mm256_store_si256((__m256i*)&summand[i], v_result);		   // Fast saving to aligned memory
        }
    } 
	
	// Case 2: Only the summand is aligned
	else if(aligned_summand) {
        for(; i + 7 < len; i += 8) {
            __m256i v_summand = _mm256_load_si256((__m256i*)&summand[i]);  // Fast loading of aligned data
            __m256i v_addend = _mm256_loadu_si256((__m256i*)&addend[i]);   // Slow loading of unaligned data
            __m256i v_result = _mm256_add_epi32(v_summand, v_addend);	   // Vector addition
            _mm256_store_si256((__m256i*)&summand[i], v_result);		   // Fast saving to aligned memory
        }
    } 
	
	// Case 3: Only the addend is aligned
	else if(aligned_addend) {
        for(; i + 7 < len; i += 8) {
            __m256i v_summand = _mm256_loadu_si256((__m256i*)&summand[i]);  // Slow loading of unaligned data
            __m256i v_addend = _mm256_load_si256((__m256i*)&addend[i]);     // Fast loading of aligned data
            __m256i v_result = _mm256_add_epi32(v_summand, v_addend);	    // Vector addition
            _mm256_storeu_si256((__m256i*)&summand[i], v_result);			// Slow saving to unaligned memoty
        }
    } 
	
	// Case 4: Both tensors are not aligned
	else {
        for(; i + 7 < len; i += 8) {
            __m256i v_summand = _mm256_loadu_si256((__m256i*)&summand[i]);  // Slow loading of unaligned data
            __m256i v_addend = _mm256_loadu_si256((__m256i*)&addend[i]);    // Slow loading of unaligned data
            __m256i v_result = _mm256_add_epi32(v_summand, v_addend);		// Vector addition
            _mm256_storeu_si256((__m256i*)&summand[i], v_result);		    // Slow saving to unaligned memoty
        }
    }
    
	// Processing the remainder
    for(; i < len; i++) summand[i] += addend[i];
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
}

// implementations of auxiliary functions for different types

/** @brief 
 * Implementation of double addition with conversion from other types
 *
 ** @details 
 * Converts addend elements to double before addition
 *
 ** @see nnl2_avx256_addinplace
 **/
static inline void nnl2_avx_add_float64_diff_type(double* summand, const Tensor* addend, size_t len, bool aligned_summand) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
	
	// Calculating the step between elements in bytes (for accessing raw data)
    size_t addend_step = get_dtype_size(addend->dtype);
    char* addend_data = (char*)addend->data;
    
    size_t i = 0;
    
	// Vector processing of 4 elements per iteration
    if(aligned_summand) {
        for(; i + 3 < len; i += 4) {
			// Loading 4 double from summand
            __m256d v_summand = _mm256_load_pd(&summand[i]);
			
			// Conversion and creation of a vector of 4 doubles
			// _mm256_set_pd fills the vector in reverse order (from oldest to youngest)
            __m256d v_addend = _mm256_set_pd(
                nnl2_convert_to_float64(addend_data + (i + 3) * addend_step, addend->dtype),
                nnl2_convert_to_float64(addend_data + (i + 2) * addend_step, addend->dtype),
                nnl2_convert_to_float64(addend_data + (i + 1) * addend_step, addend->dtype),
                nnl2_convert_to_float64(addend_data + (i + 0) * addend_step, addend->dtype)
            );
			
			// Vector addition and saving the result
            __m256d v_result = _mm256_add_pd(v_summand, v_addend);
            _mm256_store_pd(&summand[i], v_result);
        }
    } else {
		// Similarly, but with unaligned memory
		
        for(; i + 3 < len; i += 4) {
            __m256d v_summand = _mm256_loadu_pd(&summand[i]);
            __m256d v_addend = _mm256_set_pd(
                nnl2_convert_to_float64(addend_data + (i + 3) * addend_step, addend->dtype),
                nnl2_convert_to_float64(addend_data + (i + 2) * addend_step, addend->dtype),
                nnl2_convert_to_float64(addend_data + (i + 1) * addend_step, addend->dtype),
                nnl2_convert_to_float64(addend_data + (i + 0) * addend_step, addend->dtype)
            );
			
            __m256d v_result = _mm256_add_pd(v_summand, v_addend);
            _mm256_storeu_pd(&summand[i], v_result);
        }
    }
    
	// Scalar processing of the remaining elements
    for(; i < len; i++) {
        void* addend_elem = addend_data + i * addend_step;
        summand[i] += nnl2_convert_to_float64(addend_elem, addend->dtype);
    }
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
}

/** @brief 
 * Implementation of float addition with conversion from other types
 *
 ** @details 
 * Converts addend elements to float before addition
 *
 ** @see nnl2_avx256_addinplace
 **/
static inline void nnl2_avx_add_float32_diff_type(float* summand, const Tensor* addend, size_t len, bool aligned_summand) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
	
    size_t addend_step = get_dtype_size(addend->dtype);
    char* addend_data = (char*)addend->data;
    
    size_t i = 0;
    
	// Vector processing of 8 elements per iteration
    if(aligned_summand) {
        for(; i + 7 < len; i += 8) {
            __m256 v_summand = _mm256_load_ps(&summand[i]);
			
			// Creating a vector of 8 floats with conversion
			// _mm256_set_ps fills in reverse order
            __m256 v_addend = _mm256_set_ps(
                nnl2_convert_to_float32(addend_data + (i + 7) * addend_step, addend->dtype),
                nnl2_convert_to_float32(addend_data + (i + 6) * addend_step, addend->dtype),
                nnl2_convert_to_float32(addend_data + (i + 5) * addend_step, addend->dtype),
                nnl2_convert_to_float32(addend_data + (i + 4) * addend_step, addend->dtype),
                nnl2_convert_to_float32(addend_data + (i + 3) * addend_step, addend->dtype),
                nnl2_convert_to_float32(addend_data + (i + 2) * addend_step, addend->dtype),
                nnl2_convert_to_float32(addend_data + (i + 1) * addend_step, addend->dtype),
                nnl2_convert_to_float32(addend_data + (i + 0) * addend_step, addend->dtype)
            );
			
            __m256 v_result = _mm256_add_ps(v_summand, v_addend);
            _mm256_store_ps(&summand[i], v_result);
        }
    } else {
		// Similarly for unaligned memory
        for(; i + 7 < len; i += 8) {
            __m256 v_summand = _mm256_loadu_ps(&summand[i]);
            __m256 v_addend = _mm256_set_ps(
                nnl2_convert_to_float32(addend_data + (i + 7) * addend_step, addend->dtype),
                nnl2_convert_to_float32(addend_data + (i + 6) * addend_step, addend->dtype),
                nnl2_convert_to_float32(addend_data + (i + 5) * addend_step, addend->dtype),
                nnl2_convert_to_float32(addend_data + (i + 4) * addend_step, addend->dtype),
                nnl2_convert_to_float32(addend_data + (i + 3) * addend_step, addend->dtype),
                nnl2_convert_to_float32(addend_data + (i + 2) * addend_step, addend->dtype),
                nnl2_convert_to_float32(addend_data + (i + 1) * addend_step, addend->dtype),
                nnl2_convert_to_float32(addend_data + (i + 0) * addend_step, addend->dtype)
            );
			
            __m256 v_result = _mm256_add_ps(v_summand, v_addend);
            _mm256_storeu_ps(&summand[i], v_result);
        }
    }
    
	// Scalar processing of the remainder
    for(; i < len; i++) {
        void* addend_elem = addend_data + i * addend_step;
        summand[i] += nnl2_convert_to_float32(addend_elem, addend->dtype);
    }
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
}	

/** @brief 
 * Implementation of int32 addition with conversion from other types
 *
 ** @details 
 * Converts addend elements to int32 before addition
 *
 ** @see nnl2_avx256_addinplace
 **/
static inline void nnl2_avx_add_int32_diff_type(int32_t* summand, const Tensor* addend, size_t len, bool aligned_summand) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
	
    size_t addend_step = get_dtype_size(addend->dtype);
    char* addend_data = (char*)addend->data;
    
    size_t i = 0;
    
	// Vector processing of 8 elements per iteration
    if(aligned_summand) {
        for(; i + 7 < len; i += 8) {
            __m256i v_summand = _mm256_load_si256((__m256i*)&summand[i]);
			
			// Creating a vector of 8 int32s with conversion
            __m256i v_addend = _mm256_set_epi32(
                nnl2_convert_to_int32(addend_data + (i + 7) * addend_step, addend->dtype),
                nnl2_convert_to_int32(addend_data + (i + 6) * addend_step, addend->dtype),
                nnl2_convert_to_int32(addend_data + (i + 5) * addend_step, addend->dtype),
                nnl2_convert_to_int32(addend_data + (i + 4) * addend_step, addend->dtype),
                nnl2_convert_to_int32(addend_data + (i + 3) * addend_step, addend->dtype),
                nnl2_convert_to_int32(addend_data + (i + 2) * addend_step, addend->dtype),
                nnl2_convert_to_int32(addend_data + (i + 1) * addend_step, addend->dtype),
                nnl2_convert_to_int32(addend_data + (i + 0) * addend_step, addend->dtype)
            );
			
            __m256i v_result = _mm256_add_epi32(v_summand, v_addend);
            _mm256_store_si256((__m256i*)&summand[i], v_result);
        }
    } else {
		// Similarly for unaligned memory
        for(; i + 7 < len; i += 8) {
            __m256i v_summand = _mm256_loadu_si256((__m256i*)&summand[i]);
            __m256i v_addend = _mm256_set_epi32(
                nnl2_convert_to_int32(addend_data + (i + 7) * addend_step, addend->dtype),
                nnl2_convert_to_int32(addend_data + (i + 6) * addend_step, addend->dtype),
                nnl2_convert_to_int32(addend_data + (i + 5) * addend_step, addend->dtype),
                nnl2_convert_to_int32(addend_data + (i + 4) * addend_step, addend->dtype),
                nnl2_convert_to_int32(addend_data + (i + 3) * addend_step, addend->dtype),
                nnl2_convert_to_int32(addend_data + (i + 2) * addend_step, addend->dtype),
                nnl2_convert_to_int32(addend_data + (i + 1) * addend_step, addend->dtype),
                nnl2_convert_to_int32(addend_data + (i + 0) * addend_step, addend->dtype)
            );
			
            __m256i v_result = _mm256_add_epi32(v_summand, v_addend);
            _mm256_storeu_si256((__m256i*)&summand[i], v_result);
        }
    }
    
	// Scalar processing of the remainder
    for(; i < len; i++) {
        void* addend_elem = addend_data + i * addend_step;
        summand[i] += nnl2_convert_to_int32(addend_elem, addend->dtype);
    }
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
}

#endif

#ifdef OPENBLAS_AVAILABLE
    /** @brief
     * BLAS implementation of add in-place operation
	 */
	void nnl2_blas_addinplace(Tensor* summand, Tensor* addend);
#endif

/** @ingroup backend_system
 ** @brief Backend implementations for add in-place
 ** @details
 * Array follows the common backend registration pattern
 * Currently register backends:
 *  - nnl2_naive_addinplace: Basic reference implementation
 *  - nnl2_avx256_addinplace: AVX256 implementation 
 *
 ** @see nnl2_naive_addinplace
 ** @see nnl2_avx256_addinplace
 **/
Implementation addinplace_backends[] = {	
	REGISTER_BACKEND(nnl2_naive_addinplace, nnl2_naive, NAIVE_BACKEND_NAME),
	
	#if defined(__AVX__) && TENSOR_MEM_ALIGNMENT == 32
		REGISTER_BACKEND(nnl2_avx256_addinplace, nnl2_avx256, AVX256_BACKEND_NAME),
	#endif
	
	#ifdef OPENBLAS_AVAILABLE
		REGISTER_BACKEND(nnl2_blas_addinplace, nnl2_blas, BLAS_BACKEND_NAME),
	#endif	
};

#ifdef OPENBLAS_AVAILABLE

/** @brief
 * Suboptimal backend fallback for BLAS implementation
 *
 ** @details
 * Used for data types not supported by BLAS (not FLOAT64/FLOAT32)
 * Lazily initialized on first use
 */
addinplacefn addinplace_blas_suboptimal = NULL;

void nnl2_blas_addinplace(Tensor* summand, Tensor* addend) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	if (addinplace_blas_suboptimal == NULL) {
        addinplace_blas_suboptimal = ((Implementation*)nnl2_get_suboptimal_backend(addinplace_backends, 3, BLAS_BACKEND_NAME))->fn;
    }
	
	TensorType dtype_summand = summand->dtype;
	
	if(dtype_summand == INT32) { 
	    addinplace_blas_suboptimal(summand, addend);
		#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
		return;
	}
	
    TensorType dtype_addend = addend->dtype;
	
	// Additional checks at the maximum safety level
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(summand, "Summand tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(summand->data, "Summand tensor's data is NULL");
        
        NNL2_CHECK_NULL_IF_ERR_RETURN(addend, "Addend tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(addend->data, "Addend tensor's data is NULL");
    #endif
	
	// Calculating the total number of elements in the summand tensor
    size_t len_summand = product(summand->shape, summand->rank);
	
	// If the tensor is empty, exit the function
    if(len_summand == 0) return;
	
	if(dtype_summand == dtype_addend) {
        // Handling case when the tensors have the same type
		
		switch(dtype_summand) {
            case FLOAT64: {
                double* data_summand = (double*)summand->data;
                double* data_addend = (double*)addend->data;
				
				cblas_daxpy(len_summand, 1.0, data_addend, 1, data_summand, 1);
				break;
			}
			
			case FLOAT32: {
                float* data_summand = (float*)summand->data;
                float* data_addend = (float*)addend->data;
				
				cblas_saxpy(len_summand, 1.0, data_addend, 1, data_summand, 1);
				break;
			}
			
			default: {
				NNL2_TYPE_ERROR(dtype_summand);
				return;
			}
		}
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}
#endif

/**
 * @brief Function pointer for add in-place
 * @ingroup backend_systsem 
 */
addinplacefn addinplace;

/** 
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(addinplace);

/** 
 * @brief Sets the backend for add in-place
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void set_addinplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(addinplace_backends, addinplace, backend_name, current_backend(addinplace));
}

/** 
 * @brief Gets the name of the active backend for add in-place
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* get_addinplace_backend() {
	return current_backend(addinplace);
}

/** 
 * @brief Function declaration for getting all `addinplace` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(addinplace);

/**
 * @brief Function declaration for getting the number of all `dgemminplace` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(addinplace);

#endif /** NNL2_ADD_INPLACE_H **/
