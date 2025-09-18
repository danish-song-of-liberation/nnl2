#ifndef NNL2_INPLACE_FILL_H
#define NNL2_INPLACE_FILL_H

/** @brief
 * Fills the tensor with the specified value in-place
 *
 ** @details
 * The function fills all tensor elements with the specified value.
 * The operation is performed directly in the tensor memory without creating copies.
 * INT32, FLOAT32, and FLOAT64 data types are supported.
 *
 ** @param tensor
 * Pointer to the tensor structure for filling
 *
 ** @param value 
 * Pointer to a value to be filled (the type must match the dtype)
 *
 ** @param dtype
 * The data type of the tensor value and elements
 *
 ** @example
 * // Filling a tensor with integers
 * int32_t fill_value = 42;
 * naive_inplace_fill(tensor, &fill_value, INT32);
 * 
 * // Filling a tensor with floating-point numbers
 * float float_value = 3.14f;
 * naive_inplace_fill(tensor, &float_value, FLOAT32);
 *
 ** @note
 * The function will return at the very beginning if the tensor is empty without doing anything
 *
 ** @note
 * In the NNL2_SAFETY_MODE_OFF safety mode, a regular pointer is used,
 * while in other modes, a volatile pointer is used to prevent compiler optimizations
 *
 ** @return
 * Returns true (1) if the function is successful, false (0) if it is unsuccessful
 *	
 ** @exception NNL2Error
 * Throws error if tensor pointer is NULL (only in NNL2_SAFETY_MODE_MAX)
 *
 ** @exception NNL2Error  
 * Throws error if tensor data pointer is NULL (only in NNL2_SAFETY_MODE_MAX)
 *
 ** @exception NNL2Error
 * Throws error if value pointer is NULL (only in NNL2_SAFETY_MODE_MAX)
 *
 ** @exception NNL2Error
 * Throws error if unsupported data type is specified
 *
 */
bool nnl2_naive_inplace_fill(Tensor* tensor, void* value, TensorType dtype) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Validate input parameters in maximum safety mode
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		if (!tensor || !tensor->data || !value) {
			#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
				NNL2_FUNC_EXIT();
			#endif
		
			NNL2_ERROR("Incorrect tensor structure");
			return false;
		}
	#endif
	
	// Calculate total number of elements from tensor shape and rank
	size_t total_elems = product(tensor->shape, tensor->rank);	
	if (total_elems == 0) return true; // Early return for empty tensors
	
	bool result = true;
	
	switch(dtype) {
		case INT32: {
			int32_t filler = *(int32_t*)value; // Extract integer fill value
			
			// Cast tensor data to integer pointer
			#if NNL2_SAFETY_MODE == NNL2_SAFETY_MODE_OFF
				int32_t* data = (int32_t*)tensor->data;
			#else
				volatile int32_t* data = (int32_t*)tensor->data;
			#endif 
			
			// Simple scalar loop for INT32 elements
			for(size_t i = 0; i < total_elems; ++i) data[i] = filler;
			break;
		}
		
		case FLOAT32: {
			float filler = *(float*)value; // Extract float fill value
			
			// Cast tensor data to float pointer
			#if NNL2_SAFETY_MODE == NNL2_SAFETY_MODE_OFF
				float* data = (float*)tensor->data;
			#else
				volatile float* data = (float*)tensor->data;
			#endif
			
			// Simple scalar loop for FLOAT32 elements
			for(size_t i = 0; i < total_elems; ++i) data[i] = filler;
			break;
		}
		
		case FLOAT64: {
			double filler = *(double*)value; // Extract double fill value
			
			// Cast tensor data to double pointer
			#if NNL2_SAFETY_MODE == NNL2_SAFETY_MODE_OFF
				double* data = (double*)tensor->data;
			#else 
				volatile double* data = (double*)tensor->data;
			#endif 
			
			// Simple scalar loop for FLOAT64 elements
			for(size_t i = 0; i < total_elems; ++i) data[i] = filler;
			break;
		}
		
		default: {
			NNL2_TYPE_ERROR(dtype); // Error: unsupported data type
			result = false;
			break;
		}
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT(); 
	#endif
	
	return result;
}

/** @brief
 * Fills the tensor with the specified value in-place using loop unrolling optimization
 *
 ** @details
 * The function fills all tensor elements with the specified value using loop unrolling
 * technique for improved performance. The operation is performed directly in the tensor
 * memory without creating copies. INT32, FLOAT32, and FLOAT64 data types are supported
 *
 * The function uses different unroll factors for different data types:
 * INT32 - 4 elements per iteration (32 bytes)
 * FLOAT32 - 4 elements per iteration (32 bytes)  
 * FLOAT64 - 8 elements per iteration (64 bytes)
 *
 ** @param tensor
 * Pointer to the tensor structure for filling
 *
 ** @param value 
 * Pointer to a value to be filled (the type must match the dtype)]
 *
 ** @param dtype
 * The data type of the tensor value and elements
 *
 ** @note
 * In the NNL2_SAFETY_MODE_OFF safety mode, a regular pointer is used,
 * while in other modes, a volatile pointer is used to prevent compiler optimizations
 *
 ** @note
 * The function will return at the very beginning if the tensor is empty without doing anything
 *
 ** @return
 * Returns true (1) if the function is successful, false (0) if it is unsuccessful
 *
 ** @example
 * // Filling a tensor with integers using optimized unrolled version
 * int32_t fill_value = 42;
 * unroll128_inplace_fill(tensor, &fill_value, INT32);
 * 
 * // Filling a tensor with floating-point numbers
 * float float_value = 3.14f;
 * unroll128_inplace_fill(tensor, &float_value, FLOAT32);
 *
 ** @exception NNL2Error
 * Throws error if tensor pointer is NULL (only in NNL2_SAFETY_MODE_MAX)
 *
 ** @exception NNL2Error  
 * Throws error if tensor data pointer is NULL (only in NNL2_SAFETY_MODE_MAX)
 *
 ** @exception NNL2Error
 * Throws error if value pointer is NULL (only in NNL2_SAFETY_MODE_MAX)
 *
 ** @exception NNL2Error
 * Throws error if unsupported data type is specifieds
 */
bool nnl2_unroll_128_inplace_fill(Tensor* tensor, void* value, TensorType dtype) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		if (!tensor || !tensor->data || !value) {
			#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
				NNL2_FUNC_EXIT();
			#endif
			
			NNL2_ERROR("Incorrect tensor structure");
			return false;
		}
	#endif
	
	// Calculating the total number of elements in a tensor
	size_t total_elems = product(tensor->shape, tensor->rank);	
	if (total_elems == 0) return true; // Exit if the tensor is empty
	
	bool result = true;
	
	switch(dtype) {
		case INT32: {
			int32_t filler = *(int32_t*)value; // Extracting the value to fill in
			
			#if NNL2_SAFETY_MODE == NNL2_SAFETY_MODE_OFF
				int32_t* data = (int32_t*)tensor->data;	
			#else 
				volatile int32_t* data = (int32_t*)tensor->data;	
			#endif
			
			size_t i = 0;
			size_t unroll_factor = 4;
			size_t main_elems = total_elems & ~(unroll_factor - 1);
			
			// Main loop with scanning for int32 (4 elements per iteration)
			for(; i < main_elems; i += unroll_factor) {
				data[i] = filler; data[i + 1] = filler; data[i + 2] = filler; data[i + 3] = filler;
			}
			
			// Post-processing of the remaining elements
			for (; i < total_elems; i++) {
                data[i] = filler;
            }
			
			break;
		}
		
		case FLOAT32: {
			float filler = *(float*)value; // Extracting the value to fill in
			
			#if NNL2_SAFETY_MODE == NNL2_SAFETY_MODE_OFF
				float* data = (float*)tensor->data;	
			#else 
				volatile float* data = (float*)tensor->data;	
			#endif
			
			size_t i = 0;
			size_t unroll_factor = 4;
			size_t main_elems = total_elems & ~(unroll_factor - 1);
			
			// Main loop with scanning for float32 (4 elements per iteration)
			for(; i < main_elems; i += unroll_factor) {
				data[i] = filler; data[i + 1] = filler; data[i + 2] = filler; data[i + 3] = filler;
			}
			
			// Post-processing of the remaining elements
			for (; i < total_elems; i++) {
                data[i] = filler;
            }
			
			break;
		}
		
		case FLOAT64: {
			double filler = *(double*)value; // Extracting the value to fill in
			
			#if NNL2_SAFETY_MODE == NNL2_SAFETY_MODE_OFF
				double* data = (double*)tensor->data;	
			#else 
				volatile double* data = (double*)tensor->data;	
			#endif
			
			size_t i = 0;
			size_t unroll_factor = 8;
			size_t main_elems = total_elems & ~(unroll_factor - 1);
			
			// Main loop with scanning for float64 (8 elements per iteration)
			for(; i < main_elems; i += unroll_factor) {
				data[i] = filler; data[i + 1] = filler; data[i + 2] = filler; data[i + 3] = filler;
                data[i + 4] = filler; data[i + 5] = filler; data[i + 6] = filler; data[i + 7] = filler;
			}
			
			// Post-processing of the remaining elements
			for (; i < total_elems; i++) {
                data[i] = filler;
            }
			
			break;
		}
		
		default: {
			NNL2_TYPE_ERROR(dtype); // Unsupported data type error
			result = false;
			break;
		}
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
	
	return result;
}

/** @brief
 * Fills the tensor with the specified value in-place using 256-bit optimized loop unrolling
 *
 ** @details
 * The function fills all tensor elements with the specified value using loop unrolling
 * technique for improved performance. The operation is performed directly in the tensor
 * memory without creating copies. INT32, FLOAT32, and FLOAT64 data types are supported
 *
 * The function uses different unroll factors for different data types:
 * INT32 - 8 elements per iteration (32 bytes)
 * FLOAT32 - 8 elements per iteration (32 bytes)  
 * FLOAT64 - 16 elements per iteration (64 bytes)
 *
 ** @param tensor
 * Pointer to the tensor structure for filling
 *
 ** @param value 
 * Pointer to a value to be filled (the type must match the dtype)
 *
 ** @param dtype
 * The data type of the tensor value and elements
 *
 ** @note
 * In the NNL2_SAFETY_MODE_OFF safety mode, a regular pointer is used,
 * while in other modes, a volatile pointer is used to prevent compiler optimizations
 *
 ** @note
 * The function will return at the very beginning if the tensor is empty without doing anything
 *
 ** @return
 * Returns true (1) if the function is successful, false (0) if it is unsuccessful
 *
 ** @example
 * // Filling a tensor with integers using 256-bit optimized version
 * int32_t fill_value = 42;
 * unroll_256_inplace_fill(tensor, &fill_value, INT32);
 * 
 * // Filling a tensor with floating-point numbers
 * float float_value = 3.14f;
 * unroll_256_inplace_fill(tensor, &float_value, FLOAT32);
 *
 * // Filling a tensor with double precision numbers
 * double double_value = 2.71828;
 * unroll_256_inplace_fill(tensor, &double_value, FLOAT64);
 *
 ** @exception NNL2Error
 * Throws error if tensor pointer is NULL (only in NNL2_SAFETY_MODE_MAX)
 *
 ** @exception NNL2Error  
 * Throws error if tensor data pointer is NULL (only in NNL2_SAFETY_MODE_MAX)
 *
 ** @exception NNL2Error
 * Throws error if value pointer is NULL (only in NNL2_SAFETY_MODE_MAX)
 *
 ** @exception NNL2Error
 * Throws error if unsupported data type is specified
 *
 */
bool nnl2_unroll_256_inplace_fill(Tensor* tensor, void* value, TensorType dtype) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		if (!tensor || !tensor->data || !value) {
			#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
				NNL2_FUNC_EXIT();
			#endif
			
			NNL2_ERROR("Incorrect tensor structure");
			return false;
		}
	#endif
	
	// Calculating the total number of elements in a tensor
	size_t total_elems = product(tensor->shape, tensor->rank);	
	if (total_elems == 0) return true; // Exit if the tensor is empty
	
	bool result = true;
	
	switch(dtype) {
		case INT32: { 
			int32_t filler = *(int32_t*)value; // Extracting the value to fill in
			
			#if NNL2_SAFETY_MODE == NNL2_SAFETY_MODE_OFF
				int32_t* data = (int32_t*)tensor->data;	
			#else 
				volatile int32_t* data = (int32_t*)tensor->data;	
			#endif
			
			size_t i = 0;
			size_t unroll_factor = 8;
			size_t main_elems = total_elems & ~(unroll_factor - 1);
			
			// Main loop with scanning for int32 (8 elements per iteration)
			for(; i < main_elems; i += unroll_factor) {
				data[i] = filler; data[i + 1] = filler; data[i + 2] = filler; data[i + 3] = filler;
				data[i + 4] = filler; data[i + 5] = filler; data[i + 6] = filler; data[i + 7] = filler;
			}
			
			// Post-processing of the remaining elements
			for (; i < total_elems; i++) {
                data[i] = filler;
            }
			
			break;
		}
		
		case FLOAT32: {
			float filler = *(float*)value; // Extracting the value to fill in
			
			#if NNL2_SAFETY_MODE == NNL2_SAFETY_MODE_OFF
				float* data = (float*)tensor->data;	
			#else 
				volatile float* data = (float*)tensor->data;	
			#endif
			
			size_t i = 0;
			size_t unroll_factor = 8;
			size_t main_elems = total_elems & ~(unroll_factor - 1);
			
			// Main loop with scanning for float32 (8 elements per iteration)
			for(; i < main_elems; i += unroll_factor) {
				data[i] = filler; data[i + 1] = filler;	data[i + 2] = filler; data[i + 3] = filler;
				data[i + 4] = filler; data[i + 5] = filler; data[i + 6] = filler; data[i + 7] = filler;
			}
			
			// Post-processing of the remaining elements
			for (; i < total_elems; i++) {
                data[i] = filler;
            }
			
			break;
		}
		
		case FLOAT64: {
			double filler = *(double*)value; // Extracting the value to fill in
			
			#if NNL2_SAFETY_MODE == NNL2_SAFETY_MODE_OFF
				double* data = (double*)tensor->data;	
			#else 
				volatile double* data = (double*)tensor->data;	
			#endif
			
			size_t i = 0;
			size_t unroll_factor = 16;
			size_t main_elems = total_elems & ~(unroll_factor - 1);
			
			// Main loop with scanning for float64 (16 elements per iteration)
			for(; i < main_elems; i += unroll_factor) {
				data[i] = filler; data[i + 1] = filler; data[i + 2] = filler; data[i + 3] = filler;
				data[i + 4] = filler; data[i + 5] = filler; data[i + 6] = filler; data[i + 7] = filler;
				data[i + 8] = filler; data[i + 9] = filler; data[i + 10] = filler; data[i + 11] = filler;
				data[i + 12] = filler; data[i + 13] = filler; data[i + 14] = filler; data[i + 15] = filler;
			}
			
			// Post-processing of the remaining elements
			for (; i < total_elems; i++) {
                data[i] = filler; // Unsupported data type error
            }
			
			break;
		}
		
		default: {
			NNL2_TYPE_ERROR(dtype);
			result = false;
			break;
		}
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
	
	return result;
}

/** @brief
 * Fills the tensor with the specified value in-place using 512-bit optimized loop unrolling
 *
 ** @details
 * The function fills all tensor elements with the specified value using loop unrolling
 * technique for improved performance. The operation is performed directly in the tensor
 * memory without creating copies. INT32, FLOAT32, and FLOAT64 data types are supported
 *
 * The function uses different unroll factors for different data types:
 * INT32 - 16 elements per iteration (32 bytes)
 * FLOAT32 - 16 elements per iteration (32 bytes)  
 * FLOAT64 - 32 elements per iteration (64 bytes)
 *
 ** @param tensor
 * Pointer to the tensor structure for filling
 *
 ** @param value 
 * Pointer to a value to be filled (the type must match the dtype)
 *
 ** @param dtype
 * The data type of the tensor value and elements
 *
 ** @return
 * Returns true (1) if the function is successful, false (0) if it is unsuccessful
 *
 ** @note
 * In the NNL2_SAFETY_MODE_OFF safety mode, a regular pointer is used for maximum performance,
 * while in other modes, a volatile pointer is used to prevent compiler optimizations
 *
 ** @example
 * // Filling a tensor with integers using 512-bit optimized version
 * int32_t fill_value = 42;
 * unroll_512_inplace_fill(tensor, &fill_value, INT32);
 * 
 * // Filling a tensor with floating-point numbers
 * float float_value = 3.14f;
 * unroll_512_inplace_fill(tensor, &float_value, FLOAT32);
 *
 * // Filling a tensor with double precision numbers
 * double double_value = 2.71828;
 * unroll_512_inplace_fill(tensor, &double_value, FLOAT64);
 *
 ** @exception NNL2Error
 * Throws error if tensor pointer is NULL (only in NNL2_SAFETY_MODE_MAX)
 *
 ** @exception NNL2Error  
 * Throws error if tensor data pointer is NULL (only in NNL2_SAFETY_MODE_MAX)
 *
 ** @exception NNL2Error
 * Throws error if value pointer is NULL (only in NNL2_SAFETY_MODE_MAX)
 *
 ** @exception NNL2Error
 * Throws error if unsupported data type is specified
 *
 */
bool nnl2_unroll_512_inplace_fill(Tensor* tensor, void* value, TensorType dtype) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		if (!tensor || !tensor->data || !value) {
			#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
				NNL2_FUNC_EXIT();
			#endif
			
			NNL2_ERROR("Incorrect tensor structure");
			return false;
		}
	#endif
	
	// Calculating the total number of elements in a tensor
	size_t total_elems = product(tensor->shape, tensor->rank);	
	if (total_elems == 0) return true; // Exit if the tensor is empty
	
	bool result = true;
	
	switch(dtype) {
		case INT32: {
			int32_t filler = *(int32_t*)value; // Extracting the value to fill in
			
			#if NNL2_SAFETY_MODE == NNL2_SAFETY_MODE_OFF
				int32_t* data = (int32_t*)tensor->data;	
			#else 
				volatile int32_t* data = (int32_t*)tensor->data;	
			#endif
			
			size_t i = 0;
			size_t unroll_factor = 16;
			size_t main_elems = total_elems & ~(unroll_factor - 1);
			
			// Main loop with scanning for int32 (16 elements per iteration)
			for(; i < main_elems; i += unroll_factor) {
				data[i] = filler; data[i + 1] = filler; data[i + 2] = filler; data[i + 3] = filler;
				data[i + 4] = filler; data[i + 5] = filler; data[i + 6] = filler; data[i + 7] = filler;
				data[i + 8] = filler; data[i + 9] = filler; data[i + 10] = filler; data[i + 11] = filler;
				data[i + 12] = filler; data[i + 13] = filler; data[i + 14] = filler; data[i + 15] = filler;
			}
			
			// Post-processing of the remaining elements
			for (; i < total_elems; i++) {
                data[i] = filler;
            }
			
			break;
		}
		
		case FLOAT32: {
			float filler = *(float*)value; // Extracting the value to fill in
			
			#if NNL2_SAFETY_MODE == NNL2_SAFETY_MODE_OFF
				float* data = (float*)tensor->data;	
			#else 
				volatile float* data = (float*)tensor->data;	
			#endif
			
			size_t i = 0;
			size_t unroll_factor = 16;
			size_t main_elems = total_elems & ~(unroll_factor - 1);
			
			// Main loop with scanning for float32 (16 elements per iteration)
			for(; i < main_elems; i += unroll_factor) {
				data[i] = filler; data[i + 1] = filler; data[i + 2] = filler; data[i + 3] = filler;
				data[i + 4] = filler; data[i + 5] = filler; data[i + 6] = filler; data[i + 7] = filler;
				data[i + 8] = filler; data[i + 9] = filler; data[i + 10] = filler; data[i + 11] = filler;
				data[i + 12] = filler; data[i + 13] = filler; data[i + 14] = filler; data[i + 15] = filler;
			}
			
			// Post-processing of the remaining elements
			for (; i < total_elems; i++) {
                data[i] = filler;
            }
			
			break;
		}
		
		case FLOAT64: {
			double filler = *(double*)value; // Extracting the value to fill in
			
			#if NNL2_SAFETY_MODE == NNL2_SAFETY_MODE_OFF
				double* data = (double*)tensor->data;	
			#else 
				volatile double* data = (double*)tensor->data;	
			#endif
			
			size_t i = 0;
			size_t unroll_factor = 32;
			size_t main_elems = total_elems & ~(unroll_factor - 1);
			
			// Main loop with scanning for float64 (32 elements per iteration)
			for(; i < main_elems; i += unroll_factor) {
				data[i] = filler; data[i + 1] = filler; data[i + 2] = filler; data[i + 3] = filler; 
				data[i + 4] = filler; data[i + 5] = filler; data[i + 6] = filler; data[i + 7] = filler;
				data[i + 8] = filler; data[i + 9] = filler; data[i + 10] = filler; data[i + 11] = filler;
				data[i + 12] = filler; data[i + 13] = filler; data[i + 14] = filler; data[i + 15] = filler;
				data[i + 16] = filler; data[i + 17] = filler; data[i + 18] = filler; data[i + 19] = filler;
				data[i + 20] = filler; data[i + 21] = filler; data[i + 22] = filler; data[i + 23] = filler;
				data[i + 24] = filler; data[i + 25] = filler; data[i + 26] = filler; data[i + 27] = filler;
				data[i + 28] = filler; data[i + 29] = filler; data[i + 30] = filler; data[i + 31] = filler;
			}
			
			// Post-processing of the remaining elements
			for (; i < total_elems; i++) {
                data[i] = filler;
            }
			
			break;
		}
		
		default: {
			NNL2_TYPE_ERROR(dtype); // Unsupported data type error
			result = false;
			break;
		}
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
	
	return result;
}

#ifdef __AVX__
/** @brief 
 * Fills the tensor with the specified value using AVX intrinsics
 *
 ** @details 
 * The function utilizes AVX (Advanced Vector Extensions) instructions to
 * efficiently fill tensor elements with the specified value. This implementation
 * provides significant performance improvements over scalar versions by processing
 * multiple elements simultaneously using 256-bit SIMD registers
 *
 * Memory alignment is automatically detected and appropriate instructions are used:
 * Aligned stores (_mm256_store_*) for 32-byte aligned memory
 *  Unaligned stores (_mm256_storeu_*) for unaligned memory
 *
 ** @param tensor 
 * Pointer to the tensor structure to be filled
 *
 ** @param value
 * Pointer to the fill value (must match tensor data type)
 *
 ** @param dtype 
 * Data type of the tensor elements and fill value
 *
 ** @return
 * Returns true (1) if the function is successful, false (0) if it is unsuccessful
 *
 ** @note 
 * This function requires AVX support and will only be compiled if __AVX__ is defined
 *
 ** @note 
 * For optimal performance, ensure tensor memory is 32-byte aligned
 *
 ** @example
 * // Filling aligned float tensor with AVX256
 * float fill_val = 1.0f;
 * avx_inplace_fill(tensor, &fill_val, FLOAT32);
 *
 ** @exception NNL2Error
 * Throws error if unsupported data type is specified
 *
 */
bool nnl2_avx256_inplace_fill(Tensor* tensor, void* value, TensorType dtype) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		if (tensor == NULL || value == NULL || tensor->data == NULL) {
			#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
				NNL2_FUNC_EXIT();
			#endif
			
			return false; // Invalid parameters
		}
	#endif
	
	// Calculate total elements from tensor shape and rank
	size_t total_elems = product(tensor->shape, tensor->rank);
	if (total_elems == 0) return true; // Exit if the tensor is empty
	
	// Check if tensor data is 32-byte aligned for optimal AVX performance
	bool is_aligned = NNL2_IS_ALIGNED(tensor->data, NNL2_TENSOR_ALIGNMENT_32);
	
	// Warning for unaligned memory in safety modes (performance impact)
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MINI
		if(!is_aligned) {
			NNL2_WARN("In the avx256 implementation of inplace_fill, memory is not aligned to 32 bytes. Calculations may be slightly slower");
		}
	#endif
	
	bool result = true;
	
	switch(dtype) {
		case INT32: {
			int32_t filler = *(int32_t*)value; // Extract scalar fill value
			int32_t* data = (int32_t*)tensor->data; // Cast tensor data to appropriate type
			
			// Create AVX vector with 8 copies of the fill value
			__m256i avx_filler = _mm256_set1_epi32(filler);
				
			size_t avx_iters = total_elems / NNL2_INT32_ELEMENTS_PER_AVX256; // total_elems / 8
			size_t avx_processed_elems = avx_iters * NNL2_INT32_ELEMENTS_PER_AVX256; // avx_iters * 8
			
			if(is_aligned) {
				// Process aligned memory with optimized stores
				for (size_t i = 0; i < avx_iters; i++) { 
					_mm256_store_si256((__m256i*)(data + i * NNL2_INT32_ELEMENTS_PER_AVX256), avx_filler);
				}
			} else {
				// Process unaligned memory with unaligned stores
				for (size_t i = 0; i < avx_iters; i++) { 
					_mm256_storeu_si256((__m256i*)(data + i * NNL2_INT32_ELEMENTS_PER_AVX256), avx_filler);
				}
			}

			// Process remaining elements
			for (size_t j = avx_processed_elems; j < total_elems; j++) {
				data[j] = filler;
			}	
				
			break;
		}
		
		case FLOAT32: {
			float filler = *(float*)value; // Extract scalar fill value
			float* data = (float*)tensor->data; // Cast tensor data to appropriate type
			
			// Create AVX vector with 8 copies of the fill value
			__m256 avx_filler = _mm256_set1_ps(filler);
			
			size_t avx_iters = total_elems / NNL2_FLOAT32_ELEMENTS_PER_AVX256; // total_elems / 8 
			size_t avx_processed_elems = avx_iters * NNL2_FLOAT32_ELEMENTS_PER_AVX256; // avx_iters * 8
			
			if(is_aligned) {
				// Process aligned memory with optimized stores
				for (size_t i = 0; i < avx_iters; i++) {
					_mm256_store_ps(data + i * NNL2_FLOAT32_ELEMENTS_PER_AVX256, avx_filler);
				}
			} else {
				// Process unaligned memory with unaligned stores
				for (size_t i = 0; i < avx_iters; i++) { 
					_mm256_storeu_ps(data + i * NNL2_FLOAT32_ELEMENTS_PER_AVX256, avx_filler);
				}
			}

			// Process remaining elements
			for (size_t j = avx_processed_elems; j < total_elems; j++) {
				data[j] = filler;
			}
			
			break;
		}
		
		case FLOAT64: {
			double filler = *(double*)value; // Extract scalar fill value
			double* data = (double*)tensor->data; // Cast tensor data to appropriate type
			
			// Create AVX vector with 4 copies of the fill value
			__m256d avx_filler = _mm256_set1_pd(filler);
			
			size_t it = 0;
			size_t avx_limit = total_elems - (NNL2_MIN_ELEMENTS_FOR_AVX256_DOUBLE - 1);
			
			if(is_aligned) {
				// Process aligned memory with optimized stores
				for(; it < avx_limit; it += NNL2_FLOAT64_ELEMENTS_PER_AVX256) {
					_mm256_store_pd(data + it, avx_filler);
				}	
			} else {
				// Process unaligned memory with unaligned stores
				for(; it < avx_limit; it += NNL2_FLOAT64_ELEMENTS_PER_AVX256) {
					_mm256_storeu_pd(data + it, avx_filler);
				}
			}
			
			// Process remaining elements
			for(size_t j = it; j < total_elems; j++) {
				data[j] = filler;
			}
			
			break;
		}
		
		default: {
			NNL2_TYPE_ERROR(dtype); // Unsupported data type error
			result = false;
			break;
		}
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
	
	return result;
}
#endif

/** @ingroup backend_system
 ** @brief Backend implementations for inplace_fill
 ** @details
 * Array follows the common backend registration pattern
 * Currently register backends:
 *  - nnl2_unroll_128_inplace_fill: 128-bit optimized loop unrolling implementation
 *  - nnl2_unroll_256_inplace_fill: 256-bit optimized loop unrolling implementation  
 *  - nnl2_unroll_512_inplace_fill: 512-bit optimized loop unrolling implementation
 *  - nnl2_naive_inplace_fill: Basic reference implementation
 *  - nnl2_avx256_inplace_fill: AVX-256 SIMD optimized implementation (conditionally compiled)
 *
 ** @see REGISTER_BACKEND
 ** @see UNROLL_128_BACKEND_NAME
 ** @see UNROLL_256_BACKEND_NAME
 ** @see UNROLL_512_BACKEND_NAME
 ** @see AVX512_BACKEND_NAME
 ** @see NAIVE_BACKEND_NAME
 ** @see nnl2_unroll_128_inplace_fill
 ** @see nnl2_unroll_256_inplace_fill
 ** @see nnl2_unroll_512_inplace_fill
 ** @see nnl2_avx_256_inplace_fill
 ** @see nnl2_naive_inplace_fillss
 ** @see nnl2_unroll_128
 ** @see nnl2_unroll_256
 ** @see nnl2_unroll_512
 ** @see nnl2_avx256
 ** @see nnl2_naive
 **/
Implementation inplace_fill_backends[] = {
	REGISTER_BACKEND(nnl2_unroll_128_inplace_fill, nnl2_unroll_128, UNROLL_128_BACKEND_NAME),	
	REGISTER_BACKEND(nnl2_unroll_256_inplace_fill, nnl2_unroll_256, UNROLL_256_BACKEND_NAME),
	REGISTER_BACKEND(nnl2_unroll_512_inplace_fill, nnl2_unroll_512, UNROLL_512_BACKEND_NAME),
	REGISTER_BACKEND(nnl2_naive_inplace_fill, nnl2_naive, NAIVE_BACKEND_NAME),
	
	#ifdef __AVX__
		#if TENSOR_MEM_ALIGNMENT == 32
			REGISTER_BACKEND(nnl2_avx256_inplace_fill, nnl2_avx256, AVX256_BACKEND_NAME),
		#endif
	#endif
};

/**
 * @brief Function pointer for inplace_fill 
 * @ingroup backend_system 
 */
fn_inplace_fill inplace_fill;

/** 
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(inplace_fill);

/** 
 * @brief Sets the backend for inplace_fill
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void set_inplace_fill_backend(const char* backend_name) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
	    NNL2_FUNC_ENTER();
		NNL2_DEBUG("Changed backend for inplace_fill from %s to %s", CURRENT_BACKEND(inplace_fill), backend_name);	
	#endif
	
    ESET_BACKEND_BY_NAME(inplace_fill_backends, inplace_fill, backend_name, CURRENT_BACKEND(inplace_fill));
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
		NNL2_FUNC_EXIT();
	#endif
}

/** 
 * @brief Gets the name of the active backend for inplace_all
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* get_inplace_fill_backend() {
	return current_backend(inplace_fill);
}

/** 
 * @brief Function declaration for getting all `inplace_fill` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(inplace_fill);

/**
 * @brief Function declaration for getting the number of all `inplace_fill` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(inplace_fill);

#endif /** NNL2_INPLACE_FILL_H **/
