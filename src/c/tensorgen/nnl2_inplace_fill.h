#ifndef NNL2_INPLACE_FILL_H
#define NNL2_INPLACE_FILL_H

///@{
	
/** @brief
 * Threshold for enabling parallel execution 
 * of the inplace_fill operation
 */
#define NNL2_INPLACE_FILL_THREASHOLD 50000

///@}

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
	size_t total_elems = nnl2_product(tensor->shape, tensor->rank);	
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
	size_t total_elems = nnl2_product(tensor->shape, tensor->rank);	
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
	size_t total_elems = nnl2_product(tensor->shape, tensor->rank);	
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
	size_t total_elems = nnl2_product(tensor->shape, tensor->rank);	
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
	size_t total_elems = nnl2_product(tensor->shape, tensor->rank);
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
			double filler = *(double*)value;
			double* data = (double*)tensor->data;
    
			__m256d avx_filler = _mm256_set1_pd(filler);
   
			size_t avx_iters = total_elems / NNL2_FLOAT64_ELEMENTS_PER_AVX256;  // total_elems / 4
			size_t avx_processed_elems = avx_iters * NNL2_FLOAT64_ELEMENTS_PER_AVX256;  // avx_iters * 4
    
			if(is_aligned) {
				for (size_t i = 0; i < avx_iters; i++) {
					_mm256_store_pd(data + i * NNL2_FLOAT64_ELEMENTS_PER_AVX256, avx_filler);
				}
			} else {
				for (size_t i = 0; i < avx_iters; i++) {
					_mm256_storeu_pd(data + i * NNL2_FLOAT64_ELEMENTS_PER_AVX256, avx_filler);
				}
			}
    
			// Process remaining elements
			for (size_t j = avx_processed_elems; j < total_elems; j++) {
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



#ifdef NNL2_PTHREAD_AVAILABLE

/** @brief 
 * Fill tensor of type float64
 * 
 ** @param data 
 * Pointer to tensor data
 *
 ** @param total_size 
 * Total number of elements
 *
 ** @param value 
 * Fill value
 *
 ** @param num_threads 
 * Number of threads
 *
 * @param aligned 
 * Memory alignment flag
 *
 ** @return 
 * true on success
 */
bool nnl2_own_inplace_fill_float64(double* data, size_t total_size, double value, size_t num_threads, bool aligned);

/** @brief
 * Docs similiary at nnl2_own_inplace_fill_float64 doxygen
 * See them
 *
 ** @see nnl2_own_inplace_fill_float64 (declaration)
 **/
bool nnl2_own_inplace_fill_float32(float* data, size_t total_size, float value, size_t num_threads, bool aligned);

/** @brief
 * Docs similiary at nnl2_own_inplace_fill_float64 doxygen
 * See them
 *
 ** @see nnl2_own_inplace_fill_float64 (declaration)
 **/
bool nnl2_own_inplace_fill_int32(int32_t* data, size_t total_size, int32_t value, size_t num_threads, bool aligned);

/** @brief 
 * Worker function for parallel float64 fill
 * 
 ** @param arg 
 * Pointer to fill_ptask structure
 *
 ** @return 
 * NULL (for pthread.h api)
 * 
 ** @details
 * Uses AVX256 instructions for vectorized filling of 8 elements
 * per operation when AVX is available and chunk size is sufficient
 */
void* nnl2_own_pfill_float64(void* thread);

/** @brief
 * Docs similiary at nnl2_own_pfill_float64 doxygen
 * See them
 *
 ** @see nnl2_own_pfill_float64 (declaration)
 **/
void* nnl2_own_pfill_float32(void* thread);

/** @brief
 * Docs similiary at nnl2_own_pfill_float64 doxygen
 * See them
 *
 ** @see nnl2_own_pfill_float64 (declaration)
 **/
void* nnl2_own_pfill_int32(void* thread); 

/** @brief
 * Own nnl2 inplace_fill implementation
 *
 ** @details
 * Combines AVX256 SIMD optimization with multi-threading for maximum performance
 * on large tensors. Automatically detects memory alignment and uses appropriate
 * AVX instructions
 *
 ** @param tensor
 * Pointer to the tensor structure for filling
 *
 ** @param value 
 * Pointer to a value to be filled
 *
 ** @param dtype
 * The data type of the tensor value and elements
 *
 ** @return
 * Returns true (1) if the function is successful, false (0) if it is unsuccessful
 */
bool nnl2_own_inplace_fill(Tensor* tensor, void* value, TensorType dtype) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        if (!tensor || !tensor->data || !value) {
            NNL2_ERROR("Incorrect tensor structure");
            return false;
        }
    #endif
    
    size_t total_elems = nnl2_product(tensor->shape, tensor->rank);    
    if (total_elems == 0) return true; // If tensor empty exiting
    
    // Use naive implementation for small tensors
    if (total_elems < NNL2_INPLACE_FILL_THREASHOLD) {
        return nnl2_naive_inplace_fill(tensor, value, dtype);
    }
    
    bool is_aligned = NNL2_IS_ALIGNED(tensor->data, NNL2_TENSOR_ALIGNMENT_32);
    size_t num_threads = NNL2_NUM_THREADS;
    
    // Adjust thread count for small workloads
    if (total_elems < num_threads * NNL2_MIN_ELEMS_PER_THREAD) {
        num_threads = (total_elems + (NNL2_MIN_ELEMS_PER_THREAD - 1)) / NNL2_MIN_ELEMS_PER_THREAD;
        if (num_threads < NNL2_MIN_THREADS) num_threads = NNL2_MIN_THREADS;
    }
    
    bool result = true;
    
    switch(dtype) {
        case FLOAT64:  result = nnl2_own_inplace_fill_float64((double*)tensor->data, total_elems, *(double*)value, num_threads, is_aligned);  break;         
        case FLOAT32:  result = nnl2_own_inplace_fill_float32((float*)tensor->data, total_elems, *(float*)value, num_threads, is_aligned);    break;
        case INT32:    result = nnl2_own_inplace_fill_int32((int32_t*)tensor->data, total_elems, *(int32_t*)value, num_threads, is_aligned);  break; 
        
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
 * See docs at declaration
 *
 ** @see nnl2_own_inplace_fill_float64 (declaration)
 **/
bool nnl2_own_inplace_fill_float64(double* data, size_t total_size, double value, size_t num_threads, bool aligned) {
    pthread_t threads[num_threads];
    fill_ptask tasks[num_threads];
    
    // Calculate chunk size with cache-friendly alignment
    size_t cache_line_elements = NNL2_CACHE_LINE_SIZE / sizeof(double);
    size_t base_chunk = total_size / num_threads;
    
    // Align chunks to cache line boundaries
    size_t aligned_chunk = (base_chunk + cache_line_elements - 1) / cache_line_elements;
    aligned_chunk *= cache_line_elements;
    
    // Ensure minimum chunk size for vectorization
    if (aligned_chunk < cache_line_elements * 4) {
        aligned_chunk = cache_line_elements * 4;
    }
    
    // Recalculate thread distribution with aligned chunks
    size_t full_chunks = total_size / aligned_chunk;
    size_t remainder = total_size % aligned_chunk;
    
    // Adjust thread count if we have fewer chunks than threads
    size_t actual_threads = (full_chunks + (remainder > 0 ? 1 : 0));
    if (actual_threads < num_threads) {
        num_threads = actual_threads;
    }
    
    size_t current_start = 0;
    for (size_t i = 0; i < num_threads; i++) {
        size_t current_chunk = (i < full_chunks) ? aligned_chunk : remainder;
        if (current_chunk == 0) break;
        
        // Initialize task parameters for this thread
        tasks[i].data = data;
        tasks[i].start = current_start;
        tasks[i].end = current_start + current_chunk;
        tasks[i].value = &value;
        tasks[i].dtype = FLOAT64;
        tasks[i].aligned = aligned;
        
        // Create worker thread
        int status = pthread_create(&threads[i], NULL, nnl2_own_pfill_float64, &tasks[i]);
        if(status != 0) {
            NNL2_THREAD_CREATE_ERROR(status, "nnl2_own_inplace_fill_float64");
            num_threads = i;  // Adjust thread count if creation failed
            break;
        }
        
        current_start += current_chunk;
    }
    
    // Wait for all threads to complete
    for (size_t i = 0; i < num_threads; i++) {
        int join_status = pthread_join(threads[i], NULL);
        if(join_status != 0) {
            NNL2_THREAD_JOIN_ERROR(join_status, "nnl2_own_inplace_fill_float64");
        }
    }
    
    return true;
}

/** @brief
 * See docs at declaration
 *
 ** @see nnl2_own_inplace_fill_float32 (declaration)
 **/
bool nnl2_own_inplace_fill_float32(float* data, size_t total_size, float value, size_t num_threads, bool aligned) {
    pthread_t threads[num_threads];
    fill_ptask tasks[num_threads];
    
    // Calculate chunk size with cache-friendly alignment
    size_t cache_line_elements = NNL2_CACHE_LINE_SIZE / sizeof(float);
    size_t base_chunk = total_size / num_threads;
    
    // Align chunks to cache line boundaries
    size_t aligned_chunk = (base_chunk + cache_line_elements - 1) / cache_line_elements;
    aligned_chunk *= cache_line_elements;
    
    // Ensure minimum chunk size for vectorization
    if (aligned_chunk < cache_line_elements * 4) {
        aligned_chunk = cache_line_elements * 4;
    }
    
    // Recalculate thread distribution with aligned chunks
    size_t full_chunks = total_size / aligned_chunk;
    size_t remainder = total_size % aligned_chunk;
    
    // Adjust thread count if we have fewer chunks than threads
    size_t actual_threads = (full_chunks + (remainder > 0 ? 1 : 0));
    if (actual_threads < num_threads) {
        num_threads = actual_threads;
    }
    
    size_t current_start = 0;
    for (size_t i = 0; i < num_threads; i++) {
        size_t current_chunk = (i < full_chunks) ? aligned_chunk : remainder;
        if (current_chunk == 0) break;
        
        // Initialize task parameters for this thread
        tasks[i].data = data;
        tasks[i].start = current_start;
        tasks[i].end = current_start + current_chunk;
        tasks[i].value = &value;
        tasks[i].dtype = FLOAT32;
        tasks[i].aligned = aligned;
        
        // Create worker thread
        int status = pthread_create(&threads[i], NULL, nnl2_own_pfill_float32, &tasks[i]);
        if(status != 0) {
            NNL2_THREAD_CREATE_ERROR(status, "nnl2_own_inplace_fill_float32");
            num_threads = i;  // Adjust thread count if creation failed
            break;
        }
        
        current_start += current_chunk;
    }
    
    // Wait for all threads to complete
    for (size_t i = 0; i < num_threads; i++) {
        int join_status = pthread_join(threads[i], NULL);
        if(join_status != 0) {
            NNL2_THREAD_JOIN_ERROR(join_status, "nnl2_own_inplace_fill_float32");
        }
    }
    
    return true;
}

/** @brief
 * See docs at declaration
 *
 ** @see nnl2_own_inplace_fill_int32 (declaration)
 **/
bool nnl2_own_inplace_fill_int32(int32_t* data, size_t total_size, int32_t value, size_t num_threads, bool aligned) {
    pthread_t threads[num_threads];
    fill_ptask tasks[num_threads];
    
    // Calculate chunk size with cache-friendly alignment
    size_t cache_line_elements = NNL2_CACHE_LINE_SIZE / sizeof(int32_t);
    size_t base_chunk = total_size / num_threads;
    
    // Align chunks to cache line boundaries
    size_t aligned_chunk = (base_chunk + cache_line_elements - 1) / cache_line_elements;
    aligned_chunk *= cache_line_elements;
    
    // Ensure minimum chunk size for vectorization
    if (aligned_chunk < cache_line_elements * 4) {
        aligned_chunk = cache_line_elements * 4;
    }
    
    // Recalculate thread distribution with aligned chunks
    size_t full_chunks = total_size / aligned_chunk;
    size_t remainder = total_size % aligned_chunk;
    
    // Adjust thread count if we have fewer chunks than threads
    size_t actual_threads = (full_chunks + (remainder > 0 ? 1 : 0));
    if (actual_threads < num_threads) {
        num_threads = actual_threads;
    }
    
    size_t current_start = 0;
    for (size_t i = 0; i < num_threads; i++) {
        size_t current_chunk = (i < full_chunks) ? aligned_chunk : remainder;
        if (current_chunk == 0) break;
        
        // Initialize task parameters for this thread
        tasks[i].data = data;
        tasks[i].start = current_start;
        tasks[i].end = current_start + current_chunk;
        tasks[i].value = &value;
        tasks[i].dtype = INT32;
        tasks[i].aligned = aligned;
        
        // Create worker thread
        int status = pthread_create(&threads[i], NULL, nnl2_own_pfill_int32, &tasks[i]);
        if(status != 0) {
            NNL2_THREAD_CREATE_ERROR(status, "nnl2_own_inplace_fill_int32");
            num_threads = i;  // Adjust thread count if creation failed
            break;
        }
        
        current_start += current_chunk;
    }
    
    // Wait for all threads to complete
    for (size_t i = 0; i < num_threads; i++) {
        int join_status = pthread_join(threads[i], NULL);
        if(join_status != 0) {
            NNL2_THREAD_JOIN_ERROR(join_status, "nnl2_own_inplace_fill_int32");
        }
    }
    
    return true;
}

/** @brief
 * See docs at declaration
 *
 ** @see nnl2_own_pfill_float64 (declaration)
 **/
void* nnl2_own_pfill_float64(void* arg) {
    fill_ptask* task = (fill_ptask*)arg;
    
    double* data = (double*)task->data;
    double value = *(double*)task->value;
    
    size_t start = task->start;
    size_t end = task->end;
    size_t chunk_size = end - start;
    
    #ifdef NNL2_AVX256_AVAILABLE
		if (chunk_size >= 4) {
			__m256d avx_value = _mm256_set1_pd(value);
			size_t avx_iters = chunk_size / 4;
			size_t avx_processed = avx_iters * 4;
			
			// Adaptive prefetching based on chunk size
			size_t prefetch_distance = NNL2_PREFETCH_DISTANCE;
			
			if (prefetch_distance > avx_iters / 2) {
				prefetch_distance = avx_iters / 4;
			}
			
			if (prefetch_distance < 4) {
				prefetch_distance = 4;
			}
			
			if (task->aligned) {
				for (size_t i = 0; i < avx_iters; i++) {
					size_t idx = start + i * 4;
					
					// Aggressive prefetching for multiple cache lines ahead
					if (i + prefetch_distance < avx_iters) {
						size_t prefetch_idx = idx + prefetch_distance * 4;
						// Prefetch multiple cache lines
						for (int j = 0; j < NNL2_CACHE_LINES_AHEAD; j++) {
							__builtin_prefetch(data + prefetch_idx + j * (NNL2_CACHE_LINE_SIZE / sizeof(double)), 1, 3);
						}
					}
					
					_mm256_store_pd(data + idx, avx_value);
				}
			} else {
				for (size_t i = 0; i < avx_iters; i++) {
					size_t idx = start + i * 4;
					
					if (i + prefetch_distance < avx_iters) {
						size_t prefetch_idx = idx + prefetch_distance * 4;
						for (int j = 0; j < NNL2_CACHE_LINES_AHEAD; j++) {
							__builtin_prefetch(data + prefetch_idx + j * (NNL2_CACHE_LINE_SIZE / sizeof(double)), 1, 3);
						}
					}
					
					_mm256_storeu_pd(data + idx, avx_value);
				}
			}
			
			size_t remaining_start = start + avx_processed;
			if (remaining_start < end) {
				size_t remaining_size = end - remaining_start;
				
				// Process remainder in small blocks for better cache locality
				const size_t block_size = 2; // Process 2 elements at a time
				size_t block_iters = remaining_size / block_size;
				size_t block_processed = block_iters * block_size;
				
				for (size_t i = 0; i < block_iters; i++) {
					size_t idx = remaining_start + i * block_size;
					data[idx] = value;
					data[idx + 1] = value;
				}
				
				// Process final single elements
				for (size_t i = remaining_start + block_processed; i < end; i++) {
					data[i] = value;
				}
			}
		} else {
			// Small chunk - use simple sequential processing
			for (size_t i = start; i < end; i++) {
				data[i] = value;
			}
		}
    #else
		// Fallback branch - AVX not available
		for (size_t i = start; i < end; i++) {
			data[i] = value;
		}
    #endif
    
    return NULL;
}

/** @brief
 * See docs at declaration
 *
 ** @see nnl2_own_pfill_float32 (declaration)
 **/
void* nnl2_own_pfill_float32(void* arg) {
    fill_ptask* task = (fill_ptask*)arg;
    
    float* data = (float*)task->data;
    float value = *(float*)task->value;
    
    size_t start = task->start;
    size_t end = task->end;
    size_t chunk_size = end - start;
    
    #ifdef NNL2_AVX256_AVAILABLE
		if (chunk_size >= 8) {
			__m256 avx_value = _mm256_set1_ps(value);
			size_t avx_iters = chunk_size / 8;
			size_t avx_processed = avx_iters * 8;
			
			// Adaptive prefetching based on chunk size
			size_t prefetch_distance = NNL2_PREFETCH_DISTANCE;
			if (prefetch_distance > avx_iters / 2) {
				prefetch_distance = avx_iters / 4;
			}
			if (prefetch_distance < 4) {
				prefetch_distance = 4;
			}
			
			if (task->aligned) {
				for (size_t i = 0; i < avx_iters; i++) {
					size_t idx = start + i * 8;
					
					// Aggressive prefetching for multiple cache lines ahead
					if (i + prefetch_distance < avx_iters) {
						size_t prefetch_idx = idx + prefetch_distance * 8;
						// Prefetch multiple cache lines
						for (int j = 0; j < NNL2_CACHE_LINES_AHEAD; j++) {
							__builtin_prefetch(data + prefetch_idx + j * (NNL2_CACHE_LINE_SIZE / sizeof(float)), 1, 3);
						}
					}
					
					_mm256_store_ps(data + idx, avx_value);
				}
			} else {
				for (size_t i = 0; i < avx_iters; i++) {
					size_t idx = start + i * 8;
					
					if (i + prefetch_distance < avx_iters) {
						size_t prefetch_idx = idx + prefetch_distance * 8;
						for (int j = 0; j < NNL2_CACHE_LINES_AHEAD; j++) {
							__builtin_prefetch(data + prefetch_idx + j * (NNL2_CACHE_LINE_SIZE / sizeof(float)), 1, 3);
						}
					}
					
					_mm256_storeu_ps(data + idx, avx_value);
				}
			}
			
			size_t remaining_start = start + avx_processed;
			if (remaining_start < end) {
				size_t remaining_size = end - remaining_start;
				
				// Process remainder in small blocks for better cache locality
				const size_t block_size = 4; // Process 4 elements at a time
				size_t block_iters = remaining_size / block_size;
				size_t block_processed = block_iters * block_size;
				
				for (size_t i = 0; i < block_iters; i++) {
					size_t idx = remaining_start + i * block_size;
					data[idx] = value;
					data[idx + 1] = value;
					data[idx + 2] = value;
					data[idx + 3] = value;
				}
				
				// Process final elements
				for (size_t i = remaining_start + block_processed; i < end; i++) {
					data[i] = value;
				}
			}
		} else {
			// Small chunk - use simple sequential processing
			for (size_t i = start; i < end; i++) {
				data[i] = value;
			}
		}
    #else
		// Fallback branch - AVX not available
		for (size_t i = start; i < end; i++) {
			data[i] = value;
		}
    #endif
    
    return NULL;
}

/** @brief
 * See docs at declaration
 *
 ** @see nnl2_own_pfill_int32 (declaration)
 **/
void* nnl2_own_pfill_int32(void* arg) {
    fill_ptask* task = (fill_ptask*)arg;
    
    int32_t* data = (int32_t*)task->data;
    int32_t value = *(int32_t*)task->value;
    
    size_t start = task->start;
    size_t end = task->end;
    size_t chunk_size = end - start;
    
    #ifdef NNL2_AVX256_AVAILABLE
		if (chunk_size >= 8) {
			__m256i avx_value = _mm256_set1_epi32(value);
			size_t avx_iters = chunk_size / 8;
			size_t avx_processed = avx_iters * 8;
			
			// Adaptive prefetching based on chunk size
			size_t prefetch_distance = NNL2_PREFETCH_DISTANCE;
			if (prefetch_distance > avx_iters / 2) {
				prefetch_distance = avx_iters / 4;
			}
			if (prefetch_distance < 4) {
				prefetch_distance = 4;
			}
			
			if (task->aligned) {
				for (size_t i = 0; i < avx_iters; i++) {
					size_t idx = start + i * 8;
					
					// Aggressive prefetching for multiple cache lines ahead
					if (i + prefetch_distance < avx_iters) {
						size_t prefetch_idx = idx + prefetch_distance * 8;
						// Prefetch multiple cache lines
						for (int j = 0; j < NNL2_CACHE_LINES_AHEAD; j++) {
							__builtin_prefetch(data + prefetch_idx + j * (NNL2_CACHE_LINE_SIZE / sizeof(int32_t)), 1, 3);
						}
					}
					
					_mm256_store_si256((__m256i*)(data + idx), avx_value);
				}
			} else {
				for (size_t i = 0; i < avx_iters; i++) {
					size_t idx = start + i * 8;
					
					if (i + prefetch_distance < avx_iters) {
						size_t prefetch_idx = idx + prefetch_distance * 8;
						for (int j = 0; j < NNL2_CACHE_LINES_AHEAD; j++) {
							__builtin_prefetch(data + prefetch_idx + j * (NNL2_CACHE_LINE_SIZE / sizeof(int32_t)), 1, 3);
						}
					}
					
					_mm256_storeu_si256((__m256i*)(data + idx), avx_value);
				}
			}
			
			size_t remaining_start = start + avx_processed;
			if (remaining_start < end) {
				size_t remaining_size = end - remaining_start;
				
				// Process remainder in small blocks for better cache locality
				const size_t block_size = 4; // Process 4 elements at a time
				size_t block_iters = remaining_size / block_size;
				size_t block_processed = block_iters * block_size;
				
				for (size_t i = 0; i < block_iters; i++) {
					size_t idx = remaining_start + i * block_size;
					data[idx] = value;
					data[idx + 1] = value;
					data[idx + 2] = value;
					data[idx + 3] = value;
				}
				
				// Process final elements
				for (size_t i = remaining_start + block_processed; i < end; i++) {
					data[i] = value;
				}
			}
		} else {
			// Small chunk - use simple sequential processing
			for (size_t i = start; i < end; i++) {
				data[i] = value;
			}
		}
    #else
		// Fallback branch - AVX not available
		for (size_t i = start; i < end; i++) {
			data[i] = value;
		}
    #endif
    
    return NULL;
}

#endif

// I tried to add blas but it's even slower than naive implementation

/** @ingroup backend_system
 ** @brief Backend implementations for inplace_fill
 ** @details
 * Array follows the common backend registration pattern
 * Currently register backends:
 *  - nnl2_unroll_128_inplace_fill: 128-bit optimized loop unrolling implementation
 *  - nnl2_unroll_256_inplace_fill: 256-bit optimized loop unrolling implementation  
 *  - nnl2_unroll_512_inplace_fill: 512-bit optimized loop unrolling implementation
 *  - nnl2_naive_inplace_fill: Basic reference implementation
 *  - nnl2_avx256_inplace_fill: AVX-256 SIMD optimized implementation 
 *  - nnl2_own_inplace_fill: Own hyper-accelerated nnl2 implementation
 *
 ** @see REGISTER_BACKEND
 ** @see UNROLL_128_BACKEND_NAME
 ** @see UNROLL_256_BACKEND_NAME
 ** @see UNROLL_512_BACKEND_NAME
 ** @see AVX512_BACKEND_NAME
 ** @see NAIVE_BACKEND_NAME
 ** @see NNL2_OWN_NAME
 ** @see nnl2_unroll_128_inplace_fill
 ** @see nnl2_unroll_256_inplace_fill
 ** @see nnl2_unroll_512_inplace_fill
 ** @see nnl2_avx_256_inplace_fill
 ** @see nnl2_naive_inplace_fill
 ** @see nnl2_own_inplace_fill
 ** @see nnl2_unroll_128
 ** @see nnl2_unroll_256
 ** @see nnl2_unroll_512
 ** @see nnl2_avx256
 ** @see nnl2_naive
 ** @see nnl2_own
 **/
Implementation inplace_fill_backends[] = {
	REGISTER_BACKEND(nnl2_unroll_128_inplace_fill, nnl2_unroll_128, UNROLL_128_BACKEND_NAME),	
	REGISTER_BACKEND(nnl2_unroll_256_inplace_fill, nnl2_unroll_256, UNROLL_256_BACKEND_NAME),
	REGISTER_BACKEND(nnl2_unroll_512_inplace_fill, nnl2_unroll_512, UNROLL_512_BACKEND_NAME),
	REGISTER_BACKEND(nnl2_naive_inplace_fill, nnl2_naive, NAIVE_BACKEND_NAME),
	
	#if defined(NNL2_AVX256_AVAILABLE) && TENSOR_MEM_ALIGNMENT == 32
		REGISTER_BACKEND(nnl2_avx256_inplace_fill, nnl2_avx256, AVX256_BACKEND_NAME),
	#endif

	#if defined(NNL2_AVX256_AVAILABLE) && defined(NNL2_PTHREAD_AVAILABLE)
		REGISTER_BACKEND(nnl2_own_inplace_fill, nnl2_own_2, NNL2_OWN_NAME),
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
	return CURRENT_BACKEND(inplace_fill);
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
