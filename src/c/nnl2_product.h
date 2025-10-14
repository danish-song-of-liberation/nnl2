#ifndef NNL2_PRODUCT_H
#define NNL2_PRODUCT_H

/** @brief
 * Calculates the total number of elements in the tensor specified by the shape (for calculating memory)
 *
 ** @param lst
 * Pointer to an array of integers representing the tensor's shape
 *
 ** @param len
 * Length of the array `lst`, which is the number of dimensions in the tensor
 *
 ** @return 
 * Total number of elements in the tensor
 *
 ** @code
 * int shape[] = {2, 3, 4};
 * size_t num_elements = product(shape, 3); // num_elements will be 24
 ** @endcode
 **
 ** @note
 * Uses a forced inline
 *
 ** @note
 * Displays a lot of debug information at the 
 * maximum debug level. If the debug level is 
 * lower than the maximum, nothing is displayed
 *
 ** @note
 * Сan perform additional checks at a high level of safety
 *
 ** @note
 * In debug mode, it can output additional information to the console
 *
 ** @see NNL2_DEBUG_MODE
 ** @see NNL2_SAFETY_MODE
 ** @see NNL2_FORCE_INLINE
 **/
NNL2_FORCE_INLINE static size_t product(const nnl2_int32* lst, int32_t len) { // todo rename from product to nnl2_product
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_ENTER();	
		nnl2_int32 original_len = len;
	#endif
	
	// Additional checks when the debugging level is sufficient 
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		if (lst == NULL) {
			NNL2_ERROR("product(): NULL pointer passed as shape array");
			return 0;
		}
	
		if (len <= 0) {
			NNL2_ERROR("product(): Invalid length %d", len);
			return 0;
		}
	#endif
	
	switch(len) {
		// Unrolling
		case 0: return 0;  
		case 1: return lst[0];
		case 2: return lst[0] * lst[1]; 
		case 3: return lst[0] * lst[1] * lst[2];
		case 4: return lst[0] * lst[1] * lst[2] * lst[3];
		
		default: {
			size_t acc = 1;
			while (len--) {
				#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE 
					if (*lst <= 0) {
						NNL2_ERROR("Invalid dimension value %d", *lst);
						return 0;
					}
				#endif
				
				#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
					if (acc > SIZE_MAX / (size_t)(*lst)) {
						NNL2_ERROR("Multiplication overflow in product()");
					}
				#endif
				
				// Calculation
				acc *= (size_t)*lst++;
			}
			
			#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
				NNL2_DEBUG("Product calculated for shape[%d], result=%zu", original_len, acc);
				NNL2_FUNC_EXIT();
			#endif
			
			return acc;
		}
	}
}

#endif /** NNL2_PRODUCT_H **/
