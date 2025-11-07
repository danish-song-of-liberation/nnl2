#ifndef NNL2_AD_RELU_INPLACE_H
#define NNL2_AD_RELU_INPLACE_H

// NNL2

/** @file nnl2_ad_relu_inplace.h
 ** @date 2025
 ** @copyright MIT
 ** @brief In-place ReLU operation for AD tensors
 **/

/** @brief 
 * Applies in-place Rectified Linear Unit (ReLU) operation to an AD tensor
 *
 * ReLU is defined as:
 *   f(x) = max(0, x)
 *
 * This operation sets all negative values in the tensor to zero
 * while preserving positive values unchanged.
 *
 ** @param ad_tensor 
 * Pointer to the AD tensor to modify in-place
 *
 ** @exception NNL2_AD_INPLACE_FATAL
 * Terminates program if tensor requires gradients, as in-place
 * operations on tensors with gradients are not supported
 *
 ** @exception NNL2_ERROR
 * NULL tensor pointer  
 *
 ** @exception NNL2_ERROR
 * NULL data pointer (depending on safety mode)
 *
 ** @see nnl2_ad_tensor
 ** @see reluinplace()
 **
 ** @return void
 **
 ** @code
 * nnl2_ad_tensor* tensor = nnl2_ad_randn(...);
 * // Apply ReLU activation function in-place
 * nnl2_ad_inplace_relu(tensor);
 ** @endcode
 **
 ** @see NNL2_AD_INPLACE_FATAL
 ** @see NNL2_ERROR
 **/
void nnl2_ad_inplace_relu(nnl2_ad_tensor* ad_tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(!ad_tensor) {
			NNL2_ERROR("ad_tensor is NULL (in function nnl2_ad_inplace_relu)"); 
		}
		
		#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
			if(!ad_tensor->data) {
				NNL2_ERROR("ad_tensor->data is NULL (in function nnl2_ad_inplace_relu)"); 
			}
		#endif
	#endif
	
	// Prevent in-place operations on tensors that require gradients
	if(ad_tensor->requires_grad) {
		NNL2_AD_INPLACE_FATAL(".relu! (.relu in-place)", ad_tensor);
	}
	
	// Apply in-place ReLU operation to ad tensor data
	reluinplace(ad_tensor->data);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

#endif /** NNL2_AD_RELU_INPLACE_H **/
