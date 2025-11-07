#ifndef NNL2_AD_LEAKYRELU_INPLACE_H
#define NNL2_AD_LEAKYRELU_INPLACE_H

// NNL2

/** @file nnl2_ad_leakyrelu_inplace.h
 ** @date 2025
 ** @copyright MIT
 ** @brief In-place LeakyReLU operation for AD tensors
 **/

/** @brief 
 * Applies in-place LeakyReLU operation to an AD tensor
 *
 * LeakyReLU is defined as:
 *   f(x) = x if x >= 0
 *   f(x) = alpha * x if x < 0
 *
 ** @param ad_tensor 
 * Pointer to the AD tensor to modify in-place
 *
 ** @param alpha
 * Negative slope coefficient for x < 0. Typically a small
 * positive value (e.g., 0.01) to prevent "dead neurons"
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
 ** @see leakyreluinplace()
 **
 ** @return void
 **
 ** @code
 * nnl2_ad_tensor* tensor = nnl2_ad_randn(...);
 * // Apply LeakyReLU with alpha = 0.01
 * nnl2_ad_inplace_leakyrelu(tensor, 0.01f);
 ** @endcode
 **
 ** @see NNL2_AD_INPLACE_FATAL
 ** @see NNL2_ERROR
 **/
void nnl2_ad_inplace_leakyrelu(nnl2_ad_tensor* ad_tensor, nnl2_float32 alpha) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(!ad_tensor) {
			NNL2_ERROR("ad_tensor is NULL (in function nnl2_ad_inplace_leakyrelu)"); 
		}
		
		#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
			if(!ad_tensor->data) {
				NNL2_ERROR("ad_tensor->data is NULL (in function nnl2_ad_inplace_leakyrelu)"); 
			}
		#endif
	#endif
	
	// Prevent in-place operations on tensors that require gradients
	if(ad_tensor->requires_grad) {
		NNL2_AD_INPLACE_FATAL(".leaky-relu! (.leaky_relu in-place)", ad_tensor);
	}
	
	// Apply in-place LeakyReLU operation to ad tensor data
	leakyreluinplace(ad_tensor->data, alpha);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

#endif /** NNL2_AD_LEAKYRELU_INPLACE_H **/