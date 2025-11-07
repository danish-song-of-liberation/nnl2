#ifndef NNL2_AD_TANH_INPLACE_H
#define NNL2_AD_TANH_INPLACE_H

// NNL2

/** @file nnl2_ad_tanh_inplace.h
 ** @date 2025
 ** @copyright MIT
 ** @brief In-place hyperbolic tangent operation for AD tensors
 **/

/** @brief 
 * Applies in-place hyperbolic tangent operation to an AD tensor
 *
 * Tanh is defined as:
 *   f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
 *
 * The approx parameter controls whether to use exact exponential
 * calculation or an optimized approximation.
 *
 ** @param ad_tensor 
 * Pointer to the AD tensor to modify in-place
 *
 ** @param approx
 * If true, uses an approximate but faster computation of tanh.
 * If false, uses exact exponential calculation.
 *
 ** @exception NNL2_AD_INPLACE_FATAL
 * Terminates program if tensor requires gradients and is not a leaf node,
 * as in-place operations on non-leaf tensors with gradients are not supported
 *
 ** @exception NNL2_ERROR
 * NULL tensor pointer  
 *
 ** @exception NNL2_ERROR
 * NULL data pointer (depending on safety mode)
 *
 ** @see nnl2_ad_tensor
 ** @see tanhinplace()
 **
 ** @return void
 **
 ** @code
 * nnl2_ad_tensor* tensor = nnl2_ad_randn(...);
 * // Apply exact tanh activation function in-place
 * nnl2_ad_inplace_tanh(tensor, false);
 * 
 * // Apply approximate tanh for better performance
 * nnl2_ad_inplace_tanh(tensor, true);
 ** @endcode
 **
 ** @see NNL2_AD_INPLACE_FATAL
 ** @see NNL2_ERROR
 **/
void nnl2_ad_inplace_tanh(nnl2_ad_tensor* ad_tensor, bool approx) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(!ad_tensor) {
			NNL2_ERROR("ad_tensor is NULL (in function nnl2_ad_inplace_tanh)"); 
		}
		
		#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
			if(!ad_tensor->data) {
				NNL2_ERROR("ad_tensor->data is NULL (in function nnl2_ad_inplace_tanh)"); 
			}
		#endif
	#endif
	
	// Prevent in-place operations on tensors that require gradients and are not leaf nodes
	if(ad_tensor->requires_grad && !ad_tensor->is_leaf) {
		NNL2_AD_INPLACE_FATAL(".tanh! (.tanh in-place)", ad_tensor);
	}
	
	// Apply in-place tanh operation to ad tensor data
	tanhinplace(ad_tensor->data, approx);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

#endif /** NNL2_AD_TANH_INPLACE_H **/
