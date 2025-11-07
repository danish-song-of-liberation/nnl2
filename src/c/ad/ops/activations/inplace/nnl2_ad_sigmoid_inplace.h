#ifndef NNL2_AD_SIGMOID_INPLACE_H
#define NNL2_AD_SIGMOID_INPLACE_H

// NNL2

/** @file nnl2_ad_sigmoid_inplace.h
 ** @date 2025
 ** @copyright MIT
 ** @brief In-place sigmoid operation for AD tensors
 **/

/** @brief 
 * Applies in-place sigmoid operation to an AD tensor
 *
 * Sigmoid is defined as:
 *   f(x) = 1 / (1 + exp(-x))
 *
 * The approx parameter controls whether to use exact exponential
 * calculation or an optimized approximation
 *
 ** @param ad_tensor 
 * Pointer to the AD tensor to modify in-place
 *
 ** @param approx
 * If true, uses an approximate but faster computation of sigmoid
 * If false, uses exact exponential calculation
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
 ** @see sigmoidinplace()
 **
 ** @return void
 **
 ** @code
 * nnl2_ad_tensor* tensor = nnl2_ad_randn(...);
 * // Apply exact sigmoid activation function in-place
 * nnl2_ad_inplace_sigmoid(tensor, false);
 * 
 * // Apply approximate sigmoid for better performance
 * nnl2_ad_inplace_sigmoid(tensor, true);
 ** @endcode
 **
 ** @see NNL2_AD_INPLACE_FATAL
 ** @see NNL2_ERROR
 **/
void nnl2_ad_inplace_sigmoid(nnl2_ad_tensor* ad_tensor, bool approx) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(!ad_tensor) {
			NNL2_ERROR("ad_tensor is NULL (in function nnl2_ad_inplace_sigmoid)"); 
		}
		
		#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
			if(!ad_tensor->data) {
				NNL2_ERROR("ad_tensor->data is NULL (in function nnl2_ad_inplace_sigmoid)"); 
			}
		#endif
	#endif
	
	// Prevent in-place operations on tensors that require gradients and are not leaf nodes
	if(ad_tensor->requires_grad && !ad_tensor->is_leaf) {
		NNL2_AD_INPLACE_FATAL(".sigmoid! (.sigmoid in-place)", ad_tensor);
	}
	
	// Apply in-place sigmoid operation to ad tensor data
	sigmoidinplace(ad_tensor->data, approx);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

#endif /** NNL2_AD_SIGMOID_INPLACE_H **/
