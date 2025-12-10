#ifndef NNL2_AD_ASIN_INPLACE_H
#define NNL2_AD_ASIN_INPLACE_H

// NNL2

/** @file nnl2_ad_asin_inplace.h
 ** @date 2025
 ** @copyright MIT
 ** @brief In-place arcsine operation for AD tensors
 **/
 
/** @brief 
 * Applies in-place arcsine (inverse sine) operation to an AD tensor 
 *
 ** @param ad_tensor 
 * Pointer to the AD tensor to modify in-place
 *
 ** @param retain_graph 
 * If true, indicates the computation graph should be
 * preserved for backward passes. In-place operations
 * are not allowed when this is true and the tensor
 * requires gradients.
 *
 ** @exception NNL2_AD_INPLACE_FATAL
 * Terminates program if tensor requires gradients and retain_graph is true
 *
 ** @exception NNL2_ERROR
 * NULL tensor pointer  
 *
 ** @exception NNL2_ERROR
 * NULL data pointer (depending on safety mode)
 *
 ** @see nnl2_ad_tensor
 ** @see nnl2_asin_inplace()
 **
 ** @return void
 **
 ** @note
 * Input values must be in the range [-1, 1] for valid arcsine computation.
 *
 ** @code
 * nnl2_ad_tensor* tensor = nnl2_ad_ones(...);
 * // ... initialize tensor with some data in range [-1, 1]
 * nnl2_ad_asin_inplace(tensor, false); // No graph retention needed
 ** @endcode
 **
 ** @see NNL2_AD_INPLACE_FATAL
 ** @see NNL2_ERROR
 **/
void nnl2_ad_asin_inplace(nnl2_ad_tensor* ad_tensor, bool retain_graph) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(!ad_tensor) {
			NNL2_ERROR("ad_tensor is NULL (in function nnl2_ad_asin_inplace)"); 
		}
		
		#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
			if(!ad_tensor->data) {
				NNL2_ERROR("ad_tensor->data is NULL (in function nnl2_ad_asin_inplace)"); 
			}
		#endif
	#endif
	
	// Prevent in-place operations on tensors that require gradients
	// when the computation graph needs to be preserved
	if(ad_tensor->requires_grad && retain_graph) {
        NNL2_AD_INPLACE_FATAL(".asin! (.asin in-place)", ad_tensor);
    }
	
	// Apply in-place arcsine operation to ad tensor data
	nnl2_asininplace(ad_tensor->data);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

#endif /** NNL2_AD_ASIN_INPLACE_H **/
