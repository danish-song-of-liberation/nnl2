#ifndef NNL2_AD_SUB_BROADCASTING_INPLACE_H
#define NNL2_AD_SUB_BROADCASTING_INPLACE_H

// NNL2

/** @file nnl2_ad_sub_broadcasting_inplace.h
 ** @date 2025
 ** @copyright MIT
 ** @brief In-place broadcasting subtraction operation for AD tensors
 **/
 
/** @brief 
 * Applies in-place broadcasting subtraction operation to AD tensors
 *
 ** @param minuend 
 * Pointer to the AD tensor to modify in-place (minuend and result)
 *
 ** @param subtrahend 
 * Pointer to the AD tensor to subtract (broadcasted if needed)
 *
 ** @param retain_graph 
 * If true, indicates the computation graph should be
 * preserved for backward passes. In-place operations
 * are not allowed when this is true and the tensor
 * requires gradients.
 *
 ** @exception NNL2_AD_INPLACE_FATAL
 * Terminates program if minuend requires gradients and retain_graph is true
 *
 ** @exception NNL2_ERROR
 * NULL tensor pointer  
 *
 ** @exception NNL2_ERROR
 * NULL data pointer (depending on safety mode)
 *
 ** @see nnl2_ad_tensor
 ** @see sub_broadcasting_inplace()
 **
 ** @return void
 **
 ** @code
 * nnl2_ad_tensor* tensor1 = nnl2_ad_ones(...);
 * nnl2_ad_tensor* tensor2 = nnl2_ad_zeros(...);
 * nnl2_ad_sub_broadcasting_inplace(tensor1, tensor2, false); // tensor1 = tensor1 .- tensor2 (with broadcasting)
 ** @endcode
 **
 ** @see NNL2_AD_INPLACE_FATAL
 ** @see NNL2_ERROR
 **/
void nnl2_ad_sub_broadcasting_inplace(nnl2_ad_tensor* minuend, nnl2_ad_tensor* subtrahend, bool retain_graph) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(!minuend) {
			NNL2_ERROR("minuend is NULL (in function nnl2_ad_sub_broadcasting_inplace)"); 
		}
		if(!subtrahend) {
			NNL2_ERROR("subtrahend is NULL (in function nnl2_ad_sub_broadcasting_inplace)"); 
		}
		
		#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
			if(!minuend->data) {
				NNL2_ERROR("minuend->data is NULL (in function nnl2_ad_sub_broadcasting_inplace)"); 
			}
			if(!subtrahend->data) {
				NNL2_ERROR("subtrahend->data is NULL (in function nnl2_ad_sub_broadcasting_inplace)"); 
			}
		#endif
	#endif
	
	// Prevent in-place operations on tensors that require gradients
	// when the computation graph needs to be preserved
	if(minuend->requires_grad && retain_graph) {
        NNL2_AD_INPLACE_FATAL("-= (.- in-place) (broadcasting)", minuend);
    }
	
	// Apply in-place broadcasting subtraction operation: minuend = minuend .- subtrahend (with broadcasting)
    sub_broadcasting_inplace(minuend->data, subtrahend->data);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

#endif /** NNL2_AD_SUB_BROADCASTING_INPLACE_H **/
