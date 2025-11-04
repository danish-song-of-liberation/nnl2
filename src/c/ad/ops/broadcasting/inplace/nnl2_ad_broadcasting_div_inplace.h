#ifndef NNL2_AD_DIV_BROADCASTING_INPLACE_H
#define NNL2_AD_DIV_BROADCASTING_INPLACE_H

// NNL2

/** @file nnl2_ad_div_broadcasting_inplace.h
 ** @date 2025
 ** @copyright MIT
 ** @brief In-place broadcasting division operation for AD tensors
 **/
 
/** @brief 
 * Applies in-place broadcasting division operation to AD tensors
 *
 ** @param dividend 
 * Pointer to the AD tensor to modify in-place (dividend and result)
 *
 ** @param divisor 
 * Pointer to the AD tensor to use as divisor (broadcasted if needed)
 *
 ** @param retain_graph 
 * If true, indicates the computation graph should be
 * preserved for backward passes. In-place operations
 * are not allowed when this is true and the tensor
 * requires gradients.
 *
 ** @exception NNL2_AD_INPLACE_FATAL
 * Terminates program if dividend requires gradients and retain_graph is true
 *
 ** @exception NNL2_ERROR
 * NULL tensor pointer  
 *
 ** @exception NNL2_ERROR
 * NULL data pointer (depending on safety mode)
 *
 ** @see nnl2_ad_tensor
 ** @see div_broadcasting_inplace()
 **
 ** @return void
 **
 ** @code
 * nnl2_ad_tensor* tensor1 = nnl2_ad_ones(...);
 * nnl2_ad_tensor* tensor2 = nnl2_ad_zeros(...);
 * nnl2_ad_div_broadcasting_inplace(tensor1, tensor2, false); // tensor1 = tensor1 ./ tensor2 (with broadcasting)
 ** @endcode
 **
 ** @see NNL2_AD_INPLACE_FATAL
 ** @see NNL2_ERROR
 **/
void nnl2_ad_div_broadcasting_inplace(nnl2_ad_tensor* dividend, nnl2_ad_tensor* divisor, bool retain_graph) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(!dividend) {
			NNL2_ERROR("dividend is NULL (in function nnl2_ad_div_broadcasting_inplace)"); 
		}
		if(!divisor) {
			NNL2_ERROR("divisor is NULL (in function nnl2_ad_div_broadcasting_inplace)"); 
		}
		
		#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
			if(!dividend->data) {
				NNL2_ERROR("dividend->data is NULL (in function nnl2_ad_div_broadcasting_inplace)"); 
			}
			if(!divisor->data) {
				NNL2_ERROR("divisor->data is NULL (in function nnl2_ad_div_broadcasting_inplace)"); 
			}
		#endif
	#endif
	
	// Prevent in-place operations on tensors that require gradients
	// when the computation graph needs to be preserved
	if(dividend->requires_grad && retain_graph) {
		NNL2_AD_INPLACE_FATAL("/! (./ in-place) (broadcasting)", dividend);
    }
	
	// Apply in-place broadcasting division operation: dividend = dividend ./ divisor (with broadcasting)
    div_broadcasting_inplace(dividend->data, divisor->data);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

#endif /** NNL2_AD_DIV_BROADCASTING_INPLACE_H **/
