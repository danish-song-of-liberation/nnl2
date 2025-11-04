#ifndef NNL2_AD_DIV_INPLACE_H
#define NNL2_AD_DIV_INPLACE_H

// NNL2

/** @file nnl2_ad_div_inplace.h
 ** @date 2025
 ** @copyright MIT
 ** @brief In-place division operation for AD tensors
 **/
 
/** @brief 
 * Applies in-place division operation to AD tensors
 *
 ** @param result 
 * Pointer to the AD tensor to modify in-place (dividend and result)
 *
 ** @param operand 
 * Pointer to the AD tensor to use as divisor
 *
 ** @param retain_graph 
 * If true, indicates the computation graph should be
 * preserved for backward passes. In-place operations
 * are not allowed when this is true and the tensor
 * requires gradients.
 *
 ** @exception NNL2_AD_INPLACE_FATAL
 * Terminates program if result requires gradients and retain_graph is true
 *
 ** @exception NNL2_ERROR
 * NULL tensor pointer  
 *
 ** @exception NNL2_ERROR
 * NULL data pointer (depending on safety mode)
 *
 ** @see nnl2_ad_tensor
 ** @see divinplace()
 **
 ** @return void
 **
 ** @code
 * nnl2_ad_tensor* tensor1 = nnl2_ad_ones(...);
 * nnl2_ad_tensor* tensor2 = nnl2_ad_zeros(...);
 * nnl2_ad_div_inplace(tensor1, tensor2, false); // tensor1 = tensor1 / tensor2
 ** @endcode
 **
 ** @see NNL2_AD_INPLACE_FATAL
 ** @see NNL2_ERROR
 **/
void nnl2_ad_div_inplace(nnl2_ad_tensor* result, nnl2_ad_tensor* operand, bool retain_graph) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(!result) {
			NNL2_ERROR("result is NULL (in function nnl2_ad_div_inplace)"); 
		}
		if(!operand) {
			NNL2_ERROR("operand is NULL (in function nnl2_ad_div_inplace)"); 
		}
		
		#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
			if(!result->data) {
				NNL2_ERROR("result->data is NULL (in function nnl2_ad_div_inplace)"); 
			}
			if(!operand->data) {
				NNL2_ERROR("operand->data is NULL (in function nnl2_ad_div_inplace)"); 
			}
		#endif
	#endif
	
	// Prevent in-place operations on tensors that require gradients
	// when the computation graph needs to be preserved
	if(result->requires_grad && retain_graph) {
        NNL2_AD_INPLACE_FATAL("/! (division in-place)", result);
    }
	
	// Apply in-place division operation: result = result / operand
    divinplace(result->data, operand->data);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

#endif /** NNL2_AD_DIV_INPLACE_H **/
