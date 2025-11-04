#ifndef NNL2_AD_MUL_BROADCASTING_INPLACE_H
#define NNL2_AD_MUL_BROADCASTING_INPLACE_H

// NNL2

/** @file nnl2_ad_mul_broadcasting_inplace.h
 ** @date 2025
 ** @copyright MIT
 ** @brief In-place broadcasting multiplication operation for AD tensors
 **/
 
/** @brief 
 * Applies in-place broadcasting element-wise multiplication operation to AD tensors
 *
 ** @param multiplicand 
 * Pointer to the AD tensor to modify in-place (first operand and result)
 *
 ** @param multiplier 
 * Pointer to the AD tensor to multiply with (broadcasted if needed)
 *
 ** @param retain_graph 
 * If true, indicates the computation graph should be
 * preserved for backward passes. In-place operations
 * are not allowed when this is true and the tensor
 * requires gradients.
 *
 ** @exception NNL2_AD_INPLACE_FATAL
 * Terminates program if multiplicand requires gradients and retain_graph is true
 *
 ** @exception NNL2_ERROR
 * NULL tensor pointer  
 *
 ** @exception NNL2_ERROR
 * NULL data pointer (depending on safety mode)
 *
 ** @see nnl2_ad_tensor
 ** @see mul_broadcasting_inplace()
 **
 ** @return void
 **
 ** @code
 * nnl2_ad_tensor* tensor1 = nnl2_ad_ones(...);
 * nnl2_ad_tensor* tensor2 = nnl2_ad_zeros(...);
 * nnl2_ad_mul_broadcasting_inplace(tensor1, tensor2, false); // tensor1 = tensor1 .* tensor2 (with broadcasting)
 ** @endcode
 **
 ** @see NNL2_AD_INPLACE_FATAL
 ** @see NNL2_ERROR
 **/
void nnl2_ad_mul_broadcasting_inplace(nnl2_ad_tensor* multiplicand, nnl2_ad_tensor* multiplier, bool retain_graph) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(!multiplicand) {
			NNL2_ERROR("multiplicand is NULL (in function nnl2_ad_mul_broadcasting_inplace)"); 
		}
		
		if(!multiplier) {
			NNL2_ERROR("multiplier is NULL (in function nnl2_ad_mul_broadcasting_inplace)"); 
		}
		
		#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
			if(!multiplicand->data) {
				NNL2_ERROR("multiplicand->data is NULL (in function nnl2_ad_mul_broadcasting_inplace)"); 
			}
			
			if(!multiplier->data) {
				NNL2_ERROR("multiplier->data is NULL (in function nnl2_ad_mul_broadcasting_inplace)"); 
			}
		#endif
	#endif
	
	// Prevent in-place operations on tensors that require gradients
	// when the computation graph needs to be preserved
	if(multiplicand->requires_grad && retain_graph) {
        NNL2_AD_INPLACE_FATAL("*= (.* in-place) (broadcasting)", multiplicand);
    }
	
	// Apply in-place broadcasting element-wise multiplication operation: multiplicand = multiplicand .* multiplier (with broadcasting)
    mul_broadcasting_inplace(multiplicand->data, multiplier->data);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

#endif /** NNL2_AD_MUL_BROADCASTING_INPLACE_H **/
