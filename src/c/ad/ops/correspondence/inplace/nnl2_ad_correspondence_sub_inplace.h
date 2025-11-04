#ifndef NNL2_AD_SUB_DECF_INPLACE_H
#define NNL2_AD_SUB_DECF_INPLACE_H

// NNL2

/** @file nnl2_ad_sub_decf_inplace.h
 ** @date 2025
 ** @copyright MIT
 ** @brief In-place scalar subtraction operation for AD tensors
 **/
 
/** @brief 
 * Applies in-place scalar subtraction operation to AD tensors
 *
 ** @param tensor 
 * Pointer to the AD tensor to modify in-place
 *
 ** @param dec 
 * Pointer to scalar value to subtract from each element
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
 ** @see sub_decf_inplace()
 **
 ** @return void
 **
 ** @code
 * nnl2_ad_tensor* tensor = nnl2_ad_ones(...);
 * float scalar = 0.5f;
 * nnl2_ad_sub_decf_inplace(tensor, &scalar, false); // tensor = tensor .- 0.5 (scalar subtraction)
 ** @endcode
 **
 ** @see NNL2_AD_INPLACE_FATAL
 ** @see NNL2_ERROR
 **/
void nnl2_ad_sub_decf_inplace(nnl2_ad_tensor* tensor, void* dec, bool retain_graph) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(!tensor) {
			NNL2_ERROR("tensor is NULL (in function nnl2_ad_sub_decf_inplace)"); 
		}
		if(!dec) {
			NNL2_ERROR("dec is NULL (in function nnl2_ad_sub_decf_inplace)"); 
		}
		
		#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
			if(!tensor->data) {
				NNL2_ERROR("tensor->data is NULL (in function nnl2_ad_sub_decf_inplace)"); 
			}
		#endif
	#endif
	
	// Prevent in-place operations on tensors that require gradients
	// when the computation graph needs to be preserved
	if(tensor->requires_grad && retain_graph) {
        NNL2_AD_INPLACE_FATAL("-= (.- in-place) (scalar)", tensor);
    }
    
	// Apply in-place scalar subtraction operation: tensor = tensor .- dec
	sub_decf_inplace(tensor->data, dec);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

#endif /** NNL2_AD_SUB_DECF_INPLACE_H **/
