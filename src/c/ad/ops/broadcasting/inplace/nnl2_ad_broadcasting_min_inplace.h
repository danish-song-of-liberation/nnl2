#ifndef NNL2_AD_MIN_BROADCASTING_INPLACE_H
#define NNL2_AD_MIN_BROADCASTING_INPLACE_H

// NNL2

/** @file nnl2_ad_min_broadcasting_inplace.h
 ** @date 2025
 ** @copyright MIT
 ** @brief In-place broadcasting minimum operation for AD tensors
 **/
 
/** @brief 
 * Applies in-place broadcasting element-wise minimum operation to AD tensors
 *
 ** @param tensor_a 
 * Pointer to the AD tensor to modify in-place (first operand and result)
 *
 ** @param tensor_b 
 * Pointer to the AD tensor to compare with (broadcasted if needed)
 *
 ** @param retain_graph 
 * If true, indicates the computation graph should be
 * preserved for backward passes. In-place operations
 * are not allowed when this is true and the tensor
 * requires gradients.
 *
 ** @exception NNL2_AD_INPLACE_FATAL
 * Terminates program if tensor_a requires gradients and retain_graph is true
 *
 ** @exception NNL2_ERROR
 * NULL tensor pointer  
 *
 ** @exception NNL2_ERROR
 * NULL data pointer (depending on safety mode)
 *
 ** @see nnl2_ad_tensor
 ** @see min_broadcasting_inplace()
 **
 ** @return void
 **
 ** @code
 * nnl2_ad_tensor* tensor1 = nnl2_ad_ones(...);
 * nnl2_ad_tensor* tensor2 = nnl2_ad_zeros(...);
 * nnl2_ad_min_broadcasting_inplace(tensor1, tensor2, false); // tensor1 = min(tensor1, tensor2) (with broadcasting)
 ** @endcode
 **
 ** @see NNL2_AD_INPLACE_FATAL
 ** @see NNL2_ERROR
 **/
void nnl2_ad_min_broadcasting_inplace(nnl2_ad_tensor* tensor_a, nnl2_ad_tensor* tensor_b, bool retain_graph) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(!tensor_a) {
			NNL2_ERROR("tensor_a is NULL (in function nnl2_ad_min_broadcasting_inplace)"); 
		}
		if(!tensor_b) {
			NNL2_ERROR("tensor_b is NULL (in function nnl2_ad_min_broadcasting_inplace)"); 
		}
		
		#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
			if(!tensor_a->data) {
				NNL2_ERROR("tensor_a->data is NULL (in function nnl2_ad_min_broadcasting_inplace)"); 
			}
			if(!tensor_b->data) {
				NNL2_ERROR("tensor_b->data is NULL (in function nnl2_ad_min_broadcasting_inplace)"); 
			}
		#endif
	#endif
	
	// Prevent in-place operations on tensors that require gradients
	// when the computation graph needs to be preserved
	if(tensor_a->requires_grad && retain_graph) {
        NNL2_AD_INPLACE_FATAL(".min! (.min in-place) (broadcasting)", tensor_a);
    }
	
	// Apply in-place broadcasting element-wise minimum operation: tensor_a = min(tensor_a, tensor_b) (with broadcasting)
    min_broadcasting_inplace(tensor_a->data, tensor_b->data);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

#endif /** NNL2_AD_MIN_BROADCASTING_INPLACE_H **/
