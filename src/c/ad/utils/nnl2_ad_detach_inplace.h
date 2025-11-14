#ifndef NNL2_DETACH_INPLACE_H
#define NNL2_DETACH_INPLACE_H

// NNL2

/** @file nnl2_ad_detach_inplace.h
 ** @date 2025
 ** @copyright MIT License
 ** @brief Contains detach inplace function
 **/

/** @brief
 * Disconnects the tensor from the computational graph in place
 *
 ** @param ad_tensor
 * Tensor for graph disconnection
 */
void nnl2_ad_detach_inplace(nnl2_ad_tensor* ad_tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
		NNL2_FUNC_ENTER();
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		NNL2_CHECK_NULL_IF_ERR_RETURN(ad_tensor, "In function nnl2_ad_detach_inplace, ad_tensor is NULL");
	#endif 
	
	ad_tensor->requires_grad = false;  // Disable gradient tracking
	ad_tensor->is_leaf = true;		   // Make it a leaf tensor
	ad_tensor->backward_fn = NULL;	   // Remove backward function
	
	if(ad_tensor->roots != NULL) {
        free(ad_tensor->roots);
        ad_tensor->roots = NULL;
        ad_tensor->num_roots = 0;
    }
	
	// Reset extra fields that are related to computational graph
	ad_tensor->extra_multiplier = 1.0f;  // Reset any scaling factors
    ad_tensor->extra_bool = false;		 // Reset any boolean flags used in backward
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
		NNL2_FUNC_EXIT();
	#endif
}

#endif /** NNL2_DETACH_INPLACE_H **/
