#ifndef NNL2_AD_ADD_INPLACE_H
#define NNL2_AD_ADD_INPLACE_H

// NNL2

/** @file nnl2_ad_add_inplace.h
 ** @date 2025
 ** @copyright MIT
 ** @brief In-place addition operation for AD tensors
 **/
 
/** @brief 
 * Applies in-place addition operation to an AD tensor 
 *
 ** @param summand 
 * Pointer to the AD tensor to modify in-place (receives the result)
 *
 ** @param addend 
 * Pointer to the AD tensor to add to the summand
 *
 ** @param track_graph 
 * If true, indicates the computation graph should be
 * preserved for backward passes. In-place operations
 * are not allowed when this is true and the tensor
 * requires gradients.
 *
 ** @exception NNL2_AD_INPLACE_FATAL
 * Terminates program if summand requires gradients and track_graph is true
 *
 ** @exception NNL2_ERROR
 * NULL tensor pointer  
 *
 ** @exception NNL2_ERROR
 * NULL data pointer (depending on safety mode)
 *
 ** @see nnl2_ad_tensor
 ** @see addinplace()
 **
 ** @return void
 **
 ** @code
 * nnl2_ad_tensor* tensor1 = nnl2_ad_ones(...);
 * nnl2_ad_tensor* tensor2 = nnl2_ad_zeros(...);
 * nnl2_ad_add_inplace(tensor1, tensor2, false); // No graph retention needed
 ** @endcode
 **
 ** @see NNL2_AD_INPLACE_FATAL
 ** @see NNL2_ERROR
 **/
void nnl2_ad_add_inplace(nnl2_ad_tensor* summand, nnl2_ad_tensor* addend, bool track_graph) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(!summand) {
			NNL2_ERROR("summand is NULL (in function nnl2_ad_add_inplace)"); 
		}
		if(!addend) {
			NNL2_ERROR("addend is NULL (in function nnl2_ad_add_inplace)"); 
		}
		
		#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
			if(!summand->data) {
				NNL2_ERROR("summand->data is NULL (in function nnl2_ad_add_inplace)"); 
			}
			if(!addend->data) {
				NNL2_ERROR("addend->data is NULL (in function nnl2_ad_add_inplace)"); 
			}
		#endif
	#endif
	
	// Prevent in-place operations on tensors that require gradients
	// when the computation graph needs to be preserved
	if(summand->requires_grad && track_graph) {
        NNL2_AD_INPLACE_FATAL("+= (Addition in-place)", summand);
    }
	
	// Apply in-place addition operation to summand tensor data
	addinplace(summand->data, addend->data);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

#endif /** NNL2_AD_ADD_INPLACE_H **/
