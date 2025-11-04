#ifndef NNL2_AD_AXPF_INPLACE_H
#define NNL2_AD_AXPF_INPLACE_H

// NNL2

/** @file nnl2_ad_axpf_inplace.h
 ** @date 2025
 ** @copyright MIT
 ** @brief In-place scalar AXPY operation for AD tensors
 **/
 
/** @brief 
 * Applies in-place scalar AXPY operation to AD tensors (alpha * x + y)
 *
 ** @param summand 
 * Pointer to the AD tensor to modify in-place (y in AXPY)
 *
 ** @param sumend 
 * Pointer to scalar value to scale and add (x in AXPY)
 *
 ** @param alpha 
 * Scalar multiplier for the sumend value (alpha in AXPY)
 *
 ** @param retain_graph 
 * If true, indicates the computation graph should be
 * preserved for backward passes. In-place operations
 * are not allowed when this is true and the tensor
 * requires gradients.
 *
 ** @exception NNL2_AD_INPLACE_FATAL
 * Terminates program if summand requires gradients and retain_graph is true
 *
 ** @exception NNL2_ERROR
 * NULL tensor pointer  
 *
 ** @exception NNL2_ERROR
 * NULL data pointer (depending on safety mode)
 *
 ** @see nnl2_ad_tensor
 ** @see axpf_inplace()
 **
 ** @return void
 **
 ** @code
 * nnl2_ad_tensor* tensor = nnl2_ad_ones(...);
 * float scalar = 2.0f;
 * float alpha = 1.5f;
 * nnl2_ad_axpf_inplace(tensor, &scalar, alpha, false); // tensor = 1.5 * 2.0 + tensor
 ** @endcode
 **
 ** @see NNL2_AD_INPLACE_FATAL
 ** @see NNL2_ERROR
 **/
void nnl2_ad_axpf_inplace(nnl2_ad_tensor* summand, void* sumend, float alpha, bool retain_graph) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(!summand) {
			NNL2_ERROR("summand is NULL (in function nnl2_ad_axpf_inplace)"); 
		}
		if(!sumend) {
			NNL2_ERROR("sumend is NULL (in function nnl2_ad_axpf_inplace)"); 
		}
		
		#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
			if(!summand->data) {
				NNL2_ERROR("summand->data is NULL (in function nnl2_ad_axpf_inplace)"); 
			}
		#endif
	#endif
	
	// Prevent in-place operations on tensors that require gradients
	// when the computation graph needs to be preserved
	if(summand->requires_grad && retain_graph) {
        NNL2_AD_INPLACE_FATAL("axpy! (axpy in-place) (scalar)", summand);
    }

	// Apply in-place scalar AXPY operation: summand = alpha * sumend + summand
	axpf_inplace(summand->data, sumend, alpha);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

#endif /** NNL2_AD_AXPF_INPLACE_H **/
