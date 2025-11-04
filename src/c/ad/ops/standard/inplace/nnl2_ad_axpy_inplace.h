#ifndef NNL2_AD_AXPY_INPLACE_H
#define NNL2_AD_AXPY_INPLACE_H

// NNL2

/** @file nnl2_ad_axpy_inplace.h
 ** @date 2025
 ** @copyright MIT
 ** @brief In-place AXPY operation for AD tensors
 **/
 
/** @brief 
 * Applies in-place AXPY operation to AD tensors (A * X + Y)
 *
 ** @param summand 
 * Pointer to the AD tensor to modify in-place (Y in AXPY)
 *
 ** @param sumend 
 * Pointer to the AD tensor to scale and add (X in AXPY)
 *
 ** @param multiplier 
 * Scalar multiplier for the sumend tensor (A in AXPY)
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
 ** @see axpy_inplace()
 **
 ** @return void
 **
 ** @code
 * nnl2_ad_tensor* tensor_y = nnl2_ad_ones(...);
 * nnl2_ad_tensor* tensor_x = nnl2_ad_zeros(...);
 * nnl2_ad_inplace_axpy(tensor_y, tensor_x, 2.5f, false); // y = 2.5 * x + y
 ** @endcode
 **
 ** @see NNL2_AD_INPLACE_FATAL
 ** @see NNL2_ERROR
 **/
void nnl2_ad_inplace_axpy(nnl2_ad_tensor* summand, nnl2_ad_tensor* sumend, float multiplier, bool retain_graph) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(!summand) {
			NNL2_ERROR("summand is NULL (in function nnl2_ad_inplace_axpy)"); 
		}
		if(!sumend) {
			NNL2_ERROR("sumend is NULL (in function nnl2_ad_inplace_axpy)"); 
		}
		
		#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
			if(!summand->data) {
				NNL2_ERROR("summand->data is NULL (in function nnl2_ad_inplace_axpy)"); 
			}
			if(!sumend->data) {
				NNL2_ERROR("sumend->data is NULL (in function nnl2_ad_inplace_axpy)"); 
			}
		#endif
	#endif
	
	// Prevent in-place operations on tensors that require gradients
	// when the computation graph needs to be preserved
	if(summand->requires_grad && retain_graph) {
		NNL2_AD_INPLACE_FATAL("axpy", summand);
    }
	
	// Apply in-place AXPY operation: summand = multiplier * sumend + summand
    axpy_inplace(summand->data, sumend->data, multiplier);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

#endif /** NNL2_AD_AXPY_INPLACE_H **/
