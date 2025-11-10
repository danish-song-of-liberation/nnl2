#ifndef NNL2_AD_TRANSPOSITION_INPLACE_POINT_H
#define NNL2_AD_TRANSPOSITION_INPLACE_POINT_H

// NNL2

/** @file nnl2_ad_transposition_inplace.h  
 ** @date 2025
 ** @copyright MIT
 ** @brief In-place transposition operation for AD tensors
 **/
 
/** @brief 
 * Applies in-place transposition operation to an AD tensor 
 *
 ** @param ad_tensor 
 * Pointer to the AD tensor to modify in-place
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
 ** @warning
 * MATHEMATICALLY INCORRECT FOR MOST OPERATIONS!
 * While the visual representation is correct, most tensor operations
 * will produce mathematically incorrect results when applied to transposed views
 *
 ** @warning
 * This is a DESTRUCTIVE operation that modifies the input tensor.
 * The original shape and strides are permanently lost.
 *
 ** @see nnl2_ad_tensor
 ** @see nnl2_transposition_inplace()
 **
 ** @return void
 **
 ** @code
 * nnl2_ad_tensor* tensor = nnl2_ad_ones((int[]){3, 2}, 2, FLOAT64);
 * // ... initialize tensor with some data
 * nnl2_ad_inplace_transposition(tensor, false); // No graph retention needed
 * // tensor shape is now [2, 3]
 ** @endcode
 **
 ** @see NNL2_AD_INPLACE_FATAL
 ** @see NNL2_ERROR
 **/
void nnl2_ad_inplace_transposition(nnl2_ad_tensor* ad_tensor, bool retain_graph) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(!ad_tensor) {
			NNL2_ERROR("ad_tensor is NULL (in function nnl2_ad_inplace_transposition)"); 
		}
		
		#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
			if(!ad_tensor->data) {
				NNL2_ERROR("ad_tensor->data is NULL (in function nnl2_ad_inplace_transposition)"); 
			}
			
			if(!ad_tensor->data->shape) {
				NNL2_ERROR("ad_tensor->data->shape is NULL (in function nnl2_ad_inplace_transposition)"); 
			}
			
			if(!ad_tensor->data->strides) {
				NNL2_ERROR("ad_tensor->data->strides is NULL (in function nnl2_ad_inplace_transposition)"); 
			}
		#endif
	#endif
	
	// Prevent in-place operations on tensors that require gradients
	// when the computation graph needs to be preserved
	if(ad_tensor->requires_grad && retain_graph) {
		NNL2_AD_INPLACE_FATAL("transpose! (transpose in-place)", ad_tensor);
	}
	
	// Apply in-place transposition operation to ad tensor data
    nnl2_transposition_inplace(ad_tensor->data);
	
	// Also transpose the gradient tensor if it exists to maintain shape consistency
	if(ad_tensor->grad) {
		nnl2_transposition_inplace(ad_tensor->grad);
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

#endif /** NNL2_AD_TRANSPOSITION_INPLACE_POINT_H **/
