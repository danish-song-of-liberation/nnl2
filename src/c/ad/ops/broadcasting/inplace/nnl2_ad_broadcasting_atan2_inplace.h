#ifndef NNL2_AD_ATAN2_BROADCASTING_INPLACE_H
#define NNL2_AD_ATAN2_BROADCASTING_INPLACE_H

// NNL2

/** @file nnl2_ad_broadcasting_atan2_inplace.h
 ** @date 2025
 ** @copyright MIT
 ** @brief In-place broadcasting atan2 operation for AD tensors
 **/
 
/** @brief 
 * Applies in-place broadcasting element-wise atan2 operation to AD tensors
 *
 ** @param y 
 * Pointer to the AD tensor to modify in-place (y-coordinate operand and result)
 *
 ** @param x 
 * Pointer to the AD tensor for x-coordinate (broadcasted if needed)
 *
 ** @param retain_graph 
 * If true, indicates the computation graph should be
 * preserved for backward passes. In-place operations
 * are not allowed when this is true and the tensor
 * requires gradients
 *
 ** @exception NNL2_AD_INPLACE_FATAL
 * Terminates program if y requires gradients and retain_graph is true
 *
 ** @exception NNL2_ERROR
 * NULL tensor pointer  
 *
 ** @exception NNL2_ERROR
 * NULL data pointer (depending on safety mode)
 *
 ** @see nnl2_ad_tensor
 ** @see atan2_broadcasting_inplace()
 **
 ** @return void
 **
 ** @code
 * nnl2_ad_tensor* y_tensor = nnl2_ad_ones(...);
 * nnl2_ad_tensor* x_tensor = nnl2_ad_zeros(...);
 * nnl2_ad_atan2_broadcasting_inplace(y_tensor, x_tensor, false); // y_tensor = atan2(y_tensor, x_tensor) (with broadcasting)
 ** @endcode
 **
 ** @see NNL2_AD_INPLACE_FATAL
 ** @see NNL2_ERROR
 **/
void nnl2_ad_atan2_broadcasting_inplace(nnl2_ad_tensor* y, nnl2_ad_tensor* x, bool retain_graph) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(!y) {
			NNL2_ERROR("y is NULL (in function nnl2_ad_atan2_broadcasting_inplace)"); 
		}
		
		if(!x) {
			NNL2_ERROR("x is NULL (in function nnl2_ad_atan2_broadcasting_inplace)"); 
		}
		
		#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
			if(!y->data) {
				NNL2_ERROR("y->data is NULL (in function nnl2_ad_atan2_broadcasting_inplace)"); 
			}
			
			if(!x->data) {
				NNL2_ERROR("x->data is NULL (in function nnl2_ad_atan2_broadcasting_inplace)"); 
			}
		#endif
	#endif
	
	// Prevent in-place operations on tensors that require gradients
	// when the computation graph needs to be preserved
	if(y->requires_grad && retain_graph) {
        NNL2_AD_INPLACE_FATAL(".atan2! (.atan2 in-place) (broadcasting)", y);
    }
	
	// y = atan2(y, x) (with broadcasting)
    nnl2_atan2_broadcasting_inplace(y->data, x->data);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

#endif /** NNL2_AD_ATAN2_BROADCASTING_INPLACE_H **/
