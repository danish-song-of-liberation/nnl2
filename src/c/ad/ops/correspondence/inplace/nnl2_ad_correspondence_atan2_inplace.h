#ifndef NNL2_AD_ATAN2_ATAN2F_INPLACE_H
#define NNL2_AD_ATAN2_ATAN2F_INPLACE_H

// NNL2

/** @file nnl2_ad_atan2_atan2f_inplace.h
 ** @date 2025
 ** @copyright MIT
 ** @brief In-place scalar atan2 operation for AD tensors
 **/
 
/** @brief 
 * Applies in-place scalar atan2 operation to AD tensors
 *
 ** @param y 
 * Pointer to the AD tensor to modify in-place (y-coordinate)
 *
 ** @param threshold 
 * Pointer to scalar value for x-coordinate
 *
 ** @param retain_graph 
 * If true, indicates the computation graph should be
 * preserved for backward passes. In-place operations
 * are not allowed when this is true and the tensor
 * requires gradients.
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
 ** @see atan2_atan2f_inplace()
 **
 ** @return void
 **
 ** @code
 * nnl2_ad_tensor* y_tensor = nnl2_ad_ones(...);
 * float x_value = 0.5f;
 * nnl2_ad_atan2_atan2f_inplace(y_tensor, &x_value, false); // y_tensor = atan2(y_tensor, 0.5) (scalar atan2)
 ** @endcode
 **
 ** @see NNL2_AD_INPLACE_FATAL
 ** @see NNL2_ERROR
 **/
void nnl2_ad_atan2_correspondence_inplace(nnl2_ad_tensor* y, void* threshold, bool retain_graph) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(!y) {
			NNL2_ERROR("y is NULL (in function nnl2_ad_atan2_atan2f_inplace)"); 
		}
		
		if(!threshold) {
			NNL2_ERROR("threshold is NULL (in function nnl2_ad_atan2_atan2f_inplace)"); 
		}
		
		#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
			if(!y->data) {
				NNL2_ERROR("y->data is NULL (in function nnl2_ad_atan2_atan2f_inplace)"); 
			}
		#endif
	#endif
	
	// Prevent in-place operations on tensors that require gradients
	// when the computation graph needs to be preserved
	if(y->requires_grad && retain_graph) {
        NNL2_AD_INPLACE_FATAL(".atan2! (.atan2 in-place) (scalar)", y);
    }
  
    nnl2_atan2_correspondence_inplace(y->data, threshold);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

#endif /** NNL2_AD_ATAN2_ATAN2F_INPLACE_H **/
