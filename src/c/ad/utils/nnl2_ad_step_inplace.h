#ifndef NNL2_AD_STEP_INPLACE_H
#define NNL2_AD_STEP_INPLACE_H

// NNL2

/** @file nnl2_ad_step_inplace.h
 ** @brief In-place automatic differentiation optimization step functions
 ** @date 2025
 ** @copyright MIT
 **/

/** @brief 
 * Performs a single in-place optimization step using gradient descent
 *
 ** @param ad_tensor
 * Pointer to the automatic differentiation tensor to update in-place
 *
 ** @param learning_rate
 * Learning rate for the optimization step
 *
 ** @exception NNL2Error
 * If ad_tensor is NULL and safety mode is enabled
 *
 ** @exception NNL2Error
 * If tensor data or gradient is NULL and safety mode is enabled
 *
 ** @see axpy_inplace()
 **/
void nnl2_ad_step_inplace(nnl2_ad_tensor* ad_tensor, float learning_rate) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		NNL2_CHECK_NULL_IF_ERR_RETURN(ad_tensor, "In function nnl2_ad_step_inplace, ad_tensor is NULL");
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_FULL
		NNL2_CHECK_NULL_IF_ERR_RETURN(ad_tensor->data, "In function nnl2_ad_step_inplace, ad_tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(ad_tensor->grad, "In function nnl2_ad_step_inplace, ad_tensor grad is NULL");
	#endif
	
	// Perform in-place optimization step: data = data - learning_rate * grad
	axpy_inplace(ad_tensor->data, ad_tensor->grad, -learning_rate);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

#endif /** NNL2_AD_STEP_INPLACE_H **/
