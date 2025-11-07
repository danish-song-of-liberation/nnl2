#ifndef NNL2_AD_STEP_H
#define NNL2_AD_STEP_H

// NNL2

/** @file nnl2_ad_step.h
 ** @brief Automatic differentiation optimization step function
 ** @date 2025
 ** @copyright MIT
 **/

/** @brief 
 * Performs a single optimization step using gradient descent
 *
 ** @param ad_tensor
 * Pointer to the automatic differentiation tensor to update
 *
 ** @param learning_rate
 * Learning rate for the optimization step
 *
 ** @return nnl2_tensor*
 * Pointer to the updated tensor data, or NULL on failure
 *
 ** @details
 * Updates tensor data using gradient descent: data = data - learning_rate * grad
 * This is equivalent to axpy(data, grad, -learning_rate)
 *
 ** @exception NNL2Error
 * If ad_tensor is NULL and safety mode is enabled
 *
 ** @exception NNL2Error
 * If tensor data or gradient is NULL and safety mode is enabled
 *
 ** @see axpy()
 **/
nnl2_tensor* nnl2_ad_step(nnl2_ad_tensor* ad_tensor, float learning_rate) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(ad_tensor, "In function nnl2_ad_step, ad_tensor is NULL", NULL);
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_FULL
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(ad_tensor->data, "In function nnl2_ad_step, ad_tensor data is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(ad_tensor->grad, "In function nnl2_ad_step, ad_tensor grad is NULL", NULL);
	#endif

	nnl2_tensor* result = axpy(ad_tensor->data, ad_tensor->grad, -learning_rate);
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(!result) {
			NNL2_ERROR("In function nnl2_ad_step, axpy is failed. NULL will be returned");
		}
	#endif
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
	
	return result;
}

#endif /** NNL2_AD_STEP_H **/
