#ifndef NNL2_AD_TRANSPOSITION_INPLACE_H
#define NNL2_AD_TRANSPOSITION_INPLACE_H

/** @brief
 * Reverse derivative implementation for O(1) transposition
 *
 ** @param out_tensor
 * Output tensor whose gradient is being propagated backward
 *
 ** @param ad_tensor
 * Root tensor receiving propagated gradient
 *
 ** @details
 * Performs view-based transpose of output gradient and accumulates it
 * into the root gradient tensor.
 **/
static void nnl2_ad_reverse_derivative_transposition(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* ad_tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif

	if(!ad_tensor->requires_grad) return;
	if(!out_tensor->grad) return;

	Tensor* grad_y = out_tensor->grad;
	Tensor* grad_x = nnl2_transposition(grad_y);

	if(!grad_x) {
		NNL2_ERROR("Failed to perform gradient view transposition in nnl2_ad_reverse_derivative_transposition");
		return;
	}

	addinplace(ad_tensor->grad, grad_x); // accumulate

	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

#endif /** NNL2_AD_TRANSPOSITION_INPLACE_H **/
