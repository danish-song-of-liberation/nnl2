#ifndef NNL2_AD_TRANSPOSE_DERIVATIVE_H
#define NNL2_AD_TRANSPOSE_DERIVATIVE_H

/** @brief
 * Reverse derivative implementation for transpose operation
 *
 ** @param out_tensor
 * Output tensor whose gradient is being propagated backward
 *
 ** @param ad_tensor
 * Root tensor receiving propagated gradient
 *
 ** @details
 * Performs transpose of output gradient and accumulates it
 * into the root gradient tensor using the same force mode
 * as the forward pass.
 **/
static void nnl2_ad_reverse_derivative_transpose(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* ad_tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif

	if(!ad_tensor->requires_grad) return;
	if(!out_tensor->grad) return;

	// Get force flag stored during forward pass
	bool force = out_tensor->extra_bool;

	// For transpose operation: dL/dx = T(dL/dy)
	Tensor* grad_x = transpose(out_tensor->grad, force);

	if(!grad_x) {
		NNL2_ERROR("Failed to perform gradient transposition in nnl2_ad_reverse_derivative_transpose");
		return;
	}

	// Accumulate gradient into root tensor
	addinplace(ad_tensor->grad, grad_x);

	// Free temporary gradient tensor
	nnl2_free_tensor(grad_x);

	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

#endif /** NNL2_AD_TRANSPOSE_DERIVATIVE_H **/
