#ifndef NNL2_AD_AXPF_BACKWARD_H
#define NNL2_AD_AXPF_BACKWARD_H

/** @file nnl2_ad_axpf_backward.h
 ** @date 2025
 ** @copyright MIT
 ** @brief AD Backward for axpf operation
 **/

/** @brief 
 * Computes the gradient of the axpf operation in reverse mode AD
 *
 ** @param out_tensor 
 * The output tensor from the axpf operation
 *
 ** @param summand_tensor 
 * The summand input tensor to the axpf operation
 *
 ** @param sumend
 * The sumend value pointer
 *
 ** @param alpha
 * The alpha scalar value
 *
 ** @warning
 * Do not call the function directly. This is an internal function for AD
 *
 ** @note
 * MVP
 *
 ** @see addinplace
 **/
static NNL2_FORCE_INLINE void nnl2_ad_reverse_derivative_axpf(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* summand_tensor, void* sumend, float alpha) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	(void)sumend;
	(void)alpha;
	
    if(!summand_tensor->requires_grad) {
		#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
			NNL2_DEBUG("Exiting nnl2_ad_reverse_derivative_axpf because summand_tensor is not requiring gradient");
			NNL2_FUNC_EXIT();
		#endif
		
		return;
	}
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor, "In function nnl2_ad_reverse_derivative_axpf, out_tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(summand_tensor, "In function nnl2_ad_reverse_derivative_axpf, summand_tensor is NULL");
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data, "In function nnl2_ad_reverse_derivative_axpf, out_tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(summand_tensor->data, "In function nnl2_ad_reverse_derivative_axpf, summand_tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data->shape, "In function nnl2_ad_reverse_derivative_axpf, out_tensor data shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(summand_tensor->data->shape, "In function nnl2_ad_reverse_derivative_axpf, summand_tensor data shape is NULL");
	#endif
    
    addinplace(summand_tensor->grad, out_tensor->grad);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

#endif /** NNL2_AD_AXPF_BACKWARD_H **/
