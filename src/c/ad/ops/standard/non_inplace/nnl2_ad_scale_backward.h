#ifndef NNL2_AD_LOG_BACKWARD_H
#define NNL2_AD_LOG_BACKWARD_H

/** @file nnl2_ad_log_backward.h
 ** @date 2025
 ** @copyright MIT
 ** @brief AD Backward for scale operation
 **/

/** @brief 
 * Computes the gradient of the scale operation in reverse mode AD
 *
 ** @param out_tensor 
 * The output tensor from the scale operation
 *
 ** @param ad_tensor 
 * The input tensor to the scale operation
 *
 ** @param multiplier
 * The scalar multiplier value
 *
 ** @warning
 * Do not call the function directly. This is an internal function for AD
 *
 ** @note
 * MVP
 *
 ** @see scale
 ** @see nnl2_add_inplace
 ** @see nnl2_free_tensor
 **/
static NNL2_FORCE_INLINE void nnl2_ad_reverse_derivative_scale(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* ad_tensor, float multiplier) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
    if(!ad_tensor->requires_grad) {
		#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
			NNL2_DEBUG("Exiting nnl2_ad_reverse_derivative_scale because ad_tensor is not requiring gradient");
			NNL2_FUNC_EXIT();
		#endif
		
		return;
	}
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor, "In function nnl2_ad_reverse_derivative_scale, out_tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(ad_tensor, "In function nnl2_ad_reverse_derivative_scale, ad_tensor is NULL");
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data, "In function nnl2_ad_reverse_derivative_scale, out_tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(ad_tensor->data, "In function nnl2_ad_reverse_derivative_scale, ad_tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data->shape, "In function nnl2_ad_reverse_derivative_scale, out_tensor data shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(ad_tensor->data->shape, "In function nnl2_ad_reverse_derivative_scale, ad_tensor data shape is NULL");
	#endif
    
	nnl2_tensor* scaled_grad = scale(out_tensor->grad, multiplier, true);
    nnl2_add_inplace(ad_tensor->grad, scaled_grad);
    nnl2_free_tensor(scaled_grad);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

#endif /** NNL2_AD_LOG_BACKWARD_H **/
