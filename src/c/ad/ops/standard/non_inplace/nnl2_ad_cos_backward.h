#ifndef NNL2_AD_COS_BACKWARD_H
#define NNL2_AD_COS_BACKWARD_H

/** @file nnl2_ad_cos_backward.h
 ** @date 2025
 ** @copyright MIT
 ** @brief AD Backward for cosine operation
 **/

/** @brief 
 * Computes the gradient of the cosine operation in reverse mode AD
 *
 ** @param out_tensor 
 * The output tensor from the cosine operation
 *
 ** @param ad_tensor 
 * The input tensor to the cosine operation
 *
 ** @warning
 * Do not call the function directly. This is an internal function for AD
 *
 ** @note
 * MVP
 *
 ** @see nnl2_cos
 ** @see nnl2_sin
 ** @see nnl2_neg
 ** @see nnl2_mul_inplace
 ** @see nnl2_free_tensor
 **/
static NNL2_FORCE_INLINE void nnl2_ad_reverse_derivative_cos(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* ad_tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
    if(!ad_tensor->requires_grad) {
		#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
			NNL2_DEBUG("Exiting nnl2_ad_reverse_derivative_cos because ad_tensor is not requiring gradient");
			NNL2_FUNC_EXIT();
		#endif
		
		return;
	}
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor, "In function nnl2_ad_reverse_derivative_cos, out_tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(ad_tensor, "In function nnl2_ad_reverse_derivative_cos, ad_tensor is NULL");
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data, "In function nnl2_ad_reverse_derivative_cos, out_tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(ad_tensor->data, "In function nnl2_ad_reverse_derivative_cos, ad_tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data->shape, "In function nnl2_ad_reverse_derivative_cos, out_tensor shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(ad_tensor->data->shape, "In function nnl2_ad_reverse_derivative_cos, ad_tensor shape is NULL");
	#endif

    // dy/dx = -sin(x), so dL/dx = dL/dy * (-sin(x)) = -(dL/dy * sin(x))
    nnl2_tensor* sin_input = nnl2_sin(ad_tensor->data);      // sin(x)
    mulinplace(sin_input, out_tensor->grad);  // dL/dy * sin(x)
    nnl2_neginplace(sin_input);     // -(dL/dy * sin(x))
    
    nnl2_add_inplace(ad_tensor->grad, sin_input);
    
    // Free temporary tensors
    nnl2_free_tensor(sin_input);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

#endif /** NNL2_AD_COS_BACKWARD_H **/
