#ifndef NNL2_AD_SIN_BACKWARD_H
#define NNL2_AD_SIN_BACKWARD_H

/** @file nnl2_ad_sin_backward.h
 ** @date 2025
 ** @copyright MIT
 ** @brief AD Backward for sine operation
 **/

/** @brief 
 * Computes the gradient of the sine operation in reverse mode AD
 *
 ** @param out_tensor 
 * The output tensor from the sine operation
 *
 ** @param ad_tensor 
 * The input tensor to the sine operation
 *
 ** @warning
 * Do not call the function directly. This is an internal function for AD
 *
 ** @note
 * MVP
 *
 ** @see nnl2_sin
 ** @see nnl2_cos
 ** @see nnl2_mul_inplace
 ** @see nnl2_free_tensor
 **/
static NNL2_FORCE_INLINE void nnl2_ad_reverse_derivative_sin(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* ad_tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
    if(!ad_tensor->requires_grad) {
		#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
			NNL2_DEBUG("Exiting nnl2_ad_reverse_derivative_sin because ad_tensor is not requiring gradient");
			NNL2_FUNC_EXIT();
		#endif
		
		return;
	}
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor, "In function nnl2_ad_reverse_derivative_sin, out_tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(ad_tensor, "In function nnl2_ad_reverse_derivative_sin, ad_tensor is NULL");
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data, "In function nnl2_ad_reverse_derivative_sin, out_tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(ad_tensor->data, "In function nnl2_ad_reverse_derivative_sin, ad_tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data->shape, "In function nnl2_ad_reverse_derivative_sin, out_tensor shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(ad_tensor->data->shape, "In function nnl2_ad_reverse_derivative_sin, ad_tensor shape is NULL");
	#endif
    
    // dy/dx = cos(x), so dL/dx = dL/dy * cos(x)
    nnl2_tensor* cos_input = nnl2_cos(ad_tensor->data);  // cos(x)
    mulinplace(cos_input, out_tensor->grad);  // dL/dy * cos(x)
    
    nnl2_add_inplace(ad_tensor->grad, cos_input);
    
    // Free temporary tensors
    nnl2_free_tensor(cos_input);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

#endif /** NNL2_AD_SIN_BACKWARD_H **/
