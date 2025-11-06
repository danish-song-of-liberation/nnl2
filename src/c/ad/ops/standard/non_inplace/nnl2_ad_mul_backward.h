#ifndef NNL2_AD_MUL_BACKWARD_H
#define NNL2_AD_MUL_BACKWARD_H

/** @file nnl2_ad_mul_backward.h
 ** @date 2025
 ** @copyright MIT
 ** @brief AD Backward for multiplication operation
 **/

/** @brief 
 * Computes the gradient of the multiplication operation in reverse mode AD
 *
 ** @param out_tensor 
 * The output tensor from the multiplication operation
 *
 ** @param addend 
 * The first input tensor to the multiplication operation
 *
 ** @param sumend 
 * The second input tensor to the multiplication operation
 *
 ** @warning
 * Do not call the function directly. This is an internal function for AD
 *
 ** @note
 * MVP
 *
 ** @see mul
 ** @see nnl2_add_inplace
 ** @see nnl2_free_tensor
 **/
static NNL2_FORCE_INLINE void nnl2_ad_reverse_derivative_mul(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* addend, nnl2_ad_tensor* sumend) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor, "In function nnl2_ad_reverse_derivative_mul, out_tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(addend, "In function nnl2_ad_reverse_derivative_mul, addend is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(sumend, "In function nnl2_ad_reverse_derivative_mul, sumend is NULL");
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data, "In function nnl2_ad_reverse_derivative_mul, out_tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(addend->data, "In function nnl2_ad_reverse_derivative_mul, addend data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(sumend->data, "In function nnl2_ad_reverse_derivative_mul, sumend data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data->shape, "In function nnl2_ad_reverse_derivative_mul, out_tensor data shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(addend->data->shape, "In function nnl2_ad_reverse_derivative_mul, addend data shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(sumend->data->shape, "In function nnl2_ad_reverse_derivative_mul, sumend data shape is NULL");
	#endif
    
	nnl2_tensor* grad_out_a = mul(out_tensor->grad, sumend->data);
	nnl2_tensor* grad_out_b = mul(out_tensor->grad, addend->data);
	
	if(addend->requires_grad) nnl2_add_inplace(addend->grad, grad_out_a);
	if(sumend->requires_grad) nnl2_add_inplace(sumend->grad, grad_out_b);
	
	nnl2_free_tensor(grad_out_a);
	nnl2_free_tensor(grad_out_b);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

#endif /** NNL2_AD_MUL_BACKWARD_H **/
