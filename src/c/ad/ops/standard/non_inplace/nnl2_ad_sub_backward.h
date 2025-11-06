#ifndef NNL2_AD_SUB_BACKWARD_H
#define NNL2_AD_SUB_BACKWARD_H

/** @file nnl2_ad_sub_backward.h
 ** @date 2025
 ** @copyright MIT
 ** @brief AD Backward for subtraction operation
 **/

/** @brief 
 * Computes the gradient of the subtraction operation in reverse mode AD
 *
 ** @param out_tensor 
 * The output tensor from the subtraction operation
 *
 ** @param minuend 
 * The minuend input tensor to the subtraction operation
 *
 ** @param subtrahend 
 * The subtrahend input tensor to the subtraction operation
 *
 ** @warning
 * Do not call the function directly. This is an internal function for AD
 *
 ** @note
 * MVP
 *
 ** @see nnl2_add_inplace
 ** @see subinplace
 **/
static NNL2_FORCE_INLINE void nnl2_ad_reverse_derivative_sub(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* minuend, nnl2_ad_tensor* subtrahend) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor, "In function nnl2_ad_reverse_derivative_sub, out_tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(minuend, "In function nnl2_ad_reverse_derivative_sub, minuend is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(subtrahend, "In function nnl2_ad_reverse_derivative_sub, subtrahend is NULL");
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data, "In function nnl2_ad_reverse_derivative_sub, out_tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(minuend->data, "In function nnl2_ad_reverse_derivative_sub, minuend data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(subtrahend->data, "In function nnl2_ad_reverse_derivative_sub, subtrahend data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data->shape, "In function nnl2_ad_reverse_derivative_sub, out_tensor data shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(minuend->data->shape, "In function nnl2_ad_reverse_derivative_sub, minuend data shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(subtrahend->data->shape, "In function nnl2_ad_reverse_derivative_sub, subtrahend data shape is NULL");
	#endif
    
	if(minuend->requires_grad) nnl2_add_inplace(minuend->grad, out_tensor->grad);
	if(subtrahend->requires_grad) subinplace(subtrahend->grad, out_tensor->grad);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

#endif /** NNL2_AD_SUB_BACKWARD_H **/
