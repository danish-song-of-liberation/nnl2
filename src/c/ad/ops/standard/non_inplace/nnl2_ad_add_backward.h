#ifndef NNL2_AD_ADD_BACKWARD_H
#define NNL2_AD_ADD_BACKWARD_H

/** @file nnl2_ad_add_backward.h
 ** @date 2025
 ** @copyright MIT
 ** @brief AD Backward for addition operation
 **/

/** @brief 
 * Computes the gradient of the addition operation in reverse mode AD
 *
 ** @param out_tensor 
 * The output tensor from the addition operation
 *
 ** @param addend 
 * The first input tensor to the addition operation
 *
 ** @param sumend 
 * The second input tensor to the addition operation
 *
 ** @warning
 * Do not call the function directly. This is an internal function for AD
 *
 ** @note
 * MVP
 *
 ** @see nnl2_add_inplace
 **/
static NNL2_FORCE_INLINE void nnl2_ad_reverse_derivative_add(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* addend, nnl2_ad_tensor* sumend) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor, "In function nnl2_ad_reverse_derivative_add, out_tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(addend, "In function nnl2_ad_reverse_derivative_add, addend is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(sumend, "In function nnl2_ad_reverse_derivative_add, sumend is NULL");
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data, "In function nnl2_ad_reverse_derivative_add, out_tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(addend->data, "In function nnl2_ad_reverse_derivative_add, addend data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(sumend->data, "In function nnl2_ad_reverse_derivative_add, sumend data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data->shape, "In function nnl2_ad_reverse_derivative_add, out_tensor shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(addend->data->shape, "In function nnl2_ad_reverse_derivative_add, addend shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(sumend->data->shape, "In function nnl2_ad_reverse_derivative_add, sumend shape is NULL");
	#endif
    
	if(addend->requires_grad) nnl2_add_inplace(addend->grad, out_tensor->grad);
	if(sumend->requires_grad) nnl2_add_inplace(sumend->grad, out_tensor->grad);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

#endif /** NNL2_AD_ADD_BACKWARD_H **/
