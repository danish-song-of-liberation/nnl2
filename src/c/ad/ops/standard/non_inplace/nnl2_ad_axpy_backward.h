#ifndef NNL2_AD_ADD_BACKWARD_DERIVATIVE_H
#define NNL2_AD_ADD_BACKWARD_DERIVATIVE_H

/** @file nnl2_ad_add_backward_derivative.h
 ** @date 2025
 ** @copyright MIT
 ** @brief AD Backward for axpy operation
 **/

/** @brief 
 * Computes the gradient of the axpy operation in reverse mode AD
 *
 ** @param out_tensor 
 * The output tensor from the axpy operation
 *
 ** @param addend 
 * The first input tensor to the axpy operation
 *
 ** @param sumend 
 * The second input tensor to the axpy operation
 *
 ** @param multiplier
 * The multiplier scalar value
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
 ** @see addinplace
 **/
static NNL2_FORCE_INLINE void nnl2_ad_reverse_derivative_axpy(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* addend, nnl2_ad_tensor* sumend, float multiplier) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor, "In function nnl2_ad_reverse_derivative_axpy, out_tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(addend, "In function nnl2_ad_reverse_derivative_axpy, addend is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(sumend, "In function nnl2_ad_reverse_derivative_axpy, sumend is NULL");
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data, "In function nnl2_ad_reverse_derivative_axpy, out_tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(addend->data, "In function nnl2_ad_reverse_derivative_axpy, addend data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(sumend->data, "In function nnl2_ad_reverse_derivative_axpy, sumend data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data->shape, "In function nnl2_ad_reverse_derivative_axpy, out_tensor shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(addend->data->shape, "In function nnl2_ad_reverse_derivative_axpy, addend shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(sumend->data->shape, "In function nnl2_ad_reverse_derivative_axpy, sumend shape is NULL");
	#endif
    
	if(addend->requires_grad) {
        nnl2_tensor* scaled_grad = scale(out_tensor->grad, multiplier, true);
        nnl2_add_inplace(addend->grad, scaled_grad);
        nnl2_free_tensor(scaled_grad);
    }
	
	if(sumend->requires_grad) {
        addinplace(sumend->grad, out_tensor->grad);
    }
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

#endif /** NNL2_AD_ADD_BACKWARD_DERIVATIVE_H **/
