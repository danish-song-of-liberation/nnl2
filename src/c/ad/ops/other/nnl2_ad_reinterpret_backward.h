#ifndef NNL2_AD_REINTERPRET_BACKWARD_H
#define NNL2_AD_REINTERPRET_BACKWARD_H

/** @file nnl2_ad_reinterpret_backward.h
 ** @date 2025
 ** @copyright MIT
 ** @brief AD Backward for reinterpret operation
 **/

/** @brief 
 * Computes the gradient of the reinterpret operation in reverse mode AD
 *
 ** @param out_tensor 
 * The output tensor from the reinterpret operation
 *
 ** @param input_tensor 
 * The input tensor to the reinterpret operation
 *
 ** @warning
 * Do not call the function directly. This is an internal function for AD
 *
 ** @note
 * Since reinterpret creates a view (O(1) operation), gradient is directly
 * propagated to input gradient without reshaping
 *
 ** @see nnl2_add_inplace()
 **/
static NNL2_FORCE_INLINE void nnl2_ad_reverse_derivative_reinterpret(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* input_tensor) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Safety checks
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
        NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor, "In function nnl2_ad_reverse_derivative_reinterpret, out_tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(input_tensor, "In function nnl2_ad_reverse_derivative_reinterpret, input_tensor is NULL");
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data, "In function nnl2_ad_reverse_derivative_reinterpret, out_tensor data is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(input_tensor->data, "In function nnl2_ad_reverse_derivative_reinterpret, input_tensor data is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->grad, "In function nnl2_ad_reverse_derivative_reinterpret, out_tensor grad is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(input_tensor->grad, "In function nnl2_ad_reverse_derivative_reinterpret, input_tensor grad is NULL");
    #endif
    
    // Only propagate gradient if input requires gradient
    // For reinterpret, we directly add gradients since it's a view operation
    if(input_tensor->requires_grad) {
        nnl2_add_inplace(input_tensor->grad, out_tensor->grad);
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

#endif /** NNL2_AD_REINTERPRET_BACKWARD_H **/
