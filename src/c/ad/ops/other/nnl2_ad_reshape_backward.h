#ifndef NNL2_AD_RESHAPE_BACKWARD_H
#define NNL2_AD_RESHAPE_BACKWARD_H

/** @file nnl2_ad_reshape_backward.h
 ** @date 2025
 ** @copyright MIT
 ** @brief AD Backward for reshape operation
 **/
 
/// The future is a disaster 

/** @brief 
 * Computes the gradient of the reshape operation in reverse mode AD
 *
 ** @param out_tensor 
 * The output tensor from the reshape operation
 *
 ** @param input_tensor 
 * The input tensor to the reshape operation
 *
 ** @warning
 * Do not call the function directly. This is an internal function for AD
 *
 ** @note
 * Since reshape creates a copy, gradient is propagated by reshaping back to input shape
 * Uses input_tensor->data->shape directly - no need to store original shape separately
 *
 ** @see nnl2_reshape()
 ** @see nnl2_add_inplace()
 **/
static NNL2_FORCE_INLINE void nnl2_ad_reverse_derivative_reshape(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* input_tensor) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Safety checks
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
        NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor, "In function nnl2_ad_reverse_derivative_reshape, out_tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(input_tensor, "In function nnl2_ad_reverse_derivative_reshape, input_tensor is NULL");
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data, "In function nnl2_ad_reverse_derivative_reshape, out_tensor data is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(input_tensor->data, "In function nnl2_ad_reverse_derivative_reshape, input_tensor data is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->grad, "In function nnl2_ad_reverse_derivative_reshape, out_tensor grad is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(input_tensor->grad, "In function nnl2_ad_reverse_derivative_reshape, input_tensor grad is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data->shape, "In function nnl2_ad_reverse_derivative_reshape, out_tensor shape is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(input_tensor->data->shape, "In function nnl2_ad_reverse_derivative_reshape, input_tensor shape is NULL");
    #endif
    
    // Only propagate gradient if input requires gradient
    if(input_tensor->requires_grad) {
        // Reshape the output gradient back to input shape and add to input gradient
        // Use input_tensor->data->shape directly - no need to store separately
        nnl2_tensor* reshaped_grad = nnl2_reshape(out_tensor->grad, input_tensor->data->shape, input_tensor->data->rank, false);
        if(reshaped_grad) {
            nnl2_add_inplace(input_tensor->grad, reshaped_grad);
            nnl2_free_tensor(reshaped_grad);
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

#endif /** NNL2_AD_RESHAPE_BACKWARD_H **/