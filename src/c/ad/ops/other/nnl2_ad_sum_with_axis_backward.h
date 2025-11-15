#ifndef NNL2_AD_SUM_AXIS_BACKWARD_H
#define NNL2_AD_SUM_AXIS_BACKWARD_H

/** @file nnl2_ad_sum_axis_backward.h
 ** @date 2025
 ** @copyright MIT
 ** @brief AD Backward for sum with axis operation
 **/

/** @brief 
 * Compute derivative for sum with axis operation in reverse mode
 *
 ** @param output_tensor
 * The output tensor from sum with axis operation
 *
 ** @param input_tensor  
 * The input tensor to sum with axis operation
 *
 ** @details
 * Computes gradient propagation for sum operation along specific axis
 * Uses broadcasting to propagate gradients from reduced output back to input shape
 *
 ** @warning
 * Do not call the function directly. This is an internal function for AD
 *
 ** @exception NNL2Error
 * If any tensor is NULL, function returns early
 **/
static NNL2_FORCE_INLINE void nnl2_ad_reverse_derivative_sum_axis(nnl2_ad_tensor* output_tensor, nnl2_ad_tensor* input_tensor) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    if(!input_tensor->requires_grad) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_DEBUG("Exiting nnl2_ad_reverse_derivative_sum_axis because input_tensor is not requiring gradient");
            NNL2_FUNC_EXIT();
        #endif
        return;
    }
    
    // Safety checks
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
        NNL2_CHECK_NULL_IF_ERR_RETURN(output_tensor, "In function nnl2_ad_reverse_derivative_sum_axis, output_tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(input_tensor, "In function nnl2_ad_reverse_derivative_sum_axis, input_tensor is NULL");
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(output_tensor->data, "In function nnl2_ad_reverse_derivative_sum_axis, output_tensor data is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(input_tensor->data, "In function nnl2_ad_reverse_derivative_sum_axis, input_tensor data is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(output_tensor->grad, "In function nnl2_ad_reverse_derivative_sum_axis, output_tensor grad is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(input_tensor->grad, "In function nnl2_ad_reverse_derivative_sum_axis, input_tensor grad is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(output_tensor->data->shape, "In function nnl2_ad_reverse_derivative_sum_axis, output_tensor shape is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(input_tensor->data->shape, "In function nnl2_ad_reverse_derivative_sum_axis, input_tensor shape is NULL");
    #endif
    
    // Use broadcasting to propagate gradients
    // For sum along axis: dL/dinput = broadcast(dL/doutput) to input shape
    add_broadcasting_inplace(input_tensor->grad, output_tensor->grad);
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

#endif /** NNL2_AD_SUM_AXIS_BACKWARD_H **/
