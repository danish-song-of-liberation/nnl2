#ifndef NNL2_AD_SUM_WITHOUT_AXIS_BACKWARD_H
#define NNL2_AD_SUM_WITHOUT_AXIS_BACKWARD_H

/** @file nnl2_ad_sum_without_axis_backward.h
 ** @date 2025
 ** @copyright MIT
 ** @brief AD Backward for sum without axis operation
 **/

/** @brief 
 * Compute derivative for sum without axis operation in reverse mode
 *
 ** @param output_tensor
 * The output tensor from sum without axis operation
 *
 ** @param input_tensor  
 * The input tensor to sum without axis operation
 *
 ** @details
 * Computes: dL/dinput = (dL/doutput) broadcasted to input shape
 * Since sum reduces all elements to scalar, gradient is propagated
 * equally to all input elements
 *
 ** @warning
 * Do not call the function directly. This is an internal function for AD
 *
 ** @exception NNL2Error
 * If any tensor is NULL, function returns early
 **/
static NNL2_FORCE_INLINE void nnl2_ad_reverse_derivative_sum_without_axis(nnl2_ad_tensor* output_tensor, nnl2_ad_tensor* input_tensor) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    if(!input_tensor->requires_grad) {
        return;
    }
    
    // Safety checks
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
        NNL2_CHECK_NULL_IF_ERR_RETURN(output_tensor, "In function nnl2_ad_reverse_derivative_sum_without_axis, output_tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(input_tensor, "In function nnl2_ad_reverse_derivative_sum_without_axis, input_tensor is NULL");
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(output_tensor->data, "In function nnl2_ad_reverse_derivative_sum_without_axis, output_tensor data is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(input_tensor->data, "In function nnl2_ad_reverse_derivative_sum_without_axis, input_tensor data is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(output_tensor->grad, "In function nnl2_ad_reverse_derivative_sum_without_axis, output_tensor grad is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(input_tensor->grad, "In function nnl2_ad_reverse_derivative_sum_without_axis, input_tensor grad is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(output_tensor->data->shape, "In function nnl2_ad_reverse_derivative_sum_without_axis, output_tensor shape is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(input_tensor->data->shape, "In function nnl2_ad_reverse_derivative_sum_without_axis, input_tensor shape is NULL");
    #endif
    
    add_broadcasting_inplace(input_tensor->grad, output_tensor->grad);
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

#endif /** NNL2_AD_SUM_WITHOUT_AXIS_BACKWARD_H **/
