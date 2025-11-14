#ifndef NNL2_AD_SLICE_BACKWARD_H
#define NNL2_AD_SLICE_BACKWARD_H

/** @file nnl2_ad_slice_backward.h
 ** @date 2025
 ** @copyright MIT
 ** @brief AD Backward for slice operation
 **/

/** @brief 
 * Computes the gradient of the slice operation in reverse mode AD
 *
 ** @param out_tensor 
 * The output tensor from the slice operation
 *
 ** @param input_tensor 
 * The input tensor to the slice operation
 *
 ** @warning
 * Do not call the function directly. This is an internal function for AD
 *
 ** @note
 * Since slice creates a view, gradient is propagated by accumulating to the sliced region
 * Uses appropriate indexing to add gradients only to the sliced portion
 *
 ** @see nnl2_slice()
 ** @see nnl2_add_inplace()
 **/
static NNL2_FORCE_INLINE void nnl2_ad_reverse_derivative_slice(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* input_tensor, int32_t* from, int32_t* to) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Safety checks
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
        NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor, "In function nnl2_ad_reverse_derivative_slice, out_tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(input_tensor, "In function nnl2_ad_reverse_derivative_slice, input_tensor is NULL");
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data, "In function nnl2_ad_reverse_derivative_slice, out_tensor data is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(input_tensor->data, "In function nnl2_ad_reverse_derivative_slice, input_tensor data is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->grad, "In function nnl2_ad_reverse_derivative_slice, out_tensor grad is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(input_tensor->grad, "In function nnl2_ad_reverse_derivative_slice, input_tensor grad is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data->shape, "In function nnl2_ad_reverse_derivative_slice, out_tensor shape is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(input_tensor->data->shape, "In function nnl2_ad_reverse_derivative_slice, input_tensor shape is NULL");
    #endif

    if(input_tensor->requires_grad) {
		(void)to;

        int32_t rank = input_tensor->data->rank;
        int32_t correct_to[rank];
        
        for(int i = 0; i < rank; i++) {
            correct_to[i] = from[i] + out_tensor->data->shape[i] - 1;
        }
        
        nnl2_axpy_inplace_region(input_tensor->grad, out_tensor->grad, 1.0f, from, correct_to);
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

#endif /** NNL2_AD_SLICE_BACKWARD_H **/
