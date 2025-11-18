#ifndef NNL2_AD_FLAT_BACKWARD_H
#define NNL2_AD_FLAT_BACKWARD_H

/** @brief 
 * Compute derivative for flat operation in reverse mode
 *
 ** @param flat_tensor
 * The output tensor from flat operation (scalar tensor [1])
 *
 ** @param original_tensor  
 * The original input tensor to flat operation
 *
 ** @param at  
 * Flat index used for the element access
 *
 ** @param scalar_exit_p
 * Boolean indicating if the operation resulted in a scalar (always true for flat)
 *
 ** @exception NNL2Error
 * If any tensor is NULL, function returns early
 */
static void nnl2_ad_reverse_derivative_flat(nnl2_ad_tensor* flat_tensor, nnl2_ad_tensor* original_tensor, size_t at) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif

    nnl2_tensor* gradient_view = NULL;

    // Create a view to the specific flat index in the original gradient
    gradient_view = nnl2_empty((int32_t[]){ 1 }, 1, original_tensor->data->dtype);
    if(!gradient_view) {
        NNL2_ERROR("In function nnl2_ad_reverse_derivative_flat, failed to create gradient view");
        return;
    }
    
    // Set the data pointer to point to the specific flat index in original gradient
    gradient_view -> data = (void*)nnl2_get_raw_tensor_elem_at(original_tensor -> grad, at);
    if(!gradient_view->data) {
        NNL2_ERROR("In function nnl2_ad_reverse_derivative_flat, failed to get gradient view data");
        nnl2_free_tensor(gradient_view);
        return;
    }
    
    // Add the gradients: original_grad[at] += flat_grad[0]
    addinplace(gradient_view, flat_tensor -> grad);

    // Free the view container 
    nnl2_free_tensor(gradient_view);

    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

#endif /** NNL2_AD_FLAT_BACKWARD_H **/
