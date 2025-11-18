#ifndef NNL2_AD_TREFW_BACKWARD_H
#define NNL2_AD_TREFW_BACKWARD_H

/** @brief 
 * Compute derivative for trefw operation in reverse mode
 *
 ** @param trefw_tensor
 * The output tensor from trefw operation (scalar tensor [1])
 *
 ** @param original_tensor  
 * The original input tensor to trefw operation
 *
 ** @param coords  
 * Array of coordinates used for the element access
 *
 ** @param coords_len
 * Number of coordinates in the coords array
 *
 ** @param scalar_exit_p
 * Boolean indicating if the operation resulted in a scalar (always true for trefw)
 *
 ** @exception NNL2Error
 * If any tensor is NULL, function returns early
 */
static void nnl2_ad_reverse_derivative_trefw(nnl2_ad_tensor* trefw_tensor, nnl2_ad_tensor* original_tensor, int32_t* coords, int32_t coords_len) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif

    nnl2_tensor* gradient_view = NULL;

    // Create a view to the specific coordinate in the original gradient
    gradient_view = nnl2_empty((int32_t[]){ 1 }, 1, original_tensor->data->dtype);
    if(!gradient_view) {
        NNL2_ERROR("In function nnl2_ad_reverse_derivative_trefw, failed to create gradient view");
        return;
    }
    
    // Set the data pointer to point to the specific coordinate in original gradient
    gradient_view -> data = (void*)nnl2_get_raw_tensor_elem(original_tensor -> grad, coords, coords_len);
    if(!gradient_view->data) {
        NNL2_ERROR("In function nnl2_ad_reverse_derivative_trefw, failed to get gradient view data");
        nnl2_free_tensor(gradient_view);
        return;
    }
    
    // original_grad[coords] += trefw_grad[0]
    addinplace(gradient_view, trefw_tensor -> grad);

    // Free the view container
    nnl2_free_tensor(gradient_view);

    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

#endif /** NNL2_AD_TREFW_BACKWARD_H **/
