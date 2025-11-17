#ifndef NNL2_AD_VIEW_BACKWARD_H
#define NNL2_AD_VIEW_BACKWARD_H

/** @brief 
 * Compute derivative for view operation in reverse mode
 *
 ** @param view_tensor
 * The output tensor from view operation
 *
 ** @param original_tensor  
 * The original input tensor to view operation
 *
 ** @param indices  
 * Array of indices used for the view operation
 *
 ** @param num_indices
 * Number of indices in the indices array
 *
 ** @exception NNL2Error
 * If any tensor is NULL, function returns early
 */
static void nnl2_ad_reverse_derivative_view(nnl2_ad_tensor* view_tensor, nnl2_ad_tensor* original_tensor, int32_t* indices, uint8_t num_indices, bool scalar_exit_p) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif	
	
	nnl2_tensor* gradient_view = NULL;

	if(!scalar_exit_p) {
		gradient_view = (nnl2_tensor*)nnl2_view(original_tensor -> grad, indices, num_indices);
	} else {
		gradient_view = nnl2_empty((int[]){ 1 }, 1, original_tensor->data->dtype);
		gradient_view -> data = (void*)nnl2_view(original_tensor -> grad, indices, num_indices);
	}
	
	addinplace(gradient_view, view_tensor -> grad);

    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

#endif /** NNL2_AD_VIEW_BACKWARD_H **/
