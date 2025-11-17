#ifndef NNL2_AD_TREF_BACKWARD_H
#define NNL2_AD_TREF_BACKWARD_H

/** @brief 
 * Compute derivative for tref_getter operation in reverse mode
 *
 ** @param tref_tensor
 * The output tensor from tref_getter operation
 *
 ** @param original_tensor  
 * The original input tensor to tref_getter operation
 *
 ** @param indices  
 * Array of indices used for the tref_getter operation
 *
 ** @param num_indices
 * Number of indices in the indices array
 *
 ** @exception NNL2Error
 * If any tensor is NULL, function returns early
 */
static void nnl2_ad_reverse_derivative_tref_getter(nnl2_ad_tensor* tref_tensor, nnl2_ad_tensor* original_tensor, int32_t* indices, uint8_t num_indices) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif	

	nnl2_tensor* gradient_view = (nnl2_tensor*)nnl2_view(original_tensor -> grad, indices, num_indices);
	
	addinplace(gradient_view, tref_tensor -> grad);

    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

#endif /** NNL2_AD_TREF_BACKWARD_H **/
