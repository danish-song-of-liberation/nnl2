#ifndef NNL2_AD_VSTACK_BACKWARD_H
#define NNL2_AD_VSTACK_BACKWARD_H

/** @brief 
 * Compute derivative for vstack operation in reverse mode
 *
 ** @param output_tensor
 * The output tensor from vstack operation
 *
 ** @param tensora  
 * The first input tensor to vstack operation
 *
 ** @param tensorb  
 * The second input tensor to vstack operation
 *
 ** @exception NNL2Error
 * If any tensor is NULL, function returns early
 */
static void nnl2_ad_reverse_derivative_vstack(nnl2_ad_tensor* output_tensor, nnl2_ad_tensor* tensora, nnl2_ad_tensor* tensorb) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif

    if(!tensora->requires_grad && !tensorb->requires_grad) {
        return;
    }

    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
        NNL2_CHECK_NULL_IF_ERR_RETURN(output_tensor, "In function nnl2_ad_reverse_derivative_vstack, output_tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensora, "In function nnl2_ad_reverse_derivative_vstack, tensora is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensorb, "In function nnl2_ad_reverse_derivative_vstack, tensorb is NULL");
    #endif
	
	nnl2_tensor_type dtype = output_tensor->data->dtype;
	size_t type_size = get_dtype_size(dtype);

	size_t numela = nnl2_product(tensora->data->shape, tensora->data->rank);
	
	if(tensora->requires_grad) {
		if(!memcpy(tensora->grad->data, output_tensor->grad, numela * type_size)) {
			NNL2_ERROR("In function nnl2_ad_reverse_derivative_vstack failed to copy memory from output_tensor->grad to tensora->grad->data");
		}
	}
	
	if(tensorb->requires_grad) {
		size_t numelb = nnl2_product(tensorb->data->shape, tensorb->data->rank);
		void* src = (char*)output_tensor->grad + numela * type_size;
		
		if(!memcpy(tensorb->grad->data, src, numelb * type_size)) {
			NNL2_ERROR("In function nnl2_ad_reverse_derivative_vstack failed to copy memory from output_tensor->grad to tensorb->grad->data");
		}
	}		
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

#endif /** NNL2_AD_VSTACK_BACKWARD_H **/