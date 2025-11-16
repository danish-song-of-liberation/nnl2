#ifndef NNL2_AD_HSTACK_BACKWARD_H
#define NNL2_AD_HSTACK_BACKWARD_H

/** @brief 
 * Compute derivative for hstack operation in reverse mode
 *
 ** @param output_tensor
 * The output tensor from hstack operation
 *
 ** @param tensora  
 * The first input tensor to hstack operation
 *
 ** @param tensorb  
 * The second input tensor to hstack operation
 *
 ** @exception NNL2Error
 * If any tensor is NULL, function returns early
 */
static void nnl2_ad_reverse_derivative_hstack(nnl2_ad_tensor* output_tensor, nnl2_ad_tensor* tensora, nnl2_ad_tensor* tensorb) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    if(!tensora->requires_grad && !tensorb->requires_grad) {
        return;
    }
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
        NNL2_CHECK_NULL_IF_ERR_RETURN(output_tensor, "In function nnl2_ad_reverse_derivative_hstack, output_tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensora, "In function nnl2_ad_reverse_derivative_hstack, tensora is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensorb, "In function nnl2_ad_reverse_derivative_hstack, tensorb is NULL");
    #endif

    nnl2_tensor_type dtype = output_tensor->data->dtype;
    size_t type_size = get_dtype_size(dtype);

    size_t rows = output_tensor->data->shape[0];
    size_t a_cols = tensora->data->shape[1];
    size_t b_cols = tensorb->data->shape[1];

    char* grad_out = (char*)output_tensor->grad->data;
    char* grad_a = (char*)tensora->grad->data;
    char* grad_b = (char*)tensorb->grad->data;

    for(size_t i = 0; i < rows; i++) {
        if(tensora->requires_grad) {
            if(!memcpy(grad_a + i * a_cols * type_size, grad_out + i * (a_cols + b_cols) * type_size, a_cols * type_size))
				NNL2_ERROR("In function nnl2_ad_reverse_derivative_hstack, failed to memcpy");
        }
		
        if(tensorb->requires_grad) {
            if(!memcpy(grad_b + i * b_cols * type_size, grad_out + i * (a_cols + b_cols) * type_size + a_cols * type_size, b_cols * type_size))
				NNL2_ERROR("In function nnl2_ad_reverse_derivative_hstack, failed to memcpy");
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

#endif /** NNL2_AD_HSTACK_BACKWARD_H **/