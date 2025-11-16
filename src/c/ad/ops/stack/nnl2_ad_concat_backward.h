#ifndef NNL2_AD_CONCAT_BACKWARD_H
#define NNL2_AD_CONCAT_BACKWARD_H

/** @brief 
 * Compute derivative for concat operation in reverse mode
 *
 ** @param output_tensor
 * The output tensor from concat operation
 *
 ** @param tensora  
 * The first input tensor to concat operation
 *
 ** @param tensorb  
 * The second input tensor to concat operation
 *
 ** @exception NNL2Error
 * If any tensor is NULL, function returns early
 */
static void nnl2_ad_reverse_derivative_concat(nnl2_ad_tensor* output_tensor, nnl2_ad_tensor* tensora, nnl2_ad_tensor* tensorb, int axis) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif

    if(!tensora->requires_grad && !tensorb->requires_grad) return;

    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
        NNL2_CHECK_NULL_IF_ERR_RETURN(output_tensor, "output_tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensora, "tensora is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensorb, "tensorb is NULL");
    #endif

    nnl2_tensor_type dtype = output_tensor->data->dtype;
    size_t type_size = get_dtype_size(dtype);

    size_t rank = output_tensor->data->rank;
    int32_t* out_shape = output_tensor->data->shape;
    int32_t* a_shape = tensora->data->shape;
    int32_t* b_shape = tensorb->data->shape;

    size_t outer_stride = 1;  
    size_t inner_stride = 1;  

    for(int i = 0; i < axis; i++) outer_stride *= out_shape[i];
    for(size_t i = axis + 1; i < rank; i++) inner_stride *= out_shape[i];

    char* grad_out = (char*)output_tensor->grad->data;
    char* grad_a = (char*)tensora->grad->data;
    char* grad_b = (char*)tensorb->grad->data;

    size_t a_axis = a_shape[axis];
    size_t b_axis = b_shape[axis];
    size_t out_axis = out_shape[axis];

    for(size_t outer = 0; outer < outer_stride; outer++) {
        for(size_t i = 0; i < a_axis; i++) {
            if(tensora->requires_grad) {
                memcpy(grad_a + ((outer * a_axis + i) * inner_stride) * type_size, grad_out + ((outer * out_axis + i) * inner_stride) * type_size, inner_stride * type_size);
            }
        }

        for(size_t i = 0; i < b_axis; i++) {
            if(tensorb->requires_grad) {
                memcpy(grad_b + ((outer * b_axis + i) * inner_stride) * type_size, grad_out + ((outer * out_axis + a_axis + i) * inner_stride) * type_size, inner_stride * type_size);
            }
        }
    }

    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

#endif /** NNL2_AD_CONCAT_BACKWARD_H **/
