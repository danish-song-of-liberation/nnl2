#ifndef NNL2_AD_SUB_BROADCASTING_H
#define NNL2_AD_SUB_BROADCASTING_H

void nnl2_ad_reverse_backward_sub_broadcasting(nnl2_ad_tensor* tensor) {
    nnl2_ad_reverse_derivative_sub_broadcasting(tensor, tensor->roots[0], tensor->roots[1]);
}   

nnl2_ad_tensor* nnl2_ad_sub_broadcasting(nnl2_ad_tensor* minuend, nnl2_ad_tensor* subtrahend, nnl2_ad_mode ad_mode) {
    //not ready
    nnl2_ad_tensor* result = malloc(sizeof(nnl2_ad_tensor));

    result->data = sub_broadcasting(minuend->data, subtrahend->data);
    result->grad = nnl2_empty(minuend->data->shape, minuend->data->rank, minuend->data->dtype);
    result->num_roots = 2;
    result->roots = (nnl2_ad_tensor**)malloc(2 * sizeof(nnl2_ad_tensor*));
    result->roots[0] = minuend;
    result->roots[1] = subtrahend;
    result->requires_grad = minuend->requires_grad || subtrahend->requires_grad;
    result->magic_number = TENSOR_MAGIC_ALIVE;
    result->grad_initialized = false;
    result->ts_type = nnl2_type_ad;
    
    switch(ad_mode) {
        case nnl2_ad_reverse_mode: result->backward_fn = nnl2_ad_reverse_backward_sub_broadcasting; break;
        
        default: {
            NNL2_UNKNOWN_AD_MODE_ERROR(ad_mode);
            nnl2_free_ad_tensor(result);
            return NULL;
        }
    }
    
    return result;
}

#endif /** NNL2_AD_SUB_BROADCASTING_H **/
