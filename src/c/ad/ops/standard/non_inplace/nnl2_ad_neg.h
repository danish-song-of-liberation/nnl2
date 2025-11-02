#ifndef NNL2_AD_NEG_H
#define NNL2_AD_NEG_H

void nnl2_ad_reverse_backward_neg(nnl2_ad_tensor* tensor) {
    nnl2_ad_reverse_derivative_neg(tensor, tensor->roots[0]);
}   

nnl2_ad_tensor* nnl2_ad_neg(nnl2_ad_tensor* ad_tensor, nnl2_ad_mode ad_mode) {
    nnl2_ad_tensor* result = malloc(sizeof(nnl2_ad_tensor));
    
    result->data = nnl2_neg(ad_tensor->data);
    result->grad = nnl2_empty(ad_tensor->data->shape, ad_tensor->data->rank, ad_tensor->data->dtype);
    result->num_roots = 1;
    result->roots = (nnl2_ad_tensor**)malloc(sizeof(nnl2_ad_tensor*));
    result->roots[0] = ad_tensor;
    result->requires_grad = ad_tensor->requires_grad;
    result->magic_number = TENSOR_MAGIC_ALIVE;
    result->grad_initialized = false;
    result->is_leaf = false; 
    result->name = NULL;
    result->ts_type = nnl2_type_ad;
    
    switch(ad_mode) {
        case nnl2_ad_reverse_mode: result->backward_fn = nnl2_ad_reverse_backward_neg; break;
        
        default: {
            NNL2_UNKNOWN_AD_MODE_ERROR(ad_mode);
            nnl2_free_ad_tensor(result);
            return NULL;
        }
    }
    
    return result;
}

#endif /** NNL2_AD_NEG_H **/
