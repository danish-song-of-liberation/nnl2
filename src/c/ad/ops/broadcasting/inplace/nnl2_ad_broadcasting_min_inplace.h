#ifndef NNL2_AD_MIN_BROADCASTING_INPLACE_H
#define NNL2_AD_MIN_BROADCASTING_INPLACE_H

void nnl2_ad_reverse_backward_min_broadcasting_inplace(nnl2_ad_tensor* tensor_a) {
    nnl2_ad_reverse_derivative_min_broadcasting_inplace(tensor_a, tensor_a->roots[0]);
}   

void nnl2_ad_min_broadcasting_inplace(nnl2_ad_tensor* tensor_a, nnl2_ad_tensor* tensor_b, nnl2_ad_mode ad_mode) {
    if(tensor_a->requires_grad && !tensor_a->is_leaf) {
        // do something
    }
    
    min_broadcasting_inplace(tensor_a->data, tensor_b->data);
    tensor_a->num_roots = 1;
    tensor_a->roots = (nnl2_ad_tensor**)malloc(sizeof(nnl2_ad_tensor*));
    tensor_a->roots[0] = tensor_b;
    tensor_a->is_leaf = false; 
    tensor_a->ts_type = nnl2_type_ad;
    
    switch(ad_mode) {
        case nnl2_ad_reverse_mode: tensor_a->backward_fn = nnl2_ad_reverse_backward_min_broadcasting_inplace; break;
        
        default: {
            NNL2_UNKNOWN_AD_MODE_ERROR(ad_mode);
            return;
        }
    }
}

#endif /** NNL2_AD_MIN_BROADCASTING_INPLACE_H **/
