#ifndef NNL2_AD_RELU_INPLACE_H
#define NNL2_AD_RELU_INPLACE_H

void nnl2_ad_reverse_backward_inplace_relu(nnl2_ad_tensor* tensor) {
    nnl2_ad_reverse_derivative_inplace_relu(tensor, tensor->roots[0]);
}   

void nnl2_ad_inplace_relu(nnl2_ad_tensor* ad_tensor, nnl2_ad_mode ad_mode) {
    if(ad_tensor->requires_grad && !ad_tensor->is_leaf) {
        // do something
    }
    
    ad_tensor->num_roots = 1;
    ad_tensor->roots = (nnl2_ad_tensor**)malloc(sizeof(nnl2_ad_tensor*));
    ad_tensor->roots[0] = nnl2_ad_copy(ad_tensor, ad_tensor->data->dtype); 
    reluinplace(ad_tensor->data);
    ad_tensor->is_leaf = false; 
    ad_tensor->ts_type = nnl2_type_ad;
    
    switch(ad_mode) {
        case nnl2_ad_reverse_mode: ad_tensor->backward_fn = nnl2_ad_reverse_backward_inplace_relu; break;
        
        default: {
            NNL2_UNKNOWN_AD_MODE_ERROR(ad_mode);
            return;
        }
    }
}

#endif /** NNL2_AD_RELU_INPLACE_H **/
