#ifndef NNL2_AD_SCALE_INPLACE_H
#define NNL2_AD_SCALE_INPLACE_H

void nnl2_ad_reverse_backward_inplace_scale(nnl2_ad_tensor* tensor) {
    nnl2_ad_reverse_derivative_inplace_scale(tensor, tensor->roots[0], tensor->extra_multiplier);
}   

void nnl2_ad_inplace_scale(nnl2_ad_tensor* ad_tensor, nnl2_float32 multiplier, nnl2_ad_mode ad_mode) {
    if(ad_tensor->requires_grad && !ad_tensor->is_leaf) {
        // do something
    }
    
    ad_tensor->num_roots = 1;
    ad_tensor->roots = (nnl2_ad_tensor**)malloc(sizeof(nnl2_ad_tensor*));
    ad_tensor->roots[0] = nnl2_ad_copy(ad_tensor, ad_tensor->data->dtype);
    scaleinplace(ad_tensor->data, multiplier);
    ad_tensor->is_leaf = false; 
    ad_tensor->ts_type = nnl2_type_ad;
    ad_tensor->extra_multiplier = multiplier;
    
    switch(ad_mode) {
        case nnl2_ad_reverse_mode: ad_tensor->backward_fn = nnl2_ad_reverse_backward_inplace_scale; break;
        
        default: {
            NNL2_UNKNOWN_AD_MODE_ERROR(ad_mode);
            return;
        }
    }
}

#endif /** NNL2_AD_SCALE_INPLACE_H **/
