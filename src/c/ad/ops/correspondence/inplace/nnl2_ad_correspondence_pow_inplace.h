#ifndef NNL2_AD_POW_POWF_INPLACE_H
#define NNL2_AD_POW_POWF_INPLACE_H

void nnl2_ad_reverse_backward_pow_powf_inplace(nnl2_ad_tensor* tensor) {
    nnl2_ad_reverse_derivative_pow_powf_inplace(tensor, tensor->roots[0], tensor->extra_correspondence);
}   

void nnl2_ad_pow_powf_inplace(nnl2_ad_tensor* tensor, void* exponent, nnl2_ad_mode ad_mode) {
    if(tensor->requires_grad && !tensor->is_leaf) {
        // do something
    }
    
    tensor->num_roots = 1;
    tensor->roots = (nnl2_ad_tensor**)malloc(sizeof(nnl2_ad_tensor*));
    tensor->roots[0] = nnl2_ad_copy(tensor, tensor->data->dtype); 
    pow_powf_inplace(tensor->data, exponent);
    tensor->extra_correspondence = exponent;
    tensor->is_leaf = false; 
    tensor->ts_type = nnl2_type_ad;
    
    switch(ad_mode) {
        case nnl2_ad_reverse_mode: tensor->backward_fn = nnl2_ad_reverse_backward_pow_powf_inplace; break;
        
        default: {
            NNL2_UNKNOWN_AD_MODE_ERROR(ad_mode);
            return;
        }
    }
}

#endif /** NNL2_AD_POW_POWF_INPLACE_H **/
