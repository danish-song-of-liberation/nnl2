#ifndef NNL2_AD_POW_BROADCASTING_INPLACE_H
#define NNL2_AD_POW_BROADCASTING_INPLACE_H

void nnl2_ad_reverse_backward_pow_broadcasting_inplace(nnl2_ad_tensor* base) {
    nnl2_ad_reverse_derivative_pow_broadcasting_inplace(base, base->roots[0]);
}   

void nnl2_ad_pow_broadcasting_inplace(nnl2_ad_tensor* base, nnl2_ad_tensor* exponent, nnl2_ad_mode ad_mode) {
    if(base->requires_grad && !base->is_leaf) {
        // do something
    }
    
    pow_broadcasting_inplace(base->data, exponent->data);
    base->num_roots = 1;
    base->roots = (nnl2_ad_tensor**)malloc(sizeof(nnl2_ad_tensor*));
    base->roots[0] = exponent;
    base->is_leaf = false; 
    base->ts_type = nnl2_type_ad;
    
    switch(ad_mode) {
        case nnl2_ad_reverse_mode: base->backward_fn = nnl2_ad_reverse_backward_pow_broadcasting_inplace; break;
        
        default: {
            NNL2_UNKNOWN_AD_MODE_ERROR(ad_mode);
            return;
        }
    }
}

#endif /** NNL2_AD_POW_BROADCASTING_INPLACE_H **/
