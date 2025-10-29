#ifndef NNL2_AD_MUL_BROADCASTING_INPLACE_H
#define NNL2_AD_MUL_BROADCASTING_INPLACE_H

void nnl2_ad_reverse_backward_mul_broadcasting_inplace(nnl2_ad_tensor* multiplicand) {
    nnl2_ad_reverse_derivative_mul_broadcasting_inplace(multiplicand, multiplicand->roots[0]);
}   

void nnl2_ad_mul_broadcasting_inplace(nnl2_ad_tensor* multiplicand, nnl2_ad_tensor* multiplier, nnl2_ad_mode ad_mode) {
    if(multiplicand->requires_grad && !multiplicand->is_leaf) {
        // do something
    }
    
    mul_broadcasting_inplace(multiplicand->data, multiplier->data);
    multiplicand->num_roots = 1;
    multiplicand->roots = (nnl2_ad_tensor**)malloc(sizeof(nnl2_ad_tensor*));
    multiplicand->roots[0] = multiplier;
    multiplicand->is_leaf = false; 
    multiplicand->ts_type = nnl2_type_ad;
    
    switch(ad_mode) {
        case nnl2_ad_reverse_mode: multiplicand->backward_fn = nnl2_ad_reverse_backward_mul_broadcasting_inplace; break;
        
        default: {
            NNL2_UNKNOWN_AD_MODE_ERROR(ad_mode);
            return;
        }
    }
}

#endif /** NNL2_AD_MUL_BROADCASTING_INPLACE_H **/
