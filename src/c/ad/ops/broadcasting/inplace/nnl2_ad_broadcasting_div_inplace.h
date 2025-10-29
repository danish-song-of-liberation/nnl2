#ifndef NNL2_AD_DIV_BROADCASTING_INPLACE_H
#define NNL2_AD_DIV_BROADCASTING_INPLACE_H

void nnl2_ad_reverse_backward_div_broadcasting_inplace(nnl2_ad_tensor* dividend) {
    nnl2_ad_reverse_derivative_div_broadcasting_inplace(dividend, dividend->roots[0]);
}   

void nnl2_ad_div_broadcasting_inplace(nnl2_ad_tensor* dividend, nnl2_ad_tensor* divisor, nnl2_ad_mode ad_mode) {
    if(dividend->requires_grad && !dividend->is_leaf) {
        // do something
    }
    
    div_broadcasting_inplace(dividend->data, divisor->data);
    dividend->num_roots = 1;
    dividend->roots = (nnl2_ad_tensor**)malloc(sizeof(nnl2_ad_tensor*));
    dividend->roots[0] = divisor;
    dividend->is_leaf = false; 
    dividend->ts_type = nnl2_type_ad;
    
    switch(ad_mode) {
        case nnl2_ad_reverse_mode: dividend->backward_fn = nnl2_ad_reverse_backward_div_broadcasting_inplace; break;
        
        default: {
            NNL2_UNKNOWN_AD_MODE_ERROR(ad_mode);
            return;
        }
    }
}

#endif /** NNL2_AD_DIV_BROADCASTING_INPLACE_H **/
