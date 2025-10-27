#ifndef NNL2_AD_MIN_INPLACE_H
#define NNL2_AD_MIN_INPLACE_H

void nnl2_ad_reverse_backward_min_inplace(nnl2_ad_tensor* result) {
    nnl2_ad_reverse_derivative_min_inplace(result, result->roots[0]);
}   

void nnl2_ad_min_inplace(nnl2_ad_tensor* result, nnl2_ad_tensor* operand, nnl2_ad_mode ad_mode) {
    if(result->requires_grad && !result->is_leaf) {
        // do something
    }
    
    mininplace(result->data, operand->data);
    result->num_roots = 1;
    result->roots = (nnl2_ad_tensor**)malloc(sizeof(nnl2_ad_tensor*));
    result->roots[0] = operand;
    result->is_leaf = false; 
    result->ts_type = nnl2_type_ad;
    
    switch(ad_mode) {
        case nnl2_ad_reverse_mode: result->backward_fn = nnl2_ad_reverse_backward_min_inplace; break;
        
        default: {
            NNL2_UNKNOWN_AD_MODE_ERROR(ad_mode);
            return;
        }
    }
}

#endif /** NNL2_AD_MIN_INPLACE_H **/
