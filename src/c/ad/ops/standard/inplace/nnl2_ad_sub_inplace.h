#ifndef NNL2_AD_SUB_INPLACE_H
#define NNL2_AD_SUB_INPLACE_H

void nnl2_ad_reverse_backward_sub_inplace(nnl2_ad_tensor* minuend) {
    nnl2_ad_reverse_derivative_sub_inplace(minuend, minuend->roots[0]);
}   

void nnl2_ad_sub_inplace(nnl2_ad_tensor* minuend, nnl2_ad_tensor* subtrahend, nnl2_ad_mode ad_mode) {
    if(minuend->requires_grad && !minuend->is_leaf) {
        // do something
    }
    
    subinplace(minuend->data, subtrahend->data);
    minuend->num_roots = 1;
    minuend->roots = (nnl2_ad_tensor**)malloc(sizeof(nnl2_ad_tensor*));
    minuend->roots[0] = subtrahend;
    minuend->is_leaf = false; 
    minuend->ts_type = nnl2_type_ad;
    
    switch(ad_mode) {
        case nnl2_ad_reverse_mode: minuend->backward_fn = nnl2_ad_reverse_backward_sub_inplace; break;
        
        default: {
            NNL2_UNKNOWN_AD_MODE_ERROR(ad_mode);
            return;
        }
    }
}

#endif /** NNL2_AD_SUB_INPLACE_H **/