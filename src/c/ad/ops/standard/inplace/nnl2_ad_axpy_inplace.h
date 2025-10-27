#ifndef NNL2_AD_AXPY_INPLACE_H
#define NNL2_AD_AXPY_INPLACE_H

void nnl2_ad_reverse_backward_inplace_axpy(nnl2_ad_tensor* tensor) {
    nnl2_ad_reverse_derivative_inplace_axpy(tensor, tensor->roots[0], tensor->roots[1], tensor->extra_multiplier);
}   

void nnl2_ad_inplace_axpy(nnl2_ad_tensor* summand, nnl2_ad_tensor* sumend, float multiplier, nnl2_ad_mode ad_mode) {
    if(summand->requires_grad && !summand->is_leaf) {
        // do something
    }

    summand->num_roots = 2;
    summand->roots = (nnl2_ad_tensor**)malloc(2 * sizeof(nnl2_ad_tensor*));
    summand->roots[0] = nnl2_ad_copy(summand, summand->data->dtype);
    summand->roots[1] = sumend;
    axpy_inplace(summand->data, sumend->data, multiplier);
    summand->is_leaf = false; 
    summand->ts_type = nnl2_type_ad;
    summand->extra_multiplier = multiplier;
    summand->requires_grad = summand->requires_grad || sumend->requires_grad;
    
    switch(ad_mode) {
        case nnl2_ad_reverse_mode: summand->backward_fn = nnl2_ad_reverse_backward_inplace_axpy; break;
        
        default: {
            NNL2_UNKNOWN_AD_MODE_ERROR(ad_mode);
            return;
        }
    }
}

#endif /** NNL2_AD_AXPY_INPLACE_H **/
