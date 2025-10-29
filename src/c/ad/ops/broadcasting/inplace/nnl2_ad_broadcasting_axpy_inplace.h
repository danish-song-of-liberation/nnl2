#ifndef NNL2_AD_AXPY_BROADCASTING_INPLACE_H
#define NNL2_AD_AXPY_BROADCASTING_INPLACE_H

void nnl2_ad_reverse_backward_axpy_broadcasting_inplace(nnl2_ad_tensor* sumend) {
    nnl2_ad_reverse_derivative_axpy_broadcasting_inplace(sumend, sumend->roots[0], sumend->extra_multiplier);
}	

void nnl2_ad_axpy_broadcasting_inplace(nnl2_ad_tensor* sumend, nnl2_ad_tensor* axpyend, float multiplier, nnl2_ad_mode ad_mode) {
    if(sumend->requires_grad && !sumend->is_leaf) {
        // do something
    }
    
    axpy_broadcasting_inplace(sumend->data, axpyend->data, multiplier);
    sumend->num_roots = 1;
    sumend->roots = (nnl2_ad_tensor**)malloc(sizeof(nnl2_ad_tensor*));
    sumend->roots[0] = axpyend;
    sumend->is_leaf = false; 
    sumend->name = NULL;
    sumend->ts_type = nnl2_type_ad;
    sumend->extra_multiplier = multiplier; 
    
    switch(ad_mode) {
        case nnl2_ad_reverse_mode: sumend->backward_fn = nnl2_ad_reverse_backward_axpy_broadcasting_inplace; break;
        
        default: {
            NNL2_UNKNOWN_AD_MODE_ERROR(ad_mode);
            return;
        }
    }
}

#endif /** NNL2_AD_AXPY_BROADCASTING_INPLACE_H **/
