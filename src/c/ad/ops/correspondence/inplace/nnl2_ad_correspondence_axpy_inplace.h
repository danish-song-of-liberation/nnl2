#ifndef NNL2_AD_AXPF_INPLACE_H
#define NNL2_AD_AXPF_INPLACE_H

void nnl2_ad_reverse_backward_axpf_inplace(nnl2_ad_tensor* summand) {
    nnl2_ad_reverse_derivative_axpf_inplace(summand, summand->roots[0], summand->extra_correspondence, summand->extra_multiplier);
}   

void nnl2_ad_axpf_inplace(nnl2_ad_tensor* summand, void* sumend, float alpha, nnl2_ad_mode ad_mode) {
    if(summand->requires_grad && !summand->is_leaf) {
        // do something
    }
    
    summand->num_roots = 1;
    summand->roots = (nnl2_ad_tensor**)malloc(sizeof(nnl2_ad_tensor*));
    summand->roots[0] = nnl2_ad_copy(summand, summand->data->dtype);  
	axpf_inplace(summand->data, sumend, alpha);
    summand->extra_correspondence = sumend;
    summand->extra_multiplier = alpha;
    summand->is_leaf = false; 
    summand->ts_type = nnl2_type_ad;
    
    switch(ad_mode) {
        case nnl2_ad_reverse_mode: summand->backward_fn = nnl2_ad_reverse_backward_axpf_inplace; break;
        
        default: {
            NNL2_UNKNOWN_AD_MODE_ERROR(ad_mode);
            return;
        }
    }
}

#endif /** NNL2_AD_AXPF_INPLACE_H **/
