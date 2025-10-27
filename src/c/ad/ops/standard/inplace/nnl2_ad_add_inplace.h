#ifndef NNL2_AD_ADD_INPLACE_H
#define NNL2_AD_ADD_INPLACE_H

void nnl2_ad_reverse_backward_add_inplace(nnl2_ad_tensor* summand) {
    nnl2_ad_reverse_derivative_add_inplace(summand, summand->roots[0]);
}   

void nnl2_ad_add_inplace(nnl2_ad_tensor* summand, nnl2_ad_tensor* addend, nnl2_ad_mode ad_mode) {
	if(summand->requires_grad && !summand->is_leaf) {
		// do something
	}
	
    addinplace(summand->data, addend->data);
    summand->num_roots = 1;
    summand->roots = (nnl2_ad_tensor**)malloc(sizeof(nnl2_ad_tensor*));
    summand->roots[0] = addend;
    summand->is_leaf = false; 
    summand->ts_type = nnl2_type_ad;
    
    switch(ad_mode) {
        case nnl2_ad_reverse_mode: summand->backward_fn = nnl2_ad_reverse_backward_add_inplace; break;
        
        default: {
            NNL2_UNKNOWN_AD_MODE_ERROR(ad_mode);
            return;
        }
    }
}

#endif /** NNL2_AD_ADD_INPLACE_H **/
