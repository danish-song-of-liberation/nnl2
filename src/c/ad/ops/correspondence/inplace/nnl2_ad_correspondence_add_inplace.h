#ifndef NNL2_AD_ADD_INCF_INPLACE_H
#define NNL2_AD_ADD_INCF_INPLACE_H

void nnl2_ad_add_incf_inplace(nnl2_ad_tensor* tensor, void* inc) {
    if(tensor->requires_grad) {
        NNL2_AD_INPLACE_FATAL("+= (.+ in-place) (correspondence)", tensor);
    }
    
	add_incf_inplace(tensor->data, inc);
}

#endif /** NNL2_AD_ADD_INCF_INPLACE_H **/
