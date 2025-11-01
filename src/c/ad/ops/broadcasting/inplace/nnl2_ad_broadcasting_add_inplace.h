#ifndef NNL2_AD_ADD_BROADCASTING_INPLACE_H
#define NNL2_AD_ADD_BROADCASTING_INPLACE_H

void nnl2_ad_add_broadcasting_inplace(nnl2_ad_tensor* summand, nnl2_ad_tensor* addend) {
    if(summand->requires_grad) {
        NNL2_AD_INPLACE_FATAL("+= (.+ in-place) (broadcasting)", summand);
    }
    
    add_broadcasting_inplace(summand->data, addend->data);
}

#endif /** NNL2_AD_ADD_BROADCASTING_INPLACE_H **/
