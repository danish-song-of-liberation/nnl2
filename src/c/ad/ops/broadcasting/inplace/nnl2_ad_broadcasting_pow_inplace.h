#ifndef NNL2_AD_POW_BROADCASTING_INPLACE_H
#define NNL2_AD_POW_BROADCASTING_INPLACE_H

void nnl2_ad_pow_broadcasting_inplace(nnl2_ad_tensor* base, nnl2_ad_tensor* exponent) {
    if(base->requires_grad) {
        NNL2_AD_INPLACE_FATAL("^= (.^ in-place) (broadcasting)", base);
    }
    
    pow_broadcasting_inplace(base->data, exponent->data);
}

#endif /** NNL2_AD_POW_BROADCASTING_INPLACE_H **/
