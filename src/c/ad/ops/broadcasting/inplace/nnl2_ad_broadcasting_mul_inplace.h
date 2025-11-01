#ifndef NNL2_AD_MUL_BROADCASTING_INPLACE_H
#define NNL2_AD_MUL_BROADCASTING_INPLACE_H

void nnl2_ad_mul_broadcasting_inplace(nnl2_ad_tensor* multiplicand, nnl2_ad_tensor* multiplier) {
    if(multiplicand->requires_grad && !multiplicand->is_leaf) {
        NNL2_AD_INPLACE_FATAL("*= (.* in-place) (broadcasting)", multiplicand);
    }
    
    mul_broadcasting_inplace(multiplicand->data, multiplier->data);
}

#endif /** NNL2_AD_MUL_BROADCASTING_INPLACE_H **/
